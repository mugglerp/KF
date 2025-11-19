"""QR-Net 训练脚本：卡尔曼滤波器噪声协方差矩阵自适应学习

本脚本实现基于 Transformer 的 QR-Net 模型训练，用于预测卡尔曼滤波器中的：
    - Q 矩阵：过程噪声协方差（状态转移不确定性）
    - R 矩阵：测量噪声协方差（观测不确定性）

核心思路：
    1. 从真实 EEG 数据中学习噪声统计特性
    2. 通过协方差匹配损失训练网络
    3. 支持不同噪声条件下的自适应调整

数据格式：
    - 输入：带噪 EEG 观测序列 [N, T, 2]
    - 标签：干净 EEG 状态序列 [N, T, 2]
    - 第二通道：导数（derivative）或副本（copy）
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from dataloaders.EEG_all_DataLoader import EEG_all_Dataloader
from models.QRNet_model import QRTransformer


# ======================
# 核心训练工具函数
# ======================

def compute_batch_loss(model, batch_x, batch_z, mse):
    """
    计算单个 batch 的协方差匹配损失
    
    损失计算策略：
        1. 使用网络预测每个时间步的 Q 和 R 矩阵
        2. 从数据中统计真实的协方差矩阵作为监督信号
        3. 通过 MSE 损失最小化预测与真实协方差的差异
    
    真实协方差矩阵的计算：
        Q_target = E[(x_t - x_{t-1})(x_t - x_{t-1})^T]  # 状态增量的协方差
        R_target = E[(z_t - x_t)(z_t - x_t)^T]          # 测量残差的协方差
    
    Args:
        model: QRTransformer 模型实例
        batch_x: 干净状态序列 [B, T, 2]
        batch_z: 带噪观测序列 [B, T, 2]
        mse: MSE 损失函数
    
    Returns:
        loss: 标量损失值（如果序列长度不足返回 None）
    """
    B, T_cur, _ = batch_x.shape
    if T_cur < 2:
        # 序列太短无法计算时间差分，跳过
        return None

    # === 步骤 1: 网络前向传播获取预测的 Q 和 R ===
    Q_seq, R_seq = model(batch_x, batch_z)  # [B, T, 2, 2]

    # 只对 t=1..T-1 的时间步计算损失（因为需要 t-1 的信息）
    Q_pred = Q_seq[:, 1:, :, :].reshape(-1, 2, 2)  # [N', 2, 2], N' = B*(T-1)
    R_pred = R_seq[:, 1:, :, :].reshape(-1, 2, 2)  # [N', 2, 2]

    # === 步骤 2: 计算真实协方差矩阵的统计量 ===
    
    # 提取对应时间步的状态和观测
    x_now  = batch_x[:, 1:, :]   # t 时刻状态 [B, T-1, 2]
    x_prev = batch_x[:, :-1, :]  # t-1 时刻状态 [B, T-1, 2]
    z_now  = batch_z[:, 1:, :]   # t 时刻观测 [B, T-1, 2]
    z_hat  = x_now               # 预测测量 = 真实状态（假设观测模型为恒等映射）

    # 状态增量（过程噪声的体现）
    dx = x_now - x_prev  # [B, T-1, 2]
    
    # 测量残差（测量噪声的体现）
    e = z_now - z_hat    # [B, T-1, 2]

    # 展平为样本维度
    dx_flat = dx.reshape(-1, 2)  # [N', 2]
    e_flat  = e.reshape(-1, 2)   # [N', 2]

    # 去均值（协方差定义需要中心化）
    dx_flat = dx_flat - dx_flat.mean(dim=0, keepdim=True)
    e_flat  = e_flat  - e_flat.mean(dim=0, keepdim=True)

    # === 步骤 3: 计算完整 2×2 协方差矩阵 ===
    
    # Q 的真实协方差: E[dx @ dx^T]
    # 通过外积计算每个样本的协方差，然后平均
    Q_cov = dx_flat.unsqueeze(2) * dx_flat.unsqueeze(1)  # [N', 2, 2]
    Q_target = Q_cov.mean(dim=0, keepdim=True)           # [1, 2, 2]

    # R 的真实协方差: E[e @ e^T]
    R_cov = e_flat.unsqueeze(2) * e_flat.unsqueeze(1)    # [N', 2, 2]
    R_target = R_cov.mean(dim=0, keepdim=True)           # [1, 2, 2]

    # 广播到每个时间步的预测
    Q_target = Q_target.expand_as(Q_pred)  # [N', 2, 2]
    R_target = R_target.expand_as(R_pred)  # [N', 2, 2]

    # === 步骤 4: 计算 MSE 损失（包含矩阵的全部 4 个元素）===
    loss_q = mse(Q_pred, Q_target)  # Q 矩阵损失
    loss_r = mse(R_pred, R_target)  # R 矩阵损失

    # 总损失 = Q损失 + R损失
    loss = loss_q + loss_r
    return loss


def prepare_eeg_data(dataloader, num_train=10, num_val=10, num_test=10, seq_len=512):
    """
    从 EEG 数据加载器中提取训练/验证/测试数据集
    
    数据分割策略（与 EEGDenoising.py 保持完全一致）：
        - 训练集：从索引 0 开始，连续取 num_train 个窗口，每个窗口间隔 seq_len
        - 验证集：每个窗口从 index + 5120 到 index + 5632
        - 测试集：每个窗口从 index + 10240 到 index + 10752
    
    注意：这个函数完全复制 EEGDenoising.py 的数据提取逻辑
    
    Args:
        dataloader: EEG_all_Dataloader 实例，包含完整的 EEG 数据
        num_train: 训练样本数量
        num_val: 验证样本数量
        num_test: 测试样本数量
        seq_len: 每个样本的序列长度（时间窗口大小）
    
    Returns:
        六个张量，每个形状为 [N, 1, seq_len]：
            noiseEEG_train, EEG_train: 训练集的带噪/干净数据
            noiseEEG_val, EEG_val: 验证集的带噪/干净数据
            noiseEEG_test, EEG_test: 测试集的带噪/干净数据
    """
    # 初始化数据容器
    noiseEEG_train = torch.zeros((num_train, 1, seq_len))
    EEG_train = torch.zeros((num_train, 1, seq_len))
    noiseEEG_val = torch.zeros((num_val, 1, seq_len))
    EEG_val = torch.zeros((num_val, 1, seq_len))
    noiseEEG_test = torch.zeros((num_test, 1, seq_len))
    EEG_test = torch.zeros((num_test, 1, seq_len))
    
    # === 按照 EEGDenoising.py 的方式提取数据 ===
    # 训练集、验证集、测试集使用相同的循环结构
    for i in range(num_train):
        index = seq_len * i  # 计算当前窗口的起始索引
        # 训练集：从 index 开始
        noiseEEG_train[i] = dataloader.observations[0, index:index+seq_len, 0].float()
        EEG_train[i] = dataloader.dataset[0, index:index+seq_len, 0].float()
    
    for i in range(num_val):
        index = seq_len * i
        # 验证集：从 index + 5120 开始（固定偏移）
        noiseEEG_val[i] = dataloader.observations[0, index+5120:index+5632, 0].float()
        EEG_val[i] = dataloader.dataset[0, index+5120:index+5632, 0].float()
    
    for i in range(num_test):
        index = seq_len * i
        # 测试集：从 index + 10240 开始（固定偏移）
        noiseEEG_test[i] = dataloader.observations[0, index+10240:index+10752, 0].float()
        EEG_test[i] = dataloader.dataset[0, index+10240:index+10752, 0].float()
    
    return noiseEEG_train, EEG_train, noiseEEG_val, EEG_val, noiseEEG_test, EEG_test


def build_random_noise_observations(eeg_clean: torch.Tensor,
                                    snr_choices=None,
                                    seq_len: int = 512,
                                    eog_path: str = 'dataloaders/datasets/EOG_all_epochs.npy',
                                    emg_path: str = 'dataloaders/datasets/EMG_all_epochs.npy') -> torch.Tensor:
    """
    随机噪声增广：基于干净 EEG（[N, 1, T]），为每个样本生成一次带噪观测。
    
    噪声来源（随机三选一）：
      - 高斯噪声：零均值 N(0,1)，按目标 SNR 精确缩放
      - EOG 片段：从 EOG_all_epochs.npy 直接用 numpy 载入，随机截取 T 点并做零均值，按 SNR 缩放
      - EMG 片段：从 EMG_all_epochs.npy 直接用 numpy 载入，随机截取 T 点并做零均值，按 SNR 缩放
      注：不依赖额外的 DataLoader；若 EOG/EMG 文件不可读，自动回退为高斯噪声
    
    SNR 设定：从 {-3, 0, 3} dB 中随机取一值，按 SNR(dB)=10*log10(Ps/Pn) 计算噪声功率并缩放，确保与干净 EEG 的 SNR 精确匹配。
    
    参数:
        eeg_clean: 干净 EEG，形状 [N, 1, T]
        snr_choices: 备选 SNR 列表，默认 [-3, 0, 3]
        seq_len: 序列长度 T（仅用于说明）
        eog_path/emg_path: EOG/EMG numpy 文件路径
    返回:
        noisy: 带噪数据张量，形状 [N, 1, T]
    """
    if snr_choices is None:
        snr_choices = [-3, 0, 3]
    N, _, T = eeg_clean.shape
    device = eeg_clean.device

    # 预加载 EOG/EMG 数据到 CPU numpy，避免GPU/IO频繁切换
    try:
        eog_np = np.load(eog_path).reshape(-1)
    except Exception:
        eog_np = None
    try:
        emg_np = np.load(emg_path).reshape(-1)
    except Exception:
        emg_np = None

    noisy = torch.zeros((N, 1, T), device=device)

    for i in range(N):
        clean = eeg_clean[i, 0]  # [T]
        # 选择 SNR 和 噪声类型
        snr_db = int(np.random.choice(snr_choices))
        noise_type = np.random.choice(['gaussian', 'eog', 'emg'])

        # 计算信号功率 (均方功率)
        p_signal = torch.mean(clean.float()**2).item()
        # 由 SNR(dB) = 10*log10(Ps/Pn) 推得噪声功率
        p_noise = p_signal / (10 ** (snr_db / 10.0))

        if noise_type == 'gaussian' or ((noise_type == 'eog') and eog_np is None) or ((noise_type == 'emg') and emg_np is None):
            # 生成零均值高斯噪声
            n = torch.randn(T, device=device)
        elif noise_type == 'eog':
            # 随机截取一段 EOG
            if len(eog_np) >= T:
                start = np.random.randint(0, len(eog_np) - T + 1)
                seg = torch.tensor(eog_np[start:start+T], dtype=torch.float32, device=device)
            else:
                seg = torch.randn(T, device=device)
            seg = seg - torch.mean(seg)
            n = seg
        else:  # 'emg'
            if len(emg_np) >= T:
                start = np.random.randint(0, len(emg_np) - T + 1)
                seg = torch.tensor(emg_np[start:start+T], dtype=torch.float32, device=device)
            else:
                seg = torch.randn(T, device=device)
            seg = seg - torch.mean(seg)
            n = seg

        # 缩放噪声以匹配目标噪声功率
        var_n = torch.var(n)
        scale = float(np.sqrt(max(p_noise, 1e-12) / max(var_n.item(), 1e-12)))
        n_scaled = n * scale

        noisy[i, 0] = clean + n_scaled

    return noisy

def train_qr_net(mode="copy",
                 datapoints=512,
                 samples=None,
                 snr_db=0,
                 noise_color=0,
                 num_train=10,
                 num_val=10,
                 num_test=10,
                 batch_size=8,
                 epochs=300,
                 lr=1e-4,
                 wd=1e-3,
                 device=None):
    """
    QR-Net 主训练函数（遵循 EEGDenoising.py 的训练策略）
    
    训练流程：
        1. 从真实 EEG 数据加载器获取带噪/干净数据
        2. 根据 mode 构造双通道输入（copy 或 derivative）
        3. 使用协方差匹配损失训练 Transformer 模型
        4. 基于验证集保存最佳模型
        5. 在测试集上评估最终性能
    
    Args:
        mode: 第二通道模式
            - "copy": 两个通道都是 EEG 信号（冗余编码）
            - "derivative": 第二通道是 EEG 的时间导数（增强动态信息）
        datapoints: 数据窗口大小（序列长度），通常为 512
        samples: EEG 样本索引列表（None 表示使用默认 [0]）
        snr_db: 信噪比（dB），0 表示信号和噪声功率相等
        noise_color: 噪声类型
            - 0: 白噪声（频谱均匀）
            - 1: 粉噪声（1/f，低频更强）
            - 2: 棕噪声（1/f², 低频最强）
        num_train: 训练样本数量
        num_val: 验证样本数量
        num_test: 测试样本数量
        batch_size: 批次大小
        epochs: 训练轮数
        lr: 学习率（AdamW 优化器）
        wd: 权重衰减（L2 正则化系数）
        device: 计算设备（None 表示自动选择 cuda/cpu）
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("="*50)
    print(f"使用 EEG_all_Dataloader 加载数据，mode = {mode}")
    print(f"数据窗口大小 = {datapoints}, SNR = {snr_db} dB, 噪声颜色 = {noise_color}")
    print(f"训练参数: epochs={epochs}, batch_size={batch_size}, lr={lr}, wd={wd}")
    print("="*50)

    # === 第一步：使用 EEG_all_Dataloader 加载数据 ===
    if samples is None:
        samples = [0]  # 默认使用第一个样本

    # 创建数据加载器（自动处理噪声添加）
    dataloader = EEG_all_Dataloader(
        datapoints=datapoints,
        samples=samples,
        snr_db=snr_db,
        noise_color=noise_color
    )

    print(f"数据集总长度: {len(dataloader)}")
    print(f"观测数据形状: {dataloader.observations.shape}")
    print(f"干净数据形状: {dataloader.dataset.shape}")

    # === 第二步：准备训练/验证/测试数据 ===
    noiseEEG_train, EEG_train, noiseEEG_val, EEG_val, noiseEEG_test, EEG_test = \
        prepare_eeg_data(dataloader, num_train, num_val, num_test, datapoints)

    print(f"训练集: {num_train} 条, 验证集: {num_val} 条, 测试集: {num_test} 条")
    print(f"训练数据形状: noiseEEG={noiseEEG_train.shape}, EEG={EEG_train.shape}")
    
    # === 导出原始数据集（[N, 1, T] 格式）===
    if mode == "copy":
        dataset_file = "checkpoints/dataset_copy.npz"
    else:
        dataset_file = "checkpoints/dataset_derivative.npz"
    
    np.savez(
        dataset_file,
        noiseEEG_train=noiseEEG_train.cpu().numpy(),
        EEG_train=EEG_train.cpu().numpy(),
        noiseEEG_val=noiseEEG_val.cpu().numpy(),
        EEG_val=EEG_val.cpu().numpy(),
        noiseEEG_test=noiseEEG_test.cpu().numpy(),
        EEG_test=EEG_test.cpu().numpy(),
        mode=mode,
        snr_db=snr_db,
        noise_color=noise_color
    )
    print(f"原始数据集已保存到: {dataset_file}")

    # === 第三步：转换数据格式 [N, 1, T] -> [N, T, 2] ===
    
    def add_derivative(data, scale=0.5):
        """
        为单通道数据添加导数通道
        
        Args:
            data: 输入数据 [N, 1, T]
            scale: 导数缩放系数（避免数值过大）
        
        Returns:
            双通道数据 [N, T, 2]，第一通道为原始信号，第二通道为导数
        """
        N, _, T = data.shape
        # 计算时间导数（向前差分）
        deriv = data[:, :, 1:] - data[:, :, :-1]  # [N, 1, T-1]
        # 在开头补一个相同值保持长度
        deriv = torch.cat([deriv[:, :, :1], deriv], dim=2)  # [N, 1, T]
        # 缩放导数避免数值过大
        deriv = deriv * scale
        # 拼接原始信号和导数，并调整维度顺序
        return torch.cat([data, deriv], dim=1).permute(0, 2, 1)  # [N, T, 2]
    
    # 根据 mode 构造第二通道
    # 数据增强：每个干净 EEG 样本生成两次随机噪声观测（N→2N），
    #          噪声类型在 高斯/EOG/EMG 中随机选择，SNR 在 {-3,0,3} dB 中随机选择。
    num_aug = 3
    snr_choices = [-3, 0, 3]

    eeg_train_aug = EEG_train.repeat_interleave(num_aug, dim=0)  # [N*2, 1, T]
    noisy_train_aug = build_random_noise_observations(
        eeg_train_aug, snr_choices, datapoints,
        eog_path='dataloaders/datasets/EOG_all_epochs.npy',
        emg_path='dataloaders/datasets/EMG_all_epochs.npy'
    )  # [N*2, 1, T]

    if mode == "copy":
        # 两个通道都是原始/带噪信号
        x_train = torch.cat([eeg_train_aug, eeg_train_aug], dim=1).permute(0, 2, 1)
    else:  # derivative
        x_train = add_derivative(eeg_train_aug)

    # 验证/测试增广：每个干净样本生成三次噪声（N→3N），保持样本数一致性
    num_aug_val = 3
    EEG_val_aug = EEG_val.repeat_interleave(num_aug_val, dim=0)
    EEG_test_aug = EEG_test.repeat_interleave(num_aug_val, dim=0)
    if mode == "copy":
        x_val = torch.cat([EEG_val_aug, EEG_val_aug], dim=1).permute(0, 2, 1)
        x_test = torch.cat([EEG_test_aug, EEG_test_aug], dim=1).permute(0, 2, 1)
    else:
        x_val = add_derivative(EEG_val_aug)
        x_test = add_derivative(EEG_test_aug)
    
    # 训练/验证/测试均使用随机噪声：
    # - 训练集：每个干净样本生成两次噪声（已在 noisy_train_aug 中）
    # - 验证/测试：每个干净样本生成一次噪声（不增广样本数）
    noisy_val = build_random_noise_observations(
        EEG_val_aug, snr_choices, datapoints,
        eog_path='dataloaders/datasets/EOG_all_epochs.npy',
        emg_path='dataloaders/datasets/EMG_all_epochs.npy'
    )
    noisy_test = build_random_noise_observations(
        EEG_test_aug, snr_choices, datapoints,
        eog_path='dataloaders/datasets/EOG_all_epochs.npy',
        emg_path='dataloaders/datasets/EMG_all_epochs.npy'
    )

    if mode == "copy":
        z_train = torch.cat([noisy_train_aug, noisy_train_aug], dim=1).permute(0, 2, 1)
        z_val = torch.cat([noisy_val, noisy_val], dim=1).permute(0, 2, 1)
        z_test = torch.cat([noisy_test, noisy_test], dim=1).permute(0, 2, 1)
    else:
        z_train = add_derivative(noisy_train_aug)
        z_val = add_derivative(noisy_val)
        z_test = add_derivative(noisy_test)
    
    # 移动数据到计算设备（GPU/CPU）
    x_train, z_train = x_train.to(device), z_train.to(device)
    x_val, z_val = x_val.to(device), z_val.to(device)
    x_test, z_test = x_test.to(device), z_test.to(device)
    
    # === 导出转换后的数据集（[N, T, 2] 格式）===
    if mode == "copy":
        processed_file = "checkpoints/processed_data_copy.npz"
    else:
        processed_file = "checkpoints/processed_data_derivative.npz"
    
    np.savez(
        processed_file,
        x_train=x_train.cpu().numpy(),
        z_train=z_train.cpu().numpy(),
        x_val=x_val.cpu().numpy(),
        z_val=z_val.cpu().numpy(),
        x_test=x_test.cpu().numpy(),
        z_test=z_test.cpu().numpy(),
        mode=mode
    )
    print(f"处理后数据集已保存到: {processed_file}")

    # === 第四步：初始化模型 ===
    model = QRTransformer(
        in_dim=8,              # 输入特征维度 [x, z, e, dx] = 2+2+2+2
        d_model=32,            # Transformer 隐藏层维度
        nhead=4,               # 多头注意力头数
        num_layers=2,          # Transformer 编码器层数
        dim_feedforward=64,    # 前馈网络维度
        dropout=0.1,           # Dropout 概率
        max_len=datapoints,    # 最大序列长度
    ).to(device)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # === 第五步：配置优化器和损失函数 ===
    # 使用 AdamW 优化器（Adam + 权重衰减正则化）
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    # MSE 损失用于协方差矩阵匹配
    mse = nn.MSELoss()

    # 设置模型保存路径
    if mode == "copy":
        save_name = "checkpoints/qr_net_copy_eeg.pt"
    else:
        save_name = "checkpoints/qr_net_derivative_eeg.pt"
    
    # 确保保存目录存在
    os.makedirs("checkpoints", exist_ok=True)

    # === 第六步：训练循环 ===
    best_val_loss = float('inf')
    
    # 记录 loss 历史
    train_loss_history = []
    val_loss_history = []
    
    for epoch in range(1, epochs + 1):
        # ---- 训练阶段 ----
        model.train()
        # 随机打乱训练样本（使用增广后的样本数）
        n_train = x_train.shape[0]
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        num_batches = 0

        # 分批训练
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            batch_x = x_train[idx]  # [B, T, 2]
            batch_z = z_train[idx]  # [B, T, 2]

            # 计算损失
            loss = compute_batch_loss(model, batch_x, batch_z, mse)
            if loss is None:
                continue

            # 反向传播和参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        train_loss = epoch_loss / max(num_batches, 1)
        train_loss_history.append(train_loss)

        # ---- 验证阶段 ----
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():  # 不计算梯度，节省内存
            n_val = x_val.shape[0]
            for i in range(0, n_val, batch_size):
                batch_x = x_val[i:i + batch_size]
                batch_z = z_val[i:i + batch_size]
                loss = compute_batch_loss(model, batch_x, batch_z, mse)
                if loss is None:
                    continue
                val_loss_sum += loss.item()
                val_batches += 1
        val_loss = val_loss_sum / max(val_batches, 1)
        val_loss_history.append(val_loss)

        # 定期打印训练进度
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} - train_loss: {train_loss:.6f}, val_loss: {val_loss:.6f}")
        
        # 保存验证集上表现最好的模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_name)
            if epoch % 10 == 0:
                print(f"  → 保存最佳模型 (val_loss: {best_val_loss:.6f})")

    print(f"\n训练完成！最佳验证损失: {best_val_loss:.6f}")
    print(f"模型权重已保存到: {save_name}")
    
    # === 保存 loss 历史和绘制曲线 ===
    # 保存 loss 数据
    loss_data = {
        'train_loss': train_loss_history,
        'val_loss': val_loss_history,
        'best_val_loss': best_val_loss,
        'epochs': epochs
    }
    
    if mode == "copy":
        loss_file = "checkpoints/loss_history_copy.npy"
        loss_plot = "checkpoints/loss_curve_copy.png"
    else:
        loss_file = "checkpoints/loss_history_derivative.npy"
        loss_plot = "checkpoints/loss_curve_derivative.png"
    
    np.save(loss_file, loss_data)
    print(f"Loss 历史已保存到: {loss_file}")
    
    # 绘制 loss 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_loss_history, label='Train Loss', linewidth=2)
    plt.plot(range(1, epochs + 1), val_loss_history, label='Val Loss', linewidth=2)
    plt.axhline(y=best_val_loss, color='r', linestyle='--', label=f'Best Val Loss: {best_val_loss:.6f}')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Training History - {mode.upper()} Mode', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(loss_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Loss 曲线已保存到: {loss_plot}")
    
    # === 第七步：测试阶段 ===
    print("\n" + "="*50)
    print("开始测试...")
    
    # 加载最佳模型权重
    model.load_state_dict(torch.load(save_name))
    model.eval()
    
    test_loss_sum = 0.0
    test_batches = 0
    
    with torch.no_grad():
        n_test = x_test.shape[0]
        for i in range(0, n_test, batch_size):
            batch_x = x_test[i:i + batch_size]
            batch_z = z_test[i:i + batch_size]
            loss = compute_batch_loss(model, batch_x, batch_z, mse)
            if loss is None:
                continue
            test_loss_sum += loss.item()
            test_batches += 1
    
    test_loss = test_loss_sum / max(test_batches, 1)
    test_loss_db = 10 * torch.log10(torch.tensor(test_loss))
    
    print(f"测试损失 (MSE): {test_loss:.6f}")
    print(f"测试损失 (dB): {test_loss_db:.2f}")
    print("="*50)
    
    # 返回训练历史和测试结果
    return {
        'train_loss_history': train_loss_history,
        'val_loss_history': val_loss_history,
        'best_val_loss': best_val_loss,
        'test_loss': test_loss,
        'test_loss_db': test_loss_db.item()
    }


if __name__ == "__main__":
    # === 主训练配置（与 EEGDenoising.py 保持一致）===
    
    # 用于存储两个模式的结果
    results = {}
    
    # ========== 训练参数配置 ==========
    # 可以根据需要调整以下参数
    # 注意：数据文件总大小为 2,311,168 个数据点
    # 最大可用样本数约为 2,311,168 / 512 = 4514 个
    NUM_TRAIN = 3000     # 训练样本数（固定3000 → 增广后6000）
    NUM_VAL = 757        # 验证样本数（剩余的一半）
    NUM_TEST = 757       # 测试样本数（剩余的一半）
    BATCH_SIZE = 64      # 批次大小
    EPOCHS = 300         # 训练轮数
    LEARNING_RATE = 1e-4 # 学习率
    WEIGHT_DECAY = 1e-3  # 权重衰减
    
    print("\n" + "="*60)
    print("训练配置:")
    print(f"  训练样本: {NUM_TRAIN}")
    print(f"  验证样本: {NUM_VAL}")
    print(f"  测试样本: {NUM_TEST}")
    print(f"  批次大小: {BATCH_SIZE}")
    print(f"  训练轮数: {EPOCHS}")
    print(f"  学习率: {LEARNING_RATE}")
    print(f"  权重衰减: {WEIGHT_DECAY}")
    print("="*60)
    
    # ========== 训练 COPY 模式 ==========
    print("\n" + "#"*60)
    print("#" + " "*58 + "#")
    print("#" + " "*15 + "开始训练 COPY 模式" + " "*21 + "#")
    print("#" + " "*58 + "#")
    print("#"*60 + "\n")
    
    results['copy'] = train_qr_net(
        mode="copy",         # 双通道模式: "copy" 或 "derivative"
        datapoints=512,      # 序列长度（时间窗口大小）
        samples=[0],         # 使用的 EEG 样本索引列表
        snr_db=0,            # 信噪比 (dB)，0 表示信号和噪声功率相等
        noise_color=0,       # 噪声类型: 0=白噪声, 1=粉噪声, 2=棕噪声
        num_train=NUM_TRAIN,
        num_val=NUM_VAL,
        num_test=NUM_TEST,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        wd=WEIGHT_DECAY,
    )
    
    # ========== 训练 DERIVATIVE 模式 ==========
    print("\n\n" + "#"*60)
    print("#" + " "*58 + "#")
    print("#" + " "*12 + "开始训练 DERIVATIVE 模式" + " "*17 + "#")
    print("#" + " "*58 + "#")
    print("#"*60 + "\n")
    
    results['derivative'] = train_qr_net(
        mode="derivative",   # 第二通道使用导数
        datapoints=512,      # 序列长度（时间窗口大小）
        samples=[0],         # 使用相同的 EEG 样本索引列表
        snr_db=0,            # 信噪比 (dB)，0 表示信号和噪声功率相等
        noise_color=0,       # 噪声类型: 0=白噪声, 1=粉噪声, 2=棕噪声
        num_train=NUM_TRAIN,
        num_val=NUM_VAL,
        num_test=NUM_TEST,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        wd=WEIGHT_DECAY,
    )
    
    # ========== 训练完成总结 ==========
    print("\n\n" + "="*60)
    print("="*60)
    print("所有训练完成！")
    print("\n模型文件:")
    print("  - Copy 模式模型: checkpoints/qr_net_copy_eeg.pt")
    print("  - Derivative 模式模型: checkpoints/qr_net_derivative_eeg.pt")
    print("\n数据集文件:")
    print("  - Copy 模式原始数据: checkpoints/dataset_copy.npz")
    print("  - Copy 模式处理数据: checkpoints/processed_data_copy.npz")
    print("  - Derivative 模式原始数据: checkpoints/dataset_derivative.npz")
    print("  - Derivative 模式处理数据: checkpoints/processed_data_derivative.npz")
    print("\nLoss 历史:")
    print("  - Copy 模式 Loss: checkpoints/loss_history_copy.npy")
    print("  - Derivative 模式 Loss: checkpoints/loss_history_derivative.npy")
    print("\nLoss 曲线图:")
    print("  - Copy 模式: checkpoints/loss_curve_copy.png")
    print("  - Derivative 模式: checkpoints/loss_curve_derivative.png")
    print("\n测试结果对比:")
    print(f"  - Copy 模式测试损失: {results['copy']['test_loss']:.6f} ({results['copy']['test_loss_db']:.2f} dB)")
    print(f"  - Derivative 模式测试损失: {results['derivative']['test_loss']:.6f} ({results['derivative']['test_loss_db']:.2f} dB)")
    print("="*60)
    
    # ========== 绘制对比图 ==========
    plt.figure(figsize=(14, 5))
    
    # 子图1: Copy 模式
    plt.subplot(1, 2, 1)
    plt.plot(results['copy']['train_loss_history'], label='Train Loss', linewidth=2)
    plt.plot(results['copy']['val_loss_history'], label='Val Loss', linewidth=2)
    plt.axhline(y=results['copy']['best_val_loss'], color='r', linestyle='--', 
                label=f"Best: {results['copy']['best_val_loss']:.6f}")
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Loss', fontsize=11)
    plt.title('COPY Mode', fontsize=13, fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # 子图2: Derivative 模式
    plt.subplot(1, 2, 2)
    plt.plot(results['derivative']['train_loss_history'], label='Train Loss', linewidth=2)
    plt.plot(results['derivative']['val_loss_history'], label='Val Loss', linewidth=2)
    plt.axhline(y=results['derivative']['best_val_loss'], color='r', linestyle='--',
                label=f"Best: {results['derivative']['best_val_loss']:.6f}")
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Loss', fontsize=11)
    plt.title('DERIVATIVE Mode', fontsize=13, fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('checkpoints/loss_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n对比图已保存到: checkpoints/loss_comparison.png")
    print("="*60)
