import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataloaders.EEG_all_DataLoader import EEG_all_Dataloader
from models.QRNet_model import QRTransformer


# ======================
# 数据准备和训练工具函数
# ======================

def compute_batch_loss(model, batch_x, batch_z, mse):
    """
    给定一个 batch，计算 loss（完整协方差匹配）
    返回标量 loss
    """
    B, T_cur, _ = batch_x.shape
    if T_cur < 2:
        return None

    # 用 Transformer 对整段序列建模
    Q_seq, R_seq = model(batch_x, batch_z)  # [B, T, 2, 2]

    # 我们的 target 是针对 t=1..T-1 的增量 dx / 残差 e
    Q_pred = Q_seq[:, 1:, :, :].reshape(-1, 2, 2)  # [N',2,2]
    R_pred = R_seq[:, 1:, :, :].reshape(-1, 2, 2)  # [N',2,2]

    # ===== 计算完整协方差矩阵 target =====
    x_now  = batch_x[:, 1:, :]                     # [B, T-1, 2]
    x_prev = batch_x[:, :-1, :]                    # [B, T-1, 2]
    z_now  = batch_z[:, 1:, :]                     # [B, T-1, 2]
    z_hat  = x_now                                 # 预测测量 = 干净值

    dx = x_now - x_prev                            # [B, T-1, 2]
    e  = z_now - z_hat                             # [B, T-1, 2]

    dx_flat = dx.reshape(-1, 2)                    # [N', 2]
    e_flat  = e.reshape(-1, 2)

    # 去均值
    dx_flat = dx_flat - dx_flat.mean(dim=0, keepdim=True)
    e_flat  = e_flat  - e_flat.mean(dim=0, keepdim=True)

    # 完整协方差：E[xx^T]  -> [2,2]
    Q_cov = dx_flat.unsqueeze(2) * dx_flat.unsqueeze(1)   # [N',2,2]
    Q_target = Q_cov.mean(dim=0, keepdim=True)            # [1,2,2]

    R_cov = e_flat.unsqueeze(2) * e_flat.unsqueeze(1)     # [N',2,2]
    R_target = R_cov.mean(dim=0, keepdim=True)            # [1,2,2]

    # broadcast 到每个时间步预测上
    Q_target = Q_target.expand_as(Q_pred)  # [N',2,2]
    R_target = R_target.expand_as(R_pred)

    # 四个元素全部参与 MSE
    loss_q = mse(Q_pred, Q_target)
    loss_r = mse(R_pred, R_target)

    loss = loss_q + loss_r
    return loss


def prepare_eeg_data(dataloader, num_train=10, num_val=10, num_test=10, seq_len=512):
    """
    按照 EEGDenoising.py 的方式准备数据
    从 dataloader 中提取训练、验证、测试数据
    
    Args:
        dataloader: EEG_all_Dataloader 实例
        num_train: 训练样本数量
        num_val: 验证样本数量
        num_test: 测试样本数量
        seq_len: 序列长度
    
    Returns:
        noiseEEG_train, EEG_train, noiseEEG_val, EEG_val, noiseEEG_test, EEG_test
        每个形状: [N, 1, seq_len]
    """
    noiseEEG_train = torch.zeros((num_train, 1, seq_len))
    EEG_train = torch.zeros((num_train, 1, seq_len))
    noiseEEG_val = torch.zeros((num_val, 1, seq_len))
    EEG_val = torch.zeros((num_val, 1, seq_len))
    noiseEEG_test = torch.zeros((num_test, 1, seq_len))
    EEG_test = torch.zeros((num_test, 1, seq_len))
    
    # 训练集：从 0 开始
    for i in range(num_train):
        index = seq_len * i
        noiseEEG_train[i] = dataloader.observations[0, index:index+seq_len, 0].float()
        EEG_train[i] = dataloader.dataset[0, index:index+seq_len, 0].float()
    
    # 验证集：从 5120 (10*512) 开始
    for i in range(num_val):
        index = seq_len * i
        noiseEEG_val[i] = dataloader.observations[0, index+5120:index+5632, 0].float()
        EEG_val[i] = dataloader.dataset[0, index+5120:index+5632, 0].float()
    
    # 测试集：从 10240 (20*512) 开始
    for i in range(num_test):
        index = seq_len * i
        noiseEEG_test[i] = dataloader.observations[0, index+10240:index+10752, 0].float()
        EEG_test[i] = dataloader.dataset[0, index+10240:index+10752, 0].float()
    
    return noiseEEG_train, EEG_train, noiseEEG_val, EEG_val, noiseEEG_test, EEG_test


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
    按照 EEGDenoising.py 的策略训练 QR-Net
    
    Args:
        mode: "copy" 或 "derivative"
        datapoints: 数据窗口大小（序列长度）
        samples: 样本索引列表
        snr_db: 信噪比（dB），0表示信号强度和噪声强度相等
        noise_color: 噪声类型（0=白噪声, 1=粉噪声, 2=棕噪声）
        num_train: 训练样本数量
        num_val: 验证样本数量
        num_test: 测试样本数量
        batch_size: 批次大小
        epochs: 训练轮数
        lr: 学习率
        wd: 权重衰减
        device: 设备
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("="*50)
    print(f"使用 EEG_all_Dataloader 加载数据，mode = {mode}")
    print(f"数据窗口大小 = {datapoints}, SNR = {snr_db} dB, 噪声颜色 = {noise_color}")
    print(f"训练参数: epochs={epochs}, batch_size={batch_size}, lr={lr}, wd={wd}")
    print("="*50)

    # === 使用 DataLoader 构造数据集（与 EEGDenoising.py 一致）===
    if samples is None:
        samples = [0]  # 默认样本索引

    # 创建数据集
    dataloader = EEG_all_Dataloader(
        datapoints=datapoints,
        samples=samples,
        snr_db=snr_db,
        noise_color=noise_color
    )

    print(f"数据集总长度: {len(dataloader)}")
    print(f"观测数据形状: {dataloader.observations.shape}")
    print(f"干净数据形状: {dataloader.dataset.shape}")

    # ======= 按照 EEGDenoising.py 方式准备数据 =======
    noiseEEG_train, EEG_train, noiseEEG_val, EEG_val, noiseEEG_test, EEG_test = \
        prepare_eeg_data(dataloader, num_train, num_val, num_test, datapoints)

    print(f"训练集: {num_train} 条, 验证集: {num_val} 条, 测试集: {num_test} 条")
    print(f"训练数据形状: noiseEEG={noiseEEG_train.shape}, EEG={EEG_train.shape}")

    # === 转换数据格式：[N, 1, T] -> [N, T, 2] ===
    # 定义导数计算函数
    def add_derivative(data, scale=0.5):
        # data: [N, 1, T]
        N, _, T = data.shape
        deriv = data[:, :, 1:] - data[:, :, :-1]  # [N, 1, T-1]
        deriv = torch.cat([deriv[:, :, :1], deriv], dim=2)  # [N, 1, T] 补齐
        deriv = deriv * scale
        return torch.cat([data, deriv], dim=1).permute(0, 2, 1)  # [N, T, 2]
    
    # 干净数据：第二通道根据 mode 决定
    if mode == "copy":
        # 两通道都是 EEG
        x_train = torch.cat([EEG_train, EEG_train], dim=1).permute(0, 2, 1)  # [N, T, 2]
        x_val = torch.cat([EEG_val, EEG_val], dim=1).permute(0, 2, 1)
        x_test = torch.cat([EEG_test, EEG_test], dim=1).permute(0, 2, 1)
    else:  # derivative
        # 第二通道是导数
        x_train = add_derivative(EEG_train)
        x_val = add_derivative(EEG_val)
        x_test = add_derivative(EEG_test)
    
    # 带噪数据：同样处理
    if mode == "copy":
        z_train = torch.cat([noiseEEG_train, noiseEEG_train], dim=1).permute(0, 2, 1)
        z_val = torch.cat([noiseEEG_val, noiseEEG_val], dim=1).permute(0, 2, 1)
        z_test = torch.cat([noiseEEG_test, noiseEEG_test], dim=1).permute(0, 2, 1)
    else:
        z_train = add_derivative(noiseEEG_train)
        z_val = add_derivative(noiseEEG_val)
        z_test = add_derivative(noiseEEG_test)
    
    # 移动到设备
    x_train, z_train = x_train.to(device), z_train.to(device)
    x_val, z_val = x_val.to(device), z_val.to(device)
    x_test, z_test = x_test.to(device), z_test.to(device)

    # Transformer 版 QR-Net
    model = QRTransformer(
        in_dim=8,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        dropout=0.1,
        max_len=datapoints,
    ).to(device)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # 使用 AdamW 优化器（带权重衰减）
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    mse = nn.MSELoss()

    if mode == "copy":
        save_name = "checkpoints/qr_net_copy_eeg.pt"
    else:
        save_name = "checkpoints/qr_net_derivative_eeg.pt"

    # ================== 训练循环（与 EEGDenoising.py 策略一致）==================
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        # ---- 训练 ----
        model.train()
        perm = torch.randperm(num_train)
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, num_train, batch_size):
            idx = perm[i:i + batch_size]
            batch_x = x_train[idx]  # [B, T, 2]
            batch_z = z_train[idx]  # [B, T, 2]

            loss = compute_batch_loss(model, batch_x, batch_z, mse)
            if loss is None:
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        train_loss = epoch_loss / max(num_batches, 1)

        # ---- 验证 ----
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for i in range(0, num_val, batch_size):
                batch_x = x_val[i:i + batch_size]
                batch_z = z_val[i:i + batch_size]
                loss = compute_batch_loss(model, batch_x, batch_z, mse)
                if loss is None:
                    continue
                val_loss_sum += loss.item()
                val_batches += 1
        val_loss = val_loss_sum / max(val_batches, 1)

        # 每10个epoch打印一次
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} - train_loss: {train_loss:.6f}, val_loss: {val_loss:.6f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_name)
            if epoch % 10 == 0:
                print(f"  → 保存最佳模型 (val_loss: {best_val_loss:.6f})")

    print(f"\n训练完成！最佳验证损失: {best_val_loss:.6f}")
    print(f"模型权重已保存到: {save_name}")
    
    # ================== 测试阶段 ==================
    print("\n" + "="*50)
    print("开始测试...")
    model.load_state_dict(torch.load(save_name))
    model.eval()
    
    test_loss_sum = 0.0
    test_batches = 0
    with torch.no_grad():
        for i in range(0, num_test, batch_size):
            batch_x = x_test[i:i + batch_size]
            batch_z = z_test[i:i + batch_size]
            loss = compute_batch_loss(model, batch_x, batch_z, mse)
            if loss is None:
                continue
            test_loss_sum += loss.item()
            test_batches += 1
    
    test_loss = test_loss_sum / max(test_batches, 1)
    test_loss_db = 10 * torch.log10(torch.tensor(test_loss))
    
    print(f"测试损失: {test_loss:.6f}")
    print(f"测试损失 (dB): {test_loss_db:.2f}")
    print("="*50)


if __name__ == "__main__":
    # ====== 按照 EEGDenoising.py 的配置训练 ======
    train_qr_net(
        mode="copy",
        datapoints=512,      # 序列长度
        samples=[0],         # 使用的样本索引
        snr_db=0,            # 信噪比 (dB)，0表示信号强度和噪声强度相等
        noise_color=0,       # 0=白噪声, 1=粉噪声, 2=棕噪声
        num_train=10,        # 训练样本数
        num_val=10,          # 验证样本数
        num_test=10,         # 测试样本数
        batch_size=8,        # 批次大小（与EEGDenoising.py一致）
        epochs=300,          # 训练轮数（与EEGDenoising.py一致）
        lr=1e-4,             # 学习率（与EEGDenoising.py一致）
        wd=1e-3,             # 权重衰减（与EEGDenoising.py一致）
    )
