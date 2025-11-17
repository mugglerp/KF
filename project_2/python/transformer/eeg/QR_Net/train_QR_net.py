import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F


# ======================
# 1. 模型：PositionalEncoding / QRTransformer
# ======================

class PositionalEncoding(nn.Module):
    """标准 sin-cos 位置编码，支持 [B, T, D] 输入"""
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [T, D]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [T,1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )  # [D/2]

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维
        pe = pe.unsqueeze(0)  # [1, T, D]

        self.register_buffer("pe", pe)  # 不参与训练但跟随设备移动

    def forward(self, x):
        """
        x: [B, T, D]
        """
        T = x.size(1)
        return x + self.pe[:, :T, :]


class QRTransformer(nn.Module):
    """
    Transformer 版 QR-Net（完整 2x2 Q/R）:
      输入: x_seq, z_seq 形状 [B, T, 2]
      输出: Q_seq, R_seq 形状 [B, T, 2, 2]
    """
    def __init__(self,
                 in_dim=8,        # [x, z, e, dx] -> 2+2+2+2 = 8
                 d_model=32,
                 nhead=4,
                 num_layers=2,
                 dim_feedforward=64,
                 dropout=0.1,
                 max_len=1024):
        super().__init__()
        self.d_model = d_model

        # 把 8 维特征映射到 d_model 维
        self.input_proj = nn.Linear(in_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,   # 输入输出用 [B, T, D]
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 通过 Cholesky 参数化预测 Q/R：
        # 每个时间步输出 3 个值 -> 下三角 L 的 (l11, l21, l22)
        # Q = L L^T 会得到 4 个元素全非零（一般情况）
        self.q_head = nn.Linear(d_model, 3)  # -> [l11_q, l21_q, l22_q]
        self.r_head = nn.Linear(d_model, 3)  # -> [l11_r, l21_r, l22_r]

    def forward(self, x_seq, z_seq):
        """
        x_seq: [B, T, 2]  干净状态 (EEG + 第二通道)
        z_seq: [B, T, 2]  带伪迹观测

        返回:
          Q_seq: [B, T, 2, 2]
          R_seq: [B, T, 2, 2]
        """
        B, T, _ = x_seq.shape

        # e = z - x
        e_seq = z_seq - x_seq  # [B, T, 2]

        # dx_t = x_t - x_{t-1}, t=0 时补 0
        dx_seq = x_seq[:, 1:, :] - x_seq[:, :-1, :]   # [B, T-1, 2]
        dx_seq = torch.cat(
            [torch.zeros(B, 1, 2, device=x_seq.device, dtype=x_seq.dtype), dx_seq],
            dim=1
        )  # [B, T, 2]

        # 拼特征: [x, z, e, dx] -> [B, T, 8]
        feat = torch.cat([x_seq, z_seq, e_seq, dx_seq], dim=-1)

        # 线性映射 + 位置编码 + Transformer
        h = self.input_proj(feat)          # [B, T, d_model]
        h = self.pos_encoder(h)            # [B, T, d_model]
        h = self.transformer(h)            # [B, T, d_model]

        # --------- Q：Cholesky 参数化 ----------
        q_params = self.q_head(h)          # [B, T, 3]
        # 对角用 softplus 保证正，非对角不限制
        l11_q = F.softplus(q_params[..., 0]) + 1e-6   # [B,T]
        l21_q = q_params[..., 1]                      # [B,T]
        l22_q = F.softplus(q_params[..., 2]) + 1e-6   # [B,T]

        # L_q = [[l11_q, 0],
        #        [l21_q, l22_q]]
        # Q = L_q L_q^T
        Q11 = l11_q * l11_q + l21_q * l21_q           # [B,T]
        Q12 = l21_q * l22_q                           # [B,T]
        Q22 = l22_q * l22_q                           # [B,T]

        # --------- R：Cholesky 参数化 ----------
        r_params = self.r_head(h)          # [B, T, 3]
        l11_r = F.softplus(r_params[..., 0]) + 1e-6
        l21_r = r_params[..., 1]
        l22_r = F.softplus(r_params[..., 2]) + 1e-6

        R11 = l11_r * l11_r + l21_r * l21_r           # [B,T]
        R12 = l21_r * l22_r                           # [B,T]
        R22 = l22_r * l22_r                           # [B,T]

        # 组装成对称 2x2 矩阵 [B, T, 2, 2]
        Q_seq = torch.stack([
            torch.stack([Q11, Q12], dim=-1),
            torch.stack([Q12, Q22], dim=-1),
        ], dim=-2)

        R_seq = torch.stack([
            torch.stack([R11, R12], dim=-1),
            torch.stack([R12, R22], dim=-1),
        ], dim=-2)

        return Q_seq, R_seq


# ======================
# 2. 数据加载 & 伪迹加噪
# ======================

def load_epoch_file(path, max_samples, seq_len):
    """
    通用 .npy 加载:
      path: EEG_all_epochs.npy / EMG_all_epochs.npy / EOG_all_epochs.npy
      返回: [N, T]，做了截断和标准化
    """
    arr = np.load(path).astype(np.float32)   # [N_all, L]
    if arr.ndim != 2:
        raise ValueError(f"{path} 期望是 2D (N,T)，实际 shape={arr.shape}")

    # 截断到 seq_len
    L = arr.shape[1]
    if L < seq_len:
        raise ValueError(f"{path} 长度 {L} 小于 seq_len={seq_len}")
    arr = arr[:, :seq_len]                   # [N_all, seq_len]

    N_all = arr.shape[0]
    N = min(max_samples, N_all)
    arr = arr[:N]                            # [N, seq_len]

    # 全局标准化到均值0, 方差1
    mean = arr.mean()
    std = arr.std() + 1e-6
    arr = (arr - mean) / std
    return arr  # [N, T]


def build_dataset_with_artifacts(
    mode: str,
    eeg_path: str,
    emg_path: str,
    eog_path: str,
    num_samples: int = 500,
    seq_len: int = 512,
    emg_scale: float = 0.7,
    eog_scale: float = 0.7,
    white_std: float = 0.1,
    deriv_scale: float = 0.5,
):
    """
    构造带伪迹噪声的数据:
      干净 EEG: 来自 EEG_all_epochs.npy
      伪迹: EMG_all_epochs.npy + EOG_all_epochs.npy
      白噪声: N(0, white_std^2)

    mode:
      "copy"       -> x: [EEG, EEG],           z: [EEG+伪迹, EEG+伪迹]
      "derivative" -> x: [EEG, dEEG],          z: [EEG+伪迹, d(EEG+伪迹)]

    返回:
      x_data: [N, T, 2] 干净状态
      z_data: [N, T, 2] 带伪迹 + 噪声后的观测
    """

    # 1) 干净 EEG
    eeg = load_epoch_file(eeg_path, num_samples, seq_len)   # [N, T]
    N, T = eeg.shape
    s = eeg                                                 # 干净 EEG

    # 2) EMG / EOG 伪迹
    #   这里多读一些样本，随机抽 N 条叠加
    emg_all = load_epoch_file(emg_path, max_samples=2000, seq_len=seq_len)  # [Nemg, T]
    eog_all = load_epoch_file(eog_path, max_samples=2000, seq_len=seq_len)  # [Neog, T]

    Nemg = emg_all.shape[0]
    Neog = eog_all.shape[0]

    emg_idx = np.random.randint(0, Nemg, size=N)
    eog_idx = np.random.randint(0, Neog, size=N)

    emg_sel = emg_all[emg_idx]   # [N, T]
    eog_sel = eog_all[eog_idx]   # [N, T]

    artifacts = emg_scale * emg_sel + eog_scale * eog_sel     # [N, T]
    white_noise = white_std * np.random.randn(N, T).astype(np.float32)

    s_noisy = s + artifacts + white_noise                     # [N, T] 带伪迹 EEG

    # 3) 构造 2 维干净状态 x_data
    if mode == "copy":
        # 通道1 = EEG, 通道2 = EEG
        s_2ch = s[..., None]                       # [N, T, 1]
        x_np = np.repeat(s_2ch, 2, axis=-1)        # [N, T, 2]

    elif mode == "derivative":
        # 通道1 = EEG, 通道2 = 导数 dEEG
        ds = s[:, 1:] - s[:, :-1]                  # [N, T-1]
        ds = np.pad(ds, ((0, 0), (1, 0)), mode="edge")  # [N, T]
        ds = ds * deriv_scale                      # 控制导数幅度

        x_np = np.stack([s, ds], axis=-1)          # [N, T, 2]

    else:
        raise ValueError("mode 必须是 'copy' 或 'derivative'")

    # 4) 构造 2 维观测 z_data（带噪）
    if mode == "copy":
        # 通道1 = 带伪迹 EEG, 通道2 = 带伪迹 EEG
        sn_2ch = s_noisy[..., None]                # [N, T, 1]
        z_np = np.repeat(sn_2ch, 2, axis=-1)       # [N, T, 2]

    elif mode == "derivative":
        # 通道1 = 带伪迹 EEG, 通道2 = 带伪迹 EEG 的导数
        dsn = s_noisy[:, 1:] - s_noisy[:, :-1]     # [N, T-1]
        dsn = np.pad(dsn, ((0, 0), (1, 0)), mode="edge")  # [N, T]
        dsn = dsn * deriv_scale

        z_np = np.stack([s_noisy, dsn], axis=-1)   # [N, T, 2]

    # 5) 转成 Tensor
    x_data = torch.from_numpy(x_np).float()        # [N, T, 2]
    z_data = torch.from_numpy(z_np).float()        # [N, T, 2]
    return x_data, z_data


# ======================
# 3. 训练循环（含 train/val 划分）
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


def train_qr_net(mode="copy",
                 eeg_path="./data/EEG_all_epochs.npy",
                 emg_path="./data/EMG_all_epochs.npy",
                 eog_path="./data/EOG_all_epochs.npy",
                 num_samples=500,
                 seq_len=512,
                 batch_size=32,
                 epochs=50,
                 lr=1e-3,
                 val_ratio=0.2,
                 device=None):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"使用伪迹噪声构造数据，mode = {mode}")

    # === 构造数据集（干净 x_data, 带噪 z_data） ===
    x_data, z_data = build_dataset_with_artifacts(
        mode=mode,
        eeg_path=eeg_path,
        emg_path=emg_path,
        eog_path=eog_path,
        num_samples=num_samples,
        seq_len=seq_len,
        emg_scale=0.7,
        eog_scale=0.7,
        white_std=0.1,
        deriv_scale=0.5,
    )

    x_data = x_data.to(device)   # [N, T, 2]
    z_data = z_data.to(device)

    N, T, _ = x_data.shape
    print(f"总样本数 N = {N}, 序列长度 T = {T}")

    # ======= 划分训练 / 验证 =======
    N_train = int(N * (1.0 - val_ratio))
    if N_train < 1 or N_train >= N:
        raise ValueError(f"val_ratio={val_ratio} 导致 N_train={N_train} 不合理")

    x_train = x_data[:N_train]
    z_train = z_data[:N_train]
    x_val   = x_data[N_train:]
    z_val   = z_data[N_train:]

    print(f"训练集: {x_train.shape[0]} 条，验证集: {x_val.shape[0]} 条")

    # Transformer 版 QR-Net
    model = QRTransformer(
        in_dim=8,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        dropout=0.1,
        max_len=seq_len,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    if mode == "copy":
        save_name = "qr_net_tf_copy2ch_artifacts.pt"
    else:
        save_name = "qr_net_tf_derivative_artifacts.pt"

    # ================== 训练循环 ==================
    for epoch in range(1, epochs + 1):
        # ---- 训练 ----
        model.train()
        perm = torch.randperm(x_train.shape[0], device=device)
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, x_train.shape[0], batch_size):
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
            for i in range(0, x_val.shape[0], batch_size):
                batch_x = x_val[i:i + batch_size]
                batch_z = z_val[i:i + batch_size]
                loss = compute_batch_loss(model, batch_x, batch_z, mse)
                if loss is None:
                    continue
                val_loss_sum += loss.item()
                val_batches += 1
        val_loss = val_loss_sum / max(val_batches, 1)

        print(f"Epoch {epoch}/{epochs} - train_loss: {train_loss:.6f}, val_loss: {val_loss:.6f}")

    torch.save(model.state_dict(), save_name)
    print(f"训练完成，权重已保存到 {save_name}")


if __name__ == "__main__":
    # ====== 版本 A：两通道都复制 EEG ======
    train_qr_net(
        mode="copy",
        eeg_path="./data/EEG_all_epochs.npy",
        emg_path="./data/EMG_all_epochs.npy",
        eog_path="./data/EOG_all_epochs.npy",
        num_samples=1500,
        seq_len=512,
        batch_size=32,
        epochs=100,
        lr=1e-3,
        val_ratio=0.2,
    )

    # ====== 版本 B：第二维 = 导数 ======
    train_qr_net(
        mode="derivative",
        eeg_path="./data/EEG_all_epochs.npy",
        emg_path="./data/EMG_all_epochs.npy",
        eog_path="./data/EOG_all_epochs.npy",
        num_samples=1500,
        seq_len=512,
        batch_size=32,
        epochs=100,
        lr=1e-3,
        val_ratio=0.2,
    )
