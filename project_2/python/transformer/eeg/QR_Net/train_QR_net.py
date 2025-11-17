import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F


# ======================
# 1. 模型：QMatrixNet / RMatrixNet / QRNet
# ======================

class QMatrixNet(nn.Module):
    """根据 x_now, x_prev 估计 Q 矩阵（2x2 对角）"""
    def __init__(self):
        super().__init__()
        self.WQ = nn.Parameter(torch.randn(2))
        self.WK = nn.Parameter(torch.randn(2))
        self.WV = nn.Parameter(torch.randn(2))

    def forward(self, x_now, x_prev):
        """
        x_now, x_prev: [N, 2]
        返回 Q: [N, 2, 2]
        """
        dx = x_now - x_prev  # [N, 2]
        q = (dx * self.WQ).sum(dim=-1, keepdim=True)  # [N, 1]
        k = (dx * self.WK).sum(dim=-1, keepdim=True)  # [N, 1]
        v = (dx * self.WV).sum(dim=-1, keepdim=True)  # [N, 1]

        att = q * k          # [N, 1]
        out = att * v        # [N, 1]

        # 保证对角非负，避免非正定
        out = F.softplus(out) + 1e-6   # [N, 1]

        Q11 = out.view(-1)             # [N]
        Q22 = out.view(-1)             # [N]
        zeros = torch.zeros_like(Q11)

        Q = torch.stack([
            torch.stack([Q11, zeros], dim=-1),
            torch.stack([zeros, Q22], dim=-1),
        ], dim=-2)  # [N, 2, 2]
        return Q


class RMatrixNet(nn.Module):
    """根据测量残差 e = z - z_hat 估计 R 矩阵（2x2 对角）"""
    def __init__(self):
        super().__init__()
        self.WQ = nn.Parameter(torch.randn(2))
        self.WK = nn.Parameter(torch.randn(2))

    def forward(self, z, z_hat):
        """
        z, z_hat: [N, 2]
        返回 R: [N, 2, 2]
        """
        e = z - z_hat                    # [N, 2]
        q = (e * self.WQ).sum(dim=-1, keepdim=True)  # [N, 1]
        k = (e * self.WK).sum(dim=-1, keepdim=True)  # [N, 1]
        att = q * k                      # [N, 1]
        out = F.softplus(att) + 1e-6     # [N, 1]

        R11 = out.view(-1)
        R22 = out.view(-1)
        zeros = torch.zeros_like(R11)

        R = torch.stack([
            torch.stack([R11, zeros], dim=-1),
            torch.stack([zeros, R22], dim=-1),
        ], dim=-2)  # [N, 2, 2]
        return R


class QRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_net = QMatrixNet()
        self.r_net = RMatrixNet()

    def forward(self, x_now, x_prev, z, z_hat):
        Q = self.q_net(x_now, x_prev)
        R = self.r_net(z, z_hat)
        return Q, R


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
# 3. 训练循环（两版通用）
# ======================

def train_qr_net(mode="copy",
                 eeg_path="EEG_all_epochs.npy",
                 emg_path="EMG_all_epochs.npy",
                 eog_path="EOG_all_epochs.npy",
                 num_samples=500,
                 seq_len=512,
                 batch_size=32,
                 epochs=50,
                 lr=1e-3,
                 device=None):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"使用伪迹噪声构造数据，mode = {mode}")

    # === 根据模式构造数据集（干净 x_data, 带噪 z_data） ===
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
    print(f"x_data shape: {x_data.shape}, z_data shape: {z_data.shape}")

    model = QRNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    mse = nn.MSELoss()

    if mode == "copy":
        save_name = "qr_net_copy2ch_artifacts.pt"
    else:
        save_name = "qr_net_derivative_artifacts.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(N)
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            batch_x = x_data[idx]  # [B, T, 2]
            batch_z = z_data[idx]  # [B, T, 2]

            B, T_cur, _ = batch_x.shape
            if T_cur < 2:
                continue

            # 用 t=1..T-1 的时刻做训练（需要 x_prev）
            x_now  = batch_x[:, 1:, :]      # [B, T-1, 2]
            x_prev = batch_x[:, :-1, :]     # [B, T-1, 2]
            z_now  = batch_z[:, 1:, :]      # [B, T-1, 2]
            z_hat  = x_now                  # 简单认为预测测量 = 干净值

            # 展平时间维，方便一次前向
            x_now_flat  = x_now.reshape(-1, 2)   # [B*(T-1), 2]
            x_prev_flat = x_prev.reshape(-1, 2)
            z_now_flat  = z_now.reshape(-1, 2)
            z_hat_flat  = z_hat.reshape(-1, 2)

            Q_pred, R_pred = model(x_now_flat, x_prev_flat, z_now_flat, z_hat_flat)
            # Q_pred, R_pred: [B*(T-1), 2, 2]

            # ===== 简化版目标协方差 =====
            # 这里用 batch 内 + 时间维的协方差作为 target
            dx = x_now - x_prev             # [B, T-1, 2]
            e  = z_now - z_hat              # [B, T-1, 2]

            dx_flat = dx.reshape(-1, 2)     # [B*(T-1), 2]
            e_flat  = e.reshape(-1, 2)

            # 去均值
            dx_flat = dx_flat - dx_flat.mean(dim=0, keepdim=True)
            e_flat  = e_flat  - e_flat.mean(dim=0, keepdim=True)

            # 计算协方差矩阵（简化版：用整体平均）
            Q_target = dx_flat.unsqueeze(2) * dx_flat.unsqueeze(1)  # [N',2,2]
            Q_target = Q_target.mean(dim=0, keepdim=True)           # [1,2,2]

            R_target = e_flat.unsqueeze(2) * e_flat.unsqueeze(1)   # [N',2,2]
            R_target = R_target.mean(dim=0, keepdim=True)          # [1,2,2]

            # broadcast 到 Q_pred/R_pred 的 batch 上
            Q_target = Q_target.expand_as(Q_pred)
            R_target = R_target.expand_as(R_pred)

            loss_q = mse(Q_pred, Q_target)
            loss_r = mse(R_pred, R_target)

            # 正定约束：惩罚负对角
            q_diag = torch.stack([Q_pred[:, 0, 0], Q_pred[:, 1, 1]], dim=-1)  # [N',2]
            r_diag = torch.stack([R_pred[:, 0, 0], R_pred[:, 1, 1]], dim=-1)
            loss_pos = F.relu(-q_diag).mean() + F.relu(-r_diag).mean()

            loss = loss_q + loss_r + 0.1 * loss_pos

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        epoch_loss /= max(num_batches, 1)
        print(f"Epoch {epoch}/{epochs} - loss: {epoch_loss:.6f}")

    torch.save(model.state_dict(), save_name)
    print(f"训练完成，权重已保存到 {save_name}")


if __name__ == "__main__":
    # ====== 版本 A：两通道都复制 EEG ======
    train_qr_net(
        mode="copy",
        eeg_path="EEG_all_epochs.npy",
        emg_path="EMG_all_epochs.npy",
        eog_path="EOG_all_epochs.npy",
        num_samples=500,
        seq_len=512,
        batch_size=32,
        epochs=50,
        lr=1e-3,
    )

    # ====== 版本 B：第二维 = 导数 ======
    train_qr_net(
        mode="derivative",
        eeg_path="EEG_all_epochs.npy",
        emg_path="EMG_all_epochs.npy",
        eog_path="EOG_all_epochs.npy",
        num_samples=500,
        seq_len=512,
        batch_size=32,
        epochs=50,
        lr=1e-3,
    )
