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
# 2. 两种数据构造方式
# ======================

def load_clean_eeg(path, num_samples, seq_len):
    """从 EEG_all_epochs.npy 读干净 EEG，并裁剪/归一化"""
    eeg = np.load(path).astype(np.float32)  # [N_all, 512]
    eeg = eeg[:num_samples, :seq_len]       # [N, T]

    # 全局归一化到均值0、方差1
    mean = eeg.mean()
    std = eeg.std() + 1e-6
    eeg = (eeg - mean) / std
    return eeg  # [N, T]


# --- 版本 A：第二维 = 复制 EEG -----------------
def build_dataset_copy2ch(path, num_samples=500, seq_len=512, noise_std=0.2):
    """
    第一维 = EEG, 第二维 = 再复制一份 EEG
    返回:
      x_data: [N, T, 2] 干净状态
      z_data: [N, T, 2] 带噪测量
    """
    eeg = load_clean_eeg(path, num_samples, seq_len)   # [N, T]

    s = eeg[..., None]           # [N, T, 1]
    x_data = np.repeat(s, 2, axis=-1)  # [N, T, 2]  两通道相同

    x_data = torch.from_numpy(x_data)  # float32
    z_data = x_data + noise_std * torch.randn_like(x_data)
    return x_data, z_data


# --- 版本 B：第二维 = 导数 -----------------
def build_dataset_with_derivative(path, num_samples=500, seq_len=512,
                                  noise_std=0.2, deriv_scale=0.5):
    """
    第一维 = EEG, 第二维 = 导数 (差分)，再做适当缩放
    返回:
      x_data: [N, T, 2]
      z_data: [N, T, 2]
    """
    eeg = load_clean_eeg(path, num_samples, seq_len)   # [N, T]
    s = eeg                                            # [N, T]

    # 差分: d_t = s_t - s_{t-1}, 第一个点补0
    ds = s[:, 1:] - s[:, :-1]              # [N, T-1]
    ds = np.pad(ds, ((0, 0), (1, 0)), mode="edge")  # [N, T]
    ds = ds * deriv_scale                  # 缩小一下幅度，便于训练

    x = np.stack([s, ds], axis=-1)         # [N, T, 2]
    x_data = torch.from_numpy(x)           # float32

    z_data = x_data + noise_std * torch.randn_like(x_data)
    return x_data, z_data


# ======================
# 3. 训练循环（两版通用，用不同的 build_dataset 即可）
# ======================

def train_qr_net(mode="copy",
                 eeg_path="EEG_all_epochs.npy",
                 num_samples=500,
                 seq_len=512,
                 batch_size=32,
                 epochs=50,
                 lr=1e-3,
                 device=None):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # === 选用哪种数据构造方式 ===
    if mode == "copy":
        print("使用 2 通道 = [EEG, EEG] 模式")
        x_data, z_data = build_dataset_copy2ch(eeg_path, num_samples, seq_len)
        save_name = "qr_net_copy2ch.pt"
    elif mode == "derivative":
        print("使用 2 通道 = [EEG, 导数] 模式")
        x_data, z_data = build_dataset_with_derivative(eeg_path, num_samples, seq_len)
        save_name = "qr_net_derivative.pt"
    else:
        raise ValueError("mode 必须是 'copy' 或 'derivative'")

    x_data = x_data.to(device)   # [N, T, 2]
    z_data = z_data.to(device)

    N, T, _ = x_data.shape
    print(f"x_data shape: {x_data.shape}, z_data shape: {z_data.shape}")

    model = QRNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    mse = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(N)
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            batch_x = x_data[idx]  # [B, T, 2]
            batch_z = z_data[idx]  # [B, T, 2]

            B, T, _ = batch_x.shape
            if T < 2:
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
        num_samples=500,
        seq_len=512,
        batch_size=32,
        epochs=50,
        lr=1e-3,
    )
