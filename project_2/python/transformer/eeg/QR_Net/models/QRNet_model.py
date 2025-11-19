import math
import torch
import torch.nn as nn
from torch.nn import functional as F


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
                 in_dim: int = 8,        # [x, z, e, dx] -> 2+2+2+2 = 8
                 d_model: int = 32,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dim_feedforward: int = 64,
                 dropout: float = 0.1,
                 max_len: int = 1024):
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
