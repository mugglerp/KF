"""QR-Net 模型定义：基于 Transformer 的卡尔曼滤波噪声协方差矩阵预测网络

该模块实现了用于预测卡尔曼滤波器过程噪声协方差矩阵 Q 和测量噪声协方差矩阵 R 的深度学习模型。
采用 Transformer 架构提取时序特征，通过 Cholesky 分解确保协方差矩阵的半正定性。
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class PositionalEncoding(nn.Module):
    """正弦-余弦位置编码模块
    
    为 Transformer 输入序列添加位置信息，使模型能够感知序列中不同时间步的相对位置。
    使用 Attention is All You Need 论文中的标准位置编码方案。
    
    位置编码公式：
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Args:
        d_model: 模型特征维度
        max_len: 支持的最大序列长度
    """
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        # 初始化位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        
        # 位置索引 [0, 1, 2, ..., max_len-1] -> [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        
        # 频率缩放因子，控制不同维度的周期
        # div_term[i] = 1 / (10000^(2i/d_model))
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        # 偶数维度使用 sin，奇数维度使用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 添加 batch 维度 [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # 注册为 buffer：不参与梯度更新，但会随模型保存/加载和设备迁移
        self.register_buffer("pe", pe)

    def forward(self, x):
        """将位置编码加到输入特征上
        
        Args:
            x: 输入张量，形状 [B, T, D]
                B: batch size
                T: 序列长度
                D: 特征维度（必须等于 d_model）
        
        Returns:
            添加位置编码后的张量，形状 [B, T, D]
        """
        T = x.size(1)
        # 只使用前 T 个位置的编码，支持变长序列
        return x + self.pe[:, :T, :]


class QRTransformer(nn.Module):
    """基于 Transformer 的 QR 协方差矩阵预测网络
    
    该网络用于预测卡尔曼滤波器中的噪声协方差矩阵：
        - Q: 过程噪声协方差矩阵（状态转移不确定性）
        - R: 测量噪声协方差矩阵（观测不确定性）
    
    核心设计思想：
        1. 输入多模态时序特征（干净信号、带噪观测、残差、状态差分）
        2. 通过 Transformer 编码器提取全局时序依赖关系
        3. 使用 Cholesky 分解参数化确保输出协方差矩阵的半正定性
        4. 对每个时间步独立预测完整的 2×2 协方差矩阵
    
    输入输出：
        输入: 
            x_seq: 干净状态序列 [B, T, 2]  (EEG 信号及其导数/副本)
            z_seq: 带噪观测序列 [B, T, 2]  (含伪迹的 EEG 观测)
        输出:
            Q_seq: 过程噪声协方差序列 [B, T, 2, 2]
            R_seq: 测量噪声协方差序列 [B, T, 2, 2]
    
    Args:
        in_dim: 输入特征维度（默认 8 = x(2) + z(2) + e(2) + dx(2)）
        d_model: Transformer 隐藏层维度
        nhead: 多头注意力头数
        num_layers: Transformer 编码器层数
        dim_feedforward: 前馈网络维度
        dropout: Dropout 概率
        max_len: 支持的最大序列长度
    """
    def __init__(self,
                 in_dim: int = 8,
                 d_model: int = 32,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dim_feedforward: int = 64,
                 dropout: float = 0.1,
                 max_len: int = 1024):
        super().__init__()
        self.d_model = d_model

        # 输入投影层：将原始特征映射到 Transformer 特征空间
        self.input_proj = nn.Linear(in_dim, d_model)
        
        # 位置编码：为序列添加时间位置信息
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)

        # Transformer 编码器层配置
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # 使用 [Batch, Time, Feature] 格式
        )
        # 堆叠多层 Transformer 编码器
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Q/R 预测头：使用 Cholesky 分解参数化
        # 对于 2×2 对称矩阵，下三角 Cholesky 因子 L 有 3 个独立参数：
        #     L = [[l11,  0 ],
        #          [l21, l22]]
        # 协方差矩阵通过 Cov = L @ L^T 计算，自动满足半正定性
        self.q_head = nn.Linear(d_model, 3)  # 输出 [l11_q, l21_q, l22_q]
        self.r_head = nn.Linear(d_model, 3)  # 输出 [l11_r, l21_r, l22_r]

    def forward(self, x_seq, z_seq):
        """前向传播：预测每个时间步的 Q 和 R 协方差矩阵
        
        Args:
            x_seq: 干净状态序列 [B, T, 2]
                   第一通道：EEG 信号
                   第二通道：EEG 导数（derivative 模式）或 EEG 副本（copy 模式）
            z_seq: 带噪观测序列 [B, T, 2]
                   对应 x_seq 的带伪迹版本（EOG/EMG/ECG 污染）
        
        Returns:
            Q_seq: 过程噪声协方差矩阵序列 [B, T, 2, 2]
                   描述状态转移过程中的不确定性
            R_seq: 测量噪声协方差矩阵序列 [B, T, 2, 2]
                   描述观测过程中的噪声特性
        """
        B, T, _ = x_seq.shape

        # === 步骤 1: 构造多模态输入特征 ===
        
        # 测量残差：观测值与真实状态的差异
        e_seq = z_seq - x_seq  # [B, T, 2]

        # 状态差分：捕捉状态的时间变化率
        # dx_t = x_t - x_{t-1}，对于 t=0 填充零向量
        dx_seq = x_seq[:, 1:, :] - x_seq[:, :-1, :]  # [B, T-1, 2]
        dx_seq = torch.cat(
            [torch.zeros(B, 1, 2, device=x_seq.device, dtype=x_seq.dtype), dx_seq],
            dim=1
        )  # [B, T, 2]

        # 拼接所有特征通道：[干净状态, 带噪观测, 测量残差, 状态差分]
        # 形状: [B, T, 8] = [B, T, 2+2+2+2]
        feat = torch.cat([x_seq, z_seq, e_seq, dx_seq], dim=-1)

        # === 步骤 2: Transformer 编码 ===
        
        # 特征投影到 Transformer 空间
        h = self.input_proj(feat)     # [B, T, d_model]
        
        # 添加位置编码（时序信息）
        h = self.pos_encoder(h)       # [B, T, d_model]
        
        # Transformer 编码器提取全局时序依赖
        h = self.transformer(h)       # [B, T, d_model]

        # === 步骤 3: Q 矩阵预测（Cholesky 分解参数化）===
        
        # 预测 Cholesky 下三角因子的 3 个参数
        q_params = self.q_head(h)  # [B, T, 3]
        
        # 对角元素使用 softplus 确保为正（协方差矩阵对角线必须 > 0）
        l11_q = F.softplus(q_params[..., 0]) + 1e-6  # 添加小常数防止数值不稳定
        l21_q = q_params[..., 1]                      # 非对角元素无约束
        l22_q = F.softplus(q_params[..., 2]) + 1e-6

        # 通过 Q = L @ L^T 计算协方差矩阵的 4 个元素
        # L = [[l11,  0 ],    Q = [[l11^2 + l21^2,  l21*l22    ],
        #      [l21, l22]]         [l21*l22,        l22^2      ]]
        Q11 = l11_q * l11_q + l21_q * l21_q  # 方差（第一维）
        Q12 = l21_q * l22_q                   # 协方差（跨维度）
        Q22 = l22_q * l22_q                   # 方差（第二维）

        # === 步骤 4: R 矩阵预测（相同的 Cholesky 参数化）===
        
        r_params = self.r_head(h)  # [B, T, 3]
        l11_r = F.softplus(r_params[..., 0]) + 1e-6
        l21_r = r_params[..., 1]
        l22_r = F.softplus(r_params[..., 2]) + 1e-6

        R11 = l11_r * l11_r + l21_r * l21_r
        R12 = l21_r * l22_r
        R22 = l22_r * l22_r

        # === 步骤 5: 组装对称协方差矩阵 ===
        
        # Q 矩阵 [B, T, 2, 2]，注意 Q12 = Q21（对称性）
        Q_seq = torch.stack([
            torch.stack([Q11, Q12], dim=-1),  # 第一行 [Q11, Q12]
            torch.stack([Q12, Q22], dim=-1),  # 第二行 [Q21, Q22]
        ], dim=-2)

        # R 矩阵 [B, T, 2, 2]
        R_seq = torch.stack([
            torch.stack([R11, R12], dim=-1),
            torch.stack([R12, R22], dim=-1),
        ], dim=-2)

        return Q_seq, R_seq
