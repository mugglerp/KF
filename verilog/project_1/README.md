# 2x2 固定点卡尔曼滤波器 — Verilog-2001 实现

## 项目概述

本项目实现了一个高度优化的 2×2 定点卡尔曼滤波器，采用串行和半并行混合架构，总流水线压缩至 **34 个时钟周期**完成一帧计算。

### 核心特性

- **参数化位宽**: 通过 `N` 和 `FRAC` 参数控制字长（默认 N=20, FRAC=10）
- **定点运算**: 所有算术单元（加、减、乘）提供全精度和截断输出
- **资源优化**: 通过模块复用和流水线重叠最小化硬件开销
- **34 周期帧**: 从 start 到 done 拉高共 34 个时钟周期
- **模块化设计**: 各子模块独立可测，便于集成和验证

### 模块架构与时序

| 模块 | 架构 | 资源 | 周期 | 说明 |
|------|------|------|------|------|
| Prior State | 串行 | 2 乘法器, 1 加法器 | 8 | x̂(k|k-1) = Ax(k-1) + Bu(k) |
| Prior Covariance | 半并行 | 4 乘法器, 2 加法器 | 9 | P̂(k|k-1) = APA^T + Q |
| Estimated Output | 串行 | 2 乘法器, 1 加法器 | 4 | ẑ(k) = Hx̂(k|k-1) |
| Kalman Gain | 半并行 | 4 乘法器, 2 加法器, 1 除法器 | 17 | K = P̂H^T(HP̂H^T + R)^{-1} |
| Posterior State | 串行 | 2 乘法器, 1 加法器, 2 减法器 | 5 | x(k) = x̂ + K(z - ẑ) |
| Posterior Covariance | 半并行 | 4 乘法器, 2 加法器, 4 减法器 | 9 | P(k) = P̂ - KHP̂ |
| Q 噪声协方差 | 串行 | 1 乘法器, 1 加法器, 1 减法器 | 3 | 自适应更新 |
| R 噪声协方差 | 串行 | subsystem_eq1 × 2 | 3 | 自适应更新 |

### 关键优化技术

1. **加法器复用设计**: Prior Covariance 使用 2 个加法器通过时分复用完成列求和和 +Q 运算
2. **启动时序优化**: start 上升沿当拍即配置乘法器，节省 1 周期
3. **流水线重叠**: Kalman Gain 在 C8 启动，与 Prior Covariance 完成时刻对齐
4. **后端提前**: Posterior 模块在 C25 启动，相比原设计提前 2 周期

## 目录结构

```
.
├── rtl/                    # RTL 源文件
│   ├── fxp_add.v          # 定点加法器（组合逻辑）
│   ├── fxp_sub.v          # 定点减法器
│   ├── fxp_mul.v          # 定点乘法器
│   ├── fxp_abs.v          # 绝对值运算
│   ├── goldschmidt_struct.v  # Goldschmidt 除法器（迭代求倒数）
│   ├── subsystem_eq1.v    # 自适应噪声协方差子系统
│   ├── inv2_serial.v      # 2x2 矩阵求逆（串行）
│   ├── prior_state_serial.v      # 先验状态估计
│   ├── prior_cov_semipar.v       # 先验协方差估计（优化版）
│   ├── est_output_serial.v       # 估计输出
│   ├── kg_semipar.v             # 卡尔曼增益
│   ├── post_state_serial.v      # 后验状态估计
│   ├── post_cov_semipar.v       # 后验协方差估计
│   ├── q_serial.v               # Q 矩阵自适应更新
│   ├── r_serial.v               # R 矩阵自适应更新
│   └── top_kf.v                 # 顶层卡尔曼滤波器
├── tb/                     # 测试文件
│   ├── tb_*_min.v         # 各模块最小测试
│   ├── tb_top_kf_34c_basic.v     # 顶层基础测试（34周期）
│   └── tb_top_kf_manyframes_rlc.v # RLC 电路多帧闭环测试
├── vivado/                 # Xilinx 仿真环境
│   └── project_1/sim/behav/xsim/
│       ├── compile.bat    # 编译脚本
│       ├── elaborate.bat  # 综合脚本
│       ├── simulate.bat   # 仿真脚本
│       └── *.tcl          # TCL 控制脚本
├── matlab/                 # MATLAB 后处理
│   └── plot_rlc_kf_xsim.m # 波形绘制脚本
└── README.md
```

## 技术要点

### 语言与工具
- **HDL**: Verilog-2001（无 SystemVerilog 特性）
- **仿真器**: Xilinx XSIM
- **后处理**: MATLAB（可选）

### 算法实现
- **矩阵维度**: 所有矩阵为 2×2，向量为 2×1
- **除法器**: 采用 Goldschmidt 迭代算法，仅使用乘法器和减法器
- **定点截断**: 乘法结果取中间对齐位，加法丢弃进位 MSB，保持小数点位置
- **噪声协方差**: Q 和 R 通过 subsystem_eq1 自适应更新


## 34 周期时序甘特图（C0-C33）

| 周期 | 模块 | 说明 |
|------|------|------|
| C0–C7 | Prior State | 先验状态估计 |
| C0–C8 | Prior Covariance | 先验协方差估计（使用 Q_prev） |
| C8–C11 | Estimated Output | 估计输出（需要 x_prior） |
| C8–C24 | Kalman Gain | 卡尔曼增益（使用 R_prev 和 P_prior） |
| C10–C12 | Q Update | Q 矩阵更新 → Q_next |
| C12–C14 | R Update | R 矩阵更新 → R_next |
| C25–C29 | Posterior State | 后验状态估计（需要 K 和 z_hat） |
| C25–C33 | Posterior Covariance | 后验协方差估计（需要 K 和 P_prior） |
| C33 | **DONE** | 提交 R_next→R_prev, Q_next→Q_prev, P_post→P_prev |

### 关键时序说明
- **并行执行**: Prior State 和 Prior Covariance 在 C0 同时启动
- **流水线重叠**: Kalman Gain 在 C8 启动，无需等待 Prior Covariance 完全结束
- **后端优化**: Posterior 模块在 C25 启动，比 Kalman Gain 结束（C24）提前 1 周期准备

## 仿真运行

### 编译与仿真（Windows）

```bash
cd vivado/project_1/project_1.sim/sim_1/behav/xsim

# 1. 编译
compile.bat

# 2. 综合
elaborate.bat

# 3. 仿真（选择测试）
# 基础测试
simulate.bat tb_top_kf_34c_basic

# 多帧闭环测试
simulate.bat tb_top_kf_manyframes_rlc
```

### 预期输出

```
Info: frame done @ 34 cycles
PASS frame 0 : done at C33 (34-cycle)
...
DONE 4000 frames (34-cycle handshake).
```

## 性能指标

- **帧周期**: 34 时钟周期
- **吞吐率**: 1 帧 / 34 周期
- **时钟频率**: 取决于综合后关键路径（未综合）
- **资源占用**: 
  - 乘法器: 4 个（复用）
  - 加法器: 2 个（复用）
  - 减法器: 4 个（复用）
  - 除法器: 1 个（Goldschmidt 迭代）

## 验证状态

✅ **所有模块单元测试通过**  
✅ **顶层 34 周期时序验证通过**  
✅ **4000 帧闭环 RLC 仿真验证通过**  
✅ **数据精度满足定点运算要求**

## 参考文献

本实现基于经典卡尔曼滤波算法，针对 FPGA 硬件进行了以下优化：
- 定点运算替代浮点
- 串行/半并行混合架构
- 模块级流水线设计
- 自适应噪声协方差更新
