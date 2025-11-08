# Fixed-Point Kalman Filter (2x2) — Verilog-2001 Multi-File Project

- Word width: parameterizable via ``FXP_N`` (default 20). Fractional bits ``FXP_FRAC`` default to N/2.
- Arithmetic units (add, sub, mul) expose **both** full-precision and **truncated & fraction-aligned** outputs.
- Truncation is defined as taking the aligned middle bits for products and dropping the carry MSB for sums, preserving the binary point.
- Architectures implemented per your spec:
  - Prior state vector **serial** (2 mul, 1 add; 8 cycles)
  - Prior covariance **semi-parallel** (4 mul, 2 add; 8 cycles)
  - Kalman Gain **semi-parallel**, with **INV serial** (INV uses 2 mul + 1 sub in its iterative core)
  - Estimated output **serial** (2 mul, 1 add; 4 cycles)
  - Posterior state vector **serial** (2 sub, 2 mul, 1 add; 5 cycles)
  - Posterior covariance **semi-parallel** (4 mul, 2 add, 2 sub; 8 cycles)
  - Q **serial** (1 sub, 1 add, 1 mul; 3 cycles)
  - R uses **two subsystem_eq1** + subtract + modulus (3 cycles)
  - `subsystem_eq1`: `O = I3*I3*(1-I1) + I2*I1` (combinational)

> NOTE: The provided `top_kf.v` stitches blocks in a simple sequence. It asserts `done` at the end of the last block.
> To exactly reach an overall **34-cycle** pipeline as in your target, you would overlap blocks (start the next block early) and/or tighten internal schedules. The blocks here keep the requested **per-block** resources/latencies; the system-level controller can be re-timed to your 34-cycle micro-schedule.

## File Tree
```
rtl/
  include/
    fxp_types.vh
  common/
    fxp_trunc.v
    fxp_abs.v
  arith/
    fxp_add.v
    fxp_sub.v
    fxp_mul.v
    goldschmidt_serial.v
  blocks/
    subsystem_eq1.v
    inv2_serial.v
    prior_state_serial.v
    prior_cov_semipar.v
    est_output_serial.v
    post_state_serial.v
    post_cov_semipar.v
    q_serial.v
    r_block.v
  top/
    top_kf.v
```

## Synthesis Notes
- Pure Verilog-2001.
- No testbench is included (per your request).
- All matrices are 2×2 (order-2) and vectors are length-2.
- The divider is implemented via a serial **Goldschmidt** reciprocal approximator using only multipliers and subtractors; number of iterations tunable by `RECIP_ITERS`.


## 34-cycle Gantt (0-based, 0..33)

| Cycles | Block | Notes |
|---|---|---|
| 00–07 | Prior State (PS) + Prior Cov (PC) | PC uses **Q_prev** |
| 08–11 | Estimated Output (Z) | needs x_prior |
| 12–14 | R (two Eq1 + sub + |·|) → **R_next** | for next frame |
| 08–24 | Kalman Gain (KG) | uses **R_prev** and P_prior |
| 25–29 | Posterior State (PST) | needs K + z_hat |
| 25–32 | Posterior Cov (POC) | needs K + P_prior |
| 30–32 | Q (serial) → **Q_next** | for next frame |
| 33    | **DONE**; commit R_next→R_prev, Q_next→Q_prev, P_post→P_prev | frame boundary |
