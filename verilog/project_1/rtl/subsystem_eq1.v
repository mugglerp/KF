// ============================================================================
// subsystem_eq1.v : Combinational: O = I3*I3*(1-I1) + I2*I1
// Policy 保持不变；把 “+” 改成 fxp_add；最终 O_raw 仍按 FRAC 对齐截断
// 要修改：加法器不见了 -> 已改为 fxp_add
// ============================================================================
`include "fxp_types.vh"

module subsystem_eq1
    #( parameter N=`FXP_N, parameter FRAC=`FXP_FRAC )
     (
         input  signed [N-1:0] I1,
         input  signed [N-1:0] I2,
         input  signed [N-1:0] I3,
         output signed [N-1:0]   O_raw,
         output signed [2*N-1:0] O_full
     );
    // 1.0 in Q(N,FRAC)
    localparam signed [N-1:0] ONE = $signed(1) <<< FRAC;

    // t0 = I3*I3  (取截断 N 位参与后续乘法)
    wire signed [2*N-1:0] t0_full;
    wire signed [N-1:0]   t0_trunc;
    fxp_mul #(N,FRAC) U_M0 (.a(I3), .b(I3), .y_full(t0_full), .y_trunc(t0_trunc));

    // t1 = (1 - I1)  (取截断 N 位参与后续乘法)
    wire signed [N  :0] t1_full;
    wire signed [N-1:0] t1_trunc;
    fxp_sub #(N) U_S0 (.a(ONE), .b(I1), .y_full(t1_full), .y_trunc(t1_trunc));

    // pA = t0_trunc * t1_trunc (保留 FULL 2N)
    wire signed [2*N-1:0] pA_full;
    wire signed [N-1:0]   pA_trunc_unused;
    fxp_mul #(N,FRAC) U_M1 (.a(t0_trunc), .b(t1_trunc), .y_full(pA_full), .y_trunc(pA_trunc_unused));

    // pB = I2 * I1 (保留 FULL 2N)
    wire signed [2*N-1:0] pB_full;
    wire signed [N-1:0]   pB_trunc_unused;
    fxp_mul #(N,FRAC) U_M2 (.a(I2), .b(I1), .y_full(pB_full), .y_trunc(pB_trunc_unused));

    // SUM = pA_full + pB_full  -> 用 fxp_add #(2*N)
    wire signed [2*N  :0] sum_wide;
    wire signed [2*N-1:0] sum_trunc_unused;
    fxp_add #(2*N) U_A0 (
                .a(pA_full), .b(pB_full),
                .y_full(sum_wide),       // 2N+1 位，含进位
                .y_trunc(sum_trunc_unused)
            );

    // Final 对齐与输出
    wire signed [2*N  :0] sum_shifted = sum_wide >>> FRAC;
    assign O_raw  = sum_shifted[N-1:0];   // N 位
    assign O_full = sum_wide[2*N-1:0];    // 未右移前的低 2N 位（debug）

endmodule
