`include "fxp_types.vh"

module fxp_mul
    #(parameter N = `FXP_N, parameter FRAC = `FXP_FRAC)
     (
         input  signed [N-1:0] a,
         input  signed [N-1:0] b,
         output signed [2*N-1:0] y_full,   // 完整 2N 位 (2FRAC 小数)
         output signed [N-1:0]   y_trunc   // 按小数点对齐后截取 N 位
     );
    wire signed [2*N-1:0] prod = $signed(a) * $signed(b);
    assign y_full  = prod;

    // ！！对齐切片：保持符号的同时，取 prod[FRAC + N - 1 : FRAC]
    // 这等价于对 prod 做"向零截断的算术右移 FRAC 位"，然后取低 N 位。
    assign y_trunc = prod[FRAC + N - 1 : FRAC];
endmodule
