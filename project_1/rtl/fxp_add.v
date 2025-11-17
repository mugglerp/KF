module fxp_add
    #(parameter N = 20)
     (
         input  signed [N-1:0] a,
         input  signed [N-1:0] b,
         output signed [N  :0] y_full,   // N+1 位，包含进位
         output signed [N-1:0] y_trunc   // 丢弃最高位，保持同一 Q 格式
     );
    assign y_full  = $signed(a) + $signed(b);
    assign y_trunc = y_full[N-1:0]; // 纯丢位（wrap），不做饱和/舍入
endmodule