// fxp_sub.v
`include "fxp_types.vh"

module fxp_sub
    #(parameter N = `FXP_N)
     (
         input  signed [N-1:0] a,
         input  signed [N-1:0] b,
         output signed [N  :0] y_full,
         output signed [N-1:0] y_trunc
     );
    assign y_full  = $signed(a) - $signed(b);
    assign y_trunc = y_full[N-1:0];
endmodule