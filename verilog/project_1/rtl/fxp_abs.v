`include "fxp_types.vh"

module fxp_abs
    #( parameter N=`FXP_N )
     (
         input  signed [N-1:0] a,
         output signed [N-1:0] y
     );
    assign y = (a[N-1]) ? -a : a;
endmodule
