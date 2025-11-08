// ============================================================================
// subsystem_eq1.v : Combinational: O = I3*I3*(1-I1) + I2*I1
// Policy:
//   - t0 = I3*I3  -> take y_trunc (N bits, aligned to FRAC)
//   - pA = t0 * (1-I1) -> keep FULL 2N (no mid truncate)
//   - pB = I2 * I1     -> keep FULL 2N (no mid truncate)
//   - SUM = pA + pB    -> 2N(+1) add
//   - O_raw = (SUM >>> FRAC) then take N bits (final align-truncate)
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
    // 1. 1.0 in Q format
    localparam signed [N-1:0] ONE = (1'sb1 <<< FRAC);

    // 2. t0 = I3*I3 (use truncated N-bit)
    wire signed [2*N-1:0] t0_full;
    wire signed [N-1:0]   t0_trunc;
    fxp_mul #(N,FRAC) U_M0 (
                .a(I3), .b(I3),
                .y_full(t0_full),
                .y_trunc(t0_trunc)
            );

    // 3. t1 = (1 - I1) (use truncated N-bit)
    wire signed [N  :0] t1_full;
    wire signed [N-1:0] t1_trunc;
    fxp_sub #(N) U_S0 (
                .a(ONE), .b(I1),
                .y_full(t1_full),
                .y_trunc(t1_trunc)
            );

    // 4. pA = t0_trunc * t1_trunc : keep FULL 2N
    wire signed [2*N-1:0] pA_full;
    wire signed [N-1:0]   pA_trunc_unused;
    fxp_mul #(N,FRAC) U_M1 (
                .a(t0_trunc), .b(t1_trunc),
                .y_full(pA_full),
                .y_trunc(pA_trunc_unused)
            );

    // 5. pB = I2 * I1 : keep FULL 2N
    wire signed [2*N-1:0] pB_full;
    wire signed [N-1:0]   pB_trunc_unused;
    fxp_mul #(N,FRAC) U_M2 (
                .a(I2), .b(I1),
                .y_full(pB_full),
                .y_trunc(pB_trunc_unused)
            );

    // 6. SUM wide = pA_full + pB_full (2N+1 bits for safety)
    wire signed [2*N  :0] sum_wide;
    assign sum_wide = $signed({pA_full[2*N-1], pA_full}) + $signed({pB_full[2*N-1], pB_full});

    // 7. Final align-truncate: first shift into a net, then slice
    wire signed [2*N  :0] sum_shifted;   // (2N+1) >>> FRAC
    assign sum_shifted = sum_wide >>> FRAC;

    assign O_raw  = sum_shifted[N-1:0];      // take N bits after alignment
    assign O_full = sum_wide[2*N-1:0];       // expose lower 2N bits pre-align (debug)

endmodule
