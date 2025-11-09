// ============================================================================
// goldschmidt_struct.v  (piecewise-initialized, 3-iter, pure combinational)
// 1/x  (x in Q(N-FRAC).FRAC, signed two's complement)
// - Normalize |x| -> u in [0.5, 1.0)
// - Piecewise linear init: y0 = a*u + b (3 segments)
// - 3 iterations: y <- y*(2 - u*y)
// - Denormalize: recip = sign * y << (FRAC - msb_pos(|x|))
// NOTE: This version removes all 'real' usage and pre-quantizes constants
//       for N=20, FRAC=10 (SCALE=1024). Fully synthesizable.
// ============================================================================

`include "fxp_types.vh"

module goldschmidt_struct #(
    parameter integer N    = `FXP_N,     // expect 20
    parameter integer FRAC = `FXP_FRAC,  // expect 10

    // ---- Optional integer overrides (Q(FRAC)) ----
    parameter integer T1_OVERRIDE = 0,
    parameter integer T2_OVERRIDE = 0,
    parameter integer A1_OVERRIDE = 0, parameter integer B1_OVERRIDE = 0,
    parameter integer A2_OVERRIDE = 0, parameter integer B2_OVERRIDE = 0,
    parameter integer A3_OVERRIDE = 0, parameter integer B3_OVERRIDE = 0
)(
    input  signed [N-1:0] x,
    output signed [N-1:0] recip
);
    // ---------------- constants (pre-quantized for FRAC=10) ----------------
    // SCALE = 1<<FRAC = 1024
    // T1 ≈ 1/sqrt(2) = 0.70710678 -> 724
    // T2 ≈ sqrt(3)/2 = 0.86602540 -> 887
    // Segment coefficients (y0 = a*u + b), Q10:
    // seg1: a=-0.934 -> -956,  b=1.934 -> 1980
    // seg2: a=-0.707 -> -724,  b=1.707 -> 1748
    // seg3: a=-0.577 -> -590,  b=1.577 -> 1614
    localparam integer T1_AUTO = 724;
    localparam integer T2_AUTO = 887;
    localparam integer A1_AUTO = -956, B1_AUTO = 1980;
    localparam integer A2_AUTO = -724, B2_AUTO = 1748;
    localparam integer A3_AUTO = -590, B3_AUTO = 1614;

    // Actual constants (allow overrides if non-zero)
    localparam integer T1 = (T1_OVERRIDE != 0) ? T1_OVERRIDE : T1_AUTO;
    localparam integer T2 = (T2_OVERRIDE != 0) ? T2_OVERRIDE : T2_AUTO;
    localparam integer A1 = (A1_OVERRIDE != 0) ? A1_OVERRIDE : A1_AUTO;
    localparam integer B1 = (B1_OVERRIDE != 0) ? B1_OVERRIDE : B1_AUTO;
    localparam integer A2 = (A2_OVERRIDE != 0) ? A2_OVERRIDE : A2_AUTO;
    localparam integer B2 = (B2_OVERRIDE != 0) ? B2_OVERRIDE : B2_AUTO;
    localparam integer A3 = (A3_OVERRIDE != 0) ? A3_OVERRIDE : A3_AUTO;
    localparam integer B3 = (B3_OVERRIDE != 0) ? B3_OVERRIDE : B3_AUTO;

    // ---------------- helpers & constants -----------------------------
    wire signed [N-1:0] CONST_TWO = $signed(2) <<< FRAC;  // 2.0 in Q(FRAC)

    // abs(x)
    wire        x_neg  = x[N-1];
    wire signed [N-1:0] x_abs = x_neg ? -x : x;

    // x == 0 ? (avoid divide-by-zero; saturate)
    wire is_zero = (x_abs == {N{1'b0}});

    // ---------------- find MSB position of |x| ------------------------
    // find MSB position of |x| (ignore sign bit)
    function integer msb_pos_mag;
        input [N-2:0] v;
        integer i;
        begin
            msb_pos_mag = 0;
            for (i = N-2; i >= 0; i = i - 1) begin
                if (v[i]) begin
                    msb_pos_mag = i;
                    // break out of loop: Verilog-2001 does NOT support disable-label,
                    // so we return directly.
                    return; 
                end
            end
        end
    endfunction


    wire [N-2:0] mag_bits = x_abs[N-2:0];
    integer msb_i;
    reg signed [N-1:0] u_norm;   // normalized |x| to [0.5, 1.0)
    integer den_shift;           // FRAC - msb_i

    always @* begin
        msb_i = msb_pos_mag(mag_bits);
        if (msb_i > FRAC) begin
            u_norm    = x_abs >>> (msb_i - FRAC);
            den_shift = FRAC - msb_i;      // negative
        end else begin
            u_norm    = x_abs <<< (FRAC - msb_i);
            den_shift = FRAC - msb_i;      // >= 0
        end
    end

    // ---------------- piecewise y0 = a*u + b -------------------------
    wire seg1 = (u_norm < $signed(T1));
    wire seg2 = (~seg1) && (u_norm < $signed(T2));
    wire seg3 = ~(seg1 | seg2);

    wire signed [N-1:0] a_sel = seg1 ? $signed(A1) : (seg2 ? $signed(A2) : $signed(A3));
    wire signed [N-1:0] b_sel = seg1 ? $signed(B1) : (seg2 ? $signed(B2) : $signed(B3));

    // a*u
    wire signed [2*N-1:0] au_full;
    wire signed [N-1:0]   au_trunc;
    fxp_mul #(N,FRAC) UM0 (.a(a_sel), .b(u_norm), .y_full(au_full), .y_trunc(au_trunc));

    // y0 = a*u + b  （同一Q格式）
    wire signed [N:0]     y0_sum = $signed(au_trunc) + $signed(b_sel);
    wire signed [N-1:0]   y0     = y0_sum[N-1:0];

    // ---------------- 3 Goldschmidt iterations -----------------------
    // y1 = y0*(2 - u*y0)
    wire signed [2*N-1:0] t0_full;
    wire signed [N-1:0]   t0;
    fxp_mul #(N,FRAC) UM1 (.a(u_norm), .b(y0), .y_full(t0_full), .y_trunc(t0));
    wire signed [N-1:0]   f0 = $signed(CONST_TWO) - $signed(t0);
    wire signed [2*N-1:0] y1_full;
    wire signed [N-1:0]   y1;
    fxp_mul #(N,FRAC) UM2 (.a(y0), .b(f0), .y_full(y1_full), .y_trunc(y1));

    // y2 = y1*(2 - u*y1)
    wire signed [2*N-1:0] t1_full;
    wire signed [N-1:0]   t1;
    fxp_mul #(N,FRAC) UM3 (.a(u_norm), .b(y1), .y_full(t1_full), .y_trunc(t1));
    wire signed [N-1:0]   f1 = $signed(CONST_TWO) - $signed(t1);
    wire signed [2*N-1:0] y2_full;
    wire signed [N-1:0]   y2;
    fxp_mul #(N,FRAC) UM4 (.a(y1), .b(f1), .y_full(y2_full), .y_trunc(y2));

    // y3 = y2*(2 - u*y2)
    wire signed [2*N-1:0] t2_full;
    wire signed [N-1:0]   t2;
    fxp_mul #(N,FRAC) UM5 (.a(u_norm), .b(y2), .y_full(t2_full), .y_trunc(t2));
    wire signed [N-1:0]   f2 = $signed(CONST_TWO) - $signed(t2);
    wire signed [2*N-1:0] y3_full;
    wire signed [N-1:0]   y3;
    fxp_mul #(N,FRAC) UM6 (.a(y2), .b(f2), .y_full(y3_full), .y_trunc(y3));

    // ---------------- denormalize & sign -----------------------------
    wire signed [N-1:0] recip_mag_shifted =
         (den_shift >= 0) ? ($signed(y3) <<< den_shift)
                          : ($signed(y3) >>> (-den_shift));
    wire signed [N-1:0] recip_mag = recip_mag_shifted;

    // saturation constants
    wire signed [N-1:0] POS_MAX = {1'b0, {(N-1){1'b1}} };
    wire signed [N-1:0] NEG_MAX = {1'b1, {(N-1){1'b0}} } + {{(N-1){1'b0}}, 1'b1};

    wire signed [N-1:0] recip_signed      = x_neg ? -recip_mag : recip_mag;
    wire signed [N-1:0] recip_sat_on_zero = x_neg ? NEG_MAX : POS_MAX;

    assign recip = is_zero ? recip_sat_on_zero : recip_signed;

endmodule
