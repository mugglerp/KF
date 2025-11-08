// ============================================================================
// goldschmidt_struct.v  (piecewise-initialized, 3-iter, pure combinational)
// 1/x  (x in fixed-point Q(N-FRAC).FRAC, signed two's complement)
// - Normalize |x| -> u in [0.5, 1.0)
// - Piecewise linear init: y0 = a*u + b   (3 segments, constants auto-scaled)
// - 3 iterations: y <- y*(2 - u*y)
// - Denormalize: recip = sign * y * 2^(FRAC - msb_pos(|x|))
// ============================================================================
`include "fxp_types.vh"

module goldschmidt_struct
#(
    parameter integer N    = `FXP_N,
    parameter integer FRAC = `FXP_FRAC,

    // ---- 可选：整数 Override（非 0 则使用），否则用自动推导 ----
    parameter integer T1_OVERRIDE = 0,
    parameter integer T2_OVERRIDE = 0,
    parameter integer A1_OVERRIDE = 0, parameter integer B1_OVERRIDE = 0,
    parameter integer A2_OVERRIDE = 0, parameter integer B2_OVERRIDE = 0,
    parameter integer A3_OVERRIDE = 0, parameter integer B3_OVERRIDE = 0
)(
    input  signed [N-1:0] x,
    output signed [N-1:0] recip
);

    // ---------------- 常量缩放：按 FRAC 自动推导整数 ----------------
    localparam integer SCALE = (1 << FRAC);

    // Real -> Q(FRAC) 量化（四舍五入，Verilog-2001 合法写法）
    function integer q_from_real;
        input real r;
        begin
            if (r >= 0.0)
                q_from_real = $rtoi(r * (1.0 * SCALE) + 0.5);
            else
                q_from_real = $rtoi(r * (1.0 * SCALE) - 0.5);
        end
    endfunction

    // 阈值（自动）
    localparam integer T1_AUTO = q_from_real(0.7071067811865476); // 1/sqrt(2)
    localparam integer T2_AUTO = q_from_real(0.8660254037844386); // sqrt(3)/2

    // 线性系数（自动）
    localparam integer A1_AUTO = q_from_real(-0.934);
    localparam integer B1_AUTO = q_from_real( 1.934);
    localparam integer A2_AUTO = q_from_real(-0.707);
    localparam integer B2_AUTO = q_from_real( 1.707);
    localparam integer A3_AUTO = q_from_real(-0.577);
    localparam integer B3_AUTO = q_from_real( 1.577);

    // 实际使用的整数常量：优先采用 Override，否则用 AUTO
    localparam integer T1 = (T1_OVERRIDE != 0) ? T1_OVERRIDE : T1_AUTO;
    localparam integer T2 = (T2_OVERRIDE != 0) ? T2_OVERRIDE : T2_AUTO;
    localparam integer A1 = (A1_OVERRIDE != 0) ? A1_OVERRIDE : A1_AUTO;
    localparam integer B1 = (B1_OVERRIDE != 0) ? B1_OVERRIDE : B1_AUTO;
    localparam integer A2 = (A2_OVERRIDE != 0) ? A2_OVERRIDE : A2_AUTO;
    localparam integer B2 = (B2_OVERRIDE != 0) ? B2_OVERRIDE : B2_AUTO;
    localparam integer A3 = (A3_OVERRIDE != 0) ? A3_OVERRIDE : A3_AUTO;
    localparam integer B3 = (B3_OVERRIDE != 0) ? B3_OVERRIDE : B3_AUTO;

    // ---------------- helpers & constants -----------------------------
    // 2.0 in Q format
    wire signed [N-1:0] CONST_TWO = $signed(2) <<< FRAC;

    // abs(x)
    wire        x_neg  = x[N-1];
    wire signed [N-1:0] x_abs = x_neg ? -x : x;

    // special case: x == 0 -> saturation
    wire is_zero = (x_abs == {N{1'b0}});

    // ---------------- find MSB position of |x| ------------------------
    function integer msb_pos_mag;
        input [N-2:0] v;
        integer i, found;
        begin
            msb_pos_mag = 0;
            found = 0;
            for (i = N-2; i >= 0; i = i - 1)
                if (!found && v[i]) begin
                    msb_pos_mag = i;
                    found = 1;
                end
        end
    endfunction

    wire [N-2:0] mag_bits = x_abs[N-2:0];
    integer msb_i;
    reg signed [N-1:0] u_norm;   // normalized |x| to [0.5,1)
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
    wire seg1 = (u_norm < $signed(T1));             // [0.5, T1)
    wire seg2 = (~seg1) && (u_norm < $signed(T2));  // [T1, T2)
    wire seg3 = ~(seg1 | seg2);                     // [T2, 1.0)

    wire signed [N-1:0] a_sel = seg1 ? $signed(A1) : (seg2 ? $signed(A2) : $signed(A3));
    wire signed [N-1:0] b_sel = seg1 ? $signed(B1) : (seg2 ? $signed(B2) : $signed(B3));

    // a*u
    wire signed [2*N-1:0] au_full;
    wire signed [N-1:0]   au_trunc;
    fxp_mul #(N,FRAC) UM0 (.a(a_sel), .b(u_norm), .y_full(au_full), .y_trunc(au_trunc));

    // y0 = a*u + b  （对齐同一 Q 格式）
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
    // **修复：显式端口连接（Verilog-2001），避免 .name 隐式连接/语法错误**
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

    // sign & zero handling (simple saturation on x==0)
    wire signed [N-1:0] POS_MAX = {1'b0, {(N-1){1'b1}} };
    wire signed [N-1:0] NEG_MAX = ({1'b1, {(N-1){1'b0}} }) + {{(N-1){1'b0}}, 1'b1};

    wire signed [N-1:0] recip_signed      = x_neg ? -recip_mag : recip_mag;
    wire signed [N-1:0] recip_sat_on_zero = x_neg ? NEG_MAX : POS_MAX;

    assign recip = is_zero ? recip_sat_on_zero : recip_signed;

endmodule
