module goldschmidt_struct
#(
    parameter integer N    = 20,
    parameter integer FRAC = 10
)(
    input  signed [N-1:0] x,
    output signed [N-1:0] recip
);

    // ========================
    // 预计算常量（N=20,FRAC=10）
    // ========================
    localparam signed [N-1:0] T1 = 20'sd724;   // 0.7071 * 1024
    localparam signed [N-1:0] T2 = 20'sd886;   // 0.8660 * 1024

    localparam signed [N-1:0] A1 = -20'sd957;  // -0.934 * 1024
    localparam signed [N-1:0] B1 =  20'sd1981; //  1.934 * 1024
    localparam signed [N-1:0] A2 = -20'sd724;  // -0.707 * 1024
    localparam signed [N-1:0] B2 =  20'sd1748; //  1.707 * 1024
    localparam signed [N-1:0] A3 = -20'sd591;  // -0.577 * 1024
    localparam signed [N-1:0] B3 =  20'sd1616; //  1.577 * 1024

    localparam signed [N-1:0] CONST_TWO = 20'sd2048; // 2 * 1024

    // abs & zero detection
    wire x_neg = x[N-1];
    wire signed [N-1:0] x_abs = x_neg ? -x : x;
    wire is_zero = (x_abs == 0);

    // normalize magnitude
    integer msb_i;
    reg signed [N-1:0] u_norm;
    integer den_shift;

    integer i;
    always @* begin
        msb_i = 0;
        for (i=N-2; i>=0; i=i-1)
            if (x_abs[i] && (msb_i == 0))
                msb_i = i;

        if (msb_i > FRAC) begin
            u_norm    = x_abs >>> (msb_i - FRAC);
            den_shift = FRAC - msb_i;
        end else begin
            u_norm    = x_abs <<< (FRAC - msb_i);
            den_shift = FRAC - msb_i;
        end
    end

    wire seg1 = (u_norm < T1);
    wire seg2 = ~seg1 && (u_norm < T2);

    wire signed [N-1:0] a_sel = seg1 ? A1 : (seg2 ? A2 : A3);
    wire signed [N-1:0] b_sel = seg1 ? B1 : (seg2 ? B2 : B3);

    wire signed [N-1:0] au, y0, y1, y2, y3;
    wire signed [N-1:0] t0, f0, t1, f1, t2, f2;

    fxp_mul #(N,FRAC) M0(.a(a_sel),  .b(u_norm), .y_trunc(au));
    assign y0 = au + b_sel;

    fxp_mul #(N,FRAC) M1(.a(u_norm), .b(y0), .y_trunc(t0)); assign f0 = CONST_TWO - t0;
    fxp_mul #(N,FRAC) M2(.a(y0),     .b(f0), .y_trunc(y1));

    fxp_mul #(N,FRAC) M3(.a(u_norm), .b(y1), .y_trunc(t1)); assign f1 = CONST_TWO - t1;
    fxp_mul #(N,FRAC) M4(.a(y1),     .b(f1), .y_trunc(y2));

    fxp_mul #(N,FRAC) M5(.a(u_norm), .b(y2), .y_trunc(t2)); assign f2 = CONST_TWO - t2;
    fxp_mul #(N,FRAC) M6(.a(y2),     .b(f2), .y_trunc(y3));

    wire signed [N-1:0] recip_mag =
         (den_shift >= 0) ? (y3 <<< den_shift) : (y3 >>> (-den_shift));

    assign recip = is_zero ? {1'b0,{(N-1){1'b1}}} : (x_neg ? -recip_mag : recip_mag);

endmodule
