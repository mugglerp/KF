// ============================================================================
// prior_cov_semipar.v : S = A*P*A^T + Q   (2x2, semi-parallel, 8 cycles)
// Ports strictly follow the paper figure:
//   A: a00 a01 a10 a11
//   P: p00 p01 p10 p11
//   Q: q00 q01 q10 q11
// Outputs (UPPERCASE, naming per your rule):
//   P_PRIOR00 P_PRIOR01 P_PRIOR10 P_PRIOR11
// ----------------------------------------------------------------------------
// MUL keeps 2N full precision; column-sums and +Q happen in 2N domain.
// Only when feeding next stage or at final outputs we truncate to N bits
// aligned to FRAC. Q is sign-extended to 2N and left-shifted by FRAC
// to align with mul full products.
//
// Cycle map (0..7):
//   0: m0=p00*a00, m1=p01*a01, m2=p10*a00, m3=p11*a01
//   1: t00=(m0+m1)>>FRAC, t10=(m2+m3)>>FRAC
//   2: m0=p00*a10, m1=p01*a11, m2=p10*a10, m3=p11*a11
//   3: t01=(m0+m1)>>FRAC, t11=(m2+m3)>>FRAC
//   4: m0=a00*t00, m1=a01*t10, m2=a10*t00, m3=a11*t10
//   5: s00_2N=(m0+m1)+Q00_2N, s10_2N=(m2+m3)+Q10_2N
//   6: m0=a00*t01, m1=a01*t11, m2=a10*t01, m3=a11*t11
//   7: s01_2N=(m0+m1)+Q01_2N, s11_2N=(m2+m3)+Q11_2N, done=1
// ============================================================================
`include "fxp_types.vh"

module prior_cov_semipar
#( parameter integer N=`FXP_N, parameter integer FRAC=`FXP_FRAC )
(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     start,

    // A (2x2) - lowercase inputs
    input  wire signed [N-1:0]      a00, a01, a10, a11,
    // P (2x2)
    input  wire signed [N-1:0]      p00, p01, p10, p11,
    // Q (2x2)
    input  wire signed [N-1:0]      q00, q01, q10, q11,

    // control
    output reg                      done,

    // UPPERCASE data outputs with your naming rule
    output wire signed [N-1:0]      P_PRIOR00, P_PRIOR01, P_PRIOR10, P_PRIOR11
);

    // -------- 2N -> N truncation (FRAC aligned) --------
    function [N-1:0] trunc2N_to_N;
        input signed [2*N-1:0] x2n;
        begin
            trunc2N_to_N = x2n[FRAC+N-1 : FRAC];
        end
    endfunction

    // -------- Q aligned to 2N domain --------
    wire signed [2*N-1:0] Q00_2N = ({{N{q00[N-1]}}, q00}) <<< FRAC;
    wire signed [2*N-1:0] Q01_2N = ({{N{q01[N-1]}}, q01}) <<< FRAC;
    wire signed [2*N-1:0] Q10_2N = ({{N{q10[N-1]}}, q10}) <<< FRAC;
    wire signed [2*N-1:0] Q11_2N = ({{N{q11[N-1]}}, q11}) <<< FRAC;

    // -------- 4 MUL (use full outputs), 2 adders in 2N domain --------
    reg  signed [N-1:0] m0_a, m0_b, m1_a, m1_b, m2_a, m2_b, m3_a, m3_b;
    wire signed [2*N-1:0] m0_f, m1_f, m2_f, m3_f;
    wire signed [N-1:0]   _t0,_t1,_t2,_t3;  // unused trunc ports

    fxp_mul #(N,FRAC) U_M0 (.a(m0_a), .b(m0_b), .y_full(m0_f), .y_trunc(_t0));
    fxp_mul #(N,FRAC) U_M1 (.a(m1_a), .b(m1_b), .y_full(m1_f), .y_trunc(_t1));
    fxp_mul #(N,FRAC) U_M2 (.a(m2_a), .b(m2_b), .y_full(m2_f), .y_trunc(_t2));
    fxp_mul #(N,FRAC) U_M3 (.a(m3_a), .b(m3_b), .y_full(m3_f), .y_trunc(_t3));

    wire signed [2*N:0]   sum01_f, sum23_f;
    wire signed [2*N-1:0] sum01_2N, sum23_2N;
    fxp_add #(2*N) U_SUM01 (.a(m0_f), .b(m1_f), .y_full(sum01_f), .y_trunc(sum01_2N));
    fxp_add #(2*N) U_SUM23 (.a(m2_f), .b(m3_f), .y_full(sum23_f), .y_trunc(sum23_2N));

    // -------- T matrix (N domain) --------
    reg signed [N-1:0] t00, t01, t10, t11;

    // -------- S registers in 2N, N-truncated at outputs --------
    reg  signed [2*N-1:0] s00_2N, s01_2N, s10_2N, s11_2N;

    // Rename Sxx(N) to P_PRIORxx(N)
    assign P_PRIOR00 = trunc2N_to_N(s00_2N);
    assign P_PRIOR01 = trunc2N_to_N(s01_2N);
    assign P_PRIOR10 = trunc2N_to_N(s10_2N);
    assign P_PRIOR11 = trunc2N_to_N(s11_2N);

    // -------- 8-cycle controller --------
    reg [3:0] cyc;
    reg       running;

    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            cyc<=0; running<=1'b0; done<=1'b0;
            m0_a<=0; m0_b<=0; m1_a<=0; m1_b<=0; m2_a<=0; m2_b<=0; m3_a<=0; m3_b<=0;
            t00<=0; t01<=0; t10<=0; t11<=0;
            s00_2N<=0; s01_2N<=0; s10_2N<=0; s11_2N<=0;
        end else begin
            done <= 1'b0;

            if(start && !running) begin
                running<=1'b1;
                cyc<=0;
            end else if(running) begin
                cyc <= cyc + 4'd1;
            end

            case(cyc)
                // 0) T column-0: P * [a00;a01]
                4'd0: begin
                    m0_a<=p00; m0_b<=a00;
                    m1_a<=p01; m1_b<=a01;
                    m2_a<=p10; m2_b<=a00;
                    m3_a<=p11; m3_b<=a01;
                end
                // 1) latch t00,t10 (N)
                4'd1: begin
                    t00 <= trunc2N_to_N(sum01_2N);
                    t10 <= trunc2N_to_N(sum23_2N);
                end
                // 2) T column-1: P * [a10;a11]
                4'd2: begin
                    m0_a<=p00; m0_b<=a10;
                    m1_a<=p01; m1_b<=a11;
                    m2_a<=p10; m2_b<=a10;
                    m3_a<=p11; m3_b<=a11;
                end
                // 3) latch t01,t11 (N)
                4'd3: begin
                    t01 <= trunc2N_to_N(sum01_2N);
                    t11 <= trunc2N_to_N(sum23_2N);
                end
                // 4) S column-0: A * [t00;t10]
                4'd4: begin
                    m0_a<=a00; m0_b<=t00;
                    m1_a<=a01; m1_b<=t10;
                    m2_a<=a10; m2_b<=t00;
                    m3_a<=a11; m3_b<=t10;
                end
                // 5) write s00_2N, s10_2N (+Q00_2N, +Q10_2N)
                4'd5: begin
                    s00_2N <= sum01_2N + Q00_2N;
                    s10_2N <= sum23_2N + Q10_2N;
                end
                // 6) S column-1: A * [t01;t11]
                4'd6: begin
                    m0_a<=a00; m0_b<=t01;
                    m1_a<=a01; m1_b<=t11;
                    m2_a<=a10; m2_b<=t01;
                    m3_a<=a11; m3_b<=t11;
                end
                // 7) write s01_2N, s11_2N (+Q01_2N, +Q11_2N) and finish
                4'd7: begin
                    s01_2N <= sum01_2N + Q01_2N;
                    s11_2N <= sum23_2N + Q11_2N;
                    running<=1'b0; done<=1'b1;
                end
                default: ;
            endcase
        end
    end
endmodule
