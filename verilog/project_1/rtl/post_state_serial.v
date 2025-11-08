`include "fxp_types.vh"

// ============================================================================
// post_state_serial : x_post = x_prior + K * (z_meas - z_hat)
// Ports rule: inputs lowercase, matrix/vector outputs UPPERCASE
// Timing: 5 cycles (c0..c4) — unchanged
// - MUL keeps 2N full precision; sums in 2N; truncate to N at FRAC when output
// ============================================================================
module post_state_serial
#( parameter integer N=`FXP_N, parameter integer FRAC=`FXP_FRAC )
(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     start,

    // x_prior (lowercase)
    input  wire signed [N-1:0]      x00_prior, x10_prior,
    // measurements (lowercase)
    input  wire signed [N-1:0]      z00_meas,  z10_meas,
    input  wire signed [N-1:0]      z00_hat,   z10_hat,
    // K gain (lowercase) : k00=K11, k01=K12, k10=K21, k11=K22
    input  wire signed [N-1:0]      k00, k01, k10, k11,

    output reg                      done,
    // x_post (UPPERCASE)
    output reg  signed [N-1:0]      X00_post, X10_post
);

    // -------- 2N -> N truncate aligned to FRAC --------
    function [N-1:0] trunc2N_to_N;
        input signed [2*N-1:0] x2n;
        begin
            trunc2N_to_N = x2n[FRAC+N-1 : FRAC];
        end
    endfunction

    // e = z - z_hat  (N domain)
    wire signed [N  :0] e0_full, e1_full;
    wire signed [N-1:0] e0, e1;
    fxp_sub #(N) U_E0 (.a(z00_meas), .b(z00_hat), .y_full(e0_full), .y_trunc(e0));
    fxp_sub #(N) U_E1 (.a(z10_meas), .b(z10_hat), .y_full(e1_full), .y_trunc(e1));

    // 2N multipliers (full precision)
    reg  signed [N-1:0] m0_a, m0_b, m1_a, m1_b;
    wire signed [2*N-1:0] m0_full, m1_full;
    wire signed [N-1:0]   m0_tr_nc, m1_tr_nc; // unused
    fxp_mul #(N,FRAC) U_M0 (.a(m0_a), .b(m0_b), .y_full(m0_full), .y_trunc(m0_tr_nc));
    fxp_mul #(N,FRAC) U_M1 (.a(m1_a), .b(m1_b), .y_full(m1_full), .y_trunc(m1_tr_nc));

    // 2N adder for column sum
    wire signed [2*N  :0] s_full;
    wire signed [2*N-1:0] s_2N;
    fxp_add #(2*N) U_SUM (.a(m0_full), .b(m1_full), .y_full(s_full), .y_trunc(s_2N));

    // hold y = K*e (2N domain)
    reg signed [2*N-1:0] y0_2N, y1_2N;

    // x_prior extend to 2N and align by FRAC
    wire signed [2*N-1:0] x0_2N = ({{N{x00_prior[N-1]}}, x00_prior}) <<< FRAC;
    wire signed [2*N-1:0] x1_2N = ({{N{x10_prior[N-1]}}, x10_prior}) <<< FRAC;

    // -------- 5-cycle FSM (unchanged) --------
    reg [2:0] st;
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            st <= 3'd0;
            done <= 1'b0;
            X00_post <= 0;
            X10_post <= 0;
            m0_a<=0; m0_b<=0; m1_a<=0; m1_b<=0;
            y0_2N <= 0; y1_2N <= 0;
        end else begin
            done <= 1'b0;
            case(st)
                // c0: y0 = k00*e0 + k01*e1
                3'd0: begin
                    if(start) begin
                        m0_a <= k00; m0_b <= e0;
                        m1_a <= k01; m1_b <= e1;
                        st   <= 3'd1;
                    end
                end
                // c1: latch y0
                3'd1: begin
                    y0_2N <= s_2N;
                    st    <= 3'd2;
                end
                // c2: y1 = k10*e0 + k11*e1
                3'd2: begin
                    m0_a <= k10; m0_b <= e0;
                    m1_a <= k11; m1_b <= e1;
                    st   <= 3'd3;
                end
                // c3: latch y1, compute X00_post = x00_prior + y0
                3'd3: begin
                    y1_2N   <= s_2N;
                    X00_post<= trunc2N_to_N(x0_2N + y0_2N);
                    st      <= 3'd4;
                end
                // c4: compute X10_post = x10_prior + y1 ; done
                3'd4: begin
                    X10_post<= trunc2N_to_N(x1_2N + y1_2N);
                    done    <= 1'b1;
                    st      <= 3'd0;
                end
                default: st <= 3'd0;
            endcase
        end
    end
endmodule
