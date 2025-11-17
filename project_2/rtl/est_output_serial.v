// ============================================================================
// est_output_serial : z = H * x  (2x2, serial 4-cycle)
// Ports renamed: inputs lowercase, outputs UPPERCASE
// - 2 multipliers (FULL 2N precision) + 1 adder (2N domain)
// - Outputs Z00 (row0), Z10 (row1) truncated to N at FRAC
// - Latency: 4 cycles from start (c0..c3)
// ============================================================================

module est_output_serial
    #( parameter integer N=20, parameter integer FRAC=10 )
     (
         input  wire                     clk,
         input  wire                     rst_n,
         input  wire                     start,

         // H and x (lowercase)
         input  wire signed [N-1:0]      h00, h01, h10, h11,
         input  wire signed [N-1:0]      x00, x10,

         output reg                      done,
         // Z (UPPERCASE)
         output reg  signed [N-1:0]      Z00,   // = h00*x00 + h01*x10
         output reg  signed [N-1:0]      Z10    // = h10*x00 + h11*x10
     );

    // ---------- full-precision multipliers ----------
    reg  signed [N-1:0] m0_a, m0_b, m1_a, m1_b;
    wire signed [2*N-1:0] m0_f, m1_f;       // 2N full products
    wire signed [N-1:0]   m0_t_unused, m1_t_unused;

    fxp_mul #(N,FRAC) U_M0 (.a(m0_a), .b(m0_b), .y_full(m0_f), .y_trunc(m0_t_unused));
    fxp_mul #(N,FRAC) U_M1 (.a(m1_a), .b(m1_b), .y_full(m1_f), .y_trunc(m1_t_unused));

    // ---------- 2N adder ----------
    wire signed [2*N:0]   add_full;
    wire signed [2*N-1:0] add_2N;
    fxp_add #(2*N) U_ADD (.a(m0_f), .b(m1_f), .y_full(add_full), .y_trunc(add_2N));

    // helper: 2N -> N truncation aligned to FRAC
    function [N-1:0] trunc_2N_to_N;
        input signed [2*N-1:0] x2n;
        begin
            trunc_2N_to_N = x2n[FRAC+N-1 : FRAC];
        end
    endfunction

    // ---------- 4-cycle controller ----------
    // c0: load for Z00 = h00*x00 + h01*x10
    // c1: capture Z00
    // c2: load for Z10 = h10*x00 + h11*x10
    // c3: capture Z10 & done
    reg [1:0] st;

    always @(posedge clk or negedge rst_n)
    begin
        if (!rst_n)
        begin
            st   <= 2'd0;
            done <= 1'b0;
            Z00  <= {N{1'b0}};
            Z10  <= {N{1'b0}};
            m0_a <= {N{1'b0}};
            m0_b <= {N{1'b0}};
            m1_a <= {N{1'b0}};
            m1_b <= {N{1'b0}};
        end
        else
        begin
            done <= 1'b0;
            case (st)
                2'd0:
                begin
                    if (start)
                    begin
                        // c0
                        m0_a <= h00;
                        m0_b <= x00;
                        m1_a <= h01;
                        m1_b <= x10;
                        st   <= 2'd1;
                    end
                end
                2'd1:
                begin
                    // c1: capture Z00
                    Z00 <= trunc_2N_to_N(add_2N);
                    st  <= 2'd2;
                end
                2'd2:
                begin
                    // c2
                    m0_a <= h10;
                    m0_b <= x00;
                    m1_a <= h11;
                    m1_b <= x10;
                    st   <= 2'd3;
                end
                2'd3:
                begin
                    // c3: capture Z10 & finish
                    Z10  <= trunc_2N_to_N(add_2N);
                    done <= 1'b1;
                    st   <= 2'd0;
                end
                default:
                    st <= 2'd0;
            endcase
        end
    end

endmodule
