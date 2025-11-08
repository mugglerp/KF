// ============================================================================
// inv2_serial.v : True 4-cycle 2x2 matrix inverse (registered, Verilog-2001)
// inv([a b; c d]) = (1/det) * [ d -b; -c  a ]
// Cycle (start sampled at c0):
//   c0: det = ad - bc (ad,bc in parallel) -> recip_det_w (comb) -> recip_det_r (reg)
//   c1: set MUL inputs for IA/IB: m0_a = d,  m1_a = -b
//   c2: latch IA=m0_tr, IB=m1_tr; set MUL inputs for IC/ID: m0_a = -c, m1_a = a
//   c3: latch IC=m0_tr, ID=m1_tr; done=1 (one-cycle pulse)
// ============================================================================

`include "fxp_types.vh"

module inv2_serial
#( parameter integer N   = `FXP_N,
   parameter integer FRAC= `FXP_FRAC )
(
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire                   start,

    input  wire signed [N-1:0]    a, b, c, d,

    output reg                    done,
    output reg  signed [N-1:0]    IA, IB, IC, ID
);

    // ---------------- State (0..3) ----------------
    reg [1:0] st;   // 0: det/recip, 1: set IA/IB mul, 2: latch IA/IB & set IC/ID mul, 3: latch IC/ID & done

    // ---------------- Parallel mul for ad / bc ----------------
    wire signed [2*N-1:0] ad_full, bc_full;
    wire signed [N-1:0]   ad_tr,   bc_tr;

    fxp_mul #(N,FRAC) U_MUL_AD (.a(a), .b(d), .y_full(ad_full), .y_trunc(ad_tr));
    fxp_mul #(N,FRAC) U_MUL_BC (.a(b), .b(c), .y_full(bc_full), .y_trunc(bc_tr));

    // det = ad - bc
    wire signed [N:0]     det_full_w;
    wire signed [N-1:0]   det_w;
    fxp_sub #(N) U_SUB_DET (.a(ad_tr), .b(bc_tr), .y_full(det_full_w), .y_trunc(det_w));

    // recip(det)
    wire signed [N-1:0]   recip_det_w;
    reg  signed [N-1:0]   recip_det_r;
    goldschmidt_struct #(N,FRAC) U_RECIP (.x(det_w), .recip(recip_det_w));

    // ---------------- Two parallel output muls ----------------
    reg  signed [N-1:0]   m0_a, m1_a;      // selected operands (另一端始终接 recip_det_r)
    wire signed [2*N-1:0] m0_full, m1_full;
    wire signed [N-1:0]   m0_tr,   m1_tr;

    fxp_mul #(N,FRAC) U_MUL0 (.a(m0_a), .b(recip_det_r), .y_full(m0_full), .y_trunc(m0_tr));
    fxp_mul #(N,FRAC) U_MUL1 (.a(m1_a), .b(recip_det_r), .y_full(m1_full), .y_trunc(m1_tr));

    // ---------------- Control ----------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st <= 2'd0;
            done <= 1'b0;
            IA <= 0; IB <= 0; IC <= 0; ID <= 0;
            recip_det_r <= 0;
            m0_a <= 0; m1_a <= 0;
        end else begin
            done <= 1'b0;

            case (st)
                // c0: compute recip(det) and register for following stages
                2'd0: begin
                    if (start) begin
                        recip_det_r <= recip_det_w; // det_w -> recip_det_w (comb) -> reg
                        st <= 2'd1;
                    end
                end

                // c1: set mul inputs for IA/IB
                2'd1: begin
                    m0_a <= d;    // IA path
                    m1_a <= -b;   // IB path
                    st   <= 2'd2;
                end

                // c2: latch IA/IB; set next mul inputs for IC/ID
                2'd2: begin
                    IA   <= m0_tr;   // result of c1
                    IB   <= m1_tr;
                    m0_a <= -c;      // IC path
                    m1_a <=  a;      // ID path
                    st   <= 2'd3;
                end

                // c3: latch IC/ID; done
                2'd3: begin
                    IC   <= m0_tr;   // result of c2
                    ID   <= m1_tr;
                    done <= 1'b1;
                    st   <= 2'd0;
                end

                default: st <= 2'd0;
            endcase
        end
    end

endmodule
