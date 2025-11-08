// ============================================================================
// r_serial.v : Serial R block using 2 Eq.1 + MUX (3-cycle version)
// Timeline: c0 load pair0, c1 capture R11 & load pair1, c2 capture R22 & done
// - Eq.1 keeps 2N internally; subtraction in N domain; abs(N) for modulus.
// ============================================================================
`include "fxp_types.vh"

module r_serial
#( parameter integer N    = `FXP_N
 ,  parameter integer FRAC = `FXP_FRAC )
(
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire                   start,

    // Inputs per figure (strict mapping)
    input  wire signed [N-1:0]    beta,
    input  wire signed [N-1:0]    sigma2_00, sigma2_01,
    input  wire signed [N-1:0]    sigma2_10, sigma2_11,
    input  wire signed [N-1:0]    z00, z10,
    input  wire signed [N-1:0]    zhat00, zhat10,

    output reg                    done,
    output reg  signed [N-1:0]    R11, R12, R21, R22
);

    // ---------------- MUX to feed two Eq.1 blocks ----------------
    reg sel_pair;  // 0: pair0 (00/01 with z00/zhat00), 1: pair1 (10/11 with z10/zhat10)

    wire signed [N-1:0] I1L = beta;
    wire signed [N-1:0] I2L = sel_pair ? sigma2_10 : sigma2_00;
    wire signed [N-1:0] I3L = sel_pair ? z10       : z00;

    wire signed [N-1:0] I1R = beta;
    wire signed [N-1:0] I2R = sel_pair ? sigma2_11 : sigma2_01;
    wire signed [N-1:0] I3R = sel_pair ? zhat10    : zhat00;

    wire signed [N-1:0]   L_raw, R_raw;
    wire signed [2*N-1:0] L_full, R_full; // not used, kept for completeness
    subsystem_eq1 #(N,FRAC) EQL (.I1(I1L), .I2(I2L), .I3(I3L), .O_raw(L_raw), .O_full(L_full));
    subsystem_eq1 #(N,FRAC) EQR (.I1(I1R), .I2(I2R), .I3(I3R), .O_raw(R_raw), .O_full(R_full));

    // ---------------- N-domain subtraction + ABS ----------------
    wire signed [N:0]   diff_full;
    wire signed [N-1:0] diff_n;
    fxp_sub #(N) U_SUB (.a(L_raw), .b(R_raw), .y_full(diff_full), .y_trunc(diff_n));

    wire signed [N-1:0] diff_abs;
    fxp_abs  #(N) U_ABS (.a(diff_n), .y(diff_abs));

    // ---------------- 3-cycle controller ----------------
    // st=0 IDLE, 1 C0(load pair0), 2 C1(capture R11 + load pair1), 3 C2(capture R22 + done)
    reg [1:0] st;

    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            st       <= 2'd0;
            sel_pair <= 1'b0;
            done     <= 1'b0;
            R11<=0; R22<=0; R12<=0; R21<=0;
        end else begin
            done <= 1'b0;
            case(st)
                2'd0: begin
                    if(start) begin
                        sel_pair <= 1'b0;   // c0: select pair0
                        st       <= 2'd1;
                    end
                end

                2'd1: begin
                    // c1: capture result of pair0, and switch to pair1
                    R11     <= diff_abs;    // | (β,σ̂²00,z00) - (β,σ̂²01,ẑ00) |
                    sel_pair<= 1'b1;        // immediately start pair1
                    st      <= 2'd2;
                end

                2'd2: begin
                    // c2: capture result of pair1 and finish
                    R22  <= diff_abs;       // | (β,σ̂²10,z10) - (β,σ̂²11,ẑ10) |
                    R12  <= {N{1'b0}};
                    R21  <= {N{1'b0}};
                    done <= 1'b1;
                    st   <= 2'd0;
                end

                default: st <= 2'd0;
            endcase
        end
    end

endmodule
