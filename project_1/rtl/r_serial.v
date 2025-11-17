`timescale 1ns/1ps

module r_serial
#( parameter integer N=20, parameter integer FRAC=10 )
(
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire                   start,

    input  wire signed [N-1:0]    beta,
    input  wire signed [N-1:0]    sigma2_00, sigma2_01,
    input  wire signed [N-1:0]    sigma2_10, sigma2_11,
    input  wire signed [N-1:0]    z00, z10,
    input  wire signed [N-1:0]    zhat00, zhat10,

    output reg                    done,
    output reg  signed [N-1:0]    R11, R12, R21, R22
);

    reg sel_pair; // 0: pair0(z00,zhat00, σ00/σ01)  1: pair1(z10,zhat10, σ10/σ11)

    // 左/右 Eq.1（输出 O_raw 必须是 N.FRAC 对齐）
    wire signed [N-1:0] L_raw, R_raw;
    wire signed [2*N-1:0] L_full, R_full; // 仅保留以便波形观测
    subsystem_eq1 #(N,FRAC) EQL (.I1(beta),
                                 .I2(sel_pair ? sigma2_10 : sigma2_00),
                                 .I3(sel_pair ? z10       : z00),
                                 .O_raw(L_raw), .O_full(L_full));
    subsystem_eq1 #(N,FRAC) EQR (.I1(beta),
                                 .I2(sel_pair ? sigma2_11 : sigma2_01),
                                 .I3(sel_pair ? zhat10    : zhat00),
                                 .O_raw(R_raw), .O_full(R_full));

    // N 域相减取绝对值
    wire signed [N  :0] diff_full;
    wire signed [N-1:0] diff_n;
    fxp_sub #(N) U_SUB (.a(L_raw), .b(R_raw), .y_full(diff_full), .y_trunc(diff_n));

    wire signed [N-1:0] diff_abs;
    fxp_abs  #(N) U_ABS (.a(diff_n), .y(diff_abs));

    // 3-cycle FSM
    // c0(load pair0)->c1(capture R11 & switch)->c2(capture R22 & done)
    reg [1:0] st;
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            st<=2'd0; sel_pair<=1'b0; done<=1'b0;
            R11<=0; R22<=0; R12<=0; R21<=0;
        end else begin
            done<=1'b0;
            case(st)
                2'd0: if(start) begin
                          sel_pair<=1'b0; st<=2'd1;
                      end
                2'd1: begin
                          R11<=diff_abs;  // pair0
                          sel_pair<=1'b1; st<=2'd2;
                      end
                2'd2: begin
                          R22<=diff_abs;  // pair1
                          R12<=0; R21<=0;
                          done<=1'b1; st<=2'd0;
                      end
            endcase
        end
    end
endmodule
