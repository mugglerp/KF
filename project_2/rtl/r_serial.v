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

    // 轻量 Transformer：单头注意力（Q,K 为标量），特征采用测量残差 e0,e1
    localparam signed [N-1:0] ONE = $signed(1) <<< FRAC;
    localparam signed [N-1:0] WQ0 = ONE; // 训练后可更新
    localparam signed [N-1:0] WQ1 = ONE;
    localparam signed [N-1:0] WK0 = ONE;
    localparam signed [N-1:0] WK1 = ONE;

    // 计算 e0 = z00 - zhat00, e1 = z10 - zhat10
    wire signed [N  :0] e0_full, e1_full;
    wire signed [N-1:0] e0, e1;
    fxp_sub #(N) U_E0 (.a(z00), .b(zhat00), .y_full(e0_full), .y_trunc(e0));
    fxp_sub #(N) U_E1 (.a(z10), .b(zhat10), .y_full(e1_full), .y_trunc(e1));

    // Q,K 部分乘法（按特征分两拍生成并累加）
    reg  signed [N-1:0] mq_a, mq_b, mk_a, mk_b;
    wire signed [2*N-1:0] mq_full, mk_full;
    wire signed [N-1:0]   mq_tr,   mk_tr;
    fxp_mul #(N,FRAC) MQ (.a(mq_a), .b(mq_b), .y_full(mq_full), .y_trunc(mq_tr));
    fxp_mul #(N,FRAC) MK (.a(mk_a), .b(mk_b), .y_full(mk_full), .y_trunc(mk_tr));

    reg  signed [N-1:0] q_acc, k_acc;           // 累加器
    wire signed [N  :0] qsum_full, ksum_full;   // 加法器
    wire signed [N-1:0] qsum_tr,   ksum_tr;
    fxp_add #(N) U_QADD (.a(q_acc), .b(mq_tr), .y_full(qsum_full), .y_trunc(qsum_tr));
    fxp_add #(N) U_KADD (.a(k_acc), .b(mk_tr), .y_full(ksum_full), .y_trunc(ksum_tr));

    // 注意力权重（标量），以及对两路残差的加权输出
    wire signed [2*N-1:0] att_full, outL_full, outR_full;
    wire signed [N-1:0]   att_tr,   outL_tr,   outR_tr;
    fxp_mul #(N,FRAC) U_ATT  (.a(q_acc), .b(k_acc), .y_full(att_full), .y_trunc(att_tr));
    fxp_mul #(N,FRAC) U_OUTL (.a(att_tr), .b(e0),    .y_full(outL_full), .y_trunc(outL_tr));
    fxp_mul #(N,FRAC) U_OUTR (.a(att_tr), .b(e1),    .y_full(outR_full), .y_trunc(outR_tr));

    // 取绝对值（按现有 R 逻辑输出为非负）
    wire signed [N-1:0] outL_abs, outR_abs;
    fxp_abs  #(N) U_ABSL (.a(outL_tr), .y(outL_abs));
    fxp_abs  #(N) U_ABSR (.a(outR_tr), .y(outR_abs));

    // 3-cycle FSM（保持时序与握手不变）
    reg [1:0] st;
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            st<=2'd0; done<=1'b0;
            R11<=0; R22<=0; R12<=0; R21<=0;
            mq_a<=0; mq_b<=0; mk_a<=0; mk_b<=0; q_acc<=0; k_acc<=0;
        end else begin
            done<=1'b0;
            case(st)
                // c0: 采样 e0，生成 Q/K 第一部分
                2'd0: if(start) begin
                          mq_a<=WQ0; mq_b<=e0;
                          mk_a<=WK0; mk_b<=e0;
                          q_acc<=mq_tr; k_acc<=mk_tr;
                          st<=2'd1;
                      end
                // c1: 采样 e1，累加并写 R11
                2'd1: begin
                          mq_a<=WQ1; mq_b<=e1;
                          mk_a<=WK1; mk_b<=e1;
                          q_acc<=qsum_tr; k_acc<=ksum_tr;
                          R11<=outL_abs;  // 基于注意力的左通道
                          st<=2'd2;
                      end
                // c2: 写 R22 & done
                2'd2: begin
                          R22<=outR_abs;  // 基于注意力的右通道
                          R12<=0; R21<=0;
                          done<=1'b1; st<=2'd0;
                      end
            endcase
        end
    end
endmodule
