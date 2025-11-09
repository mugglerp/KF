`timescale 1ns/1ps
`include "fxp_types.vh"

module q_serial
    #( parameter integer N=`FXP_N, parameter integer FRAC=`FXP_FRAC )
     (
         input  wire                     clk,
         input  wire                     rst_n,
         input  wire                     start,

         input  wire signed [N-1:0]      x00_now, x01_now,
         input  wire signed [N-1:0]      x00_prev, x01_prev,

         output reg                      done,
         output reg  signed [N-1:0]      Q11, Q12, Q21, Q22
     );

    // 状态：两次差分平方后做一次加法得到 q_next
    localparam [1:0] S0=2'd0, S1=2'd1, S2=2'd2;
    reg [1:0] st;

    // 选择当前/上一帧的分量
    wire signed [N-1:0] sel_now  = (st==S0) ? x00_now  : x01_now;
    wire signed [N-1:0] sel_prev = (st==S0) ? x00_prev : x01_prev;

    // 差分
    wire signed [N  :0] dx_full;
    wire signed [N-1:0] dx;
    fxp_sub #(N) U_SUB (
                .a(sel_now),
                .b(sel_prev),
                .y_full(dx_full),
                .y_trunc(dx)
            );

    // 差分平方（全精度，再按 FRAC 对齐到 Q 格式）
    wire signed [2*N-1:0] prod_2N;
    wire signed [N-1:0]   prod_tr_unused;
    fxp_mul #(N,FRAC) U_MUL (
                .a(dx),
                .b(dx),
                .y_full(prod_2N),
                .y_trunc(prod_tr_unused)
            );

    // 暂存两个分量的平方结果（已对齐到 N 位 Q 格式）
    reg signed [N-1:0] t00_q, t11_q;

    // 使用 fxp_add 做加法（y_trunc 已做 N 位截断）
    wire signed [N  :0] sum_full;
    wire signed [N-1:0] sum_trunc;
    fxp_add #(N) U_ADD (
                .a(t00_q),
                .b(t11_q),
                .y_full(sum_full),
                .y_trunc(sum_trunc)
            );

    // q_next = (t00_q + t11_q) / 2 ；保持同一 Q 格式
    wire signed [N-1:0] q_next = sum_trunc >>> 1;

    always @(posedge clk or negedge rst_n)
    begin
        if(!rst_n)
        begin
            st<=S0;
            done<=1'b0;
            t00_q<=0;
            t11_q<=0;
            Q11<=0;
            Q22<=0;
            Q12<=0;
            Q21<=0;
        end
        else
        begin
            done<=1'b0;
            case(st)
                S0:
                    if(start)
                    begin
                        t00_q <= prod_2N >>> FRAC;  // 对齐到 Q 格式
                        st    <= S1;
                    end
                S1:
                begin
                    t11_q <= prod_2N >>> FRAC;  // 对齐到 Q 格式
                    st    <= S2;
                end
                S2:
                begin
                    // 对角等于 q_next，非对角为 0（按你现在的设定）
                    Q11<=q_next;
                    Q22<=q_next;
                    Q12<=0;
                    Q21<=0;
                    done<=1'b1;
                    st<=S0;
                end
            endcase
        end
    end
endmodule
