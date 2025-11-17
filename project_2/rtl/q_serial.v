`timescale 1ns/1ps

module q_serial
    #( parameter integer N=20, parameter integer FRAC=10 )
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

    // 轻量 Transformer：单头注意力（Q,K,V 为标量），特征采用两路差分 dx0,dx1
    localparam signed [N-1:0] ONE = $signed(1) <<< FRAC;
    localparam signed [N-1:0] WQ0 = ONE; // 训练后可更新
    localparam signed [N-1:0] WQ1 = ONE;
    localparam signed [N-1:0] WK0 = ONE;
    localparam signed [N-1:0] WK1 = ONE;
    localparam signed [N-1:0] WV0 = ONE;
    localparam signed [N-1:0] WV1 = ONE;

    // 选择当前/上一帧的分量
    wire signed [N-1:0] sel_now  = (st==S0) ? x00_now  : x01_now;
    wire signed [N-1:0] sel_prev = (st==S0) ? x00_prev : x01_prev;

    // 差分（串行得到 dx0 与 dx1）
    wire signed [N  :0] dx_full;
    wire signed [N-1:0] dx;
    fxp_sub #(N) U_SUB (
                .a(sel_now),
                .b(sel_prev),
                .y_full(dx_full),
                .y_trunc(dx)
            );

    // 三路乘法用于生成部分 Q,K,V 分量
    reg  signed [N-1:0] mq_a, mq_b, mk_a, mk_b, mv_a, mv_b;
    wire signed [2*N-1:0] mq_full, mk_full, mv_full;
    wire signed [N-1:0]   mq_tr,   mk_tr,   mv_tr;
    fxp_mul #(N,FRAC) MQ (.a(mq_a), .b(mq_b), .y_full(mq_full), .y_trunc(mq_tr));
    fxp_mul #(N,FRAC) MK (.a(mk_a), .b(mk_b), .y_full(mk_full), .y_trunc(mk_tr));
    fxp_mul #(N,FRAC) MV (.a(mv_a), .b(mv_b), .y_full(mv_full), .y_trunc(mv_tr));

    // 累加器（N 位域，保持同一 Q 格式）
    reg  signed [N-1:0] q_acc, k_acc, v_acc;
    wire signed [N  :0] qsum_full, ksum_full, vsum_full;
    wire signed [N-1:0] qsum_tr,   ksum_tr,   vsum_tr;
    fxp_add #(N) U_QADD (.a(q_acc), .b(mq_tr), .y_full(qsum_full), .y_trunc(qsum_tr));
    fxp_add #(N) U_KADD (.a(k_acc), .b(mk_tr), .y_full(ksum_full), .y_trunc(ksum_tr));
    fxp_add #(N) U_VADD (.a(v_acc), .b(mv_tr), .y_full(vsum_full), .y_trunc(vsum_tr));

    // 注意力与输出（标量）：att = q*k；out = att*v
    wire signed [2*N-1:0] att_full, out_full;
    wire signed [N-1:0]   att_tr,   out_tr;
    fxp_mul #(N,FRAC) U_ATT (.a(q_acc), .b(k_acc), .y_full(att_full), .y_trunc(att_tr));
    fxp_mul #(N,FRAC) U_OUT (.a(att_tr), .b(v_acc), .y_full(out_full), .y_trunc(out_tr));

    always @(posedge clk or negedge rst_n)
    begin
        if(!rst_n)
        begin
            st<=S0;
            done<=1'b0;
            q_acc<=0; k_acc<=0; v_acc<=0;
            mq_a<=0; mq_b<=0; mk_a<=0; mk_b<=0; mv_a<=0; mv_b<=0;
            Q11<=0; Q22<=0; Q12<=0; Q21<=0;
        end
        else
        begin
            done<=1'b0;
            case(st)
                // S0：处理 dx0 = x00_now - x00_prev
                S0:
                    if(start)
                    begin
                        // Q,K,V 第一部分
                        mq_a<=WQ0; mq_b<=dx;
                        mk_a<=WK0; mk_b<=dx;
                        mv_a<=WV0; mv_b<=dx;
                        // 采样到累加器
                        q_acc<=mq_tr; k_acc<=mk_tr; v_acc<=mv_tr;
                        st <= S1;
                    end
                // S1：处理 dx1 = x01_now - x01_prev，并累加
                S1:
                begin
                    mq_a<=WQ1; mq_b<=dx;
                    mk_a<=WK1; mk_b<=dx;
                    mv_a<=WV1; mv_b<=dx;
                    q_acc<=qsum_tr; // q_acc + mq_tr
                    k_acc<=ksum_tr; // k_acc + mk_tr
                    v_acc<=vsum_tr; // v_acc + mv_tr
                    st <= S2;
                end
                // S2：一次组合计算注意力与输出，写 Q
                S2:
                begin
                    Q11<=out_tr;
                    Q22<=out_tr;
                    Q12<=0;
                    Q21<=0;
                    done<=1'b1;
                    st<=S0;
                end
            endcase
        end
    end
endmodule
