`timescale 1ns/1ps
`include "fxp_types.vh"

// ============================================================================
// q_serial.v : Q = ((Δx1)^2 + (Δx2)^2)/2 * I
// - 3 cycles timing: S0->Δx1^2, S1->Δx2^2, S2->sum/2->done
// - dx, mul 均用组合电路
// - 保证输出为 Q(N.FRAC) 定点格式
// ============================================================================
module q_serial
    #( parameter integer N    = `FXP_N,
       parameter integer FRAC = `FXP_FRAC )
     (
         input  wire                     clk,
         input  wire                     rst_n,
         input  wire                     start,

         input  wire signed [N-1:0]      x00_now, x01_now,
         input  wire signed [N-1:0]      x00_prev, x01_prev,

         output reg                      done,
         output reg  signed [N-1:0]      Q11, Q12, Q21, Q22
     );

    // FSM
    localparam [1:0] S0=2'd0, S1=2'd1, S2=2'd2;
    reg [1:0] st;

    // 选择器（复用一次减法器/乘法器）
    wire signed [N-1:0] sel_now  = (st==S0) ? x00_now  : x01_now;
    wire signed [N-1:0] sel_prev = (st==S0) ? x00_prev : x01_prev;

    // Δx
    wire signed [N  :0] dx_full;
    wire signed [N-1:0] dx;
    fxp_sub #(N) U_SUB (.a(sel_now), .b(sel_prev), .y_full(dx_full), .y_trunc(dx));

    // Δx^2 : 2N 精度
    wire signed [2*N-1:0] prod_2N;
    wire signed [N-1:0]   prod_tr_unused;
    fxp_mul #(N,FRAC) U_MUL (.a(dx), .b(dx), .y_full(prod_2N), .y_trunc(prod_tr_unused));

    // 暂存 Δx^2（对齐到 N.FRAC）
    reg signed [N-1:0] t00_q, t11_q;

    // sum/2（组合 next）
    wire signed [N:0]     sum_q_next =
         $signed({t00_q[N-1],t00_q}) + $signed({t11_q[N-1],t11_q}); // N+1 bits

    wire signed [N-1:0]   q_next = sum_q_next >>> 1; // *0.5, 算术移位

    // 主时序
    always @(posedge clk or negedge rst_n)
    begin
        if(!rst_n)
        begin
            st   <= S0;
            done <= 1'b0;
            t00_q <= {N{1'b0}};
            t11_q <= {N{1'b0}};
            Q11 <= 0;
            Q22 <= 0;
            Q12 <= 0;
            Q21 <= 0;
        end
        else
        begin
            done <= 1'b0;

            case(st)
                // c0
                S0:
                begin
                    if(start)
                    begin
                        t00_q <= prod_2N >>> FRAC;
                        st    <= S1;
                    end
                end

                // c1
                S1:
                begin
                    t11_q <= prod_2N >>> FRAC;
                    st    <= S2;
                end

                // c2
                S2:
                begin
                    Q11  <= q_next;
                    Q22  <= q_next;
                    Q12  <= 0;
                    Q21  <= 0;
                    done <= 1'b1;
                    st   <= S0;
                end
            endcase
        end
    end
endmodule
