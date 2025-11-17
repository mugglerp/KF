// ============================================================================
// prior_state_serial (Fig.c strict) with 1-cycle settle after s/t ready
// Ax -> t00,t10 ; Bu -> s00,s10 ; x = t + s
// Resources: 2x MUL + 1x ADD
// Latency  : 8 cycles (0..7)  [state 5 is the settle/wait cycle]
// 要修改:加法器并没有被使用 -> 已改为全流程复用 fxp_add
// ============================================================================
module prior_state_serial
    #( parameter integer N=20,
       parameter integer FRAC=10 )
     (
         input  wire                   clk,
         input  wire                   rst_n,
         input  wire                   start,

         // Inputs strictly as in figure (lowercase)
         input  wire signed [N-1:0]    x00, x10,
         input  wire signed [N-1:0]    a00, a01, a10, a11,
         input  wire signed [N-1:0]    u00, u10,
         input  wire signed [N-1:0]    b00, b01, b10, b11,

         // Outputs: data in UPPERCASE (control 'done' kept lowercase)
         output reg                    done,
         output reg  signed [N-1:0]    X_PRIOR00,
         output reg  signed [N-1:0]    X_PRIOR10
     );
    // 2 multipliers
    reg  signed [N-1:0] m0a,m0b,m1a,m1b;
    wire signed [2*N-1:0] p0,p1;
    wire signed [N-1:0]   p0_t_unused,p1_t_unused;
    fxp_mul #(N,FRAC) U_M0(.a(m0a),.b(m0b),.y_full(p0),.y_trunc(p0_t_unused));
    fxp_mul #(N,FRAC) U_M1(.a(m1a),.b(m1b),.y_full(p1),.y_trunc(p1_t_unused));

    // Figure(c) intermediates
    reg signed [2*N-1:0] t00, t10; // Ax row results (2N, 小数点在 2*FRAC)
    reg signed [2*N-1:0] s00, s10; // Bu row results (2N, 小数点在 2*FRAC)

    // 统一的 2N 加法器：前半段加 p0+p1，后半段加 t+s
    reg [3:0] st; // 0..7

    // 根据状态选择加法器输入（只保留 1 颗 fxp_add）
    wire signed [2*N-1:0] add_in_a =
         (st==6) ? t00 :
         (st==7) ? t10 : p0;

    wire signed [2*N-1:0] add_in_b =
         (st==6) ? s00 :
         (st==7) ? s10 : p1;

    wire signed [2*N  :0] add_full_2N;
    wire signed [2*N-1:0] add_2N; // fxp_add 的“同宽截断”（去最高位）
    fxp_add #(2*N) U_A0(.a(add_in_a), .b(add_in_b), .y_full(add_full_2N), .y_trunc(add_2N));

    // 将 2N 对齐回 N（按 FRAC 右移）：这一步不是 fxp_add 的同宽截断能替代的
    wire signed [N-1:0] add_ts_to_N = add_2N[FRAC+N-1 : FRAC];

    always @(posedge clk or negedge rst_n)
    begin
        if(!rst_n)
        begin
            st<=0;
            done<=1'b0;
            X_PRIOR00<=0;
            X_PRIOR10<=0;
            t00<=0;
            t10<=0;
            s00<=0;
            s10<=0;
            m0a<=0;
            m0b<=0;
            m1a<=0;
            m1b<=0;
        end
        else
        begin
            done<=1'b0;
            case(st)
                // 0) t00 = a00*x00 + a01*x10
                0:
                begin
                    if(start)
                    begin
                        m0a<=a00;
                        m0b<=x00;
                        m1a<=a01;
                        m1b<=x10;
                        st<=1;
                    end
                end
                // 1) latch t00, prepare s00 = b00*u00 + b01*u10
                1:
                begin
                    t00 <= add_2N;                  // 用同一 fxp_add 的结果
                    m0a<=b00;
                    m0b<=u00;
                    m1a<=b01;
                    m1b<=u10;
                    st<=2;
                end
                // 2) latch s00, prepare t10 = a10*x00 + a11*x10
                2:
                begin
                    s00 <= add_2N;
                    m0a<=a10;
                    m0b<=x00;
                    m1a<=a11;
                    m1b<=x10;
                    st<=3;
                end
                // 3) latch t10, prepare s10 = b10*u00 + b11*u10
                3:
                begin
                    t10 <= add_2N;
                    m0a<=b10;
                    m0b<=u00;
                    m1a<=b11;
                    m1b<=u10;
                    st<=4;
                end
                // 4) latch s10, then go to settle cycle
                4:
                begin
                    s10 <= add_2N;
                    m0a<=0;
                    m0b<=0;
                    m1a<=0;
                    m1b<=0; // freeze
                    st<=5;
                end
                // 5) settle/wait one cycle for stability
                5:
                begin
                    st<=6;
                end
                // 6) X_PRIOR00 = t00 + s00 （复用同一 fxp_add，再对齐到 N）
                6:
                begin
                    X_PRIOR00 <= add_ts_to_N;
                    st<=7;
                end
                // 7) X_PRIOR10 = t10 + s10 ; done
                7:
                begin
                    X_PRIOR10 <= add_ts_to_N;
                    done <= 1'b1;
                    st<=0;
                end
                default:
                    st<=0;
            endcase
        end
    end
endmodule
