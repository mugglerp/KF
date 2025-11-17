// ============================================================================
// prior_cov_semipar.v : S = A*P*A^T + Q   (2x2, semi-parallel, 8 cycles)
// Ports strictly follow the paper figure:
//   A: a00 a01 a10 a11
//   P: p00 p01 p10 p11
//   Q: q00 q01 q10 q11
// Outputs (UPPERCASE, naming per your rule):
//   P_PRIOR00 P_PRIOR01 P_PRIOR10 P_PRIOR11
// ----------------------------------------------------------------------------
// MUL keeps 2N full precision; column-sums and +Q happen in 2N domain.
// Only when feeding next stage or at final outputs we truncate to N bits
// aligned to FRAC. Q is sign-extended to 2N and left-shifted by FRAC
// to align with mul full products.
//
// New 2-adder schedule (start at C0, done pulse at C7, total 8 cycles from start):
//   C0(start): cyc=1, config P*A^T col0
//   C1: cyc=2, T col0 -> (t00,t10), config P*A^T col1
//   C2: cyc=3, T col1 -> (t01,t11), config A*T col0
//   C3: cyc=4, S col0 sum -> (s00_tmp,s10_tmp), config A*T col1
//   C4: cyc=5, S col0 + Q00/Q10 -> (s00_2N,s10_2N)
//   C5: cyc=6, S col1 sum -> (s01_tmp,s11_tmp)
//   C6: cyc=7, S col1 + Q01/Q11 -> (s01_2N,s11_2N)
//   C7: done=1
// ============================================================================

module prior_cov_semipar
    #(
         parameter integer N    = 20,
         parameter integer FRAC = 10
     )
     (
         input  wire                     clk,
         input  wire                     rst_n,
         input  wire                     start,

         // A (2x2) - lowercase inputs
         input  wire signed [N-1:0]      a00, a01, a10, a11,
         // P (2x2)
         input  wire signed [N-1:0]      p00, p01, p10, p11,
         // Q (2x2)
         input  wire signed [N-1:0]      q00, q01, q10, q11,

         // control
         output reg                      done,

         // UPPERCASE data outputs
         output wire signed [N-1:0]      P_PRIOR00,
         output wire signed [N-1:0]      P_PRIOR01,
         output wire signed [N-1:0]      P_PRIOR10,
         output wire signed [N-1:0]      P_PRIOR11
     );

    // -------- 2N -> N truncation (FRAC aligned) --------
    function [N-1:0] trunc2N_to_N;
        input signed [2*N-1:0] x2n;
        begin
            trunc2N_to_N = x2n[FRAC+N-1 : FRAC];
        end
    endfunction

    // -------- Q aligned to 2N domain --------
    wire signed [2*N-1:0] Q00_2N = ({{N{q00[N-1]}}, q00}) <<< FRAC;
    wire signed [2*N-1:0] Q01_2N = ({{N{q01[N-1]}}, q01}) <<< FRAC;
    wire signed [2*N-1:0] Q10_2N = ({{N{q10[N-1]}}, q10}) <<< FRAC;
    wire signed [2*N-1:0] Q11_2N = ({{N{q11[N-1]}}, q11}) <<< FRAC;

    // -------- 4 MUL (keep 2N full products) --------
    reg  signed [N-1:0]      m0_a, m0_b, m1_a, m1_b, m2_a, m2_b, m3_a, m3_b;
    wire signed [2*N-1:0]    m0_f, m1_f, m2_f, m3_f;
    wire signed [N-1:0]      _t0, _t1, _t2, _t3; // unused truncated ports

    fxp_mul #(N,FRAC) U_M0 (.a(m0_a), .b(m0_b), .y_full(m0_f), .y_trunc(_t0));
    fxp_mul #(N,FRAC) U_M1 (.a(m1_a), .b(m1_b), .y_full(m1_f), .y_trunc(_t1));
    fxp_mul #(N,FRAC) U_M2 (.a(m2_a), .b(m2_b), .y_full(m2_f), .y_trunc(_t2));
    fxp_mul #(N,FRAC) U_M3 (.a(m3_a), .b(m3_b), .y_full(m3_f), .y_trunc(_t3));

    // -------- 2 可复用的 2N 加法器 --------
    reg  signed [2*N-1:0]    add0_a, add0_b, add1_a, add1_b;
    wire signed [2*N:0]      add0_f, add1_f;
    wire signed [2*N-1:0]    add0_2N, add1_2N;

    fxp_add #(2*N) U_ADD0 (
                .a      (add0_a),
                .b      (add0_b),
                .y_full (add0_f),
                .y_trunc(add0_2N)
            );

    fxp_add #(2*N) U_ADD1 (
                .a      (add1_a),
                .b      (add1_b),
                .y_full (add1_f),
                .y_trunc(add1_2N)
            );

    // -------- T matrix (N domain) --------
    reg signed [N-1:0] t00, t01, t10, t11;

    // -------- S registers in 2N domain --------
    reg signed [2*N-1:0] s00_2N, s01_2N, s10_2N, s11_2N;

    // 中间列和 (2N) 用来做 +Q
    reg signed [2*N-1:0] s00_tmp, s01_tmp, s10_tmp, s11_tmp;

    // 输出截断到 N 位
    assign P_PRIOR00 = trunc2N_to_N(s00_2N);
    assign P_PRIOR01 = trunc2N_to_N(s01_2N);
    assign P_PRIOR10 = trunc2N_to_N(s10_2N);
    assign P_PRIOR11 = trunc2N_to_N(s11_2N);

    // -------- 8-cycle controller --------
    reg [3:0] cyc;
    reg       running;

    // 加法器输入组合选择：根据 cyc 复用
    always @*
    begin
        // 默认 0，避免综合出锁存器
        add0_a = {2*N{1'b0}};
        add0_b = {2*N{1'b0}};
        add1_a = {2*N{1'b0}};
        add1_b = {2*N{1'b0}};

        if (running)
        begin
            case (cyc)
                // cyc=1,2,3,5 : 用于列求和 sum(m0+m1), sum(m2+m3)
                4'd1,
                4'd2,
                4'd3,
                4'd5:
                begin
                    add0_a = m0_f;
                    add0_b = m1_f;
                    add1_a = m2_f;
                    add1_b = m3_f;
                end

                // cyc=4 : S 第 0 列 + Q00/Q10
                4'd4:
                begin
                    add0_a = s00_tmp;
                    add0_b = Q00_2N;
                    add1_a = s10_tmp;
                    add1_b = Q10_2N;
                end

                // cyc=6 : S 第 1 列 + Q01/Q11
                4'd6:
                begin
                    add0_a = s01_tmp;
                    add0_b = Q01_2N;
                    add1_a = s11_tmp;
                    add1_b = Q11_2N;
                end

                default:
                begin
                    // 其它拍不使用输出
                end
            endcase
        end
    end

    // 时序控制 + 寄存器更新
    always @(posedge clk or negedge rst_n)
    begin
        if (!rst_n)
        begin
            cyc      <= 4'd0;
            running  <= 1'b0;
            done     <= 1'b0;

            m0_a <= 0;
            m0_b <= 0;
            m1_a <= 0;
            m1_b <= 0;
            m2_a <= 0;
            m2_b <= 0;
            m3_a <= 0;
            m3_b <= 0;

            t00 <= 0;
            t01 <= 0;
            t10 <= 0;
            t11 <= 0;

            s00_2N <= 0;
            s01_2N <= 0;
            s10_2N <= 0;
            s11_2N <= 0;

            s00_tmp <= 0;
            s01_tmp <= 0;
            s10_tmp <= 0;
            s11_tmp <= 0;
        end
        else
        begin
            done <= 1'b0;

            // 启动：接受 start 的当拍作为 C0，配置 P*A^T 第 0 列
            if (start && !running)
            begin
                running <= 1'b1;
                cyc     <= 4'd1;

                // C0(start) : P * [a00;a01] -> 用于 T 的第 0 列
                m0_a <= p00;
                m0_b <= a00;
                m1_a <= p01;
                m1_b <= a01;
                m2_a <= p10;
                m2_b <= a00;
                m3_a <= p11;
                m3_b <= a01;
            end
            else if (running)
            begin
                cyc <= cyc + 4'd1;
            end

            if (running)
            begin
                case (cyc)
                    // ----------------------------------------------------------------
                    // C1: T 第 0 列 (P*A^T col0) -> t00, t10
                    //     同拍配置 P*A^T col1
                    // ----------------------------------------------------------------
                    4'd1:
                    begin
                        t00 <= trunc2N_to_N(add0_2N); // sum(m0+m1)
                        t10 <= trunc2N_to_N(add1_2N); // sum(m2+m3)

                        // 配置 P*A^T 第 1 列
                        m0_a <= p00;
                        m0_b <= a10;
                        m1_a <= p01;
                        m1_b <= a11;
                        m2_a <= p10;
                        m2_b <= a10;
                        m3_a <= p11;
                        m3_b <= a11;
                    end

                    // ----------------------------------------------------------------
                    // C2: T 第 1 列 (P*A^T col1) -> t01, t11
                    //     同拍配置 A*T 第 0 列 (A * [t00;t10])
                    // ----------------------------------------------------------------
                    4'd2:
                    begin
                        t01 <= trunc2N_to_N(add0_2N);
                        t11 <= trunc2N_to_N(add1_2N);

                        // A * [t00; t10] -> S 第 0 列
                        m0_a <= a00;
                        m0_b <= t00;
                        m1_a <= a01;
                        m1_b <= t10;
                        m2_a <= a10;
                        m2_b <= t00;
                        m3_a <= a11;
                        m3_b <= t10;
                    end

                    // ----------------------------------------------------------------
                    // C3: S 第 0 列列求和 -> s00_tmp, s10_tmp
                    //     同拍配置 A*T 第 1 列 (A * [t01;t11])
                    // ----------------------------------------------------------------
                    4'd3:
                    begin
                        s00_tmp <= add0_2N; // sum(m0+m1) for col0
                        s10_tmp <= add1_2N; // sum(m2+m3) for col0

                        // A * [t01; t11] -> S 第 1 列
                        m0_a <= a00;
                        m0_b <= t01;
                        m1_a <= a01;
                        m1_b <= t11;
                        m2_a <= a10;
                        m2_b <= t01;
                        m3_a <= a11;
                        m3_b <= t11;
                    end

                    // ----------------------------------------------------------------
                    // C4: S 第 0 列 + Q00/Q10 -> s00_2N, s10_2N
                    // ----------------------------------------------------------------
                    4'd4:
                    begin
                        s00_2N <= add0_2N; // s00_tmp + Q00_2N
                        s10_2N <= add1_2N; // s10_tmp + Q10_2N
                    end

                    // ----------------------------------------------------------------
                    // C5: S 第 1 列列求和 -> s01_tmp, s11_tmp
                    // ----------------------------------------------------------------
                    4'd5:
                    begin
                        s01_tmp <= add0_2N; // sum(m0+m1) for col1
                        s11_tmp <= add1_2N; // sum(m2+m3) for col1
                    end

                    // ----------------------------------------------------------------
                    // C6: S 第 1 列 + Q01/Q11 -> s01_2N, s11_2N
                    // ----------------------------------------------------------------
                    4'd6:
                    begin
                        s01_2N <= add0_2N; // s01_tmp + Q01_2N
                        s11_2N <= add1_2N; // s11_tmp + Q11_2N
                    end

                    // ----------------------------------------------------------------
                    // C7: done 拉高一拍
                    // ----------------------------------------------------------------
                    4'd7:
                    begin
                        running <= 1'b0;
                        done    <= 1'b1;
                    end

                    default:
                        ;
                endcase
            end
        end
    end

endmodule
