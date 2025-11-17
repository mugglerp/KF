// ============================================================================
// post_cov_semipar.v : P_post = P_prior - (K*H)*P_prior  (2x2 semiparallel)
// - 4x MUL: y_full=2N (2*FRAC fractional), y_trunc unused
// - 2x ADD: 输入/输出均在 2N 域（与乘积对齐）
// - 2x SUB: 在 2N 域做 P - I，然后只在最终输出处截短回 N 位
// - Latency: 8 cycles (0..7), done 在 cyc==7 拉高
// - 符号扩展与小数点对齐：P(N,FRAC) -> P_2N = (sign-extend)<<FRAC 使小数位从 FRAC 对齐到 2*FRAC
// ============================================================================

module post_cov_semipar
    #( parameter integer N=20, parameter integer FRAC=10 )
     (
         input  wire clk,
         input  wire rst_n,
         input  wire start,

         input  wire signed [N-1:0] k00,k01,k10,k11,
         input  wire signed [N-1:0] h00,h01,h10,h11,
         input  wire signed [N-1:0] p_prior00,p_prior01,p_prior10,p_prior11,

         output reg  done,
         output wire signed [N-1:0] P_post00,P_post01,P_post10,P_post11
     );

    // ---------------- Helpers ----------------
    // N.FRAC -> 2N.(2*FRAC)
    function signed [2*N-1:0] fxp_ext_2N;
        input signed [N-1:0] xN;
        begin
            fxp_ext_2N = ({{N{xN[N-1]}}, xN}) <<< FRAC; // 先符号扩展，再左移 FRAC, 小数位对齐到 2*FRAC
        end
    endfunction

    // 2N.(2*FRAC) -> N.FRAC （向零截短）
    function signed [N-1:0] trunc_2N_to_N;
        input signed [2*N-1:0] x2N;
        begin
            trunc_2N_to_N = x2N[FRAC+N-1 : FRAC];
        end
    endfunction

    // ---------------- Frame counter (0..7) ----------------
    reg [3:0] cyc;
    reg       running;
    always @(posedge clk or negedge rst_n)
    begin
        if(!rst_n)
        begin
            running <= 1'b0;
            cyc <= 4'd0;
            done <= 1'b0;
        end
        else
        begin
            done <= 1'b0;
            if(start && !running)
            begin
                running <= 1'b1;
                cyc     <= 4'd1;
            end
            else if(running)
            begin
                if(cyc==4'd8)
                begin
                    running <= 1'b0;
                    done    <= 1'b1;
                end
                else
                begin
                    cyc <= cyc + 4'd1;
                end
            end
        end
    end

    // ---------------- 4 MULs (full 2N) ----------------
    reg  signed [N-1:0] m0_a, m0_b, m1_a, m1_b, m2_a, m2_b, m3_a, m3_b;
    wire signed [2*N-1:0] m0_f, m1_f, m2_f, m3_f;
    wire signed [N-1:0]   m0_t_unused, m1_t_unused, m2_t_unused, m3_t_unused;

    fxp_mul #(N,FRAC) U_M0 (.a(m0_a), .b(m0_b), .y_full(m0_f), .y_trunc(m0_t_unused));
    fxp_mul #(N,FRAC) U_M1 (.a(m1_a), .b(m1_b), .y_full(m1_f), .y_trunc(m1_t_unused));
    fxp_mul #(N,FRAC) U_M2 (.a(m2_a), .b(m2_b), .y_full(m2_f), .y_trunc(m2_t_unused));
    fxp_mul #(N,FRAC) U_M3 (.a(m3_a), .b(m3_b), .y_full(m3_f), .y_trunc(m3_t_unused));

    // ---------------- 2N 域列求和 ----------------
    wire signed [2*N:0] sum01_full, sum23_full;
    wire signed [2*N-1:0] sum01_2N, sum23_2N;

    fxp_add #(2*N) U_SUM01 (.a(m0_f), .b(m1_f), .y_full(sum01_full), .y_trunc(sum01_2N));
    fxp_add #(2*N) U_SUM23 (.a(m2_f), .b(m3_f), .y_full(sum23_full), .y_trunc(sum23_2N));

    // ---------------- KH 中间值（t 符合乘法器输入 n 位） ----------------
    reg signed [N-1:0] Kh00_n, Kh01_n, Kh10_n, Kh11_n;

    // ---------------- P 扩展到 2N（2*FRAC 小数） ----------------
    wire signed [2*N-1:0] p_prior00_2N = fxp_ext_2N(p_prior00);
    wire signed [2*N-1:0] p_prior01_2N = fxp_ext_2N(p_prior01);
    wire signed [2*N-1:0] p_prior10_2N = fxp_ext_2N(p_prior10);
    wire signed [2*N-1:0] p_prior11_2N = fxp_ext_2N(p_prior11);

    // ---------------- I = (K*H)*P （2N 域） ----------------
    reg  signed [2*N-1:0] I00_2N, I01_2N, I10_2N, I11_2N;

    // ---------------- 2N 域减法：P - I ----------------
    wire signed [2*N:0]  sub11_f, sub12_f, sub21_f, sub22_f;
    wire signed [2*N-1:0] PPOST11_2N, PPOST12_2N, PPOST21_2N, PPOST22_2N;

    fxp_sub #(2*N) U_SUB11 (.a(p_prior00_2N), .b(I00_2N), .y_full(sub11_f), .y_trunc(PPOST11_2N));
    fxp_sub #(2*N) U_SUB12 (.a(p_prior01_2N), .b(I01_2N), .y_full(sub12_f), .y_trunc(PPOST12_2N));
    fxp_sub #(2*N) U_SUB21 (.a(p_prior10_2N), .b(I10_2N), .y_full(sub21_f), .y_trunc(PPOST21_2N));
    fxp_sub #(2*N) U_SUB22 (.a(p_prior11_2N), .b(I11_2N), .y_full(sub22_f), .y_trunc(PPOST22_2N));

    // ---------------- 输出（N 位，截短） ----------------
    assign P_post00 = trunc_2N_to_N(PPOST11_2N);
    assign P_post01 = trunc_2N_to_N(PPOST12_2N);
    assign P_post10 = trunc_2N_to_N(PPOST21_2N);
    assign P_post11 = trunc_2N_to_N(PPOST22_2N);

    // ---------------- 8 拍时序 ----------------
    // 0: 计算 KH 第 1 行 (Kh00,Kh01) 的四个乘法
    // 1: 截短保存 Kh00_n,Kh01_n
    // 2: 计算 KH 第 2 行 (Kh10,Kh11) 的四个乘法
    // 3: 截短保存 Kh10_n,Kh11_n
    // 4: 计算 I00,I01 ：用 KH 第 1 行 与 P 的两列
    // 5: 采样 I00_2N,I01_2N
    // 6: 计算 I10,I11 ：用 KH 第 2 行 与 P 的两列
    // 7: 采样 I10_2N,I11_2N；此时减法与输出已就绪；done 在上面逻辑中拉高
    always @(posedge clk or negedge rst_n)
    begin
        if(!rst_n)
        begin
            m0_a<=0;
            m0_b<=0;
            m1_a<=0;
            m1_b<=0;
            m2_a<=0;
            m2_b<=0;
            m3_a<=0;
            m3_b<=0;
            Kh00_n<=0;
            Kh01_n<=0;
            Kh10_n<=0;
            Kh11_n<=0;
            I00_2N<=0;
            I01_2N<=0;
            I10_2N<=0;
            I11_2N<=0;
        end
        else if(running)
        begin
            case(cyc)
                4'd1:
                begin
                    // Kh00 = k00*h00 + k01*h10
                    // Kh01 = k00*h01 + k01*h11
                    m0_a <= k00;
                    m0_b <= h00;
                    m1_a <= k01;
                    m1_b <= h10;
                    m2_a <= k00;
                    m2_b <= h01;
                    m3_a <= k01;
                    m3_b <= h11;
                end
                4'd2:
                begin
                    Kh00_n <= trunc_2N_to_N(sum01_2N);
                    Kh01_n <= trunc_2N_to_N(sum23_2N);
                end
                4'd3:
                begin
                    // Kh10 = k10*h00 + k11*h10
                    // Kh11 = k10*h01 + k11*h11
                    m0_a <= k10;
                    m0_b <= h00;
                    m1_a <= k11;
                    m1_b <= h10;
                    m2_a <= k10;
                    m2_b <= h01;
                    m3_a <= k11;
                    m3_b <= h11;
                end
                4'd4:
                begin
                    Kh10_n <= trunc_2N_to_N(sum01_2N);
                    Kh11_n <= trunc_2N_to_N(sum23_2N);
                end
                4'd5:
                begin
                    // I00 = Kh00*p_prior00 + Kh01*p_prior10
                    // I01 = Kh00*p_prior01 + Kh01*p_prior11
                    m0_a <= Kh00_n;
                    m0_b <= p_prior00;
                    m1_a <= Kh01_n;
                    m1_b <= p_prior10;
                    m2_a <= Kh00_n;
                    m2_b <= p_prior01;
                    m3_a <= Kh01_n;
                    m3_b <= p_prior11;
                end
                4'd6:
                begin
                    I00_2N <= sum01_2N;
                    I01_2N <= sum23_2N;
                end
                4'd7:
                begin
                    // I10 = Kh10*p_prior00 + Kh11*p_prior10
                    // I11 = Kh10*p_prior01 + Kh11*p_prior11
                    m0_a <= Kh10_n;
                    m0_b <= p_prior00;
                    m1_a <= Kh11_n;
                    m1_b <= p_prior10;
                    m2_a <= Kh10_n;
                    m2_b <= p_prior01;
                    m3_a <= Kh11_n;
                    m3_b <= p_prior11;
                end
                4'd8:
                begin
                    I10_2N <= sum01_2N;
                    I11_2N <= sum23_2N;
                end
                default:
                    ;
            endcase
        end
    end

endmodule