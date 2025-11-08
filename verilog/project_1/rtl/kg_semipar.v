// ============================================================================
// kg_semipar.v : 2x2 Kalman Gain (semi-parallel, 4xMUL full-precision + 2xADD)
// - MUL 保持 2N 精度；列求和在 2N 域完成；仅在需要 n 位时做中位截短
// - R 先符号扩展到 2N 并左移 FRAC 与乘积对齐
// - 固定 17 拍时序：0..16（在 16 拍结束并由外层计数逻辑拉高 done）
// - 本版调用 inv2_serial（4-cycle）：start@8 → done@11
// - 按你的要求：K 的两次乘法与寄存由 12/13/14 改为 13/14/15
// ============================================================================

`include "fxp_types.vh"

module kg_semipar #(
        parameter integer N    = `FXP_N,
        parameter integer FRAC = `FXP_FRAC
    ) (
        input  wire                   clk,
        input  wire                   rst_n,
        input  wire                   start,

        // ---- Inputs (lowercase) ----
        // P_prior
        input  wire signed [N-1:0]    p_prior00, p_prior01, p_prior10, p_prior11,
        // H
        input  wire signed [N-1:0]    h00, h01, h10, h11,
        // R
        input  wire signed [N-1:0]    r00, r01, r10, r11,

        // ---- Control ----
        output reg                    done,

        // ---- Outputs (UPPERCASE) ----
        output wire signed [N-1:0]    K00, K01, K10, K11
    );

    // ---------------- 17-cycle frame counter ----------------
    reg [4:0] cyc;
    reg       running;
    always @(posedge clk or negedge rst_n)
    begin
        if (!rst_n)
        begin
            running <= 1'b0;
            cyc     <= 5'd0;
            done    <= 1'b0;
        end
        else
        begin
            done <= 1'b0;
            if (start && !running)
            begin
                running <= 1'b1;
                cyc     <= 5'd0;
            end
            else if (running)
            begin
                if (cyc == 5'd16)
                begin
                    running <= 1'b0;
                    done    <= 1'b1;
                end
                else
                begin
                    cyc <= cyc + 5'd1;
                end
            end
        end
    end

    // ---------------- Helpers ----------------
    // R 对齐到 2N (与乘法 full 对齐)
    wire signed [2*N-1:0] R00_2N = ({{N{r00[N-1]}}, r00}) <<< FRAC;
    wire signed [2*N-1:0] R01_2N = ({{N{r01[N-1]}}, r01}) <<< FRAC;
    wire signed [2*N-1:0] R10_2N = ({{N{r10[N-1]}}, r10}) <<< FRAC;
    wire signed [2*N-1:0] R11_2N = ({{N{r11[N-1]}}, r11}) <<< FRAC;

    function [N-1:0] trunc2N_to_N;
        input signed [2*N-1:0] x2n;
        begin
            trunc2N_to_N = x2n[FRAC+N-1 : FRAC];
        end
    endfunction

    // ---------------- 4 MULs (full 2N) ----------------
    reg  signed [N-1:0] m0_a, m0_b, m1_a, m1_b, m2_a, m2_b, m3_a, m3_b;
    wire signed [2*N-1:0] m0_f, m1_f, m2_f, m3_f;
    wire signed [N-1:0]   m0_t, m1_t, m2_t, m3_t; // 截短口保留（有处用到）
    fxp_mul #(N, FRAC) U_M0 (.a(m0_a), .b(m0_b), .y_full(m0_f), .y_trunc(m0_t));
    fxp_mul #(N, FRAC) U_M1 (.a(m1_a), .b(m1_b), .y_full(m1_f), .y_trunc(m1_t));
    fxp_mul #(N, FRAC) U_M2 (.a(m2_a), .b(m2_b), .y_full(m2_f), .y_trunc(m2_t));
    fxp_mul #(N, FRAC) U_M3 (.a(m3_a), .b(m3_b), .y_full(m3_f), .y_trunc(m3_t));

    // 列求和（2N 域）
    wire signed [2*N:0]   sum01_full, sum23_full;
    wire signed [2*N-1:0] sum01_2N,   sum23_2N;
    fxp_add #(2*N) U_SUM01 (.a(m0_f), .b(m1_f), .y_full(sum01_full), .y_trunc(sum01_2N));
    fxp_add #(2*N) U_SUM23 (.a(m2_f), .b(m3_f), .y_full(sum23_full), .y_trunc(sum23_2N));

    // ---------------- T 与 S ----------------
    reg  signed [N-1:0]   t00, t01, t10, t11;            // n 域
    reg  signed [2*N-1:0] s00_2N, s01_2N, s10_2N, s11_2N;// 2N 域
    wire signed [N-1:0]   s00_n = trunc2N_to_N(s00_2N);
    wire signed [N-1:0]   s01_n = trunc2N_to_N(s01_2N);
    wire signed [N-1:0]   s10_n = trunc2N_to_N(s10_2N);
    wire signed [N-1:0]   s11_n = trunc2N_to_N(s11_2N);

    // ---------------- INV：4-cycle 串行逆 ----------------
    reg  start_inv;
    wire inv_done;
    wire signed [N-1:0] si00, si01, si10, si11;

    inv2_serial #(N, FRAC) U_INV (
        .clk   (clk),
        .rst_n (rst_n),
        .start (start_inv),
        .a     (s00_n),
        .b     (s01_n),
        .c     (s10_n),
        .d     (s11_n),
        .done  (inv_done),
        .IA    (si00),
        .IB    (si01),
        .IC    (si10),
        .ID    (si11)
    );

    // ---------------- K（寄存输出寄存器） ----------------
    reg  signed [N-1:0] K00_r, K01_r, K10_r, K11_r;
    assign K00 = K00_r;
    assign K01 = K01_r;
    assign K10 = K10_r;
    assign K11 = K11_r;

    // ---------------- Datapath schedule（保持 17 拍；K 的两步整体后移 1 拍） ----------------
    always @(posedge clk or negedge rst_n)
    begin
        if (!rst_n)
        begin
            m0_a<=0; m0_b<=0; m1_a<=0; m1_b<=0; m2_a<=0; m2_b<=0; m3_a<=0; m3_b<=0;
            t00<=0; t01<=0; t10<=0; t11<=0;
            s00_2N<=0; s01_2N<=0; s10_2N<=0; s11_2N<=0;
            K00_r<=0; K01_r<=0; K10_r<=0; K11_r<=0;
            start_inv<=1'b0;
        end
        else if (running)
        begin
            // 默认拉低 start_inv（单拍脉冲）
            start_inv <= 1'b0;

            case (cyc)
                // 0..1 : t00,t10 = P*H(:,0)
                5'd0:
                begin
                    m0_a<=p_prior00; m0_b<=h00;
                    m1_a<=p_prior01; m1_b<=h01;
                    m2_a<=p_prior10; m2_b<=h00;
                    m3_a<=p_prior11; m3_b<=h01;
                end
                5'd1:
                begin
                    t00 <= trunc2N_to_N(sum01_2N);
                    t10 <= trunc2N_to_N(sum23_2N);
                end

                // 2..3 : t01,t11 = P*H(:,1)
                5'd2:
                begin
                    m0_a<=p_prior00; m0_b<=h10;
                    m1_a<=p_prior01; m1_b<=h11;
                    m2_a<=p_prior10; m2_b<=h10;
                    m3_a<=p_prior11; m3_b<=h11;
                end
                5'd3:
                begin
                    t01 <= trunc2N_to_N(sum01_2N);
                    t11 <= trunc2N_to_N(sum23_2N);
                end

                // 4..5 : S(:,0) = H * [t00;t10] + R(:,0)
                5'd4:
                begin
                    m0_a<=h00; m0_b<=t00;
                    m1_a<=h01; m1_b<=t10;
                    m2_a<=h10; m2_b<=t00;
                    m3_a<=h11; m3_b<=t10;
                end
                5'd5:
                begin
                    s00_2N <= sum01_2N + R00_2N;
                    s10_2N <= sum23_2N + R10_2N;
                end

                // 6..7 : S(:,1) = H * [t01;t11] + R(:,1)
                5'd6:
                begin
                    m0_a<=h00; m0_b<=t01;
                    m1_a<=h01; m1_b<=t11;
                    m2_a<=h10; m2_b<=t01;
                    m3_a<=h11; m3_b<=t11;
                end
                5'd7:
                begin
                    s01_2N <= sum01_2N + R01_2N;
                    s11_2N <= sum23_2N + R11_2N;
                end

                // 8 : 启动 INV（4-cycle）：start@8 → done@11
                5'd8:
                begin
                    start_inv <= 1'b1;    // 单拍脉冲
                end

                // 13 : 使用 INV 结果，装载 K01/K11 的乘法   <-- 从原 12 后移 1 拍
                5'd13:
                begin
                    // K01 = t00*si01 + t01*si11
                    // K11 = t10*si01 + t11*si11
                    m0_a<=t00; m0_b<=si01;
                    m1_a<=t01; m1_b<=si11;
                    m2_a<=t10; m2_b<=si01;
                    m3_a<=t11; m3_b<=si11;
                end

                // 14 : 寄存 K01/K11，同时装载 K00/K10 的乘法   <-- 从原 13 后移 1 拍
                5'd14:
                begin
                    K01_r <= trunc2N_to_N(sum01_2N);
                    K11_r <= trunc2N_to_N(sum23_2N);
                    // 下一列：K00 = t00*si00 + t01*si10
                    //        K10 = t10*si00 + t11*si10
                    m0_a<=t00; m0_b<=si00;
                    m1_a<=t01; m1_b<=si10;
                    m2_a<=t10; m2_b<=si00;
                    m3_a<=t11; m3_b<=si10;
                end

                // 15 : 寄存 K00/K10（从原 14 后移 1 拍）
                5'd15:
                begin
                    K00_r <= trunc2N_to_N(sum01_2N);
                    K10_r <= trunc2N_to_N(sum23_2N);
                end

                // 16 : 空拍；计数器在本拍结束后置 done=1（外层计数逻辑）
                default: ;
            endcase
        end
    end

endmodule
