// ============================================================================
// top_kf.v : Kalman Filter Top (2x2), STRICT 34-cycle frame controller
// ----------------------------------------------------------------------------
// Cycle map (0..33, total 34 cycles):
//   00 : start PS(prior_state_serial) + PC(prior_cov_semipar)
//   08 : start Z (est_output_serial)
//   10 : start KG (kg_semipar) using R_prev  +  start Q (q_serial)   <-- 同拍
//   12 : start R (r_serial)  → 产出 R_next（供下一帧）
//   26 : start PST (post_state_serial)  [needs K, zhat, x_prior]
//        start POC (post_cov_semipar)   [needs K, P_prior]
//   33 : frame DONE; commit P_post -> P_prev, R_next -> R_prev, Q_next -> Q_prev
// Notes:
// - PC 使用 Q_prev（上一帧结果）；KG 使用 R_prev（上一帧结果）。
// - R/Q 在本帧并行计算产生 R_next/Q_next，用于下一帧，解耦关键路径。
// ============================================================================

`include "fxp_types.vh"

module top_kf
    #( parameter integer N=`FXP_N, parameter integer FRAC=`FXP_FRAC )
     (
         input  wire                     clk,
         input  wire                     rst_n,
         input  wire                     start,          // pulse to begin a 34-cycle frame

         // ---- System matrices & vectors (lowercase inputs) ----
         input  wire signed [N-1:0]      a00, a01, a10, a11,
         input  wire signed [N-1:0]      b00, b01, b10, b11,
         input  wire signed [N-1:0]      h00, h01, h10, h11,
         input  wire signed [N-1:0]      x00_prev, x10_prev,
         input  wire signed [N-1:0]      u00, u10,

         // measurements for this frame
         input  wire signed [N-1:0]      z00_meas, z10_meas,

         // r_serial extra inputs (strict per your spec)
         input  wire signed [N-1:0]      beta,
         input  wire signed [N-1:0]      sigma2_00, sigma2_01,
         input  wire signed [N-1:0]      sigma2_10, sigma2_11,

         // ---- Outputs of this frame ----
         output reg                      done,           // asserted exactly on cycle 33
         output wire signed [N-1:0]      X00_post, X10_post
     );

    // ----------------------------------------------------------------------------
    // Global frame counter: down-counter 33..0  →  derive cyc = 0..33
    // ----------------------------------------------------------------------------
    reg       running;
    reg [5:0] cnt;       // 33..0
    wire [5:0] cyc = 6'd33 - cnt;      // 0..33 only valid when running=1

    // 原始：在 cnt==0 拉高 done 并清 running
    // 修改后：在 cnt==1 拉高 done；在 cnt==0 仅清 running（提交仍在 cnt==0）

    always @(posedge clk or negedge rst_n)
    begin
        if(!rst_n)
        begin
            running <= 1'b0;
            cnt     <= 6'd0;
            done    <= 1'b0;
        end
        else
        begin
            done <= 1'b0; // one-cycle pulse by default

            if(start && !running)
            begin
                running <= 1'b1;
                cnt     <= 6'd33;   // 仍然把 start 当拍作为“装载”，下一拍是 C0
            end
            else if(running)
            begin
                if(cnt == 6'd1)
                begin
                    // 在 C33（对外口径）拉高 done，同时把计数推进到 0
                    done <= 1'b1;
                    cnt  <= 6'd0;
                end
                else if(cnt == 6'd0)
                begin
                    // C33 的下一拍：完成提交并退出运行态
                    running <= 1'b0;
                end
                else
                begin
                    cnt <= cnt - 6'd1;
                end
            end
        end
    end

    // 提交时刻保持不变：仍在“内部 C33”
    wire commit_c33 = running && (cnt==6'd0);


    // ----------------------------------------------------------------------------
    // Frame-latency decoupled states: R_prev / Q_prev / P_prev
    // ----------------------------------------------------------------------------
    localparam signed [N-1:0] ONE = $signed(1) <<< FRAC;

    // R_prev / Q_prev（上一帧保留，默认单位对角）
    reg signed [N-1:0] R11_prev,R12_prev,R21_prev,R22_prev;
    reg signed [N-1:0] Q11_prev,Q12_prev,Q21_prev,Q22_prev;

    // P_prev（上一帧的 P_post）
    reg signed [N-1:0] P_post00_prev, P_post01_prev, P_post10_prev, P_post11_prev;

    // 本帧新计算出的 R_next / Q_next / P_post_w（在 C33 提交为 *_prev）
    reg signed [N-1:0] R11_next,R12_next,R21_next,R22_next;
    reg signed [N-1:0] Q11_next,Q12_next,Q21_next,Q22_next;

    wire signed [N-1:0] P_post00_w, P_post01_w, P_post10_w, P_post11_w;

    // reset & commit（在帧末 C33 提交）
    always @(posedge clk or negedge rst_n)
    begin
        if(!rst_n)
        begin
            R11_prev<=ONE;
            R22_prev<=ONE;
            R12_prev<=0;
            R21_prev<=0;
            Q11_prev<=ONE;
            Q22_prev<=ONE;
            Q12_prev<=0;
            Q21_prev<=0;
            P_post00_prev<=ONE;
            P_post11_prev<=ONE;
            P_post01_prev<=0;
            P_post10_prev<=0;
        end
        else if(commit_c33)
        begin
            // commit R/Q
            R11_prev<=R11_next;
            R12_prev<=R12_next;
            R21_prev<=R21_next;
            R22_prev<=R22_next;
            Q11_prev<=Q11_next;
            Q12_prev<=Q12_next;
            Q21_prev<=Q21_next;
            Q22_prev<=Q22_next;
            // commit P_post -> P_prev
            P_post00_prev<=P_post00_w;
            P_post01_prev<=P_post01_w;
            P_post10_prev<=P_post10_w;
            P_post11_prev<=P_post11_w;
        end
    end

    // ----------------------------------------------------------------------------
    // Ready latches (frame-scoped) -- "更宽松"的关键
    // ----------------------------------------------------------------------------
    reg kg_ready, z_ready, pc_ready;
    always @(posedge clk or negedge rst_n)
    begin
        if(!rst_n)
        begin
            kg_ready <= 1'b0;
            z_ready  <= 1'b0;
            pc_ready <= 1'b0;
        end
        else
        begin
            // 帧开始清零
            if (start && !running)
            begin
                kg_ready <= 1'b0;
                z_ready  <= 1'b0;
                pc_ready <= 1'b0;
            end
            // 任意拍捕获 done → 置位保持到帧末
            if (kg_done)
                kg_ready <= 1'b1;
            if (z_done)
                z_ready  <= 1'b1;
            if (pc_done)
                pc_ready <= 1'b1;
            // 帧末清零
            if (commit_c33)
            begin
                kg_ready <= 1'b0;
                z_ready  <= 1'b0;
                pc_ready <= 1'b0;
            end
        end
    end

    // ----------------------------------------------------------------------------
    // Start pulses (single-cycle, with readiness gating)
    // ----------------------------------------------------------------------------
    wire ps_start  = running && (cyc==6'd0);
    wire pc_start  = running && (cyc==6'd0);
    wire z_start   = running && (cyc==6'd8);
    wire kg_start  = running && (cyc==6'd10);   // 与 q 同拍
    wire q_start   = running && (cyc==6'd10);   // 与 kg 同拍
    wire r_start   = running && (cyc==6'd12);

    // 关键：固定 C26 启动（满足依赖才发脉冲）
    // - KG 17-cycle 从 C10 起，典型在 C26 进入就绪，所以用 kg_ready 兜底
    // - Z 4-cycle 从 C8 起，C12 前就绪；PC 8-cycle 从 C0 起，C8 前就绪
    // 如果你的 KG 实际在 C27 才 done，可把 pst_start 改到 (cyc==27) && kg_ready；
    // 但 POC 8-cycle 要在 C26 起以保证 C33 前结束（不要推迟）。
    wire pst_start = running && (cyc==6'd26) && kg_ready && z_ready;
    wire poc_start = running && (cyc==6'd26) && kg_ready && pc_ready;

    // ----------------------------------------------------------------------------
    // 1) Prior state (serial, 8 cyc) : X_PRIOR**
    // ----------------------------------------------------------------------------
    wire ps_done;
    wire signed [N-1:0] X_PRIOR00, X_PRIOR10;

    prior_state_serial #(N,FRAC) U_PS (
                           .clk(clk), .rst_n(rst_n), .start(ps_start),
                           .x00(x00_prev), .x10(x10_prev),
                           .a00(a00), .a01(a01), .a10(a10), .a11(a11),
                           .u00(u00), .u10(u10),
                           .b00(b00), .b01(b01), .b10(b10), .b11(b11),
                           .done(ps_done),
                           .X_PRIOR00(X_PRIOR00), .X_PRIOR10(X_PRIOR10)
                       );

    // ----------------------------------------------------------------------------
    // 2) Prior covariance (semi-parallel, 8 cyc) : P_PRIOR** (uses Q_prev)
    // ----------------------------------------------------------------------------
    wire pc_done;
    wire signed [N-1:0] P_PRIOR00, P_PRIOR01, P_PRIOR10, P_PRIOR11;

    prior_cov_semipar #(N,FRAC) U_PC (
                          .clk(clk), .rst_n(rst_n), .start(pc_start),
                          .a00(a00), .a01(a01), .a10(a10), .a11(a11),
                          .p00(P_post00_prev), .p01(P_post01_prev), .p10(P_post10_prev), .p11(P_post11_prev),
                          .q00(Q11_prev), .q01(Q12_prev), .q10(Q21_prev), .q11(Q22_prev),
                          .done(pc_done),
                          .P_PRIOR00(P_PRIOR00), .P_PRIOR01(P_PRIOR01),
                          .P_PRIOR10(P_PRIOR10), .P_PRIOR11(P_PRIOR11)
                      );

    // ----------------------------------------------------------------------------
    // 3) Estimated output (serial, 4 cyc) : Z00,Z10
    // ----------------------------------------------------------------------------
    wire z_done;
    wire signed [N-1:0] Z00, Z10;

    est_output_serial #(N,FRAC) U_Z (
                          .clk(clk), .rst_n(rst_n), .start(z_start),
                          .h00(h00), .h01(h01), .h10(h10), .h11(h11),
                          .x00(X_PRIOR00), .x10(X_PRIOR10),
                          .done(z_done),
                          .Z00(Z00), .Z10(Z10)
                      );

    // ----------------------------------------------------------------------------
    // 4) KG (semi-parallel, 17 cyc) : K00..K11   - uses P_PRIOR** and R_prev
    // ----------------------------------------------------------------------------
    wire kg_done;
    wire signed [N-1:0] K00, K01, K10, K11;

    kg_semipar #(N,FRAC) U_KG (
                   .clk(clk), .rst_n(rst_n), .start(kg_start),
                   .p_prior00(P_PRIOR00), .p_prior01(P_PRIOR01),
                   .p_prior10(P_PRIOR10), .p_prior11(P_PRIOR11),
                   .h00(h00), .h01(h01), .h10(h10), .h11(h11),
                   .r00(R11_prev), .r01(R12_prev), .r10(R21_prev), .r11(R22_prev),
                   .done(kg_done),
                   .K00(K00), .K01(K01), .K10(K10), .K11(K11)
               );

    // ----------------------------------------------------------------------------
    // 5) Q (serial, 3 cyc) : Q_next - 与 KG 同拍启动（C10）
    // ----------------------------------------------------------------------------
    wire q_done;
    wire signed [N-1:0] Q11_w, Q12_w, Q21_w, Q22_w;

    q_serial #(N,FRAC) U_Q (
                 .clk(clk), .rst_n(rst_n), .start(q_start),
                 .x00_now(X_PRIOR00), .x01_now(X_PRIOR10),
                 .x00_prev(x00_prev), .x01_prev(x10_prev),
                 .done(q_done),
                 .Q11(Q11_w), .Q12(Q12_w), .Q21(Q21_w), .Q22(Q22_w)
             );
    always @(posedge clk or negedge rst_n)
    begin
        if(!rst_n)
        begin
            Q11_next<=ONE;
            Q22_next<=ONE;
            Q12_next<=0;
            Q21_next<=0;
        end
        else if(q_done)
        begin
            Q11_next<=Q11_w;
            Q12_next<=Q12_w;
            Q21_next<=Q21_w;
            Q22_next<=Q22_w;
        end
    end

    // ----------------------------------------------------------------------------
    // 6) R (serial, 3 cyc) : R_next - 供下一帧
    // ----------------------------------------------------------------------------
    wire r_done;
    wire signed [N-1:0] R11_w, R12_w, R21_w, R22_w;

    r_serial #(N,FRAC) U_R (
                 .clk(clk), .rst_n(rst_n), .start(r_start),
                 .beta(beta),
                 .sigma2_00(sigma2_00), .sigma2_01(sigma2_01),
                 .sigma2_10(sigma2_10), .sigma2_11(sigma2_11),
                 .z00(z00_meas), .z10(z10_meas),
                 .zhat00(Z00), .zhat10(Z10),
                 .done(r_done),
                 .R11(R11_w), .R12(R12_w), .R21(R21_w), .R22(R22_w)
             );
    always @(posedge clk or negedge rst_n)
    begin
        if(!rst_n)
        begin
            R11_next<=ONE;
            R22_next<=ONE;
            R12_next<=0;
            R21_next<=0;
        end
        else if(r_done)
        begin
            R11_next<=R11_w;
            R12_next<=R12_w;
            R21_next<=R21_w;
            R22_next<=R22_w;
        end
    end

    // ----------------------------------------------------------------------------
    // 7) Posterior state (serial, 5 cyc) : X00_post, X10_post
    // ----------------------------------------------------------------------------
    wire pst_done;
    post_state_serial #(N,FRAC) U_PST (
                          .clk(clk), .rst_n(rst_n), .start(pst_start),
                          .x00_prior(X_PRIOR00), .x10_prior(X_PRIOR10),
                          .z00_meas(z00_meas), .z10_meas(z10_meas),
                          .z00_hat(Z00), .z10_hat(Z10),
                          .k00(K00), .k01(K01), .k10(K10), .k11(K11),
                          .done(pst_done),
                          .X00_post(X00_post), .X10_post(X10_post)
                      );

    // ----------------------------------------------------------------------------
    // 8) Posterior covariance (semi-parallel, 8 cyc) : P_post**
    // ----------------------------------------------------------------------------
    wire poc_done;
    post_cov_semipar #(N,FRAC) U_POC (
                         .clk(clk), .rst_n(rst_n), .start(poc_start),
                         .k00(K00), .k01(K01), .k10(K10), .k11(K11),
                         .h00(h00), .h01(h01), .h10(h10), .h11(h11),
                         .p_prior00(P_PRIOR00), .p_prior01(P_PRIOR01),
                         .p_prior10(P_PRIOR10), .p_prior11(P_PRIOR11),
                         .done(poc_done),
                         .P_post00(P_post00_w), .P_post01(P_post01_w),
                         .P_post10(P_post10_w), .P_post11(P_post11_w)
                     );

endmodule
