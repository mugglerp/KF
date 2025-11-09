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
    reg [5:0] cnt;                 // 33..0
    wire [5:0] cyc = 6'd33 - cnt;  // 0..33 (valid when running=1)

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
            done <= 1'b0; // one-cycle pulse
            if(start && !running)
            begin
                running <= 1'b1;
                cnt     <= 6'd33;       // next clock becomes C0
            end
            else if(running)
            begin
                if(cnt == 6'd1)
                begin
                    // externally-visible C33
                    done <= 1'b1;
                    cnt  <= 6'd0;
                end
                else if(cnt == 6'd0)
                begin
                    // one cycle after C33
                    running <= 1'b0;
                end
                else
                begin
                    cnt <= cnt - 6'd1;
                end
            end
        end
    end

    // commit moment = internal C33 (cnt==0 while still running)
    wire commit_c33 = running && (cnt==6'd0);

    // ----------------------------------------------------------------------------
    // Declarations that are referenced early (avoid "used before declaration")
    // ----------------------------------------------------------------------------
    // Sub-module done signals (declared early so they can be used below)
    wire ps_done, pc_done, z_done, kg_done, q_done, r_done, pst_done, poc_done;

    // Sub-module outputs that may be tapped by later logic
    wire signed [N-1:0] X_PRIOR00, X_PRIOR10;
    wire signed [N-1:0] P_PRIOR00, P_PRIOR01, P_PRIOR10, P_PRIOR11;
    wire signed [N-1:0] Z00, Z10;
    wire signed [N-1:0] K00, K01, K10, K11;
    wire signed [N-1:0] Q11_w, Q12_w, Q21_w, Q22_w;
    wire signed [N-1:0] R11_w, R12_w, R21_w, R22_w;

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
    // Start pulses (single-cycle)
    // ----------------------------------------------------------------------------
    // 导出这些内部起始脉冲名，方便 testbench 分层探针（dut.ps_start 等）
    wire ps_start  = running && (cyc==6'd0);
    wire pc_start  = running && (cyc==6'd0);
    wire z_start   = running && (cyc==6'd8);
    wire kg_start  = running && (cyc==6'd10);  // 与 q 同拍
    wire q_start   = running && (cyc==6'd10);  // 与 kg 同拍
    wire r_start   = running && (cyc==6'd12);

    // ----------------------------------------------------------------------------
    // Ready latches (frame-scoped)
    // （必须在 kg_done/z_done/pc_done 提前声明之后，才能在此使用）
    // ----------------------------------------------------------------------------
    reg kg_ready, z_ready, pc_ready;
    always @(posedge clk or negedge rst_n)
    begin
        if(!rst_n)
        begin
            kg_ready <= 1'b0;
            z_ready <= 1'b0;
            pc_ready <= 1'b0;
        end
        else
        begin
            // 帧开始清零
            if (start && !running)
            begin
                kg_ready <= 1'b0;
                z_ready <= 1'b0;
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
                z_ready <= 1'b0;
                pc_ready <= 1'b0;
            end
        end
    end

    // —— K 就绪判定（当拍 done 也算就绪，避免同沿可见性问题）——
    wire kg_ok = kg_done | kg_ready;

    // KG 需要就绪后再发后续的 pst/poc 脉冲（更稳）
    wire pst_start = running && (cyc==6'd26) && kg_ok;  // 需要 K 与 zhat
    wire poc_start = running && (cyc==6'd26) && kg_ok;  // 需要 K 与 P_prior

    // ----------------------------------------------------------------------------
    // 1) Prior state (serial, 8 cyc)
    // ----------------------------------------------------------------------------
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
    // 2) Prior covariance (semi-parallel, 8 cyc)
    // ----------------------------------------------------------------------------
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
    // 3) Estimated output (serial, 4 cyc)
    // ----------------------------------------------------------------------------
    est_output_serial #(N,FRAC) U_Z (
                          .clk(clk), .rst_n(rst_n), .start(z_start),
                          .h00(h00), .h01(h01), .h10(h10), .h11(h11),
                          .x00(X_PRIOR00), .x10(X_PRIOR10),
                          .done(z_done),
                          .Z00(Z00), .Z10(Z10)
                      );

    // ----------------------------------------------------------------------------
    // 4) KG (semi-parallel, 17 cyc)
    // ----------------------------------------------------------------------------
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
    // 5) Q (serial, 3 cyc)
    // ----------------------------------------------------------------------------
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
    // 6) R (serial, 3 cyc)
    // ----------------------------------------------------------------------------
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
    // 7) Posterior state (serial, 5 cyc)
    // ----------------------------------------------------------------------------
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
    // 8) Posterior covariance (semi-parallel, 8 cyc)
    // ----------------------------------------------------------------------------
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
