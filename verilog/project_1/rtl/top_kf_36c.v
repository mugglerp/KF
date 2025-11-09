`include "fxp_types.vh"

module top_kf
    #( parameter integer N=`FXP_N, parameter integer FRAC=`FXP_FRAC )
     (
         input  wire                     clk,
         input  wire                     rst_n,
         input  wire                     start,

         input  wire signed [N-1:0]      a00,a01,a10,a11,
         input  wire signed [N-1:0]      b00,b01,b10,b11,
         input  wire signed [N-1:0]      h00,h01,h10,h11,
         input  wire signed [N-1:0]      x00_prev,x10_prev,
         input  wire signed [N-1:0]      u00,u10,
         input  wire signed [N-1:0]      z00_meas,z10_meas,

         input  wire signed [N-1:0]      beta,
         input  wire signed [N-1:0]      sigma2_00,sigma2_01,sigma2_10,sigma2_11,

         output reg                      done,           // assert at C35
         output wire signed [N-1:0]      X00_post, X10_post
     );

    // 36-cycle frame: load 35, run down to 0; done at cyc==35 (cnt==0)
    localparam integer FRAME_CYCLES = 36;
    localparam [5:0]   LOAD_VAL     = FRAME_CYCLES-1; // 35

    reg       running;
    reg [5:0] cnt;                      // 35..0
    wire [5:0] cyc = LOAD_VAL - cnt;    // 0..35 当且仅当 running=1 有效

    always @(posedge clk or negedge rst_n)
    begin
        if (!rst_n)
        begin
            running <= 1'b0;
            cnt     <= 6'd0;
            done    <= 1'b0;
        end
        else
        begin
            done <= 1'b0; // 默认 1 拍脉冲

            if (start && !running)
            begin
                running <= 1'b1;
                cnt     <= LOAD_VAL;          // 装载 35，下一拍就是 C0
            end
            else if (running)
            begin
                if (cnt == 6'd0)
                begin
                    // 当拍是 C35：对外口径拉高 done；同时允许本拍做 commit
                    done    <= 1'b1;
                    running <= 1'b0;            // 下一拍退出运行
                    // cnt 保持 0（无所谓），或可选 cnt <= 0;
                end
                else
                begin
                    cnt <= cnt - 6'd1;
                end
            end
        end
    end

    // 提交时刻保持与口径一致：在 C35 当拍
    wire commit_c35 = running && (cnt == 6'd0);

    // ---------------- previous-frame registers ----------------
    localparam signed [N-1:0] ONE = $signed(1) <<< FRAC;

    reg signed [N-1:0] R11_prev,R12_prev,R21_prev,R22_prev;
    reg signed [N-1:0] Q11_prev,Q12_prev,Q21_prev,Q22_prev;

    reg signed [N-1:0] P_post00_prev,P_post01_prev,P_post10_prev,P_post11_prev;

    wire signed [N-1:0] P_post00_w,P_post01_w,P_post10_w,P_post11_w;

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
        else if(commit_c35)
        begin
            // R/Q 从 r_serial/q_serial 的 next 提交
            R11_prev<=R11_next;
            R12_prev<=R12_next;
            R21_prev<=R21_next;
            R22_prev<=R22_next;
            Q11_prev<=Q11_next;
            Q12_prev<=Q12_next;
            Q21_prev<=Q21_next;
            Q22_prev<=Q22_next;
            // P_post -> P_prev
            P_post00_prev<=P_post00_w;
            P_post01_prev<=P_post01_w;
            P_post10_prev<=P_post10_w;
            P_post11_prev<=P_post11_w;
        end
    end

    // ---------------- start pulses (36c) ----------------
    wire ps_start  = running && (cyc==6'd0);
    wire pc_start  = running && (cyc==6'd0);   // prior_cov_semipar 9-cycle (0..8)
    wire z_start   = running && (cyc==6'd8);   // est_output 4-cycle (8..11)
    wire kg_start  = running && (cyc==6'd10);  // kg_semipar 17-cycle (10..26)
    wire q_start   = running && (cyc==6'd10);  // q_serial 3-cycle
    wire r_start   = running && (cyc==6'd12);  // r_serial 3-cycle
    wire pst_start = running && (cyc==6'd27);  // post_state 5-cycle (27..31)
    wire poc_start = running && (cyc==6'd27);  // post_cov_semipar 9-cycle (27..35)

    // ---------------- modules ----------------
    // 1) Prior state (serial, 8 cyc)
    wire ps_done;
    wire signed [N-1:0] X_PRIOR00, X_PRIOR10;
    prior_state_serial #(N,FRAC) U_PS (
                           .clk(clk),.rst_n(rst_n),.start(ps_start),
                           .x00(x00_prev),.x10(x10_prev),
                           .a00(a00),.a01(a01),.a10(a10),.a11(a11),
                           .u00(u00),.u10(u10),
                           .b00(b00),.b01(b01),.b10(b10),.b11(b11),
                           .done(ps_done),
                           .X_PRIOR00(X_PRIOR00),.X_PRIOR10(X_PRIOR10)
                       );

    // 2) Prior covariance (semi-parallel, **9 cyc**)
    wire pc_done;
    wire signed [N-1:0] P_PRIOR00, P_PRIOR01, P_PRIOR10, P_PRIOR11;
    prior_cov_semipar #(N,FRAC) U_PC (
                          .clk(clk),.rst_n(rst_n),.start(pc_start),
                          .a00(a00),.a01(a01),.a10(a10),.a11(a11),
                          .p00(P_post00_prev),.p01(P_post01_prev),.p10(P_post10_prev),.p11(P_post11_prev),
                          .q00(Q11_prev),.q01(Q12_prev),.q10(Q21_prev),.q11(Q22_prev),
                          .done(pc_done),
                          .P_PRIOR00(P_PRIOR00),.P_PRIOR01(P_PRIOR01),
                          .P_PRIOR10(P_PRIOR10),.P_PRIOR11(P_PRIOR11)
                      );

    // 3) Estimated output (serial, 4 cyc)
    wire z_done;
    wire signed [N-1:0] Z00, Z10;
    est_output_serial #(N,FRAC) U_Z (
                          .clk(clk),.rst_n(rst_n),.start(z_start),
                          .h00(h00),.h01(h01),.h10(h10),.h11(h11),
                          .x00(X_PRIOR00),.x10(X_PRIOR10),
                          .done(z_done), .Z00(Z00),.Z10(Z10)
                      );

    // 4) KG (semi-parallel, 17 cyc) uses P_prior and R_prev
    wire kg_done;
    wire signed [N-1:0] K00,K01,K10,K11;
    kg_semipar #(N,FRAC) U_KG (
                   .clk(clk),.rst_n(rst_n),.start(kg_start),
                   .p_prior00(P_PRIOR00),.p_prior01(P_PRIOR01),
                   .p_prior10(P_PRIOR10),.p_prior11(P_PRIOR11),
                   .h00(h00),.h01(h01),.h10(h10),.h11(h11),
                   .r00(R11_prev),.r01(R12_prev),.r10(R21_prev),.r11(R22_prev),
                   .done(kg_done), .K00(K00),.K01(K01),.K10(K10),.K11(K11)
               );

    // 5) Q (serial, 3 cyc)
    wire q_done;
    wire signed [N-1:0] Q11_w,Q12_w,Q21_w,Q22_w;
    q_serial #(N,FRAC) U_Q (
                 .clk(clk),.rst_n(rst_n),.start(q_start),
                 .x00_now(X_PRIOR00),.x01_now(X_PRIOR10),
                 .x00_prev(x00_prev),.x01_prev(x10_prev),
                 .done(q_done),
                 .Q11(Q11_w),.Q12(Q12_w),.Q21(Q21_w),.Q22(Q22_w)
             );
    reg signed [N-1:0] Q11_next,Q12_next,Q21_next,Q22_next;
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

    // 6) R (serial, 3 cyc)
    wire r_done;
    wire signed [N-1:0] R11_w,R12_w,R21_w,R22_w;
    r_serial #(N,FRAC) U_R (
                 .clk(clk),.rst_n(rst_n),.start(r_start),
                 .beta(beta),
                 .sigma2_00(sigma2_00),.sigma2_01(sigma2_01),
                 .sigma2_10(sigma2_10),.sigma2_11(sigma2_11),
                 .z00(z00_meas),.z10(z10_meas),
                 .zhat00(Z00),.zhat10(Z10),
                 .done(r_done),
                 .R11(R11_w),.R12(R12_w),.R21(R21_w),.R22(R22_w)
             );
    reg signed [N-1:0] R11_next,R12_next,R21_next,R22_next;
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

    // 7) Posterior state (serial, 5 cyc)
    wire pst_done;
    post_state_serial #(N,FRAC) U_PST (
                          .clk(clk),.rst_n(rst_n),.start(pst_start),
                          .x00_prior(X_PRIOR00),.x10_prior(X_PRIOR10),
                          .z00_meas(z00_meas),.z10_meas(z10_meas),
                          .z00_hat(Z00),.z10_hat(Z10),
                          .k00(K00),.k01(K01),.k10(K10),.k11(K11),
                          .done(pst_done),
                          .X00_post(X00_post),.X10_post(X10_post)
                      );

    // 8) Posterior covariance (semi-parallel, **9 cyc**)
    wire poc_done;
    post_cov_semipar #(N,FRAC) U_POC (
                         .clk(clk),.rst_n(rst_n),.start(poc_start),
                         .k00(K00),.k01(K01),.k10(K10),.k11(K11),
                         .h00(h00),.h01(h01),.h10(h10),.h11(h11),
                         .p_prior00(P_PRIOR00),.p_prior01(P_PRIOR01),
                         .p_prior10(P_PRIOR10),.p_prior11(P_PRIOR11),
                         .done(poc_done),
                         .P_post00(P_post00_w),.P_post01(P_post01_w),
                         .P_post10(P_post10_w),.P_post11(P_post11_w)
                     );

endmodule
