`timescale 1ns/1ps
// 纯 Verilog-2001，自带 fxp()，两帧测量/系统常量内嵌
module tb_top_kf_34c;

    // ===== Parameters =====
    localparam integer N    = 20;   // 20/10
    localparam integer FRAC = 10;
    localparam integer CLK_PERIOD_NS = 10;  // 100MHz
    localparam integer EPS_LSB = 2;         // 允许 ±2 LSB

    // ===== Clk/Reset/Start =====
    reg clk, rst_n, start;
    initial
    begin
        clk = 1'b0;
        forever
            #(CLK_PERIOD_NS/2) clk = ~clk;
    end
    task do_reset;
        begin
            rst_n = 1'b0;
            start = 1'b0;
            repeat (5) @(posedge clk);
            rst_n = 1'b1;
            @(posedge clk);
        end
    endtask
    task pulse_start;
        begin
            @(posedge clk);
            start <= 1'b1;
            @(posedge clk);
            start <= 1'b0;
        end
    endtask
    task wait_done;
        begin
            @(posedge clk);
            while(!done)
                @(posedge clk);
            @(posedge clk); // commit 拍
        end
    endtask

    // ===== DUT ports =====
    reg  signed [N-1:0] a00, a01, a10, a11;
    reg  signed [N-1:0] b00, b01, b10, b11;
    reg  signed [N-1:0] h00, h01, h10, h11;
    reg  signed [N-1:0] x00_prev, x10_prev;
    reg  signed [N-1:0] u00, u10;
    reg  signed [N-1:0] z00_meas, z10_meas;
    reg  signed [N-1:0] beta;
    reg  signed [N-1:0] sigma2_00, sigma2_01, sigma2_10, sigma2_11;

    wire done;
    wire signed [N-1:0] X00_post, X10_post;

    top_kf #(.N(N), .FRAC(FRAC)) dut (
               .clk(clk), .rst_n(rst_n), .start(start),
               .a00(a00), .a01(a01), .a10(a10), .a11(a11),
               .b00(b00), .b01(b01), .b10(b10), .b11(b11),
               .h00(h00), .h01(h01), .h10(h10), .h11(h11),
               .x00_prev(x00_prev), .x10_prev(x10_prev),
               .u00(u00), .u10(u10),
               .z00_meas(z00_meas), .z10_meas(z10_meas),
               .beta(beta),
               .sigma2_00(sigma2_00), .sigma2_01(sigma2_01),
               .sigma2_10(sigma2_10), .sigma2_11(sigma2_11),
               .done(done),
               .X00_post(X00_post), .X10_post(X10_post)
           );

    // ===== 小工具 =====
    function integer abs_int;
        input integer v;
        begin
            abs_int=(v<0)?-v:v;
        end
    endfunction
    // real -> Q(N,FRAC) 饱和
    function signed [N-1:0] fxp;
        input real r;
        real maxr, minr, scaled;
        integer t;
        begin
            maxr = ( (1<<(N-1)) - 1 ) / (1.0 * (1<<FRAC));
            minr = ( -(1<<(N-1))     ) / (1.0 * (1<<FRAC));
            if (r > maxr)
                r = maxr;
            if (r < minr)
                r = minr;
            scaled = r * (1<<FRAC);
            t = $rtoi(scaled);
            fxp = t[N-1:0];
        end
    endfunction

    // ===== 参考 KF 所需实数（全部模块顶层声明）=====
    real A00,A01,A10,A11, B00,B01,B10,B11, H00,H01,H10,H11;
    real R11,R22, Q11,Q22;
    real x_prev0, x_prev1, P00,P01,P10,P11;
    real u0,u1;
    real z0[0:1], z1[0:1];       // 两帧测量
    real xpost0[0:1], xpost1[0:1];

    real beta;
    real sig2_d_0, sig2_d_1;
    real sig2_zh_0, sig2_zh_1;
    real dx0, dx1, q_from_dx;

    // 这些是计算中的临时量，也一并放到模块顶层：
    real xp0,xp1;                         // x_prior
    real Pp00,Pp01,Pp10,Pp11;             // P_prior
    real S00,S01,S10,S11, detS;           // S = Pp + R
    real K00,K01,K10,K11;                 // Kalman 增益
    real y0,y1, zhat0, zhat1;
    real d2, zh2, r_est_0, r_est_1;
    real IK00,IK01,IK10,IK11, n00,n01,n10,n11;
    real t00,t01,t10,t11;                  // 临时矩阵乘加
    real Q0, R0;                           // 启动值


    // ===== 34-cycle 断言（C33 必须到）=====
    integer frame_cyc;
    reg running, done_q;
    always @(posedge clk or negedge rst_n)
    begin
        if (!rst_n)
        begin
            running<=0;
            frame_cyc<=0;
            done_q<=0;
        end
        else
        begin
            done_q <= done;
            if (start && !running)
            begin
                running<=1;
                frame_cyc<=0;
            end
            else if (running)
            begin
                frame_cyc <= frame_cyc + 1;
                if (!done_q && done)
                begin
                    if (frame_cyc != 33)
                    begin
                        $display("**ERROR** done @ C%0d (expect C33)", frame_cyc);
                        $stop;
                    end
                    running <= 1'b0;
                end
                if (frame_cyc==34 && !done_q && !done)
                begin
                    $display("**ERROR** missed done@C33");
                    $stop;
                end
            end
        end
    end

    // ===== 参考 KF（常量 Q/R），两帧 =====

    // ========================= 自适应参考：两帧 =========================
    // 约定：H = I2；R_k 依式(11),(12),(13)用 z 与 zhat（由 x_prior 预测）做；
    //      Q_{k+1} 依式(10)由 Δx_post(k) 生成（在本例里 k=0 的 Q 用 Q0 预置）。
    task ref_kf_two_frames_adaptiveQR;
        integer k;
        begin
            // —— 系统矩阵（可观）——
            A00=1.0;
            A01=1.0;
            A10=0.0;
            A11=1.0;
            B00=0.5;
            B01=0.0;
            B10=1.0;
            B11=0.0;
            H00=1.0;
            H01=0.0;
            H10=0.0;
            H11=1.0;

            // 两帧测量（直接常量）
            z0[0]= 1.25;
            z1[0]= -0.80;
            z0[1]= 2.10;
            z1[1]=  0.35;

            // 自适应参数与启动
            beta = 0.90;
            Q0   = 0.01;
            R0   = 0.04;

            sig2_d_0  = R0;
            sig2_d_1  = R0;
            sig2_zh_0 = R0;
            sig2_zh_1 = R0;

            x_prev0=0.0;
            x_prev1=0.0;
            P00=1.0;
            P01=0.0;
            P10=0.0;
            P11=1.0;
            u0=0.0;
            u1=0.0;

            // ===== 帧 0 =====
            xp0 = A00*x_prev0 + A01*x_prev1 + B00*u0 + B01*u1;
            xp1 = A10*x_prev0 + A11*x_prev1 + B10*u0 + B11*u1;

            zhat0 = H00*xp0 + H01*xp1;
            zhat1 = H10*xp0 + H11*xp1;

            d2 = z0[0]*z0[0];
            sig2_d_0  = beta*sig2_d_0  + (1.0-beta)*d2;
            zh2= zhat0*zhat0;
            sig2_zh_0 = beta*sig2_zh_0 + (1.0-beta)*zh2;
            r_est_0 = sig2_d_0 - sig2_zh_0;
            if (r_est_0<0.0)
                r_est_0 = -r_est_0;

            d2 = z1[0]*z1[0];
            sig2_d_1  = beta*sig2_d_1  + (1.0-beta)*d2;
            zh2= zhat1*zhat1;
            sig2_zh_1 = beta*sig2_zh_1 + (1.0-beta)*zh2;
            r_est_1 = sig2_d_1 - sig2_zh_1;
            if (r_est_1<0.0)
                r_est_1 = -r_est_1;

            R11 = r_est_0;
            R22 = r_est_1;
            Q11 = Q0;
            Q22 = Q0;

            // P_prior = A*P*A' + Q
            Pp00 = A00*P00 + A01*P10;
            Pp01 = A00*P01 + A01*P11;
            Pp10 = A10*P00 + A11*P10;
            Pp11 = A10*P01 + A11*P11;

            t00 = Pp00*A00 + Pp01*A01;
            t01 = Pp00*A10 + Pp01*A11;
            t10 = Pp10*A00 + Pp11*A01;
            t11 = Pp10*A10 + Pp11*A11;
            Pp00 = t00 + Q11;
            Pp01 = t01 + 0.0;
            Pp10 = t10 + 0.0;
            Pp11 = t11 + Q22;

            // K = Pp/(Pp+R)
            S00=Pp00+R11;
            S01=Pp01;
            S10=Pp10;
            S11=Pp11+R22;
            detS = S00*S11 - S01*S10;
            K00 = ( Pp00*S11 - Pp01*S10)/detS;
            K01 = (-Pp00*S01 + Pp01*S00)/detS;
            K10 = ( Pp10*S11 - Pp11*S10)/detS;
            K11 = (-Pp10*S01 + Pp11*S00)/detS;

            y0 = z0[0]-xp0;
            y1 = z1[0]-xp1;
            xpost0[0] = xp0 + K00*y0 + K01*y1;
            xpost1[0] = xp1 + K10*y0 + K11*y1;

            IK00=1.0-K00;
            IK01=-K01;
            IK10=-K10;
            IK11=1.0-K11;
            n00 = IK00*Pp00 + IK01*Pp10;
            n01 = IK00*Pp01 + IK01*Pp11;
            n10 = IK10*Pp00 + IK11*Pp10;
            n11 = IK10*Pp01 + IK11*Pp11;
            P00=n00;
            P01=n01;
            P10=n10;
            P11=n11;

            dx0 = xpost0[0]-x_prev0;
            dx1 = xpost1[0]-x_prev1;
            q_from_dx = (dx0*dx0 + dx1*dx1)/2.0;
            Q11 = q_from_dx;
            Q22 = q_from_dx;

            x_prev0 = xpost0[0];
            x_prev1 = xpost1[0];

            // ===== 帧 1 =====
            xp0 = A00*x_prev0 + A01*x_prev1 + B00*u0 + B01*u1;
            xp1 = A10*x_prev0 + A11*x_prev1 + B10*u0 + B11*u1;

            zhat0 = H00*xp0 + H01*xp1;
            zhat1 = H10*xp0 + H11*xp1;

            d2 = z0[1]*z0[1];
            sig2_d_0  = beta*sig2_d_0  + (1.0-beta)*d2;
            zh2= zhat0*zhat0;
            sig2_zh_0 = beta*sig2_zh_0 + (1.0-beta)*zh2;
            r_est_0 = sig2_d_0 - sig2_zh_0;
            if (r_est_0<0.0)
                r_est_0 = -r_est_0;

            d2 = z1[1]*z1[1];
            sig2_d_1  = beta*sig2_d_1  + (1.0-beta)*d2;
            zh2= zhat1*zhat1;
            sig2_zh_1 = beta*sig2_zh_1 + (1.0-beta)*zh2;
            r_est_1 = sig2_d_1 - sig2_zh_1;
            if (r_est_1<0.0)
                r_est_1 = -r_est_1;

            R11 = r_est_0;
            R22 = r_est_1;

            Pp00 = A00*P00 + A01*P10;
            Pp01 = A00*P01 + A01*P11;
            Pp10 = A10*P00 + A11*P10;
            Pp11 = A10*P01 + A11*P11;

            t00 = Pp00*A00 + Pp01*A01;
            t01 = Pp00*A10 + Pp01*A11;
            t10 = Pp10*A00 + Pp11*A01;
            t11 = Pp10*A10 + Pp11*A11;
            Pp00 = t00 + Q11;
            Pp01 = t01 + 0.0;
            Pp10 = t10 + 0.0;
            Pp11 = t11 + Q22;

            S00=Pp00+R11;
            S01=Pp01;
            S10=Pp10;
            S11=Pp11+R22;
            detS = S00*S11 - S01*S10;
            K00 = ( Pp00*S11 - Pp01*S10)/detS;
            K01 = (-Pp00*S01 + Pp01*S00)/detS;
            K10 = ( Pp10*S11 - Pp11*S10)/detS;
            K11 = (-Pp10*S01 + Pp11*S00)/detS;

            y0 = z0[1]-xp0;
            y1 = z1[1]-xp1;
            xpost0[1] = xp0 + K00*y0 + K01*y1;
            xpost1[1] = xp1 + K10*y0 + K11*y1;
        end
    endtask



    // ===== Test flow（2 帧）=====
    integer d0, d1;
    initial
    begin
        ref_kf_two_frames_adaptiveQR();  // 先算参考（全部 real，块首声明）

        // 写入 DUT 的常量（与参考系统一致）
        a00 = fxp(1.0);
        a01 = fxp(1.0);
        a10 = fxp(0.0);
        a11 = fxp(1.0);
        b00 = fxp(0.5);
        b01 = fxp(0.0);
        b10 = fxp(1.0);
        b11 = fxp(0.0);
        h00 = fxp(1.0);
        h01 = fxp(0.0);
        h10 = fxp(0.0);
        h11 = fxp(1.0);

        x00_prev = fxp(0.0);
        x10_prev = fxp(0.0);
        u00 = fxp(0.0);
        u10 = fxp(0.0);

        // r_serial 额外输入：beta、sigma^2 的“上一帧估计”启动值
        beta      = fxp(0.90);
        sigma2_00 = fxp(0.04);
        sigma2_01 = fxp(0.00);
        sigma2_10 = fxp(0.00);
        sigma2_11 = fxp(0.04);

        do_reset();

        // Frame 0
        z00_meas = fxp(1.25);
        z10_meas = fxp(-0.80);
        pulse_start();
        wait_done();
        d0 = X00_post - fxp(xpost0[0]);
        d1 = X10_post - fxp(xpost1[0]);
        $display("F0  DUT=[%0d %0d]  GOLD=[%0d %0d]  d=[%0d %0d] LSB",
                 X00_post, X10_post, fxp(xpost0[0]), fxp(xpost1[0]), d0, d1);

        repeat(10) @(posedge clk);

        // Frame 1
        z00_meas = fxp(2.10);
        z10_meas = fxp(0.35);
        pulse_start();
        wait_done();
        d0 = X00_post - fxp(xpost0[1]);
        d1 = X10_post - fxp(xpost1[1]);
        $display("F1  DUT=[%0d %0d]  GOLD=[%0d %0d]  d=[%0d %0d] LSB",
                 X00_post, X10_post, fxp(xpost0[1]), fxp(xpost1[1]), d0, d1);

        $finish;
    end

endmodule
