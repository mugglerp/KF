`timescale 1ns/1ps
`include "fxp_types.vh"

// Many-frame RLC closed-loop testbench (36-cycle frame)
// State/matrix per (14)(15); 100 Hz sine (10 A peak); small gaussian-like noise.
module tb_top_kf_manyframes_rlc;

    // ---------------- Params / fixed-point ----------------
    localparam integer N    = `FXP_N;       // e.g. 20
    localparam integer FRAC = `FXP_FRAC;    // e.g. 10
    localparam integer CLK_PERIOD_NS = 10;  // 100 MHz

    // === 关键：将帧长设为 36 拍 ===
    localparam integer FRAME_CYCLES  = 36;  // C0..C35，done 何时到由 DUT 决定
    localparam integer HARD_LIMIT   = FRAME_CYCLES + 4;  // 容忍+4拍（可调）
    localparam integer FRAMES        = 4000;
    localparam integer LOG_EVERY     = 100;

    // 仅用于正弦相位
    localparam real FS_HZ  = 10000.0;                // 10 kHz
    localparam real TWO_PI = 6.283185307179586;
    localparam integer S   = (1<<FRAC);

    // ---------------- Clock / reset / start ----------------
    reg clk, rst_n, start;
    initial
    begin
        clk=1'b0;
        forever
            #(CLK_PERIOD_NS/2) clk=~clk;
    end
    task do_reset;
        begin
            rst_n=1'b0;
            start=1'b0;
            repeat(5) @(posedge clk);
            rst_n=1'b1;
            @(posedge clk);
        end
    endtask
    task pulse_start;
        begin
            @(posedge clk);
            start<=1'b1;
            @(posedge clk);
            start<=1'b0;
        end
    endtask
    task wait_done;
        integer local_cyc;
        begin
            local_cyc = 0;
            @(posedge clk);
            // 等待 done = 1，同时记录实际用时
            while(!done)
            begin
                local_cyc = local_cyc + 1;
                if (local_cyc == HARD_LIMIT)
                begin
                    $display("**WARN** frame took >= %0d cycles before done asserted.", HARD_LIMIT);
                end
                @(posedge clk);
            end
            // 追加一个提交拍
            @(posedge clk);
        end
    endtask

    // ---------------- DUT ports ----------------
    reg  signed [N-1:0] a00,a01,a10,a11;
    reg  signed [N-1:0] b00,b01,b10,b11;
    reg  signed [N-1:0] h00,h01,h10,h11;
    reg  signed [N-1:0] x00_prev,x10_prev;
    reg  signed [N-1:0] u00,u10;
    reg  signed [N-1:0] z00_meas,z10_meas;
    reg  signed [N-1:0] beta;
    reg  signed [N-1:0] sigma2_00,sigma2_01,sigma2_10,sigma2_11;

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

    // ---------------- helpers ----------------
    function signed [N-1:0] fxp; // real -> Q(N,FRAC)
        input real r;
        real maxr,minr,scaled;
        integer t;
        begin
            maxr = ( (1<<(N-1)) - 1 ) / (1.0*(1<<FRAC));
            minr = ( -(1<<(N-1))    ) / (1.0*(1<<FRAC));
            if (r>maxr)
                r=maxr;
            if (r<minr)
                r=minr;
            scaled = r * (1<<FRAC);
            t = $rtoi(scaled);
            fxp = t[N-1:0];
        end
    endfunction
    function real fxp_to_real;
        input signed [N-1:0] x;
        begin
            fxp_to_real = x / (1.0*S);
        end
    endfunction

    // Poor-man gaussian ~N(0,1) (需要一个形参符合 Verilog-2001)
    function real gauss01;
        input dummy;
        integer i;
        real s;
        begin
            s=0.0;
            for(i=0;i<12;i=i+1)
                s = s + (($random & 32'hFFFF)/65536.0);
            gauss01 = s - 6.0;
        end
    endfunction

    // ---------------- RLC model (discrete) per (14)(15) ----------------
    // A = [[ 0.5004 -0.0005], [-1.0005 0]]
    // B = [[-0.5004 0], [1.0005 0]]
    // H = [[ 0.2502 -1.0003], [0.000250 -0.001]]
    real A00r,A01r,A10r,A11r, B00r,B10r, H00r,H01r,H10r,H11r;
    real il, vc, il_next, vc_next, vR2, iR2;
    real it_amp, it_freq, it_val, noise_sigma_it, noise_sigma_z;
    real it_noise, z_noise0, z_noise1, theta;

    // 仅作统计（非强校验）
    integer frame_cyc;
    reg running, done_q;
    always @(posedge clk or negedge rst_n)
    begin
        if(!rst_n)
        begin
            running<=0;
            done_q<=0;
            frame_cyc<=0;
        end
        else
        begin
            done_q<=done;
            if(start && !running)
            begin
                running<=1;
                frame_cyc<=0;
            end
            else if(running)
            begin
                frame_cyc<=frame_cyc+1;
                if(!done_q && done)
                begin
                    $display("Info: frame done @ %0d cycles", frame_cyc);
                    running<=0;
                end
            end
        end
    end

    // -------------------- Main stimulus (many frames) --------------------
    integer k, d0, d1;
    integer fd;
    real il_hat, vc_hat, vR2_hat, iR2_hat;

    initial
    begin : MAIN
        // matrices
        A00r= 0.5004;
        A01r=-0.0005;
        A10r=-1.0005;
        A11r=0.0;
        B00r=-0.5004;
        B10r= 1.0005;
        H00r= 0.2502;
        H01r=-1.0003;
        H10r=0.000250;
        H11r=-0.001;

        // input & noise
        it_amp=10.0;
        it_freq=100.0;
        noise_sigma_it=1e-3;
        noise_sigma_z=1e-3;

        // map to DUT
        a00=fxp(A00r);
        a01=fxp(A01r);
        a10=fxp(A10r);
        a11=fxp(A11r);
        b00=fxp(B00r);
        b01=fxp(0.0);
        b10=fxp(B10r);
        b11=fxp(0.0);
        h00=fxp(H00r);
        h01=fxp(H01r);
        h10=fxp(H10r);
        h11=fxp(H11r);

        // seeds
        x00_prev=fxp(0.0);
        x10_prev=fxp(0.0);
        u00=fxp(0.0);
        u10=fxp(0.0);
        beta=fxp(0.90);
        sigma2_00=fxp(0.04);
        sigma2_01=fxp(0.0);
        sigma2_10=fxp(0.0);
        sigma2_11=fxp(0.04);

        il=0.0;
        vc=0.0;

        do_reset();

        // 可选：记录到 xsim 工作目录
        fd = $fopen("rlc_trace.csv","w");
        $fdisplay(fd,"k,il,il_hat,vc,vc_hat,vR2,vR2_hat,iR2,iR2_hat");

        for (k=0; k<FRAMES; k=k+1)
        begin
            // 生成输入
            theta    = TWO_PI*(it_freq/FS_HZ)*k;
            it_noise = noise_sigma_it * gauss01(1'b0);
            it_val   = it_amp * $sin(theta) + it_noise;

            // x_k = A x_{k-1} + B [it;0]
            il_next = A00r*il + A01r*vc + B00r*it_val;
            vc_next = A10r*il + A11r*vc + B10r*it_val;
            il = il_next;
            vc = vc_next;

            // KF 输入
            u00 = fxp(it_val);   // ★关键修复点
            u10 = fxp(0.0);

            // z_k = H x_k + noise
            z_noise0 = noise_sigma_z * gauss01(1'b0);
            z_noise1 = noise_sigma_z * gauss01(1'b0);
            vR2 = H00r*il + H01r*vc + z_noise0;
            iR2 = H10r*il + H11r*vc + z_noise1;

            // 驱动一帧
            z00_meas = fxp(vR2);
            z10_meas = fxp(iR2);
            pulse_start();
            wait_done();   // <- 36-cycle 帧：严格等 done 再继续

            // 回写 prev（闭环）
            x00_prev = X00_post;
            x10_prev = X10_post;

            // 估计输出（不访问 DUT 内部）
            il_hat  = fxp_to_real(X00_post);
            vc_hat  = fxp_to_real(X10_post);
            vR2_hat = H00r*il_hat + H01r*vc_hat;
            iR2_hat = H10r*il_hat + H11r*vc_hat;

            // 记录 / 打印
            if ((k%LOG_EVERY)==0)
            begin
                d0 = X00_post - fxp(il);
                d1 = X10_post - fxp(vc);
                $display("k=%0d  it=%f  z=[%f %f]  x_ref=[%f %f]  d=[%0d %0d]LSB",
                         k, it_val, vR2, iR2, il, vc, d0, d1);
            end
            $fdisplay(fd,"%0d,%f,%f,%f,%f,%f,%f,%f,%f",
                      k, il, il_hat, vc, vc_hat, vR2, vR2_hat, iR2, iR2_hat);
        end

        $display("DONE %0d frames (36-cycle handshake).", FRAMES);
        $fclose(fd);
        $finish;
    end

endmodule
