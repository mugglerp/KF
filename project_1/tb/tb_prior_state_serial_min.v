`timescale 1ns/1ps

// tb_prior_state_serial_min.v
module tb_prior_state_serial_min;
    localparam integer N    = 20;
    localparam integer FRAC = 10;
    localparam integer S    = (1<<FRAC);
    localparam integer CLK_PERIOD_NS = 10;
    localparam integer EPS = 1;   // 允许误差 ±1 LSB

    // --------- 时钟/复位/启动 ----------
    reg clk, rst_n, start;
    initial
    begin
        clk=1'b0;
        forever
            #(CLK_PERIOD_NS/2) clk=~clk;
    end
    task pulse_start;
        begin
            @(posedge clk);
            start<=1'b1;
            @(posedge clk);
            start<=1'b0;
        end
    endtask
    task do_reset;
        begin
            rst_n=1'b0;
            start=1'b0;
            repeat(5) @(posedge clk);
            rst_n=1'b1;
            @(posedge clk);
        end
    endtask
    task wait_done;
        begin
            @(posedge clk);
            while(!done)
                @(posedge clk);
            @(posedge clk);
        end
    endtask

    // --------- DUT 端口 ----------
    reg  signed [N-1:0] a00,a01,a10,a11;
    reg  signed [N-1:0] b00,b01,b10,b11;
    reg  signed [N-1:0] x00_prev,x10_prev;
    reg  signed [N-1:0] u00,u10;
    wire done;
    wire signed [N-1:0] X_PRIOR00, X_PRIOR10;

    prior_state_serial #(N,FRAC) dut (
                           .clk(clk), .rst_n(rst_n), .start(start),
                           .x00(x00_prev), .x10(x10_prev),
                           .a00(a00), .a01(a01), .a10(a10), .a11(a11),
                           .u00(u00), .u10(u10),
                           .b00(b00), .b01(b01), .b10(b10), .b11(b11),
                           .done(done),
                           .X_PRIOR00(X_PRIOR00), .X_PRIOR10(X_PRIOR10)
                       );

    // --------- 固定点小工具（乘法取对齐截断） ----------
    function signed [N-1:0] mul_trunc;
        input signed [N-1:0] p, q;
        reg   signed [2*N-1:0] prod;
        begin
            prod = $signed(p) * $signed(q);
            mul_trunc = prod[FRAC+N-1:FRAC]; // 向零截断的算术右移 FRAC
        end
    endfunction

    function integer abs_i;
        input integer v;
        begin
            abs_i=(v<0)?-v:v;
        end
    endfunction

    // --------- 用简单常量：A=[[1,1],[0,1]]; B=[[1/2,0],[1,0]]; x_prev=[1.0;-0.5]; u=[0.25; 0] ----------
    integer d0,d1;
    reg signed [N-1:0] xp0_g, xp1_g;

    initial
    begin
        // 参数
        a00 = 1*S;
        a01 = 1*S;
        a10 = 0;
        a11 = 1*S;
        b00 = (S>>1);
        b01 = 0;
        b10 = 1*S;
        b11 = 0;

        x00_prev = 1*S;         // 1.0
        x10_prev = - (S>>1);    // -0.5
        u00      = (S>>2);      // 0.25
        u10      = 0;

        do_reset();
        pulse_start();
        wait_done();

        // 期望：xp0 = a00*x0 + a01*x1 + b00*u0 + b01*u1
        xp0_g =  mul_trunc(a00,x00_prev)
              + mul_trunc(a01,x10_prev)
              + mul_trunc(b00,u00)
              + mul_trunc(b01,u10);

        // xp1 = a10*x0 + a11*x1 + b10*u0 + b11*u1
        xp1_g =  mul_trunc(a10,x00_prev)
              + mul_trunc(a11,x10_prev)
              + mul_trunc(b10,u00)
              + mul_trunc(b11,u10);

        d0 = X_PRIOR00 - xp0_g;
        d1 = X_PRIOR10 - xp1_g;

        $display("PRIOR  DUT=[%0d %0d]  GOLD=[%0d %0d]  d=[%0d %0d] LSB",
                 X_PRIOR00, X_PRIOR10, xp0_g, xp1_g, d0, d1);

        if (abs_i(d0)>EPS || abs_i(d1)>EPS)
            $fatal(1, "prior_state_serial mismatch");
        else
            $display("prior_state_serial PASS");
        $finish;
    end
endmodule
