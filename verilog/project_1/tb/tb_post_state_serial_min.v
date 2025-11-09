`timescale 1ns/1ps
`include "fxp_types.vh"

// tb_post_state_serial_min.v
module tb_post_state_serial_min;
    localparam integer N    = `FXP_N;
    localparam integer FRAC = `FXP_FRAC;
    localparam integer S    = (1<<FRAC);
    localparam integer CLK_PERIOD_NS = 10;
    localparam integer EPS = 2;

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
    function integer abs_i;
        input integer v;
        begin
            abs_i=(v<0)?-v:v;
        end
    endfunction

    // DUT ports (严格对齐你的接口)
    reg  signed [N-1:0] x00_prior, x10_prior;
    reg  signed [N-1:0] z00_meas,  z10_meas;
    reg  signed [N-1:0] z00_hat,   z10_hat;
    reg  signed [N-1:0] k00,k01,k10,k11;
    wire done;
    wire signed [N-1:0] X00_post, X10_post;

    post_state_serial #(N,FRAC) dut (
                          .clk(clk), .rst_n(rst_n), .start(start),
                          .x00_prior(x00_prior), .x10_prior(x10_prior),
                          .z00_meas(z00_meas), .z10_meas(z10_meas),
                          .z00_hat(z00_hat),   .z10_hat(z10_hat),
                          .k00(k00), .k01(k01), .k10(k10), .k11(k11),
                          .done(done),
                          .X00_post(X00_post), .X10_post(X10_post)
                      );

    // 期望：x_prior=[1.0,-0.5], H=I ⇒ z_hat = x_prior
    // z_meas=[2.0,1.0], K=0.5*I ⇒ x_post=[1.5, 0.25]
    integer d0,d1;
    reg signed [N-1:0] x0_exp, x1_exp;

    initial
    begin
        x00_prior = 1*S;        // 1.0
        x10_prior = -(S>>1);    // -0.5
        z00_meas  = 2*S;        // 2.0
        z10_meas  = 1*S;        // 1.0
        z00_hat   = x00_prior;  // H=I
        z10_hat   = x10_prior;

        k00 = (S>>1);
        k11 = (S>>1);   // 0.5
        k01 = 0;
        k10 = 0;

        x0_exp = 1*S + (S>>1);        // 1.5*S
        x1_exp = -(S>>1) + ( (S>>1) * 3 / 2 ); // 0.25*S  (0.5*1.5 = 0.75 ; -0.5+0.75=0.25)

        do_reset();
        pulse_start();
        wait_done();

        d0 = X00_post - x0_exp;
        d1 = X10_post - x1_exp;

        $display("POST  DUT=[%0d %0d]  GOLD=[%0d %0d]  d=[%0d %0d] LSB",
                 X00_post, X10_post, x0_exp, x1_exp, d0, d1);

        if (abs_i(d0)>EPS || abs_i(d1)>EPS)
            $fatal(1,"post_state_serial mismatch");
        else
            $display("post_state_serial PASS");
        $finish;
    end
endmodule
