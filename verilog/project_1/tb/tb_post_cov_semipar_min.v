`timescale 1ns/1ps
`include "fxp_types.vh"

module tb_post_cov_semipar_min;
    localparam integer N=`FXP_N, FRAC=`FXP_FRAC, S=(1<<FRAC);
    reg clk,rst_n,start;
    reg signed [N-1:0] k00,k01,k10,k11;
    reg signed [N-1:0] h00,h01,h10,h11;
    reg signed [N-1:0] p00,p01,p10,p11;
    wire done;
    wire signed [N-1:0] P00,P01,P10,P11;

    post_cov_semipar #(N,FRAC) dut(
                         .clk(clk),.rst_n(rst_n),.start(start),
                         .k00(k00),.k01(k01),.k10(k10),.k11(k11),
                         .h00(h00),.h01(h01),.h10(h10),.h11(h11),
                         .p_prior00(p00),.p_prior01(p01),.p_prior10(p10),.p_prior11(p11),
                         .done(done),
                         .P_post00(P00),.P_post01(P01),.P_post10(P10),.P_post11(P11)
                     );

    initial
    begin
        clk=0;
        forever
            #5 clk=~clk;
    end
    task pulse_start;
        begin
            @(posedge clk);
            start<=1;
            @(posedge clk);
            start<=0;
        end
    endtask
    integer cyc;
    always @(posedge clk or negedge rst_n) if(!rst_n)
            cyc<=0;
        else if(start)
            cyc<=1;
        else if(cyc!=0 && !done)
            cyc<=cyc+1;

    initial
    begin
        rst_n=0;
        start=0;
        k00=S>>>1;
        k11=S>>>1;
        k01=0;
        k10=0;   // K = 0.5 I
        h00=S;
        h11=S;
        h01=0;
        h10=0;           // H = I
        p00=S;
        p11=S;
        p01=0;
        p10=0;           // P_prior = I
        repeat(2) @(posedge clk);
        rst_n=1;
        @(posedge clk);
        pulse_start();
        wait(done);
        $display("P_post diag=[%0d %0d]  EXP=%0d", P00,P11, S>>>1);
        if (P00!==(S>>>1) || P11!==(S>>>1) || P01!==0 || P10!==0)
            $display("post_cov_semipar FAIL (value)");
        else if (cyc!=8)
            $display("post_cov_semipar FAIL: done at %0d (exp 8)",cyc);
        else
            $display("post_cov_semipar PASS");
        $finish;
    end
endmodule
