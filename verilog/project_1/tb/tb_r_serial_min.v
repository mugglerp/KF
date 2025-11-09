`timescale 1ns/1ps
`include "fxp_types.vh"

module tb_r_serial_min;
    localparam integer N=`FXP_N, FRAC=`FXP_FRAC, S=(1<<FRAC);
    reg clk,rst_n,start;
    reg  signed [N-1:0] beta,s00,s01,s10,s11,z00,z10,zh0,zh1;
    wire done;
    wire signed [N-1:0] R11,R12,R21,R22;

    r_serial #(N,FRAC) dut(
                 .clk(clk),.rst_n(rst_n),.start(start),
                 .beta(beta),
                 .sigma2_00(s00),.sigma2_01(s01),
                 .sigma2_10(s10),.sigma2_11(s11),
                 .z00(z00),.z10(z10),.zhat00(zh0),.zhat10(zh1),
                 .done(done),.R11(R11),.R12(R12),.R21(R21),.R22(R22)
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

    // 计算 eq1 的期望（全部在 Q 域整数里）
    function integer eq1;
        input integer beta_q, sig2_q, z_q;
        integer one, z2;
        begin
            one = (1<<FRAC);
            z2  = (z_q*z_q) >>> FRAC; // z^2 (对齐同 Q)
            eq1 = (((one-beta_q)*z2) >>> FRAC) + ((beta_q*sig2_q) >>> FRAC);
        end
    endfunction

    integer e11,e22;
    initial
    begin
        rst_n=0;
        start=0;
        beta = (3*S)>>2;          // 0.75
        s00  = 1*S;
        s01  = 0;
        s10  = (3*S)>>1;
        s11=(1*S)>>2;
        z00  = 2*S;
        zh0  = 1*S;
        z10  = (3*S)>>1;
        zh1=(1*S)>>2;
        repeat(3) @(posedge clk);
        rst_n=1;
        @(posedge clk);

        e11 = eq1(beta,s00,z00) - eq1(beta,s01,zh0);
        if (e11<0)
            e11=-e11;
        e22 = eq1(beta,s10,z10) - eq1(beta,s11,zh1);
        if (e22<0)
            e22=-e22;

        pulse_start();
        @(posedge clk);
        while(!done)
            @(posedge clk);
        $display("R11=%0d R22=%0d  EXP=[%0d %0d] LSB", R11,R22,e11,e22);
        if (R11!==e11 || R22!==e22 || R12!==0 || R21!==0)
            $display("r_serial FAIL");
        else if (cyc!=3)
            $display("r_serial FAIL: done at %0d (exp 3)",cyc);
        else
            $display("r_serial PASS");
        $finish;
    end
endmodule
