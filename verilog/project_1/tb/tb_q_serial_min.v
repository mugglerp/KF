`timescale 1ns/1ps
`include "fxp_types.vh"

module tb_q_serial_min;
    localparam integer N=`FXP_N, FRAC=`FXP_FRAC, S=(1<<FRAC);
    reg clk,rst_n,start;
    reg  signed [N-1:0] x00_now,x01_now,x00_prev,x01_prev;
    wire done;
    wire signed [N-1:0] Q11,Q12,Q21,Q22;

    q_serial #(N,FRAC) dut(
                 .clk(clk),.rst_n(rst_n),.start(start),
                 .x00_now(x00_now),.x01_now(x01_now),
                 .x00_prev(x00_prev),.x01_prev(x01_prev),
                 .done(done),.Q11(Q11),.Q12(Q12),.Q21(Q21),.Q22(Q22)
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

    integer expQ;
    initial
    begin
        rst_n=0;
        start=0;
        // Δx = [2.0, 1.0] => Q = ((2^2+1^2)/2)*I = 2.5*I
        x00_prev=0;
        x01_prev=0;
        x00_now = 2*S;
        x01_now = 1*S;
        repeat(3) @(posedge clk);
        rst_n=1;
        @(posedge clk);
        pulse_start();
        @(posedge clk);
        while(!done)
            @(posedge clk);
        expQ = (5*S)/2;  // 2.5*S = 2560 for FRAC=10
        $display("Q11=%0d Q22=%0d (LSB)  EXP=%0d", Q11,Q22,expQ);
        if (Q11!==expQ || Q22!==expQ)
            $display("q_serial FAIL");
        else if (cyc!=3)
            $display("q_serial FAIL: done at %0d (exp 3)",cyc);
        else
            $display("q_serial PASS");
        $finish;
    end
endmodule
