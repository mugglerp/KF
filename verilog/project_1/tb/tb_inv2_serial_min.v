`timescale 1ns/1ps
`include "fxp_types.vh"

module tb_inv2_serial_min;
    localparam integer N=`FXP_N, FRAC=`FXP_FRAC, S=(1<<FRAC);
    reg clk,rst_n,start;
    reg  signed [N-1:0] a,b,c,d;
    wire done;
    wire signed [N-1:0] IA,IB,IC,ID;

    inv2_serial #(N,FRAC) dut(.clk(clk),.rst_n(rst_n),.start(start),
                              .a(a),.b(b),.c(c),.d(d), .done(done), .IA(IA),.IB(IB),.IC(IC),.ID(ID));

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
        a=2*S;
        b=0;
        c=0;
        d=4*S;
        repeat(2) @(posedge clk);
        rst_n=1;
        @(posedge clk);
        pulse_start();
        @(posedge clk);
        while(!done)
            @(posedge clk);
        $display("IA=%0d IB=%0d IC=%0d ID=%0d  EXP=[%0d 0 0 %0d]", IA,IB,IC,ID, (S>>1),(S>>2));
        if (IA!==(S>>1) || ID!==(S>>2) || IB!==0 || IC!==0)
            $display("inv2_serial FAIL");
        else if (cyc!=4)
            $display("inv2_serial FAIL: done at %0d (exp 4)",cyc);
        else
            $display("inv2_serial PASS");
        $finish;
    end
endmodule
