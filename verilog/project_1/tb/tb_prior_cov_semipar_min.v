`timescale 1ns/1ps

module tb_prior_cov_semipar_min;
    localparam integer N=20, FRAC=10, S=(1<<FRAC);
    reg clk,rst_n,start;
    reg  signed [N-1:0] a00,a01,a10,a11;
    reg  signed [N-1:0] p00,p01,p10,p11;
    reg  signed [N-1:0] q00,q01,q10,q11;
    wire done;
    wire signed [N-1:0] P00,P01,P10,P11;

    prior_cov_semipar #(N,FRAC) dut(
                          .clk(clk),.rst_n(rst_n),.start(start),
                          .a00(a00),.a01(a01),.a10(a10),.a11(a11),
                          .p00(p00),.p01(p01),.p10(p10),.p11(p11),
                          .q00(q00),.q01(q01),.q10(q10),.q11(q11),
                          .done(done),
                          .P_PRIOR00(P00),.P_PRIOR01(P01),.P_PRIOR10(P10),.P_PRIOR11(P11)
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
        a00=S;
        a11=S;
        a01=0;
        a10=0;           // A = I
        p00=S;
        p11=S;
        p01=0;
        p10=0;           // P_prev = I
        q00=S>>2;
        q11=S>>2;
        q01=0;
        q10=0;     // Q = 0.25 I
        repeat(2) @(posedge clk);
        rst_n=1;
        @(posedge clk);
        pulse_start();
        wait(done);
        $display("P_prior diag=[%0d %0d]  EXP=%0d", P00,P11, S + (S>>2));
        if (P00!==(S+(S>>2)) || P11!==(S+(S>>2)) || P01!==0 || P10!==0)
            $display("prior_cov_semipar FAIL");
        else if (cyc!=8)
            $display("prior_cov_semipar FAIL: done at %0d (exp 8)",cyc);
        else
            $display("prior_cov_semipar PASS");
        $finish;
    end
endmodule
