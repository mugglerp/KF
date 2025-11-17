`timescale 1ns/1ps

module tb_top_kf_34c_basic;
    localparam integer N=20, FRAC=10;
    localparam integer S = (1<<FRAC);

    reg clk,rst_n,start;
    initial
    begin
        clk=0;
        forever
            #5 clk=~clk;
    end

    // matrices: A=I, B=0, H=I
    reg  signed [N-1:0] a00,a01,a10,a11;
    reg  signed [N-1:0] b00,b01,b10,b11;
    reg  signed [N-1:0] h00,h01,h10,h11;
    reg  signed [N-1:0] x00_prev,x10_prev,u00,u10;
    reg  signed [N-1:0] z00_meas,z10_meas;
    reg  signed [N-1:0] beta, sigma2_00,sigma2_01,sigma2_10,sigma2_11;

    wire done;
    wire signed [N-1:0] X00_post,X10_post;

    top_kf #(N,FRAC) dut (
                   .clk(clk),.rst_n(rst_n),.start(start),
                   .a00(a00),.a01(a01),.a10(a10),.a11(a11),
                   .b00(b00),.b01(b01),.b10(b10),.b11(b11),
                   .h00(h00),.h01(h01),.h10(h10),.h11(h11),
                   .x00_prev(x00_prev),.x10_prev(x10_prev),
                   .u00(u00),.u10(u10),
                   .z00_meas(z00_meas),.z10_meas(z10_meas),
                   .beta(beta),
                   .sigma2_00(sigma2_00),.sigma2_01(sigma2_01),
                   .sigma2_10(sigma2_10),.sigma2_11(sigma2_11),
                   .done(done),
                   .X00_post(X00_post),.X10_post(X10_post)
               );

    task pulse_start;
        begin
            @(posedge clk);
            start<=1'b1;
            @(posedge clk);
            start<=1'b0;
        end
    endtask

    integer cyc;
    always @(posedge clk or negedge rst_n)
    begin
        if(!rst_n)
            cyc<=-1;
        else if(start)
            cyc<=0;
        else if(cyc>=0 && !done)
            cyc<=cyc+1;
    end
    integer f;
    initial
    begin
        // reset & constants
        rst_n=0;
        start=0;
        a00=S;
        a11=S;
        a01=0;
        a10=0;
        b00=0;
        b01=0;
        b10=0;
        b11=0;
        h00=S;
        h11=S;
        h01=0;
        h10=0;
        x00_prev=0;
        x10_prev=0;
        u00=0;
        u10=0;
        z00_meas=0;
        z10_meas=0;
        beta = (S*9)/10;   // 0.9
        sigma2_00 = S/16;
        sigma2_11 = S/16;
        sigma2_01=0;
        sigma2_10=0;

        repeat(3) @(posedge clk);
        rst_n=1;
        @(posedge clk);

        // 调用 2 帧，检查每帧 34 拍

        for(f=0; f<2; f=f+1)
        begin
            pulse_start();
            @(posedge clk);
            while(!done)
                @(posedge clk);
            if(cyc!==33)
                $display("**FAIL** frame %0d done at cyc=%0d (exp 33)", f, cyc);
            else
                $display("PASS frame %0d : done at C33 (34-cycle)", f);
            @(posedge clk);  // 甯ч棿闅斾竴鎷?
        end
        $finish;
    end
endmodule
