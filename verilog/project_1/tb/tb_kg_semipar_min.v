`timescale 1ns/1ps

// Minimal testbench for kg_semipar (H=I, P=I, R=0.25I → K=0.8I)
module tb_kg_semipar_min;
    localparam integer N    = 20;
    localparam integer FRAC = 10;
    localparam integer CLK_PERIOD_NS = 10;
    localparam integer EPS_LSB = 2;

    // clk/reset/start
    reg clk, rst_n, start;
    initial
    begin
        clk=0;
        forever
            #(CLK_PERIOD_NS/2) clk=~clk;
    end
    task do_reset;
        begin
            rst_n=0;
            start=0;
            repeat(5) @(posedge clk);
            rst_n=1;
            @(posedge clk);
        end
    endtask
    task pulse_start;
        begin
            @(posedge clk);
            start<=1;
            @(posedge clk);
            start<=0;
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

    // fxp helper
    function signed [N-1:0] fxp;
        input real r;
        real maxr,minr,scaled;
        integer t;
        begin
            maxr = (((1<<(N-1))-1)) / (1.0*(1<<FRAC));
            minr = (-(1<<(N-1)))    / (1.0*(1<<FRAC));
            if (r>maxr)
                r=maxr;
            if (r<minr)
                r=minr;
            scaled = r * (1<<FRAC);
            t = $rtoi(scaled);
            fxp = t[N-1:0];
        end
    endfunction
    function integer abs_int;
        input integer v;
        begin
            abs_int=(v<0)?-v:v;
        end
    endfunction

    // DUT ports (per your kg_semipar)
    reg  signed [N-1:0] p00,p01,p10,p11;
    reg  signed [N-1:0] h00,h01,h10,h11;
    reg  signed [N-1:0] r00,r01,r10,r11;
    wire signed [N-1:0] K00,K01,K10,K11;
    wire done;

    kg_semipar #(N,FRAC) dut (
                   .clk(clk), .rst_n(rst_n), .start(start),
                   .p_prior00(p00), .p_prior01(p01), .p_prior10(p10), .p_prior11(p11),
                   .h00(h00), .h01(h01), .h10(h10), .h11(h11),
                   .r00(r00), .r01(r01), .r10(r10), .r11(r11),
                   .done(done),
                   .K00(K00), .K01(K01), .K10(K10), .K11(K11)
               );

    // GOLD
    reg signed [N-1:0] K00_g,K01_g,K10_g,K11_g;
    integer d00,d01,d10,d11;

    initial
    begin
        // Inputs: P=I, H=I, R=0.25 I
        p00=fxp(1.0);
        p01=fxp(0.0);
        p10=fxp(0.0);
        p11=fxp(1.0);
        h00=fxp(1.0);
        h01=fxp(0.0);
        h10=fxp(0.0);
        h11=fxp(1.0);
        r00=fxp(0.25);
        r01=fxp(0.0);
        r10=fxp(0.0);
        r11=fxp(0.25);

        // GOLD K = 0.8 I
        K00_g=fxp(0.8);
        K11_g=fxp(0.8);
        K01_g=fxp(0.0);
        K10_g=fxp(0.0);

        do_reset();
        pulse_start();
        wait_done();

        d00 = K00 - K00_g;
        d01 = K01 - K01_g;
        d10 = K10 - K10_g;
        d11 = K11 - K11_g;

        $display("K DUT=[%0d %0d; %0d %0d]  GOLD=[%0d %0d; %0d %0d]  d=[%0d %0d %0d %0d] LSB",
                 K00,K01,K10,K11, K00_g,K01_g,K10_g,K11_g, d00,d01,d10,d11);

        if (abs_int(d00)<=EPS_LSB && abs_int(d01)<=EPS_LSB &&
                abs_int(d10)<=EPS_LSB && abs_int(d11)<=EPS_LSB)
            $display("kg_semipar PASS");
        else
            $display("kg_semipar FAIL");

        $finish;
    end
endmodule
