`timescale 1ns/1ps

module tb_est_output_serial_min;
  localparam integer N=20, FRAC=10, S=(1<<FRAC);
  reg clk,rst_n,start;
  reg  signed [N-1:0] h00,h01,h10,h11;
  reg  signed [N-1:0] x00,x10;
  wire done;
  wire signed [N-1:0] Z00,Z10;

  est_output_serial #(N,FRAC) dut(
    .clk(clk),.rst_n(rst_n),.start(start),
    .h00(h00),.h01(h01),.h10(h10),.h11(h11),
    .x00(x00),.x10(x10), .done(done), .Z00(Z00),.Z10(Z10)
  );

  initial begin clk=0; forever #5 clk=~clk; end
  task pulse_start; begin @(posedge clk); start<=1; @(posedge clk); start<=0; end endtask
  integer cyc; always @(posedge clk or negedge rst_n) if(!rst_n) cyc<=0; else if(start) cyc<=1; else if(cyc!=0 && !done) cyc<=cyc+1;

  initial begin
    rst_n=0; start=0;
    h00=S; h11=S; h01=0; h10=0;      // H = I
    x00=(3*S)>>1; x10=-(3*S)>>2;     // [1.5, -0.75]
    repeat(2) @(posedge clk); rst_n=1; @(posedge clk);
    pulse_start();
    @(posedge clk); while(!done) @(posedge clk);
    $display("Z=[%0d %0d] EXP=[%0d %0d]", Z00,Z10,x00,x10);
    if (Z00!==x00 || Z10!==x10) $display("est_output_serial FAIL");
    else if (cyc!=4) $display("est_output_serial FAIL: done at %0d (exp 4)",cyc);
    else $display("est_output_serial PASS");
    $finish;
  end
endmodule
