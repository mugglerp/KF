`timescale 1ns/1ps
`include "fxp_types.vh"

module tb_top_kf_34c;
  // -------- Clock & Reset --------
  reg clk = 0;
  always #5 clk = ~clk; // 100MHz
  reg rst_n = 0;
  initial begin
    rst_n = 0;
    repeat(5) @(posedge clk);
    rst_n = 1;
  end

  // -------- Params / DUT IO --------
  localparam integer N    = `FXP_N;
  localparam integer FRAC = `FXP_FRAC;
  localparam signed [N-1:0] ONE = $signed(1) <<< FRAC;

  reg start;
  // System matrices (先给出接近单位的稳定配置)
  reg  signed [N-1:0] a00,a01,a10,a11;
  reg  signed [N-1:0] b00,b01,b10,b11;
  reg  signed [N-1:0] h00,h01,h10,h11;

  reg  signed [N-1:0] x00_prev, x10_prev;
  reg  signed [N-1:0] u00, u10;

  // measurements
  reg  signed [N-1:0] z00_meas, z10_meas;

  // r_serial extras
  reg  signed [N-1:0] beta;
  reg  signed [N-1:0] sigma2_00, sigma2_01;
  reg  signed [N-1:0] sigma2_10, sigma2_11;

  wire done;
  wire signed [N-1:0] X00_post, X10_post;

  // -------- Cycle counter for checking C33 --------
  integer cyc;
  reg running;
  always @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
      cyc <= 0; running <= 0;
    end else begin
      if(start && !running) begin
        running <= 1; cyc <= 0;
      end else if(running) begin
        cyc <= cyc + 1;
        if(done) running <= 0;
      end
    end
  end

  // -------- Instantiate DUT --------
  top_kf #(.N(N), .FRAC(FRAC)) DUT (
    .clk(clk), .rst_n(rst_n), .start(start),

    .a00(a00), .a01(a01), .a10(a10), .a11(a11),
    .b00(b00), .b01(b01), .b10(b10), .b11(b11),
    .h00(h00), .h01(h01), .h10(h10), .h11(h11),

    .x00_prev(x00_prev), .x10_prev(x10_prev),
    .u00(u00), .u10(u10),

    .z00_meas(z00_meas), .z10_meas(z10_meas),

    .beta(beta),
    .sigma2_00(sigma2_00), .sigma2_01(sigma2_01),
    .sigma2_10(sigma2_10), .sigma2_11(sigma2_11),

    .done(done),
    .X00_post(X00_post), .X10_post(X10_post)
  );

  // -------- Stimulus --------
  task kick_one_frame;
  begin
    @(posedge clk);
    start <= 1'b1;
    @(posedge clk);
    start <= 1'b0;
  end
  endtask

  initial begin
    // 默认数值（可替换为你论文里的配置）
    a00 = ONE;   a01 = 0;     a10 = 0;     a11 = ONE;
    b00 = ONE;   b01 = 0;     b10 = 0;     b11 = ONE;
    h00 = ONE;   h01 = 0;     h10 = 0;     h11 = ONE;

    x00_prev = 0; x10_prev = 0;
    u00 = 0; u10 = 0;

    z00_meas = 0; z10_meas = 0;

    beta = ONE;              // 可理解为 R 的调节因子
    sigma2_00 = ONE; sigma2_01 = 0;
    sigma2_10 = 0;   sigma2_11 = ONE;

    start = 0;

    // 等待复位
    @(posedge rst_n);
    @(posedge clk);

    // ===== 帧1 =====
    kick_one_frame();
    wait(done);
    $display("[T=%0t] Frame1 done at cyc=%0d, X_post=(%0d,%0d)", $time, cyc, X00_post, X10_post);
    if (cyc !== 33) begin
      $display("**ERROR** done should assert at C33, got C%0d", cyc);
      $stop;
    end

    // 提交发生在同拍 done 处（C33），下一拍就进入新一帧的 prev
    @(posedge clk);

    // ===== 帧2 =====
    // 给点小扰动，检查跨帧寄存提交
    z00_meas = ONE >>> 2; // 0.25
    z10_meas = -(ONE >>> 2);

    kick_one_frame();
    wait(done);
    $display("[T=%0t] Frame2 done at cyc=%0d, X_post=(%0d,%0d)", $time, cyc, X00_post, X10_post);
    if (cyc !== 33) begin
      $display("**ERROR** done should assert at C33 in frame2, got C%0d", cyc);
      $stop;
    end

    $display("All good. Strict 34-cycle frame controller passes smoke test.");
    $finish;
  end
endmodule
