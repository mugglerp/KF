// ============================================================================
// fxp_types.vh : Fixed-point configuration (Verilog-2001)
// ----------------------------------------------------------------------------
// - N: total word width (signed two's complement)
// - FRAC: number of fractional bits (decimal point between bit FRAC and FRAC-1)
//   By default FRAC = N/2 as in the paper's quantization split.
// ============================================================================
`ifndef FXP_TYPES_VH
`define FXP_TYPES_VH

`ifndef FXP_N
`define FXP_N 20
`endif

`ifndef FXP_FRAC
`define FXP_FRAC (`FXP_N/2)
`endif

`endif  // FXP_TYPES_VH
