`timescale 1ns / 1ps

////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer:
//
// Create Date:   11:13:38 04/23/2016
// Design Name:   multi_cycle_cpu
// Module Name:   F:/new_lab/7_multi_cycle_cpu/tb.v
// Project Name:  multi_cycle_cpu
// Target Device:  
// Tool versions:  
// Description: 
//
// Verilog Test Fixture created by ISE for module: multi_cycle_cpu
//
// Dependencies:
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
////////////////////////////////////////////////////////////////////////////////

module tb;

    // Inputs
    reg clk;
    reg resetn;
    reg [4:0] rf_addr;
    reg [31:0] mem_addr;

    // Outputs
    wire [31:0] rf_data;
    wire [31:0] mem_data;
    wire [31:0] IF_pc;
    wire [31:0] IF_inst;
    wire [31:0] ID_pc;
    wire [31:0] EXE_pc;
    wire [31:0] MEM_pc;
    wire [31:0] WB_pc;
    wire [31:0] display_state;

    // Wires for new debug signals from UUT
    wire [31:0] debug_dm_addr_tb;
    wire [31:0] debug_dm_wdata_tb;
    wire [31:0] debug_dm_rdata_tb;
    wire [ 3:0] debug_dm_wen_tb;
    wire [31:0] debug_rf_wdata_tb;
    wire [ 4:0] debug_rf_wdest_tb;
    wire        debug_rf_wen_tb;

    // Instantiate the Unit Under Test (UUT)
    multi_cycle_cpu uut (
        .clk(clk), 
        .resetn(resetn), 
        .rf_addr(rf_addr), 
        .mem_addr(mem_addr), 
        .rf_data(rf_data), 
        .mem_data(mem_data), 
        .IF_pc(IF_pc), 
        .IF_inst(IF_inst), 
        .ID_pc(ID_pc), 
        .EXE_pc(EXE_pc), 
        .MEM_pc(MEM_pc), 
        .WB_pc(WB_pc), 
        .display_state(display_state),

        // Connect new debug ports
        .debug_dm_addr(debug_dm_addr_tb),
        .debug_dm_wdata(debug_dm_wdata_tb),
        .debug_dm_rdata(debug_dm_rdata_tb),
        .debug_dm_wen(debug_dm_wen_tb),
        .debug_rf_wdata(debug_rf_wdata_tb),
        .debug_rf_wdest(debug_rf_wdest_tb),
        .debug_rf_wen(debug_rf_wen_tb)
    );

    initial begin
        // Initialize Inputs
        clk = 0;
        resetn = 0;
        rf_addr = 5'd0; // Initially observe R0
        mem_addr = 32'd0;

        // Wait 100 ns for global reset to finish
        #100;
        resetn = 1;
        $display("Time: %0t ns, CPU Reset Released.", $time);

        // After ADDIU $s0, $0, 0xAAAA (Inst 1, PC=0) completes (WB ~50ns after reset release)
        #60; // Wait until ~160ns. WB should be around 150ns.
        rf_addr = 5'd16; // Observe $s0 (R16)
        $display("Time: %0t ns, TB: Observing R[16] ($s0) after ADDIU $s0, $0, 0xAAAA. Expected: 0x0000AAAA", $time);

        // After NOTRD $t0, $s0 (Inst 2, PC=4) completes (WB ~50ns after previous, so ~100ns after reset release)
        #50; // Wait until ~210ns. WB should be around 200ns.
        rf_addr = 5'd8;  // Observe $t0 (R8)
        $display("Time: %0t ns, TB: Observing R[8] ($t0) after NOTRD $t0, $s0. Expected: ~0x0000AAAA = 0xFFFF5555", $time);

        // After NEG $t1, $t0 (Inst 3, PC=8) completes (WB ~50ns after previous, so ~150ns after reset release)
        #50; // Wait until ~260ns. WB should be around 250ns.
        rf_addr = 5'd9;  // Observe $t1 (R9)
        $display("Time: %0t ns, TB: Observing R[9] ($t1) after NEG $t1, $t0. Expected: -0xFFFF5555 = 0x0000AAAB", $time);

        // After INC $t2, $t1 (Inst 4, PC=C) completes (WB ~50ns after previous, so ~200ns after reset release)
        #50; // Wait until ~310ns. WB should be around 300ns.
        rf_addr = 5'd10; // Observe $t2 (R10)
        $display("Time: %0t ns, TB: Observing R[10] ($t2) after INC $t2, $t1. Expected: 0x0000AAAB + 1 = 0x0000AAAC", $time);

        // Restore observations for original test.coe sequence if needed, or add more new ones.
        // For now, we can let it run and observe the debug signals.
        // Original tb.v display points started after this.
        // To align with original timings, we are at ~310ns. Original sequence starts effectively at PC=0x10

        #50; // Give some time before potentially changing rf_addr for original instructions. Current total time ~360ns.

        // Observe R[1] after original ADDIU $1, $0, 1 (Original Inst 1, now Inst 5, PC=0x10) (WB ~50ns after PC=0x10 starts, so ~200+50 = 250ns from this point if PC=0x10 started at ~310ns, total ~560ns)
        // The CPU takes 5 cycles per simple instruction. First 4 instructions take ~20 cycles = 200ns. So PC=0x10 starts IF at ~100ns(reset) + 200ns = 300ns.
        // WB for PC=0x10 is 4 cycles after its IF, so IF@300, ID@310, EXE@320, MEM@330, WB@340.
        // So at ~350ns total time, R[1] should be updated.
        // Current time is ~310ns from reset. Adding #60 should get us to ~370ns to be safe.
        rf_addr = 5'd1; 
        $display("Time: %0t ns, TB: Observing R[1] for result of original ADDIU $1, $0, 1 (expected: 1)", $time);

        #60; // Wait till ~430ns total. (Original Inst 1 (now 5) WB was ~340-350ns)

        // Observe R[2] after original SLL $2, $1, 4 (Original Inst 2, now Inst 6, PC=0x14)
        // WB for PC=0x14 is 4 cycles after its IF. IF@350, WB@390.
        rf_addr = 5'd2;
        $display("Time: %0t ns, TB: Observing R[2] for result of original SLL $2, $1, 4 (expected: R[1]<<4 = 0x10)", $time);

        // ... (rest of the original tb.v observations can be added here, adjusting timings)
        // For simplicity, I will stop explicit register checking here for now.

    end
   always #5 clk=~clk;
endmodule

