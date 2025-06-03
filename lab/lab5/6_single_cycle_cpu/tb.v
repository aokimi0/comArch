`timescale 1ns / 1ps

////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer:
//
// Create Date:   16:49:44 04/19/2016
// Design Name:   single_cycle_cpu
// Module Name:   F:/new_lab/6_single_cycle_cpu/tb.v
// Project Name:  single_cycle_cpu
// Target Device:  
// Tool versions:  
// Description: 
//
// Verilog Test Fixture created by ISE for module: single_cycle_cpu
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
    wire [31:0] cpu_pc;
    wire [31:0] cpu_inst;
    wire [31:0] debug_op1_from_cpu;
    wire [31:0] debug_op2_from_cpu;

    // Instantiate the Unit Under Test (UUT)
    single_cycle_cpu uut (
        .clk(clk), 
        .resetn(resetn), 
        .rf_addr(rf_addr), 
        .mem_addr(mem_addr), 
        .rf_data(rf_data), 
        .mem_data(mem_data), 
        .cpu_pc(cpu_pc), 
        .cpu_inst(cpu_inst),
        .cpu_debug_op1(debug_op1_from_cpu),
        .cpu_debug_op2(debug_op2_from_cpu)
    );

    initial begin
        // Initialize Inputs
        clk = 0;
        resetn = 0;
        rf_addr = 1;
        mem_addr = 0;

        // Wait 100 ns for global reset to finish
        #100;
        resetn = 1; // De-assert reset

        $display("Starting instruction-by-instruction (waveform-focused) verification...\n");
        // At this point, PC = 0x00

        // --- Instruction @ PC=0x00: addiu $1, $0, 10 ---
        #10; // CPU executes instruction. PC becomes 0x04. $1 is now 10.
        $display("=== After 'addiu $1, $0, 10' (PC was 0x00) ===");
        rf_addr = 1;   // Set rf_addr to observe $1
        #2;            // Allow rf_data to update for waveform and display
        $display("  $1 (Destination) = %0d (0x%h)\n", rf_data, rf_data);
        #8;            // Hold this view for the remainder of an 10ns "observation slot"

        // --- Instruction @ PC=0x04: addiu $2, $0, 20 ---
        #10; // CPU executes instruction. PC becomes 0x08. $2 is now 20.
        $display("=== After 'addiu $2, $0, 20' (PC was 0x04) ===");
        rf_addr = 2;   // Set rf_addr to observe $2
        #2;
        $display("  $2 (Destination) = %0d (0x%h)\n", rf_data, rf_data);
        #8;

        // --- Instruction @ PC=0x08: addu $3, $1, $2 ---
        #10; // CPU executes instruction. PC becomes 0x0C. $3 is now 30.
        $display("=== After 'addu $3, $1, $2' (PC was 0x08) ===");
        $display("  Verifying $3 = $1 (10) + $2 (20)");
        rf_addr = 3;   // Set rf_addr to observe $3 (the destination)
        #2;
        $display("    Result  $3 = %0d (0x%h) (Expected: 30)\n", rf_data, rf_data);
        #8;

        // --- Instruction @ PC=0x0C: addiu $4, $0, 7 ---
        #10; // CPU executes. PC becomes 0x10. $4 is now 7.
        $display("=== After 'addiu $4, $0, 7' (PC was 0x0C) ===");
        rf_addr = 4;
        #2;
        $display("  $4 (Destination) = %0d (0x%h)\n", rf_data, rf_data);
        #8;

        // --- Instruction @ PC=0x10: addiu $5, $0, 13 ---
        #10; // CPU executes. PC becomes 0x14. $5 is now 13.
        $display("=== After 'addiu $5, $0, 13' (PC was 0x10) ===");
        rf_addr = 5;
        #2;
        $display("  $5 (Destination) = %0d (0x%h)\n", rf_data, rf_data);
        #8;

        // --- Instruction @ PC=0x14: and $6, $4, $5 ---
        #10; // CPU executes. PC becomes 0x18. $6 is now 5.
        $display("=== After 'and $6, $4, $5' (PC was 0x14) ===");
        $display("  Verifying $6 = $4 (7) & $5 (13)");
        rf_addr = 6;   // Set rf_addr to observe $6 (the destination)
        #2;
        $display("    Result  $6 = %0d (0x%h) (Expected: 5)\n", rf_data, rf_data);
        #8;

        // --- Instruction @ PC=0x18: NOP ---
        #10; // CPU executes NOP. PC becomes 0x1C. Registers $1-$6 should be unchanged.
        $display("=== After NOP (PC was 0x18) ===");
        rf_addr = 6; // Keep observing $6 for example, or change to $0
        #2;
        $display("  Value of $6 (should still be 5) = %0d (0x%h)\n", rf_data, rf_data);
        #8;

        $stop; // Stop simulation
    end
    always #5 clk=~clk;
endmodule

