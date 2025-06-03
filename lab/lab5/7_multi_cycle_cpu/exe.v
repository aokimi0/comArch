`timescale 1ns / 1ps
//*************************************************************************
//   > �ļ���: exe.v
//   > ����  :������CPU��ִ��ģ��
//   > ����  : LOONGSON
//   > ����  : 2016-04-14
//*************************************************************************
module exe(                         // ִ�м�
    input              EXE_valid,   // ִ�м���Ч�ź�
    input      [152:0] ID_EXE_bus_r,// ID->EXE
    output             EXE_over,    // EXEģ��ִ�����
    output     [105:0] EXE_MEM_bus, // EXE->MEM����
    
    //չʾPC
    output     [ 31:0] EXE_pc
);
//-----{ID->EXE����}begin
    //EXE��Ҫ�õ�����Ϣ
    //ALU����Դ�������Ϳ����ź�
    wire [14:0] alu_control_wire;
    wire [31:0] alu_operand1_wire;
    wire [31:0] alu_operand2_wire;

    //ôҪõload/storeϢ
    wire [3:0] mem_control_wire;
    wire [31:0] store_data_wire;
                          
    //дҪõϢ
    wire       rf_wen_wire;
    wire [4:0] rf_wdest_wire;
    
    //pc
    wire [31:0] pc_wire;

    // Deconstruct the ID_EXE_bus_r according to decode.v's new structure
    // {alu_control_internal (15), alu_operand1_internal (32), alu_operand2_internal (32), mem_control (4), store_data (32), rf_wen_internal (1), rf_wdest_internal (5), pc (32)}
    assign {alu_control_wire,    // 15 bits
            alu_operand1_wire,   // 32 bits
            alu_operand2_wire,   // 32 bits
            mem_control_wire,    // 4 bits
            store_data_wire,     // 32 bits
            rf_wen_wire,         // 1 bit
            rf_wdest_wire,       // 5 bits
            pc_wire              // 32 bits
           } = ID_EXE_bus_r;
//-----{ID->EXE}end

//-----{ALU}begin
    wire [31:0] alu_result_wire;

    alu alu_module(
        .alu_control  (alu_control_wire ),  // I, 15 (was 12), ALUź
        .alu_src1     (alu_operand1_wire), // I, 32, ALU1
        .alu_src2     (alu_operand2_wire), // I, 32, ALU2
        .alu_result   (alu_result_wire  )  // O, 32, ALU
    );
//-----{ALU}end

//-----{EXEִ}begin
    //Ƕڵģ
    //ALU㶼����һ�������
    //��EXEģ��һ�ľ���������в���
    //��EXE_valid����EXE_over�ź�
    assign EXE_over = EXE_valid;
//-----{EXEִ�����}end

//-----{EXE->MEM����}begin
    // EXE_MEM_bus width is [105:0] and should remain unchanged as alu_control is not part of it.
    // {mem_control (4), store_data (32), alu_result (32), rf_wen (1), rf_wdest (5), pc (32)}
    // Total: 4 + 32 + 32 + 1 + 5 + 32 = 106 bits.
    assign EXE_MEM_bus = {mem_control_wire, store_data_wire,   
                          alu_result_wire,               
                          rf_wen_wire, rf_wdest_wire,          
                          pc_wire};
//-----{EXE->MEM}end

//-----{չʾEXEģPCֵ}begin
    assign EXE_pc = pc_wire;
//-----{չʾEXEģPCֵ}end
endmodule
