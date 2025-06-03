`timescale 1ns / 1ps
//*************************************************************************
//   > �ļ���: single_cycle_cpu.v
//   > ����  :������CPUģ�飬��ʵ��16��ָ��
//   >        ָ��rom������ram�������첽�����ݣ��Ա㵥����CPU��ʵ��
//   > ����  : LOONGSON
//   > ����  : 2016-04-14
//*************************************************************************
`define STARTADDR 32'd0  // ������ʼ��ַ
module single_cycle_cpu(
    input clk,    // ʱ��
    input resetn,  // ��λ�źţ��͵�ƽ��Ч

    //display data
    input  [ 4:0] rf_addr,
    input  [31:0] mem_addr,
    output [31:0] rf_data,
    output [31:0] mem_data,
    output [31:0] cpu_pc,
    output [31:0] cpu_inst,

    // New debug outputs for ALU operands
    output wire [31:0] cpu_debug_op1,
    output wire [31:0] cpu_debug_op2
    );

//---------------------------------{ȡָ}begin------------------------------------//
    reg  [31:0] pc;
    wire [31:0] next_pc;
    wire [31:0] seq_pc;
    wire [31:0] jbr_target;
    wire jbr_taken;

    // ��һָ���ַ��seq_pc=pc+4
    assign seq_pc[31:2]    = pc[31:2] + 1'b1;
    assign seq_pc[1:0]     = pc[1:0];
    // ��ָ���ָ����ת��Ϊ��ת��ַ������Ϊ��һָ��
    assign next_pc = jbr_taken ? jbr_target : seq_pc;
    always @ (posedge clk)    // PC���������
    begin
        if (!resetn) begin
            pc <= `STARTADDR; // ��λ��ȡ������ʼ��ַ
        end
        else begin
            pc <= next_pc;    // ����λ��ȡ��ָ��
        end
    end

    wire [31:0] inst_addr;
    wire [31:0] inst;
    assign inst_addr = pc;  // ָ���ַ��ָ���32λ
    inst_rom inst_rom_module(         // ָ��洢��
        .addr      (inst_addr[6:2]),  // I, 5,ָ���ַ
        .inst      (inst          )   // O, 32,ָ��
    );
    assign cpu_pc = pc;       //display pc
    assign cpu_inst = inst;
//----------------------------------{ȡָ}end-------------------------------------//

//---------------------------------{����}begin------------------------------------//
    wire [5:0] op;       
    wire [4:0] rs;       
    wire [4:0] rt;       
    wire [4:0] rd;       
    wire [4:0] sa;      
    wire [5:0] funct;    
    wire [15:0] imm;     
    wire [15:0] offset;  
    wire [25:0] target;  

    assign op     = inst[31:26];  // ������
    assign rs     = inst[25:21];  // Դ������1
    assign rt     = inst[20:16];  // Դ������2
    assign rd     = inst[15:11];  // Ŀ�������
    assign sa     = inst[10:6];   // �����򣬿��ܴ��ƫ����
    assign funct  = inst[5:0];    // ������
    assign imm    = inst[15:0];   // ������
    assign offset = inst[15:0];   // ��ַƫ����
    assign target = inst[25:0];   // Ŀ���ַ

    wire [31:0] rs_value, rt_value;

    wire op_zero;  // ȫ0
    wire sa_zero;  // saȫ0
    assign op_zero = ~(|op);
    assign sa_zero = ~(|sa);
    
    // ʵ��ָ���б�
    wire inst_ADDU, inst_SUBU , inst_SLT, inst_AND;
    wire inst_NOR , inst_OR   , inst_XOR, inst_SLL;
    wire inst_SRL , inst_ADDIU, inst_BEQ, inst_BNE;
    wire inst_LW  , inst_SW   , inst_LUI, inst_J;

    assign inst_ADDU  = op_zero & sa_zero    & (funct == 6'b100001);// �޷��żӷ�
    assign inst_SUBU  = op_zero & sa_zero    & (funct == 6'b100011);// �޷��ż���
    assign inst_SLT   = op_zero & sa_zero    & (funct == 6'b101010);// С������λ
    assign inst_AND   = op_zero & sa_zero    & (funct == 6'b100100);// �߼�������
    assign inst_NOR   = op_zero & sa_zero    & (funct == 6'b100111);// �߼��������
    assign inst_OR    = op_zero & sa_zero    & (funct == 6'b100101);// �߼�������
    assign inst_XOR   = op_zero & sa_zero    & (funct == 6'b100110);// �߼��������
    assign inst_SLL   = op_zero & (rs==5'd0) & (funct == 6'b000000);// �߼�����
    assign inst_SRL   = op_zero & (rs==5'd0) & (funct == 6'b000010);// �߼�����
    assign inst_ADDIU = (op == 6'b001001);                  // �������޷��żӷ�
    assign inst_BEQ   = (op == 6'b000100);                  // �ж������ת
    assign inst_BNE   = (op == 6'b000101);                  // �жϲ�����ת
    assign inst_LW    = (op == 6'b100011);                  // ���ڴ�װ��
    assign inst_SW    = (op == 6'b101011);                  // ���ڴ�洢
    assign inst_LUI   = (op == 6'b001111);                  // ������װ�ظ߰��ֽ�
    assign inst_J     = (op == 6'b000010);                  // ֱ����ת

    // ��������ת�ж�
    wire        j_taken;
    wire [31:0] j_target;
    assign j_taken  = inst_J;
    // ��������תĿ���ַ��PC={PC[31:28],target<<2}
    assign j_target = {pc[31:28], target, 2'b00};

    //��֧��ת
    wire        beq_taken;
    wire        bne_taken;
    wire [31:0] br_target;
    assign beq_taken = (rs_value == rt_value);       // BEQ��ת������GPR[rs]=GPR[rt]
    assign bne_taken = ~beq_taken;                   // BNE��ת������GPR[rs]��GPR[rt]
    assign br_target[31:2] = pc[31:2] + {{14{offset[15]}}, offset};
    assign br_target[1:0]  = pc[1:0];    // ��֧��תĿ���ַ��PC=PC+offset<<2

    //��תָ�����ת�źź���תĿ���ַ
    assign jbr_taken = j_taken              // ָ����ת����������ת �� �����֧��ת����
                     | inst_BEQ & beq_taken
                     | inst_BNE & bne_taken;
    assign jbr_target = j_taken ? j_target : br_target;

    // �Ĵ�����
    wire rf_wen;
    wire [4:0] rf_waddr;
    wire [31:0] rf_wdata;

    regfile rf_module(
        .clk    (clk      ),  // I, 1
        .wen    (rf_wen   ),  // I, 1
        .raddr1 (rs       ),  // I, 5
        .raddr2 (rt       ),  // I, 5
        .waddr  (rf_waddr ),  // I, 5
        .wdata  (rf_wdata ),  // I, 32
        .rdata1 (rs_value ),  // O, 32
        .rdata2 (rt_value ),   // O, 32

        //display rf
        .test_addr(rf_addr),
        .test_data(rf_data)
    );
    
    
    // ���ݵ�ִ��ģ���ALUԴ�������Ͳ�����
    wire inst_add, inst_sub, inst_slt,inst_sltu;
    wire inst_and, inst_nor, inst_or, inst_xor;
    wire inst_sll, inst_srl, inst_sra,inst_lui;
    assign inst_add = inst_ADDU | inst_ADDIU | inst_LW | inst_SW; // ���ӷ�����ָ��
    assign inst_sub = inst_SUBU; // ����
    assign inst_slt = inst_SLT;  // С����λ
    assign inst_sltu= 1'b0;      // ��δʵ��
    assign inst_and = inst_AND;  // �߼���
    assign inst_nor = inst_NOR;  // �߼����
    assign inst_or  = inst_OR;   // �߼���
    assign inst_xor = inst_XOR;  // �߼����
    assign inst_sll = inst_SLL;  // �߼�����
    assign inst_srl = inst_SRL;  // �߼�����
    assign inst_sra = 1'b0;      // ��δʵ��
    assign inst_lui = inst_LUI;  // ������װ�ظ�λ

    wire [31:0] sext_imm;
    wire   inst_shf_sa;    //ʹ��sa����Ϊƫ������ָ��
    wire   inst_imm_sign;  //����������������չ��ָ��
    assign sext_imm      = {{16{imm[15]}}, imm};// ������������չ
    assign inst_shf_sa   = inst_SLL | inst_SRL;
    assign inst_imm_sign = inst_ADDIU | inst_LUI | inst_LW | inst_SW;
    
    wire [31:0] alu_operand1;
    wire [31:0] alu_operand2;
    wire [11:0] alu_control;
    assign alu_operand1 = inst_shf_sa ? {27'd0,sa} : rs_value;
    assign alu_operand2 = inst_imm_sign ? sext_imm : rt_value;
    assign alu_control = {inst_add,        // ALU�����룬���ȱ���
                          inst_sub,
                          inst_slt,
                          inst_sltu,
                          inst_and,
                          inst_nor,
                          inst_or, 
                          inst_xor,
                          inst_sll,
                          inst_srl,
                          inst_sra,
                          inst_lui};

    assign cpu_debug_op1 = alu_operand1;
    assign cpu_debug_op2 = alu_operand2;
//----------------------------------{}end-------------------------------------//

//---------------------------------{ִ��}begin------------------------------------//
    wire [31:0] alu_result;

    alu alu_module(
        .alu_control  (alu_control ),  // I, 12, ALU�����ź�
        .alu_src1     (alu_operand1),  // I, 32, ALU������1
        .alu_src2     (alu_operand2),  // I, 32, ALU������2
        .alu_result   (alu_result  )   // O, 32, ALU���
    );
//----------------------------------{ִ��}end-------------------------------------//
    
//---------------------------------{�ô�}begin------------------------------------//
    wire [3 :0] dm_wen;
    wire [31:0] dm_addr;
    wire [31:0] dm_wdata;
    wire [31:0] dm_rdata;
    assign dm_wen   = {4{inst_SW}} & resetn;    // �ڴ�дʹ��,��resetn״̬����Ч
    assign dm_addr  = alu_result;               // �ڴ�д��ַ��ΪALU���
    assign dm_wdata = rt_value;                 // �ڴ�д���ݣ�Ϊrt�Ĵ���ֵ
    data_ram data_ram_module(
        .clk   (clk         ),  // I, 1,  ʱ��
        .wen   (dm_wen      ),  // I, 1,  дʹ��
        .addr  (dm_addr[6:2]),  // I, 32, ����ַ
        .wdata (dm_wdata    ),  // I, 32, д����
        .rdata (dm_rdata    ),  // O, 32, ������

        //display mem
        .test_addr(mem_addr[6:2]),
        .test_data(mem_data     )
    );
//----------------------------------{�ô�}end-------------------------------------//

//---------------------------------{д��}begin------------------------------------//
    wire inst_wdest_rt;   // �Ĵ�����д���ַΪrt��ָ��
    wire inst_wdest_rd;   // �Ĵ�����д���ַΪrd��ָ��
    assign inst_wdest_rt = inst_ADDIU | inst_LW | inst_LUI;
    assign inst_wdest_rd = inst_ADDU | inst_SUBU | inst_SLT | inst_AND | inst_NOR
                          | inst_OR   | inst_XOR  | inst_SLL | inst_SRL;                   
    // �Ĵ�����дʹ���źţ��Ǹ�λ״̬����Ч
    assign rf_wen   = (inst_wdest_rt | inst_wdest_rd) & resetn;
    assign rf_waddr = inst_wdest_rd ? rd : rt;        // �Ĵ�����д��ַrd��rt
    assign rf_wdata = inst_LW ? dm_rdata : alu_result;// д�ؽ����Ϊload�����ALU���
//----------------------------------{д��}end-------------------------------------//
endmodule
