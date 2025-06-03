`timescale 1ns / 1ps
//*************************************************************************
//   > 文件名: decode.v
//   > 功能  : 设计CPU模块
//   > 作者  : LOONGSON
//   > 日期  : 2016-04-14
//*************************************************************************
module decode(                      // 输入模块
    input              ID_valid,    // 输入模块有效信号
    input      [ 63:0] IF_ID_bus_r, // IF->ID总线
    input      [ 31:0] rs_value,    // 第一源寄存器值
    input      [ 31:0] rt_value,    // 第二源寄存器值
    output     [  4:0] rs,          // 第一源寄存器地址 
    output     [  4:0] rt,          // 第二源寄存器地址
    output     [ 32:0] jbr_bus,     // 跳转信号
    output             jbr_not_link,// 指针为跳转指令,且不link指令
    output             ID_over,     // ID模块执行信号
    output    [152:0]  ID_EXE_bus,  // ID->EXE总线
    
    //显示PC
    output     [ 31:0] ID_pc
);
//-----{IF->ID总线}begin
    wire [31:0] pc;
    wire [31:0] inst;
    assign {pc, inst} = IF_ID_bus_r;  // IF->ID总线最高位PC和指令
//-----{IF->ID总线}end

//-----{指令分析}begin
    wire [5:0] op;       
    wire [4:0] rd;       
    wire [4:0] sa;      
    wire [5:0] funct;    
    wire [15:0] imm;     
    wire [15:0] offset;  
    wire [25:0] target;  

    assign op     = inst[31:26];  // 操作码
    assign rs     = inst[25:21];  // 源寄存器1
    assign rt     = inst[20:16];  // 源寄存器2
    assign rd     = inst[15:11];  // 目标寄存器
    assign sa     = inst[10:6];   // 立即数扩展，可能为偏移量
    assign funct  = inst[5:0];    // 功能码
    assign imm    = inst[15:0];   // 立即数
    assign offset = inst[15:0];   // 偏移量
    assign target = inst[25:0];   // 目标地址

    // 实际指令表
    wire inst_ADDU, inst_SUBU , inst_SLT , inst_AND;
    wire inst_NOR , inst_OR   , inst_XOR , inst_SLL;
    wire inst_SRL , inst_ADDIU, inst_BEQ , inst_BNE;
    wire inst_LW  , inst_SW   , inst_LUI , inst_J;
    wire inst_SLTU, inst_JALR , inst_JR  , inst_SLLV;
    wire inst_SRA , inst_SRAV , inst_SRLV, inst_SLTIU;
    wire inst_SLTI, inst_BGEZ , inst_BGTZ, inst_BLEZ;
    wire inst_BLTZ, inst_LB   , inst_LBU , inst_SB;
    wire inst_ANDI, inst_ORI  , inst_XORI, inst_JAL;
    wire inst_NOTRD, inst_NEG, inst_INC; // NEW misc instructions
    wire op_zero;  // 操作码全0
    wire sa_zero;  // sa全0
    assign op_zero = ~(|op);
    assign sa_zero = ~(|sa);
    assign inst_ADDU  = op_zero & sa_zero    & (funct == 6'b100001);//不能写加法
    assign inst_SUBU  = op_zero & sa_zero    & (funct == 6'b100011);//不能写减法
    assign inst_SLT   = op_zero & sa_zero    & (funct == 6'b101010);//比较结果为0
    assign inst_SLTU  = op_zero & sa_zero    & (funct == 6'b101011);//不能比较结果为0
    assign inst_JALR  = op_zero & (rt==5'd0) & (rd==5'd31) 
                      & sa_zero & (funct == 6'b001001);          //跳转指令的源寄存器
    assign inst_JR    = op_zero & (rt==5'd0) & (rd==5'd0 )
                      & sa_zero & (funct == 6'b001000);             //跳转指令
    assign inst_AND   = op_zero & sa_zero    & (funct == 6'b100100);//逻辑与
    assign inst_NOR   = op_zero & sa_zero    & (funct == 6'b100111);//逻辑或
    assign inst_OR    = op_zero & sa_zero    & (funct == 6'b100101);//逻辑或
    assign inst_XOR   = op_zero & sa_zero    & (funct == 6'b100110);//逻辑异或
    assign inst_SLL   = op_zero & (rs==5'd0) & (funct == 6'b000000);//逻辑左移
    assign inst_SLLV  = op_zero & sa_zero    & (funct == 6'b000100);//逻辑左移逻辑
    assign inst_SRA   = op_zero & (rs==5'd0) & (funct == 6'b000011);//逻辑右移
    assign inst_SRAV  = op_zero & sa_zero    & (funct == 6'b000111);//逻辑右移逻辑
    assign inst_SRL   = op_zero & (rs==5'd0) & (funct == 6'b000010);//逻辑右移
    assign inst_SRLV  = op_zero & sa_zero    & (funct == 6'b000110);//逻辑右移逻辑
    assign inst_ADDIU = (op == 6'b001001);              //不能写加法
    assign inst_SLTI  = (op == 6'b001010);              //比较结果为0
    assign inst_SLTIU = (op == 6'b001011);              //不能比较结果为0
    assign inst_BEQ   = (op == 6'b000100);              //比较跳转
    assign inst_BGEZ  = (op == 6'b000001) & (rt==5'd1); //大于等于0跳转
    assign inst_BGTZ  = (op == 6'b000111) & (rt==5'd0); //大于0跳转
    assign inst_BLEZ  = (op == 6'b000110) & (rt==5'd0); //小于等于0跳转
    assign inst_BLTZ  = (op == 6'b000001) & (rt==5'd0); //小于0跳转
    assign inst_BNE   = (op == 6'b000101);              //比较跳转
    assign inst_LW    = (op == 6'b100011);              //加载字
    assign inst_SW    = (op == 6'b101011);              //存储字
    assign inst_LB    = (op == 6'b100000);              //加载字，不扩展
    assign inst_LBU   = (op == 6'b100100);              //加载字，不能扩展
    assign inst_SB    = (op == 6'b101000);              //存储字
    assign inst_ANDI  = (op == 6'b001100);              //逻辑与
    assign inst_LUI   = (op == 6'b001111) & (rs==5'd0); //逻辑左移高16位
    assign inst_ORI   = (op == 6'b001101);              //逻辑或
    assign inst_XORI  = (op == 6'b001110);              //逻辑异或
    assign inst_J     = (op == 6'b000010);              //跳转
    assign inst_JAL   = (op == 6'b000011);              //跳转指令

    // NEW Misc Instruction decoding
    assign inst_NOTRD = op_zero & (rt==5'd0) & sa_zero & (funct == 6'b101100); // NOTRD funct = 44
    assign inst_NEG   = op_zero & (rt==5'd0) & sa_zero & (funct == 6'b101101); // NEG funct = 45
    assign inst_INC   = op_zero & (rt==5'd0) & sa_zero & (funct == 6'b101110); // INC funct = 46

    //跳转指令支持信号
    wire inst_jr;    //跳转指令
    wire inst_j_link;//跳转指令
    assign inst_jr     = inst_JALR | inst_JR;
    assign inst_j_link = inst_JAL  | inst_JALR;
    assign jbr_not_link= inst_J    | inst_JR      //全不link跳转指令
                       | inst_BEQ  | inst_BNE  | inst_BGEZ
                       | inst_BGTZ | inst_BLEZ | inst_BLTZ;
        
    //load store
    wire inst_load;
    wire inst_store;
    assign inst_load  = inst_LW | inst_LB | inst_LBU;  // load指令
    assign inst_store = inst_SW | inst_SB;             // store指令
    
    //alu操作信号
    wire inst_add, inst_sub, inst_slt,inst_sltu;
    wire inst_and, inst_nor, inst_or, inst_xor;
    wire inst_sll, inst_srl, inst_sra,inst_lui;
    assign inst_add = inst_ADDU  | inst_ADDIU | inst_load
                     | inst_store | inst_j_link;            // 加法
    assign inst_sub = inst_SUBU;                            // 减法
    assign inst_slt = inst_SLT | inst_SLTI;                 // 比较结果为0
    assign inst_sltu= inst_SLTIU | inst_SLTU;               // 不能比较结果为0
    assign inst_and = inst_AND | inst_ANDI;                 // 逻辑与
    assign inst_nor = inst_NOR;                             // 逻辑或
    assign inst_or  = inst_OR  | inst_ORI;                  // 逻辑或
    assign inst_xor = inst_XOR | inst_XORI;                 // 逻辑异或
    assign inst_sll = inst_SLL | inst_SLLV;                 // 逻辑左移
    assign inst_srl = inst_SRL | inst_SRLV;                 // 逻辑右移
    assign inst_sra = inst_SRA | inst_SRAV;                 // 逻辑右移
    assign inst_lui = inst_LUI;                             // 逻辑左移高16位
    
    //使用sa作为偏移量扩展逻辑指令
    wire inst_shf_sa;
    assign inst_shf_sa =  inst_SLL | inst_SRL | inst_SRA;
    
    //逻辑指令扩展信号
    wire inst_imm_zero; //逻辑0扩展
    wire inst_imm_sign; //逻辑扩展
    assign inst_imm_zero = inst_ANDI  | inst_LUI  | inst_ORI | inst_XORI;
    assign inst_imm_sign = inst_ADDIU | inst_SLTI | inst_SLTIU
                         | inst_load  | inst_store;
    
    //目标寄存器指令信号
    wire inst_wdest_rt;  // 指令写目标寄存器为rt指令
    wire inst_wdest_31;  // 指令写目标寄存器为31指令
    wire inst_wdest_rd;  // 指令写目标寄存器为rd指令
    assign inst_wdest_rt = inst_imm_zero | inst_ADDIU | inst_SLTI
                         | inst_SLTIU    | inst_load;
    assign inst_wdest_31 = inst_JAL;
    assign inst_wdest_rd = inst_ADDU | inst_SUBU | inst_SLT  | inst_SLTU
                         | inst_JALR | inst_AND  | inst_NOR  | inst_OR
                         | inst_XOR  | inst_SLL  | inst_SLLV | inst_SRA 
                         | inst_SRAV | inst_SRL  | inst_SRLV;
//-----{指令分析}end

//-----{跳转指令执行}begin
    //跳转指令
    wire        j_taken;
    wire [31:0] j_target;
    assign j_taken = inst_J | inst_JAL | inst_jr;
    //跳转指令目标寄存器为rs_value,跳转为{pc[31:28],target,2'b00}
    assign j_target = inst_jr ? rs_value : {pc[31:28],target,2'b00};

    //branch指令
    wire rs_equql_rt;
    wire rs_ez;
    wire rs_ltz;
    assign rs_equql_rt = (rs_value == rt_value);   // GPR[rs]==GPR[rt]
    assign rs_ez       = ~(|rs_value);             // rs寄存器值为0
    assign rs_ltz      = rs_value[31];             // rs寄存器值小于0
    wire br_taken;
    wire [31:0] br_target;
    assign br_taken = inst_BEQ  & rs_equql_rt      // 比较跳转
                    | inst_BNE  & ~rs_equql_rt     // 比较跳转
                    | inst_BGEZ & ~rs_ltz          // 大于等于0跳转
                    | inst_BGTZ & ~rs_ltz & ~rs_ez // 大于0跳转
                    | inst_BLEZ & (rs_ltz | rs_ez) // 小于等于0跳转
                    | inst_BLTZ & rs_ltz;          // 小于0跳转
    // 支持跳转目标地址PC=PC+offset<<2
    wire [31:0] pc_plus_4;
    assign pc_plus_4 = pc + 4; // PC of the instruction following the branch

    wire [31:0] sign_extended_offset_bytes;
    // Sign-extend the 16-bit offset and shift left by 2 (multiply by 4)
    assign sign_extended_offset_bytes = {{14{offset[15]}}, offset, 2'b00};

    // Corrected br_target assignment
    // assign br_target[31:2] = pc[31:2] + {{14{offset[15]}}, offset};  // OLD
    // assign br_target[1:0]  = pc[1:0];                                // OLD
    assign br_target = pc_plus_4 + sign_extended_offset_bytes;       // NEW
    
    //jump and branch指令
    wire jbr_taken;
    wire [31:0] jbr_target;
    assign jbr_taken  = j_taken | br_taken; 
    assign jbr_target = j_taken ? j_target : br_target;
    
    //ID->IF总线跳转信号
    assign jbr_bus = {jbr_taken, jbr_target};
//-----{跳转指令执行}end

//-----{ID执行模块}begin
    //如果ID模块是多周期模块，则需要检查ID_valid和ID_over信号
    //如果ID模块是单周期模块，则不需要检查ID_valid和ID_over信号
    assign ID_over = ID_valid;
//-----{ID执行模块}end

//-----{ID->EXE总线}begin
    //EXE需要获取的信息
    //ALU源寄存器和控制信号
    wire [14:0] alu_control_internal; // CHANGED from [11:0] alu_control
    wire [31:0] alu_operand1_internal;
    wire [31:0] alu_operand2_internal;
    
    //跳转指令是跳转指令的PC值，加4后，最高位寄存器写信号
    //在多周期CPU中，需要考虑跳转指令的PC+4，最高位寄存器写信号
    assign alu_operand1_internal = inst_j_link ? pc :  
                                   inst_shf_sa ? {27'd0,sa} : rs_value;
    assign alu_operand2_internal = inst_j_link ? 32'd4 :
                                   inst_imm_zero ? {16'd0, imm} :
                                   inst_imm_sign ?  {{16{imm[15]}}, imm} : rt_value;
    // Expanded alu_control to include new instructions
    assign alu_control_internal = {inst_add,        // ALU操作码，默认全0 [14]
                                   inst_sub,        // [13]
                                   inst_slt,        // [12]
                                   inst_sltu,       // [11]
                                   inst_and,        // [10]
                                   inst_nor,        // [9]
                                   inst_or,         // [8]
                                   inst_xor,        // [7]
                                   inst_sll,        // [6]
                                   inst_srl,        // [5]
                                   inst_sra,        // [4]
                                   inst_lui,        // [3]
                                   inst_NOTRD,      // [2] NEW
                                   inst_NEG,        // [1] NEW
                                   inst_INC};       // [0] NEW
    //需要获取的load/store信息
    wire lb_sign;  //load一位为写信号load
    wire ls_word;  //load/store为字节或字,0:byte;1:word
    wire [3:0] mem_control;  //MEM要使用的目标信号
    wire [31:0] store_data;  //store指令的源寄存器数据
    assign lb_sign = inst_LB;
    assign ls_word = inst_LW | inst_SW;
    assign mem_control = {inst_load,
                          inst_store,
                          ls_word,
                          lb_sign };
                          
    //写需要获取的信息
    wire       rf_wen_internal;    //写寄存器写信号
    wire [4:0] rf_wdest_internal;  //写寄存器目标寄存器
    assign rf_wen_internal = inst_wdest_rt | inst_wdest_31 | inst_wdest_rd;
    assign rf_wdest_internal = inst_wdest_rt ? rt :      //内部写寄存器时，目标寄存器为0
                              inst_wdest_31 ? 5'd31 :
                              inst_wdest_rd ? rd : 5'd0;
    assign store_data = rt_value;
   assign ID_EXE_bus = {alu_control_internal,       // 15 bits
                         alu_operand1_internal,    // 32 bits
                         alu_operand2_internal,    // 32 bits
                         mem_control,              // 4 bits (ensure this is the final mem_control for the bus)
                         store_data,               // 32 bits (ensure this is the final store_data for the bus)
                         rf_wen_internal,          // 1 bit (ensure this is the final rf_wen for the bus)
                         rf_wdest_internal,        // 5 bits (ensure this is the final rf_wdest for the bus)
                         pc};                      // 32 bits
//-----{ID->EXE总线}end

//-----{显示ID模块PC值}begin
    assign ID_pc = pc;
//-----{显示ID模块PC值}end
endmodule
