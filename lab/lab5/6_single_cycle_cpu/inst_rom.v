`timescale 1ns / 1ps
//*************************************************************************
//   > �ļ���: inst_rom.v
//   > ����  ���첽ָ��洢��ģ�飬���üĴ�������ɣ����ƼĴ�����
//   >         ��Ƕ��ָ�ֻ�����첽��
//   > ����  : LOONGSON
//   > ����  : 2016-04-14
//*************************************************************************
module inst_rom(
    input      [4 :0] addr, // ָ���ַ
    output reg [31:0] inst       // ָ��
    );

    wire [31:0] inst_rom[19:0];  // ָ��洢�����ֽڵ�ַ7'b000_0000~7'b111_1111
    //------------- ָ����� ---------|ָ���ַ|--- ���ָ�� -----|- ָ���� -----
    // Test ADDU: $3 = $1 + $2
    assign inst_rom[ 0] = 32'h2401000A; // addiu $1, $0, 10     ($1 = 10)
    assign inst_rom[ 1] = 32'h24020014; // addiu $2, $0, 20     ($2 = 20)
    assign inst_rom[ 2] = 32'h00221821; // addu $3, $1, $2      ($3 = 10 + 20 = 30)

    // Test AND: $6 = $4 & $5
    assign inst_rom[ 3] = 32'h24040007; // addiu $4, $0, 7      ($4 = 7)
    assign inst_rom[ 4] = 32'h2405000D; // addiu $5, $0, 13     ($5 = 13)
    assign inst_rom[ 5] = 32'h00853024; // and $6, $4, $5       ($6 = 7 & 13 = 5)

    // Fill rest with NOPs
    assign inst_rom[ 6] = 32'h00000000; // NOP
    assign inst_rom[ 7] = 32'h00000000; // NOP
    assign inst_rom[ 8] = 32'h00000000; // NOP
    assign inst_rom[ 9] = 32'h00000000; // NOP
    assign inst_rom[10] = 32'h00000000; // NOP
    assign inst_rom[11] = 32'h00000000; // NOP
    assign inst_rom[12] = 32'h00000000; // NOP
    assign inst_rom[13] = 32'h00000000; // NOP
    assign inst_rom[14] = 32'h00000000; // NOP
    assign inst_rom[15] = 32'h00000000; // NOP
    assign inst_rom[16] = 32'h00000000; // NOP
    assign inst_rom[17] = 32'h00000000; // NOP
    assign inst_rom[18] = 32'h00000000; // NOP
    assign inst_rom[19] = 32'h00000000; // NOP

    //ָ,ȡ4ֽ
    always @(*)
    begin
        case (addr)
            5'd0 : inst <= inst_rom[0 ];
            5'd1 : inst <= inst_rom[1 ];
            5'd2 : inst <= inst_rom[2 ];
            5'd3 : inst <= inst_rom[3 ];
            5'd4 : inst <= inst_rom[4 ];
            5'd5 : inst <= inst_rom[5 ];
            5'd6 : inst <= inst_rom[6 ];
            5'd7 : inst <= inst_rom[7 ];
            5'd8 : inst <= inst_rom[8 ];
            5'd9 : inst <= inst_rom[9 ];
            5'd10: inst <= inst_rom[10];
            5'd11: inst <= inst_rom[11];
            5'd12: inst <= inst_rom[12];
            5'd13: inst <= inst_rom[13];
            5'd14: inst <= inst_rom[14];
            5'd15: inst <= inst_rom[15];
            5'd16: inst <= inst_rom[16];
            5'd17: inst <= inst_rom[17];
            5'd18: inst <= inst_rom[18];
            5'd19: inst <= inst_rom[19];
            default: inst <= 32'd0;
        endcase
    end
endmodule