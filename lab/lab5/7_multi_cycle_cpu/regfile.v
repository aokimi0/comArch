`timescale 1ns / 1ps
//*************************************************************************
//   > �ļ���: regfile.v
//   > ����  ���Ĵ�����ģ�飬ͬ��д���첽��
//   > ����  : LOONGSON
//   > ����  : 2016-04-14
//*************************************************************************
module regfile(
    input             clk,
    input             resetn,
    input             wen,
    input      [4 :0] raddr1,
    input      [4 :0] raddr2,
    input      [4 :0] waddr,
    input      [31:0] wdata,
    output reg [31:0] rdata1,
    output reg [31:0] rdata2,
    input      [4 :0] test_addr,
    output reg [31:0] test_data
    );
    reg [31:0] rf[31:0];
    integer i;
     
    // three ported register file
    // read two ports combinationally
    // write third port on rising edge of clock
    // register 0 hardwired to 0

    always @(posedge clk or negedge resetn)
    begin
        if (!resetn) begin
            for (i = 0; i < 32; i = i + 1) begin
                rf[i] <= 32'b0;
            end
        end else if (wen) begin
            if (waddr != 5'b0) begin
                rf[waddr] <= wdata;
            end
        end
    end
     
    //˿�1
    always @(*)
    begin
        if (raddr1 == 5'b0)
            rdata1 = 32'b0;
        else
            rdata1 = rf[raddr1];
    end
    //˿2
    always @(*)
    begin
        if (raddr2 == 5'b0)
            rdata2 = 32'b0;
        else
            rdata2 = rf[raddr2];
    end
     //Զ˿ڣĴֵʾڴ
    always @(*)
    begin
        if (test_addr == 5'b0)
            test_data = 32'b0;
        else
            test_data = rf[test_addr];
    end
endmodule
