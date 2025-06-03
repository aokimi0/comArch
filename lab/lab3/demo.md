计算机组成原理第三次实验报告	 
 
姓名：杨澍   学号：2311673  班级：张金老师
 
实验目的：	 根据《CPU设计实战》书中的第三章讲解，完成Lab2的三个实验，并撰写实验报告。
1、	针对任务一寄存器堆实验，完成仿真，在感想收获中思考并回答问题：为什么寄存器堆要设计成“两读一写”？
2、	针对任务二同步ram和异步ram实验，可以参考实验指导手册中的存储器实验，注意同步和异步需要分开建工程，然后仿真，在感想收获中分析同步ram和异步ram各自的特点和区别。
3、	针对任务三，重点介绍清楚发现bug、修改bug和验证的过程，在感想收获中总结使用 vivado调试的经验步骤。
 
实验过程：	 实验一：寄存器堆仿真	 
实验过程：	 首先将下发的代码导入到vivado中，并运行仿真，观察波形图。
![](./pic/1.png)
如上为得出的波形图。
其中 rdata1输出的数据为 raddr1 地址所储存的数据， rdata2输出的数据为 raddr2 地址所储存的数据。
当we为1时，寄存器堆会将 wdata 写入 waddr 地址。注意到对一个地址不能同时进行读写操作，会优先进行写操作。图中的 X 表示我们对一个地址进行写操作的同时进行读操作，返回值为不定值。
实验心得：	 CPU指令集中包含许多指令，是将两个数从内存中读取出来，进行操作后写入第三个内存地址中（如 ADD R1, R2, R3），两读一写的设计可以让这一类操作在一个时钟周期内完成。
同时，在执行其他指令时，两读一写的形式可以让读写操作同时进行，在执行完读操作后，不需要等下一个时钟周期就可以执行写操作。
 
实验二：同步内存与异步内存	 
同步ram	 
首先将实验代码导入vivado中，并按照书中的指导，从 IP catalog中导入同步内存，并命名为block_ram，便于调用。
之后，编译代码并进行仿真。
 ![](./pic/2.png)
仿真波形图如上。
同步ram只会在clock的上升沿进行读取操作，只有在wen和clock均为1时，会执行写入操作。注意到图中addr为00f1时，rdata为11223344，然而00f1地址此时的值应该为0，这是因为在上一个时钟周期内，addr为00f0，而我们又在00f0内写入了11223344，因此此时读出的实际上是00f0地址上的数据。同理，在下一个时钟周期，addr为00f0，rdata却是0，实际上是读取了00f1地址上的数据。
异步ram	 
首先将实验代码导入到vivado中，并按照书中指导，从IP catalog导入异步内存，并命名为 distributed_ram 。
之后，编译代码进行仿真。
![](./pic/3.png)
仿真波形图如上。
由于异步ram和同步ram使用了相同的testbench，因此在测试时的输入数据完全相同。为了便于比较，我们仍然选择与同步ram相同的一段数据进行分析。
注意到异步ram的读写操作与同步ram不完全相同，同步ram的读操作会在 clock的上升沿读
取数据，而异步ram则在接受到addr的输入后可以立刻输出地址上的值。观察图中的波形
图，addr为00f1时，rdata为0，addr为00f0时，raddr立刻变为11223344，也就是之前赋值到地址00f0的值。对于写操作，异步内存会在wen为1时执行。执行写入操作时无法再进行读取。
实验心得	 
同步ram由时钟进行统一控制，操作在时钟上升沿或者下降沿触发。容易控制时序，适合用于组成高速系统。
异步ram不接受时钟信号，由其他控制信号直接进行控制，例如本次实验中的wen，写使能信号。然而其不使用时钟信号的特性导致其不适合运用在告诉系统中，容易产生延迟。
同步RAM和异步RAM的本质区别在于时序控制方式，这一差异直接影响了它们的性能、复杂度和适用场景。
实验三：debug_test	 首先将代码导入vivado，编译并得到仿真波形图，便于进行debug。
![](./pic/4.png) 
得到的波形图如上。
注意到当前的主要问题在于num_a_g的输出值为不定值X，而num_csn的输出值为高阻值
Z。
首先根据书中关于高阻值Z的描述，可能是一个wire变量从来没有被赋值，于是我们分析所有num_csn出现的位置，看它应该在哪里被赋值，但实际上没有被赋值。
注意到在show_num中出现了语句 assign num_csn = 8'b01111111，因此错误应该发生在这之前。
注意到在show_sw函数调用show_num函数的过程中，将num_csn写成了num_scn，因此导致端口不统一，无法将值传入num_csn中，将这个错误修改后再运行，得到波形图如下：
![](./pic/5.png)
注意到图中num_csn的值变成了7f，而num_a_g的值仍然是不定值。
查询书中得知，X不定值可能是由 reg 变量没有被赋值或者多驱动导致的。因此我们观察 num_a_g在何处被赋值。观察到show_num函数的末尾，将nxt_a_g赋值给num_a_g，而 nxt_a_g的值是根据show_data选择得到。因此回去观察show_data的值，发现show <= 
~switch 一行被注释掉了，将这句话接触注释后，重新编译程序。
在运行仿真时，遇到了波形停止错误，根据vivado给出的 warning信息，发现在 show_data 的值为6时无法完成nxt_a_g的赋值，注意到nxt_a_g后面的三目表达式没有show_data为6时的值，因此我们要手动添加。
此时代码中仍然存在一个组合环路可能导致波形停止错误，代码中将keep_a_g的值赋值给 nxt_a_g，又将nxt_a_g+num_a_g的值赋值给keep_a_g。这一错误也有可能导致输出错误。
只需要将 ssign keep_a_g = nxt_a_g+num_a_g更改为assign keep_a_g = num_a_g 即可。
最后一个问题是越沿采样。在show_sw的always语句中，show_data_r = show_data采用了非
阻塞赋值，而应该使用非阻塞赋值，但使用非阻塞赋值可能导致程序执行出现问题，
show_data_r将被赋值为show_data被赋值之前的值，因此可以直接修改为 show_data_r = switch。
此时再编译运行仿真，得到波形图如下：
![](./pic/6.png)
最终修改后的完整代码如下：
module show_sw (
    input             clk,          
    input             resetn,     
    input      [3 :0] switch,    //input
    output     [7 :0] num_csn,   //new value   
    output     [6 :0] num_a_g,      
    output     [3 :0] led        //previous value
);
//1. get switch data //2. show switch data in digital number:
//   only show 0~9 //   if >=10, digital number keep old data.
//3. show previous switch data in led. //   can show any switch data.
reg [3:0] show_data; reg [3:0] show_data_r; reg [3:0] prev_data;
//new value always @(posedge clk) begin

    show_data <= ~switch;     show_data_r <= switch; end //previous value always @(posedge clk) begin
    if(!resetn)
    begin
        prev_data <= 4'd0;
    end
    else if(show_data_r != show_data)
    begin
        prev_data <= show_data_r;
    end end
//show led: previous value assign led = ~prev_data;
//show number: new value show_num u_show_num(
        .clk        (clk      ),
        .resetn     (resetn   ),
        .show_data  (show_data),         .num_csn    (num_csn  ),
        .num_a_g    (num_a_g  )
);
endmodule
//---------------------------{digital number}begin----------------------// module show_num (
    input             clk,          
    input             resetn,     
    input      [3 :0] show_data,
    output     [7 :0] num_csn,      
    output reg [6 :0] num_a_g      
);
//digital number display assign num_csn = 8'b0111_1111;
wire [6:0] nxt_a_g;
always @(posedge clk) begin
    if ( !resetn )
    begin
        num_a_g <= 7'b0000000;
    end
    else
    begin
        num_a_g <= nxt_a_g;
    end end
//keep unchange if show_dtaa>=10 wire [6:0] keep_a_g; assign     keep_a_g = num_a_g;
assign nxt_a_g = show_data==4'd0 ? 7'b1111110 :   //0                  show_data==4'd1 ? 7'b0110000 :   //1
                 show_data==4'd2 ? 7'b1101101 :   //2
                 show_data==4'd3 ? 7'b1111001 :   //3
                 show_data==4'd4 ? 7'b0110011 :   //4
                 show_data==4'd5 ? 7'b1011011 :   //5                  show_data==4'd6 ? 7'b1011111 :
                 show_data==4'd7 ? 7'b1110000 :   //7
                 show_data==4'd8 ? 7'b1111111 :   //8
                 show_data==4'd9 ? 7'b1111011 :   //9                                    keep_a_g   ; endmodule //----------------------------{digital number}end------------------
------//
实验心得	 
在使用vivado进行调试时，一定要善用波形图进行分析，观察波形图上出现的错误，再结合可能导致错误的原因去分析代码，这样才能更块的发现错误。
同时在读代码的过程中也一定要细心，根据波形图分析只能将错误锁定在一个范围，有时打错一个字母就可能出现bug，细心分析代码才能完成debug。
