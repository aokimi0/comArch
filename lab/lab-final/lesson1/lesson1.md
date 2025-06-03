基础题：智能矩阵乘法优化挑战
已知两个矩阵：矩阵 A（大小 N × M），矩阵 B（大小 M × P）：

问题一： 请完成标准的矩阵乘算法，并支持浮点型输入，输出矩阵为 C = A × B，并对随机生成的的双进度浮点数矩阵输入，验证输出是否正确（N=1024，M=2048, P=512，N、M和P也可为任意的大数）；

问题二： 请采用至少一种方法加速以上矩阵运算算法，鼓励采用多种优化方法和混合优化方法；理论分析优化算法的性能提升，并可通过rocm-smi、hipprof、hipgdb等工具进行性能分析和检测，以及通过柱状图、折线图等图形化方式展示性能对比；

1. 基本优化思路
1.1 多线程并行化加速
   通过多线程并行化加速计算过程，可充分利用多核CPU的计算资源，可采用OpenMP（Open Multi-Processing）实现矩阵乘法计算优化。

1.2 子块并行优化
   子块并行（Block-wise Parallelization）是矩阵乘法中的一种优化技术，可通过局部计算降低内存访问延迟；为OpenMP或其他并行机制提供更细粒度、更均匀的工作划分，适用于大规模矩阵，特别适合在多核CPU上运行。

1.3 多进程并行优化
   使用MPI（Message Passing Interface）实现矩阵乘法的多进程优化，其核心思想是将大矩阵按行或块划分给不同进程，利用进程间通信协同完成整个计算。适用于分布式系统或多节点多核并行平台，能突破单机内存和计算瓶颈。

1.4 DCU加速计算
   通过国产高性能异构加速器、曙光DCU（Dawn Computing Unit），加速AI训练、推理和高性能计算场景。DCU与NVIDIA GPU特性类似，支持大规模并行计算，但通常通过HIP C++编程接口进行开发，兼容CUDA语义。

注：HIP（Heterogeneous-Compute Interface for Portability）是AMD公司在2016年提出的符合CUDA编程模型的、自由的、开源的C++编程接口和核函数语言。

1.4 其他计算优化方法或混合优化
   除了以上并行机制，还有多种计算优化方法和混合优化策略，可进一步提升矩阵计算效率。如内存访问优化，混合并行优化等。

2. 基本编译环境介绍
1）g++编译和执行文件C++程序

g++ -o outputfile sourcefile.cpp
./outputfile
2）MPI和OpenMP两种并行编程模型来编译和执行C++程序

mpic++ -fopenmp -o outputfile sourcefile.cpp
./outputfile
3）曙光DCU编译和执行执行C++程序

hipcc source_dcu.cpp -o outputfile_dcu
./outputfile_dcu
注：也可以采用其他的编译执行方法

3. 示例代码框架获取
   系统镜像基于Ubuntu，Ubuntu的基本常用命令均可使用。基本代码框架文件位于共享目录/public/SothisAI/learning_center/下：

lesson1_sourcefile.cpp和lesson1_sourcefile_dcu.cpp 为实训课程1的代码框架
lesson2_sourcefile_mlp_forward.cpp 为实训课程2的代码框架
lesson3_sourcefile_mlp.cpp 为实训课程3的代码框架