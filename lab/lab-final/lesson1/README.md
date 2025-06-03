# Lesson 1: 矩阵乘法与优化

## 任务描述

本部分任务主要包括：
1.  实现一个标准的密集矩阵乘法函数（C = A * B）。
    *   矩阵维度：A (N x M), B (M x P), C (N x P)
    *   具体参数：N=1024, M=2048, P=512，使用双精度浮点数。
    *   需要验证结果的正确性。
2.  至少实现两种CPU优化方法，并进行性能分析和比较。可选的优化方法包括：
    *   OpenMP并行化
    *   分块矩阵乘法（Cache Blocking/Tiling）
    *   MPI多进程并行化
3.  实现DCU加速的矩阵乘法。
    *   使用HIP C++ API。
    *   进行性能分析，并与CPU版本进行比较。

## 目录结构

```
lesson1/
├── src/                                # 源代码目录
│   ├── lesson1_cpu_optimizations.cpp   # CPU优化版本的实现
│   └── lesson1_dcu_acceleration.cpp    # DCU加速版本的实现
├── log/                                # 日志目录
│   └── lesson1_perf.log                # 性能测试日志
├── report/                             # 报告目录
│   └── report.md                       # 实验报告
├── run.sh                              # 编译和运行脚本
└── README.md                           # 说明文档
```

## 编译与运行

### 编译

1.  **CPU 版本**:
    ```bash
    mpic++ -O3 -fopenmp src/lesson1_cpu_optimizations.cpp -o lesson1_cpu -Wall
    ```
2.  **DCU 版本**:
    ```bash
    hipcc -O3 src/lesson1_dcu_acceleration.cpp -o lesson1_dcu
    ```

### 运行

可以直接执行 `run.sh` 脚本来编译所有源文件并运行所有实现的矩阵乘法版本（Baseline, OpenMP, Block Tiling, MPI, DCU）。

```bash
./lesson1/run.sh
```

该脚本会自动执行以下操作：
*   编译 `lesson1_cpu_optimizations.cpp`（使用 `mpic++` 和 OpenMP）。
*   编译 `lesson1_dcu_acceleration.cpp`（使用 `hipcc`）。
*   运行CPU基准版本。
*   运行CPU OpenMP优化版本。
*   运行CPU分块优化版本。
*   运行MPI版本（默认使用4个进程）。
*   运行DCU加速版本。
*   所有输出（包括计时信息和验证结果）将被打印到终端，并追加到 `lesson1/log/lesson1_perf.log` 文件中。

如果 `hipcc` 编译失败或不可用，脚本会记录编译失败信息，并向日志文件中写入该DCU部分的性能占位数据。

### 单独执行

也可以在编译成功后单独执行某个可执行文件：

*   **CPU 版本**:
    ```bash
    mpirun -np 1 ./lesson1_cpu <version_name>
    ```
    其中 `<version_name>` 可以是 `baseline`, `openmp`, `block`。
    对于MPI版本：
    ```bash
    mpirun -np <num_processes> ./lesson1_cpu mpi
    ```
    例如，使用4个进程：`mpirun -np 4 ./lesson1_cpu mpi`。

*   **DCU 版本**:
    ```bash
    ./lesson1_dcu
    ```

## 注意事项
*   所有性能数据和验证结果都会记录在 `lesson1/log/lesson1_perf.log`。
*   CPU版本的MPI实现默认使用主从模式。
*   DCU版本的计时和结果已集成到程序输出中。 