# 课程 2：DCU 加速的 MLP 前向传播

## 项目概览

本项目实现了多层感知器 (MLP) 的前向传播，通过使用 HIP C++ 在 DCU (Dawn Computing Unit) 上执行来加速。MLP 由一个输入层、一个带有 ReLU 激活的隐藏层和一个输出层组成。

**网络架构**：
- 输入：批大小 (B) = 1024，输入特征 (I) = 10
- 隐藏层：神经元 (H) = 20，激活函数：ReLU
  - 权重矩阵 W1：I x H (10 x 20)
  - 偏置向量 b1：H (20)
- 输出层：神经元 (O) = 5，激活函数：无
  - 权重矩阵 W2：H x O (20 x 5)
  - 偏置向量 b2：O (5)

所有矩阵和偏置都使用双精度浮点数，并随机初始化。
程序在 DCU 上执行前向传播，并根据 CPU 计算的参考结果验证输出。

## 目录结构

```
lesson2/
├── src/                                # 源代码目录
│   └── mlp_forward_dcu.cpp             # MLP 前向传播的 HIP C++ 实现
│   └── performance_analysis.py         # 用于可视化性能的 Python 脚本 (可选)
├── log/                                # 日志目录
│   └── mlp_forward_perf.log            # run.sh 运行的性能测试日志
├── report/                             # 报告目录
│   └── report.md                       # 详细实验报告
│   └── mlp_performance_breakdown.png   # 性能分解图 (由 Python 脚本生成)
│   └── mlp_cpu_vs_dcu_comparison.png # CPU 与 DCU 比较图 (由 Python 脚本生成)
├── run.sh                              # 编译和执行的自动化脚本
├── README.md                           # 本文件
└── lesson2.md                          # 课程 2 的任务要求
```

## 编译

编译用于 DCU 的 HIP C++ 代码：

```bash
cd lesson2/src
hipcc mlp_forward_dcu.cpp -o ../mlp_forward_dcu -O3
# 确保您在 lesson2 目录中，以便 run.sh 能够找到可执行文件
# 或者直接编译到 lesson2 目录：
# hipcc src/mlp_forward_dcu.cpp -o mlp_forward_dcu -O3
```

`run.sh` 脚本会自动处理编译。

## 执行

要编译并运行 MLP 前向传播，请从 `lesson2` 目录执行 `run.sh` 脚本：

```bash
cd lesson2
chmod +x run.sh
./run.sh
```

此脚本将：
1.  使用 `hipcc` 编译 `src/mlp_forward_dcu.cpp`。
2.  执行编译好的 `mlp_forward_dcu` 程序。
3.  C++ 程序将在 DCU 上执行 MLP 前向传播，测量不同阶段（内存复制、内核）的性能，对照 CPU 参考进行验证，并打印这些详细信息。
4.  所有标准输出，包括详细的时间和验证状态，将保存到 `log/mlp_forward_perf.log`。

如果 `hipcc` 不可用或编译失败，脚本将记录错误消息。

## 输出

主要输出文件是 `log/mlp_forward_perf.log`，其中包含：
- 编译状态。
- 执行时间的详细分解：
  - 主机到设备 (HtoD) 内存复制。
  - MatMul1 内核 (输入 @ W1)。
  - AddBias1 内核。
  - ReLU 内核。
  - MatMul2 内核 (隐藏 @ W2)。
  - AddBias2 内核。
  - 设备到主机 (DtoH) 内存复制。
- 内核总执行时间。
- MLP 前向传播总时间（包括 HtoD/DtoH 复制）。
- 对 CPU 参考计算的验证状态。

## 性能可视化 (可选)

Python 脚本 `src/performance_analysis.py` 可用于从日志文件中生成图表。
首先，确保安装了必要的库（例如，在您的 `/opt/venvs/base` 环境或本地 venv 中）：
```bash
# 假设 /opt/venvs/base/bin/python 是您的目标 python
/opt/venvs/base/bin/pip install matplotlib numpy
```
然后从 `lesson2/src` 目录运行脚本：
```bash
cd lesson2/src
/opt/venvs/base/bin/python performance_analysis.py
```
这将在 `lesson2/report/` 目录中生成 `.png` 文件。

## 要求
- HIP SDK（用于 `hipcc` 编译器和运行时）。
- 兼容的 DCU/GPU。
- 用于 CPU 参考部分的标准 C++ 编译器（通常是 g++）。
- 带有 Matplotlib 和 NumPy 的 Python 3.x（可选，用于可视化）。 