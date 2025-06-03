# Lesson 2: MLP 前向传播 DCU 加速

## 任务描述

本部分任务旨在实现一个简单的多层感知机 (MLP) 的前向传播过程，并利用 DCU (HIP C++) 进行加速。

MLP 结构如下：
*   **输入层**: 10个特征 (I=10)
*   **隐藏层**: 20个神经元 (H=20)，激活函数为 ReLU
*   **输出层**: 5个神经元 (O=5)，无激活函数（或视为线性激活）
*   **批量大小**: B=1024

前向传播计算过程：
1.  `Z1 = X * W1 + B1`  (X: BxI, W1: IxH, B1: 1xH, Z1: BxH)
2.  `H1 = ReLU(Z1)`    (H1: BxH)
3.  `Z2 = H1 * W2 + B2` (W2: HxO, B2: 1xO, Z2: BxO)
4.  `Y = Z2`            (Y: BxO，最终输出)

所有权重 (W1, B1, W2, B2) 和输入数据 (X) 均为双精度浮点数。

## 目录结构

```
lesson2/
├── src/
│   └── mlp_forward_dcu.cpp     # MLP 前向传播 DCU (HIP C++) 实现
├── log/
│   └── mlp_forward_perf.log    # 性能测试日志
├── report/
│   └── report.md               # 实验报告
├── run.sh                      # 编译和运行脚本
└── README.md                   # 说明文档
```

## 编译与运行

### 编译

```bash
hipcc -O3 src/mlp_forward_dcu.cpp -o mlp_forward_dcu
```

### 运行

可以直接执行 `run.sh` 脚本来编译并运行 MLP 前向传播的 DCU 版本：

```bash
./lesson2/run.sh
```

该脚本会自动执行以下操作：
*   编译 `src/mlp_forward_dcu.cpp`（使用 `hipcc`）。
*   运行 DCU 加速的 MLP 前向传播程序。
*   程序内部会首先在 CPU 上计算参考结果，然后执行 DCU 计算。
*   输出包括各个计算阶段（数据拷贝、核函数）的计时信息、总执行时间以及 DCU 计算结果与 CPU 参考结果的验证。
*   所有输出将打印到终端，并追加到 `lesson2/log/mlp_forward_perf.log` 文件中。

如果 `hipcc` 编译失败或不可用，脚本会记录编译失败信息，并向日志文件中写入该DCU部分的性能占位数据。

### 单独执行

编译成功后，也可以直接运行可执行文件：

```bash
./mlp_forward_dcu
```

## 注意事项

*   程序内部会随机初始化输入数据 X、权重 W1, B1, W2, B2。
*   DCU 版本的计时信息（包括 HtoD 拷贝、各个核函数执行时间、DtoH 拷贝、总核函数时间、总MLP前向传播时间）由程序内部通过 HIP 事件 API 测量并打印。
*   最终输出结果会与 CPU 计算的参考结果进行验证。
*   所有性能数据和验证结果都会记录在 `lesson2/log/mlp_forward_perf.log`。 