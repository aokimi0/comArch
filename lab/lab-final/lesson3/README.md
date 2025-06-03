# Lesson 3: LEO卫星通信带宽预测 MLP 训练与推理 (DCU 加速)

## 任务描述

本部分任务旨在实现一个完整的 MLP (多层感知机) 模型，用于预测低地球轨道 (LEO) 卫星的通信带宽。任务包括数据预处理、MLP模型在DCU上的训练、以及在测试集上的推理与评估。

**数据集**: `lesson3/starlink_bw.json` (一个包含历史带宽数据的JSON数组)。

**MLP 结构** (可调整，以下为建议参数):
*   **输入层**: `INPUT_DIM` (例如 10)，表示使用过去10个时间点的数据预测下一个。
*   **隐藏层**: `HIDDEN_DIM` (例如 64)，ReLU 激活函数。
*   **输出层**: `OUTPUT_DIM` (1)，预测未来一个时间点的带宽。

**主要步骤**:
1.  **数据加载与预处理**: 
    *   从 JSON 文件加载带宽数据。
    *   对数据进行归一化处理 (Min-Max Scaling)。
    *   使用滑动窗口方法创建时间序列样本 (X, y)，其中 X 是历史数据，y 是待预测数据。
    *   划分训练集和测试集。
2.  **MLP 模型训练 (DCU)**:
    *   在 DCU 上实现 MLP 的前向传播、损失计算 (MSE)、反向传播和参数更新 (SGD)。
    *   权重和偏置在主机端初始化，然后拷贝到DCU。
    *   训练过程在 DCU 上完成，包括所有梯度计算和参数更新。
    *   定期记录训练损失。
3.  **模型推理与评估 (DCU)**:
    *   使用训练好的模型在测试集上进行预测。
    *   计算测试集上的 MSE 损失 (归一化和反归一化后)。
    *   展示部分预测结果与真实值的对比。

所有计算密集型操作（前向、反向、更新）均应在 DCU 上通过 HIP C++ 实现。

## 目录结构

```
lesson3/
├── src/
│   └── mlp_train_dcu.cpp         # MLP 训练与推理 DCU (HIP C++) 实现
├── data/
│   └── starlink_bw.json          # 带宽数据集
├── log/
│   └── mlp_train_perf.log        # 训练与测试日志
├── report/
│   └── report.md               # 实验报告
├── run.sh                      # 编译和运行脚本
└── README.md                   # 说明文档
```

## 编译与运行

### 编译

```bash
hipcc -O3 src/mlp_train_dcu.cpp -o mlp_train_dcu
```

### 运行

可以直接执行 `run.sh` 脚本来编译并运行 MLP 的训练和测试：

```bash
./lesson3/run.sh
```

该脚本会自动执行以下操作：
*   编译 `src/mlp_train_dcu.cpp`（使用 `hipcc`）。
*   运行 MLP 训练和测试程序。
*   程序内部会执行数据加载、预处理、模型训练（在DCU上）、模型测试（在DCU上）。
*   输出包括：
    *   数据加载和预处理信息。
    *   初始权重拷贝到DCU的耗时。
    *   每个训练周期的平均损失和执行时间 (细分为HtoD, Kernels, DtoH时间)。
    *   测试阶段的归一化和反归一化 MSE。
    *   部分测试样本的预测值与真实值对比。
*   所有输出将打印到终端，并追加到 `lesson3/log/mlp_train_perf.log` 文件中。

如果 `hipcc` 编译失败或不可用，脚本会记录编译失败信息，并向日志文件中写入该DCU部分的性能占位数据。

### 单独执行

编译成功后，也可以直接运行可执行文件：

```bash
./mlp_train_dcu
```

## 注意事项

*   程序内部使用固定的超参数，如 `INPUT_DIM`, `HIDDEN_DIM`, `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE` 等，这些参数在 `mlp_train_dcu.cpp` 文件头部定义。
*   所有与DCU相关的操作，包括内存分配、数据拷贝、核函数启动和同步，都使用HIP API。
*   训练过程中的性能计时（每周期 HtoD, Kernels, DtoH 时间）由程序内部通过HIP事件 API测量并打印。
*   CPU辅助函数用于数据预处理、结果验证以及在DCU无法工作时的完整CPU路径计算（但在正常DCU流程中，这些CPU函数仅用于生成参考或辅助，核心计算在DCU）。
*   最终的性能和训练日志记录在 `lesson3/log/mlp_train_perf.log`。 