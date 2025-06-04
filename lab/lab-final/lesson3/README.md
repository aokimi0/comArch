# Lesson 3: 基于MLP的星地通信带宽预测与DCU训练加速

## 1. 项目概览

本项目旨在使用多层感知机（MLP）神经网络，根据提供的星地通信历史带宽数据（`data/starlink_bw.json`），预测未来的通信带宽。关键实现包括：

- **MLP模型**：包含输入层、一个ReLU激活的隐藏层和线性输出层，用于回归预测。
- **数据处理**：通过滑动窗口生成样本，并进行最小-最大归一化处理。
- **训练与推断**：实现MLP的前向传播、反向传播（计算梯度）和随机梯度下降（SGD）优化器进行模型训练。训练完成后，在测试集上评估模型性能。
- **DCU加速**：C++代码利用HIP API在DCU（曙光异构加速卡）上执行训练和推断的核心计算，包括数据传输（HtoD, DtoH）和核心计算（Kernels）。
- **性能分析**：生成详细的性能日志，并使用Python脚本 (`src/performance_analysis.py`) 解析日志，可视化训练损失、训练时间分解和预测结果。

## 2. 目录结构

```
lesson3/
├── data/
│   └── starlink_bw.json        # 原始带宽数据
├── log/
│   └── mlp_train_perf.log      # C++程序输出的性能日志
├── report/
│   ├── report.md                 # 实验报告
│   ├── lesson3_training_loss.png # 训练损失图
│   ├── lesson3_training_time_breakdown.png # 训练时间分解图
│   └── lesson3_predictions_vs_actual.png   # 预测对比图
├── src/
│   ├── mlp_train_dcu.cpp       # C++实现的MLP训练与推断（DCU加速）
│   └── performance_analysis.py # Python性能分析与可视化脚本
├── README.md                   # 本说明文件
└── run.sh                      # 编译和运行C++程序的便捷脚本
```

## 3. 环境要求

- **C++编译器**: 支持C++17标准。推荐使用 `hipcc` 或 `g++` (例如版本9.x或更高)。
- **Python环境**: Python 3.6+。
    - **依赖库**: `matplotlib`, `numpy`。可以通过pip安装：
      ```bash
      pip install matplotlib numpy
      ```
- **（可选）mplfonts**: 如果需要在matplotlib图表中使用中文字体，可以安装 `mplfonts` 并进行初始化（脚本中已包含英文备选方案）。
  ```bash
  pip install mplfonts
  ```
  然后在Python脚本中：
  ```python
  # from mplfonts.bin.cli import init
  # init()
  # matplotlib.rcParams['font.family'] = 'Source Han Sans CN'
  ```

## 4. 数据说明

- **`data/starlink_bw.json`**: 一个JSON文件，包含一个名为 `"bandwidth"` 的键，其值为一个表示历史带宽数据的浮点数数组。
- **数据预处理**: C++程序会加载此数据，使用滑动窗口（大小为10）生成输入特征和目标值，然后将数据归一化到 `[0, 1]` 区间进行训练。

## 5. 如何编译和运行

### 5.1. 使用 `run.sh` 脚本 (推荐)

`run.sh` 脚本会自动处理编译和运行步骤，并将输出重定向到日志文件。

在 `lesson3` 目录下执行：
```bash
./run.sh
```
该脚本会：
1. 优先使用 `hipcc` 编译 `src/mlp_train_dcu.cpp` 以在DCU上运行。
2. 如果 `hipcc` 不可用，则回退到使用 `g++` 编译（此时DCU特定代码将通过宏不被编译）。
3. 执行编译生成的可执行文件 `mlp_train_dcu`。
4. 所有标准输出和错误输出都将保存到 `log/mlp_train_perf.log`。

### 5.2. 手动编译和运行

#### 编译

根据你的环境选择以下命令之一，在 `lesson3` 目录下执行：

- **使用g++**:
  ```bash
  g++ src/mlp_train_dcu.cpp -o mlp_train_dcu -std=c++17 -I./src -O2
  ```
- **使用hipcc (用于DCU)**:
  ```bash
  hipcc src/mlp_train_dcu.cpp -o mlp_train_dcu --std=c++17 -I./src -O2
  ```

#### 运行C++程序

```bash
./mlp_train_dcu > log/mlp_train_perf.log 2>&1
```
或者，如果不重定向输出：
```bash
./mlp_train_dcu
```

### 5.3. 运行Python分析脚本

在C++程序运行并生成 `log/mlp_train_perf.log` 后，可以运行Python脚本来生成图表：

确保当前目录是 `lesson3` 或者能够正确解析脚本中的相对路径。
从 `lesson3` 目录运行：
```bash
/opt/venvs/base/bin/python src/performance_analysis.py 
```
或者如果你的Python环境已配置好，可以直接：
```bash
python3 src/performance_analysis.py
```
脚本会将生成的图表保存在 `report/` 目录下。

## 6. 输出说明

- **`log/mlp_train_perf.log`**: 包含了C++程序运行时的详细输出，包括数据加载信息、每个训练epoch的性能（损失、HtoD、Kernels、DtoH时间）以及测试阶段的性能（MSE、延迟、吞吐量）和部分预测样本。
- **`report/lesson3_training_loss.png`**: 显示训练过程中每个epoch的平均MSE损失变化。
- **`report/lesson3_training_time_breakdown.png`**: 可视化每个训练epoch中HtoD、Kernels和DtoH时间占比。
- **`report/lesson3_predictions_vs_actual.png`**: 展示测试集中部分样本的预测带宽值与实际带宽值的对比。
- **`report/report.md`**: 详细的实验报告，总结了模型设计、实验设置、性能结果和分析。

## 7. DCU加速说明

本项目的C++代码 (`mlp_train_dcu.cpp`) 利用HIP (Heterogeneous-compute Interface for Portability) API 来实现在DCU上的加速。
- **HIP API**: 用于管理设备内存、执行数据传输（主机与设备之间）以及启动在DCU上并行执行的计算内核。
- **内核函数**: 特定的计算密集型操作（如矩阵运算、激活函数等）被编写为HIP内核函数，在DCU设备上由大量线程并行执行。
- **性能计时**: 通过HIP事件API精确测量数据传输和内核执行的时间，以评估DCU加速的性能。

这种方法利用了DCU的大规模并行处理能力，以加速MLP模型的训练和推断过程。

## 8. 作者信息

- **姓名**: 廖望
- **学号**: 2210556 