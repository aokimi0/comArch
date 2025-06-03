# Lesson 2 实验报告：MLP 前向传播 DCU 加速 (廖望 2210556)

## 1. 引言

本项目旨在设计并实现一个简单的三层多层感知机 (MLP) 的前向传播过程，并重点利用 DCU (Data Center Unit, 使用 HIP C++) 进行计算加速。MLP 是深度学习中的基础模型，其前向传播涉及一系列的矩阵运算和激活函数计算，是典型的可并行化计算密集型任务。

**实验目标**：
*   在 DCU 上实现 MLP 前向传播的各个计算层（矩阵乘法、偏置加法、ReLU激活）。
*   管理 DCU 内存，完成主机与设备间的数据传输。
*   在 CPU 端实现相同的 MLP 前向传播逻辑作为参考，用于验证 DCU 计算结果的正确性。
*   测量并分析 DCU 实现的性能，包括数据传输时间和各核函数的执行时间。

## 2. 实验环境

*   **主机操作系统**: WSL Ubuntu 24.04 (Kernel: Linux 5.15.167.4-microsoft-standard-WSL2)
*   **DPU容器操作系统**: Ubuntu 20.04.1 LTS (Focal Fossa), running on DPU/DCU
*   **主机CPU**: Intel(R) Core(TM) i7-10700K CPU @ 3.80GHz (8 Cores, 16 Threads)
*   **DCU/DPU**: 
    *   **加速卡型号**: 曙光异构加速卡 (DPU)
    *   **板载CPU**: HYGON C86 7185 32-core Processor @ 2.0GHz (32 Cores, 32 Threads)
    *   **显存**: 16GB HBM2
*   **编译器**: hipcc (ROCm 5.x on DPU container)
*   **主要库**: HIP Runtime

## 3. MLP 模型结构与算法设计

### 3.1. MLP 结构

*   **输入层 (X)**: 维度 `BATCH x I` (1024 x 10)
*   **隐藏层 (H1)**:
    *   权重 `W1`: `I x H` (10 x 20)
    *   偏置 `B1`: `1 x H` (1 x 20) (实际存储为 `H`)
    *   计算: `Z1 = X * W1 + B1`
    *   激活: `H1 = ReLU(Z1)`
*   **输出层 (Y)**:
    *   权重 `W2`: `H x O` (20 x 5)
    *   偏置 `B2`: `1 x O` (1 x 5) (实际存储为 `O`)
    *   计算: `Z2 = H1 * W2 + B2`
    *   输出: `Y = Z2` (线性输出)
*   **数据类型**: 所有数据（输入、权重、偏置）均为双精度浮点数 (`double`)。

### 3.2. DCU 实现思路

1.  **数据初始化与内存分配**:
    *   在主机端随机初始化输入数据 X、权重 W1, B1, W2, B2。
    *   使用 `hipMalloc` 在DCU设备端为 X, W1, B1, W2, B2 以及中间结果 Z1 (复用为 H1), Z2 (复用为 Y) 分配内存。
2.  **数据传输 (Host-to-Device, HtoD)**:
    *   使用 `hipMemcpy` 将初始化好的 X, W1, B1, W2, B2 从主机内存拷贝到DCU设备内存。
3.  **核函数设计与执行**: 为 MLP 的每个主要计算步骤设计专门的 HIP 核函数：
    *   `matmul_kernel`: 通用的矩阵乘法核函数。用于计算 `X * W1` 和 `H1 * W2`。
        *   每个线程计算结果矩阵中的一个元素。
    *   `add_bias_kernel`: 矩阵加偏置核函数。用于计算 `Z1 + B1` 和 `Z2 + B2`。
        *   每个线程处理结果矩阵的一个元素，加上对应列的偏置值。
    *   `relu_kernel`: ReLU 激活函数核函数。用于计算 `ReLU(Z1)`。
        *   每个线程处理一个元素，应用 `fmax(0.0, element)`。
    *   所有核函数均使用 `hipLaunchKernelGGL` 启动，并配置合适的线程块和网格维度。
    *   在关键核函数执行前后使用 `hipDeviceSynchronize()` 确保执行完成（特别是在依赖前一核函数结果时，或在计时场景下）。
4.  **数据传输 (Device-to-Host, DtoH)**:
    *   使用 `hipMemcpy` 将最终的输出结果 Y 从DCU设备内存拷贝回主机内存。
5.  **资源释放**:
    *   使用 `hipFree` 释放所有在DCU设备端分配的内存。
    *   使用 `hipEventDestroy` 销毁用于计时的事件对象。

### 3.3. CPU 参考实现

在主机端使用 C++ 实现与DCU路径完全相同的 MLP 前向传播逻辑，包括：
*   `matmul_cpu`: CPU版本的矩阵乘法。
*   `add_bias_cpu`: CPU版本的偏置加法。
*   `relu_cpu`: CPU版本的ReLU激活。
该CPU计算结果 (`h_Y_cpu_ref`) 将用于验证DCU计算结果 (`h_Y_dcu`) 的正确性。

### 3.4. 结果验证

比较DCU计算得到的输出 `h_Y_dcu`