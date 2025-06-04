# Lesson 1: 矩阵乘法与优化

## 项目概述

本项目实现了标准密集矩阵乘法(C = A × B)的多种优化方案，包括CPU多核优化和DCU异构加速，旨在探索不同硬件架构下的性能优化策略。

**实验规模**: A(1024×2048), B(2048×512), C(1024×512), 双精度浮点数

## 核心功能

### CPU优化实现
1. **基准版本**: 标准三重循环实现
2. **OpenMP并行化**: 利用多核CPU并行计算
3. **分块矩阵乘法**: 改善缓存局部性的分块优化
4. **MPI多进程**: 分布式内存并行计算

### DCU加速实现
- **HIP C++核函数**: 异构并行计算加速
- **内存管理优化**: 高效的数据传输策略
- **线程块配置**: 16×16线程块优化设计

## 目录结构

```
lesson1/
├── src/                                    # 源代码目录
│   ├── lesson1_cpu_optimizations.cpp       # CPU优化版本实现
│   ├── lesson1_dcu_acceleration.cpp        # DCU加速版本实现
│   └── performance_analysis.py             # 性能分析可视化脚本
├── log/                                    # 性能测试日志
│   └── lesson1_perf.log                    # 详细性能数据记录
├── report/                                 # 实验报告目录
│   ├── report.md                           # 完整实验报告
│   ├── performance_analysis.png            # 性能分析图表
│   └── scalability_analysis.png            # 可扩展性分析图表
├── run.sh                                  # 自动化测试脚本
├── README.md                               # 项目说明文档
└── lesson1.md                              # 任务要求说明
```

## 快速开始

### 环境要求
- **编译器**: GCC 9.0+ (支持C++17)
- **并行库**: OpenMP 4.0+
- **分布式计算**: Open MPI 3.0+ (可选)
- **异构计算**: HIP/ROCm 5.0+ (DCU加速，可选)
- **Python环境**: Python 3.8+ (图表生成，可选)

### 一键运行
```bash
cd lesson1
chmod +x run.sh
./run.sh
```

该脚本将自动:
- 检测编译环境和依赖
- 编译所有版本的实现
- 运行性能测试
- 生成详细的性能日志

### 手动编译与运行

#### CPU版本编译
```bash
# 完整功能版本 (需要MPI)
mpic++ -O3 -fopenmp -DENABLE_MPI src/lesson1_cpu_optimizations.cpp -o lesson1_cpu -Wall

# 仅OpenMP版本 (无需MPI)
g++ -O3 -fopenmp src/lesson1_cpu_optimizations.cpp -o lesson1_cpu -Wall

# 基础版本 (无并行库)
g++ -O3 src/lesson1_cpu_optimizations.cpp -o lesson1_cpu -Wall
```

#### 运行CPU测试
```bash
# 基准版本
./lesson1_cpu baseline

# OpenMP优化
./lesson1_cpu openmp

# 分块优化
./lesson1_cpu block

# MPI多进程 (需要MPI编译)
mpirun -np 4 ./lesson1_cpu mpi
```

#### DCU版本编译与运行
```bash
# 编译DCU版本
hipcc -O3 src/lesson1_dcu_acceleration.cpp -o lesson1_dcu

# 运行DCU加速测试
./lesson1_dcu
```

## 性能分析

### 生成可视化图表
```bash
cd src
pip install matplotlib numpy mplfonts
python performance_analysis.py
```

生成的图表包括:
- 执行时间对比 (对数坐标)
- 加速比分析
- GFLOPS性能对比
- CPU vs DCU效率对比
- 并行可扩展性分析

### 关键性能指标

| 实现方案 | 执行时间 | 加速比 | GFLOPS | 并行效率 |
|---------|---------|--------|--------|----------|
| CPU基准 | 28.75s | 1.00x | 0.75 | - |
| OpenMP | 3.99s | 7.22x | 5.40 | 45.1% |
| 分块优化 | 4.10s | 7.01x | 5.26 | 43.8% |
| MPI(4进程) | 7.81s | 3.68x | 2.76 | 92.0% |
| **DCU加速** | **0.125s** | **230x** | **173.7** | **-** |

## 技术特点

### CPU优化技术
1. **多线程并行**: OpenMP指令级并行化
2. **内存优化**: 分块算法改善缓存局部性
3. **分布式计算**: MPI主从模式并行
4. **混合策略**: OpenMP+分块的组合优化

### DCU加速技术
1. **SIMT并行**: 大规模线程并行计算
2. **内存合并**: 优化全局内存访问模式
3. **数据传输**: 高效的Host-Device数据管理
4. **核函数优化**: 针对矩阵乘法的专用核函数

### 代码特性
- **跨平台兼容**: 支持有/无MPI环境编译
- **参数可配置**: 矩阵维度和算法参数可调
- **结果验证**: 所有实现都有正确性验证
- **性能监控**: 详细的计时和性能指标收集

## 实验结果

### 主要发现
1. **DCU加速效果显著**: 相比CPU基准版本实现230倍加速
2. **OpenMP扩展性良好**: 16线程实现7.22倍加速比
3. **分块优化有效**: 在改善缓存利用率方面表现出色
4. **MPI适用分布式**: 为大规模并行计算提供基础

### 性能瓶颈分析
1. **CPU内存带宽**: 限制了OpenMP的进一步扩展
2. **MPI通信开销**: 影响了多进程的加速效果
3. **DCU数据传输**: 占总时间约23%，有优化空间

## 扩展应用

### 适用场景
- 科学计算和数值仿真
- 机器学习模型训练
- 信号处理和图像处理
- 高性能计算基准测试

### 优化方向
- 集成BLAS库优化实现
- 支持混合精度计算
- 实现多GPU/DCU并行
- 开发自适应参数调优

## 参考资料

- [OpenMP官方文档](https://www.openmp.org/)
- [MPI标准规范](https://www.mpi-forum.org/)
- [HIP编程指南](https://rocm.docs.amd.com/projects/HIP/en/latest/)
- [矩阵乘法优化技术综述](https://dl.acm.org/doi/10.1145/3291035)

## 作者信息

**实验者**: 廖望 (学号: 2210556)  
**课程**: 计算机体系结构实验  
**时间**: 2025年6月  
**环境**: WSL Ubuntu 24.04, 曙光DCU异构加速平台 