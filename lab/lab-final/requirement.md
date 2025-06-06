
# 2025年全国大学生计算机系统能力大赛——智能计算创新设计赛（先导杯）南开大学校内赛培训核心内容
## 一、赛事基本信息
- **主题**：“智能计算，智享未来”
- **时间**：5月14日-6月4日
- **参赛对象**：南开大学在校本科生，个人形式参赛
- **主办方**：南开大学
- **培训时间**：2025年5月

## 二、赛程与奖项设置
### （一）赛程安排
- **赛题提交时间**：6月4日24:00前
- **提交方式**：发送至邮箱nkucs_org_2025s@163.com

### （二）奖项设置
|奖项|名额|奖励内容|
|----|----|----|
|特等奖|10名|奖金1000元、荣誉证书|
|一等奖|40名|奖金200元、荣誉证书|
|二等奖|70名|荣誉证书|
|优胜奖|若干|荣誉证书|

## 三、赛题设置与要求
### （一）基础题：智能矩阵乘法优化挑战（必做）
- **任务**：实现标准矩阵乘算法（\(C = A×B\)，\(N=1024\)，\(M=2048\)，\(P=512\)），并至少采用两种方法加速运算。
- **优化方向**：
  - 多线程并行化（OpenMP）
  - 子块并行优化（降低内存访问延迟）
  - 多进程并行（MPI）
  - DCU加速（曙光DCU，HIP C++接口）
- **工具支持**：
  - C++、DTK编程语言
  - 编译工具：g++、mpic++（支持OpenMP/MPI）、hipcc（DCU编译）
  - 性能分析工具：hipprof、hipgdb、rocm-smi

### （二）进阶题1：基于矩阵乘法的MLP实现与性能优化
- **任务**：实现三层MLP神经网络（输入层1024×10，隐藏层10×20，输出层20×5），使用DCU加速，完成前向传播与批处理。
- **关键步骤**：
  1. 内存初始化与数据准备（随机生成输入和权重矩阵）
  2. DCU内存分配与数据拷贝
  3. 隐藏层计算（矩阵乘法+ReLU激活）
  4. 输出层计算（矩阵乘法+偏置）
- **激活函数**：ReLU（\(f(x)=max(0, x)\)）

### （三）进阶题2：基于MLP的低轨卫星带宽预测优化
- **任务**：设计MLP网络实现LEO卫星下行带宽预测，支持前向传播、反向传播与梯度下降训练，使用DCU加速。
- **数据输入**：10维历史带宽序列（\(t_0-t_9\)），预测\(t_{10}\)时刻带宽。
- **关键流程**：
  1. **数据预处理**：滑动窗口构造样本，归一化处理
  2. **网络设计**：多层感知机结构，激活函数可选ReLU
  3. **反向传播**：基于链式法则计算梯度，更新权重与偏置
  4. **性能评估**：在DCU上测试训练与推理速度，评估预测精度
- **数据集**：starlink_bw.json（已上传至测试环境）

## 四、算力资源与平台
- **实训平台**：“光合开发者社区”网站，提供题目详情与代码框架。
- **测试平台**：“超算互联网”网站，支持算力资源与代码编辑。

## 五、其他说明
- **提交内容**：实验报告、实验代码。
- **评分规则**：基础题满分100分，进阶题每道20分（可累计至总分，最高100分）。