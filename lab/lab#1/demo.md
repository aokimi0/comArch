# EQ-Bench：大型语言模型情商能力评估

## 摘要

本文介绍EQ-Bench评测基准，一种专为评估大型语言模型(LLMs)情商能力设计的新型评测方法。该方法通过分析模型对对话中角色情感状态的预测能力，评估其理解复杂情感和社会互动的能力。研究结果表明，EQ-Bench能有效区分模型情感智能水平，其评分与主流基准高度相关(如MMLU相关系数达0.97)，且具有较强的防游戏化特性。研究还发现，引入自我修正机制平均可提升模型表现9.3%，尤其对初始表现较弱的模型效果显著。

## 1. 引言

随着LLMs在人工智能领域的飞速发展，评估方法日益多元化，但多数评测聚焦于传统认知能力，忽略了情感理解和社交互动能力。Paech (2024)[1]提出的EQ-Bench填补了这一空白，专注评估模型的情商能力，这被认为是AI系统与人类有效互动的关键。

## 2. 评测方法

EQ-Bench通过测量模型理解和预测对话情境中情感状态的能力，评估其情商水平。评测过程包含三个关键环节：场景呈现、情感评分和结果计算。

在测试中，模型首先接收一段由GPT-4生成的对话文本，这些对话通常包含情感冲突或紧张情境。随后，模型需要预测对话结束时特定角色可能体验的四种不同情绪的强度，评分范围为0-10分。为消除绝对评分差异影响，系统要求所有情绪评分总和为10，确保结果可比性。

### 2.1 测试流程

模型评测采用以下问题格式：

```plaintext
At the end of this dialogue, Jane would feel:  
Surprised:  
Confused:  
Angry:  
Forgiving:
```

为建立参考标准，研究团队使用以下提示词让GPT-4生成测试集标注：

```plaintext
Your task is to predict the likely emotional responses of a character in this dialogue:

Cecilia: You know, your words have power, Brandon. More than you might think.
Brandon: I'm well aware, Cecilia. It's a critic's job to wield them.
Cecilia: But do you understand the weight of them? The lives they can shatter?
Brandon: Art is not for the faint-hearted. If you can't handle the critique, you're in the wrong industry.
Cecilia: It's not about handling criticism, Brandon. It's about understanding the soul of the art. You dissect it like a cold, lifeless body on an autopsy table.

[End dialogue]

At the end of this dialogue, Brandon would feel...
Offended
Empathetic
Confident
Dismissive

Give each of these possible emotions a score from 0-10 for the relative intensity that they are likely to be feeling each. Then critique your answer by thinking it through step by step. Finally, give your revised scores.
```

### 2.2 评分机制

模型得分计算采用差距评估法：
- 单题得分 = 10 - 模型预测与参考答案的情感强度差距总和
- 所有评分经标准化处理，确保情感强度总和为10

研究还比较了两种评分模式的效果差异：
- **初始评分**：模型直接给出的第一次评分，反映直觉理解能力
- **修订评分**：模型经过自我审视和步骤式思考后的评分，体现反思调整能力

数据显示，引入自我修正机制平均可提升模型表现9.3%，特别对初始表现较弱的模型效果明显。这表明"think step-by-step"的思考方式能有效改善模型的情感理解能力。

## 3. 数据集设计

EQ-Bench数据集构建遵循以下原则：

- **数据来源**：所有测试对话由GPT-4生成，避免对特定模型或开发者偏向
- **多样性**：确保对话场景涵盖多元社交情境和情感类型
- **评分一致性**：采用统一标准评估，确保结果可比性

## 4. 实验结果与分析

![各模型EQ-Bench得分对比](./fig/2.png)

![初始评分与修订评分对比](./fig/1.png)

### 4.1 与现有基准的相关性

EQ-Bench与主流基准显示高度相关：
- 与MMLU基准相关系数达0.97
- 与HellaSwag等其他主流基准也呈强相关性

这表明EQ-Bench能有效反映模型的广义智能水平。

### 4.2 区分能力评估

实验证明EQ-Bench具备优异区分能力：
- 评分分布均匀，无明显"天花板"或"地板"效应
- 能清晰区分不同模型的情感处理能力差异

### 4.3 自我修正效果分析

"思考步骤式"自我审查显著提升模型表现：
- 平均性能提升9.3%
- 对初始低分模型提升更为明显
- 少数高表现模型可能出现略微回退现象

## 5. 讨论

EQ-Bench评估框架相比传统基准具有以下优势：

- **与用户感知一致性**：评分结果与用户对模型智能性感知高度吻合
- **防游戏化特性**：难以通过简单训练特定数据集提高得分
- **通用性**：适用于评估不同技术架构的模型

情感智能作为通用智能的核心组成，EQ-Bench提供了评估LLM全面能力的重要维度。

## 6. 结论

本研究证明EQ-Bench是评估语言模型情感智能的有效工具。通过测量模型理解和推理复杂情感的能力，它提供了评估LLM广义智能的新视角。结果表明情感理解是AI系统发展的关键能力，EQ-Bench在衡量模型情感智力方面展现出卓越的应用潜力。

未来研究可扩展EQ-Bench框架，纳入更广泛的文化背景和社交情境，进一步增强其作为通用评估标准的适用性。

## 参考文献

[1] Paech, S., et al. (2024). "EQ-Bench: An Emotional Intelligence Benchmark for Large Language Models."