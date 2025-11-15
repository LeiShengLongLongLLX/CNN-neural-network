# 神经网络介绍

## 一、神经网络架构总概

### 1.1 卷积神经网络（CNN）

#### 1. 特点

* 通过卷积核提取局部空间特征。
* 参数共享，减少计算量。
* 层级结构逐渐捕捉复杂特征。
* 适合处理有空间结构的数据（图像、视频）。

#### 2. 应用场景

* 图像分类（ImageNet等）
* 目标检测（Faster R-CNN, YOLO）
* 图像分割（U-Net）
* 视频分析
* 医学影像诊断

#### 3. 代表论文

* LeCun et al., *“Gradient-Based Learning Applied to Document Recognition”*, 1998.
* Krizhevsky et al., *“ImageNet Classification with Deep Convolutional Neural Networks”*, 2012 (AlexNet).
* He et al., *“Deep Residual Learning for Image Recognition”*, 2015 (ResNet).

#### 4. 代码资源

* PyTorch官方示例：torchvision.models
* TensorFlow官方示例：tf.keras.applications

---

### 1.2 循环神经网络（RNN）

#### 1. 特点

* 适合序列数据，能够处理时间依赖。
* LSTM和GRU变体解决梯度消失问题。
* 训练相对较慢，不利于长序列并行。

#### 2. 应用场景

* 语音识别
* 机器翻译
* 时间序列预测
* 文本生成

#### 3. 代表论文

* Hochreiter & Schmidhuber, *“Long Short-Term Memory”*, 1997.
* Cho et al., *“Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation”*, 2014 (GRU).

#### 4. 代码资源

* TensorFlow RNN教程
* PyTorch RNN示例

---

### 1.3 Transformer

#### 1. 特点

* 基于自注意力机制，能捕获全局依赖。
* 高度并行，训练效率高。
* 需要大量数据和计算资源。

#### 2. 应用场景

* 机器翻译
* 文本理解和生成
* 图像分类（ViT）
* 语音处理

#### 3. 代表论文

* Vaswani et al., *“Attention Is All You Need”*, 2017.
* Devlin et al., *“BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding”*, 2018.
* Dosovitskiy et al., *“An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale”*, 2020 (ViT).

#### 4. 代码资源

* Hugging Face Transformers库
* TensorFlow官方Transformer示例

---

### 1.4 图神经网络（GNN）

#### 1. 特点

* 专为图结构设计，能够聚合邻居节点信息。
* 多种变体（GCN, GAT等）。
* 计算复杂度高，扩展性是挑战。

#### 2. 应用场景

* 社交网络分析
* 知识图谱推理
* 化学分子性质预测
* 推荐系统

#### 3. 代表论文

* Kipf & Welling, *“Semi-Supervised Classification with Graph Convolutional Networks”*, 2017.
* Veličković et al., *“Graph Attention Networks”*, 2018.

#### 4. 代码资源

* PyTorch Geometric库
* Deep Graph Library (DGL)

---

### 1.5 生成对抗网络（GAN）

#### 1. 特点

* 生成模型，通过对抗训练提高生成样本质量。
* 训练不稳定，模式崩溃问题。
* 可生成高质量图像、音频等。

#### 2. 应用场景

* 图像生成和编辑
* 超分辨率
* 数据增强
* 风格迁移

#### 3. 代表论文

* Goodfellow et al., *“Generative Adversarial Nets”*, 2014.
* Radford et al., *“Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks”*, 2015 (DCGAN).
* Zhu et al., *“Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks”*, 2017 (CycleGAN).

#### 4. 代码资源

* TensorFlow GAN库
* PyTorch GAN教程

---

### 1.6 液态神经网络（Liquid Neural Networks）

#### 1. 特点

* 受生物神经系统启发，利用动态系统状态。
* 适应非平稳时间序列，动态调整。
* 理论和应用尚处早期。

#### 2. 应用场景

* 动态环境控制
* 时序数据处理（复杂非线性动力学）
* 神经形态计算

#### 3. 代表论文

* Pandarinath et al., *“Inferring single-trial neural population dynamics using sequential auto-encoders”*, 2018.
* Liquid Time-constant Networks 相关工作。

#### 4. 代码资源

* 研究代码为主，无统一库。

---

### 1.7 KAN 神经网络（Kolmogorov–Arnold Network）

#### 1. 特点

* 基于 Kolmogorov–Arnold 表示定理，强调由一维函数组合表示多维函数。
* 权重不是标量，而是可学习的一维函数（如B样条、分段函数等）。
* 可解释性强，每条边都是一个显式函数。
* 参数量少，拟合能力强，适合小规模与科学计算任务。

#### 2. 应用场景

* 数学建模与科学计算（微分方程、物理系统拟合）
* 小规模结构化数据任务
* 强可解释性模型场景

#### 3. 代表论文

* Liu et al., *“Kan: Kolmogorov–Arnold Networks”*, 2024.

#### 4. 代码资源

* 官方原型代码（基于PyTorch）可在论文附录或社区仓库找到。

---

### 1.8 总结对比

| 网络          | 优点              | 缺点             | 适用场景          | 代码资源链接简述                     |
| ----------- | --------------- | -------------- | ------------- | ---------------------------- |
| CNN         | 局部特征提取强，参数少，训练快 | 长距离依赖弱，需大量标注数据 | 图像、视频分析       | torchvision/models, TF.keras |
| RNN         | 适合序列建模，捕获时间依赖   | 训练慢，长序列依赖弱     | 语音、文本、时序      | TensorFlow RNN, PyTorch      |
| Transformer | 全局依赖，自注意力，高并行   | 资源消耗大，需大数据     | NLP、图像、文本生成   | Hugging Face, TensorFlow     |
| GNN         | 处理图结构，关系建模强     | 计算复杂，扩展性有限     | 社交、知识图谱、分子    | PyTorch Geometric, DGL       |
| GAN         | 高质量生成，丰富多样      | 训练不稳定，模式崩溃     | 图像生成、数据增强     | TF-GAN, PyTorch GAN教程        |
| 液态神经网络      | 动态适应能力强，受生物启发   | 理论和实践尚早，训练复杂   | 动态环境时序，神经形态计算 | 研究代码                         |
| KAN         | 可解释性强，函数级连接，参数少 | 不适合大规模模型，计算代价高 | 科学计算、小规模数据    | 研究代码与社区实现                    |


---

## 二、CNN（Convolutional Neural Network）卷积神经网络

### 2.1 CNN是什么？

*   **定义**：卷积神经网络是一种专为处理**网格状数据**（如图像、视频、音频）而设计的深度学习模型。
*   **核心思想**：通过**卷积** 操作来自动、高效地学习数据的空间层次特征。
*   **与传统神经网络的对比**：
    *   **传统神经网络（如全连接网络）**：将输入数据（如图像）展平为一维向量，会完全丢失空间信息，且参数数量巨大，容易过拟合。
    *   **CNN**：保留了数据的空间结构，通过**局部连接**和**权值共享** 大大减少了参数数量，使其更高效、更易于训练。

### 2.2 为什么CNN特别适合图像处理？

1.  **局部相关性**：图像中一个像素与其周围像素的关系最紧密，与遥远像素的关系较弱。CNN的卷积操作正是关注局部区域。
2.  **平移不变性**：无论一只猫在图像的左上角还是右下角，它都是一只猫。CNN通过池化操作和层级结构，使得网络对目标的位置变化不敏感。
3.  **尺度不变性**：通过多层卷积和池化，CNN可以从底层边缘、纹理，到中层部分器官，再到高层整体物体，逐步构建出对图像的尺度鲁棒性理解。

---

### 2.3 CNN的核心组件

一个典型的CNN由以下几部分组成：

#### 2.3.1 卷积层 - 特征提取的核心

*   **目的**：使用**滤波器（或称为卷积核）** 在输入数据上滑动，提取局部特征（如边缘、角点、颜色块）。
*   **关键概念**：
    *   **滤波器**：一个小尺寸的权重矩阵（如3x3, 5x5）。不同的滤波器用于提取不同的特征。
    *   **感受野**：滤波器在输入图像上每次覆盖的区域大小。
    *   **步长**：滤波器每次移动的像素数。步长越大，输出特征图尺寸越小。
    *   **填充**：在输入图像边缘填充一圈像素（通常用0填充）。目的是为了控制输出特征图的尺寸。
    *   **深度**：一个卷积层通常使用多个滤波器，每个滤波器会产生一个**特征图**。所有这些特征图堆叠起来，就构成了该卷积层的输出。
*   **工作机制**：滤波器在输入上滑动，在每个位置进行**点乘** 求和，再加上一个偏置项，最终生成特征图。

#### 2.3.2 激活函数 - 引入非线性

*   **目的**：为网络引入非线性因素，使其能够学习并模拟复杂的非线性关系。没有它，多层网络就等价于一个单层线性模型。
*   **常用函数**：
    *   **ReLU（修正线性单元）**：`f(x) = max(0, x)`。目前最常用，因为它能有效缓解梯度消失问题，且计算简单。
    *   **Sigmoid / Tanh**：在早期使用，现在多用于输出层（如二分类）。

#### 2.3.3 池化层 - 降维和保持平移不变性

*   **目的**：对特征图进行**下采样**，减少数据尺寸和参数量，防止过拟合，同时扩大后续卷积层的感受野，并赋予网络一定的平移不变性。
*   **特点**：没有需要学习的参数。
*   **常用方法**：
    *   **最大池化**：取池化窗口内的最大值。效果最好，最常用。
    *   **平均池化**：取池化窗口内的平均值。
*   **工作机制**：类似卷积，有一个窗口和步长，在特征图上滑动，但执行的是最大或平均操作。

#### 2.3.4. 全连接层 - 最终分类

*   **目的**：位于网络的末端，将前面学习到的所有分布式特征映射到最终的样本标签空间（进行分类或回归）。
*   **工作机制**：将最后一层卷积或池化输出的多维特征图**展平** 成一个一维向量，然后像传统神经网络一样进行连接。通常最后会接一个**Softmax** 函数，输出每个类别的概率。

**总结表**

| 组件 | 主要功能 | 关键优势 |
| :--- | :--- | :--- |
| **卷积层** | 提取局部空间特征 | 权值共享、局部连接，大幅减少参数 |
| **激活函数** | 引入非线性 | 使网络能拟合复杂函数（ReLU最常用） |
| **池化层** | 下采样，压缩特征图 | 降维、防止过拟合、提供平移不变性 |
| **全连接层** | 将特征映射到样本标签 | 完成最终的分类或回归任务 |

---

好的，我们在您原有笔记的基础上，保持格式和内容不变，直接添加YOLO和MobileNet的内容。

---

### 2.4 经典的CNN网络架构

*   **LeNet-5 (1998)**：CNN的开山之作，用于手写数字识别，结构为：`卷积 -> 池化 -> 卷积 -> 池化 -> 全连接 -> 输出`。
*   **AlexNet (2012)**：在ImageNet大赛中一战成名，真正开启了深度学习热潮。它更深，使用了ReLU、Dropout等技术。
*   **VGGNet (2014)**：探索了网络深度，通过堆叠小的3x3卷积核来替代大的卷积核，结构非常规整。
*   **GoogLeNet (2014)**：引入了**Inception模块**，在增加网络深度和宽度的同时，减少了参数量。
*   **ResNet (2015)**：提出了**残差块** 和**跳跃连接**，解决了极深网络的梯度消失和退化问题，使得构建上百甚至上千层的网络成为可能。
*   **MobileNet (2017)**：引入了**深度可分离卷积**为核心构建块，专为移动和嵌入式设备等资源受限环境设计，在速度和体积上实现了极致优化。
*   **YOLO (2016)**：**You Only Look Once** 的缩写，是目标检测领域的革命性框架。它将检测任务视为一个回归问题，实现了端到端的训练和极快的推理速度，催生了一系列实时检测应用。其核心思想是使用CNN骨干网络进行特征提取，再通过检测头直接预测边界框和类别概率。

---

### 2.5 CNN的工作流程总结

1.  **输入**：原始图像。
2.  **特征提取（前向传播）**：
    *   图像经过多个 **“卷积 -> 激活 -> 池化”** 的模块。
    *   底层网络学习到**低级特征**（边缘、颜色）。
    *   中层网络学习到**中级特征**（纹理、部分器官）。
    *   高层网络学习到**高级特征**（整体物体，如“猫 脸”、“车轮”）。
3.  **分类**：高级特征被送入**全连接层**，最终通过**Softmax**输出每个类别的概率。
4.  **训练（反向传播）**：
    *   计算预测结果与真实标签的**损失**。
    *   使用**优化器（如SGD, Adam）**，通过**反向传播**算法，将损失误差从后向前传递，并更新网络中所有滤波器和权重参数。
    *   重复此过程，直到模型收敛。

---

### 2.6 CNN的应用领域（远超图像）

*   **计算机视觉**：图像分类、目标检测、图像分割、人脸识别。
*   **自然语言处理**：文本分类、情感分析、机器翻译（通过一维卷积处理序列）。
*   **游戏与机器人**：AlphaGo下围棋，机器人导航。
*   **医疗**：医学影像分析（CT、MRI片子诊断）。

---

## 三、RNN（Recurrent Neural Network）循环神经网络

### 3.1 什么是 RNN？

* **RNN（Recurrent Neural Network）循环神经网络** 是一种专门处理序列数据（时间序列、文本序列等）的神经网络。
* 它能够通过“循环结构”将之前时刻的信息（记忆）传递到当前时刻，具备记忆和上下文关联能力。

---

### 3.2 RNN 的基本结构

* RNN 的关键是**隐藏状态（hidden state）**，通常用 $( h_t )$ 表示，第 $( t )$ 个时间步的隐藏状态。

* 每个时间步的输入是当前输入 $( x_t )$ 和上一时刻隐藏状态 $( h_{t-1} )$，输出当前隐藏状态 $( h_t )$：

  $$
  h_t = \sigma\left(W_{hx} x_t + W_{hh} h_{t-1} + b_h\right)
  $$

* 输出层（如果有）根据 $( h_t )$ 计算输出：

  $$
  y_t = \phi\left(W_{yh} h_t + b_y\right)
  $$

* 其中，$( W_{hx}, W_{hh}, W_{yh} )$ 是权重矩阵，$( b_h, b_y )$ 是偏置，$( \sigma )$ 和 $( \phi )$ 是激活函数。

---

### 3.3 RNN 的工作流程

1. 输入序列 $( x_1, x_2, \dots, x_T )$ 一步步送入网络。
2. 网络根据当前输入和上一时刻隐藏状态更新新的隐藏状态。
3. 每一步都产生输出（或只在最后一步输出），隐藏状态带有之前所有时刻的信息。
4. 最后通过 **误差反向传播算法（BPTT）** 训练。

---

### 3.4 RNN 的特点

| 特点            | 说明                          |
| ------------- | --------------------------- |
| **处理序列数据**    | 能够处理变长输入，适合时间序列、文本等         |
| **有记忆能力**     | 通过隐藏状态存储之前信息，捕捉上下文依赖        |
| **参数共享**      | 各时间步共享同一组参数，减少模型参数量         |
| **梯度消失/爆炸问题** | 长序列训练时容易出现梯度消失或爆炸，影响学习效果    |
| **训练难度**      | BPTT 需要反向传播通过时间，计算量大且梯度问题突出 |

---

### 3.5 RNN 的数学公式

* 隐藏状态更新：

$$
  h_t = \tanh(W_{hx} x_t + W_{hh} h_{t-1} + b_h)  
$$

* 输出计算：

$$  
  y_t = \text{softmax}(W_{yh} h_t + b_y)
$$

---

### 3.6 RNN 的缺点及改进

#### 1. 缺点

* **梯度消失和梯度爆炸**，导致无法学习长距离依赖。
* 只能顺序计算，无法并行处理，训练速度慢。

#### 2. 改进模型

| 模型       | 作用                           |
| -------- | ---------------------------- |
| **LSTM** | 引入门控机制，解决长依赖问题，保留重要信息，忘记无用信息 |
| **GRU**  | LSTM 简化版，计算更高效，效果接近 LSTM     |

---

### 3.7 RNN 的应用场景

| 任务类型       | 说明                  |
| ---------- | ------------------- |
| **自然语言处理** | 语言模型、机器翻译、文本生成、情感分析 |
| **语音识别**   | 声音信号序列识别            |
| **时间序列预测** | 股价预测、天气预测           |
| **视频分析**   | 动作识别、事件检测           |
| **推荐系统**   | 用户行为序列建模            |

---

### 3.8 RNN 示例结构图

```
x1 → [RNN cell] → h1 → y1
        ↓
x2 → [RNN cell] → h2 → y2
        ↓
x3 → [RNN cell] → h3 → y3
```

每个时刻的隐藏状态 $( h_t )$ 连接到下一个时刻，形成链式结构。

---

### 3.9 RNN 训练关键技术

* **BPTT（Back Propagation Through Time）**：时间维度上的反向传播算法。
* **梯度裁剪（Gradient Clipping）**：防止梯度爆炸。
* **正则化**：如 dropout，防止过拟合。
* **序列截断**：限制序列长度，避免内存占用过大。

---

### 3.10 简单 PyTorch RNN 代码示例

```python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, hn = self.rnn(x)
        out = self.fc(out[:, -1, :])  # 取最后时间步输出
        return out

# 假设输入是批次大小=1，序列长度=5，特征维度=10
model = SimpleRNN(input_size=10, hidden_size=20, output_size=2)
input_data = torch.randn(1, 5, 10)
output = model(input_data)
print(output)
```

---

### 3.11 总结

| 方面   | 内容                           |
| ---- | ---------------------------- |
| 定义   | 适合序列数据处理的神经网络，具有记忆能力         |
| 核心结构 | 通过隐藏状态传递信息，形成循环结构            |
| 优点   | 能捕捉时间依赖关系，参数共享               |
| 缺点   | 梯度消失爆炸，训练难度大，难长距离依赖          |
| 改进模型 | LSTM 和 GRU 解决了梯度问题，提高了长期记忆能力 |
| 典型应用 | 语言模型、语音识别、时间序列预测、视频分析等       |

---

## 四、RNN变种（LSTM与GRU）

### 4.1 为什么需要变种RNN？

* 传统 RNN 存在 **梯度消失和梯度爆炸** 问题，难以学习长期依赖。
* 这限制了传统 RNN 在长序列任务上的表现。
* 为解决此问题，设计了带门控机制的变种网络：**LSTM** 和 **GRU**。

---

### 4.2 LSTM（Long Short-Term Memory）长短时记忆网络

#### 1. 结构特点

* 引入了**细胞状态（Cell State）**，可以像传送带一样传递信息，减少信息丢失。
* 设计了三个门控单元控制信息流：

  * **遗忘门（Forget Gate）**：决定哪些信息丢弃
  * **输入门（Input Gate）**：决定哪些新信息写入
  * **输出门（Output Gate）**：决定最终输出信息

#### 2. 关键公式

$$

\begin{aligned}
f_t &= \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \quad &\text{遗忘门} \\
i_t &= \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \quad &\text{输入门} \\
\tilde{C}*t &= \tanh(W_C \cdot [h*{t-1}, x_t] + b_C) \quad &\text{候选细胞状态} \\
C_t &= f_t * C_{t-1} + i_t * \tilde{C}*t \quad &\text{细胞状态更新} \\
o_t &= \sigma(W_o \cdot [h*{t-1}, x_t] + b_o) \quad &\text{输出门} \\
h_t &= o_t * \tanh(C_t) \quad &\text{隐藏状态输出}
\end{aligned}

$$

#### 3. 优缺点

| 优点               | 缺点            |
| ---------------- | ------------- |
| 解决了梯度消失，能捕捉长距离依赖 | 结构复杂，计算资源消耗较大 |
| 适用多种序列任务         | 参数多，训练时间长     |

---

### 4.3 GRU（Gated Recurrent Unit）门控循环单元

#### 1. 结构特点

* GRU 是 LSTM 的简化版，合并了遗忘门和输入门为一个**更新门（Update Gate）**。
* 同时引入了**重置门（Reset Gate）**，控制如何结合新输入和过去信息。
* 结构更简单，计算更快。

#### 2. 关键公式

$$
\begin{aligned}
z_t &= \sigma(W_z \cdot [h_{t-1}, x_t]) \quad &\text{更新门} \\ 
r_t &= \sigma(W_r \cdot [h_{t-1}, x_t]) \quad &\text{重置门} \\
\tilde{h}*t &= \tanh(W \cdot [r_t * h*{t-1}, x_t]) \quad &\text{候选隐藏状态} \\
h_t &= (1 - z_t) * h_{t-1} + z_t * \tilde{h}_t \quad &\text{最终隐藏状态}
\end{aligned}
$$

#### 3. 优缺点

| 优点                  | 缺点            |
| ------------------- | ------------- |
| 结构简单，计算速度快          | 表达能力略逊于 LSTM  |
| 在许多任务上表现接近甚至优于 LSTM | 对某些复杂任务可能不够强大 |

---

### 4.4 LSTM vs GRU 对比总结

| 特点    | LSTM             | GRU          |
| ----- | ---------------- | ------------ |
| 结构复杂度 | 三个门（遗忘门、输入门、输出门） | 两个门（更新门、重置门） |
| 计算效率  | 较慢               | 较快           |
| 参数数量  | 多                | 少            |
| 记忆能力  | 更强（适合长序列）        | 好（适合中等长度序列）  |
| 训练难度  | 较大               | 较小           |
| 适用场景  | 需要长距离依赖的复杂任务     | 轻量级任务，资源有限环境 |

---

### 4.5 典型应用

* **LSTM**：机器翻译、语音识别、文本生成、时间序列预测
* **GRU**：实时语音识别、嵌入式设备、推荐系统等轻量级任务

---

### 4.6 总结

| 方面   | 内容                           |
| ---- | ---------------------------- |
| 目的   | 解决传统RNN梯度消失问题                |
| 结构   | LSTM三个门，GRU两个门               |
| 优势   | 长期依赖学习能力强（LSTM），结构简单速度快（GRU） |
| 选择建议 | 需要高性能和长依赖选LSTM，轻量快速选GRU      |

---

### 4.7 简单PyTorch示例（LSTM和GRU）

```python
import torch
import torch.nn as nn

# LSTM 示例
lstm = nn.LSTM(input_size=10, hidden_size=20, batch_first=True)
input_data = torch.randn(5, 3, 10)  # batch=5, seq_len=3, feature=10
output, (hn, cn) = lstm(input_data)

# GRU 示例
gru = nn.GRU(input_size=10, hidden_size=20, batch_first=True)
output, hn = gru(input_data)
```

---

## 五、Transformer神经网络

### 5.1 什么是 Transformer？

* Transformer 是 2017 年由 Vaswani 等人在论文《Attention Is All You Need》中提出的一种**基于注意力机制的神经网络架构**。
* 它彻底改变了自然语言处理（NLP）领域，摆脱了传统 RNN 依赖顺序计算的限制。
* Transformer 可以**并行处理序列数据**，更高效捕捉长距离依赖。

---

### 5.2 Transformer 的核心组件

#### 1. **自注意力机制（Self-Attention）**

* 传统序列模型处理序列时往往只关注相邻或局部信息，而自注意力机制能**让序列中每个元素与所有其他元素相互“关注”**，捕获全局依赖。

* 计算过程包括：

  * 对输入序列中每个元素，分别计算三个向量：**查询（Query, Q）**、**键（Key, K）**、**值（Value, V）**，通过三个不同的线性变换得到。
  * 计算查询和所有键的点积，反映当前元素与序列其他元素的相关性，得到注意力得分矩阵。
  * 对注意力得分做 softmax 归一化，得到权重分布。
  * 用权重对值向量加权求和，得到当前元素的上下文表示。

* 公式为：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

其中 $(d_k)$ 是键向量的维度，缩放因子防止点积值过大导致梯度消失。

---

#### 2. **多头注意力（Multi-Head Attention）**

* 单一的自注意力头可能限制模型捕获的关系范围。

* 多头注意力将注意力机制分成多个头（head），每个头学习序列在不同子空间的表示。

* 每个头执行独立的线性变换，计算注意力，最终将所有头的输出拼接并线性变换，提升模型表达力。

* 优势：

  * 捕获多样化的语义关系
  * 增强模型对复杂依赖的理解

---

#### 3. **位置编码（Positional Encoding）**

* Transformer 不同于 RNN，**没有顺序信息的天然感知能力**，需要显式注入位置信息。
* 通过向输入嵌入中添加位置编码，使模型了解序列元素的顺序。
* Vaswani 等人提出用**正弦和余弦函数**计算位置编码，形式为：

$$
PE_{(pos, 2i)} = \sin \left( \frac{pos}{10000^{2i/d_{model}}} \right) 
$$

$$
 \quad PE_{(pos, 2i+1)} = \cos \left( \frac{pos}{10000^{2i/d_{model}}} \right)
$$

* 这种设计有利于模型捕捉相对位置关系。

---

#### 4. **前馈神经网络（Feed-Forward Network, FFN）**

* 在每个编码器/解码器层内，注意力子层后面有一个位置独立的前馈全连接网络。
* 结构通常为两层线性变换，中间使用非线性激活函数（如 ReLU）。
* 作用是对每个序列位置的表示进行非线性变换，增强模型能力。

---

#### 5. **层归一化（Layer Normalization）和残差连接（Residual Connection）**

* 为了加速训练和稳定梯度，Transformer 在每个子层（注意力层和前馈层）前后添加了**残差连接**和**层归一化**。
* **残差连接**允许信号直接跳过子层，缓解深层网络训练中的梯度消失。
* **层归一化**对激活值做归一化，提高训练稳定性和收敛速度。

---

### 5.3 Transformer 的基本结构（详细）

Transformer 的整体架构是 **Encoder-Decoder** 结构，具体如下：

#### 1. **Encoder**

* 由多个相同的编码器层堆叠组成（通常6层）。

* 每个编码器层包含两个子层：

  * **多头自注意力层（Multi-Head Self-Attention）**：处理输入序列内部的依赖关系。
  * **前馈全连接层（Feed-Forward Network）**：对每个位置的表示独立进行非线性变换。

* 每个子层后都有残差连接和层归一化。

* Encoder 负责将输入序列编码成高维语义表示，捕获序列内部复杂关系。

---

#### 2. **Decoder**

* 由多个相同的解码器层堆叠组成（通常6层）。

* 每个解码器层包含三个子层：

  * **掩码多头自注意力层（Masked Multi-Head Self-Attention）**：保证解码时只能访问当前位置之前的输出，防止信息泄露。
  * **编码器-解码器注意力层（Encoder-Decoder Attention）**：关注编码器输出，结合输入序列信息。
  * **前馈全连接层（Feed-Forward Network）**。

* 每个子层后都有残差连接和层归一化。

* Decoder 负责逐步生成输出序列，结合已生成的上下文和输入信息。

---

#### 3. **输入与输出处理**

* **输入序列**经过词嵌入（Embedding）和位置编码后送入 Encoder。
* **输出序列**通过类似的嵌入和位置编码输入 Decoder。
* Decoder 最终通过线性层和 softmax 预测下一个词。

---

#### 4. **整体流程示意**

```
输入序列 → 词嵌入 + 位置编码 → Encoder Layer × N → 编码输出
                                             ↓
              Decoder Layer × N（含编码器-解码器注意力）→ 输出序列
```

---

### 5.4 Transformer 的优势

| 优势          | 说明                     |
| ----------- | ---------------------- |
| **并行计算**    | 无需像 RNN 一样逐步计算，训练更快    |
| **长距离依赖建模** | 自注意力直接计算全序列相关性，捕获长距离依赖 |
| **灵活性高**    | 适用各种序列任务（NLP、CV、语音等）   |
| **可扩展性强**   | 随模型参数增加，性能持续提升         |

---

### 5.5 Transformer 的应用领域

* **自然语言处理（NLP）**
  机器翻译、文本生成、问答系统、文本分类、命名实体识别等

* **计算机视觉（CV）**
  图像分类（Vision Transformer）、目标检测（DETR）、图像生成等

* **语音处理**
  语音识别、语音合成等

* **推荐系统**
  用户行为序列建模，提升推荐效果

---

### 5.6 Transformer 的发展和变体

| 模型/变体                        | 特点                        |
| ---------------------------- | ------------------------- |
| **BERT**                     | Encoder-only，预训练双向语言理解模型  |
| **GPT**                      | Decoder-only，自回归语言生成模型    |
| **T5**                       | Encoder-Decoder，统一文本到文本任务 |
| **Vision Transformer (ViT)** | 将 Transformer 应用于图像领域     |
| **DeiT**                     | 轻量化 ViT，适合中小规模数据集         |
| **Swin Transformer**         | 层次化结构，提升视觉任务性能和效率         |

---

### 5.7 简单 Transformer 架构图

```
输入序列 → 位置编码 → Encoder Layer × N → 输出编码
                                     ↓
                            Decoder Layer × N → 输出序列
```

---

### 5.8 简化 PyTorch 自注意力示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, queries):
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # Linear projections
        values = self.values(values).view(N, value_len, self.heads, self.head_dim)
        keys = self.keys(keys).view(N, key_len, self.heads, self.head_dim)
        queries = self.queries(queries).view(N, query_len, self.heads, self.head_dim)

        # Calculate energy scores
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys]) / (self.head_dim ** 0.5)

        attention = torch.softmax(energy, dim=3)

        out = torch.einsum("nhqk,nkhd->nqhd", [attention, values]).reshape(N, query_len, self.embed_size)

        out = self.fc_out(out)
        return out
```

---

### 5.9 总结

| 方面   | 内容                          |
| ---- | --------------------------- |
| 定义   | 基于自注意力机制的神经网络架构，替代RNN进行序列建模 |
| 核心技术 | 自注意力、多头注意力、位置编码、残差连接、层归一化   |
| 优势   | 并行计算、捕捉长距离依赖、高效、灵活          |
| 应用   | NLP、CV、语音、推荐系统等多领域广泛应用      |
| 发展   | BERT、GPT、ViT等众多变体，推动AI进步    |

---

## 六、GNN（Graph Neural Network）图神经网络

### 6.1 什么是GNN？

**Graph Neural Network（图神经网络）**是一类专门用于处理 **图结构数据** 的神经网络。
图由 **节点（Node）** 和 **边（Edge）** 组成，例如：

* 社交网络：人（节点）+ 关系（边）
* 化学分子：原子（节点）+ 化学键（边）
* 推荐系统：用户与商品
* 知识图谱：实体与关系
* 交通网络：路口与道路

图数据是非欧式结构，不像图像/序列那样有规则网格，因此需要专门的神经网络模型来处理。

GNN 的核心思想是：

> **节点通过消息传递（Message Passing）向邻居交换信息，从而更新表示（Embedding）**

---

### 6.2 GNN 的核心思想：消息传递（Message Passing）

所有 GNN 基本流程如下：

#### 6.2.1 **聚合邻居信息（Aggregate）**

$$
m_v = \text{AGG}{h_u : u \in \mathcal{N}(v)}
$$

#### 6.2.2 **组合（更新）节点特征（Update）**

$$
h_v' = \text{UPDATE}(h_v, m_v)
$$

其中：

* $(h_v)$：节点自身特征
* $(h_v')$：更新后的特征
* $(\mathcal{N}(v))$：节点 v 的邻居集合

> **聚合方法可以是：求和、平均、最大池化、注意力等**

GNN 就是在不断“传消息”：

> 节点从邻居获得信息 → 更新自身状态 → 再传播出去

经过多层 GNN 后，节点能“看到”多跳邻居的信息。

---

### 6.3 GNN 基本结构：Message Passing Neural Network (MPNN)

一般一层 GNN 的结构如下：

```
邻居节点特征（Neighbors）
      ↓
  消息聚合（Aggregation）
      ↓
  信息更新（Update）
      ↓
  新的节点表示（Embedding）
```

常见的聚合函数（AGG）：

* Sum（求和）
* Mean（平均）
* Max pooling（最大值）
* Attention（注意力加权）
* LSTM/GRU 聚合

常见的 Update 方法：

* MLP
* GRU
* 残差连接 + LayerNorm

---

### 6.4 常见的 GNN 模型

#### 6.4.1 GCN —— Graph Convolutional Network（图卷积网络）

最经典的 GNN，由 Kipf & Welling 提出（2017）

思想：
**邻居特征加权平均 → 线性变换 → 激活函数**

更新公式：

$$
H^{(l+1)} = \sigma\left( \tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}H^{(l)}W^{(l)} \right)
$$

特点：

* 简单、高效
* 大量图任务的基线模型
* 缺点：信息容易过度平滑（oversmoothing）

---

#### 6.4.2 GraphSAGE（采样式 GNN）

SAGE（2017）主要用于 **大规模图（百万节点）**。

核心特点：

* **采样邻居**（避免邻居太多导致计算爆炸）
* 支持多种聚合方式（mean、pooling、LSTM）
* 更适合工业场景的图表示学习

---

#### 6.4.3 GAT —— Graph Attention Network

引入 **注意力机制（Attention）**：

$$
\alpha_{ij} = \text{softmax}(a(W h_i, W h_j))
$$

邻居贡献不再平均，而是根据注意力系数动态加权。

优势：

* 学习哪些邻居重要
* 比 GCN 表达能力更强
* 避免过度平滑

---

#### 6.4.4 GIN —— Graph Isomorphism Network

提出为了解决 GCN 不能区分某些图结构的问题。

采用：

$$
h_v^{(k)} = \text{MLP}\left((1 + \epsilon)h_v^{(k-1)} + \sum_{u \in N(v)} h_u^{(k-1)}\right)
$$

特点：

* 表达能力接近图同构测试（WL test）
* 是最强的图分类模型之一

---

#### 6.4.5 其他常见 GNN

| 模型        | 特点                |
| --------- | ----------------- |
| **R-GCN** | 处理多关系图（如知识图谱）     |
| **HAN**   | 处理异质图（节点/边多类型）    |
| **MPNN**  | GNN 的通用范式         |
| **PIN**   | 大规模工业图推荐系统常用      |
| **HGT**   | 处理异构图 Transformer |

---

### 6.5 GNN 的应用场景

#### 1. 社交网络分析

* 好友推荐
* 社区发现
* 关系预测
* 虚假账号检测

---

#### 2. 化学与生物（分子图）

* 分子性质预测
* 药物发现（Drug Discovery）
* 蛋白质结构分析

---

#### 3. 推荐系统

* 用户与商品的二部图
* 序列推荐
* 关系推断（Graph Embedding）
* 工业界大量采用（阿里、Pinterest、Twitter）

---

#### 4. 知识图谱

* Link Prediction（推断关系）
* 知识补全
* 知识增强推理

---

#### 5. 交通网络

* 路网分析
* 路况预测
* 轨迹预测

---

#### 6. 图像处理（作为结构数据处理）

* 超像素划分
* 点云识别
* Mesh 处理

---

### 6.6 GNN 的优势与缺点

#### 优势

* 能处理**非欧式数据**
* 结构表达能力强
* 可建模关系与拓扑结构
* 在社交/推荐/分子领域效果极佳
* 与 Transformer 可以结合

---

#### 缺点

* 难以并行（不像 Transformer）
* 训练成本高，图规模大时困难
* 多层传播导致 over-smoothing
* 一些模型表达能力有限

---

### 6.7 PyTorch 实现一个简单 GCN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, H, A):
        # A 是邻接矩阵，包含自连接（A + I）
        D = torch.diag(torch.sum(A, dim=1))
        D_inv_sqrt = torch.inverse(torch.sqrt(D))
        A_norm = D_inv_sqrt @ A @ D_inv_sqrt

        return F.relu(self.linear(A_norm @ H))


class SimpleGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, out_dim)

    def forward(self, H, A):
        H = self.gcn1(H, A)
        H = self.gcn2(H, A)
        return H
```

---

### 6.8 总结

| 内容       | 说明                        |
| -------- | ------------------------- |
| **定义**   | 处理图结构数据的神经网络              |
| **核心机制** | 消息传递（message passing）     |
| **关键操作** | 邻居聚合（AGG）+ 信息更新（UPDATE）   |
| **代表模型** | GCN、GraphSAGE、GAT、GIN     |
| **应用领域** | 推荐系统、社交网络、化学分子、知识图谱、交通网络等 |
| **优势**   | 处理非欧式数据、表达力强              |
| **缺点**   | 并行性差、训练复杂度高               |

---

## 七、GAN（Generative Adversarial Network）生成对抗网络

### 7.1 GAN 简介

* **全称**：Generative Adversarial Network
* **提出者**：Ian Goodfellow 等人，2014年论文《Generative Adversarial Nets》
* **核心思想**：由两个神经网络模型组成—— **生成器（Generator）** 和**判别器（Discriminator）**，通过博弈（对抗）训练，生成器学习“造假”数据，判别器学习区分真伪数据。最终使生成器能够生成非常逼真的数据样本。

---

### 7.2 GAN 的核心结构

```
输入：随机噪声 z ~ p_z(z)

生成器 G：将随机噪声 z 转换为伪造样本 G(z)

判别器 D：输入样本 x，输出其真实性概率 D(x)

训练目标：
- G 尽力生成让 D 判别为真的假样本
- D 尽力区分真样本和假样本
```

---

#### 7.2.1 生成器（Generator）

* 作用：将随机噪声向量映射成“假”样本，使其尽可能逼近真实数据分布。
* 通常是一个神经网络（MLP、CNN、Transformer等）
* 目标：最大程度“骗过”判别器。

---

#### 7.2.2 判别器（Discriminator）

* 作用：判断输入的样本是真实数据还是生成器生成的假数据。
* 输出一个概率值，表示样本真实性。
* 通常是一个二分类神经网络。

---

### 7.3 GAN 的数学表达

GAN 的训练是一个两人零和博弈，目标函数是：

$$
\min_G \max_D V(D, G) = \mathbb{E}*{x \sim p*{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]
$$

* 判别器 D 目标：最大化区分真样本和假样本的正确率
* 生成器 G 目标：最小化判别器成功识别假样本的概率，即让 $D(G(z))$ 趋近于 1


> 训练过程是交替优化 D 和 G：
>
> 1. 固定 G，优化 D
> 2. 固定 D，优化 G

---

### 7.4 GAN 训练流程

1. 从真实数据集中采样一批真实样本
2. 从随机噪声分布采样一批噪声，生成假样本
3. 训练判别器 D，最大化判别真假样本的准确率
4. 训练生成器 G，使判别器更容易判断假样本为真样本
5. 重复步骤1-4，直到生成器生成样本足够逼真

---

### 7.5 GAN 的变体与改进

| 变体名称            | 主要改进点                              | 应用/特点        |
| --------------- | ---------------------------------- | ------------ |
| DCGAN           | 用卷积网络替代 MLP，提升图像生成质量               | 图像生成领域经典网络   |
| WGAN            | 用 Wasserstein 距离替代 JS 散度，缓解训练不稳定问题 | 更稳定训练，改善模式崩溃 |
| Conditional GAN | 条件输入标签，引导生成特定类别数据                  | 有监督条件生成      |
| CycleGAN        | 无监督图像域转换，保持循环一致性                   | 图像风格转换       |
| StyleGAN        | 利用样式混合，控制图像生成风格和细节                 | 高质量人脸生成      |
| BigGAN          | 大规模 GAN，提升生成图像质量                   | 高分辨率复杂图像生成   |

---

### 7.6 GAN 的应用场景

* **图像生成**：人脸生成、艺术画作、超分辨率重建
* **图像翻译**：黑白图上色、图像风格转换、图像修复
* **文本生成**（结合其他模型）
* **数据增强**：生成罕见类别样本，提升模型泛化能力
* **音频生成**：语音合成、音乐创作
* **医学影像**：病灶增强、数据扩充

---

### 7.7 GAN 的挑战与问题

| 挑战点   | 说明                   | 解决方法                                                  |
| ----- | -------------------- | ----------------------------------------------------- |
| 训练不稳定 | 生成器和判别器训练容易失衡，出现模式崩溃 | WGAN、谱归一化、渐进式训练等                                      |
| 模式崩溃  | 生成器只生成一类样本，缺乏多样性     | 多样性损失、多判别器、改进网络结构                                     |
| 难以评价  | 生成样本质量的定量评价缺乏统一标准    | Inception Score (IS)、Fréchet Inception Distance (FID) |
| 训练时间长 | GAN 训练往往计算资源消耗大      | 轻量模型设计、分布式训练                                          |

---

### 7.8 代码示例（PyTorch 简易版）

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 生成器
class Generator(nn.Module):
    def __init__(self, noise_dim=100, img_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.ReLU(),
            nn.Linear(256, img_dim),
            nn.Tanh()
        )
    def forward(self, z):
        return self.net(z)

# 判别器
class Discriminator(nn.Module):
    def __init__(self, img_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# 训练步骤示意（不完整）
# G = Generator()
# D = Discriminator()
# criterion = nn.BCELoss()
# optimizer_G = optim.Adam(G.parameters(), lr=0.0002)
# optimizer_D = optim.Adam(D.parameters(), lr=0.0002)
```

---

### 7.9 总结

* GAN 是一种基于博弈论的生成模型，通过生成器和判别器的对抗训练，实现从随机噪声生成逼真数据。
* 它开创了生成模型的新时代，促进了图像、音频等多媒体领域的发展。
* 尽管面临训练难、模式崩溃等问题，但各种改进方法使 GAN 在很多应用中表现优异。

---

## 八、KAN（Kolmogorov–Arnold Networks）神经网络

### 8.1 背景与理论基础

### Kolmogorov–Arnold 表示定理

* **定理内容**：
  由苏联数学家安德烈·柯尔莫哥洛夫（Andrey Kolmogorov）和维亚切斯拉夫·阿诺尔德（Vladimir Arnold）提出，定理证明了：

  > **任意多维连续函数 $(f(x_1, x_2, \dots, x_n))$ 可以表示成若干一维连续函数的有限和的形式。**

* **数学表达**：
  
  $$
  f(x_1, \dots, x_n) = \sum_{q=0}^{2n} \Phi_q\left(\sum_{p=1}^n \phi_{p,q}(x_p)\right)
  $$

  其中 $(\Phi_q)$ 和 $(\phi_{p,q})$ 是一维连续函数。

* **意义**：
  该定理为多层神经网络（特别是三层网络）能够逼近任意函数提供了理论基础。

---

### 8.2 KAN 神经网络结构

* KAN 是基于上述定理思想设计的网络架构，通常分为三层：

| 层级    | 功能描述                              |
| ----- | --------------------------------- |
| 输入层   | 接收多维输入变量 (x_1, x_2, \dots, x_n)   |
| 第一隐藏层 | 实现一维函数 (\phi_{p,q}(x_p)) 的映射      |
| 第二隐藏层 | 将第一层输出的加权和作为输入，计算 (\Phi_q(\cdot)) |
| 输出层   | 对第二层多个输出结果求和，得到最终函数输出 (f(x))      |

* **特点**：网络结构明确，层级划分清晰，且每个隐藏层节点对应一个一维函数。

---

### 8.3 数学与实现细节

* **网络表达**：

  [
  \hat{f}(x) = \sum_{q=0}^{2n} \Phi_q\left(\sum_{p=1}^n \phi_{p,q}(x_p)\right)
  ]

* 其中，(\phi_{p,q}) 和 (\Phi_q) 可以用神经网络中的激活函数拟合。

* 由于每层都是一维映射，训练时可以将每个子函数单独拟合，提高学习效率。

---

### 8.4 KAN 与传统神经网络的比较

| 特点    | KAN 神经网络                  | 传统 MLP 网络    |
| ----- | ------------------------- | ------------ |
| 理论基础  | 依据 Kolmogorov–Arnold 表示定理 | 通用逼近定理       |
| 结构    | 三层明确分层：一维映射 + 加权和 + 汇总    | 多层全连接，激活函数堆叠 |
| 参数数量  | 依赖于一维函数数量和加权系数            | 层间全连接，参数较多   |
| 训练复杂度 | 可分解为多个一维函数学习              | 整体参数联合训练     |
| 逼近能力  | 可逼近任意连续函数                 | 可逼近任意连续函数    |

---

### 8.5 优缺点

#### 优点

* **理论保证**：基于严格的数学定理，保证任意连续函数的逼近能力。
* **结构清晰**：网络层级与数学定理直接对应，易于理解。
* **可分解训练**：可将多维函数逼近任务分解为多个一维函数逼近，可能提高训练效率。

#### 缺点

* **实际训练难度**：尽管理论上存在函数分解，但实际确定一维函数 (\phi) 和 (\Phi) 并不容易。
* **实现复杂**：与传统神经网络相比，实际应用较少，训练框架和工具支持有限。
* **参数量和规模限制**：需要设计合理的一维函数数目和网络规模，避免过拟合或欠拟合。

---

### 8.6 应用领域

* 主要用于理论研究和函数逼近问题。
* 在某些特殊的工程问题中，有助于对高维函数进行分解和建模。
* 对于多维非线性系统建模、信号处理、控制系统设计等领域有潜在价值。

---

### 8.7 参考文献与资料

* Kolmogorov, A. N. "On the representation of continuous functions of several variables by superpositions of continuous functions of one variable and addition." Doklady Akademii Nauk SSSR 114.5 (1957): 953-956.
* Arnold, V. I. "On functions of three variables." Doklady Akademii Nauk SSSR 114 (1957): 679-681.
* Hecht-Nielsen, R. "Kolmogorov’s mapping neural network existence theorem." Proceedings of the IEEE International Conference on Neural Networks. 1987.
* 《神经网络与深度学习》 - Michael Nielsen（部分章节涉及函数逼近理论）

---

### 8.8 小结

* KAN 是一种基于Kolmogorov–Arnold表示定理设计的三层网络，理论上可以通过组合一维连续函数逼近任意多维连续函数。
* 该网络结构提供了数学上对多层神经网络逼近能力的理解和支持。
* 实际应用中较少直接使用，但为神经网络的普适逼近性质提供了重要理论基础。

---


