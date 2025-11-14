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

* LeCun et al., *“Gradient-Based Learning Applied to Document Recognition”*, 1998. [链接](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)
* Krizhevsky et al., *“ImageNet Classification with Deep Convolutional Neural Networks”*, 2012 (AlexNet). [链接](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
* He et al., *“Deep Residual Learning for Image Recognition”*, 2015 (ResNet). [链接](https://arxiv.org/abs/1512.03385)

#### 4. 代码资源

* PyTorch官方示例：[torchvision.models](https://pytorch.org/vision/stable/models.html)
* TensorFlow官方示例：[tf.keras.applications](https://www.tensorflow.org/api_docs/python/tf/keras/applications)

---

### 1.2. 循环神经网络（RNN）

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

* Hochreiter & Schmidhuber, *“Long Short-Term Memory”*, 1997. [链接](https://www.bioinf.jku.at/publications/older/2604.pdf)
* Cho et al., *“Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation”*, 2014 (GRU). [链接](https://arxiv.org/abs/1406.1078)

#### 4. 代码资源

* TensorFlow RNN教程：[RNN with TensorFlow](https://www.tensorflow.org/tutorials/text/text_classification_rnn)
* PyTorch RNN示例：[RNN in PyTorch](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html)

---

### 1.3. Transformer

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

* Vaswani et al., *“Attention Is All You Need”*, 2017. [链接](https://arxiv.org/abs/1706.03762)
* Devlin et al., *“BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding”*, 2018. [链接](https://arxiv.org/abs/1810.04805)
* Dosovitskiy et al., *“An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale”*, 2020 (ViT). [链接](https://arxiv.org/abs/2010.11929)

#### 4. 代码资源

* Hugging Face Transformers库：[GitHub](https://github.com/huggingface/transformers)
* TensorFlow官方Transformer示例：[TensorFlow Transformer](https://www.tensorflow.org/tutorials/text/transformer)

---

### 1.4. 图神经网络（GNN）

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

* Kipf & Welling, *“Semi-Supervised Classification with Graph Convolutional Networks”*, 2017. [链接](https://arxiv.org/abs/1609.02907)
* Veličković et al., *“Graph Attention Networks”*, 2018. [链接](https://arxiv.org/abs/1710.10903)

#### 4. 代码资源

* PyTorch Geometric库：[GitHub](https://github.com/pyg-team/pytorch_geometric)
* Deep Graph Library (DGL)：[GitHub](https://github.com/dmlc/dgl)

---

### 1.5. 生成对抗网络（GAN）

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

* Goodfellow et al., *“Generative Adversarial Nets”*, 2014. [链接](https://arxiv.org/abs/1406.2661)
* Radford et al., *“Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks”*, 2015 (DCGAN). [链接](https://arxiv.org/abs/1511.06434)
* Zhu et al., *“Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks”*, 2017 (CycleGAN). [链接](https://arxiv.org/abs/1703.10593)

#### 4. 代码资源

* TensorFlow GAN库：[GitHub](https://github.com/tensorflow/gan)
* PyTorch GAN教程：[PyTorch GAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)

---

### 1.6. 液态神经网络（Liquid Neural Networks）

#### 1. 特点

* 受生物神经系统启发，利用动态系统状态。
* 适应非平稳时间序列，动态调整。
* 理论和应用尚处早期。

#### 2. 应用场景

* 动态环境控制
* 时序数据处理（复杂非线性动力学）
* 神经形态计算

#### 3. 代表论文

* Pandarinath et al., *“Inferring single-trial neural population dynamics using sequential auto-encoders”*, 2018 (相关动态状态模型). [链接](https://www.nature.com/articles/s41467-018-05450-y)
* Recent review and works由Liquid Time-constant Networks等提出，参考： [https://arxiv.org/abs/2102.03260](https://arxiv.org/abs/2102.03260)

#### 4. 代码资源

* 目前无官方成熟库，多为研究代码，相关代码常见于论文附录或GitHub搜索。

---

### 1.7 总结对比

| 网络          | 优点              | 缺点             | 适用场景          | 代码资源链接简述                     |
| ----------- | --------------- | -------------- | ------------- | ---------------------------- |
| CNN         | 局部特征提取强，参数少，训练快 | 长距离依赖弱，需大量标注数据 | 图像、视频分析       | torchvision/models, TF.keras |
| RNN         | 适合序列建模，捕获时间依赖   | 训练慢，长序列依赖弱     | 语音、文本、时序      | TensorFlow RNN, PyTorch      |
| Transformer | 全局依赖，自注意力，高并行   | 资源消耗大，需大数据     | NLP、图像、文本生成   | Hugging Face, TensorFlow     |
| GNN         | 处理图结构，关系建模强     | 计算复杂，扩展性有限     | 社交、知识图谱、分子    | PyTorch Geometric, DGL       |
| GAN         | 高质量生成，丰富多样      | 训练不稳定，模式崩溃     | 图像生成、数据增强     | TF-GAN, PyTorch GAN教程        |
| 液态神经网络      | 动态适应能力强，受生物启发   | 理论和实践尚早，训练复杂   | 动态环境时序，神经形态计算 | 研究代码，暂无统一库                   |

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

## 四、Transformer神经网络