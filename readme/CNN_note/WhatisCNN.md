# CNN(Covlution Neural Network)

## 一、CNN总概

### 1.1 CNN是什么？

*   **定义**：卷积神经网络是一种专为处理**网格状数据**（如图像、视频、音频）而设计的深度学习模型。
*   **核心思想**：通过**卷积** 操作来自动、高效地学习数据的空间层次特征。
*   **与传统神经网络的对比**：
    *   **传统神经网络（如全连接网络）**：将输入数据（如图像）展平为一维向量，会完全丢失空间信息，且参数数量巨大，容易过拟合。
    *   **CNN**：保留了数据的空间结构，通过**局部连接**和**权值共享** 大大减少了参数数量，使其更高效、更易于训练。

### 1.2 为什么CNN特别适合图像处理？

1.  **局部相关性**：图像中一个像素与其周围像素的关系最紧密，与遥远像素的关系较弱。CNN的卷积操作正是关注局部区域。
2.  **平移不变性**：无论一只猫在图像的左上角还是右下角，它都是一只猫。CNN通过池化操作和层级结构，使得网络对目标的位置变化不敏感。
3.  **尺度不变性**：通过多层卷积和池化，CNN可以从底层边缘、纹理，到中层部分器官，再到高层整体物体，逐步构建出对图像的尺度鲁棒性理解。

---

### 1.3 CNN的核心组件

一个典型的CNN由以下几部分组成：

---

#### 1.3.1 卷积层 - 特征提取的核心

*   **目的**：使用**滤波器（或称为卷积核）** 在输入数据上滑动，提取局部特征（如边缘、角点、颜色块）。
*   **关键概念**：
    *   **滤波器**：一个小尺寸的权重矩阵（如3x3, 5x5）。不同的滤波器用于提取不同的特征。
    *   **感受野**：滤波器在输入图像上每次覆盖的区域大小。
    *   **步长**：滤波器每次移动的像素数。步长越大，输出特征图尺寸越小。
    *   **填充**：在输入图像边缘填充一圈像素（通常用0填充）。目的是为了控制输出特征图的尺寸。
    *   **深度**：一个卷积层通常使用多个滤波器，每个滤波器会产生一个**特征图**。所有这些特征图堆叠起来，就构成了该卷积层的输出。
*   **工作机制**：滤波器在输入上滑动，在每个位置进行**点乘** 求和，再加上一个偏置项，最终生成特征图。

---

#### 1.3.2 激活函数 - 引入非线性

*   **目的**：为网络引入非线性因素，使其能够学习并模拟复杂的非线性关系。没有它，多层网络就等价于一个单层线性模型。
*   **常用函数**：
    *   **ReLU（修正线性单元）**：`f(x) = max(0, x)`。目前最常用，因为它能有效缓解梯度消失问题，且计算简单。
    *   **Sigmoid / Tanh**：在早期使用，现在多用于输出层（如二分类）。

---

#### 1.3.3 池化层 - 降维和保持平移不变性

*   **目的**：对特征图进行**下采样**，减少数据尺寸和参数量，防止过拟合，同时扩大后续卷积层的感受野，并赋予网络一定的平移不变性。
*   **特点**：没有需要学习的参数。
*   **常用方法**：
    *   **最大池化**：取池化窗口内的最大值。效果最好，最常用。
    *   **平均池化**：取池化窗口内的平均值。
*   **工作机制**：类似卷积，有一个窗口和步长，在特征图上滑动，但执行的是最大或平均操作。

---

#### 1.3.4. 全连接层 - 最终分类

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

### 1.4 经典的CNN网络架构

*   **LeNet-5 (1998)**：CNN的开山之作，用于手写数字识别，结构为：`卷积 -> 池化 -> 卷积 -> 池化 -> 全连接 -> 输出`。
*   **AlexNet (2012)**：在ImageNet大赛中一战成名，真正开启了深度学习热潮。它更深，使用了ReLU、Dropout等技术。
*   **VGGNet (2014)**：探索了网络深度，通过堆叠小的3x3卷积核来替代大的卷积核，结构非常规整。
*   **GoogLeNet (2014)**：引入了**Inception模块**，在增加网络深度和宽度的同时，减少了参数量。
*   **ResNet (2015)**：提出了**残差块** 和**跳跃连接**，解决了极深网络的梯度消失和退化问题，使得构建上百甚至上千层的网络成为可能。
*   **MobileNet (2017)**：引入了**深度可分离卷积**为核心构建块，专为移动和嵌入式设备等资源受限环境设计，在速度和体积上实现了极致优化。
*   **YOLO (2016)**：**You Only Look Once** 的缩写，是目标检测领域的革命性框架。它将检测任务视为一个回归问题，实现了端到端的训练和极快的推理速度，催生了一系列实时检测应用。其核心思想是使用CNN骨干网络进行特征提取，再通过检测头直接预测边界框和类别概率。

---

### 1.5 CNN的工作流程总结

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

### 1.6 CNN的应用领域（远超图像）

*   **计算机视觉**：图像分类、目标检测、图像分割、人脸识别。
*   **自然语言处理**：文本分类、情感分析、机器翻译（通过一维卷积处理序列）。
*   **游戏与机器人**：AlphaGo下围棋，机器人导航。
*   **医疗**：医学影像分析（CT、MRI片子诊断）。

---

### 1.7 常见CNN卷积神经网络的分类

#### 1.7.1 根据任务类型分类

#### （1）分类模型（Classification Backbone）

只负责提取图像特征：

* **LeNet（最早）**
* **AlexNet**
* **VGG**
* **GoogleNet（Inception）**
* **ResNet**
* **DenseNet**
* **MobileNet**
* **ShuffleNet**
* **EfficientNet**

这些模型本身不能直接检测，只作为 **“主干网络”**，上面可以挂目标检测头。

---

#### （2）目标检测模型（Object Detection）

直接输出：

* 何处有物体（bbox）
* 什么类（label）

主要分两派：

---

##### ● 两阶段（Two-Stage）

“先生成候选框 → 再分类回归”

代表：R-CNN系列

* **R-CNN**
* **Fast R-CNN**
* **Faster R-CNN**
* **Mask R-CNN**

特点：

* 精度高
* 推理慢

---

##### ● 单阶段（One-Stage）

端到端直接预测：

> * **YOLO 系列**
>
>   * YOLOv1 → v8/11/YOLOX/YOLOv7……
> * **SSD / SSD-lite**
> * **RetinaNet**
> * **FCOS**
> * **CenterNet**

特点：

* 快
* 工业部署最多

目前：

> YOLO = 工程应用的事实标准

---

#### （3）图像分割模型（Segmentation）

* FCN
* U-Net
* DeepLab
* PSPNet
* Mask R-CNN（实例分割）
* Segment Anything（Transformer 时代）

---

#### 1.7.2 根据技术路线分类

##### （1）传统 CNN 架构

全卷积体系：

* VGG
* ResNet
* DenseNet
* MobileNet
* ShuffleNet
* EfficientNet

特点：

✔ 硬件加速完善
✔ 端侧部署最佳
✔ 工业仍大量使用

---

##### （2）Transformer 架构

以自注意力机制取代卷积：

* ViT（视觉Transformer）
* Swin Transformer
* DETR / DINO / RT-DETR（检测）
* Segment Anything（分割）

特点：

✔ SOTA 精度
✔ 大模型趋势主线
✘ 端着设备算力压力大

---

##### （3）混合架构（CNN + Transformer）

既有卷积也有注意力：

* MobileViT
* EfficientFormer
* EdgeViT
* ConvNeXt（CNN风格ViT）

目前实际落地很快。

---
# 分类模型

分类模型只负责提取图像特征，例如：

* **LeNet（最早）**
* **AlexNet**
* **VGG**
* **GoogleNet（Inception）**
* **ResNet**
* **DenseNet**
* **MobileNet**
* **ShuffleNet**
* **EfficientNet**

这些模型本身不能直接检测，只作为 **“主干网络”**，上面可以挂目标检测头。

## 一、Lenet5

### 1.1 背景介绍

LeNet-5 是由 Yann LeCun 等人在1998年提出的卷积神经网络（CNN）模型，最初用于手写数字识别（MNIST 数据集）。它是现代深度学习中经典的卷积神经网络雏形，对后续的CNN发展有重要影响。

---

### 1.2 网络架构

LeNet-5 主要由以下几部分组成：

| 层名称 | 类型   | 输入尺寸         | 输出尺寸         | 参数说明              | 参数数量（含偏置） |
| --- | ---- | ------------ | ------------ | ----------------- | --------- |
| 输入层 | —    | 1 × 32 × 32  | 1 × 32 × 32  | 灰度图像输入            | 0         |
| C1  | 卷积层  | 1 × 32 × 32  | 6 × 28 × 28  | 6个卷积核，大小5×5，步长1   | 156       |
| S2  | 池化层  | 6 × 28 × 28  | 6 × 14 × 14  | 平均池化，窗口2×2，步长2    | 0         |
| C3  | 卷积层  | 6 × 14 × 14  | 16 × 10 × 10 | 16个卷积核，大小5×5，部分连接 | 2,416     |
| S4  | 池化层  | 16 × 10 × 10 | 16 × 5 × 5   | 平均池化，窗口2×2，步长2    | 0         |
| C5  | 卷积层  | 16 × 5 × 5   | 120 × 1 × 1  | 120个卷积核，大小5×5，全连接 | 48,120    |
| F6  | 全连接层 | 120          | 84           | 全连接层              | 10,164    |
| 输出层 | 全连接层 | 84           | 10           | 对应10类数字的输出        | 850       |

---

### 1.3 各层功能及原理详解

#### 1. 输入层

* 输入为 32×32 像素的灰度图像（单通道），
* 原始MNIST图像为28×28，LeNet-5采用了32×32的输入尺寸，方便边缘处理。

#### 2. C1卷积层

* 使用6个5×5卷积核，对输入进行卷积操作，步长为1，
* 输出6个28×28的特征图（通道），
* 参数总数计算：

$$
5 \times 5 \times 1 \times 6 + 6 = 150 + 6 = 156
$$

* 功能：提取图像的局部特征，如边缘、纹理。

#### 3. S2池化层（子采样层）

* 对每个通道做平均池化，窗口大小2×2，步长2，

* 输出特征图尺寸减半，变为6×14×14，

* 无需学习参数。

* 功能：降低特征图尺寸，减少计算量和过拟合风险，同时增强平移不变性。

#### 4. C3卷积层（部分连接卷积层）

* 有16个5×5卷积核，输入通道为6，
* 采用“部分连接”，不是每个输出通道都连接所有输入通道，减少参数量和计算，
* 输出16个10×10特征图。
* 参数总数计算：

$$
5 \times 5 \times 6 \times 16 + 16 = 2400 + 16 = 2,416
$$

* 功能：提取更复杂的组合特征，部分连接提升表达能力同时控制参数规模。

#### 5. S4池化层

* 结构与S2相同，平均池化，窗口2×2，步长2，

* 输出16×5×5特征图。

* 作用同S2，进一步降低尺寸和参数。

#### 6. C5卷积层（全连接卷积层）

* 卷积核大小为5×5，输入16个通道，
* 因为输入特征图为5×5，卷积后输出为1×1，
* 实际上等价于全连接层，输出120个神经元。
* 参数总数计算：

$$
5 \times 5 \times 16 \times 120 + 120 = 48,000 + 120 = 48,120
$$

* 功能：综合所有特征，形成高层次表达。

#### 7. F6全连接层

* 输入120个神经元，输出84个神经元，
* 参数总数计算：

$$
120 \times 84 + 84 = 10,080 + 84 = 10,164
$$

* 作用是进一步整合高层特征。

#### 8. 输出层

* 84维输入，10维输出，
* 对应10个数字类别的得分（logits），
* 参数总数计算：

$$
84 \times 10 + 10 = 840 + 10 = 850
$$

---

### 1.4 LeNet-5的核心原理

* **局部感受野**：卷积核只看输入局部区域，提取局部特征。
* **权值共享**：同一个卷积核在整个输入空间滑动，减少参数。
* **下采样（池化）**：降低特征图尺寸，提升平移不变性，减少计算量。
* **层级特征抽取**：从简单边缘到复杂结构的逐层特征学习。
* **部分连接**：C3层部分连接设计，平衡表达能力和参数规模。
* **端到端训练**：通过反向传播联合优化所有参数。

---

### 1.5 LeNet-5的影响和局限

#### 影响

* 是最早成功应用于图像识别的CNN架构之一，
* 为后续更深层、更复杂的网络（如AlexNet、VGG、ResNet）奠定基础。

#### 局限

* 网络较浅，容量有限，难以处理复杂大规模数据集，
* 只适合灰度小图像，未用批量归一化、激活函数优化等现代技术。

---

### 1.6 总结

| 优点         | 局限               |
| ---------- | ---------------- |
| 参数少，训练快    | 深度和宽度有限          |
| 结构简单，易理解   | 表达能力有限           |
| 权值共享、局部感受野 | 未使用现代优化技术（如ReLU） |

---

### 1.7 LeNet-5 总参数量

所有可训练参数加起来约为：

$$
156 + 0 + 2,416 + 0 + 48,120 + 10,164 + 850 = 61,706
$$

**总参数量约 61,706 个。**

---

### 1.8 参考资料

* Yann LeCun, et al., “Gradient-Based Learning Applied to Document Recognition,” Proceedings of the IEEE, 1998.
* PyTorch官方文档及LeNet-5代码实现示例。

---

## 二、Alexnet

### 2.1 AlexNet简介

* **提出时间**：2012年
* **作者**：Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton
* **背景**：AlexNet在2012年ImageNet竞赛（ILSVRC-2012）中大幅领先，推动了深度学习在计算机视觉领域的快速发展。
* **核心贡献**：

  * 深度卷积神经网络在大规模图像分类任务中的成功应用
  * 使用ReLU激活函数提升训练速度
  * 引入Dropout减少过拟合
  * 利用数据增强提高模型泛化能力
  * 首次成功使用GPU加速训练

---

### 2.2 AlexNet网络结构

AlexNet输入是**224×224×3**彩色图片，输出为1000类（ImageNet分类任务）。

#### 网络层次组成：

| 层数 | 类型              | 参数细节                | 输出尺寸      |
| -- | --------------- | ------------------- | --------- |
| 1  | 卷积层（Conv1）      | 96个11×11卷积核，步长4，填充0 | 55×55×96  |
| 2  | 最大池化层（MaxPool1） | 3×3池化核，步长2          | 27×27×96  |
| 3  | 卷积层（Conv2）      | 256个5×5卷积核，步长1，填充2  | 27×27×256 |
| 4  | 最大池化层（MaxPool2） | 3×3池化核，步长2          | 13×13×256 |
| 5  | 卷积层（Conv3）      | 384个3×3卷积核，步长1，填充1  | 13×13×384 |
| 6  | 卷积层（Conv4）      | 384个3×3卷积核，步长1，填充1  | 13×13×384 |
| 7  | 卷积层（Conv5）      | 256个3×3卷积核，步长1，填充1  | 13×13×256 |
| 8  | 最大池化层（MaxPool3） | 3×3池化核，步长2          | 6×6×256   |
| 9  | 全连接层（FC6）       | 4096个神经元            | 4096      |
| 10 | 全连接层（FC7）       | 4096个神经元            | 4096      |
| 11 | 全连接层（FC8）       | 1000个神经元（对应1000分类）  | 1000      |
| 12 | Softmax层        | 归一化输出为概率            | 1000      |

#### PyTorch伪代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        
        # 5个卷积层（含激活与池化）
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),  # Conv1
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),  # LRN
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),  # Conv2
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),  # LRN
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),  # Conv3
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),  # Conv4
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),  # Conv5
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # 3个全连接层（含Dropout）
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),  # 输入尺寸依据最后池化层输出调整
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            
            nn.Linear(4096, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)                   # 特征提取层
        x = torch.flatten(x, 1)               # 展平多维输入为(batch_size, -1)
        x = self.classifier(x)                 # 分类层
        return x
```

#### 说明：

* `LocalResponseNorm`实现了论文中的LRN层，PyTorch中自带实现。
* Dropout用于防止过拟合。
* 输入图像尺寸通常是`224×224×3`，输出为`num_classes`分类概率（通过训练时搭配`CrossEntropyLoss`使用）。
* 训练时一般用`Softmax`层结合`CrossEntropyLoss`，推理时直接用`nn.Softmax(dim=1)`计算概率。

---

### 2.3 关键技术细节与创新点

#### 1. **ReLU激活函数**

* 代替传统的sigmoid或tanh，ReLU函数 ( f(x) = \max(0, x) ) 极大加速了训练收敛速度。

#### 2. **局部响应归一化（LRN）**

* 用于增强神经元之间的竞争，提高泛化能力。
* 作用在Conv1和Conv2层输出上。

#### 3. **Dropout**

* 在全连接层（FC6和FC7）引入Dropout，随机“丢弃”一部分神经元，减少过拟合。

#### 4. **数据增强**

* 通过图像平移、翻转、颜色变化等方式生成训练样本，提高模型鲁棒性。

#### 5. **GPU并行训练**

* 利用两块GPU并行计算，将模型和数据分配在两块GPU上加速训练。
* 分GPU架构设计，Conv层中特定卷积核分布在不同GPU。

#### 6. **大卷积核+大步长**

* 第1层采用11×11卷积核和步长为4，快速降低输入图像空间大小。

---

### 2.4 AlexNet的优势与不足

#### 优势

* 开创了大规模深度卷积神经网络的先河。
* 在ImageNet分类任务中准确率大幅领先传统方法。
* 显著缩短训练时间，促进GPU加速深度学习普及。
* 通过Dropout和数据增强缓解过拟合。

#### 不足

* 网络较大，参数量多（约6000万个参数），计算开销大。
* 结构相对简单，后续网络（如VGG、ResNet）在结构设计上更优。
* 局部响应归一化（LRN）后来被批量归一化（BatchNorm）取代。

---

### 2.5 AlexNet的应用与影响

* AlexNet奠定了深度卷积神经网络在计算机视觉领域的基础。
* 促进了后续更深、更复杂网络的设计（如VGG, ResNet等）。
* 推动了GPU深度学习硬件与软件生态发展。
* AlexNet原理被广泛应用于图像分类、目标检测、图像分割等任务。

---

### 2.6 总结

| 方面   | 说明                                |
| ---- | --------------------------------- |
| 代表意义 | 深度学习首次在大型图像分类比赛中取得突破              |
| 网络结构 | 5个卷积层 + 3个全连接层 + ReLU激活 + Dropout |
| 关键技术 | ReLU，LRN，Dropout，数据增强，GPU并行训练     |
| 参数规模 | 约6000万个参数                         |
| 影响力  | 开启了深度卷积神经网络在视觉领域的广泛应用             |

---

## 三、MobileNet

### 3.1 为什么要有 MobileNet？

传统 CNN（如 VGG、ResNet）计算量非常大，特别是普通卷积：

* 参数量高
* FLOPs 大
* 部署成本高
* 不适合手机、嵌入式、边缘设备

**MobileNet 的目标：**

> 在保证精度的情况下，大幅降低参数量和计算量，让 CNN 能在手机端轻松运行。

---

### 3.2 核心思想：Depthwise Separable Convolution（深度可分离卷积）

MobileNet 的关键创新，就是 **把普通卷积拆成两步**：

---

#### 3.2.1 普通卷积 Conv2D 的问题

普通卷积计算复杂度：

$$
D_K^2 \cdot M \cdot N \cdot D_F^2
$$

这里$D_K$为卷积核尺寸大小，$D_F$为特征图尺寸大小，$M$为输入通道数，$N$为输出通道数

所需参数量

$$
D_K^2 \cdot M \cdot N
$$

例如输入通道=32，输出通道=64，卷积核=3×3，则：

* 参数量 = 3×3×32×64
* 计算量巨大

---

#### 3.2.2 深度可分离卷积的结构

> 深度可分离卷积 = **Depthwise Conv（逐通道卷积） + Pointwise Conv（1×1卷积）**

---

#### **A. Depthwise Convolution（逐通道卷积）**

* 每个输入通道单独使用一个卷积核
* 不改变通道数
* 计算非常轻量

参数量：

$$
D_K^2 \cdot M
$$

这里$D_K$为卷积核尺寸大小，$M$为输入通道数

---

#### **B. Pointwise Convolution（1×1卷积）**

* 用 1×1 卷积完成通道混合
* 决定输出通道数

参数量：

$$
M \cdot N
$$

---

#### 3.2.3 与普通卷积的计算量对比

普通卷积：

$$
D_K^2 \cdot M \cdot N
$$

深度可分离卷积：

$$
D_K^2 \cdot M + M \cdot N
$$

节省比率：

$$
\frac{
D_K^2 \cdot M \cdot N
}{
D_K^2 \cdot M + M \cdot N
}
\approx 8\text{~}9 \text{倍}
$$

**MobileNet 能比普通卷积快 8～9 倍，模型也更轻。**

---

### 3.3 MobileNet V1 架构特征

MobileNet V1（2017）第一次提出深度可分离卷积，结构非常简单：

* 输入：224×224
* 第一层为普通卷积（stride=2）
* 后续卷积全部用 **depthwise + pointwise**
* 使用 ReLU6（防止量化损失）
* 最终使用 1024 通道全连接分类

#### 参数量：

* 约 **4.2M 参数**
* FLOPs = **569M**（比 VGG16 的 15.5G 小 27 倍）

特点：

* 非常轻量
* 推理速度极快
* 适合移动端部署

---

### 3.4 MobileNet V2（2018）结构特征

V2 引入两个关键创新：

---

#### 3.4.1 线性瓶颈（Linear Bottleneck）

传统卷积最后使用非线性（ReLU），会损失信息。

MobileNetV2：

> 在低维空间使用 **线性层（无 ReLU）** 避免特征破坏。

---

#### 3.4.2 倒残差结构（Inverted Residual）

ResNet 残差块是“压缩 → 卷积 → 扩张”。

MobileNetV2 反过来：

> 先扩张通道 → depthwise → 压缩通道

好处：

* 特征信息保留更充分
* 计算量更低
* 支持轻量级残差结构

---

#### 3.4.3 MobileNetV2 的参数量

* 仅 **3.4M 参数**
* 300M FLOPs

比 V1 更快、更准。

---

### 3.5 MobileNet V3（2019）结构特征

采用 NAS（神经网络结构搜索）+ V2 结构改进：

---

#### **关键创新：**

#### Squeeze-and-Excitation (SE) 模块

增强通道注意力机制。

#### h-swish 激活函数

更高精度、更好量化兼容性。

#### 更优结构搜索

让网络结构自动最优。

#### 版本：

* MobileNetV3 **Small**（超轻量）
* MobileNetV3 **Large**（兼顾精度）

---

### 3.6 MobileNet V1 V2 V3 与 LeNet5 对比


| 项目           | **LeNet5**     | **MobileNet V1**         | **MobileNet V2**                   | **MobileNet V3**         |
| ------------ | -------------- | ------------------------ | ---------------------------------- | ------------------------ |
| **发表年份**     | 1998           | 2017                     | 2018                               | 2019                     |
| **主要任务**     | MNIST 手写数字识别   | ImageNet 分类              | ImageNet 分类、移动端部署                  | 自动化模型搜索 + 移动端 SOTA       |
| **输入大小**     | 32×32          | 224×224                  | 224×224                            | 224×224                  |
| **参数量**      | ~60K           | ~4.2M                    | ~3.4M                              | ~2.9M（最小版）               |
| **卷积方式**     | 普通卷积           | 深度可分离卷积（DW+PW）           | 倒残差结构（Inverted Residual）+ DW 卷积    | 加入 SE、h-swish、NAS 优化     |
| **网络特色**     | 简单的 CNN + 池化   | 将 3×3 卷积拆成 DW 和 PW，计算量极低 | 引入扩展层（Expand）与压缩层（Projection），提升精度 | 自动搜索结构（NAS）+ 更高效注意力（SE）  |
| **激活函数**     | tanh           | ReLU                     | ReLU6                              | h-swish（更平滑更高效）          |
| **是否使用残差结构** | 否              | 否                        | ✔ 是（Inverted Residual）             | ✔ 是（改进版 residual）        |
| **是否轻量化**    | 否              | ✔ 轻量化开山之作                | ✔ 更高效，精度提升                         | ✔ 性能/速度顶尖的轻量化模型          |
| **适用平台**     | CPU / GPU      | 手机、嵌入式、IoT               | 手机、IoT、实时推理                        | 手机 AI 芯片、IoT、边缘设备        |
| **主要贡献**     | 提出卷积池化框架       | 大幅减少计算量 & 参数             | 用更深结构提升精度                          | 在保证轻量化的同时提升精度到接近大型模型     |
| **创新结构**     | C1-S2-C3-S4-C5 | DW + PW 卷积               | Inverted Residual Block            | SE Block + h-swish + NAS |
| **计算量**      | 极低             | 569M FLOPs               | 300M FLOPs                         | 219M FLOPs（V3-large）     |


---

### 3.7 MobileNet 的优缺点

#### 优点

* 参数量极低（3~4M）
* FLOPs 小，速度快
* 适合手机、物联网、嵌入式
* 训练速度快
* 量化友好

---

#### 缺点

* 精度比不上 ResNet/ViT
* Depthwise Conv 对 GPU 不太友好（并行性差）
* 在高精度任务上不足
* 对训练超参数敏感

---

## 四、VGG

### 4.1 VGG 是什么？

VGG（Visual Geometry Group Network）由牛津大学 VGG 团队提出，发表于 2014 年 ILSVRC（ImageNet）比赛中，取得优异成绩。

VGG 的提出目标：

> **研究“网络越深，性能是否越好”**

它的贡献不是新结构，而是：

* 使用 **固定小卷积核（3×3）堆叠**
* 更深层次的卷积堆叠网络
* 证明了“深层 CNN 能显著提升表现”

VGG 是现代卷积网络的经典基础，结构清晰、规则、易复现，广泛用于：

* 分类
* 检测（SSD）
* 迁移学习
* 特征提取

---

### 4.2 VGG 的核心特点

#### **1️⃣ 小卷积核（3×3）+ 多层堆叠**

不同于 AlexNet（7×7、11×11 大卷积核），VGG 全部使用：

```
3×3 卷积（stride=1）
2×2 最大池化（stride=2）
```

---

##### 为什么 3×3 好？

一次 5×5 卷积等价于两次 3×3 卷积：

| 方式        | 参数量 | 可表达能力 |
| --------- | --- | ----- |
| 5×5 卷积    | 大   | 一次感受  |
| 3×3 + 3×3 | 小   | 逐层抽象  |

优势：

* 多一次非线性（ReLU）
* 更少参数
* 更深模型
* 更强表示能力

三层 3×3 等价于 7×7，但参数更少。

---

#### **2️⃣ 网络更深**

VGG 分为多个版本（层数不同）：

* **VGG-11**
* **VGG-13**
* **VGG-16**
* **VGG-19**

其中：

> **VGG-16（13 Conv + 3 FC）最经典**

VGG 证明：

> 在合理结构下，CNN 越深性能越好。

---

#### **3️⃣ 网络结构统一**

全网络只包含：

```
3×3 Conv
2×2 MaxPool
FC 最后分类
```

简单、规则、好扩展，是重要工程模板。

---

### 4.3 VGG 网络整体结构

#### 1. 网络结构

以 **VGG-16** 为例：

```
输入（224×224）
↓
卷积 3×3 ×2     → 64 通道
最大池化
↓
卷积 3×3 ×2     → 128 通道
最大池化
↓
卷积 3×3 ×3     → 256 通道
最大池化
↓
卷积 3×3 ×3     → 512 通道
最大池化
↓
卷积 3×3 ×3     → 512 通道
最大池化
↓
Flatten
FC → FC → FC → Softmax
```

#### 2. VGG 结构细节说明

##### 1）卷积部分（Feature Extractor）

* 卷积核：3×3
* 步长：1
* 填充：1（不改变 Feature Map 尺寸）
* 激活：ReLU

每次池化将分辨率减半：

```
224 → 112 → 56 → 28 → 14 → 7
```

##### 2）分类器（Classifier）

经典配置：

```
4096
4096
1000（ImageNet）
```

全连接层巨大，是 VGG 最大参数来源。

---

#### 3. 典型参数规模（以 VGG-16 为例）

* 参数量：**约 138M**
* 计算量大
* 存储占用高（当年难上 GPU）

VGG 也是：

> “深度网络庞大模型时代”的重要代表。

---

#### 4. VGG 的 PyTorch 典型代码结构

##### 卷积模块

```python
import torch.nn as nn

def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers += [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ]
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)
```

##### VGG-16 结构

```python
class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.features = nn.Sequential(
            vgg_block(2, 3, 64),
            vgg_block(2, 64, 128),
            vgg_block(3, 128, 256),
            vgg_block(3, 256, 512),
            vgg_block(3, 512, 512),
        )

        self.classifier = nn.Sequential(
            nn.Linear(7*7*512, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
```

---

### 4.4 VGG 的优缺点

#### 优点

| 优点         | 解释                     |
| ---------- | ---------------------- |
| 结构统一、简洁    | 只有 3×3 Conv 和 2×2 Pool |
| 更深提升性能     | 实验证明深度有效               |
| 更小卷积更高表达能力 | 3×3 堆叠 > 大卷积核          |
| 工程意义重大     | 后续 CNN 都按它的“堆卷积”思想演化   |
| 适合迁移学习     | 从 VGG 提取特征常效果很好        |

特别是在：

* 小数据集
* 分类任务

中表现优秀。

---

#### 缺点

| 缺点         | 原因         |
| ---------- | ---------- |
| 参数量特别大     | FC 和大量通道   |
| 计算量大       | 不适合边端设备    |
| 没解决退化与梯度消失 | ResNet 才解决 |
| 训练成本高      | 昂贵的显存占用    |

因此：

> VGG 是结构经典，但不是现代实时模型。

---

### 4.5 VGG 与 AlexNet / ResNet 对比

| 特征  | AlexNet | VGG   | ResNet   |
| --- | ------- | ----- | -------- |
| 提出  | 2012    | 2014  | 2015     |
| 风格  | 大卷积核    | 小卷积堆叠 | 残差连接     |
| 深度  | 8       | 16~19 | 50~1000+ |
| 参数量 | 中       | 非常大   | 中        |
| 难训练 | 一般      | 较难    | 容易       |

VGG 是：

> 从 “浅 CNN 到深 CNN 的过渡时代代表”。

---

### 4.6 VGG 的研究意义

VGG 的贡献不在“小技巧”，而在：

> **标准化了现代 CNN 的设计思想。**

影响深远：

* Conv → Conv → Pool 的经典结构
* 小卷积核堆叠成为默认选项
* 后续网络（ResNet、DenseNet、MobileNet）都继承其思想

VGG 是：

> “现代深度 CNN 的正规军起点”。

---

### 4.7 总结

> **VGG 是通过大量 3×3 小卷积核堆叠构建的深层 CNN，结构简洁、性能强，是深度学习从浅层走向深层的关键里程碑。**

---

## 五、Resnet

### 5.1 ResNet 是什么？

ResNet（Residual Network）是 **卷积神经网络中里程碑式的结构**，由何凯明团队提出，发表于 2015 年，在 ImageNet 中获得第一名。

它解决了一个长期存在的问题：

> “网络越深，效果反而下降。”

深度 CNN 会出现：

* 准确率下降（退化问题）
* 梯度消失/梯度爆炸
* 网络难以训练

ResNet 用 **残差连接（skip connection / shortcut）**，让网络轻松训练百层甚至千层深度的 CNN。

---

### 5.2 核心思想：残差学习（Residual Learning）

传统深度网络要直接学习：

```
H(x) —— 输入到输出的映射
```

ResNet 认为网络更容易学习：

```
F(x) = H(x) - x   （残差）
=> H(x) = F(x) + x
```

也就是：

> 让网络学习“变化量”，而不是“全部映射”。

因此有：

#### 残差结构（Residual Block）

```
输入 x
→ 卷积（Conv + BN + ReLU） × 2
得到 F(x)
输出 = F(x) + x
```

公式：

$$
y = F(x) + x
$$

这个 $+ x$ 就是 **跳跃连接（Shortcut Connection）**。

优势：

* 梯度可以不经过层层卷积直接传到前面
* 解决梯度消失
* 网络再深性能不会下降

---

### 5.3 残差结构图示

#### 一个 Residual Block（基础版）

```
x ──────────────┐
↓               │
Conv → BN → ReLU
Conv → BN
└────── + ──────┘
     ReLU
```

如果输入输出维度一致，直接加；维度不一致，就加入 **1×1 卷积调整通道数**，称为 “Projection Shortcut”。

---

#### 两类残差块

#### 1）ResNet-18/34 使用 **Basic Block**

```
Conv(3×3)
Conv(3×3)
+ shortcut
```

结构简单，适合浅层网络。

---

#### 2）ResNet-50/101/152 使用 **Bottleneck Block**

```
1×1 Conv  降维
3×3 Conv  特征提取
1×1 Conv  升维
+ shortcut
```

原因：

* 深层网络如果直接堆叠 3×3 卷积，会计算量巨大
* 先用 1×1 降维 × 再升维，**减少计算量，提高性能**

结构类似：

```
x ───────────────────────────┐
 ↓                           │
1×1 Conv → 3×3 Conv → 1×1 Conv
└─────────────── + ──────────┘
        ReLU
```

---

### 5.4 ResNet 典型结构

#### 1）输入阶段

```
7×7 卷积 + BN + ReLU
↓
3×3 最大池化
```

#### 2）中间阶段（4 个残差阶段）

例如：ResNet-18

```
Stage1: 2 个 Basic Block
Stage2: 2 个 Basic Block
Stage3: 2 个 Basic Block
Stage4: 2 个 Basic Block
```

越往后通道越多、分辨率越小。

#### 3）输出阶段

```
全局平均池化（GAP）
全连接（分类）
```

---

### 5.5 典型模型参数规模

| 网络         | 深度  | 参数量（M） | Top-5 Error（ImageNet） |
| ---------- | --- | ------ | --------------------- |
| ResNet-18  | 18  | 11M    | 10.5%                 |
| ResNet-34  | 34  | 21M    | 7.8%                  |
| ResNet-50  | 50  | 25M    | 6.7%                  |
| ResNet-101 | 101 | 44M    | 6.3%                  |
| ResNet-152 | 152 | 60M    | 5.8%                  |

大幅度降低错误率，同时允许网络更深。

---

### 5.6 为什么 ResNet 能训练很深？

#### 1. 梯度可以“直通”前层

在反向传播中：

$$
y = F(x) + x
$$

$$
dy/dx = (dF/dx) + 1
$$

即使：

$$
dF/dx → 0
$$

仍有：

$$
dy/dx = 1
$$

梯度不会消失。

---

#### 2. 解决网络退化问题

传统深层网络：

```
更深 ≠ 更准
往往更差
```

ResNet：

```
更深 = 更准
```

因为：

* 通过残差结构
* 优化难度降低
* 更易保持高表达能力

---

### 5.7 Resnet优缺点

#### 优点

✓ 解决退化与梯度消失/爆炸
✓ 加深网络仍能提升性能
✓ 泛化能力强
✓ 结构简单、工程价值高
✓ 成为大量现代网络的基础（YOLO、FPN、Mask R-CNN）

#### 缺点

✗ 参数量仍较大
✗ 特征重用效率不如 DenseNet
✗ 在移动端不是最优解（MobileNet 更轻量）

---

### 5.8 ResNet 的 PyTorch 典型代码与训练细节

#### 训练细节

在深度 CNN 的训练中还引入：

* Batch Normalization（BN）
* Xavier / He 初始化
* 全局平均池化代替全连接
* SGD with momentum

ResNet 把这些统一、系统化，**成为现代 CNN 的标准框架**。

#### 残差块

```python
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2   = nn.BatchNorm2d(out_channels)

        # 如果通道或尺寸不一致，使用 1×1 卷积做 shortcut
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        return F.relu(out + residual)
```

---

### 5.9 ResNet 的意义

ResNet提升了两个时代级标准：

#### CNN 结构标准化

后续网络（DenseNet、MobileNet、FPN、YOLO 等）几乎都引入：

* shortcut
* identity mapping

#### 将深度网络推进百层/千层时代

ResNet-152 就已经是 **152 层深度**，但依然好训练、不退化。

#### ResNet 与 VGG 对比

| 对比项   | VGG  | ResNet  |
| ----- | ---- | ------- |
| 核心思想  | 堆卷积  | 加残差连接   |
| 可训练深度 | ~20层 | 1000+ 层 |
| 收敛速度  | 慢    | 快       |
| 退化问题  | 严重   | 解决      |
| 性能    | 较弱   | 远强      |

ResNet 是对传统“堆层”思路的革命。

---

### 5.10 总结

> **ResNet 通过引入残差连接成功解决深度 CNN 退化问题，使网络可以更深、更准、更好训练，是现代深度学习网络的奠基结构。**

---

# 目标检测模型
## 一、YOLO

### 1.1 YOLO 是什么？

YOLO（You Only Look Once）是一类 **端到端、单阶段（One-stage）目标检测模型**。

传统目标检测（R-CNN 系列）流程是两阶段：

1. 区域候选（生成可能含物体的框）
2. 对这些候选框做分类与回归

YOLO 则一次完成全部：

> **输入一张图片 → 通过一次 CNN 前向传播 → 直接输出所有目标的类别与位置**

所以特点是：

* **快**（实时检测）
* **端到端训练**
* **单次前向就能得到结果**

---

### 1.2 YOLO 的核心思想

YOLO 将目标检测视作 **一个回归问题**：

> 从图像像素直接回归：
>
> ```
> Bounding Box (x, y, w, h) +
> Confidence Score +
> Class Probability
> ```

图片被划分为 **S × S 网格（grid cell）**，每个网格负责预测该区域中的目标。

---

### 1.3 YOLO 输出格式（以 YOLOv1 为例）

图片被划分为：

```
S × S（如 7×7）
```

每个网格输出：

```
B 个边界框（如 2 个）
每个框包含：
  (x, y, w, h, confidence)
C 种类别概率
```

所以每个网格输出：

```
B×5 + C
```

整张图的输出是一个$（S × S × (B×5 + C)）$的张量。

---

### 1.4 YOLO 的预测内容解释

#### (1) Bounding Box

* `(x, y)`：相对于该网格 cell 的位置偏移（范围 0~1）
* `(w, h)`：相对于整张图的宽高
* 网络学习的是框的回归值

#### (2) Confidence Score

表示该格子的预测框：

```
Confidence = IOU(predicted box, ground truth box)
```

反映“框是否对”以及“框的位置是否准”。

#### (3) Class Probability

表示该网格内每类目标的概率，如：

```
Dog: 0.8
Cat: 0.1
Car: 0.05
...
```

最终：

```
Class score = Confidence × Class Probability
```

这就是推理时的最终得分。

---

### 1.5 YOLO 网络架构（以 YOLOv1 为例）

YOLOv1 使用：

* 卷积层提取特征
* 全连接层输出最终回归结果

典型结构：

```
卷积层 + 最大池化  × 多次
→ flatten
→ 全连接层（输出 S×S×(B×5+C) 维向量）
```

YOLOv1 借鉴 GoogLeNet 灵感：

* 不使用很深的网络
* 强调计算量效率

---

### 1.6 YOLO 的损失函数

YOLO 的损失由三部分组成：

#### (1) 位置回归损失

对：

```
x, y, sqrt(w), sqrt(h)
```

做均方误差（MSE）

使用开根号是为了：

* 缩小大框和小框的尺度差异
* 稳定训练

---

#### (2) 置信度损失

```
有物体的网格：希望 confidence → IOU
无物体的网格：希望 confidence → 0
```

YOLO 会对“无物体的格子置信度损失”给予更低权重（通常 λ_noobj = 0.5），防止背景过多影响训练。

---

#### (3) 分类损失

对有物体的 cell：

```
类别概率用 MSE 回归
```

---

#### 总体损失函数形式：

```
Loss = 位置损失 + 有物体置信度损失 + 无物体置信度损失 + 分类损失
```

每一项有不同权重，用于训练稳定性。

---

### 1.7 YOLO 的优缺点

#### YOLO的优点
| 优点     | 解释                    |
| ------ | --------------------- |
| 快      | 单次前向，无候选框生成，适合实时检测    |
| 端到端    | 输入图片就可以得到最终结果         |
| 全局处理目标 | 不像 R-CNN 只看 proposals |
| 泛化能力强  | 错误检测少，假阳性较低           |

---

#### YOLO 的缺点（V1 阶段）

1. 对小物体识别不够好
   因为每个 grid cell 只能负责少量对象。
2. 位置回归不够精确
   因为直接回归 box，而非更结构化的预测方式。
3. 分类在网格级预测
   降低了精度。

这些问题在后代版本中逐渐解决。

---

### 1.8 YOLO 的发展演进

#### ● YOLOv1（2016）

* 提出了 YOLO 概念
* 单次检测
* 快但精度不高

#### ● YOLOv2（YOLO9000）

改进：

* 引入 **Anchor**
* 采用更好的 backbone（Darknet-19）
* 多尺度训练（multi-scale）

速度与精度大幅提升。

#### ● YOLOv3

进一步提升：

* Darknet-53 backbone
* 三尺度检测（small/medium/big）
* 更强的特征表达

成为较长时间主流版本。

#### ● YOLOv4

* 改用 CSPDarknet
* 更高性能
* 强化推理与训练技巧（DropBlock、Mosaic 等）

#### ● YOLOv5（Ultralytics）

* PyTorch 重写
* 工程化、易部署、训练体验极佳

#### ● YOLOv7 / YOLOv8

* 更高精度
* 更快
* 继续强化多任务、多场景应用

---

### 1.9 YOLO 与其它目标检测算法对比

| 方法       | 类型        | 思路              | 代表网络                  |
| -------- | --------- | --------------- | --------------------- |
| R-CNN 系列 | Two-stage | “先找框，再分类”       | R-CNN / Fast / Faster |
| SSD      | One-stage | Anchor + 多尺度    | SSD                   |
| YOLO     | One-stage | **直接回归位置 + 类别** | YOLO all              |

YOLO 是最强调“速度”的检测路线。

---

### 1.10 YOLO 推理流程（简化）与代码框架

#### 推理流程
```
1）输入图片（resize，如 640×640）
2）CNN 提取特征（Backbone）
3）检测头输出框与类别分数
4）过滤置信度低的检测框
5）NMS（非极大值抑制）去除重叠框
6）得到最终检测结果
```

---

#### 代码逻辑核心

```python
import torch
import torch.nn as nn
from torchvision.ops import nms


class YOLO(nn.Module):
    def __init__(self, backbone, neck, head):
        super(YOLO, self).__init__()
        self.backbone = backbone   # 提取特征
        self.neck = neck           # FPN / PANet 等特征融合（YOLOv3+）
        self.head = head           # 检测头，输出框 + 类别

    def forward(self, x):
        """
        x: 输入图片 (B, C, H, W)
        return: 最终检测框
        """
        # 1. CNN特征提取
        features = self.backbone(x)

        # 2. 多层特征融合（可选 YOLOv3+ 才有 neck）
        features = self.neck(features)

        # 3. 检测头输出（通常会有3个尺度）
        outputs = self.head(features)

        # 4. 将 raw 输出转为 (B, boxes, 85)
        #   包含:
        #   cx, cy, w, h, obj_conf, class_scores...
        boxes, scores, classes = self.decode(outputs)

        # 5. 非极大值抑制
        final_boxes = self.apply_nms(boxes, scores, classes)

        return final_boxes


    def decode(self, outputs):
        """
        YOLO 的 decode 过程：
        1) 根据 anchor 解码 (cx, cy, w, h)
        2) sigmoid 置信度
        3) softmax / sigmoid 类别
        """
        # Pseudocode:
        boxes = ...
        scores = ...
        classes = ...
        return boxes, scores, classes


    def apply_nms(self, boxes, scores, classes, iou_thr=0.5):
        """
        用 torchvision.ops.nms 去除重叠框
        """
        keep = nms(boxes, scores, iou_thr)

        return boxes[keep], scores[keep], classes[keep]

```

这就是 YOLO 的常见推理流程。

---

### 1.11 总结

> **YOLO 是一种以回归方式解决目标检测问题的单阶段模型，特点是快、端到端、工程价值高，是实时检测的主流方法。**

---

## 二、R-CNN系列

### 2.1 什么是 R-CNN？

R-CNN 全名：

> **Regions with CNN features**

是 **2013 年 Ross Girshick 提出的目标检测方法**，是将 CNN 引入目标检测的开山之作。

在 R-CNN 之前，检测模型一般是：

* 手工特征（HOG、SIFT …）
* 传统分类器（SVM）

R-CNN首次提出：

> **先生成候选框（Region Proposal），再用 CNN 分类。**

一举把检测性能从 mAP≈35% 提升到 ≈66%，开启现代深度目标检测时代。

---

### 2.2 R-CNN 的工作流程

流程分三步：

```
输入图像
 ├─ ① Selective Search 生成 ~2000 个候选框
 ├─ ② 对每个候选框裁图 + CNN 提特征
 ├─ ③ SVM 分类 + 回归器做边框修正
输出：检测框 + 类别
```

逐步拆解：

---

#### ① 生成候选框（Region Proposal）

* 使用 **Selective Search**
* 输出约 **2000 个 region proposals**
* 带有“可能是物体的区域”

---

#### ② CNN 提取特征

* 每个候选框：

  * 从原图裁剪 → resize → 输入 CNN
  * 得到 4096 维特征向量（通常是 AlexNet）

---

#### ③ 分类与回归

* 特征送入 **SVM 分类器**
* 位置送入 **线性回归器**微调边框位置

---

#### R-CNN 最大缺点

| 问题   | 原因               |
| ---- | ---------------- |
| 计算太慢 | 每张图处理 2000 次 CNN |
| 训练复杂 | 分三步训练            |
| 浪费计算 | 候选框大量重叠，特征重复算    |

因此提出 **Fast R-CNN**。

---

### 2.3 Fast R-CNN（2015）

核心改进只有一句话：

> **整张图只做一次 CNN！**

---

#### Fast R-CNN 流程

```
输入图像
 ├─ CNN：提整张图特征图
 ├─ RoI Pooling：从特征图中裁出 proposal 特征
 ├─ 全连接层
 ├─ 一次性输出 分类 + 回归
```

**最大创新点：RoI Pooling**

* 把任意大小的候选框
* “池化”成固定大小（如 7×7）
* 这样才能送入 FC

---

#### Fast R-CNN 优点

| 优点       | 说明           |
| -------- | ------------ |
| CNN 只算一次 | 比 R-CNN 快几十倍 |
| 端到端训练    | 分类 + 回归一起预测  |
| 准确率更高    | 特征共享带来收益     |

但又有一个问题：

> 候选框（region proposal）仍然依赖 Selective Search，每张图要 **2秒**，仍然很慢。

于是出现 **Faster R-CNN**。

---

###  2.4 Faster R-CNN（2016）

Faster R-CNN 解决最后的问题：

> **取消 Selective Search，用 RPN 网络替代。**

RPN：Region Proposal Network（区域建议网络）

---

#### Faster R-CNN 流程

```
输入图像
 ├─ CNN：一次提特征图
 ├─ RPN：直接生成候选框（anchor）
 ├─ RoI Pooling：裁特征
 ├─ Fast R-CNN head
 ├─ 输出：分类 + 回归
```

所有步骤完全在神经网络内部完成。

---

#### RPN 的思想

* 在特征图每个位置生成 **9 个 anchor**
* 预测：

  * 是不是物体（二分类）
  * 边框偏移量（回归）

RPN 和检测网络 **共享前面的 CNN**，因此：

* 高效
* End-to-End
* 二者形成统一框架

#### Faster R-CNN 的优势

| 对比    | Faster R-CNN       |
| ----- | ------------------ |
| 候选框生成 | 完全由 CNN 预测         |
| 速度    | 比 Fast RCNN 提升 10× |
| 精度    | COCO / VOC 仍是 SOTA |

至此，R-CNN 系列形成强大的两阶段检测框架。

---

### 2.5 Mask R-CNN（2017）

> **Faster R-CNN 的升级版，不仅检测，还能预测实例分割 Mask。**

新增一个分支：

```
类别 + 边框 + 分割
```

架构：

```
输入 → Backbone → RPN → RoIAlign → 三个分支：
    ├─ 分类
    ├─ 回归
    ├─ 分割（FCN输出掩码）
```

最大创新：**RoIAlign**

* Faster R-CNN 的 RoI Pooling 有量化误差
* RoIAlign 使用双线性插值，不丢细节
* 分割精度大幅提升

Mask R-CNN 适用于：

* 实例分割
* 姿态估计（如 Keypoint R-CNN）
* 图像理解场景

---

### 2.6 R-CNN 系列对比表

| 模型           | 特点                   | Proposal 来源      | 速度    | 代表时间 |
| ------------ | -------------------- | ---------------- | ----- | ---- |
| R-CNN        | 第一次用 CNN 做检测         | Selective Search | 最慢    | 2013 |
| Fast R-CNN   | 整图一次 CNN，RoI Pooling | Selective Search | 快     | 2015 |
| Faster R-CNN | RPN 生成 proposal      | RPN              | 更快    | 2016 |
| Mask R-CNN   | 加分割分支，RoIAlign       | RPN              | 稍慢但更强 | 2017 |

---

### 2.7 R-CNN 系列的核心创新演化

```
R-CNN
  └─ 用 CNN 提特征，但速度极慢
Fast R-CNN
  └─ CNN 前提一次，RoI Pooling 改变速度格局
Faster R-CNN
  └─ 候选框也让 CNN 来预测（RPN）
Mask R-CNN
  └─ 加上分割分支，RoIAlign 提升精度
```

R-CNN 系列奠定了：

* 两阶段检测（Two-stage detector）
* Anchor + RoI 的检测哲学

直到后来：

* SSD、YOLO 才发展出一阶段检测体系。

---

### 2.8 R-CNN系列与YOLO，SSD卷积神经网络对比

参考您提供的MobileNet与LeNet5系列对比表格格式，下面是R-CNN系列、YOLO系列和SSD三大目标检测主流算法的详细对照表格：

| 项目      | R-CNN系列                         | YOLO系列                                 | SSD                |
| ------- | ------------------------------- | -------------------------------------- | ------------------ |
| 代表模型    | R-CNN, Fast R-CNN, Faster R-CNN | YOLOv1, YOLOv2, YOLOv3, YOLOv4, YOLOv5 | SSD300, SSD512     |
| 发表年份    | 2014（R-CNN）至2015（Faster R-CNN）  | 2016（YOLOv1）至2020+（YOLOv5等）            | 2016               |
| 主要任务    | 目标检测                            | 目标检测                                   | 目标检测               |
| 输入大小    | 依具体实现，通常多种尺寸                    | 448×448（YOLOv1），其他尺寸可变                 | 300×300 或 512×512  |
| 检测策略    | 先生成候选区域（Region Proposal），后分类和回归 | 端到端直接预测边界框和类别                          | 多尺度特征图直接预测边界框和类别   |
| 网络结构    | 基于卷积网络+区域建议网络（Faster R-CNN中）    | 单个卷积网络                                 | 基于卷积网络的多尺度特征金字塔结构  |
| 速度      | 较慢（尤其是R-CNN和Fast R-CNN）         | 快，适合实时                                 | 较快，介于R-CNN和YOLO之间  |
| 准确率     | 高，尤其Faster R-CNN表现优异            | 平衡速度和准确率                               | 准确率良好，特别对小目标表现较好   |
| 特征提取    | 使用深度卷积网络（如VGG, ResNet）          | 使用深度卷积网络（如Darknet）                     | 使用深度卷积网络（如VGG16）   |
| 是否端到端训练 | 部分端到端，Faster R-CNN较接近端到端        | 完全端到端训练                                | 完全端到端训练            |
| 优点      | 准确率高，检测精度好                      | 速度快，适合实时应用                             | 多尺度检测，兼顾速度和精度      |
| 缺点      | 速度慢，计算复杂度高                      | 对小目标检测精度较低                             | 相比YOLO速度稍慢，模型复杂度较高 |
| 主要创新点   | 区域提议+卷积特征抽取                     | 端到端单网络预测                               | 多尺度特征图预测           |
| 应用平台    | GPU加速服务器                        | 移动端、嵌入式设备、实时系统                         | GPU服务器及中端设备        |
| 激活函数    | ReLU                            | Leaky ReLU / Mish（后期版本）                | ReLU               |
| 是否使用锚框  | 是（候选框机制）                        | 是（锚框机制）                                | 是（多尺度锚框）           |
| 损失函数    | 分类损失 + 边框回归损失                   | 分类损失 + 边框回归损失                          | 分类损失 + 边框回归损失      |

