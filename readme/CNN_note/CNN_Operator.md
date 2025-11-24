# CNN算子

## 一、 总概（Overview）

CNN 的本质是一系列“算子（Operator）”作用于输入张量（Tensor），不断进行线性变换、非线性激活、通道融合、形态变换，最终输出网络预测结果。

对芯片和加速器而言：**算子是硬件执行的最小计算单位**，其性能直接决定芯片的吞吐、延迟和功耗。

---

## 1.1 算子（Operator）

### 1.1.1 计算类算子（Compute Operators）

计算类算子通常是计算量最大的，也是 AI 加速器优化的核心目标。

| 算子              | 说明                   | 代表公式                  |
| --------------- | -------------------- | --------------------- |
| 卷积 Conv         | CNN 的核心算子，用卷积核提取局部特征 | `Y = X * W + b`       |
| 矩阵乘 GEMM        | 完整连接层或卷积转换之后的通用形式    | `C = A × B`           |
| 激活函数 Activation | 引入非线性，使模型具有表达能力      | `ReLU, Sigmoid, GELU` |

特点：
✔ 占据网络 70%+ 的算力需求
✔ 硬件优化重点：MAC 数量、并行度、访存复用、流水化

---

### 1.1.2 数据变换类算子（Data Layout / Shape Operators）

这类算子本身计算量不大，但影响布局、带宽和数据利用率，是硬件调度的关键。

| 算子             | 作用                          |
| -------------- | --------------------------- |
| Reshape        | 改变 Tensor 形状但不改变内容          |
| Transpose      | 维度交换，常用于 NCHW ↔ NHWC        |
| Pooling        | 降维与特征压缩，如 MaxPool / AvgPool |
| Slice / Concat | 张量裁切、拼接                     |
| Upsample       | 上采样，用于分辨率提升                 |

硬件重点关注：

* 访存模式（stride access）
* Cache locality（空间/时间局部性）
* 避免无用搬运（Data reorder）

---

### 1.1.3 正则化与归一化算子（Normalization Operators）

| 算子                       | 场景             | 特点             |
| ------------------------ | -------------- | -------------- |
| Batch Normalization (BN) | CNN 标准结构       | 支持加速融合到 Conv 里 |
| Layer Normalization (LN) | Transformer 常用 | 与 batch 无关     |
| Group Norm               | 小 batch / 端侧模型 | 更稳定            |

BN 融合：
在推理（Inference）阶段，可以将 BN 的 α、β 合并进卷积权重和偏置中，减少一次访存和运算，芯片中常作为算子“优化手段”。

---

## 1.2 张量（Tensor）

> 张量是 CNN 中的核心数据结构

定义：

> 张量 = 多维数组（Multi-Dimensional Array）

---

### 1.2.1 数据维度（Rank）

常见维度：

| Tensor   | Rank | 示例              |
| -------- | ---- | --------------- |
| 标量       | 0    | `3.14`          |
| 向量       | 1    | `[128]`         |
| 矩阵       | 2    | `[128 × 256]`   |
| 特征图/卷积张量 | 4    | `N × C × H × W` |

CNN 中最典型的是 **四维张量**：

```
N：Batch size
C：Channel（通道）
H：Height（高度）
W：Width（宽度）
```

---

### 1.2.2 张量布局（Layout）

#### ① NCHW（Row-major）

```
Batch → Channel → Height → Width
```

* PyTorch 默认
* 计算友好、连续访问友好
* 适合加速器 SIMD + Tile 复用

#### ② NHWC

* TensorFlow 与 ARM 平台常用
* 有利于 depthwise conv 的连续访存

#### ③ Tile Blocking（片块化布局）

硬件为了提高缓存命中率，会把大张量分 Tile 存储，例如：

```
C blocked into C/16 groups
```

优势：

* 提高数据复用
* 符合 systolic array / MAC 阵列读取模式
* 减少 DRAM 往返

---

### 1.2.3 dtype（数据格式）

Tensor 上每个数据的存储格式，例如：

| 类型   | 示例                 |
| ---- | ------------------ |
| 浮点   | FP32 / FP16 / BF16 |
| 定点   | INT8 / INT4        |
| 混合精度 | FP16 accum FP32    |

dtype 影响：

* 精度（能否收敛）
* 存储空间
* 访存带宽
* 芯片算力（TOPS / TFLOPS）

---

## 1.3 参数（Parameters）

参数包含模型训练过程中可学习的全部数值。

---

### 1.3.1 权重（Weights）

* 卷积核参数，形如：

```
W: (C_out × C_in × K × K)
```

* 全连接层：

```
W: (input_dim × output_dim)
```

占模型参数 90%+。

---

### 1.3.2 偏置（Bias）

* 按通道或神经元加法修正输出
* 大小一般为：

```
C_out
```

* 可在推理中融合进 BN 或卷积权重

---

### 1.3.3 参数规模与带宽开销

存储量（Bytes）：

```
Param_Size = NumParams × dtype_bit / 8
```

例如 ResNet50 参数约 26M：

| dtype | 占用      |
| ----- | ------- |
| FP32  | ≈ 100MB |
| INT8  | ≈ 25MB  |

在 AI 加速器中，
**参数是否能放进片上 SRAM，影响吞吐和能耗。**

---

## 1.4 数据类型（Numeric Type）

### 1.4.1 浮点（Floating Point）

| 类型   | 位宽          | 特点             |
| ---- | ----------- | -------------- |
| FP32 | 32-bit      | 标准训练精度         |
| FP16 | 16-bit IEEE | 精度与速度折中        |
| BF16 | 16-bit      | FP32 动态范围，适合训练 |

FP16 / BF16 可 **减半带宽，提高吞吐**。

---

### 1.4.2 整型（INT8 / INT4）

特点：

* 量化后的推理常用
* 访存带宽 4～8 倍缩减
* AI 芯片计算密度翻倍

| 类型   | 常见用途      |
| ---- | --------- |
| INT8 | 工业标准端侧推理  |
| INT4 | 更高压缩、牺牲精度 |

---

### 1.4.3 混合精度（AMP：Automatic Mixed Precision）

业界标准配置：

```
FP16 → computation
FP32 → accumulation
```

优势：

* 训练速度 ×2～×6
* 收敛不受影响

---

### 1.4.4 量化 / 反量化（Quant / Dequant）

#### 量化公式

将浮点变成整型：

```
INT = FP / scale + zero_point
```

#### 反量化

执行推理后恢复为浮点：

```
FP = (INT - zero_point) × scale
```

硬件重要指标：

* 是否支持 **weight-only quant**
* 是否支持 **ACT+W 双量化**
* 是否支持 **on-chip dequant**

决定吞吐、能耗与模型精度。

---

## 1.5 总结

CNN 算子体系结构可以理解为：

> Tensor + Operator + Numeric Format + Parameters

在 AI 芯片中：

* 计算算子决定算力（TOPS / TFLOPS）
* 张量布局决定访存效率
* dtype 决定吞吐和能耗
* 参数大小决定片上存储与带宽压力

掌握这些概念，有助于：

✔ AI 芯片架构设计
✔ CNN 推理优化
✔ Tile 算法实现
✔ Cache/Bandwidth 建模
✔ LUT / MAC / Systolic Array 评估

---

# 二、CNN计算类算子（Compute Operators）

## 2.1 卷积算子（Conv）

### 1. 数学定义

卷积算子是局部加权求和操作，定义为：

$$
O[n][c_o][h][w] = \sum_{c_i=0}^{C_{in}-1} \sum_{i=0}^{K-1} \sum_{j=0}^{K-1} I[n][c_i][h \cdot S + i][w \cdot S + j] \times W[c_o][c_i][i][j] + B[c_o]
$$

* $N$ 是 batch 大小
* $C_{in}$, $C_{out}$ 是输入、输出通道数
* $(H, W)$ 是特征图高度宽度
* $K$ 是卷积核大小
* $S$ 是步长（stride）
* $B$ 是偏置向量

---

### 2. 输入输出张量格式

常用布局包括：

* **NCHW**（Batch, Channel, Height, Width）：CUDA默认
* **NHWC**（Batch, Height, Width, Channel）：TensorFlow多用
* **Blocked / Tiled**：硬件向量单元对齐友好，提升数据访问效率

卷积输入张量形状：

$$
I \in \mathbb{R}^{N \times C_{in} \times H_{in} \times W_{in}}
$$

卷积核权重形状：

$$
W \in \mathbb{R}^{C_{out} \times C_{in} \times K \times K}
$$

输出张量形状：

$$
O \in \mathbb{R}^{N \times C_{out} \times H_{out} \times W_{out}}
$$

其中：

$$
H_{out} = \frac{H_{in} + 2P - K}{S} + 1, \quad W_{out} = \frac{W_{in} + 2P - K}{S} + 1
$$

$P$ 是 padding（填充）。

---

### 3. 计算量（MACs 计算）

卷积算子总乘加次数为：

$$
MACs = N \times C_{out} \times H_{out} \times W_{out} \times (C_{in} \times K^2)
$$

* **乘加算子（Multiply-Accumulate）**是 AI 计算中的核心性能指标
* 2 × MACs 为 FLOPs 计数（乘法 + 加法）

由于计算量大，是 CNN 推理和训练的瓶颈。

---

### 4. 数据访问与重用

卷积的访存特性：

* **输入激活（Feature Map）**在空间和通道上高度重用（滑动窗口重叠）
* **卷积核权重**对整个输出特征图重复使用
* **输出激活**进行累加更新

硬件优化依赖：

* **缓存策略**（Tile Blocking，Loop Tiling）减少访存
* **数据复用**最大化芯片内存带宽利用率
* **流水线调度**和 **并行计算单元**协同

访存瓶颈常大于计算瓶颈，影响性能。

---

### 5. 常见优化方法

1. **im2col + GEMM**

   * 将卷积转化为矩阵乘法，提高通用计算库利用率（如 BLAS）
   * 但增加内存消耗和中间数据拷贝

2. **Winograd 卷积**

   * 减少乘法次数，特别是 3×3 卷积
   * 计算量约减半，带宽需求降低

3. **FFT 卷积**

   * 利用频域卷积定理，适合大核卷积
   * 计算量优势明显，但实现复杂

4. **深度可分离卷积（Depthwise + Pointwise）**

   * 降低计算复杂度，广泛用于轻量级网络如 MobileNet

5. **Sparse 卷积与剪枝**

   * 利用参数稀疏性跳过无效计算

---

### 6. 在硬件（NPU）上的实现特点

* **数据流架构**：如 Weight Stationary、Output Stationary、No Local Reuse 等
* **片上缓存**：优化数据复用，减少访存压力
* **并行度**：多计算单元同时工作，充分利用算力
* **定点计算**：INT8/INT4 量化减低功耗和带宽
* **权重融合**：如卷积 + BN 合并，减少计算和访存
* **流水线和指令集支持**：专用指令加速卷积计算

---

## 2.2 矩阵乘（GEMM）

### 1. 矩阵乘数学形式

基本形式：

$$
C = A \times B
$$

其中：

* $A\in \mathbb{R}^{M \times K}$
* $B\in \mathbb{R}^{K \times N}$
* $C\in \mathbb{R}^{M \times N}$

元素计算：

$$
C_{ij} = \sum_{k=1}^{K} A_{ik} \times B_{kj}
$$

---

### 2. 为什么 Conv 能转成 GEMM

卷积可通过**im2col**方法转成矩阵乘：

* 输入特征图通过 **im2col** 转换为矩阵 $(A)$
* 卷积核展平为矩阵 $(B)$
* 计算矩阵乘 $(C = A \times B)$ 得到卷积结果

该方法利用高度优化的 GEMM 库（如 cuBLAS）提升性能。

---

### 3. 矩阵分块（Tile）

大矩阵乘分为多个小块（tile）处理：

* 提升缓存命中率
* 利用寄存器和片上缓存降低访存延迟
* 适配硬件向量宽度（SIMD/VPU）

Tile 参数设计对性能影响极大。

---

### 4. SIMD / VPU 加速

* SIMD（单指令多数据）指令集对矩阵乘并行计算提供硬件支持
* VPU（向量处理单元）能同时计算多个矩阵元素乘加
* 需高效数据排列以保证连续访问和带宽利用

---

### 5. Sparse GEMM

* 利用矩阵稀疏特性跳过零元素乘法
* 需要特殊存储格式（如 CSR、COO）
* 芯片支持稀疏计算可显著降低计算量和能耗

---

## 2.3 激活函数（Activation）

### 1. 常见函数定义和性质

| 函数           | 公式                          | 特性          | 应用           |
| ------------ | --------------------------- | ----------- | ------------ |
| ReLU         | $(f(x) = \max(0,x))$          | 稀疏激活，非线性    | 广泛应用，简单高效    |
| Sigmoid      | $(f(x) = \frac{1}{1+e^{-x}})$ | 饱和，梯度消失风险   | 二分类输出层       |
| Tanh         | $(f(x) = \tanh(x))$           | 输出区间 [-1,1] | 传统网络中使用      |
| Leaky ReLU   | $(f(x) = \max(\alpha x, x))$  | 防止死神经元      | 改进版ReLU      |
| GELU         | $(f(x) = x \Phi(x))$          | 平滑近似ReLU    | Transformer等 |
| SiLU / Swish | $(f(x) = x \cdot sigmoid(x))$ | 平滑，性能好      | 现代网络         |

---

### 2. 推理中是否 In-place

* 许多激活函数支持**In-place计算**，直接覆盖输入内存，节省内存开销
* ReLU 是典型 In-place 函数
* Sigmoid、Tanh 由于需要保留输入数据用于反向传播，训练阶段通常不开启 In-place

---

### 3. LUT / CORDIC / Piecewise Approx

硬件实现中，非线性激活函数采用近似计算：

* **LUT（查找表）**：速度快，存储开销，适合小范围和低精度
* **CORDIC 算法**：迭代计算三角函数、指数函数等，无乘法硬件需求
* **分段线性逼近（Piecewise Linear Approximation）**：计算量和精度平衡

---

### 4. 指令级加速与混合精度

* 现代 AI 芯片支持激活函数专用指令（如 ReLU 指令）
* 混合精度计算（FP16/INT8）降低算力和带宽消耗
* 激活函数的定点实现需结合量化方案设计

---

### 5. 总结

| 章节            | 核心内容                               |
| ------------- | ---------------------------------- |
| 2.1 卷积算子      | 局部加权和操作，计算量大，访存复杂，支持多种卷积优化及硬件数据流设计 |
| 2.2 矩阵乘（GEMM） | 通用高效计算核心，卷积转 GEMM，矩阵分块与 SIMD 加速关键  |
| 2.3 激活函数      | 非线性引入，激活函数种类多，硬件中多采用近似算法与专用指令加速    |

---

# 三、CNN数据变换类算子

