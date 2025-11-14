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

## 二、CNN卷积神经网络