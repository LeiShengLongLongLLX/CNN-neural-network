# 项目介绍
## 项目结构
```makefile
cnn_project/
│
├── include/                    # 所有头文件（.h）放这里
│   ├── tensor.h                # 张量定义及相关操作接口
│   ├── conv2d.h                # 卷积算子接口
│   ├── relu.h                  # ReLU激活函数接口
│   ├── pooling.h               # 池化算子接口
│   ├── fc.h                    # 全连接层接口
|   |
│   ├── utils.h                 # 工具函数声明（打印、内存管理等）
│   └── cnn.h                   # CNN主流程及模型声明
│
├── src/                        # 源代码文件（.c）
│   ├── tensor.c                # 张量实现
│   ├── conv2d.c                # 卷积算子实现
│   ├── relu.c                  # ReLU实现
│   ├── pooling.c               # 池化实现
│   ├── fc.c                    # 全连接层实现
│   ├── utils.c                 # 工具函数实现
│   └── cnn.c                   # CNN主流程实现（调用各算子，模型推理流程）
│
├── models/                     # 模型相关数据文件（权重、偏置等）
│   ├── lenet_weights.bin       # 示例：LeNet模型权重二进制文件
│   └── ...                    # 其它模型文件
│
├── tests/                      # 测试代码和测试数据
│   ├── test_tensor.c           # 张量单元测试
│   ├── test_conv2d.c           # 卷积算子测试
│   ├── test_fc.c               # 全连接层测试
│   └── test_main.c             # 综合测试或示例程序
│
├── tools/                      # 辅助脚本或工具（比如模型权重转换等）
│   └── weight_converter.py     # Python脚本示例
│
├── Makefile                   # 构建脚本（编译规则）
├── README.md                  # 项目说明文档
└── LICENSE                    # 开源协议（如果有）
```