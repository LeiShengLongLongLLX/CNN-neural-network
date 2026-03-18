# 项目介绍
## 项目结构
```CSS
CNN_C_Project/
│
├── include/
│   ├── tensor.h
│   ├── conv2d.h
│   ├── pooling.h
│   ├── relu.h
│   ├── fc.h
│   ├── utils.h
│   │
│   ├── model_lenet5.h
│   └── model_mobilenet.h
│
├── src/
│   ├── operator/
│   │   ├── tensor.c
│   │   ├── conv2d.c
│   │   ├── pooling.c
│   │   ├── relu.c
│   │   └── fc.c
│   │
│   ├── models/
│   │   ├── lenet5.c
│   │   └── mobilenet.c
│   │
│   ├── utils.c
│   └── main.c
│
├── models/
│   ├── lenet5/
│   │   └── weights.bin
│   └── mobilenet/
│       └── weights.bin
│
└── Makefile
```