#ifndef CONV2D_H
#define CONV2D_H

#include "tensor.h"

// conv2d.c 中实现的函数声明
// 带 stride、padding 的通用卷积算子
// output 的尺寸必须由你提前计算并创建好
// input: 输入张量
// kernel: 卷积核张量
// output: 输出张量
// stride: 卷积步长
// padding: 填充大小
void Conv2D(const Tensor* input, const Tensor* kernel, Tensor* output,
            int stride, int padding);

#endif
