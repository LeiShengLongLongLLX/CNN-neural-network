#ifndef MAXPOOL_H
#define MAXPOOL_H

#include "tensor.h"

// MaxPool（支持一般 kernel_size, stride）
// output 的尺寸必须由你提前计算并创建好
// 逐元素池化操作
// input: 输入张量
// output: 输出张量
// kernel_size: 池化核大小
// stride: 池化步长
void maxpool_forward(const Tensor* input, Tensor* output,
                     int kernel_size, int stride);

#endif // MAXPOOL_H