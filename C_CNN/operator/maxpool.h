#ifndef MAXPOOL_H
#define MAXPOOL_H

#include "tensor.h"

// MaxPool（支持一般 kernel_size, stride）
// output 的尺寸必须由你提前计算并创建好
void maxpool_forward(const Tensor* input, Tensor* output,
                     int kernel_size, int stride);

#endif // MAXPOOL_H