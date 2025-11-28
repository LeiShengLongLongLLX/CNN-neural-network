#ifndef RELU_H
#define RELU_H

#include "tensor.h"

// 逐元素 ReLU：output = max(0, input)
// input: 输入张量
// output: 输出张量
void relu_forward(const Tensor* input, Tensor* output);

#endif // RELU_H
