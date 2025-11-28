#ifndef FLATTEN_H
#define FLATTEN_H

#include "tensor.h"

// 将 [N, C, H, W] 展平成 [N, 1, 1, C*H*W]
// input: 输入张量
// output: 输出张量
void Flatten(const Tensor* input, Tensor* output);

#endif
