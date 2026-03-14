#ifndef FC_H
#define FC_H

#include "tensor.h"

// 全连接层计算：
// input: [N, 1, 1, input_features]
// weights: [output_features, 1, 1, input_features]
// bias: 偏置 
// bias是一维数组，长度为 output_features，允许为 NULL 表示无偏置
// output: [N, 1, 1, output_features]
void FullyConnected(const Tensor* input, const Tensor* weights, const float* bias, Tensor* output);

#endif
