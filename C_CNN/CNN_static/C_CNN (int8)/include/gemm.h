#ifndef GEMM_H
#define GEMM_H

#include "tensor.h"

// 全连接层实现，使用矩阵乘法计算
// input: [N, input_features, 1, 1] (A[M, K])
// weights: [output_features, input_features, 1, 1] (B[N, K])
// bias: 偏置
// bias是一维数组，长度为 output_features，允许为 NULL 表示无偏置
// output: [N, output_features, 1, 1] (C[M, N])
void gemm_int8(const Tensor* A, const Tensor* B, const int8_t* bias, Tensor* C);

#endif