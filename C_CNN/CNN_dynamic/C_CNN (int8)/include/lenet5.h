#ifndef LENET5_H
#define LENET5_H

#include "tensor.h"

// 使用已有算子在 C 端搭建的 LeNet-5 前向推理（int8）
// 约定：
// - 输入张量 input 形状：N=1, C=1, H=32, W=32
// - 卷积核权重张量：
//   * w_c1: [Cout=6, Cin=1,  H=5, W=5]
//   * w_c3: [Cout=16,Cin=6,  H=5, W=5]
//   * w_c5: [Cout=120,Cin=16,H=5, W=5]
// - 全连接层（GEMM）权重张量：
//   * w_f6: [Cout=84, Cin=120, H=1, W=1]  （120 -> 84）
//   * w_f7: [Cout=10, Cin=84,  H=1, W=1]  （84 -> 10）
// - bias 为对应输出通道长度的一维数组，可以为 NULL 表示无偏置
// - output 形状：N=1, C=10, H=1, W=1（10 类 logits）

void lenet5_forward_int8(
    const Tensor* input,
    const Tensor* w_c1, const int8_t* b_c1,
    const Tensor* w_c3, const int8_t* b_c3,
    const Tensor* w_c5, const int8_t* b_c5,
    const Tensor* w_f6, const int8_t* b_f6,
    const Tensor* w_f7, const int8_t* b_f7,
    Tensor* output);

#endif // LENET5_H

