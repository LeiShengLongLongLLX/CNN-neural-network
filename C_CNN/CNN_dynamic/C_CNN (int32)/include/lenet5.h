#ifndef LENET5_H
#define LENET5_H

#include "tensor.h"

// 使用已有算子在 C 端搭建的 LeNet-5 前向推理（int32）

void lenet5_forward_int32(
    const Tensor* input,
    const Tensor* w_c1, const int32_t* b_c1,
    const Tensor* w_c3, const int32_t* b_c3,
    const Tensor* w_c5, const int32_t* b_c5,
    const Tensor* w_f6, const int32_t* b_f6,
    const Tensor* w_f7, const int32_t* b_f7,
    Tensor* output);

#endif // LENET5_H

