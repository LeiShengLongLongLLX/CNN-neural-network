#include "../../include/lenet5.h"
#include "../../include/conv2d.h"
#include "../../include/relu.h"
#include "../../include/maxpool.h"
#include "../../include/flatten.h"
#include "../../include/gemm.h"

// 结构同 int8 版，只是数据类型为 int32，对应使用 *_int32 算子

void lenet5_forward_int32(
    const Tensor* input,
    const Tensor* w_c1, const int32_t* b_c1,
    const Tensor* w_c3, const int32_t* b_c3,
    const Tensor* w_c5, const int32_t* b_c5,
    const Tensor* w_f6, const int32_t* b_f6,
    const Tensor* w_f7, const int32_t* b_f7,
    Tensor* output)
{
    // C1: conv5x5, 1->6, 32x32 -> 28x28
    Tensor c1_out   = Tensor_init(1, 6, 28, 28);
    Tensor c1_relu  = Tensor_init(1, 6, 28, 28);

    Conv2D_int32(input, w_c1, b_c1, &c1_out, 1, 0);
    relu_int32(&c1_out, &c1_relu);

    // S2: maxpool2x2, 6×28×28 -> 6×14×14
    Tensor s2_out = Tensor_init(1, 6, 14, 14);
    maxpool2D_int32(&c1_relu, &s2_out, 2, 2);

    // C3: conv5x5, 6->16, 14x14 -> 10x10
    Tensor c3_out  = Tensor_init(1, 16, 10, 10);
    Tensor c3_relu = Tensor_init(1, 16, 10, 10);

    Conv2D_int32(&s2_out, w_c3, b_c3, &c3_out, 1, 0);
    relu_int32(&c3_out, &c3_relu);

    // S4: maxpool2x2, 16×10×10 -> 16×5×5
    Tensor s4_out = Tensor_init(1, 16, 5, 5);
    maxpool2D_int32(&c3_relu, &s4_out, 2, 2);

    // C5: conv5x5, 16->120, 5x5 -> 1x1
    Tensor c5_out  = Tensor_init(1, 120, 1, 1);
    Tensor c5_relu = Tensor_init(1, 120, 1, 1);

    Conv2D_int32(&s4_out, w_c5, b_c5, &c5_out, 1, 0);
    relu_int32(&c5_out, &c5_relu);

    // flatten: [1,120,1,1] -> [1,120,1,1]
    Tensor flat = Tensor_init(1, 120, 1, 1);
    Flatten_int32(&c5_relu, &flat);

    // F6: GEMM 120 -> 84 + ReLU
    Tensor f6_out  = Tensor_init(1, 84, 1, 1);
    Tensor f6_relu = Tensor_init(1, 84, 1, 1);

    gemm_int32(&flat, w_f6, b_f6, &f6_out);
    relu_int32(&f6_out, &f6_relu);

    // F7: GEMM 84 -> 10（输出层，无激活）
    gemm_int32(&f6_relu, w_f7, b_f7, output);

    // 释放中间张量
    Tensor_free(&c1_out);
    Tensor_free(&c1_relu);
    Tensor_free(&s2_out);
    Tensor_free(&c3_out);
    Tensor_free(&c3_relu);
    Tensor_free(&s4_out);
    Tensor_free(&c5_out);
    Tensor_free(&c5_relu);
    Tensor_free(&flat);
    Tensor_free(&f6_out);
    Tensor_free(&f6_relu);
}

