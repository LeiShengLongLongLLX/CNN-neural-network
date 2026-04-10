#include <stdio.h>
#include <stdlib.h>

#include "../include/tensor.h"
#include "../include/lenet5.h"

// 说明：
// - 这是一个“模型链路连通性”测试，用于验证 Lenet5 前向计算可以完整跑通。
// - 权重全部填 1、bias 全部填 0，因此输出仅用于观察数值范围/溢出行为，不代表真实分类结果。
// - 输入/权重/输出张量形状约定与 `lenet5_forward_int32` 的声明一致。

static void fill_input(Tensor* t)
{
    printf("input data allocated at %p\n", t->data);
    printf("  shape: %dx%dx%dx%d\n", t->N, t->C, t->H, t->W);

    int size = t->N * t->C * t->H * t->W;

    printf("Total elements: %d\n", size);
    printf("Memory range: %p - %p\n", t->data, t->data + size - 1);
    
    for (int i = 0; i < size; ++i) {
        t->data[i] = (int32_t)i;
    }
}

static void fill_weights_ones(Tensor* t)
{
    printf("weights data allocated at %p\n", t->data);
    printf("  shape: %dx%dx%dx%d\n", t->N, t->C, t->H, t->W);

    int size = t->N * t->C * t->H * t->W;

    printf("Total elements: %d\n", size);
    printf("Memory range: %p - %p\n", t->data, t->data + size - 1);

    for (int i = 0; i < size; ++i) {
        t->data[i] = i;
    }
}

static void fill_bias_zero(int32_t* b, int len)
{
    printf("bias data allocated at %p\n", b);
    printf("  shape: %d\n", len);
    printf("Total elements: %d\n", len);
    printf("Memory range: %p - %p\n", b, b + len - 1);

    for (int i = 0; i < len; ++i) {
        b[i] = 1;
    }
}

int main(void)
{
    printf("===== LeNet-5 int32 Test Start =====\n");
    Tensor_arena_reset();

    // 输入：N=1, C=1, H=32, W=32
    Tensor input = Tensor_init(1, 1, 32, 32);

    fill_input(&input);
    
    // {
    //     int size = input.N * input.C * input.H * input.W;
    //     for (int i = 0; i < size; ++i) {
    //         input.data[i] = 1;
    //     }
    // }

    // 权重张量（与 LeNet-5 结构对应）：
    // C1: 6×1×5×5, C3: 16×6×5×5, C5: 120×16×5×5
    // F6: 84×120,  F7: 10×84 （全连接由 GEMM 实现）
    Tensor w_c1 = Tensor_init(6, 1, 5, 5);
    Tensor w_c3 = Tensor_init(16, 6, 5, 5);
    Tensor w_c5 = Tensor_init(120, 16, 5, 5);
    Tensor w_f6 = Tensor_init(84, 120, 1, 1);
    Tensor w_f7 = Tensor_init(10, 84, 1, 1);

    fill_weights_ones(&w_c1);
    fill_weights_ones(&w_c3);
    fill_weights_ones(&w_c5);
    fill_weights_ones(&w_f6);
    fill_weights_ones(&w_f7);

    // bias：每层输出通道一个 bias（可传 NULL 表示无 bias）
    int32_t b_c1[6];
    int32_t b_c3[16];
    int32_t b_c5[120];
    int32_t b_f6[84];
    int32_t b_f7[10];

    fill_bias_zero(b_c1, 6);
    fill_bias_zero(b_c3, 16);
    fill_bias_zero(b_c5, 120);
    fill_bias_zero(b_f6, 84);
    fill_bias_zero(b_f7, 10);

    // 输出：N=1, C=10, H=1, W=1（10 类 logits）
    Tensor output = Tensor_init(1, 10, 1, 1);

    // 前向推理
    lenet5_forward_int32(
        &input,
        &w_c1, b_c1,
        &w_c3, b_c3,
        &w_c5, b_c5,
        &w_f6, b_f6,
        &w_f7, b_f7,
        &output);

    // 打印输出向量
    printTensor(&output, "LeNet5 Output (int32)");

    Tensor_free(&input);
    Tensor_free(&w_c1);
    Tensor_free(&w_c3);
    Tensor_free(&w_c5);
    Tensor_free(&w_f6);
    Tensor_free(&w_f7);
    Tensor_free(&output);

    printf("===== LeNet-5 int32 Test End =====\n");
    return 0;
}

