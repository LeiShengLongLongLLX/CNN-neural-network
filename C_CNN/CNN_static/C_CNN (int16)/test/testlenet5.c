#include <stdio.h>
#include <stdlib.h>

#include "../include/tensor.h"
#include "../include/lenet5.h"

// 说明：
// - 这是一个“模型链路连通性”测试（能跑通 Conv/ReLU/Pool/Flatten/GEMM），不是精度对齐测试。
// - 权重全部填 1、bias 全部填 0，因此输出数值没有实际分类意义。
// - 输入/权重/输出张量形状约定与 `lenet5_forward_int16` 的声明一致。

static void fill_input(Tensor* t)
{
    int size = t->N * t->C * t->H * t->W;
    for (int i = 0; i < size; ++i) {
        t->data[i] = (int16_t)(i % 1024);
    }
}

static void fill_weights_ones(Tensor* t)
{
    int size = t->N * t->C * t->H * t->W;
    for (int i = 0; i < size; ++i) {
        t->data[i] = 1;
    }
}

static void fill_bias_zero(int16_t* b, int len)
{
    for (int i = 0; i < len; ++i) {
        b[i] = 0;
    }
}

int main(void)
{
    printf("===== LeNet-5 int16 Test Start =====\n");
    Tensor_arena_reset();

    // 输入：N=1, C=1, H=32, W=32
    Tensor input = Tensor_init(1, 1, 32, 32);
    fill_input(&input);

    // 权重张量（与 LeNet-5 结构对应）：
    // C1: 6×1×5×5, C3: 16×6×5×5, C5: 120×16×5×5
    // F6: 84×120,  F7: 10×84 （在工程里用 GEMM 实现全连接）
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
    int16_t b_c1[6];
    int16_t b_c3[16];
    int16_t b_c5[120];
    int16_t b_f6[84];
    int16_t b_f7[10];

    fill_bias_zero(b_c1, 6);
    fill_bias_zero(b_c3, 16);
    fill_bias_zero(b_c5, 120);
    fill_bias_zero(b_f6, 84);
    fill_bias_zero(b_f7, 10);

    // 输出：N=1, C=10, H=1, W=1（10 类 logits）
    Tensor output = Tensor_init(1, 10, 1, 1);

    // 前向推理
    lenet5_forward_int16(
        &input,
        &w_c1, b_c1,
        &w_c3, b_c3,
        &w_c5, b_c5,
        &w_f6, b_f6,
        &w_f7, b_f7,
        &output);

    // 打印输出向量
    printTensor(&output, "LeNet5 Output (int16)");

    Tensor_free(&input);
    Tensor_free(&w_c1);
    Tensor_free(&w_c3);
    Tensor_free(&w_c5);
    Tensor_free(&w_f6);
    Tensor_free(&w_f7);
    Tensor_free(&output);

    printf("===== LeNet-5 int16 Test End =====\n");
    return 0;
}

