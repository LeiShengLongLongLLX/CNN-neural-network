#include <stdio.h>
#include <stdlib.h>

#include "../include/tensor.h"
#include "../include/lenet5.h"

// 简单填充函数：输入为 0..(size-1)，权重全部为 1，bias 为 0
static void fill_input(Tensor* t)
{
    int size = t->N * t->C * t->H * t->W;
    for (int i = 0; i < size; ++i) {
        t->data[i] = (int8_t)(i % 128);
    }
}

static void fill_weights_ones(Tensor* t)
{
    int size = t->N * t->C * t->H * t->W;
    for (int i = 0; i < size; ++i) {
        t->data[i] = 1;
    }
}

static void fill_bias_zero(int8_t* b, int len)
{
    for (int i = 0; i < len; ++i) {
        b[i] = 0;
    }
}

int main(void)
{
    printf("===== LeNet-5 int8 Test Start =====\n");

    // 1. 构造输入：1×1×32×32
    Tensor input = Tensor_init(1, 1, 32, 32);
    fill_input(&input);

    // 2. 构造各层权重与 bias（这里只是简单填充，用于功能连通性测试）
    Tensor w_c1 = Tensor_init(6, 1, 5, 5);    // C1: 6x1x5x5
    Tensor w_c3 = Tensor_init(16, 6, 5, 5);   // C3: 16x6x5x5
    Tensor w_c5 = Tensor_init(120, 16, 5, 5); // C5: 120x16x5x5
    Tensor w_f6 = Tensor_init(84, 120, 1, 1); // F6: 84x120
    Tensor w_f7 = Tensor_init(10, 84, 1, 1);  // F7: 10x84

    fill_weights_ones(&w_c1);
    fill_weights_ones(&w_c3);
    fill_weights_ones(&w_c5);
    fill_weights_ones(&w_f6);
    fill_weights_ones(&w_f7);

    int8_t b_c1[6];
    int8_t b_c3[16];
    int8_t b_c5[120];
    int8_t b_f6[84];
    int8_t b_f7[10];

    fill_bias_zero(b_c1, 6);
    fill_bias_zero(b_c3, 16);
    fill_bias_zero(b_c5, 120);
    fill_bias_zero(b_f6, 84);
    fill_bias_zero(b_f7, 10);

    // 3. 输出张量：1×10×1×1
    Tensor output = Tensor_init(1, 10, 1, 1);

    // 4. 运行前向推理
    lenet5_forward_int8(
        &input,
        &w_c1, b_c1,
        &w_c3, b_c3,
        &w_c5, b_c5,
        &w_f6, b_f6,
        &w_f7, b_f7,
        &output);

    // 5. 打印输出
    printTensor(&output, "LeNet5 Output");

    // 6. 释放内存
    Tensor_free(&input);
    Tensor_free(&w_c1);
    Tensor_free(&w_c3);
    Tensor_free(&w_c5);
    Tensor_free(&w_f6);
    Tensor_free(&w_f7);
    Tensor_free(&output);

    printf("===== LeNet-5 int8 Test End =====\n");
    return 0;
}

