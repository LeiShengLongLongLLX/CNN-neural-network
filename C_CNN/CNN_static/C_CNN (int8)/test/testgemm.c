#include <stdio.h>
#include <stdlib.h>

#include "../include/tensor.h"
#include "../include/gemm.h"

// 填充 input 为 0..n 递增数列
void fill_input_data(Tensor *t)
{
    int size = t->N * t->C * t->H * t->W;
    for (int i = 0; i < size; i++)
        t->data[i] = (int8_t)i;
}

// 填充 weights 为 1
void fill_weights_data(Tensor *t)
{
    int size = t->N * t->C * t->H * t->W;
    for (int i = 0; i < size; i++)
        t->data[i] = (int8_t)1;
}

// 填充 bias 为 0
void fill_bias_data(int8_t *bias, int size)
{
    for (int i = 0; i < size; i++)
        bias[i] = (int8_t)0;
}

int main()
{
    printf("===== GEMM (FC) Test Start =====\n");
    Tensor_arena_reset();

    // 假设 input 是 batch=1，input_features=4
    int batch_size      = 1;
    int input_features  = 4;
    int output_features = 3;

    // A: [M, K] -> [batch_size, input_features]
    Tensor input = Tensor_init(batch_size, input_features, 1, 1);

    // B: [N, K] -> [output_features, input_features]
    Tensor weights = Tensor_init(output_features, input_features, 1, 1);

    // bias: [output_features]
    int8_t bias[3];

    // C: [M, N] -> [batch_size, output_features]
    Tensor output = Tensor_init(batch_size, output_features, 1, 1);

    fill_input_data(&input);
    fill_weights_data(&weights);
    fill_bias_data(bias, output_features);

    // 使用 GEMM 实现全连接层：C = A * B + bias
    gemm_int8(&input, &weights, bias, &output);

    printTensor(&input,   "Input");
    printTensor(&weights, "Weights");
    // Bias 为一维数组，这里简单打印
    printf("===== Bias =====\n");
    for (int i = 0; i < output_features; i++)
        printf("%d ", bias[i]);
    printf("\n\n");

    printTensor(&output,  "Output (GEMM FC)");

    Tensor_free(&input);
    Tensor_free(&weights);
    Tensor_free(&output);

    printf("===== GEMM (FC) Test End =====\n");
    return 0;
}

