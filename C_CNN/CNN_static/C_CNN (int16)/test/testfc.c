#include <stdio.h>
#include <stdlib.h>

#include "../include/tensor.h"
#include "../include/fc.h"

// 填充 input 为 0..n递增数列
void fill_input_data(Tensor *t)
{
    int size = t->N * t->C * t->H * t->W;
    for (int i = 0; i < size; i++)
        t->data[i] = (int16_t)i;
}

// 填充 weights 为 1
void fill_weights_data(Tensor *t)
{
    int size = t->N * t->C * t->H * t->W;
    for (int i = 0; i < size; i++)
        t->data[i] = (int16_t)1;
}

// 填充 bias 为 0
void fill_bias_data(int16_t *bias, int size)
{
    for (int i = 0; i < size; i++)
        bias[i] = (int16_t)0;
}

int main()
{
    printf("===== FC Test Start =====\n");
    Tensor_arena_reset();

    // 假设 input 是 batch=1，C=1，H=1，W=4 (即4维向量)
    Tensor input = Tensor_init(1, 1, 1, 4);

    // weights 形状是 输出神经元数 × 输入神经元数，即 N=3（输出），C=4（输入），H=1，W=1
    Tensor weights = Tensor_init(3, 1, 1, 4);

    // bias 形状是 N=3，C=1，H=1，W=1
    int16_t bias[3];

    // output 形状是 batch=1，C=1，H=1，W=3 (即3维向量)
    Tensor output = Tensor_init(1, 1, 1, 3);

    fill_input_data(&input);
    fill_weights_data(&weights);
    fill_bias_data(bias, 3);

    FullyConnected(&input, &weights, bias, &output);

    printTensor(&input,   "Input");
    printTensor(&weights, "Weights");
    //printTensor(bias,    "Bias");
    printTensor(&output,  "Output (FC)");

    Tensor_free(&input);
    Tensor_free(&weights);
    //Tensor_free(&bias);
    Tensor_free(&output);

    printf("===== FC Test End =====\n");
    return 0;
}
