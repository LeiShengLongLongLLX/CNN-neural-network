#include <stdio.h>
#include <stdlib.h>

#include "../include/tensor.h"
#include "../include/flatten.h"

// 填充 input 为 0..n递增数列
void fill_input_data(Tensor *t)
{
    int size = t->N * t->C * t->H * t->W;
    for (int i = 0; i < size; i++)
        t->data[i] = (float)i;
}

int main()
{
    printf("===== Flatten Test Start =====\n");

    Tensor input = Tensor_init(1, 2, 2, 2); // N=1, C=2, H=2, W=2

    fill_input_data(&input);

    // flatten 后输出是 N x (C*H*W) x 1 x 1  (常见形态)
    int flattened_size = input.C * input.H * input.W;
    Tensor output = Tensor_init(input.N, flattened_size, 1, 1);

    Flatten(&input, &output);

    printTensor(input,  "Input");
    printTensor(output, "Output (Flatten)");

    Tensor_free(&input);
    Tensor_free(&output);

    printf("===== Flatten Test End =====\n");
    return 0;
}
