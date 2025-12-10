#include <stdio.h>
#include <stdlib.h>

#include "../include/tensor.h"
#include "../include/maxpool.h"

// 填充 input 为 0..n递增数列
void fill_input_data(Tensor *t)
{
    int size = t->N * t->C * t->H * t->W;
    for (int i = 0; i < size; i++)
        t->data[i] = (float)i;
}

int main()
{
    printf("===== MaxPool Test Start =====\n");

    Tensor input  = Tensor_init(1, 1, 4, 4);

    fill_input_data(&input);

    int pool_size = 2;
    int stride = 2;

    int H_out = (input.H - pool_size) / stride + 1;
    int W_out = (input.W - pool_size) / stride + 1;
    Tensor output = Tensor_init(input.N, input.C, H_out, W_out);

    maxpool2D(&input, &output, pool_size, stride);

    printTensor(input,  "Input");
    printTensor(output, "Output (MaxPool)");

    Tensor_free(&input);
    Tensor_free(&output);

    printf("===== MaxPool Test End =====\n");
    return 0;
}
