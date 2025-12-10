#include <stdio.h>
#include <stdlib.h>

#include "../include/tensor.h"
#include "../include/relu.h"

// 填充 input 为负数和正数混合，便于测试ReLU效果
void fill_input_data(Tensor *t)
{
    int size = t->N * t->C * t->H * t->W;
    for (int i = 0; i < size; i++) {
        t->data[i] = (float)(i - 4);  // -4, -3, ..., 0, 1, ...
    }
}

int main()
{
    printf("===== ReLU Test Start =====\n");

    Tensor input  = Tensor_init(1, 1, 3, 3);
    Tensor output = Tensor_init(1, 1, 3, 3);

    fill_input_data(&input);

    relu(&input, &output);

    printTensor(input,  "Input");
    printTensor(output, "Output (ReLU)");

    Tensor_free(&input);
    Tensor_free(&output);

    printf("===== ReLU Test End =====\n");
    return 0;
}
