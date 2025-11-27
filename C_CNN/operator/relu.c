#include "relu.h"

void relu_forward(const Tensor* input, Tensor* output)
{
    int total = input->N * input->C * input->H * input->W;

    for (int i = 0; i < total; i++) {
        float x = input->data[i];
        output->data[i] = (x > 0.0f) ? x : 0.0f;
    }
}
