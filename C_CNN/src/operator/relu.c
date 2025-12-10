#include "../../include/relu.h"

// 逐元素 ReLU 激活函数实现
// input: 输入张量
// output: 输出张量

void relu_forward(const Tensor* input, Tensor* output)
{
    int total = input->N * input->C * input->H * input->W;  // 计算总元素数量

    for (int i = 0; i < total; i++)  // 遍历所有元素
    {
        float x = input->data[i];
        output->data[i] = (x > 0.0f) ? x : 0.0f;    // ReLU 操作
    }
}
