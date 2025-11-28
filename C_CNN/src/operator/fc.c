#include "fc.h"
#include <stddef.h>

// 计算全连接层，支持带偏置bias
// input: 输入张量[N, 1, 1, input_features]
// weights: 权重张量[output_features, 1, 1, input_features]
// bias: 偏置 
// bias是一维数组，长度为 output_features，允许为 NULL 表示无偏置
// output: 输出张量[N, 1, 1, output_features]

void FullyConnected(const Tensor* input, const Tensor* weights, const float* bias, Tensor* output)
{
    int N = input->N;      // 批量大小
    int input_features = input->W;   // 假设输入是 [N, 1, 1, input_features]
    int output_features = weights->N; // weights 是 [output_features, 1, 1, input_features]

    for (int n = 0; n < N; n++)    // 遍历批量
    {
        for (int o = 0; o < output_features; o++) // 遍历输出向量元素
        {
            float sum = 0.0f;
            for (int i = 0; i < input_features; i++)  // 向量点乘 input * weights
            {
                float v_in = input->data[IDX4(n, 0, 0, i, 1, 1, input_features)];
                float w = weights->data[IDX4(o, 0, 0, i, 1, 1, input_features)];
                sum += v_in * w;
            }
            if (bias != NULL)  // 加上偏置
                sum += bias[o];
            
            output->data[IDX4(n, 0, 0, o, 1, 1, output_features)] = sum; // 存储输出向量
        }
    }
}
