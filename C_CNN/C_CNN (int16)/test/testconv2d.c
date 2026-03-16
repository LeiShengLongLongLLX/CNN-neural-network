#include <stdio.h>
#include <stdlib.h>
#include "../include/tensor.h"
#include "../include/conv2d.h"

// 生成固定数据便于测试
void fill_input_data(Tensor *t)
{
    int size = t->N * t->C * t->H * t->W;
    for (int i = 0; i < size; i++)
        t->data[i] = (int16_t)i;
}

void fill_kernel_data(Tensor *k)
{
    int size = k->N * k->C * k->H * k->W;
    for (int i = 0; i < size; i++)
        k->data[i] = (int16_t)1;       // 简单卷积核：全部 1
}

int main()
{
    printf("===== Conv2D Test Start =====\n");

    // 1. 创建 feature / Kernel
    Tensor feature  = Tensor_init(1, 1, 8, 8);// Input: 1×1×32×32
    Tensor kernel = Tensor_init(1, 1, 3, 3);// Kernel: 6×1×5×5 (输出通道数=6，输入通道数=1，卷积核大小=5×5)

    fill_input_data(&feature);

    fill_kernel_data(&kernel);

    // 2. 计算 Output 尺寸 (valid 卷积)
    int stride = 1;
    int padding = 0;

    int H_out = (feature.H + 2 * padding - kernel.H) / stride + 1; 
    int W_out = (feature.W + 2 * padding - kernel.W) / stride + 1;

    Tensor output = Tensor_init(feature.N, kernel.N, H_out, W_out);

    // 3. 调用卷积
    Conv2D_int16(&feature, &kernel, NULL, &output, stride, padding);  // stride=1, padding=0

    // 4. 打印结果
    printTensor(&feature, "Input");
    printTensor(&kernel, "Kernel");
    printTensor(&output, "Output");

    // 5. 释放内存
    Tensor_free(&feature);
    Tensor_free(&kernel);
    Tensor_free(&output);

    printf("===== Conv2D Test Finished =====\n");
    return 0;
}
