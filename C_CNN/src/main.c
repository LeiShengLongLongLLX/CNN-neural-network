#include "tensor.h"

int main()
{
    // Input (1,1,3,3)
    Tensor Input = Tensor_init(1, 1, 3, 3);

    // Kernel (1,1,3,3)
    Tensor Kernel = Tensor_init(1, 1, 3, 3);

    // Output (1,1,1,1)
    Tensor Output = Tensor_init(1, 1, 1, 1);

    // Input 数据：0~8
    for (int i = 0; i < 9; i++)
        Input.data[i] = (float)i;

    // Kernel 数据：全 1
    for (int i = 0; i < 9; i++)
        Kernel.data[i] = 1.0f;

    Output.data[0] = 0.0f;

    // 打印初始状态
    printTensor(Input, "Input");
    printTensor(Kernel, "Kernel");
    printTensor(Output, "Output");

    // 执行卷积
    Conv2D(Input, Kernel, Output);

    // 打印输出
    printf("===== Conv Output =====\n");
    for (int h = 0; h < Output.H; h++) {
        for (int w = 0; w < Output.W; w++) {
            printf("%6.2f ", Output.data[
                IDX4(0, 0, h, w, Output.C, Output.H, Output.W)
            ]);
        }
        printf("\n");
    }

    // 释放内存
    Tensor_free(&Input);
    Tensor_free(&Kernel);
    Tensor_free(&Output);

    return 0;
}
