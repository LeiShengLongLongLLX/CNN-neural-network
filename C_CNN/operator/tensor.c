#include <stdio.h>
#include <stdlib.h>

#include "tensor.h"

// 张量初始化
Tensor Tensor_init(int N, int C, int H, int W)
{
    Tensor t;
    t.N = N;
    t.C = C;
    t.H = H;
    t.W = W;

    int size = N * C * H * W;   // 所需内存空间大小
    t.data = (float*)malloc(sizeof(float) * size);   // 分配内存

    return t;
}


// 释放张量内存
void Tensor_free(Tensor* t)
{
    if (t->data) free(t->data);
    t->data = NULL;
}

// 打印张量内容
void printTensor(Tensor t, const char* name)
{
    printf("===== %s Tensor =====\n", name);
    printf("Shape: N=%d, C=%d, H=%d, W=%d\n", t.N, t.C, t.H, t.W);

    int index = 0;
    for (int n = 0; n < t.N; n++) {
        for (int c = 0; c < t.C; c++) {
            printf("  [N=%d, C=%d]\n", n, c);
            for (int h = 0; h < t.H; h++) {
                for (int w = 0; w < t.W; w++) {
                    printf("%.2f ", t.data[index++]);
                }
                printf("\n");
            }
            printf("\n");
        }
    }
}
