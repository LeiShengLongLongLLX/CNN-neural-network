#include <stdio.h>

#include "../../include/tensor.h"

static int32_t g_tensor_arena[TENSOR_ARENA_CAPACITY];
static int g_tensor_arena_used = 0;

void Tensor_arena_reset(void)
{
    g_tensor_arena_used = 0;
}

// 张量初始化
Tensor Tensor_init(int N, int C, int H, int W)
{
    Tensor t;
    t.N = N;     // 设置张量batch_size
    t.C = C;     // 设置张量通道数
    t.H = H;     // 设置张量高度
    t.W = W;     // 设置张量宽度

    int size = N * C * H * W;   // 所需元素数量
    if (size <= 0 || g_tensor_arena_used + size > TENSOR_ARENA_CAPACITY) {
        t.data = NULL;
        return t;
    }
    t.data = &g_tensor_arena[g_tensor_arena_used];
    g_tensor_arena_used += size;

    return t;
}


// 释放张量内存
void Tensor_free(Tensor* t)
{
    t->data = NULL;
}

// 打印张量内容
void printTensor(const Tensor *t, const char* name)
{
    printf("===== %s Tensor =====\n", name);
    printf("Shape: N=%d, C=%d, H=%d, W=%d\n", t->N, t->C, t->H, t->W);

    int index = 0;
    for (int n = 0; n < t->N; n++) {
        for (int c = 0; c < t->C; c++) {
            printf("C=%d:\n", c);
            for (int h = 0; h < t->H; h++) {
                for (int w = 0; w < t->W; w++) {
                    printf("%d ", t->data[index++]); // 打印张量
                }
                printf("\n");
            }
            printf("\n");
        }
    }
}
