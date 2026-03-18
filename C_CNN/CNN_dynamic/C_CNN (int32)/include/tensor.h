#ifndef TENSOR_H
#define TENSOR_H

#include <stdint.h>
#include <limits.h>

// NCHW 索引展开
// n: batch index
// c: channel index
// h: height index
// w: width index
// C: total channels
// H: total height
// W: total width
#define IDX4(n, c, h, w, C, H, W) \
    ((((n) * C + (c)) * H + (h)) * W + (w))

// Tensor 定义     四维张量
typedef struct {
    int32_t* data;    // 数据指针
    int N;          // 批量大小
    int C;          // 通道数
    int H;          // 高度
    int W;          // 宽度
} Tensor;

static inline int32_t saturate_int32_i64(int64_t x)
{
    if (x > INT32_MAX) return INT32_MAX;
    if (x < INT32_MIN) return INT32_MIN;
    return (int32_t)x;
}

// tensor.c 中实现的函数声明
Tensor Tensor_init(int N, int C, int H, int W);
void   Tensor_free(Tensor* t);
void   printTensor(const Tensor* t, const char* name);

#endif
