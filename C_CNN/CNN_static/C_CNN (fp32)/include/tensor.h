#ifndef TENSOR_H
#define TENSOR_H

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
    float* data;    // 数据指针
    int N;          // 批量大小
    int C;          // 通道数
    int H;          // 高度
    int W;          // 宽度
} Tensor;

// tensor.c 中实现的函数声明
// CNN_static: 使用静态 arena 分配，避免 malloc/free。
// 容量单位：元素个数（不是字节）。可在编译时 -DTENSOR_ARENA_CAPACITY=... 覆盖。
#ifndef TENSOR_ARENA_CAPACITY
#define TENSOR_ARENA_CAPACITY (200000)
#endif

void   Tensor_arena_reset(void);
Tensor Tensor_init(int N, int C, int H, int W);
void   Tensor_free(Tensor* t);
void   printTensor(Tensor t, const char* name);

#endif
