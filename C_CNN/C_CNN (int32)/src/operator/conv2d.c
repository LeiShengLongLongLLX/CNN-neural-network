#include "../../include/conv2d.h"

// 卷积算子
// output 的尺寸必须由你提前计算并创建好
// 没有饱和处理，结果可能会溢出
void Conv2D_int32(const Tensor* input, const Tensor* kernel, Tensor* output, int stride, int padding) 
{
    int N      = input->N;        // 输入张量的batch size
    int C_in   = input->C;        // 输入张量的通道数
    int H_in   = input->H;        // 输入张量的高度 
    int W_in   = input->W;        // 输入张量的宽度

    int K      = kernel->H;       // 卷积核的高度（假设卷积核为正方形）

    int C_out  = kernel->N;       // 输出张量的通道数（卷积核的数量）
    int H_out = output->H;        // 输出张量的高度
    int W_out = output->W;        // 输出张量的宽度

    for (int n = 0; n < N; n++)   // 遍历批量
    {
        for (int co = 0; co < C_out; co++) // 遍历输出通道
        {
            for (int oh = 0; oh < H_out; oh++)      // oh ow是输出的高宽坐标  遍历输出张量高度
            {
                for (int ow = 0; ow < W_out; ow++)  // 遍历输出张量宽度
                {
                    int32_t sum = 0;

                    // h_start / w_start 是卷积核对应当前输出像素时，在输入特征图上的窗口左上角坐标
                    int h_start = oh * stride - padding;    // 计算卷积核在输入张量上的起始位置（高）
                    int w_start = ow * stride - padding;    // 计算卷积核在输入张量上的起始位置（宽）

                    for (int ci = 0; ci < C_in; ci++)
                    {
                        for (int kh = 0; kh < K; kh++)
                        {
                            for (int kw = 0; kw < K; kw++)
                            {
                                // 计算卷积核点在输入张量上的位置
                                int ih = h_start + kh;
                                int iw = w_start + kw;

                                int32_t v_in = 0;

                                // 超出边界处理为0填充
                                if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in)
                                {
                                    v_in = input->data[
                                        IDX4(n, ci, ih, iw, C_in, H_in, W_in)
                                    ]; // input[n][ci][ih][iw]
                                }

                                int32_t v_k = kernel->data[
                                    IDX4(co, ci, kh, kw, C_in, K, K)
                                ]; // kernel[co][ci][kh][kw]

                                sum += v_in * v_k;
                            }
                        }
                    }

                    // 写入输出张量
                    output->data[
                        IDX4(n, co, oh, ow, C_out, H_out, W_out)
                    ] = sum; // output[n][co][oh][ow]
                }
            }
        }
    }
}
