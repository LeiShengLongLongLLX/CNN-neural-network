#include "cov2d.h"

// 卷积算子
void Conv2D(Tensor input, Tensor kernel, Tensor output)
{
    int N      = input.N;        // 输入张量的batch size
    int C_in   = input.C;        // 输入张量的通道数
    int H_in   = input.H;        // 输入张量的高度
    int W_in   = input.W;        // 输入张量的宽度

    int C_out  = kernel.N;       // 输出张量的通道数
    int K      = kernel.H;       // 卷积核的高度（假设卷积核为正方形）    

    int H_out = output.H;
    int W_out = output.W;

    for (int n = 0; n < N; n++)
    {
        for (int co = 0; co < C_out; co++)
        {
            for (int h = 0; h < H_out; h++)
            {
                for (int w = 0; w < W_out; w++)
                {
                    float sum = 0.0f;

                    for (int ci = 0; ci < C_in; ci++)
                    {
                        for (int kh = 0; kh < K; kh++)
                        {
                            for (int kw = 0; kw < K; kw++)
                            {
                                int ih = h + kh;
                                int iw = w + kw;

                                float v_in = input.data[
                                    IDX4(n, ci, ih, iw, C_in, H_in, W_in)
                                ];

                                float v_k = kernel.data[
                                    IDX4(co, ci, kh, kw, C_in, K, K)
                                ];

                                sum += v_in * v_k;
                            }
                        }
                    }

                    output.data[
                        IDX4(n, co, h, w, C_out, H_out, W_out)
                    ] = sum;
                }
            }
        }
    }
}
