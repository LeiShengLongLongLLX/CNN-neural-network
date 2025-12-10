#include <float.h>
#include "../../include/maxpool.h"

// MaxPool函数（支持一般 kernel_size, stride）
// output 张量的尺寸必须由你提前计算并创建好
// 逐元素池化操作
// input: 输入张量
// output: 输出张量
// kernel_size: 池化核大小
// stride: 池化步长

void maxpool_forward(const Tensor* input, Tensor* output,
                     int kernel_size, int stride)
{
    int N = input->N;          // 批量大小
    int C = input->C;          // 通道数
    int H_in = input->H;       // 输入高度
    int W_in = input->W;       // 输入宽度

    int H_out = output->H;     // 输出高度
    int W_out = output->W;     // 输出宽度

    for (int n = 0; n < N; n++)   // 遍历批量
    {                     
        for (int c = 0; c < C; c++)    // 遍历通道
        {                 
            for (int oh = 0; oh < H_out; oh++)  // 遍历输出高度
             {      
                for (int ow = 0; ow < W_out; ow++)  // 遍历输出宽度
                {  

                    float max_val = -FLT_MAX;  // 初始化值为极小值

                    // Pool kernel window
                    for (int kh = 0; kh < kernel_size; kh++) {        // 遍历池化核高度
                        for (int kw = 0; kw < kernel_size; kw++) {    // 遍历池化核宽度

                            int ih = oh * stride + kh;   // 计算输入张量对应位置的高度
                            int iw = ow * stride + kw;   // 计算输入张量对应位置的宽度

                            // 读取输入张量的值
                            float v = input->data[
                                IDX4(n, c, ih, iw, C, H_in, W_in)
                            ];

                            // 取最大值
                            if (v > max_val) max_val = v;
                        }
                    }

                    // 将最大值写入输出张量
                    output->data[
                        IDX4(n, c, oh, ow, C, H_out, W_out)
                    ] = max_val;
                }
            }
        }
    }
}
