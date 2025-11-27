#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define IDX3(c,h,w,H,W)   ((c)*(H)*(W) + (h)*(W) + (w))
#define IDX4(co,ci,h,w,CI,H,W)  (((co)*(CI) + (ci))*(H)*(W) + (h)*(W) + (w))

// Tensor结构（简化）  batch size N=1 当前这个张量是三维的
typedef struct {
    float* data;        // Tensor数据指针，指向存储的数据
    int N;              // batch size
    int C;              // channels
    int H;              // height
    int W;              // width
} Tensor;

//============================
//  conv 计算：保存到 conv_temp
//============================
void Conv_compute(
    Tensor input,            // [C_in][H_in][W_in]
    Tensor kernel,           // [C_out][C_in][K][K]
    Tensor conv_temp,        // [C_out][C_in][H_out][W_out] (accumulate)
    int stride,
    int padding)

{
    int C_in = input.C;
    int H_in = input.H;
    int W_in = input.W;

    int C_out = kernel.C;
    int K = kernel.H;   // 假定 K×K
    int H_out = conv_temp.H;
    int W_out = conv_temp.W;   // 不应该是kernel.W?

    // 逐通道卷积
    for (int co = 0; co < C_out; co++) {
        for (int ci = 0; ci < C_in; ci++) {

            for (int ho = 0; ho < H_out; ho++) {
                for (int wo = 0; wo < W_out; wo++) {

                    // 当前输出像素对应输入特征图的左上角坐标
                    int h0 = ho * stride - padding;
                    int w0 = wo * stride - padding;

                    // 每个卷积核元素
                    for (int kh = 0; kh < K; kh++) {
                        int ih = h0 + kh;
                        if (ih < 0 || ih >= H_in) continue;

                        for (int kw = 0; kw < K; kw++) {
                            int iw = w0 + kw;
                            if (iw < 0 || iw >= W_in) continue;

                            float v_in = input.data[IDX3(ci, ih, iw, H_in, W_in)];
                            float v_k  = kernel.data[IDX4(co, ci, kh, kw, C_in, K, K)];

                            conv_temp.data[IDX4(co, ci, ho, wo, C_in, H_out, W_out)] += v_in * v_k;
                        }
                    }
                }
            }
        }
    }
}


//=====================================
//   通道求和 + 加上 bias
//=====================================
void Conv_reduce_and_bias(
    Tensor conv_temp,   // [C_out][C_in][H_out][W_out]
    float* bias,
    Tensor output)      // [C_out][H_out][W_out]
{
    int C_out = output.C;
    int H_out = output.H;
    int W_out = output.W;
    int C_in  = conv_temp.C;

    for (int co = 0; co < C_out; co++) {
        for (int ho = 0; ho < H_out; ho++) {
            for (int wo = 0; wo < W_out; wo++) {

                float acc = 0.0f;

                for (int ci = 0; ci < C_in; ci++) {
                    acc += conv_temp.data[IDX4(co, ci, ho, wo, C_in, H_out, W_out)];
                }

                // 写入输出
                output.data[IDX3(co, ho, wo, H_out, W_out)] = 
                    bias ? (acc + bias[co]) : acc;
            }
        }
    }
}


//=====================================
//     主接口：Conv2D
//=====================================
void Conv2D(
    Tensor input,
    Tensor output,
    Tensor kernel,
    float* bias,
    float* workspace,    // conv_temp 外部提供
    int stride,
    int padding)
{
    int C_out = kernel.C;
    int C_in  = input.C;

    int K = kernel.H;  // K × K

    int H_out = (input.H + 2*padding - K) / stride + 1;
    int W_out = (input.W + 2*padding - K) / stride + 1;

    // 设置 conv_temp Tensor
    Tensor conv_temp = {
        .data = workspace,
        .C = C_out,
        .H = C_in,
        .W = H_out * W_out    // flatten (C_out,C_in,H_out,W_out)
    };

    int temp_size = C_out * C_in * H_out * W_out;
    memset(conv_temp.data, 0, sizeof(float) * temp_size);

    //======== 卷积累积 ========
    Conv_compute(input, kernel, conv_temp, stride, padding);

    //======== 求和 + bias ========
    Conv_reduce_and_bias(conv_temp, bias, output);
}
