// input: ptr to N*C_in*H_in*W_in (we assume N=1 for simplicity)
// weight: C_out*C_in*K*K
// bias: C_out or NULL
// output: N*C_out*H_out*W_out (caller分配)
// stride, padding

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// conv2d for single batch (N=1), NCHW layout
void conv2d_nchw(
    const float *input,    // [C_in][H_in][W_in]
    const float *weight,   // [C_out][C_in][K][K]
    const float *bias,     // [C_out] 或 NULL
    float *output,         // [C_out][H_out][W_out]
    int C_in, int H_in, int W_in,
    int C_out, int K, int stride, int padding)
{
    int H_out = (H_in + 2*padding - K) / stride + 1;
    int W_out = (W_in + 2*padding - K) / stride + 1;

    // 初始化输出为 bias 或 0
    for (int co = 0; co < C_out; ++co) {
        for (int ho = 0; ho < H_out; ++ho) {
            for (int wo = 0; wo < W_out; ++wo) {
                int out_idx = (co * H_out + ho) * W_out + wo;
                output[out_idx] = bias ? bias[co] : 0.0f;
            }
        }
    }

    // 直接卷积（四层循环）
    for (int co = 0; co < C_out; ++co) {
        for (int ci = 0; ci < C_in; ++ci) {
            for (int ho = 0; ho < H_out; ++ho) {
                for (int wo = 0; wo < W_out; ++wo) {
                    float acc = 0.0f;
                    int in_h0 = ho * stride - padding;
                    int in_w0 = wo * stride - padding;
                    for (int kh = 0; kh < K; ++kh) {
                        int ih = in_h0 + kh;
                        if (ih < 0 || ih >= H_in) continue;
                        for (int kw = 0; kw < K; ++kw) {
                            int iw = in_w0 + kw;
                            if (iw < 0 || iw >= W_in) continue;
                            int in_idx = (ci * H_in + ih) * W_in + iw;
                            int w_idx = ((co * C_in + ci) * K + kh) * K + kw;
                            acc += input[in_idx] * weight[w_idx];
                        }
                    }
                    int out_idx = (co * H_out + ho) * W_out + wo;
                    output[out_idx] += acc;
                }
            }
        }
    }
}
