#include "../../include/flatten.h"

// 展平函数
// 将 [N, C, H, W] 展平成 [N, 1, 1, C*H*W]
// input: 输入张量
// output: 输出张量
void Flatten(const Tensor* input, Tensor* output)
{
    int N = input->N;       // 批量大小
    int C = input->C;       // 通道数
    int H = input->H;       // 高度
    int W = input->W;       // 宽度

    int out_features = C * H * W; // 展平后的长度

    for (int n = 0; n < N; n++)      // 遍历批量
    {
        int flat_index = 0;   // 输出一维张量的索引

        for (int c = 0; c < C; c++)      // 遍历通道
        {
            for (int h = 0; h < H; h++)  // 遍历高度
            {
                for (int w = 0; w < W; w++)  // 遍历宽度
                {
                    int in_index = IDX4(n, c, h, w, C, H, W);
                    int out_index = IDX4(n, 0, 0, flat_index, 1, 1, out_features);

                    output->data[out_index] = input->data[in_index];    
                    flat_index++;
                }
            }
        }
    }
}
