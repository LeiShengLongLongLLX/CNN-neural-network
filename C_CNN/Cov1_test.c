#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define IDX4(n, c, h, w, C, H, W) \
    ((((n) * C + (c)) * H + (h)) * W + (w))


typedef struct {
    float* data;   // 数据存储在一维数组中
    int N;         // batch size
    int C;         // channel
    int H;         // height
    int W;         // width
} Tensor;


// 初始化 Tensor
Tensor Tensor_init(int N, int C, int H, int W)
{
    Tensor t;
    t.N = N;
    t.C = C;
    t.H = H;
    t.W = W;
    int size = N * C * H * W;
    t.data = (float*)malloc(sizeof(float) * size);
	return t;
}

// 释放 Tensor 内存
void Tensor_free(Tensor* t)
{
    if (t->data) free(t->data);
    t->data = NULL;
}

void printTensor(Tensor t, const char* name) {
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


// Conv2D 操作，NCHW 格式
void Conv2D(
    Tensor input,
    Tensor kernel,
    Tensor output)
{
    int N      = input.N;
    int C_in   = input.C;
    int H_in   = input.H;
    int W_in   = input.W;

    int C_out  = kernel.N;
    int K      = kernel.H;  // 卷积核大小 K×K

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

                    // 卷积核累加
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


int main()
{
    // 创建 Input (N=1, C=1, H=3, W=3)
    Tensor Input = Tensor_init(1, 1, 3, 3);

    // 创建 Kernel (N=1, C=1, H=3, W=3)
    Tensor Kernel = Tensor_init(1, 1, 3, 3);

    // 创建 Output (简单先分配 1×1×1×1，后面卷积再改)
    Tensor Output = Tensor_init(1, 1, 1, 1);

    // 给 Input 填入简单数据：0,1,2,3,...
    for (int i = 0; i < 9; i++) {
        Input.data[i] = (float)i;
    }

    // 给 Kernel 填入简单数据：全部为 1
    for (int i = 0; i < 9; i++) {
        Kernel.data[i] = 1.0f;
    }

    // Output 默认填 0
    Output.data[0] = 0.0f;


    // 打印三者内容
    printTensor(Input,  "Input");
    printTensor(Kernel, "Kernel");
    printTensor(Output, "Output");

    //========================
    // 4. 调用卷积
    //========================
    Conv2D(Input, Kernel, Output);

    //========================
    // 5. 打印输出结果
    //========================
    printf("Output:\n");
    for (int h = 0; h < Output.H; h++)
    {
        for (int w = 0; w < Output.W; w++)
        {
            float v = Output.data[
                IDX4(0, 0, h, w, Output.C, Output.H, Output.W)
            ];
            printf("%6.2f ", v);
        }
        printf("\n");
    }

    //========================
    // 6. 释放内存
    //========================
    Tensor_free(&Input);
    Tensor_free(&Kernel);
    Tensor_free(&Output);

    return 0;
}
