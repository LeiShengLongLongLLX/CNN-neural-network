/*
 * testlenet5_1.c — LeNet5 int32 推理连通性测试
 *
 * 使用 C_CNN 下的导出头文件：
 *   - ../../../lenet5_modelweight/lenet5_weights_int32.h
 *   - ../../../input_headers/testimg_inputs_int32.h
 *
 * 说明：
 *   - 权重与输入均为「按张量对称量化」的 int32；各张量带有各自的 float scale（见 .h）。
 *   - 当前 C 算子 Conv2D_int32 / gemm_int32 内部按整数直接相乘累加，不会乘 scale，
 *     因此数值上通常无法与 PyTorch 浮点推理一致；本测试主要用于验证链路可跑通、无崩溃。
 *   - 若需对齐训练精度，需要在整条链路上统一定点格式或在算子中引入 scale/移位。
 *
 * 编译示例（在 test/ 目录下，按你工程实际路径增删 src）：
 *   gcc -std=c99 -O2 -I../include -o testlenet5_1 \
 *       testlenet5_1.c \
 *       ../src/model/lenet5.c \
 *       ../src/operator/tensor.c \
 *       ../src/operator/conv2d.c \
 *       ../src/operator/relu.c \
 *       ../src/operator/maxpool.c \
 *       ../src/operator/flatten.c \
 *       ../src/operator/gemm.c
 */

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "../include/lenet5.h"
#include "../include/tensor.h"

#include "../../../lenet5_modelweight/lenet5_weights_int32.h"
#include "../../../input_headers/testimg_inputs_int32.h"

#define INPUT_H 32
#define INPUT_W 32
#define INPUT_LEN (INPUT_H * INPUT_W)

static Tensor wrap_weight(int N, int C, int H, int W, const int32_t *p)
{
    Tensor t;
    t.N = N;
    t.C = C;
    t.H = H;
    t.W = W;
    /* 算子只读 kernel；头文件中为 const static，此处去掉 const 以满足 Tensor 成员类型 */
    t.data = (int32_t *)(uintptr_t)p;
    return t;
}

static int argmax10(const Tensor *out)
{
    int best_i = 0;
    int32_t best_v = out->data[0];
    for (int i = 1; i < 10; ++i) {
        if (out->data[i] > best_v) {
            best_v = out->data[i];
            best_i = i;
        }
    }
    return best_i;
}

int main(void)
{
    static const struct {
        const char *tag;
        const int32_t *pixels;
        int expected_digit;
    } cases[] = {
        {"input_zero",  input_zero,  0},
        {"input_one",   input_one,   1},
        {"input_two",   input_two,   2},
        {"input_three", input_three, 3},
        {"input_four",  input_four,  4},
        {"input_five",  input_five,  5},
        {"input_six",   input_six,   6},
        {"input_seven", input_seven, 7},
        {"input_eight", input_eight, 8},
        {"input_nine",  input_nine,  9},
    };

    printf("===== LeNet5 int32 testlenet5_1 (exported .h weights + inputs) =====\n");

    Tensor w_c1 = wrap_weight(6, 1, 5, 5, conv1_weight);
    Tensor w_c3 = wrap_weight(16, 6, 5, 5, conv2_weight);
    Tensor w_c5 = wrap_weight(120, 16, 5, 5, conv3_weight);
    Tensor w_f6 = wrap_weight(84, 120, 1, 1, fc1_weight);
    Tensor w_f7 = wrap_weight(10, 84, 1, 1, fc2_weight);

    for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ++ci) {
        Tensor_arena_reset();

        Tensor input = Tensor_init(1, 1, INPUT_H, INPUT_W);
        Tensor output = Tensor_init(1, 10, 1, 1);
        if (input.data == NULL || output.data == NULL) {
            fprintf(stderr, "Tensor_init failed (arena too small? case %s)\n", cases[ci].tag);
            return 1;
        }

        memcpy(input.data, cases[ci].pixels, (size_t)INPUT_LEN * sizeof(int32_t));

        lenet5_forward_int32(
            &input,
            &w_c1, conv1_bias,
            &w_c3, conv2_bias,
            &w_c5, conv3_bias,
            &w_f6, fc1_bias,
            &w_f7, fc2_bias,
            &output);

        int pred = argmax10(&output);
        printf("[%s] expected digit %d \n argmax class %d \n (output_vector: ", cases[ci].tag,
               cases[ci].expected_digit, pred);
        for (int k = 0; k < 10; ++k) {
            printf("%d%s", (int)output.data[k], (k < 9) ? ", " : "");
        }
        printf(")\n");

        Tensor_free(&input);
        Tensor_free(&output);
    }

    printf("===== testlenet5_1 end =====\n");
    return 0;
}
