// #include <stdio.h>
// #include <stdint.h>
#include "../../include/gemm.h"

// 矩阵乘法
// 全连接层，输入K个神经元，输出N个神经元
// input: A: [M, K], B^T: [N, K], C: [M, N], Bias: [N]
// output: C = A * B + Bias

void gemm_int8(const Tensor* A, 
    const Tensor* B,
    const int8_t* bias, 
    Tensor* C) {

    int M = A->N; // batch size
    int K = A->C; // input features
    int N = C->C; // output features

    for (int i = 0; i < M; i++) {       // 遍历batch
        for (int j = 0; j < N; j++) {   // 遍历output features
            int64_t sum = 0;

            for (int k = 0; k < K; k++) {
                // 遍历input features
                int a_idx = i * K + k;
                int b_idx = j * K + k;

                // 另一种实现方式
                // int a_idx = IDX4(i, k, 0, 0, K, 1, 1); // A[i][k][0][0]
                // int b_idx = IDX4(j, k, 0, 0, N, 1, 1); // B^T[j][k][0][0]

                int32_t a_val = A->data[a_idx]; // A[i][k][0][0]
                int32_t b_val = B->data[b_idx]; // (B^T)[j][k][0][0]

                //printf("  A[%d][%d] (input features %d)=%d, B[%d][%d] (input weights %d)=%d, result=%d\n",
                //    i, k, a_idx, a_val, j, k, b_idx, b_val, a_val * b_val);

                sum += (int64_t)a_val * (int64_t)b_val; // C[i][j] += A[i][k] * B^T[j][k]
            }

            if (bias) {
                sum += bias[j];
                // printf("  bias[%d]=%d\n", j, bias[j]);
            }

            // printf("  sum = %lld (0x%llx)\n", sum, sum);

            C->data[i * N + j] = (int32_t)sum;
            // printf("  C[%d][%d] = %d (0x%x)\n\n", i, j, C[i * N + j], C[i * N + j]);
        }
    }
}

// int main() {
//     printf("=== Gemm Test Start ===\n");

//     int M = 2, K = 3, N = 3;

//     // input
//     int32_t A[2][3] = {
//         {1, 2, 3},
//         {4, 5, 6}
//     };

//     int32_t B[3][3] = {
//         {7, 8, 9},
//         {10, 11, 12},
//         {13, 14, 15}
//     };

//     int32_t bias[3] = { 0, 0, 0 };
//     int32_t C[2][3];

//     printf("original input:\n");
//     printf("A[2][3]:\n");
//     for (int i = 0; i < 2; i++) {
//         for (int j = 0; j < 3; j++) {
//             printf("%d ", A[i][j]);
//         }
//         printf("\n");
//     }

//     printf("\nB[3][3]:\n");
//     for (int i = 0; i < 3; i++) {
//         for (int j = 0; j < 3; j++) {
//             printf("%d ", B[i][j]);
//         }
//         printf("\n");
//     }

//     printf("\nB to one-dimensional array (one-dimensional array):\n");
//     int32_t* B_ptr = (int32_t*)B;
//     for (int idx = 0; idx < 9; idx++) {
//         printf("B[%d] = %d\n", idx, B_ptr[idx]);
//     }

//     printf("\ncall gemm_int32...\n\n");

//     // convert to int32_t*
//     gemm_int32(M, N, K,
//         (int32_t*)A,     // convert to int32_t*
//         (int32_t*)B,     // convert to int32_t*
//         bias,
//         (int32_t*)C);    // convert to int32_t*

//     printf("\nresult C[2][3]:\n");
//     for (int i = 0; i < M; i++) {
//         for (int j = 0; j < N; j++) {
//             printf("%d ", C[i][j]);
//         }
//         printf("\n");
//     }

//     return 0;
// }