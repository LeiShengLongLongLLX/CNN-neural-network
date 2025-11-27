#include<stdio.h>
#include<math.h>
#include<malloc.h>

// 输入输出尺寸结构体
typedef struct IO_size {
	int C_in;
	int H_in;
	int W_in;

	int C_out;
	int H_out;
	int W_out;
}IO_size;

// MaxPool(conv_size, p_input, p_output, kernel_size, stride, padding)
void MaxPool (IO_size size, float* p_input, float* p_output, int kernel_size, int stride, int padding)
{
	int C = 0, H = 0, W = 0, H_out = 0, W_out = 0, H_kernel = 0, W_kernel = 0;
	float temp = -99999.0; // 为了保证比较大小的时候足够小
	for (C = 0; C < size.C_in; C++)
	{
		for (H = 0, H_out = 0; H + kernel_size <= size.H_in; H += stride, H_out++)
		{
			for (W = 0, W_out = 0; W + kernel_size <= size.W_in; W += stride, W_out++)
			{
				temp = -99999.0; // 下一个滑动窗，再次赋值
				for (H_kernel = 0; H_kernel < kernel_size; H_kernel++)
				{
					for (W_kernel = 0; W_kernel < kernel_size; W_kernel++)
					{
						temp = *(p_input + (C * size.H_in * size.W_in) + ((H + H_kernel) * (size.W_in)) + (W + W_kernel)) > temp ? *(p_input + (C * size.H_in * size.W_in) + ((H + H_kernel) * (size.W_in)) + (W + W_kernel)) : temp;
					}
				}
				*(p_output + (C * size.H_out * size.W_out) + (H_out * size.W_out) + W_out) = temp;
			}
		}
	}
}

// MaxPool层的输出函数
void MaxPool_print(float* p_output, IO_size size)
{
	for (int C = 0; C < size.C_out; C++)
	{
		printf("%d\n", C);
		for (int H = 0; H < size.H_out; H++)
		{
			for (int W = 0; W < size.W_out; W++)
			{
				//printf("output_MaxPool[%d][%d][%d] = %.4f\n", C, H, W, output[C][H][W]);
				printf("%.4e ", *(p_output + (C * size.H_out * size.W_out) + (H * size.W_out) + W));
			}
			printf("\n");
		}
		printf("\n\n");
	}
}

