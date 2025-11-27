#include<stdio.h>
#include<math.h>

// 输入输出尺寸结构体
typedef struct IO_size {
	int C_in;
	int H_in;
	int W_in;

	int C_out;
	int H_out;
	int W_out;
}IO_size;

// Flatten(conv_size, p_input, p_output)
void Flatten(IO_size size, float* p_input, float* p_output)
{
	for (int out = 0; out < size.C_in * size.H_in * size.W_in; out++)
	{
		*(p_output + out) = *(p_input + out); // 其实这一步对于一维指针完全是脱裤子放屁。。。
	}
}

// Flatten(conv_size, p_output)
void Flatten_print(IO_size size, float* p_output)
{
	for (int out = 0; out < size.C_in * size.H_in * size.W_in; out++)
	{
		printf("output[%d] = %.6e\n", out, *(p_output + out));
	}
}