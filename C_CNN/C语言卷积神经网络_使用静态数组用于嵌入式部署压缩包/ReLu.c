#include<stdio.h>
#include<math.h>

// 卷积输入输出尺寸结构体
typedef struct IO_size {
	int C_in;
	int H_in;
	int W_in;

	int C_out;
	int H_out;
	int W_out;
}IO_size;

// ReLu(size, p_input, p_output) ReLu函数生成一个新的输出数组
void ReLu(IO_size size, float* p_input, float* p_output)
{
	for(int out = 0; out < size.C_out * size.H_out * size.W_out; out++)
	{
		if (*(p_input + out) < 0)
		{
			*(p_output + out) = 0; // 如果小于0则变为0
		}
		else 
		{
			*(p_output + out) = *(p_input + out); // 如果大于等于0则变为原始数据，不做改变;
		}
	}
}


// // ReLu_print(size, p_output) 
void ReLu_print(IO_size size, float* p_output)
{
	for (int out = 0; out < size.C_out * size.H_out * size.W_out; out++)
	{
		printf("output[%d] = %.6f\n", out, *(p_output + out));
	}
}