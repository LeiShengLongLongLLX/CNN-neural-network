#include<stdio.h>

// max_arr(int input_size, float* p_input)
int max_arr(int input_size, float* p_input)
{
	int out_temp = 0; // 用于记录最大的数字的数组序号
	float temp = 0; // 用于记录最大的数字
	for (int out = 0; out < input_size; out++)
	{
		if (temp < *(p_input + out))
		{
			temp = *(p_input + out); // 如果遍历的数字大于temp，替换temp
			out_temp = out; // 记录此时的数组序号
		}
		else
		{
			; // 否则不执行操作
		}
	}
	return out_temp;
}