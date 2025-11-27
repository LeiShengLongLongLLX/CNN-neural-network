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

// Conv(IO_size, p_input, p_output, p_kernel, p_bias, in_channels, out_channels, kernel_size, stride, padding)
void Conv (IO_size size, float* p_input, float* p_output, float* conv_temp, float* p_kernel, float* p_bias, int in_channels, int out_channels, int kernel_size, int stride, int padding)
{
	// memset(conv_temp, 0, sizeof(conv_temp)); // 清零数组
	
	for (int C_out = 0; C_out < out_channels; C_out++)
	{
		for (int C_in = 0; C_in < in_channels; C_in++)
		{
			for (int H = 0, H_output = 0; H + kernel_size <= size.H_in; H += stride, H_output++)
			{
				for (int W = 0, W_output = 0; W + kernel_size <= size.W_in; W += stride, W_output++)
				{
					for (int H_kernel = 0; H_kernel < kernel_size; H_kernel++)
					{
						for (int W_kernel = 0; W_kernel < kernel_size; W_kernel++)
						{
							// p_output_temp[C_out][C_in][H_output][W_output] += p_input[C_in][H + H_kernel][W + W_kernel] * p_kernel[C_out][C_in][H_kernel][W_kernel]
							*(conv_temp + C_out * size.C_in * size.H_out * size.W_out + C_in * size.H_out * size.W_out + H_output * size.W_out + W_output) += *(p_input + (C_in * size.H_in * size.W_in + (H + H_kernel) * size.W_in + (W + W_kernel))) * *(p_kernel + C_out * size.C_in * kernel_size * kernel_size + C_in * kernel_size * kernel_size + H_kernel * kernel_size + W_kernel);
						};
					};
				};
			};
		};
	};

	//为了方便深度可分离卷积的实现，这里每一通道单独计算卷积
	for (int C_out = 0; C_out < out_channels; C_out++)
	{
		for (int H = 0; H < size.H_out; H++)
		{
			for (int W = 0; W < size.W_out; W++)
			{
				for (int C_in = 0; C_in < in_channels; C_in++)
				{
					*(p_output + (C_out * size.H_out * size.W_out + H * size.W_out + W)) += *(conv_temp + C_out * size.C_in * size.H_out * size.W_out + C_in * size.H_out * size.W_out + H * size.W_out + W);
				}
				*(p_output + C_out * size.H_out * size.W_out + H * size.W_out + W) += *(p_bias + C_out); // 增加偏置
			}
		}
	}

}

// output[OUTPUT_SIZE_3_C][OUTPUT_SIZE_3_H][OUTPUT_SIZE_3_W]
void Conv_print(float* p_output, IO_size size)
{
    for (int C = 0; C < size.C_out; C++)
    {
		printf("%d\n", C);
        for (int H = 0; H < size.H_out; H++)
        {
            for (int W = 0; W < size.W_out; W++)
            {
				printf("%.6e ", *(p_output + C * size.H_out * size.W_out + H * size.W_out + W));
			}
            printf("\n");
        }
        printf("\n\n");
    }
}

