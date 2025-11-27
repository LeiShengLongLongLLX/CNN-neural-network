#ifndef __CONV_H__
#define __CONV_H__

void Conv(IO_size size, float* p_input, float* p_output, float* conv_temp, float* p_kernel, float* p_bias, int in_channels, int out_channels, int kernel_size, int stride, int padding);
void Conv_print(float* p_output, IO_size size);

#endif
