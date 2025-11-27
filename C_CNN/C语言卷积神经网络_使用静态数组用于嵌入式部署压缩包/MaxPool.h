#ifndef __MAXPOOL_H__
#define __MAXPOOL_H__

void MaxPool(IO_size size, float* p_input, float* p_output, int kernel_size, int stride, int padding);
void MaxPool_print(float* p_output, IO_size size);

#endif
