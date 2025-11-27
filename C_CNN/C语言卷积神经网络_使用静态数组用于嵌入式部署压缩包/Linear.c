#include<stdio.h>
#include<math.h>

// Linear(p_input, p_output, p_weight, p_bias, in_features, out_features)
void Linear (float* p_input, float* p_output, float* p_weight, float* p_bias, int in_features, int out_features)
{
	for (int out = 0; out < out_features; out++)
	{
		for (int in = 0; in < in_features; in++)
		{
			*(p_output + out) += *(p_input + in) * *(p_weight + out * in_features + in);
		}
		*(p_output + out) += *(p_bias + out);
	}
}


// Linear_print(p_output, out_features)
void Linear_print(float* p_output, int out_features)
{
	for (int out = 0; out < out_features; out++)
	{
		printf("output[%d] = %.6e\n", out, *(p_output + out));
	}
}