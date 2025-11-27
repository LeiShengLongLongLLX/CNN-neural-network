/*************************************************************/
/*
@author zzy
@date 2024 / 7 / 23
@说明
1 本程序的各个神经网络函数以及参数是以pytorch框架作为参考的，所以例如卷积层中的参数in_channels表示输入的数据通道数，参考pytorch函数即可
2 卷积函数Conv和池化函数MaxPool没有padding功能，padding直接设置为0即可
3 本程序只做部署推理，没有训练
4 所有的权重和偏置以及输入数据均在头文件中
5 本程序是在VS2022软件下编写，其他的环境可能各个文件的引用会有问题
6 本程序现有历程是一个LeNet-5网络做手写数字识别
7 本程序在VS2022中直接点击调试按钮或者按F5即可运行
*/
/*************************************************************/


/***********************还需优化的地方***************************/
// 1 因为我不会多维数组指针，所以只能使用一维指针，然后手动计算数组的寻址，后面应该用多维数组指针
// 2 我不会用头文件定义函数，因为有个结构体，导致会出现各种结构体被重新定义的报错，这个嵌套很麻烦，所以干脆我就不用头文件了
// 3 卷积和池化都没有padding功能，后面应该增加
/***********************还需优化的地方***************************/


/***********************添加卷积、池化等层的操作流程***************************/
// 0 最开始的一步是需要将对应层的权重等参数从pytorch输出的结果中复制到对应层的头文件中，并且定义每一层的输入输出尺寸
// 需要在data_define.h头文件中使用IO_size变量来定义每一层的输入输出尺寸，之后把每一层的参数复制到头文件中
// 每一层的参数我都在变量的最后用这个层的编号加以区分了。比如第0层卷积核的权重就是kernel_0[OUTPUT_SIZE_0_C][INPUT_SIZE_0_C][KERNEL_SIZE_0_H][KERNEL_SIZE_0_W]
// 注意C语言的数组定义数据符号是{}，python则是[]，所以可以先把pytorch中的输出结果复制到txt文件中，然后查找替换所有的[]变成{}
// 这里每一个层的参数我都使用一个单独的头文件进行定义，所以主函数要引用
// 1 首先将现有主函数中的某一个卷积或者池化层一整段直接复制到主函数中对应的层数中：
/*******************第0层卷积*************************/
// ...
// ...
// ...
/*******************第0层卷积*************************/
// 2 然后更改输入输出尺寸，更改本层的参数（stride，padding等）
// 这里每一层的参数我都在变量的最后用这个层的编号加以区分了。比如第0层卷积的输入通道就是in_channels_0
// 3 然后需要把输入数据、卷积核、偏置等指针指向对应的数组
// 4 之后的生成输出数据动态数组不需要改变，只要第2步中的size填写正确，这里就没问题
// 5 然后调用对应的函数，把每一个类型填写进去就好，每一个参量是什么我在对应的.c，例如Conv.c
// 6 最后是可以选择使用XXX（xxx代表对应层的类型，例如Conv）_print()打印输出这一层的结果，平时不需要打印这一层结果可以直接注释掉
/***********************添加卷积、池化等层的操作流程***************************/


#include<stdio.h>
#include<math.h>
#include<malloc.h>
#include<stdlib.h>

#include"data_define.h" // 输入数据头文件
#include"weight_and_bias_layer0.h" // 第0层权重以及偏置参数头文件
#include"weight_and_bias_layer2.h" // 第2层权重以及偏置参数头文件
#include"weight_and_bias_layer5.h" // 第5层权重以及偏置参数头文件
#include"weight_and_bias_layer7.h" // 第7层权重以及偏置参数头文件
#include"weight_and_bias_layer9.h" // 第9层权重以及偏置参数头文件

// 引入conv,maxpool等函数的头文件,一定在data_define.h后面引入，因为IO_size结构体在data_define.h中定义的，否则会找不到结构体类型
#include"Conv.h"
#include"Flatten.h"
#include"Linear.h"
#include"MaxPool.h"
#include"ReLU.h"
#include"max_arr.h"

 /*******************初始化生成配置参数变量*************************/

    // 每一层的输入输出尺寸结构体
IO_size size = { 0 };

// in_channels：输入通道数 out_channels：输出通道数 kernel_size：卷积，池化核尺寸 stride：卷积步长 padding：卷积填充尺寸
int in_channels = 0, out_channels = 0, kernel_size = 0, stride = 0, padding = 0;

// p_input：输入数据指针 p_output：输出数据指针 p_kernel：卷积核权重数组指针 p_bias：卷积偏置数组指针 p_weight：全连接层权重数组指针 p_conv_temp:卷积计算的临时数组指针
float* p_input = NULL; float* p_output = NULL; float* p_kernel = NULL; float* p_bias = NULL; float* p_weight = NULL; float* p_conv_temp = NULL;

// 使用静态数组来存储输入和输出，两个数组的大小需要保证大于等于每一层的输出大小，2个数组轮流使用
// 例如第0层输出存储到buffer0，那么第1层输入就是buffer0，输出就是buffer1，第2层输入buffer1，输出buffer0，依次轮流
// 嵌入式部署尽量不要使用malloc等动态数组，因为无MMU，即内存管理单元，无法实现对内存进行动态映射，会发生很多奇奇怪怪的BUG
float mem_buffer_0[6144] = { 0.0f };
float mem_buffer_1[6144] = { 0.0f };
float conv_temp[6144] = { 0.0f }; // 生成一个临时数组，保存卷积的临时结果，使用数组不使用malloc的动态数组，动态数组在嵌入式设备由于没有中会发生各种BUG

/*******************初始化生成配置参数变量*************************/

// 利用sequential组织神经网络的结构，输入为数据的地址，返回一个输出的指针
float* sequential(float* input_data) {
    /*******************第0层卷积*************************/
    // 更改输入输出尺寸
    size = size_0;

    // 指针指向对应的数组
    p_input = input_data;
    p_output = mem_buffer_0;
    memset(p_output, 0, sizeof(conv_temp)); // 清零数组
    p_conv_temp = conv_temp;
    memset(p_conv_temp, 0, sizeof(mem_buffer_0)); // 清零数组
    p_kernel = &kernel_0[0][0][0][0];
    p_bias = bias_0;

    // 第0层参数编辑以及运算
    in_channels = 1, out_channels = 6, kernel_size = 5, stride = 1, padding = 0;
    Conv(size, p_input, p_output, p_conv_temp, p_kernel, p_bias, in_channels, out_channels, kernel_size, stride, padding);
    //Conv_print(p_output, size); // 输出第一层卷积输出
    /*******************第0层卷积*************************/
    // 运行状态提示
    printf("Layer0 Finished!!!\n\r");

    /*******************第1层池化*************************/
    // 更改输入输出尺寸
    size = size_1;

    // 指针指向对应的数组
    p_input = p_output;
    p_output = mem_buffer_1;
    memset(p_output, 0, sizeof(mem_buffer_0)); // 清零数组

    // 第1层参数编辑
    kernel_size = 2, stride = 2, padding = 0;
    MaxPool(size, p_input, p_output, kernel_size, stride, padding);
    //MaxPool_print(p_output_1, size);

    /*******************第1层池化*************************/
    // 运行状态提示
    printf("Layer1 Finished!!!\n\r");


    /*******************第2层卷积*************************/
    // 更改卷积的输入输出尺寸
    size = size_2;

    // 指针指向对应的数组
    p_input = p_output;
    p_output = mem_buffer_0;
    memset(p_output, 0, sizeof(conv_temp)); // 清零数组
    p_conv_temp = conv_temp;
    memset(p_conv_temp, 0, sizeof(mem_buffer_0)); // 清零数组
    p_kernel = &kernel_2[0][0][0][0];
    p_bias = bias_2;

    // 第2层参数编辑
    in_channels = 6, out_channels = 16, kernel_size = 5, stride = 1, padding = 0;
    // Conv(conv_size, p_input, p_output, p_kernel, p_bias, in_channels, out_channels, kernel_size, stride, padding)
    Conv(size, p_input, p_output, conv_temp, p_kernel, p_bias, in_channels, out_channels, kernel_size, stride, padding);
    //Conv_print(p_output_2, size);

    /*******************第2层卷积*************************/
    // 运行状态提示
    printf("Layer2 Finished!!!\n\r");

    /*******************第3层池化*************************/
    // 更改池化的输入输出尺寸
    size = size_3;

    // 指针指向对应的数组
    p_input = p_output;
    p_output = mem_buffer_1;
    memset(p_output, 0, sizeof(mem_buffer_0)); // 清零数组

    // 第3层参数编辑
    kernel_size = 2, stride = 2, padding = 0;
    // MaxPool(conv_size, p_input, p_output, kernel_size, stride, padding)
    MaxPool(size, p_input, p_output, kernel_size, stride, padding);
    //MaxPool_print(p_output_3, size);

    /*******************第3层池化*************************/
    // 运行状态提示
    printf("Layer3 Finished!!!\n\r");


    /*******************第4层平坦层*************************/
    // 更改输入输出尺寸
    size = size_4;

    // 指针指向对应的数组
    p_input = p_output;
    p_output = mem_buffer_0;
    memset(p_output, 0, sizeof(mem_buffer_0)); // 清零数组

    // Flatten(conv_size, p_input, p_output)
    Flatten(size, p_input, p_output);
    //Flatten_print(size, p_output_3);

    /*******************第4层平坦层*************************/
    // 运行状态提示
    printf("Layer4 Finished!!!\n\r");


    /*******************第5层全连接*************************/
    // 更改输入输出尺寸
    size = size_5;

    // 指针指向对应的数组
    p_input = p_output;
    p_output = mem_buffer_1;
    memset(p_output, 0, sizeof(mem_buffer_0)); // 清零数组
    p_weight = &weight_5[0][0];
    p_bias = bias_5;

    // 第5层参数编辑
    in_channels = 256, out_channels = 120;
    // Linear(p_input, p_output, p_weight, p_bias, in_features, out_features)
    Linear(p_input, p_output, p_weight, p_bias, in_channels, out_channels);
    //Linear_print(p_output, out_channels);

    /*******************第5层全连接*************************/
    // 运行状态提示
    printf("Layer5 Finished!!!\n\r");


    /*******************第6层激活层*************************/
    // 更改输入输出尺寸
    size = size_6;

    // 指针指向对应的数组
    p_input = p_output;
    p_output = mem_buffer_0;
    memset(p_output, 0, sizeof(mem_buffer_0)); // 清零数组

    // ReLu(size, p_input, p_output) ReLu函数生成一个新的输出数组
    ReLu(size, p_input, p_output);
    //ReLu_print(size, p_output);

    /*******************第6层激活层*************************/
    // 运行状态提示
    printf("Layer6 Finished!!!\n\r");


    /*******************第7层全连接*************************/
    // 更改输入输出尺寸
    size = size_7;

    // 指针指向对应的数组
    p_input = p_output;
    p_output = mem_buffer_1;
    memset(p_output, 0, sizeof(mem_buffer_0)); // 清零数组
    p_weight = &weight_7[0][0];
    p_bias = bias_7;

    // 第7层参数编辑
    in_channels = 120, out_channels = 84;
    // Linear(p_input, p_output, p_weight, p_bias, in_features, out_features)
    Linear(p_input, p_output, p_weight, p_bias, in_channels, out_channels);
    //Linear_print(p_output, out_channels);

    /*******************第7层全连接*************************/
    // 运行状态提示
    printf("Layer7 Finished!!!\n\r");


    /*******************第8层激活层*************************/
    // 更改输入输出尺寸
    size = size_8;

    // 指针指向对应的数组
    p_input = p_output;
    p_output = mem_buffer_0;
    memset(p_output, 0, sizeof(mem_buffer_0)); // 清零数组

    // ReLu(size, p_input, p_output) ReLu函数生成一个新的输出数组
    ReLu(size, p_input, p_output);
    //ReLu_print(size, p_output);

    /*******************第8层激活层*************************/
    // 运行状态提示
    printf("Layer8 Finished!!!\n\r");

    /*******************第9层全连接*************************/
    // 更改输入输出尺寸
    size = size_9;

    // 指针指向对应的数组
    p_input = p_output;
    p_output = mem_buffer_1;
    memset(p_output, 0, sizeof(mem_buffer_0)); // 清零数组
    p_weight = weight_9[0];
    p_bias = bias_9;

    // 第9层参数编辑
    in_channels = 84, out_channels = 10;
    // Linear(p_input, p_output, p_weight, p_bias, in_features, out_features)
    Linear(p_input, p_output, p_weight, p_bias, in_channels, out_channels);
    Linear_print(p_output, out_channels);

    /*******************第9层全连接*************************/
    // 运行状态提示
    printf("Layer9 Finished!!!\n\r");

    return p_output;
};

int main() {
    p_input = &input_data[0][0][0];
    p_output = sequential(p_input);
    
    /*******************输出最终结果*************************/
    // max(int input_size, float* p_input)
    int output = max_arr(10, p_output); // 将可能性最大的输出为类型序号

    printf("Input number is %d", output);
    /*******************输出最终结果*************************/

    return 0;
}