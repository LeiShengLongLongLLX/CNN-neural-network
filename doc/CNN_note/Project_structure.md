# C语言结构化工程开发

## 一、C语言文件介绍

### 1.1 `.h` 文件（头文件）

---
#### 1. `.h` 文件是什么

`.h` 文件（Header File，头文件）是 C/C++ 程序中用于**声明（Declaration）**各种内容的文件。
它本质上是一个**文本文件**，通过 `#include` 指令被其他 `.c` 或 `.cpp` 文件引用，让多个源文件共享相同的声明。

> **特点：**
>
> * 只包含“声明”，不包含“定义”（除特殊情况）。
> * 是 C 项目“模块化设计”的基础。
> * 帮助代码分离，提高可维护性和可复用性。
> * 通过 include 让多个源文件共用函数、结构体、类型、宏等。

---

#### 2. `.h` 文件有什么作用

##### （1）函数声明（Function Declaration）

让其他 `.c` 文件能调用该函数，而不需要知道函数的具体实现内容。

例如：

```c
void relu(float* input, float* output, int size);
```

---

##### （2）数据结构声明（struct、typedef）

让多个模块共享同一个结构体类型。

```c
typedef struct {
    float* data;
    int N, C, H, W;
} Tensor;
```

---

##### （3）宏定义（#define）

把通用常量、配置统一放在头文件。

```c
#define KERNEL_SIZE 3
#define MAX_POOL 2
```

---

##### （4）类型声明（enum、typedef）

```c
typedef unsigned int uint32;
```

---

##### （5）外部变量声明（extern）

避免多文件重复定义全局变量。

```c
extern int g_debug_flag;
```

---

##### （6）减少代码重复、实现模块化

把接口和实现分开：

* `.h` —— 声明接口（别人怎样调用你的模块）
* `.c` —— 实现接口（具体怎么实现）

这样你的代码结构清晰且易维护。

---

#### 3. `.h` 文件应该怎么写

##### （1）加入 include guard 防止重复包含

标准写法：

```c
#ifndef RELU_H
#define RELU_H

// 内容 ...

#endif
```

作用：避免重复定义导致编译报错。

---

##### （2）只写“声明”，不写“定义”

头文件中通常放：

* 函数声明
* 结构体声明
* 宏定义
* typedef
* extern 声明

头文件中**不应该写：**

* 函数的具体实现（除 static inline 特例）
* 全局变量定义（应该使用 `extern`）

---

##### （3）必要的引用放在 `.h` 里还是 `.c` 里？

| 类型                          | 是否应放在 `.h`               |
| --------------------------- | ------------------------ |
| `<stdio.h> <stdlib.h>` 等标准库 | **一般不放**（除非头文件声明用到了这些类型） |
| `<stdint.h>` 等类型相关头         | 如果头文件用到这些类型，**必须放**      |
| 自己写的头文件                     | 需要依赖时才 include           |

规则很简单：
**只 include 你这个头文件“真正需要”使用到的内容。**

---

##### （4）头文件建议的规范写法

举个标准模板：

```h
#ifndef RELU_H
#define RELU_H

#endif // RELU_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>   // 头文件真的使用到了才 include

// 宏定义
#define ACTIVATION_RELU 1

// 类型定义
typedef struct {
    float* data;
    int size;
} Tensor1D;

// 函数声明
void relu(const float* input, float* output, int size);

#ifdef __cplusplus
}
#endif

```

这是工业级规范。

---

### 1.2 `.c` 文件（源文件）的职责

`.c` 文件是 **具体实现执行逻辑的地方**。

#### 1. 实现头文件中声明的函数

例如 tensor.c：

```c
Tensor Tensor_init(...) { ... }
void Tensor_free(...)   { ... }
```

conv2d.c：

```c
void Conv2D(...) { ... }
```

#### 2. 必须 include 对应的 .h

确保函数声明、结构体一致：

```c
#include "tensor.h"
```

#### 3. .c 文件之间互不共享内容

每个 .c 文件编译时互相独立，因此必须通过头文件共享信息。

> 不要在 .c 中 include 另一个 .c
>
> 这是非专业、会导致重复定义、链接失败的写法。

---

### 1.3 main.c 的角色

main.c 的职责是：
**调用接口、组织流程，不承担任何底层逻辑实现。**

它只需要 include 头文件：

```c
#include "tensor.h"
```

它不需要知道 Tensor_init 或 Conv2D 的实现细节。

---

### 1.4 为什么所有 .c 都必须 include .h？

因为：

* main.c 要知道 Tensor、Conv2D 的声明
* tensor.c 也要知道 Tensor（为了初始化）
* conv2d.c 必须知道 Tensor 结构体、IDX4 宏

每个源文件独立，因此每个都必须 include 它需要的内容。

---

## 二、编译与链接流程

编译器执行两步：

---

### 第 1 步：编译（compile）

每个 .c 单独变成 .o：

```
gcc -c main.c     → main.o
gcc -c tensor.c   → tensor.o
gcc -c conv2d.c   → conv2d.o
```

此阶段：

> ✔ 各 .c 分开编译
>
> ✔ 互相不知道对方的内容
> 
> ✔ 仅通过 `.h` 知道函数原型 & 结构体定义

---

### 第 2 步：链接（link）

把所有 .o 文件链接成一个可执行文件：

```
gcc main.o tensor.o conv2d.o -o conv_test
```

链接器会去 tensor.o 和 conv2d.o 里查找函数实现。

---


## 三、工程结构介绍 
### 3.1 C语言项目常见工程结构（专业项目）

```
project/
│
├─ include/
│   └─ tensor.h
│
├─ src/
│   ├─ tensor.c
│   ├─ conv2d.c
│   └─ main.c
│
└─ build/
```

专业 C 工程一般会分：

* include → 头文件
* src     → 源文件
* build   → 编译输出

以后做 CNN 库、算子库、驱动程序，都可以这样写。

---

### 3.2 好处

#### ✔ 1. 工程可扩展

例如你要加：

* ReLU.c
* MaxPool.c
* Softmax.c
* MatMul.c

只需要：

* 在 .h 中声明
* 在 .c 中实现

main.c 无需修改。

#### ✔ 2. 结构清晰、模块化

每个算子独立一个 .c 文件，不会混乱。

#### ✔ 3. 程序可维护

别人一眼能看懂整个工程结构。

#### ✔ 4. 便于编译优化

可以对 conv2d.c 单独开启 -O3 或 RVV 优化。

---

### 3.3 工程模板（适合你所有未来工程）

以后新建 C 工程就按这套模板写：

---

#### **头文件 (.h)**

* 定义数据结构（Tensor、Layer、Node…）
* 宏
* extern 全局变量
* 所有函数声明
* include guard

---

#### **源文件 (.c)**

* 实现 .h 中声明的函数
* 每个模块一个 .c
* include 对应的 .h

---

#### **main.c**

* include 顶层头文件
* 调用函数
* 不放实现

---

