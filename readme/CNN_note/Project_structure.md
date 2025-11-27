# C语言结构化工程开发

## 一、C语言文件介绍

### 1.1 `.h` 文件（头文件）的用途与规则

---

头文件 `.h` 的主要作用：

#### 1. 共享结构体定义（struct / typedef）

所有需要使用 Tensor 的 `.c` 都必须知道 Tensor 的定义，因此放在 `.h`：

```c
typedef struct {
    float* data;
    int N, C, H, W;
} Tensor;
```

#### 2. 共享宏（宏函数 / 常量）

如 NCHW 展开宏必须在所有源文件中一致：

```c
#define IDX4(n, c, h, w, C, H, W) ...
```

#### 3. 声明函数原型（function prototype）

声明但不实现：

```c
Tensor Tensor_init(int N, int C, int H, int W);
void Conv2D(Tensor input, Tensor kernel, Tensor output);
```

#### 4. 防止重复包含（include guard）

防止头文件被多次 include：

```c
#ifndef TENSOR_H
#define TENSOR_H
...
#endif
```

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

> ❌ 不要在 .c 中 include 另一个 .c
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

