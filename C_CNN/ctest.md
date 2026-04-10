# CNN_test（int8 / int16 / int32）算子测试参数汇总

说明：维度与张量填充均以各 `test/*.c` 中 **`Tensor_init` 与填充函数** 为准；存储为 **NCHW**，线性下标 `i` 与 `(n,c,h,w)` 对应关系为：自 `w` 最快变，依次为 `h`、`c`、`n`。int8、int16、int32 三套用例形状与下列**整数取值模式**相同，仅元素位宽不同。

MAC 计数：Conv2D 为 **324**；GEMM 为 **12**；Flatten、MaxPool、ReLU 为 **0**。

---

## 1. Conv2D — `test/testconv2d.c`

下列为 **int32 语义下的整数**；int8、int16 为同一数值的截断/饱和（本用例均在 small 正数范围，通常与 int32 一致）。

```text
输入 feature  N=1 C=1 H=8 W=8  （单通道，平面 8×8，行=h、列=w）

  0   1   2   3   4   5   6   7
  8   9  10  11  12  13  14  15
 16  17  18  19  20  21  22  23
 24  25  26  27  28  29  30  31
 32  33  34  35  36  37  38  39
 40  41  42  43  44  45  46  47
 48  49  50  51  52  53  54  55
 56  57  58  59  60  61  62  63

卷积核 kernel  N=1 C=1 H=3 W=3  （全 1）

  1   1   1
  1   1   1
  1   1   1

bias：无（NULL）

输出 output  N=1 C=1 H=6 W=6  （stride=1，padding=0，valid 3×3 卷积、核全 1 时的求和结果）

 81  90  99 108 117 126
153 162 171 180 189 198
225 234 243 252 261 270
297 306 315 324 333 342
369 378 387 396 405 414
441 450 459 468 477 486
```

| 项目 | 值 |
|------|-----|
| 输入形状 | N=1，C=1，H=8，W=8 |
| 卷积核形状 | N=1，C=1，H=3，W=3 |
| bias | 无 |
| stride | 1 |
| padding | 0 |
| 输出形状 | N=1，C=1，H=6，W=6 |
| MACs | 324 |

API：`Conv2D_int8` / `Conv2D_int16` / `Conv2D_int32`。

---

## 2. Flatten — `test/testflatten.c`

线性填充顺序与 NCHW 一致：`i=0..7` 依次写入 `data[i]`。

```text
输入  N=1 C=2 H=2 W=2

  通道 c=0（H×W）        通道 c=1（H×W）
  0   1                  4   5
  2   3                  6   7

输出  N=1 C=8 H=1 W=1  （一行 8 个数，与上表按内存顺序相同）

  0   1   2   3   4   5   6   7
```

| 项目 | 值 |
|------|-----|
| 输入形状 | N=1，C=2，H=2，W=2 |
| 输出形状 | N=1，C=8，H=1，W=1 |
| MACs | 0 |

API：`Flatten_int8` / `Flatten_int16` / `Flatten_int32`。

---

## 3. GEMM / 全连接 — `test/testgemm.c`

`Tensor_init` 中：batch 放在 N，特征放在 C，H=W=1。下列按 **N×C 矩阵** 形式展示（每行一个 N 切片，列对应 C）。

```text
input   N=1 C=4 H=1 W=1  → 视作 1×4

  0   1   2   3

weights N=3 C=4 H=1 W=1  → 视作 3×4，元素全为 1

  1   1   1   1
  1   1   1   1
  1   1   1   1

bias 长度 3

  0   0   0

output  N=1 C=3 H=1 W=1  → 视作 1×3（与 gemm 实现及上述权重、输入一致时的结果）

  6   6   6
```

| 项目 | 值 |
|------|-----|
| batch（N 维） | 1 |
| 输入特征数（C 维） | 4 |
| 输出特征数（weights 的 N 维） | 3 |
| input 形状 | N=1，C=4，H=1，W=1 |
| weights 形状 | N=3，C=4，H=1，W=1 |
| bias 长度 | 3，取值均为 0 |
| output 形状 | N=1，C=3，H=1，W=1 |
| MACs | 12 |

API：`gemm_int8` / `gemm_int16` / `gemm_int32`。

---

## 4. MaxPool2D — `test/testmaxpool.c`

`pool_size=2`，`stride=2`，无额外 padding（与测试代码中输出尺寸计算一致）。

```text
输入  N=1 C=1 H=4 W=4

  0   1   2   3
  4   5   6   7
  8   9  10  11
 12  13  14  15

输出  N=1 C=1 H=2 W=2  （2×2 池化、步长 2，每格取窗口内最大值）

  5   7
 13  15
```

| 项目 | 值 |
|------|-----|
| 输入形状 | N=1，C=1，H=4，W=4 |
| 池化窗口 | 2 |
| 池化步长 | 2 |
| padding | 0 |
| 输出形状 | N=1，C=1，H=2，W=2 |
| MACs | 0 |

API：`maxpool2D_int8` / `maxpool2D_int16` / `maxpool2D_int32`。

---

## 5. ReLU — `test/testrelu.c`

输入：`data[i] = i - 4`，`i = 0..8`。输出：逐元素 `max(0, x)`（与通常 ReLU 一致时的结果如下）。

```text
输入  N=1 C=1 H=3 W=3

 -4  -3  -2
 -1   0   1
  2   3   4

输出  N=1 C=1 H=3 W=3

  0   0   0
  0   0   1
  2   3   4
```

| 项目 | 值 |
|------|-----|
| 输入形状 | N=1，C=1，H=3，W=3 |
| 输出形状 | N=1，C=1，H=3，W=3 |
| MACs | 0 |

API：`relu_int8` / `relu_int16` / `relu_int32`。

---

## 目录对照

| 精度 | 路径 |
|------|------|
| int8 | `C_CNN (int8)/test/` |
| int16 | `C_CNN (int16)/test/` |
| int32 | `C_CNN (int32)/test/` |

---

## 备注

- `testconv2d.c` 中若仍有与 `Tensor_init(1,1,8,8)`、`Tensor_init(1,1,3,3)` 不符的注释，以代码与本文档为准。
- 若修改测试用例填充或形状，请同步更新本文档中的数字块与表格。
