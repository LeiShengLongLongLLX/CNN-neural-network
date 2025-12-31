# python

# 一、Python 总概

## 1.1 Python 是什么语言

### 1. 高级语言的特点

**Python 属于高级编程语言（High-Level Language）**，其核心特点是：

1. **强抽象**

   * 程序员不直接接触：

     * 内存分配 / 释放
     * 寄存器
     * 指令级细节
   * 更关注「**做什么**」，而不是「**怎么在硬件上做**」

2. **自动内存管理**

   * 通过 **垃圾回收（GC）+ 引用计数**
   * 不需要 `malloc / free`、`new / delete`

3. **动态类型**

   * 变量不需要声明类型
   * 类型在运行时确定

   ```python
   a = 10      # int
   a = "hi"    # str（合法）
   ```

4. **解释执行（为主）**

   * 源码不会直接编译为机器码
   * 由解释器逐步执行（准确说是执行字节码）

5. **开发效率高**

   * 代码量少
   * 语法接近自然语言
   * 非常适合：

     * 快速原型
     * 自动化
     * 数据处理

**一句话总结**

> Python 是一门以「**可读性 + 开发效率**」优先的高级动态语言。

---

### 2. 与 C / C++ 的抽象层级对比

| 维度       | Python | C    | C++       |
| -------- | ------ | ---- | --------- |
| 抽象层级     | 很高     | 低    | 中         |
| 是否需要声明类型 | 否      | 是    | 是         |
| 内存管理     | 自动     | 手动   | 手动 / RAII |
| 是否贴近硬件   | 否      | 非常贴近 | 较贴近       |
| 运行效率     | 较低     | 很高   | 很高        |
| 开发效率     | 很高     | 较低   | 中         |

#### 抽象层级示意（从底层到上层）

```text
硬件
 ↓
汇编
 ↓
C
 ↓
C++
 ↓
Python
```

#### **工程视角理解**

* **C / C++**：

  > “我控制机器”
* **Python**：

  > “我指挥程序做事”

💡 很多大型系统是：

* **底层性能关键部分**：C / C++
* **上层逻辑、工具、调度**：Python

---

### 3. 典型应用领域

#### ① 脚本 / 工具开发

* 自动化脚本
* 文件处理
* 日志分析
* 批量操作

```bash
python build.py
python gen_config.py
```

#### ② 后端开发

* Web 服务
* REST API
* 微服务

常见框架：

* Flask
* Django
* FastAPI

#### ③ 数据 / AI / 科学计算

* 数据分析（NumPy / Pandas）
* 可视化（Matplotlib）
* 机器学习（PyTorch / TensorFlow）

> **这是 Python 的“王牌领域”**

#### ④ 自动化 / 运维 / 测试

* CI 脚本
* 自动测试
* 设备控制
* 工程流程自动化

> **在工程实践中，Python 常作为“胶水语言”**

---

## 1.2 Python 的运行机制

### 1. Python 解释器

**Python 解释器 = 执行 Python 程序的核心程序**

最常见的是：

* **CPython（官方实现）**

你输入：

```bash
python main.py
```

本质是：

```text
python.exe / python3
  ↓
读取 main.py
  ↓
解析 → 编译为字节码 → 执行
```

Python 不是“直接解释源码”，而是：

> **源码 → 字节码 → 解释执行**

---

### 2. `.py` 文件是如何被执行的

执行流程👇

#### 第一步：词法 / 语法分析

* 检查语法是否正确
* 生成抽象语法树（AST）

#### 第二步：编译为字节码

* 生成 `.pyc`
* 存放在 `__pycache__/`

```text
main.py
↓
main.cpython-311.pyc
```

字节码是：

* **与平台无关**
* **介于源码和机器码之间**

#### 第三步：解释器执行字节码

* Python 虚拟机（PVM）逐条解释执行

对比 C：

| C          | Python          |
| ---------- | --------------- |
| 源码 → 机器码   | 源码 → 字节码 → 解释执行 |
| 直接跑在 CPU 上 | 跑在 Python 虚拟机上  |

---

## 1.3 依赖库

### 1. 什么是标准库？

**Python 自带的库**

* 安装 Python 时自动拥有
* 不需要额外安装

常见标准库：

* `os`（操作系统）
* `sys`（解释器交互）
* `math`
* `time`
* `threading`
* `subprocess`

```python
import os
print(os.getcwd())
```

> **注：标准库 = Python 的“官方工具箱”**

---

### 2. 什么是第三方库？

* 非 Python 官方
* 由社区 / 公司维护
* 需要额外安装

常见第三方库：

* `numpy`
* `pandas`
* `requests`
* `matplotlib`
* `torch`

工程实践中：

> **80% 的能力来自第三方库**

---

### 3. pip

**pip = Python 官方包管理工具**

功能：

* 安装
* 卸载
* 管理版本

```bash
pip install numpy
pip uninstall numpy
pip list
```

pip 本质是：

> **从 PyPI（Python 包仓库）下载并安装库**

---

## 1.4 Python 环境管理

### 1. 虚拟环境

**虚拟环境 = 独立的 Python 运行环境**

解决的问题：

* 不同项目依赖冲突
* 不污染系统 Python

```text
系统 Python
 ├─ 项目A → numpy 1.23
 └─ 项目B → numpy 2.0
```

---

### 2. 系统 Python VS 项目 Python

| 对比项     | 系统 Python | 项目 Python |
| ------- | --------- | --------- |
| 作用      | 系统工具      | 项目运行      |
| 是否建议乱装库 | ❌         | ✅         |
| 是否可删除   | ❌         | ✅         |

**工程规范**

> 一个项目 = 一个 Python 环境

---

### 3. venv / virtualenv / conda 的定位区别

| 工具         | 定位       | 特点           |
| ---------- | -------- | ------------ |
| venv       | 官方轻量方案   | Python 自带    |
| virtualenv | venv 增强版 | 功能更全         |
| conda      | 环境 + 包管理 | 支持非 Python 包 |

**推荐认知**

* 学习 / 普通项目：`venv`
* 科学计算 / AI：`conda`

---

## 1.5 Python 常用开发工具

### 1. VS Code

* 轻量
* 插件丰富
* 适合工程 + 多语言

**工程师首选**

---

### 2. PyCharm

* Python 专用 IDE
* 智能提示强
* 适合 Python 深度开发

---

### 3. Anaconda

* Python 发行版
* 自带：

  * conda
  * 科学计算库
* 常用于：

  * 数据分析
  * AI / 机器学习

---

## 1,6 总结

> **Python 是一门高抽象、高效率、强生态的高级语言，
> 在现代工程中常作为“工具层 / 调度层 / 数据层”的核心角色。

---

# 二、基础概念

## 2.1 字面量（Literal）

**字面量**是指在代码中**直接写出来的值**，是程序中最基本、不可再拆分的组成单位。

### 1. 常见字面量类型

#### （1）数字字面量

```python
10        # 整数
3.14      # 浮点数
0b1010    # 二进制
0o17      # 八进制
0x1A      # 十六进制
```

> 注：Python 中整数**没有位宽限制**（不区分 int32 / int64）。

---

#### （2）字符串字面量

```python
"hello"
'world'
"""多行字符串"""
```

特点：

* 单引号 / 双引号等价
* 三引号支持多行

---

#### （3）布尔字面量

```python
True
False
```

⚠️ 注意：

* 首字母必须大写
* 本质是整数的子类（`True == 1`）

---

#### （4）空值字面量

```python
None
```

含义：

* 表示“没有值 / 未定义”
* 类似 C 中的 `NULL`，但语义更丰富

---

### 2. 字面量的工程理解

对比 C：

```c
int a = 10;
```

对比 Python：

```python
a = 10
```

#### **字面量先于变量存在**

> 变量只是“名字”，字面量才是“值”。

---

## 2.2 变量与常量

Python 中，**变量本质是“名字绑定（name binding）”**，而不是“内存盒子”。

---

### 2.2.1 变量

#### 1. 变量的定义

```python
a = 10
b = "hello"
```

含义：

* `a`、`b` 是名字
* `=` 表示绑定（不是赋值到内存）

Python 中变量：

* 不需要声明类型
* 类型在运行时决定

---

#### 2. 动态类型特性

```python
x = 10
x = "abc"
```

* 合法
* 名字 `x` 重新绑定到了新对象

对比 C：

```c
int x = 10;
x = "abc";   // 非法
```

---

#### 3. 多变量绑定

```python
a = b = 10
x, y = 1, 2
```

解包赋值是 Python 的重要特性。

---

### 2.2.2 常量

Python **没有语法级常量关键字**（没有 `const`）。

#### 1. 常量的约定写法

```python
PI = 3.1415926
MAX_LEN = 1024
```

约定：

* 全大写
* 只读语义靠“自觉”

---

#### 2. 不可变对象 ≠ 常量

```python
a = 10
a = 20   # 合法
```

* `10` 是不可变对象
* 但变量 `a` 仍然可以重新绑定

**不可变 ≠ 常量**

---

## 2.3 注释（Comment）

注释用于：

* 解释代码逻辑
* 提高可维护性
* 不参与程序执行

---

### 1. 单行注释

```python
# 这是单行注释
x = 10  # 行尾注释
```

---

### 2. 多行注释（常见写法）

Python 没有真正的多行注释语法，常用：

```python
"""
这是多行注释
（本质是字符串）
"""
```

工程实践中：

* 多用于模块 / 函数文档

---

## 2.4 标识符（Identifier）

**标识符**是程序中用于命名的符号。

### 1. 标识符命名规则

合法标识符：

* 字母 / 下划线 / 数字
* 不能以数字开头

```python
var1
_var
my_name
```

非法示例：

```python
1abc   # ❌
a-b    # ❌
```

---

### 2. 关键字（不能作为标识符）

```python
if, for, while, def, class, return, import
```

关键字是语言保留字。

---

### 3. 命名规范（工程建议）

#### Python官方 (PEP8) 给出的建议

- 变量 / 函数：`snake_case`
- 类名：`PascalCase（CamelCase）`
- 常量：`ALL_CAPS`
- 模块名：`snake_case`
- 私有成员：`_name`
- 魔法方法：`__name__`

---

| 场景      | 推荐命名法               |
| ------- | ------------------- |
| 变量 / 函数 | `snake_case`        |
| **类名**  | **PascalCase（大驼峰）** |
| 常量      | `ALL_CAPS`          |

也就是说
```python
class FileReader:     # ✅ 大驼峰（PascalCase）
    pass

def read_file():      # ✅ snake_case
    pass

MAX_SIZE = 1024       # ✅ ALL_CAPS
```

---

## 2.5 运算符（Operator）

运算符用于**对字面量和变量进行运算**。

---

### 1. 算术运算符

```python
+  -  *  /  //  %  **
```

示例：

```python
5 / 2    # 2.5
5 // 2   # 2
```

`/` 永远返回浮点数。

---

### 2. 比较运算符

```python
==  !=  >  <  >=  <=
```

返回值：

```python
True / False
```

---

### 3. 逻辑运算符

```python
and
or
not
```

Python 使用**单词逻辑运算符**，不是 `&& || !`

---

### 4. 赋值运算符

```python
=
+= -= *= /= //=
```

---

### 5. 成员与身份运算符（Python 特有）

```python
in
not in
is
is not
```

示例：

```python
a = [1, 2]
b = a

a is b      # True（同一对象）
a == b      # True（值相等）
```

**`is` 比较的是对象身份，不是值**

---

## 2.6 总结

1. 字面量是程序中最基本的值
2. 变量是名字绑定，而不是内存盒子
3. Python 没有真正的常量，只是约定
4. 注释不参与执行，但决定代码质量
5. 运算符是程序“运转起来”的核心工具

---

# 三、数据类型

## 3.1 数值类型（Numeric Types）

数值类型用于表示**数学意义上的数值**，在 Python 中是**对象**，不是裸值。

---

### 1. 整型（int）

#### （1）基本特性

```python
a = 10
b = -3
c = 12345678901234567890
```

* Python 的 `int`：

  * **没有位宽限制**
  * 自动扩展精度
* 不区分 `int / long`

#### 对比 C：

```c
int a;   // 位宽固定
```

---

#### （2）进制表示

```python
0b1010   # 二进制
0o17     # 八进制
0x1A     # 十六进制
```

---

#### （3）不可变特性

```python
x = 10
x += 1
```

实际发生的是：

* 创建新 `int` 对象
* `x` 重新绑定

---

### 2. 浮点型（float）

#### （1）基本特性

```python
a = 3.14
b = 1.0
```

* 基于 **IEEE 754 双精度**
* 本质是 C 的 `double`

---

#### （2）精度问题

```python
0.1 + 0.2 == 0.3   # False
```

工程建议：

* 不用 `==` 比较浮点数
* 使用误差范围

---

### 3. 复数型（complex）

```python
z = 1 + 2j
```

* `j` 表示虚部
* 支持基本运算

```python
z.real
z.imag
```

常用于：

* 信号处理
* 科学计算

---

## 3.2 序列类型（Sequence Types）

序列类型的共同特征：

* 有顺序
* 可索引
* 可切片
* 可迭代

---

### 1. 字符串（str）

#### （1）基本特性

```python
s = "hello"
```

* 本质是 **不可变字符序列**
* 支持索引 / 切片

```python
s[0]
s[1:4]
```

---

#### （2）不可变性

```python
s[0] = 'H'   # ❌
```

修改字符串 = 创建新字符串。

---

### 2. 列表（list）

#### （1）基本特性

```python
lst = [1, 2, 3]
```

* **可变序列**
* 元素类型可不同

---

#### （2）常见操作

```python
lst.append(4)
lst[0] = 10
```

列表是 Python 中最常用的容器。

---

### 3. 元组（tuple）

#### （1）基本特性

```python
t = (1, 2, 3)
```

* **不可变序列**
* 常用于：

  * 多值返回
  * 结构化数据

---

#### （2）单元素元组

```python
t = (1,)   # 注意逗号
```

---

## 3.3 布尔型（bool）

布尔型用于**逻辑判断**。

```python
True
False
```

### 1. 与 int 的关系

```python
isinstance(True, int)   # True
```

* `bool` 是 `int` 的子类
* `True == 1`

工程注意：

> 语义上要区分“布尔”和“数值”。

---

## 3.4 字典（dict）

字典是 **键值映射结构（Key-Value）**。

---

### 1. 基本用法

```python
d = {
    "name": "Alice",
    "age": 18
}
```

* Key 必须可哈希
* Value 任意类型

---

### 2. 工程特性

* 查找速度快（哈希表）
* 无序（逻辑无序，3.7+ 保留插入顺序）

字典是 Python 的**核心数据结构**。

---

## 3.5 集合类型（Set Types）

集合用于表示 **不重复元素的无序集合**。

---

### 1. 可变集合（set）

```python
s = {1, 2, 3}
```

特点：

* 元素唯一
* 可增删

```python
s.add(4)
```

---

### 2. 不可变集合（frozenset）

```python
fs = frozenset([1, 2, 3])
```

特点：

* 不可变
* 可作为 dict 的 key

用于：

* 集合的集合
* 只读配置

---

## 3.6 空值类型（NoneType）

### 1. None 的含义

```python
x = None
```

表示：

* 没有值
* 占位
* 函数无返回

---

### 2. 工程语义

```python
def func():
    pass
```

等价于：

```python
def func():
    return None
```

`None` 是一个对象，不是 0，也不是 False。

---

## 3.7 总结

1. Python 所有数据都是对象
2. 类型决定行为，而不是变量
3. 可变 / 不可变是理解 Python 的核心
4. 字典和集合是性能与结构设计关键
5. None 是 Python 语义的重要组成部分

---

# 四、Python内建容器（Container）

## 4.1 什么是 Python 容器

在 Python 中，**容器（Container）**指的是：

> **能够容纳多个对象，并以某种规则组织和访问这些对象的数据类型。**

### 容器的共同特征

* 可以包含多个元素
* 元素本身仍然是 Python 对象
* 通常可以被遍历（`for`）

### 注意区分：

* **容器**：`list / dict / set / tuple / str`
* **标量**：`int / float / bool / None`

---

## 4.2 Python 内建“核心容器”一览

Python 的内建核心容器可以分为 **三大类**：

> **序列（Sequence） / 映射（Mapping） / 集合（Set）**

---

### 4.2.1 序列容器（Sequence）

序列容器的共同特性：

* **有顺序**
* **可索引**
* **可切片**
* **允许元素重复**

#### 1. 字符串（str）

##### （1）本质与特性

```python
s = "hello"
```

* 本质：**字符的不可变序列**
* 有顺序
* 不可变
* 支持索引、切片、遍历

```python
s[0]      # 'h'
s[1:4]    # 'ell'
```

---

##### （2）常用操作

1️⃣ 定义

```python
s = "hello"
s = 'world'
```

字符串可以使用单引号或双引号定义。

2️⃣ 下标索引

```python
s = "hello"

s[0]     # 'h'
s[-1]    # 'o'
```

字符串是**有序序列**，支持正向和负向索引。

3️⃣ 遍历

```python
for ch in s:
    print(ch)
```

遍历得到的是 **单个字符（字符串长度为 1）**。

4️⃣ 常用方法（字符串方法）

```python
len(s)
```

```python
s.upper()
s.lower()
s.replace("l", "L")
s.strip()
```

所有字符串方法**都不会修改原字符串**。

---

###### ⚠ 注意事项（非常重要）

1. **修改字符串 ≠ 原地修改**
2. 每一次字符串操作都会生成 **新的字符串对象**

```python
s = "abc"
s.upper()

print(s)   # 仍然是 "abc"
```

> **字符串是不可变的序列类型，
> 所有“修改”操作本质上都是创建新字符串。**

---

##### （3）典型使用场景

* 文本处理
* 协议解析
* 配置 / 日志

---

#### 2. 列表（list）

##### （1）本质与特性

```python
lst = [1, 2, 3]
```

* 有顺序
* 可重复
* **可变**
* 元素类型可以不同
* 本质是 **对象引用的动态序列**

---

##### （2）列表定义

* 1. 使用方括号 `[]`

```python
lst = [1, 2, 3]
lst = ["a", 1, 3.14]
```

---

* 2. 使用 `list()` 构造

```python
lst = list()
lst = list("abc")     # ['a', 'b', 'c']
lst = list(range(5))  # [0, 1, 2, 3, 4]
```

`list()` 接受 **可迭代对象** 作为参数。

---

* 3. 空列表

```python
lst = []
```

---

##### （3）列表索引

* 1. 正向索引（从 0 开始）

```python
lst = [10, 20, 30, 40]

lst[0]   # 10
lst[2]   # 30
```

---

* 2. 负向索引（从 -1 开始）

```python
lst[-1]  # 40
lst[-2]  # 30
```

---

* 3. 切片（slice）

```python
lst[1:3]     # [20, 30]
lst[:2]      # [10, 20]
lst[2:]      # [30, 40]
lst[::2]     # [10, 30]
```

切片返回的是 **新列表**。

---

##### （4）越界访问

```python
lst[10]      # IndexError
```

Python 会进行边界检查，不会产生未定义行为。

---

##### （5）常用操作（列表的方法）

下面是**工程中最常用、必须熟悉的列表 API**。

---

1️⃣ 添加元素

```python
lst.append(x)        # 末尾添加一个元素
lst.extend(iterable) # 追加多个元素
lst.insert(i, x)     # 在指定位置插入
```

示例：

```python
lst = [1, 2]
lst.append(3)        # [1, 2, 3]
lst.extend([4, 5])   # [1, 2, 3, 4, 5]
lst.insert(0, 10)    # [10, 1, 2, 3, 4, 5]
```

工程建议：

* `append` 最常用
* `insert` 代价较高（涉及元素整体移动）

---

2️⃣ 删除元素

```python
lst.pop()        # 删除并返回最后一个
lst.pop(i)       # 删除指定索引
lst.remove(x)    # 删除第一个等于 x 的元素
lst.clear()      # 清空列表
```

示例：

```python
lst = [1, 2, 3, 2]
lst.pop()        # 返回 2
lst.remove(2)    # 删除第一个 2
```

区别重点：

* `pop` 按 **位置**
* `remove` 按 **值**

---

3️⃣ 查询与统计

```python
lst.count(x)     # 统计 x 出现次数
lst.index(x)     # 返回 x 第一次出现的索引
```

```python
len(lst)         # 列表长度
x in lst         # 成员判断
```

---

4️⃣ 排序与反转

```python
lst.sort()                 # 原地排序
lst.sort(reverse=True)
lst.reverse()              # 原地反转
```

```python
sorted(lst)                # 返回新列表
```

> `sort()` 改变原列表
> 
> `sorted()` 不改变原列表

---

5️⃣ 修改元素

```python
lst[0] = 100
lst[1:3] = [7, 8]
```

---

##### （6）性能与工程注意点

* `append`：摊还 O(1)
* `insert / remove`：O(n)
* 列表适合：

  * 顺序访问
  * 末尾追加
* 不适合：

  * 频繁头部插入
  * 大规模查找（可考虑 `set` / `dict`）

典型使用场景

* 动态数据收集
* 顺序数据处理
* 临时缓存
* 作为其他数据结构的基础（栈 / 队列）

---

#### 3. 元组（tuple）

##### （1）本质与特性

```python
t = (1, 2, 3)
```

* 有顺序
* 可重复
* **不可变**，一旦定义完成就不可修改
* 通常表示“结构”

---

##### （2）使用细节

1️⃣ 元组的定义

**（1）基本定义方式**

```python
t = (1, 2, 3)
```

**（2）省略括号定义（Python 特性）**

```python
t = 1, 2, 3
```

本质仍然是元组，逗号 `,` 才是关键。

**（3）单元素元组（重点）**

```python
t = (1,)
```

❌ 错误写法：

```python
t = (1)    # 这是 int，不是 tuple
```

判断依据：

```python
type(t)    # <class 'tuple'>
```

---

**（4）空元组**

```python
t = ()
```

---

2️⃣ 元组的操作方法

* 1. 解包（tuple unpacking）

```python
a, b = (1, 2)
```

本质是：

```python
a = 1
b = 2
```

左右元素个数必须一致：

```python
a, b = (1, 2, 3)   # ❌ ValueError
```

---

* 2. 多变量交换（经典用法）

```python
a, b = b, a
```

无需临时变量，这是 Python 的语言级支持。

---

* 3. 索引与切片（只读）

```python
t = (10, 20, 30)

t[0]      # 10
t[1:3]    # (20, 30)
```

❌ 不允许修改：

```python
t[0] = 100   # TypeError
```

---

* 4. 常用内建操作（非修改）

```python
len(t)
max(t)
min(t)
```

```python
t.count(20)
t.index(30)
```

注：元组的方法非常少，这是刻意的设计。

---

* 5. 元组的“浅不可变性”（重要理解）

```python
t = (1, [2, 3])
t[1].append(4)   # 合法
```

不可变的是：

* 元组中**元素引用**
* 不是元素对象本身

---

##### （3）典型使用场景

* 函数多返回值
* 不可变配置
* 作为 dict 的 key（前提：元素可哈希）

---

### 4.2.2 映射容器（Mapping）

---

#### 4. 字典（dict）

##### （1）本质与特性

```python
d = {"name": "Alice", "age": 18}
```

* **键值对（key → value）**
* Key 必须唯一
* Key 必须可哈希
* Value 任意类型

dict **不是序列**，访问方式是：

```python
d["name"]
```

---

##### （2）常用操作
1. 定义字典
2. 新增、更新、删除元素
3. 清空元素
4. 获取key
5. 遍历
```python
d["age"] = 20
d.get("score", 0)
d.keys()
d.values()
d.items()
```

---

##### （3）顺序问题（工程重点）

* Python 3.7+ 保留**插入顺序**
* 但 **dict 的本质仍是映射，不应依赖顺序逻辑**

---

##### （4）典型使用场景

* 查表
* 配置
* JSON / 数据结构表达

---

### 4.2.3 集合容器（Set）

集合容器的共同特性：

* **逻辑无序**
* **元素唯一**
* 不支持索引

---

#### 5. 可变集合（set）

##### （1）本质与特性

```python
s = {1, 2, 3}
```

* 无序
* 不重复
* 可变

---

##### （2）常用操作
1. 集合定义
2. 添加、移除、随机取出元素
3. 清空集合
4. 交、并、差集
5. 遍历

```python
s.add(4)
s.remove(2)
```

```python
s1 | s2    # 并集
s1 & s2    # 交集
s1 - s2    # 差集
```

---
好，这一节我**完全按你给出的 5 点框架来写**，并且保持与你前面字符串、列表那一套**“定义 → 操作 → 注意事项”的笔记风格一致**，可以直接放进同一章里。

---

##### （2）常用操作

1️⃣ 集合定义

```python
s = {1, 2, 3}
```

```python
s = set([1, 2, 3])
```

注意：

* `{}` 表示 **空字典**
* 空集合必须用：

```python
s = set()
```

---

2️⃣ 添加、移除、随机取出元素

**添加元素**

```python
s.add(4)
```

---

**移除指定元素**

```python
s.remove(2)     # 元素不存在会报错
```

```python
s.discard(2)    # 元素不存在不会报错
```

---

**随机取出一个元素**

```python
x = s.pop()
```

`pop()` 取出的是 **任意元素**，不是“最后一个”。

---

3️⃣ 清空集合

```python
s.clear()
```

清空后集合仍然存在，只是变为空集合。

---

4️⃣ 交、并、差集

```python
s1 | s2    # 并集
s1 & s2    # 交集
s1 - s2    # 差集
s1 ^ s2    # 对称差集
```

等价方法形式：

```python
s1.union(s2)
s1.intersection(s2)
s1.difference(s2)
s1.symmetric_difference(s2)
```

推荐在表达数学含义时使用运算符形式，更直观。

---

5️⃣ 遍历

```python
for x in s:
    print(x)
```

注：遍历顺序是 **不确定的**，不能依赖顺序。

---

###### ⚠ 注意事项（非常重要）

1. 集合是 **无序** 的
2. 集合中的元素必须 **可哈希**
3. 集合不支持下标访问：

```python
s[0]   # ❌ TypeError
```

---

##### （3）典型使用场景

* 去重
* 成员判断
* 集合运算

---

#### 6. 不可变集合（frozenset）

##### （1）本质与特性

```python
fs = frozenset([1, 2, 3])
```

* 无序
* 不重复
* **不可变**
* 可哈希

---

##### （2）典型使用场景

* 作为 dict 的 key
* 只读集合
* 高级数据结构

---

## 4.3 对照总结表（核心）

| 容器        | 有序 | 可重复   | 可变 | 可索引 | 本质    |
| --------- | -- | ----- | -- | --- | ----- |
| str       | ✅  | ✅     | ❌  | ✅   | 字符序列  |
| list      | ✅  | ✅     | ✅  | ✅   | 可变序列  |
| tuple     | ✅  | ✅     | ❌  | ✅   | 不可变序列 |
| dict      | ❌  | Key ❌<br> Value ✅ | ✅  | ❌   | 键值映射  |
| set       | ❌  | ❌     | ✅  | ❌   | 数学集合  |
| frozenset | ❌  | ❌     | ❌  | ❌   | 不变集合  |

* dict 不应作为顺序容器使用。
* 无序表明不支持下标索引

---

## 4.4 通用操作

### 4.4.1 容器通用方法

#### 1. len()
#### 2. max()
#### 3. min()

---

### 4.4.2 容器转换

### 4.4.3 排序 sorted()

---

好，这一小节你拆得**非常专业**，已经是「容器 → 抽象接口 → 通用算法」的思路了 👍
我按你给的 **4.4 通用操作**框架，写一份**可直接并入第四章的正式总结笔记**。

---

## 4.4 通用操作

Python 的内建容器虽然类型不同，但提供了一组**高度统一的通用操作接口**，这是 Python “一切皆对象 + 鸭子类型”设计思想的体现

### 4.4.1 容器通用方法

#### 1. `len()`

```python
len(container)
```

**作用：**

* 返回容器中**元素个数**

**适用容器：**

* `str`
* `list`
* `tuple`
* `set`
* `dict`（返回 key 的数量）

**示例：**

```python
len([1, 2, 3])          # 3
len("hello")           # 5
len({"a": 1, "b": 2})  # 2
```

---

#### 2. `max()`

```python
max(container)
```

**作用：**

* 返回容器中的**最大元素**

**适用条件：**

* 元素之间 **必须可比较**
* 类型需一致或定义了比较规则

**示例：**

```python
max([1, 5, 3])          # 5
max("abc")              # 'c'
max({"a": 1, "b": 2})   # 'b'（比较 key）
```

⚠️ 注意：

```python
max([1, "a", 3])  # TypeError
```

---

#### 3. `min()`

```python
min(container)
```

**作用：**

* 返回容器中的**最小元素**

**示例：**

```python
min([1, 5, 3])    # 1
min("abc")        # 'a'
```

---

### 4.4.2 容器转换

Python 提供了一组**类型构造函数**，用于在容器之间转换。

---

#### 常见转换函数

| 函数        | 说明          |
| --------- | ----------- |
| `list()`  | 转为列表        |
| `tuple()` | 转为元组        |
| `set()`   | 转为集合        |
| `dict()`  | 转为字典（需成对数据） |

---

#### 示例

```python
list("abc")        # ['a', 'b', 'c']
tuple([1, 2, 3])   # (1, 2, 3)
set([1, 2, 2, 3])  # {1, 2, 3}
```

字典转换：

```python
dict([("a", 1), ("b", 2)])
```

⚠️ 注意：

* `set` 会 **去重**
* `dict` 转换要求 **二元结构**

---

### 4.4.3 排序 `sorted()`

```python
sorted(iterable)
```

**作用：**

* 对可迭代对象进行排序
* **返回新列表**
* 不修改原容器

---

#### 基本使用

```python
sorted([3, 1, 2])     # [1, 2, 3]
sorted("cba")         # ['a', 'b', 'c']
```

---

#### 对字典排序

```python
d = {"b": 2, "a": 1}
sorted(d)             # ['a', 'b']
```

* 默认按 **key 排序**

---

#### 常用参数

```python
sorted(iterable, reverse=True)
```

```python
sorted(data, key=len)
```

---

#### 对比：`sorted()` vs `list.sort()`

| 项目     | `sorted()` | `list.sort()` |
| ------ | ---------- | ------------- |
| 是否通用   | 所有可迭代对象    | 仅 list        |
| 是否原地修改 | 否          | 是             |
| 返回值    | 新列表        | None          |

---

## 4.5 重要补充（工程理解）

### 1️⃣ 可变 vs 不可变是设计核心

* 可变：`list / dict / set`
* 不可变：`str / tuple / frozenset`

决定：

* 是否能作为 dict 的 key
* 是否安全共享

---

### 2️⃣ 容器里存的是“引用”

```python
a = [1, 2]
b = a
```

* `a` 和 `b` 指向同一对象

---

## 4.6 本章总结

1. Python 核心容器分为三类：序列、映射、集合
2. 序列有顺序，集合去重，字典是映射
3. 可变性决定设计方式
4. 选对容器，代码复杂度直接下降

---

# 五、程序流程结构

## 5.1 选择结构（条件分支）

选择结构用于：

> **根据条件的真假，决定执行哪一段代码。**

Python 中的条件判断基于 **布尔表达式（True / False）**。

---

### 5.1.1 `if / elif / else`

#### （1）基本语法结构

```python
if condition:
    # 条件为 True 时执行
elif another_condition:
    # 前面条件不满足，且该条件为 True
else:
    # 所有条件都不满足
```

关键点：

* 使用 **冒号 `:`**
* 使用 **缩进** 表示代码块（而不是 `{}`）
* `elif` 和 `else` 都是可选的

---

#### （2）条件表达式

```python
x > 0
x == 10
x in [1, 2, 3]
```

条件的结果一定是：

```python
True 或 False
```

---

#### （3）示例

```python
score = 85

if score >= 90:
    print("A")
elif score >= 60:
    print("B")
else:
    print("C")
```

---

#### （4）工程建议

* 条件从**最严格 / 最特殊**写到**最宽泛**
* 避免条件重叠
* 不要写过长的 `if-elif` 链（>5 个通常需要重构）

---

### 5.1.2 `if` 嵌套

#### （1）什么是 if 嵌套

```python
if condition1:
    if condition2:
        ...
```

即：**一个 if 语句内部再写 if**

---

#### （2）示例

```python
age = 20
vip = True

if age >= 18:
    if vip:
        print("允许进入 VIP 区")
```

---

#### （3）工程注意事项（非常重要）

* 嵌套层数过深 → 可读性急剧下降
* 通常 **超过 3 层就需要优化设计**

#### 常见优化方式：

* 使用 `elif`
* 使用提前 `return`
* 合并条件

```python
if age >= 18 and vip:
    print("允许进入 VIP 区")
```

---

## 5.2 循环结构

循环结构用于：

> **重复执行一段代码，直到条件不满足或遍历完成。**

---

### 5.2.1 `for` 循环

#### （1）for 的本质（非常重要）

Python 的 `for` 不是“计数循环”，而是：

> **遍历一个可迭代对象（iterable）**

---

#### （2）基本语法

```python
for item in iterable:
    # 使用 item
```

---

#### （3）示例

```python
for x in [1, 2, 3]:
    print(x)
```

```python
for ch in "hello":
    print(ch)
```

---

#### （4）适用场景

* 遍历列表、字符串、字典、集合
* 顺序处理数据
* Python 中**最常用的循环**

---

### 5.2.2 `while` 循环

#### （1）基本语法

```python
while condition:
    # 条件为 True 时反复执行
```

---

#### （2）示例

```python
i = 0
while i < 5:
    print(i)
    i += 1
```

---

#### （3）注意事项

* 必须确保循环条件**最终会变为 False**
* 否则会产生 **死循环**

while 更适合：

* 不确定循环次数
* 条件驱动型逻辑

---

### 5.2.3 `range`

`range` 是一个**用于生成整数序列的内建工具**，常配合 `for` 使用。

---

#### （1）基本用法

```python
range(stop)
range(start, stop)
range(start, stop, step)
```

---

#### （2）示例

```python
for i in range(5):
    print(i)      # 0~4
```

```python
for i in range(1, 10, 2):
    print(i)      # 1,3,5,7,9
```

> 注：`range` **不生成列表**，而是一个惰性序列对象。

---

### 5.2.4 `continue` 与 `break`

---

#### （1）`break` —— 终止循环

```python
for i in range(10):
    if i == 5:
        break
    print(i)
```

效果：循环在 `i == 5` 时直接结束。

---

#### （2）`continue` —— 跳过本次循环

```python
for i in range(5):
    if i == 2:
        continue
    print(i)
```

效果：跳过 `i == 2`，继续下一次。

---

#### （3）工程使用建议

* `break`：用于提前结束（性能优化）
* `continue`：用于过滤无效数据
* 不要滥用，避免逻辑混乱

---

### 5.2.5 嵌套循环

#### （1）什么是嵌套循环

```python
for i in range(3):
    for j in range(2):
        print(i, j)
```

---

#### （2）执行逻辑

* 外层循环每执行一次
* 内层循环完整执行一遍

---

#### （3）工程重点（必须理解）

* 嵌套循环 → **时间复杂度迅速上升**
* 两层：O(n²)
* 三层：O(n³)

实战中应警惕：

* 不必要的嵌套
* 可用数据结构（dict / set）替代

---

## 5.3 总结

1. 程序流程由 **选择结构 + 循环结构** 组成
2. `if` 决定执行路径
3. `for` 是 Python 最核心的循环方式
4. `while` 适合条件驱动逻辑
5. 嵌套结构影响可读性和性能
6. 控制流设计是“写好程序”的关键能力

---

# 六、函数（Function）

函数是 **Python 程序的基本组织单元**。
通过函数，可以将一段逻辑封装起来，实现**复用、抽象与模块化**，是从“写脚本”迈向“写程序”的关键一步。

## 6.1 什么是函数

### 1. 函数的定义与作用

函数是：

* 一段**可被重复调用的代码**
* 接收输入（参数）
* 执行处理逻辑
* 可选择性返回结果

函数的核心作用：

* **代码复用**：避免重复代码
* **模块化**：拆分复杂问题
* **抽象能力**：隐藏实现细节
* **可维护性**：修改集中在一处

---

### 2. 函数的基本结构

```python
def add(a, b):
    result = a + b
    return result
```

函数由四部分组成：

1. `def`：定义函数
2. 函数名：标识函数
3. 参数列表：函数输入
4. 函数体 + 返回值

---

## 6.2 函数的定义与调用

### 1. `def` 关键字

```python
def greet():
    print("Hello Python")
```

* 使用 `def` 定义函数
* 函数名遵循 **snake_case**
* 函数体必须缩进

---

### 2. 函数的调用

```python
greet()
```

调用函数时：

* 程序跳转到函数内部执行
* 执行完毕后返回调用位置

---

### 3. 函数的执行顺序（调用栈）

```python
def f1():
    f2()

def f2():
    print("in f2")

f1()
```

执行顺序：

1. 调用 `f1`
2. `f1` 调用 `f2`
3. `f2` 执行完成
4. 返回 `f1`
5. 程序继续执行

> 本质：**后进先出（栈结构）**

---

## 6.3 参数机制（重点）

### 1. 位置参数

```python
def sub(a, b):
    return a - b

sub(5, 3)
```

* 按位置传参
* 顺序不能错

---

### 2. 关键字参数

```python
sub(b=3, a=5)
```

* 指定参数名
* 顺序可打乱
* 可读性更好

---

### 3. 默认参数

```python
def power(x, n=2):
    return x ** n
```

* 默认参数必须放在**参数列表末尾**
* 默认参数在**函数定义时绑定**

⚠️ **不要用可变对象作为默认参数**

---

### 4. 可变参数

可变参数，用于调用函数时传递参数数量不确定时的场景

#### （1）`*args`：接收任意位置参数

```python
def add_all(*args):
    return sum(args)
```

* `args` 本质是 **元组**

---

#### （2）`**kwargs`：接收任意关键字参数

```python
def show_info(**kwargs):
    print(kwargs)
```

* `kwargs` 本质是 **字典**

---

## 6.4 返回值

### 1. `return` 的作用

* 结束函数执行
* 向调用方返回结果

---

### 2. 单返回值

```python
def square(x):
    return x * x
```

---

### 3. 多返回值（元组）

Python 中函数表面上可以“返回多个值”，但**语言层面始终只返回一个对象**。

```python
def calc(a, b):
    return a + b, a - b
```

---

#### 本质说明

上述写法等价于：

```python
return (a + b, a - b)
```

即：

* Python 会自动将多个返回值 **打包（packing）** 成一个 **元组（tuple）**
* 函数的真实返回值类型是 `tuple`

---

#### 接收方式一：单变量接收（元组）

```python
result = calc(5, 3)
print(result)        # (8, 2)
print(type(result))  # tuple
```

此时：

* `result` 指向一个**新创建的元组对象**
* 元组中的每个元素对应一个返回结果

---

#### 接收方式二：多变量解包接收（推荐）

```python
s, d = calc(5, 3)
```

这是 **序列解包（unpacking）** 语法，等价于：

```python
_tmp = calc(5, 3)
s = _tmp[0]
d = _tmp[1]
```

解包只是语法糖，**元组一定已经存在**。

---

#### 注意事项

1. **返回值个数与接收变量个数必须一致**

```python
a, b = calc(5, 3)   # ✅
a = calc(5, 3)      # ✅（接收整个元组）
a, b, c = calc(5, 3)  # ❌ ValueError
```

2. 多返回值的顺序 **由返回语句顺序决定**

---

### 4. `None` 返回值

在 Python 中，如果函数**没有显式使用 `return` 返回值**，那么：

> **函数的返回值默认为 `None`**

#### （1）未写 `return`

```python
def f():
    x = 10
```

```python
ret = f()
print(ret)
```

输出：

```text
None
```

#### （2）显式 `return None`

```python
def f():
    return None
```

注：这与**不写 `return` 的效果完全一致**。

#### （3）`return` 但不带值

```python
def f():
    if True:
        return
```

等价于：

```python
return None
```

#### （4）`None` 的语义含义

`None` 表示：

* 没有值
* 空结果
* 占位符
* 执行完成但无有效返回数据

常用于：

* 仅执行动作（副作用）的函数
* 初始化占位
* 失败 / 未命中状态

#### （5）工程建议

* 有返回值的函数：**必须 return**
* 无返回值的函数：**明确设计为返回 None**
* 不要混用“有时返回值、有时不返回”

---

### 5. `return` vs `print`

| 对比项  | return | print |
| ---- | ------ | ----- |
| 返回结果 | ✅      | ❌     |
| 用于逻辑 | ✅      | ❌     |
| 用于调试 | ❌      | ✅     |

---

## 6.5 嵌套函数

**嵌套函数（Nested Function）**指：

> **在一个函数内部再定义另一个函数**

---

### 1. 基本概念与形式

```python
def outer():
    def inner():
        print("inner")
    inner()
```

* `inner` 是 `outer` 的**局部函数**
* 只能在 `outer` 内部使用

---

### 2. 作用域关系（LEGB 中的 E）

```python
def outer():
    x = 10
    def inner():
        print(x)
    inner()
```

* `x` 属于 **Enclosing 作用域**
* `inner` 可以访问外层函数变量

这是 **LEGB 中的 E（Enclosing）**

---

### 3. 修改外层函数变量：`nonlocal`

```python
def outer():
    x = 0
    def inner():
        nonlocal x
        x += 1
    inner()
    print(x)
```

输出：

```text
1
```

说明：

* `nonlocal` 用于声明：
  **变量来自外层函数作用域**

⚠️ 不使用 `nonlocal` 会报错或创建新局部变量。

---

### 4. 嵌套函数的常见用途

#### （1）逻辑分层（辅助函数）

```python
def process(data):
    def validate(d):
        return d > 0
    if validate(data):
        print("ok")
```

* 内部逻辑不对外暴露
* 减少全局命名污染

---

#### （2）代码复用（局部工具）

```python
def calc(a, b):
    def add(x, y):
        return x + y
    return add(a, b)
```

---

#### （3）封装状态（闭包前置）

```python
def counter():
    cnt = 0
    def inc():
        nonlocal cnt
        cnt += 1
        return cnt
    return inc
```

---

### 5. 返回嵌套函数（闭包）

```python
f = counter()
print(f())   # 1
print(f())   # 2
```

* 内层函数“记住”了外层变量
* 这就是 **闭包（Closure）**

---

### 6. 嵌套函数 vs 普通函数

| 对比项     | 嵌套函数  | 普通函数 |
| ------- | ----- | ---- |
| 定义位置    | 函数内部  | 模块级  |
| 可见范围    | 外层函数内 | 全模块  |
| 可访问外层变量 | ✅     | ❌    |
| 是否支持闭包  | ✅     | ❌    |

---

### 7. 常见错误与注意点

---

#### （1）误以为能修改外层变量

```python
def outer():
    x = 1
    def inner():
        x = 2   # 新局部变量
```

❌ 不会修改 `outer` 的 `x`

---

#### （2）忘记 `nonlocal`

```python
def outer():
    x = 1
    def inner():
        x += 1   # UnboundLocalError
```

---

### 8. 工程建议

* 嵌套函数适合：

  * 辅助逻辑
  * 临时工具
  * 闭包
* 不宜嵌套过深（可读性差）
* 复杂逻辑建议拆为模块级函数

---

## 6.6 作用域（Scope）

### 1. 局部变量

```python
def f():
    x = 10
```

* 仅在函数内部可见

---

### 2. 全局变量

```python
x = 10

def f():
    print(x)
```

* 可在整个模块中使用

---

### 3. `global` 关键字

```python
x = 0

def inc():
    global x
    x += 1
```

⚠️ 工程中应 **尽量避免使用**

---

### 4. LEGB 规则

查找变量顺序：

1. Local —— 局部作用域
2. Enclosing —— 外层函数作用域（嵌套函数）
3. Global —— 全局作用域
4. Built-in —— 内建作用域

---

## 6.7 函数与数据类型

### 1. 参数传递机制

Python 采用：

> **对象引用传递（call by object reference）**

---

### 2. 可变对象 vs 不可变对象

| 类型  | 示例                | 函数内修改  |
| --- | ----------------- | ------ |
| 不可变 | int / str / tuple | 不影响外部  |
| 可变  | list / dict / set | 可能影响外部 |

---

### 3. 函数对容器的影响

```python
def add_item(lst):
    lst.append(1)
```

* 会修改原列表

---

## 6.8 函数的文档与规范

### 1. 文档字符串（docstring）

```python
def add(a, b):
    """返回 a 与 b 的和"""
    return a + b
```

* 三个引号之间的内容是函数的说明文档，跟注释差不多，帮助更好理解代码

---

### 2. 类型注解（基础）

```python
def add(a: int, b: int) -> int:
    return a + b
```

* 不影响运行
* 提高可读性

---

### 3. 命名规范

* 函数名：`snake_case`
* 动词 + 名词
* 避免歧义

---

## 6.9 函数与方法

---

## 6.10 函数的工程实践建议

### 1. 单一职责原则

> 一个函数只做一件事

---

### 2. 控制函数长度

* 建议 ≤ 30 行
* 过长应拆分

---

### 3. 减少副作用

* 尽量返回新值
* 少修改外部变量

---

### 4. 返回值设计

* 有结果 → `return`
* 执行动作 → 无返回或返回状态

---

## 6.11 总结

* 函数是 Python 程序的核心抽象单元
* 理解参数机制是关键
* 正确处理作用域与可变对象
* 工程中应重视函数设计，而不仅是语法

---

# 七、函数进阶（Function Advanced）

> 核心思想一句话概括：
> 
> **在 Python 中，函数和普通数据一样，是“一等对象”**

## 7.1 函数是一等对象

### 1. 什么叫“一等对象”？

函数在 Python 中：

* 可以赋值给变量
* 可以作为参数传递
* 可以作为返回值返回
* 可以存进容器（list / dict）

这点是 Python 和 C / C++ 的**本质差异之一**

---

### 2. 函数赋值

#### 基本示例

```python
def add(a, b):
    return a + b

f = add     # 没有加括号
```

此时：

* `add` 是函数对象
* `f` 指向同一个函数对象

```python
print(f(2, 3))   # 5
```

#### 注意点（非常重要）

```python
f = add      # 赋值的是函数本身
f = add()    # ❌ 调用函数，把返回值赋给 f
```

---

### 3. 函数作为参数

#### 示例：把函数传给另一个函数

```python
def calc(a, b, fn):
    return fn(a, b)

def add(a, b):
    return a + b

def mul(a, b):
    return a * b

print(calc(2, 3, add))  # 5
print(calc(2, 3, mul))  # 6
```

#### 本质理解

```text
fn = add
fn(a, b) → add(a, b)
```

传递的是**调用函数的逻辑**

这就是 **回调函数（callback）** 的原型。

---

### 4. 函数作为返回值

#### 示例

```python
def make_adder(x):
    def adder(y):
        return x + y
    return adder
```

使用：

```python
add5 = make_adder(5)
print(add5(10))    # 15
```

#### 发生了什么？

* `make_adder` 返回了一个函数对象
* 返回的函数“记住了” `x`

👉 这是 **闭包的前置知识**

---

## 7.2 高阶函数（Higher-Order Function）

### 1. 定义

**满足以下任一条件的函数**：

* 接收函数作为参数
* 返回函数作为结果

---

### 2. 常见内置高阶函数

#### 1️⃣ `map(function, iterable)`

```python
nums = [1, 2, 3, 4]

res = map(lambda x: x * 2, nums)
print(list(res))   # [2, 4, 6, 8]
```

---

#### 2️⃣ `filter(function, iterable)`

```python
nums = [1, 2, 3, 4, 5]

res = filter(lambda x: x % 2 == 0, nums)
print(list(res))   # [2, 4]
```

---

#### 3️⃣ `sorted(iterable, key=...)`

```python
words = ["apple", "banana", "kiwi"]

print(sorted(words, key=len))
```

`key` 接收的是 **函数**

---

## 7.3 lambda 匿名函数

### 1. 什么是 lambda？

* 没有名字
* 只写一行
* 自动 `return`

#### 语法

```python
lambda 参数: 表达式
```

---

### 2. 示例

```python
add = lambda a, b: a + b
print(add(2, 3))
```

等价于：

```python
def add(a, b):
    return a + b
```

---

### 3. lambda 的使用场景

✔ 临时用一次
✔ 配合高阶函数
❌ 不适合复杂逻辑

```python
sorted(words, key=lambda s: s[-1])
```

---

## 7.4 闭包（Closure）

### 1. 什么是闭包？

> **内部函数引用了外部函数的变量，且外部函数已经执行结束**

---

### 2. 经典示例

```python
def outer(x):
    def inner(y):
        return x + y
    return inner
```

```python
f = outer(10)
print(f(5))   # 15
```

---

### 闭包的三个条件（考试级）

1️⃣ 有嵌套函数
2️⃣ 内部函数使用外部变量
3️⃣ 外部函数返回内部函数

---

### `nonlocal` 的作用

```python
def counter():
    count = 0
    def inc():
        nonlocal count
        count += 1
        return count
    return inc
```

`nonlocal` 用于：

* **修改** 外层函数的局部变量
* 否则 Python 会当作新局部变量

---

### 闭包的意义

* 保存状态
* 不使用全局变量
* 函数“带记忆”

👉 装饰器的核心基础

---

## 7.5 装饰器（Decorator）

### 1. 本质一句话

> **装饰器 = 高阶函数 + 闭包 + 语法糖**

---

### 2. 最原始的函数增强写法

```python
def decorator(fn):
    def wrapper():
        print("before")
        fn()
        print("after")
    return wrapper
```

使用：

```python
def hello():
    print("hello")

hello = decorator(hello)
hello()
```

---

### 使用装饰器语法糖 `@`

```python
@decorator
def hello():
    print("hello")
```

等价于：

```python
hello = decorator(hello)
```

---

### 带参数的装饰器

```python
def decorator(fn):
    def wrapper(*args, **kwargs):
        print("before")
        res = fn(*args, **kwargs)
        print("after")
        return res
    return wrapper
```

---

### 装饰器常见用途

* 日志记录
* 权限校验
* 性能计时
* 缓存（`lru_cache`）

---

## 7.6 总结

> Python 的函数不仅是“可调用代码”，更是**可以被传递、包装、增强、保存状态的对象**。
>
> **装饰器不是新东西，只是函数对象能力的自然结果。**

---

# 八、文件操作管理

## 8.1 文件基础知识

### 1. 文件编码

* **编码的概念**：
  文件是字节序列，编码决定字节与字符的对应关系。
* **常见编码**：

  * **ASCII**：只支持基本英文字符
  * **UTF-8**：常用的国际标准，支持所有Unicode字符
  * **GBK/GB2312**：中文编码，兼容较差，不推荐使用
* **Python中的编码问题**：

  * 默认文件读写以UTF-8编码为主
  * 打开文件时可通过`encoding`参数指定编码，避免中文乱码
  * 例：`open("file.txt", "r", encoding="utf-8")`

---

### 2. 常见文件类型（文件拓展名）

| 类型    | 说明             | 常见拓展名                           |
| ----- | -------------- | ------------------------------- |
| 文本文件  | 可读文本内容         | `.txt`, `.csv`, `.log`, `.json` |
| 二进制文件 | 非文本文件，保存原始字节数据 | `.bin`, `.jpg`, `.png`, `.exe`  |
| 压缩文件  | 多文件或大文件压缩包     | `.zip`, `.tar`, `.gz`           |

---

## 8.2 文件基础操作（增删改查）

### 1. 打开文件

* 使用`open()`函数打开文件
* 常用参数：

  * 文件路径（字符串）
  * 模式（mode）: `'r'`（读，默认）、`'w'`（写，覆盖）、`'a'`（追加）、`'b'`（二进制模式）
  * 编码（encoding）：通常文本文件用`utf-8`
* 示例：

```python
f = open("test.txt", "r", encoding="utf-8")
```

---

### 2. 读取文件

* `read(size=-1)`：读取全部或指定字节数
* `readline()`：读取一行
* `readlines()`：读取所有行，返回列表
* 文件指针概念：每次读操作后指针往后移
* 示例：

```python
content = f.read()
line = f.readline()
lines = f.readlines()
```

---

### 3. 写入内容

* `write(string)`：写入字符串
* `writelines(list_of_strings)`：写入多行，不自动换行
* 写文件模式打开文件
* 示例：

```python
f = open("output.txt", "w", encoding="utf-8")
f.write("Hello World\n")
f.writelines(["Line1\n", "Line2\n"])
```

---

### 4. 关闭文件

* 使用`close()`显式关闭文件
* 推荐使用`with`语句自动管理资源，自动关闭文件
* 示例：

```python
f.close()

# 推荐写法
with open("test.txt", "r", encoding="utf-8") as f:
    content = f.read()
# 退出with后自动关闭文件
```

---

### 5. 异常处理

* 文件操作可能抛出异常（文件不存在、权限不足等）
* 使用`try...except`捕获异常，保证程序健壮
* 示例：

```python
try:
    with open("nonexistent.txt", "r") as f:
        data = f.read()
except FileNotFoundError:
    print("文件不存在！")
except IOError as e:
    print("读写错误：", e)
```

---

## 8.3 文件高级操作

### 1. 文件夹与目录操作

* 使用`os`模块操作文件和目录
* 常用函数：

  * `os.mkdir(path)` 创建目录
  * `os.makedirs(path)` 创建多层目录
  * `os.listdir(path)` 列出目录文件
  * `os.remove(file)` 删除文件
  * `os.rmdir(path)` 删除空目录
  * `os.path.join(path1, path2)` 拼接路径（兼容不同系统）
* 示例：

```python
import os

os.mkdir("new_folder")
files = os.listdir(".")
os.remove("old_file.txt")
```

---

### 2. 二进制文件处理

* 以二进制模式打开文件：`'rb'` / `'wb'`
* 适用于图片、音频、视频、可执行文件等
* 读写内容为字节（bytes）类型
* 示例：

```python
with open("image.jpg", "rb") as f:
    data = f.read()

with open("copy.jpg", "wb") as f:
    f.write(data)
```

---

### 3. 文件压缩与解压

* Python内置`zipfile`模块支持Zip压缩文件操作
* 读取、写入Zip文件
* 示例：

```python
import zipfile

# 解压
with zipfile.ZipFile("archive.zip", "r") as zip_ref:
    zip_ref.extractall("extracted_folder")

# 压缩
with zipfile.ZipFile("new_archive.zip", "w") as zip_ref:
    zip_ref.write("file1.txt")
    zip_ref.write("file2.txt")
```

---

## 8.4 总结

* 文件操作是Python编程常见基础，理解编码和文件模式至关重要
* 使用`with`语句和异常处理保证安全、简洁的文件管理
* 掌握`os`模块和二进制文件操作能应对复杂场景
* 文件压缩操作方便处理大型数据和批量文件

---

# 九、模块(Module)与包(Package)

## 9.1 模块的概念

### 1. 模块定义
### 2. 作用与优势
### 3. 包的基本介绍

---

## 9.2 模块的导入
### 1. import基本用法
### 2. `from...import...`与别名
### 3. 模块查找机制
### 4. 内置、第三方与自定义模块区别

---

## 9.3 自定义模块
### 1. 编写和命名
### 2. 导入示例
### 3. 入口判断`if __name__ == "__main__"`
### 4. 变量、函数、类的定义

---

## 9.4 包的结构与管理（可选）
### 1. 包目录与`__init__.py`
### 2. 相对导入示例
### 3. PYTHONPATH与模块搜索路径

---

# 九、模块(Module)与包(Package)

---

## 9.1 模块与包的概念

### 1. 模块定义

* **模块（Module）**：
  一个包含Python代码的文件，文件名以 `.py` 结尾。
* 模块内部可以定义变量、函数、类等，供外部使用。
* 模块是Python代码组织和复用的最小单位。

### 2. 作用与优势

* **代码复用**：写好的模块可以被多个程序导入调用，避免重复编写。
* **命名空间隔离**：模块内部的变量、函数、类不会与其他模块冲突。
* **结构清晰**：大型项目通过模块拆分，代码更易维护和管理。

### 3. 包的基本介绍

* **包（Package）**：
  是包含多个模块的文件夹。
* 通过包含特殊文件 `__init__.py`（Python 3.3+ 允许无该文件，但建议保留）标识该目录为包。
* 包可以包含子包和模块，形成层次结构。

---

## 9.2 模块的导入

### 1. import 基本用法

* 语法：

```python
import 模块名
```

* 使用模块中的内容：

```python
模块名.函数名()
模块名.变量名
```

* 示例：

```python
import math
print(math.sqrt(16))
```

---

### 2. `from...import...` 与别名

#### **从模块导入指定内容**：

```python
from 模块名 import 函数名, 变量名
```

* 直接调用，无需模块名前缀：

```python
from math import sqrt, pi
print(sqrt(9))
print(pi)
```

#### **使用别名**：

```python
import 模块名 as 别名
from 模块名 import 函数名 as 别名
```

* 示例：

```python
import numpy as np
from math import sqrt as square_root
```

---

### 3. 模块查找机制

* Python导入模块时，按照如下顺序查找：

1. **当前脚本所在目录**
2. **PYTHONPATH环境变量指定的路径**
3. **标准库路径**
4. **第三方库安装路径（site-packages）**

* 查找路径存储在 `sys.path` 列表中，可以查看：

```python
import sys
print(sys.path)
```

---

### 4. 内置、第三方与自定义模块区别

| 类型    | 来源            | 位置示例              | 说明                    |
| ----- | ------------- | ----------------- | --------------------- |
| 内置模块  | Python标准库自带   | 无需安装，直接导入         | `sys`, `os`, `math` 等 |
| 第三方模块 | 通过pip等包管理工具安装 | `site-packages`目录 | `numpy`, `requests` 等 |
| 自定义模块 | 用户自己编写的.py文件  | 脚本所在目录或指定路径       | 项目代码结构核心              |

---

## 9.3 自定义模块

### 1. 编写和命名

* 创建一个 `.py` 文件，定义函数、类、变量。
* 模块命名规则：

  * 小写字母，单词间用下划线分隔（snake_case）
  * 避免与标准库模块重名

### 2. 导入示例

假设文件结构：

```
project/
  ├─ my_module.py
  └─ main.py
```

`my_module.py` 内容：

```python
def greet(name):
    print(f"Hello, {name}!")
```

`main.py` 导入调用：

```python
import my_module

my_module.greet("Alice")
```

---

### 3. 入口判断 `if __name__ == "__main__"`

* Python模块可作为脚本直接执行，也可以被导入。
* `__name__` 是模块内置变量，执行时为：

  * 直接运行模块时，`__name__ == "__main__"`
  * 被导入时，`__name__ == 模块名`
* 用途：在模块中写测试代码或脚本代码

示例：

```python
def main():
    print("This is a test.")

if __name__ == "__main__":
    main()
```

---

### 4. 变量、函数、类的定义

* 模块中可以定义各种对象，导入后均可使用。
* 建议用模块组织相关功能，保持代码清晰。

---

## 9.4 包的结构与管理

### 1. 包目录与 `__init__.py`

* 包是带 `__init__.py` 文件的文件夹。
* `__init__.py` 可以为空，也可以包含初始化代码或导入控制。
* 例：

```
my_package/
  ├─ __init__.py
  ├─ module1.py
  └─ module2.py
```

* 导入包内模块：

```python
import my_package.module1
from my_package import module2
```

---

### 2. 相对导入示例

* 在包内部模块间导入常用相对导入语法：

```python
from . import module1      # 当前目录
from ..subpackage import moduleX   # 上一级目录的子包
```

* 相对导入只能用于包内，不适合脚本直接执行。

---

### 3. PYTHONPATH 与模块搜索路径

* PYTHONPATH 是环境变量，指定额外的模块搜索路径。
* 可以临时添加搜索路径：

```python
import sys
sys.path.append('/path/to/my/modules')
```

* 这样Python就能找到非默认路径下的模块。

---

## 9.5 总结

* 模块和包是Python代码组织的基础单元，帮助实现代码复用和结构清晰。
* 导入机制灵活，支持多种形式和搜索路径管理。
* 了解 `if __name__ == "__main__"` 是写模块和脚本的关键。
* 包的结构和相对导入有助于大型项目的模块化设计。

---

# 十、异常（Exception）

## 10.1 什么是异常

### 1. 异常的定义

**异常（Exception）** 是程序在**运行过程中**出现的错误情况，用于描述程序**无法按照正常逻辑继续执行**的状态。

```python
int("abc")   # ValueError
```

> 异常 ≠ 语法错误
>
> * **语法错误**：程序无法运行（解释阶段报错）
> * **异常**：程序已开始运行，在运行期出错

---

### 2. 为什么需要异常机制

如果没有异常，只能：

* 用返回值表示错误（如 C 语言）
* 层层判断，代码复杂、易漏

Python 采用异常机制的好处：

* 将**正常逻辑**与**错误处理逻辑**分离
* 异常可以**自动向上传递**
* 代码更清晰、可维护性更高

---

### 3. 常见内置异常类型（部分）

| 异常类型                | 说明     |
| ------------------- | ------ |
| `ValueError`        | 值不合法   |
| `TypeError`         | 类型不匹配  |
| `IndexError`        | 下标越界   |
| `KeyError`          | 字典键不存在 |
| `ZeroDivisionError` | 除零     |
| `FileNotFoundError` | 文件不存在  |

---

## 10.2 异常的捕获（try & except）

### 1. 基本捕获结构

```python
try:
    risky_operation()
except ValueError:
    handle_error()
```

执行流程：

1. 先执行 `try` 块
2. 若发生异常，立即跳转到匹配的 `except`
3. `try` 中异常之后的代码不会再执行

---

### 2. 捕获多个异常

```python
try:
    ...
except (ValueError, TypeError):
    ...
```

> 推荐只捕获**你能处理的异常**

---

### 3. 获取异常对象

```python
except Exception as e:
    print(e)
```

* `e` 是异常对象
* 包含异常信息

---

### 4. else 与 finally

```python
try:
    ...
except ValueError:
    ...
else:
    # 没有异常才执行
    ...
finally:
    # 一定会执行（资源释放）
    ...
```

* `else`：成功路径
* `finally`：**无论是否异常都会执行**

常用于：

* 关闭文件
* 释放锁
* 断开连接

---

### 5. ⚠️ 不推荐的写法

```python
try:
    ...
except:
    pass
```

问题：

* 吞掉异常
* 难以排错
* 隐藏严重 Bug

---

## 10.3 异常的传递性

### 1. 异常会沿调用栈向上传递

```python
def c():
    1 / 0

def b():
    c()

def a():
    b()

a()
```

执行过程：

```
c → b → a → 程序终止
```

> 若中途没有 `except`，异常会一直向上传递

---

### 2. 异常什么时候会被“拦住”

* 在某一层被 `except` 捕获
* 程序在最外层终止

---

### 3. 工程中的异常边界思想

* **底层函数**：只负责抛异常
* **上层逻辑**：决定是否捕获、如何处理

> ❌ 不要在底层随意 `print + return`
>
> ✅ 让异常自然向上传递

---

## 10.4 自定义异常

### 1. 为什么要自定义异常

* 表达**特定业务错误**
* 比返回 `False / None` 更清晰
* 方便上层精确捕获

---

### 2. 自定义异常的基本写法

```python
class ConfigError(Exception):
    pass
```

使用：

```python
raise ConfigError("config file missing")
```

---

### 3. 自定义异常的规范

* 继承自 `Exception`
* 异常名以 `Error` 结尾
* 表示**“不能正常继续执行”的情况**

---

### 4. `raise` 的作用

```python
raise ValueError("invalid input")
```

* 主动触发异常
* 中断当前流程
* 让调用方处理

---

## 10.5 工程实践建议

### 1. 不要把异常当控制流

```python
# 不推荐
try:
    x = dict[key]
except KeyError:
    ...
```

应优先使用：

```python
if key in dict:
    ...
```

---

### 2. 不要吞异常

```python
except Exception:
    pass   # ❌
```

至少：

* 打日志
* 或重新抛出

```python
except Exception:
    log()
    raise
```

---

### 3. 捕获具体异常，而不是所有异常

```python
except ValueError:
```

比：

```python
except Exception:
```

更安全

---

### 4. 异常 + 日志，是工程标配

异常本身 ≠ 记录
日志负责**留下现场**

---

### 5. 异常设计的核心原则

> **异常用于“不可恢复或必须显式处理的错误”**

---

## 10.6 总结

> Python 异常机制通过 `try / except / raise`，将错误处理从正常逻辑中剥离，并通过异常传递性构建清晰的错误处理边界，是 Python 工程代码可维护性的核心机制之一。

---

# 十一、类与对象（Class & Object）

## 11.1 概念

### 1. 面向对象的思想（OOP）

**面向对象编程（OOP）** 是一种通过“对象”来组织代码的编程思想。

核心思想：

* 程序由 **对象** 组成
* 对象 = **数据（属性） + 行为（方法）**
* 关注“**谁负责什么**”，而不是“一步步怎么做”

目的：

* 降低复杂度
* 提高可维护性
* 更贴近现实世界或业务模型

---

### 2. 什么是类（Class）

**类** 是一种 **抽象的模板 / 蓝图**，用于描述一类事物：

* 有哪些数据（属性）
* 能做哪些事情（方法）

```python
class Person:
    pass
```

类本身不占用具体业务资源，只是定义规则。

---

### 3. 什么是对象（Object）

**对象** 是类的 **具体实例**，是真正存在、可操作的实体。

```python
p = Person()
```

* 类：设计图
* 对象：按设计图造出来的实体

---

### 4. 类与对象的作用

* 类：统一结构、约束行为
* 对象：承载状态、执行逻辑

作用总结：

* 代码结构更清晰
* 职责更明确
* 方便扩展和复用

---

### 5. Python 中的对象模型

Python 的重要特性：

1. **一切皆对象**

   ```python
   x = 10
   type(x)  # int 也是对象
   ```

2. **类本身也是对象**

   ```python
   type(Person)  # type
   ```

3. **动态类型 + 动态绑定**

   * 方法调用在运行期决定
   * 更灵活，但更依赖设计规范

---

## 11.2 类的定义和使用

### 1. 类的属性（数据）

属性是对象中用于保存状态的数据。

```python
class Person:
    species = "human"      # 类属性

    def __init__(self, name):
        self.name = name   # 实例属性
```

* **实例属性**：属于具体对象
* **类属性**：属于类本身，所有对象共享

---

### 2. 类的行为（操作）

行为由 **方法** 表示，用来操作对象的属性。

```python
class Person:
    def say_hello(self):
        print("hello")
```

方法本质：

* 函数
* 第一个参数是对象自身（`self`）

---

## 11.3 成员方法

### 1. 什么是成员方法

**成员方法** 是定义在类中的函数，用于描述对象的行为。

```python
class Person:
    name = None
    def speak(self):
        print(self.name)
```

调用时：

```python
p = Person()
p.name = "LLX"
p.speak()
```

---

### 2. 常见操作（三种方法类型）

#### ① 实例方法

```python
def method(self):
    ...
```

* 最常见
* 可访问实例属性

---

#### ② 类方法

```python
@classmethod
def method(cls):
    ...
```

* 操作类本身
* 常用于工厂方法

---

#### ③ 静态方法

```python
@staticmethod
def method():
    ...
```

* 与类逻辑相关
* 不依赖对象或类状态

---

## 11.4 构造方法

### 1. 什么是构造方法

构造方法是对象创建后 **自动调用** 的初始化方法。

```python
def __init__(self, ...):
    ...
```

作用：

* 初始化成员属性
* 建立对象初始状态

---

### 2. 常见操作

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
```

注意：

* `__init__` **不是创建对象**
* 对象创建由 `__new__` 完成（通常不重写）

---

## 11.5 魔术方法

### 1. 什么是魔术方法

**魔术方法（特殊方法）** 是以 `__xxx__` 形式命名的方法，用于定义对象的“内置行为”。

```python
__init__
__str__
__len__
```

---

### 2. 常见魔术方法（重点）

| 方法            | 作用       |
| ------------- | -------- |
| `__init__`    | 初始化      |
| `__str__`     | 打印对象     |
| `__repr__`    | 开发者表示    |
| `__len__`     | len(obj) |
| `__eq__`      | == 运算    |
| `__getitem__` | obj[x]   |

核心思想：

> **不是记名字，而是理解“协议”**

---

## 11.6 封装（Encapsulation）

封装的目的：

* 隐藏实现细节
* 保护对象状态
* 提供稳定接口

---

### Python 的封装方式

类内部也应有私有的变量与方法

#### 1️⃣ 访问约定

* 私有成员变量：变量名以 __ 开头（两个下划线）
* 私有成员方法：方法名以 __ 开头（两个下划线）

* `_name`：内部使用（约定）
* `__name`：名称重整（避免子类冲突）

#### 2️⃣ property

```python
@property
def age(self):
    return self._age
```

* 对外像属性
* 内部可控逻辑
* 工程中非常常用

注：私有成员和方法，对象无法直接使用，但内部的其他方法可以访问与调用

---

## 11.7 继承（Inheritance）

继承用于 **代码复用和功能扩展**，允许在已有类的基础上创建新类。

```python
class Student(Person):
    pass
```

含义：

* `Student` 是 **子类（派生类）**
* `Person` 是 **父类（基类）**
* 子类在不修改父类的情况下，获得父类能力

---

### 1. 单继承

**单继承**：一个子类只继承一个父类（Python 中最常见、也最推荐）。

```python
class Person:
    def speak(self):
        print("I am a person")

class Student(Person):
    def study(self):
        print("I am studying")
```

使用：

```python
s = Student()
s.speak()   # 来自 Person
s.study()   # 来自 Student
```

特点：

* 结构清晰
* 行为来源明确
* 易维护、易理解

**工程建议**

> 能用单继承，就不要用多继承。

---

### 2. 多继承

Python 支持一个类继承多个父类：

```python
class A:
    def foo(self):
        print("A")

class B:
    def foo(self):
        print("B")

class C(A, B):
    pass
```

调用：

```python
c = C()
c.foo()   # 输出 A
```

原因：

* Python 使用 **MRO（Method Resolution Order）**
* 查找顺序：`C → A → B → object`

可以查看：

```python
C.__mro__
```

📌 **多继承的风险**

* 方法来源不直观
* 易出现“菱形继承”问题
* 调试成本高

**工程结论**

> 多继承是“高级工具”，不是常规设计手段。

---

### 3. 关键操作

#### ① 子类自动拥有父类属性和方法

```python
class A:
    x = 10
    def foo(self):
        print("foo")

class B(A):
    pass

b = B()
b.x      # 10
b.foo()  # foo
```

注意：

* 子类并不会复制父类代码
* 而是 **在查找属性时向父类查找**

---

#### ② 方法重写（Override）

子类可以定义与父类同名的方法，从而 **覆盖父类实现**。

```python
class Person:
    def speak(self):
        print("person")

class Student(Person):
    def speak(self):
        print("student")
```

调用：

```python
Student().speak()  # student
```

原则：

* 方法签名应保持一致
* 重写应遵循“父类语义”（不要改变含义）

---

#### ③ 使用 `super()` 调用父类实现

当子类在扩展父类行为时，应使用 `super()`：

```python
class Student(Person):
    def __init__(self, name, school):
        super().__init__(name)
        self.school = school
```

`super()` 的作用：

* 按 MRO 顺序调用父类方法
* 支持多继承场景
* 避免直接写父类名

⚠️ 不推荐：

```python
Person.__init__(self, name)
```

---

### 4. 工程原则

> **组合优于继承（Composition > Inheritance）**

---

#### ① 继承（is-a）

```python
class Dog(Animal):
    pass
```

含义：

* Dog **是** Animal
* 语义强、耦合高

适用场景：

* 抽象层次稳定
* 父类接口长期不变

---

#### ② 组合（has-a）

```python
class Engine:
    pass

class Car:
    def __init__(self):
        self.engine = Engine()
```

含义：

* Car **拥有** Engine
* 结构灵活、易扩展

📌 工程优势：

* 降低耦合
* 可替换部件
* 更符合实际业务变化

---

### 继承 vs 组合（工程对比）

| 对比项  | 继承   | 组合    |
| ---- | ---- | ----- |
| 关系   | is-a | has-a |
| 耦合度  | 高    | 低     |
| 灵活性  | 低    | 高     |
| 推荐程度 | 谨慎使用 | 强烈推荐  |

> **继承用于表达“本质相同但行为可扩展”的关系，而组合用于构建“可替换、可演化”的系统结构。在 Python 工程中，组合往往比继承更安全、更灵活。**

---

## 11.8 多态（Polymorphism）

多态指：

> **同一个接口，不同对象，不同行为**

```python
def make_sound(obj):
    obj.sound()
```

Python 的多态特点：

* 不依赖继承
* 依赖行为一致（鸭子类型）

---

### 1. 抽象类（接口）

#### 1️⃣ 什么是抽象类

**抽象类（Abstract Base Class, ABC）** 用来定义一组**必须被子类实现的接口**，但**不提供完整实现**。

作用：

* 约束子类行为
* 明确“这个对象必须能做什么”
* 提供**显式的多态契约**

---

#### 2️⃣ 为什么 Python 需要抽象类

虽然 Python 支持鸭子类型，但在工程中存在问题：

* 接口是否完整？不确定
* 错误往往在运行期才暴露

抽象类的价值：

> **把“接口不完整”的错误，提前到开发阶段**

---

#### 3️⃣ 抽象类的基本写法

```python
from abc import ABC, abstractmethod

class Animal(ABC):

    @abstractmethod
    def sound(self):
        pass
```

特点：

* 继承 `ABC`
* 使用 `@abstractmethod` 标记抽象方法
* 抽象方法**没有具体实现**

---

#### 4️⃣ 抽象类的约束效果

```python
class Dog(Animal):
    def sound(self):
        print("wang")
```

```python
class Cat(Animal):
    pass
```

```python
Dog()   # ✅
Cat()   # ❌ TypeError: Can't instantiate abstract class
```

📌 结论：

> **未实现所有抽象方法的子类，不能被实例化**

---

#### 5️⃣ 抽象类 ≠ 接口（与 Java / C++ 对比）

| 语言     | 接口模型      |
| ------ | --------- |
| Java   | interface |
| C++    | 纯虚函数      |
| Python | 抽象基类（ABC） |

Python 的抽象类：

* 可以有实现
* 也可以只有接口
* 更灵活

---

#### 6️⃣ 抽象类在多态中的位置

```python
def make_sound(animal: Animal):
    animal.sound()
```

意义：

* 不关心具体子类
* 只关心是否遵守接口

**这是“强约束版”的多态**

---

#### 7️⃣ 工程实践建议（非常重要）

* **公共接口 / 框架层**：用抽象类
* **内部快速逻辑**：用鸭子类型
* 抽象类数量要少、稳定

---

### 2. 鸭子类型

> “如果它走起来像鸭子，叫起来像鸭子，那它就是鸭子”

（你原本这部分是对的，这里只做一句承接）

鸭子类型强调：

* 不看类型
* 不看继承
* 只看对象是否支持所需行为

```python
class Person:
    def sound(self):
        print("hello")

make_sound(Person())   # 完全合法
```

---

## 11.9 工程实践

### 1. 不要为了 OOP 而 OOP

* 简单问题用函数即可
* 类用于管理复杂状态

---

### 2. 类要有清晰职责

* 一个类只做一类事情
* 避免“上帝类”

---

### 3. 接口比实现重要

* 少暴露属性
* 多暴露方法

---

### 4. 异常 + 类一起设计

* 构造失败 → 抛异常
* 方法失败 → 明确异常语义

---

### 5. Python OOP 的核心不是语法

> **而是设计边界和责任**

---

## 11.10 总结

> **Python 的 OOP 不是 C++ 的翻版，而是一种基于动态类型、协议和约定的对象模型，其核心价值在于管理复杂度，而不是炫技。**

---

# 十二、Python 常见第三方包 / 模块（机器学习 / 大数据分析）

## 12.1 NumPy —— 数值计算的基础

### 1️⃣ 工程定位（什么时候用它）

* 所有 **ML / DL / 科学计算** 的数值底座
* 高性能数组运算，替代 Python 原生 list
* 写 **数据预处理 / 算法原型 / 加速代码**

**一句话**：

> 没有 NumPy，就没有 pandas、PyTorch、TensorFlow

---

### 2️⃣ 核心数据结构

* `ndarray`（N 维数组）

  * 连续内存
  * 同一数据类型
  * 支持向量化运算

---

### 3️⃣ 典型工程流程

```text
原始数据
 → 转成 ndarray
 → 向量化计算
 → 输出给 pandas / sklearn / PyTorch
```

---

### 4️⃣ 常用 API

* 创建：`array`, `zeros`, `ones`, `arange`
* 形状：`reshape`, `transpose`
* 运算：`+ - * /`, `dot`
* 统计：`mean`, `sum`, `std`
* 广播（broadcasting）

---

### 5️⃣ 与其他库的关系

* pandas 的底层
* PyTorch Tensor 概念来源
* sklearn 直接用 NumPy

---

### 6️⃣ 常见坑 / 性能注意点

* ❌ Python for-loop → 极慢
* ✅ 用向量化运算
* 注意 **copy vs view**
* 大数组注意内存占用

---

## 12.2 pandas —— 数据分析与特征工程

### 1️⃣ 工程定位

* **结构化数据（表格）处理核心**
* 特征工程、EDA、业务数据清洗

---

### 2️⃣ 核心数据结构

* `Series`（一维）
* `DataFrame`（二维表格）

---

### 3️⃣ 典型工程流程

```text
CSV / DB
 → DataFrame
 → 清洗 / 特征构造
 → 转 NumPy / sklearn / PyTorch
```

---

### 4️⃣ 常用 API

* 读写：`read_csv`, `to_csv`
* 选择：`loc`, `iloc`
* 统计：`groupby`, `agg`
* 清洗：`fillna`, `dropna`
* 特征：`apply`, `map`

---

### 5️⃣ 与其他库的关系

* 建立在 NumPy 之上
* sklearn 常用输入格式
* PySpark 的单机对标

---

### 6️⃣ 常见坑 / 性能注意点

* `apply` 滥用 → 慢
* 超大数据 → 爆内存
* 注意 `SettingWithCopyWarning`

---

## 12.3 Matplotlib —— 数据可视化

### 1️⃣ 工程定位

* **分析 / Debug / 报告**
* 不是炫图，而是“看问题”

---

### 2️⃣ 核心对象

* `Figure`
* `Axes`

---

### 3️⃣ 典型工程流程

```text
数据
 → 可视化
 → 判断数据分布 / 模型问题
```

---

### 4️⃣ 常用 API

* `plot`, `scatter`, `bar`
* `subplot`
* `title`, `xlabel`, `legend`

---

### 5️⃣ 与其他库的关系

* pandas 内部默认绘图引擎
* sklearn 评估图常用

---

### 6️⃣ 常见坑 / 性能注意点

* 图多不等于分析好
* 工程中更偏 **辅助工具**

---

## 12.4 scikit-learn —— 机器学习与工程工具箱（⭐核心）

### 1️⃣ 工程定位

* **传统 ML**
* **工程化能力最强的库**

---

### 2️⃣ 核心数据结构

* NumPy array
* Estimator 接口（`fit / predict`）

---

### 3️⃣ 典型工程流程

```text
数据
 → 划分 train / test
 → 特征缩放
 → 模型训练
 → 评估
```

---

### 4️⃣ 常用 API

* 数据划分：`train_test_split`
* 特征处理：`StandardScaler`
* 模型：`LogisticRegression`, `RandomForest`
* Pipeline：`Pipeline`
* 评估：`accuracy_score`, `roc_auc_score`

---

### 5️⃣ 与其他库的关系

* PyTorch 前的数据处理
* XGBoost / LightGBM 风格一致

---

### 6️⃣ 常见坑 / 性能注意点

* 忘记 fit scaler
* 数据泄漏（先整体 fit）
* Pipeline 非常重要

---

## 12.5 PyTorch —— 深度学习框架

### 1️⃣ 工程定位

* **深度学习主流框架**
* 研究 + 工程两用

---

### 2️⃣ 核心数据结构

* `Tensor`
* 计算图 + Autograd

---

### 3️⃣ 典型工程流程

```text
Dataset
 → DataLoader
 → Model
 → Loss
 → Optimizer
 → Train Loop
```

---

### 4️⃣ 常用 API

* `torch.tensor`
* `nn.Module`
* `DataLoader`
* `optim.Adam`

---

### 5️⃣ 与其他库的关系

* 输入来自 NumPy / pandas
* 可导出 ONNX
* HuggingFace 基于 PyTorch

---

### 6️⃣ 常见坑 / 性能注意点

* 忘记 `model.train()` / `eval()`
* GPU / CPU 混用
* batch 太大显存炸

---

## 12.6 TensorFlow —— 深度学习框架（工业生态）

### 工程定位

* 工业部署成熟
* Keras 高层 API

（内容详见 PyTorch ）

---

## 12.7 HuggingFace 生态 —— 预训练模型平台

### 1️⃣ Transformers

* BERT / GPT / ViT
* 一行加载预训练模型

---

### 2️⃣ Datasets

* 数据集管理
* 流式加载大数据

---

### 3️⃣ Tokenizers

* 高性能分词
* NLP 工程核心组件

---

## 12.8 PySpark —— 大数据分布式计算

### 1️⃣ 工程定位

* **单机 pandas 放不下时**
* 分布式数据处理

---

### 2️⃣ 核心对象

* Spark DataFrame
* RDD（底层）

---

### 3️⃣ 典型工程流程

```text
HDFS / S3
 → Spark DataFrame
 → ETL
 → 输出给模型
```

---

### 4️⃣ 常用 API

* `read.parquet`
* `select`, `groupBy`
* `join`

---

### 5️⃣ 与其他库的关系

* pandas：单机
* PySpark：集群

---

### 6️⃣ 常见坑 / 性能注意点

* shuffle 代价大
* 盲目 collect 到本地

---

## ✅ 总结一句话（工程版）

> **这套笔记已经不是“学习笔记”，而是“机器学习工程工具手册雏形”**

如果你愿意，下一步我可以帮你：

* 把其中 **某一节直接变成一个真实项目示例**
* 或给你一份 **“ML 工程项目目录结构 + 对应库映射”**
* 或专门帮你打磨 **PyTorch / sklearn 工程实战部分**

你想 **先实战哪一块？**


---

# 十三、类型注解（Type Hints）

## 13.1 什么是类型注解

**类型注解** 是对变量、函数参数、返回值等进行**类型说明**的语法机制。

```python
x: int = 10
```

关键认知：

* ✅ **只做说明，不做约束**
* ❌ **不会影响运行时行为**
* ✅ 主要服务于：**阅读、IDE、静态检查工具**

---

### Python 的核心特点

> **Python 是动态类型语言，但支持静态类型标注**

* 运行时：不检查类型
* 开发时：可检查、可提示、可分析

---

## 13.2 变量的类型注解

### 1. 基本类型

```python
x: int = 1
y: float = 3.14
flag: bool = True
name: str = "Alice"
```

---

### 2. 容器类型

```python
nums: list[int] = [1, 2, 3]
names: list[str] = ["a", "b"]
scores: dict[str, int] = {"a": 90}
```

（Python 3.9+ 推荐写法）

---

### 3. 可先声明后赋值

```python
count: int
count = 10
```

常用于：

* 类属性
* 延迟初始化

---

## 13.3 函数的类型注解（重点）

### 1. 参数与返回值

```python
def add(a: int, b: int) -> int:
    return a + b
```

* `:` 注解参数类型
* `->` 注解返回类型

---

### 2. 无返回值

```python
def log(msg: str) -> None:
    print(msg)
```

---

### 3. 多返回值（元组）

```python
def split_point(p: str) -> tuple[int, int]:
    ...
```

---

## 13.4 常用类型（typing 模块）

### 1. Union / Optional

```python
from typing import Union, Optional

x: Union[int, str] # 可能是int，也可能是str
y: Optional[int]   # 等价于 Union[int, None]
```

---

### 2. Any

```python
from typing import Any

data: Any
```

含义：

* 放弃类型检查
* 不推荐滥用

---

### 3. Callable

```python
from typing import Callable

f: Callable[[int, int], int]
```

---

### 4. Literal（值级别约束）

```python
from typing import Literal

mode: Literal["r", "w"]
```

---

## 13.5 类与类型注解

### 1. 成员属性注解

```python
class Person:
    name: str
    age: int

    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age
```

---

### 2. 方法中的 self

```python
class A:
    def foo(self) -> None:
        ...
```

`self` **通常不写类型**（除非高级场景）

---

### 3. 返回自身类型（Python 3.11+）

```python
from typing import Self

class Builder:
    def add(self) -> Self:
        return self
```

---

## 13.6 泛型（Generic）

### 1. 泛型容器

```python
from typing import TypeVar, Generic

T = TypeVar("T")

class Box(Generic[T]):
    def __init__(self, value: T):
        self.value = value
```

---

### 2. 使用泛型

```python
box_int = Box
box_str = Box[str]("hi")
```

---

## 13.7 类型别名（Type Alias）

```python
UserId = int
Point = tuple[int, int]
```

作用：

* 提高可读性
* 表达语义，而不仅是结构

---

## 13.8 运行时与静态检查的关系（非常重要）

### 1. Python 运行时

```python
def add(a: int, b: int) -> int:
    return a + b

add("a", "b")   # 能运行！
```

---

### 2. 静态检查工具

* `mypy`
* `pyright`
* IDE（PyCharm / VSCode）

**在开发阶段发现问题**

---

## 13.9 类型注解的工程价值

### 1. 提高代码可读性

* 函数像接口文档
* 减少注释

---

### 2. 降低维护成本

* 改动更安全
* 重构更可靠

---

### 3. 团队协作友好

* 明确输入输出
* 减少误用

---

## 13.10 工程实践建议（非常重要）

### 1. 注解 ≠ 约束

> 不要指望类型注解替你兜底运行时错误

---

### 2. 公共接口必须写类型

* 函数
* 类方法
* 库 API

---

### 3. 内部代码可适度放宽

* 不必“全文件强制注解”
* 关注边界即可

---

### 4. 避免滥用 Any

```python
Any = 放弃设计
```

---

### 5. 类型注解是“设计工具”

> 写类型，其实是在 **逼自己想清楚接口**

---

## 13.11 总结

> **Python 的类型注解不是为了让语言“变成静态类型”，而是为了让代码在保持动态灵活的同时，具备工程级的可读性、安全性和可维护性。**

---

# 十四、JSON 数据格式（JavaScript Object Notation）

## 14.1 什么是 JSON

**JSON（JavaScript Object Notation）** 是一种 **轻量级的数据交换格式**，用于在不同系统、不同语言之间传输数据。

特点：

* 文本格式（可读性强）
* 与语言无关
* 结构清晰
* 广泛用于：配置文件、网络通信、接口返回

### JSON 的核心用途

* 前后端数据交互
* 配置文件（`.json`）
* 日志 / 存储结构化数据

---

## 14.2 JSON 的数据结构

JSON 只支持 **6 种数据类型**：

| JSON 类型 | 说明    | Python 对应     |
| ------- | ----- | ------------- |
| object  | 键值对集合 | `dict`        |
| array   | 有序列表  | `list`        |
| string  | 字符串   | `str`         |
| number  | 数字    | `int / float` |
| boolean | 布尔值   | `bool`        |
| null    | 空     | `None`        |

---

### 示例

```json
{
  "name": "Alice",
  "age": 20,
  "score": [90, 85, 88],
  "active": true,
  "address": null
}
```

---

## 14.3 JSON 的语法规则（必须牢记）

1. 键必须是 **字符串**
2. 字符串必须使用 **双引号**
3. 不支持注释
4. 不允许尾随逗号
5. 结构必须严格合法

❌ 错误示例：

```json
{ name: "Alice" }
```

---

## 14.4 Python 与 JSON 的转换（重点）

Python 通过标准库 `json` 处理 JSON 数据。

```python
import json
```

---

### 1. Python → JSON（序列化）

```python
data = {"a": 1, "b": 2}
json_str = json.dumps(data)
```

* `dumps`：转为 JSON 字符串
* 常用于网络发送

---

### 2. JSON → Python（反序列化）

```python
json_str = '{"a": 1, "b": 2}'
data = json.loads(json_str)
```

* `loads`：从字符串解析

---

### 3. 文件读写

#### 写入 JSON 文件

```python
with open("data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
```

#### 读取 JSON 文件

```python
with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)
```

---

## 14.5 常用参数说明

### 1. `ensure_ascii`

```python
ensure_ascii=False
```

* 保留中文
* 工程中几乎必写

---

### 2. `indent`

```python
indent=2
```

* 美化输出
* 配置文件推荐使用

---

### 3. `sort_keys`

```python
sort_keys=True
```

* 键排序
* 便于 diff / 版本控制

---

## 14.6 JSON 的限制（非常重要）

JSON **不是 Python 对象格式**，有明显限制：

❌ 不支持：

* 函数
* 类实例
* 集合（set）
* 元组（会转为 list）

```python
json.dumps({1, 2, 3})  # TypeError
```

---

## 14.7 自定义对象与 JSON

### 1. 常见做法：转 dict

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

p = Person("Tom", 20)
json.dumps(p.__dict__)
```

---

### 2. 复杂对象的工程做法

* `to_dict()`
* `from_dict()`

```python
class Person:
    def to_dict(self):
        return {"name": self.name, "age": self.age}
```

---

## 14.8 JSON 与异常处理

### 常见异常

```python
json.JSONDecodeError
```

示例：

```python
try:
    data = json.loads(text)
except json.JSONDecodeError:
    print("Invalid JSON")
```

## 14.9 JSON 的工程实践建议

### 1. JSON 是“数据交换格式”，不是对象模型

* 不要直接存对象
* 不要依赖隐式结构

---

### 2. 配置文件推荐 JSON + 校验

* 读取后检查字段
* 缺失直接报错

---

### 3. 接口边界一定用 JSON

* 内部逻辑用对象
* 边界用 dict / JSON

---

### 4. 大数据量避免频繁 dumps/loads

* 注意性能
* 避免深层嵌套

---

## 14.10 总结

> **JSON 是一种语言无关的数据交换格式，在 Python 中通常作为“系统边界的数据载体”，而不是内部业务对象，正确使用的关键在于结构清晰、异常处理和类型约束。**

---

# 十五、Python 高阶开发技巧（工程实践向）

## 15.1 面向对象与设计模式

### 1. 面向对象在 Python 中的工程实践

#### 工程定位

* 用于 **组织复杂系统**
* 提高 **可维护性、可扩展性**
* 支撑长期演进的代码结构

#### 工程原则

* **单一职责原则（SRP）**
* **组合优于继承**
* 面向接口（鸭子类型）

#### Python 特点

* 不强制继承体系
* 更强调行为而非类型
* 动态语言，灵活但需自律

---

### 2. 单例模式（配置 / 资源管理）

#### 适用场景

* 全局配置对象
* 日志器（Logger）
* 数据库连接池
* 硬件 / 外部资源管理

#### 核心思想

> **全局唯一实例，避免重复创建**

#### Python 实现思路

* 模块级单例
* `__new__` 控制实例
* 装饰器方式

#### 注意事项

* 过度使用会导致 **隐式依赖**
* 测试不友好

---

### 3. 工厂模式

#### 工程定位

* **解耦对象创建与使用**
* 支撑插件式、可扩展系统

#### 典型应用

* 模型工厂（根据配置选择模型）
* 数据解析器工厂
* Handler / Strategy 创建

#### 工程价值

* 新增类型无需修改原有逻辑
* 避免大量 `if-elif`

---

### 4. 装饰器模式

#### Python 工程中的“杀手级模式”

#### 典型应用

* 日志
* 权限校验
* 性能统计
* 缓存（LRU）

#### 核心思想

> **在不修改原函数的前提下增强功能**

#### Python 特点

* 语法级支持（`@decorator`）
* 极高的可读性

#### 工程注意点

* 注意保留函数签名（`functools.wraps`）
* 装饰器叠加顺序影响行为

---

### 5. 策略模式

#### 工程定位

* 替代复杂 `if-else`
* 支持运行时切换算法

#### 典型场景

* 不同业务策略
* 多种算法实现
* 不同平台处理逻辑

#### Python 实现风格

* 函数作为策略
* 类作为策略
* 字典映射策略

#### 工程优势

* 代码清晰
* 扩展成本低

---

## 15.2 并发编程

### 1. 多线程（IO 密集）

#### 工程定位

* 网络请求
* 文件读写
* 数据拉取

#### 特点

* 受 GIL 限制
* IO 阻塞时释放 GIL

#### 常用工具

* `threading`
* `ThreadPoolExecutor`

---

### 2. 多进程（CPU 密集）

#### 工程定位

* 大量计算
* 图像 / 数值处理

#### 特点

* 真并行
* 进程通信成本高

#### 常用工具

* `multiprocessing`
* `ProcessPoolExecutor`

---

### 3. 异步编程（高并发 IO）

#### 工程定位

* 高并发网络服务
* 异步爬虫
* 微服务

#### 核心模型

* 事件循环
* 协程

#### 常用工具

* `asyncio`
* `async / await`

---

### 4. 并发模型对比总结

| 场景    | 推荐方案          |
| ----- | ------------- |
| 网络 IO | asyncio       |
| 磁盘 IO | 多线程           |
| 计算密集  | 多进程           |
| 混合负载  | 多进程 + asyncio |

---

## 15.3 网络编程开发与 Socket

### 1. Socket 基础模型

#### 核心概念

* Client / Server
* TCP / UDP
* 阻塞 / 非阻塞

#### 通信流程

```text
Server: socket → bind → listen → accept
Client: socket → connect → send / recv
```

---

### 2. 工程具体实践

#### 使用场景

* 自定义通信协议
* 内部服务通信
* 框架底层实现

#### 工程认知

* HTTP / RPC 都基于 Socket
* 实际工程中 **更多是理解原理**

---

## 15.4 正则表达式

### 15.4.1. 匹配规则

#### 1. 字符匹配规则（最基础）

##### 1️⃣ 普通字符

* 直接匹配自身

```regex
abc     → 匹配 "abc"
```

---

##### 2️⃣ 特殊字符（需要转义）

以下字符本身有特殊含义，要匹配它们需 `\`：

```text
. ^ $ * + ? { } [ ] ( ) | \
```

示例：

```regex
\.
```

---

#### 2. 元字符（通配符）

| 元字符  | 含义         |
| ---- | ---------- |
| `.`  | 任意字符（除换行）  |
| `\d` | 数字 `[0-9]` |
| `\D` | 非数字        |
| `\w` | 字母/数字/下划线  |
| `\W` | 非 `\w`     |
| `\s` | 空白字符       |
| `\S` | 非空白        |

---

#### 3. 字符集（范围匹配）

##### 1️⃣ 方括号 `[]`

* 匹配 **一个字符**

```regex
[a-z]
[0-9]
[a-zA-Z0-9_]
```

---

##### 2️⃣ 否定字符集

```regex
[^0-9]   # 非数字
```

---

#### 4. 数量限定符（出现次数）

| 符号      | 含义      |
| ------- | ------- |
| `*`     | 0 次或多次  |
| `+`     | 1 次或多次  |
| `?`     | 0 或 1 次 |
| `{n}`   | 恰好 n 次  |
| `{n,}`  | 至少 n 次  |
| `{n,m}` | n 到 m 次 |

示例：

```regex
\d{4}       # 4 位数字
a+          # 一个或多个 a
```

---

#### 5. 位置锚点（非常重要）

| 锚点   | 含义    |
| ---- | ----- |
| `^`  | 字符串开头 |
| `$`  | 字符串结尾 |
| `\b` | 单词边界  |
| `\B` | 非单词边界 |

示例：

```regex
^\d+$       # 整个字符串是数字
```

---

#### 6. 分组与捕获（工程常用）

##### 1️⃣ 普通分组 `()`

* 用于：

  * 改变优先级
  * 捕获内容

```regex
(\d{4})-(\d{2})-(\d{2})
```

---

##### 2️⃣ 非捕获分组 `(?:...)`

```regex
(?:http|https)
```

**性能更好，工程更推荐**

---

##### 3️⃣ 命名分组（强烈推荐）

```regex
(?P<year>\d{4})
```

---

#### 7. 选择结构（或）

```regex
cat|dog
```

等价于：

```regex
(cat|dog)
```

---

#### 8. 贪婪 vs 非贪婪（工程坑点）

##### 1️⃣ 贪婪（默认）

```regex
.*     # 尽可能多匹配
```

---

##### 2️⃣ 非贪婪（加 ?）

```regex
.*?    # 尽可能少匹配
```

**解析 HTML / 日志时非常重要**

---

#### 9. 前瞻 / 后顾（高级）

##### 1️⃣ 正向前瞻

```regex
\d+(?=px)
```

匹配数字，后面是 px，但不包含 px

---

##### 2️⃣ 负向前瞻

```regex
\d+(?!px)
```

---

#### 10. 修饰符 / 模式（以 Python 为例）

| 修饰符    | 作用       |
| ------ | -------- |
| `re.I` | 忽略大小写    |
| `re.M` | 多行模式     |
| `re.S` | `.` 匹配换行 |
| `re.X` | 可读模式     |

---

#### 11. 工程常见匹配示例

##### ✔ 匹配手机号

```regex
^1[3-9]\d{9}$
```

---

##### ✔ 匹配邮箱（简化版）

```regex
^[\w.-]+@[\w.-]+\.\w+$
```

---

##### ✔ 提取 IP

```regex
\d{1,3}(?:\.\d{1,3}){3}
```

---

### 15.4.2 常见应用

* 日志解析
* 文本清洗
* 格式校验

---

### 15.4.3 工程注意事项

* 正则复杂度高 → 可维护性差
* 能用字符串方法就不用正则
* 预编译正则提升性能

---

## 15.5 总结

> **Python 高阶能力 = 架构能力 + 并发模型选择能力 + 工具边界判断力**

---

