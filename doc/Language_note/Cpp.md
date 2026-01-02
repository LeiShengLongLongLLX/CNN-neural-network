# Cpp_language

# 一、Cpp基础语法概念

## 1.1 关键字（Keywords）

### 1. 定义

**关键字**是 C++ 语言保留的、具有固定语义的单词，**不能作为标识符使用**。

### 2. 常见关键字分类

#### ▶ 类型相关

```cpp
int    float    double    char    bool    void
```

#### ▶ 流程控制

```cpp
if    else    switch    case    for    while    do
break    continue    goto    return
```

#### ▶ 面向对象

```cpp
class    struct    public    private    protected
virtual    friend    this
```

#### ▶ 内存与修饰

```cpp
const    static    sizeof    typedef
```

#### ▶ C++ 特有（对比 C）

```cpp
new    delete    reference(&)
```

#### **记忆原则**：
看见关键字，说明**语言层面要参与语义解释**，不是普通名字。

---

## 1.2 标识符（Identifier）

### 1. 什么是标识符

**程序员自己定义的名字**，用于表示：

* 变量
* 函数
* 类
* 命名空间

### 2. 命名规则（硬规则）

✔ 只能包含：字母、数字、下划线
✔ 不能以数字开头
✔ 不能是关键字
✔ 区分大小写

```cpp
int value;
int _count;
int value2;

// ❌ 错误
int 2value;
int int;
```

### 3. 命名规范（软规则，工程重要）

* 变量 / 函数：`lower_case` 或 `camelCase`
* 类名：`PascalCase`
* 常量：`ALL_CAPS`

**好命名 = 可读性 + 可维护性**

---

## 1.3 注释（Comment）

### 1. 单行注释

```cpp
// 这是单行注释
```

### 2. 多行注释

```cpp
/*
   这是多行注释
*/
```

### 3. 工程建议

* 注释**解释“为什么”**
* 不要重复“代码已经表达的内容”

❌ 坏注释：

```cpp
i++;  // i 加 1
```

✔ 好注释：

```cpp
i++;  // 跳过已处理的数据
```

---

## 1.4 数据类型（Data Types）

### 1. 基本数据类型

| 类型     | 含义    |
| ------ | ----- |
| int    | 整型    |
| float  | 单精度浮点 |
| double | 双精度浮点 |
| char   | 字符    |
| bool   | 布尔    |
| void   | 无类型   |

```cpp
int a = 10;
double pi = 3.14;
bool flag = true;
```

### 2. sizeof 运算符

```cpp
sizeof(int);     // 查看类型占用字节数
sizeof(a);       // 查看变量大小
```

**大小与平台有关，不要写死假设**

---

## 1.5 输入与输出（I/O）

### 1. 标准输出 `cout`

```cpp
#include <iostream>
using namespace std;

cout << "Hello" << endl;
```

* `<<`：插入运算符
* `endl`：换行 + 刷新缓冲区

### 2. 标准输入 `cin`

```cpp
int x;
cin >> x;
```

`cin` 会根据**变量类型自动解析输入**

---

## 1.6 运算符（Operators）

### 1. 算术运算符

```cpp
+  -  *  /  %
```

### 2. 关系运算符

```cpp
==  !=  >  <  >=  <=
```

### 3. 逻辑运算符

```cpp
&&   ||   !
```

### 4. 赋值运算符

```cpp
=   +=   -=   *=   /=
```

注：**注意 `=` 和 `==` 的区别（常见错误）**

---

## 1.7 流程控制

### 1. 分支语句

#### ▶ if / else

```cpp
if (x > 0) {
    // ...
} else {
    // ...
}
```

#### ▶ switch

```cpp
switch (n) {
case 1:
    break;
default:
    break;
}
```

📌 `switch` 只能用于**整型或枚举类型**

---

### 2. 循环语句

#### ▶ while

```cpp
while (condition) {
}
```

#### ▶ do-while

```cpp
do {
} while (condition);
```

#### ▶ for

```cpp
for (int i = 0; i < n; i++) {
}
```

`for` 是**最常用、最安全**的循环结构

---

### 3. 跳转语句

```cpp
break;      // 跳出循环
continue;   // 跳过本次循环
goto;       // 不推荐使用
```

**goto 在现代 C++ 中几乎不使用**

---

## 1.8 C++ 中的引用（Reference）

### 1. 引用的本质

> **引用是变量的别名**

```cpp
int a = 10;
int& b = a;
```

* `b` 和 `a` 指向**同一块内存**
* 修改 `b` 就是修改 `a`

### 2. 引用的特点

✔ 必须初始化
✔ 不能再绑定到其他对象
✔ 比指针更安全

### 3. 引用 vs 指针（初步）

| 项目     | 引用  | 指针   |
| ------ | --- | ---- |
| 是否为空   | ❌   | ✔    |
| 是否可变指向 | ❌   | ✔    |
| 使用方式   | 直接用 | 需解引用 |

**引用是 C++ 相对 C 的重要升级**

---

# 二、数组指针结构体（与C类似）

## 2.1 数组（Array）

### 1. 数组的定义

数组是 **相同类型元素的连续内存集合**。

```cpp
int a[5] = {1, 2, 3, 4, 5};
```

* `a` 是数组名
* `5` 是元素个数
* 每个元素类型相同

---

### 2. 数组的内存特点（重点）

```cpp
int a[5];
```

内存中表现为：

```
a[0] a[1] a[2] a[3] a[4]
```

* 连续存储
* 可通过下标随机访问
* 下标从 **0** 开始

> 注：**数组名表示首元素地址（在表达式中）**

---

### 3. 数组访问

```cpp
a[0] = 10;
cout << a[2];
```

⚠️ **C++ 不做越界检查**

```cpp
a[10] = 100;   // 未定义行为（非常危险）
```

---

### 4. 数组名与 sizeof

```cpp
sizeof(a);        // 整个数组大小
sizeof(a[0]);     // 单个元素大小
```

注：在同一作用域中，`sizeof(a)` 才有效。

---

### 5. 数组作为函数参数（退化）

```cpp
void func(int arr[]) {
}
```

⚠️ 数组传参时：

> **数组会退化为指针**

```cpp
sizeof(arr);   // 指针大小
```

---

## 2.2 指针（Pointer）

### 1. 指针的定义

指针是 **存储地址的变量**。

```cpp
int a = 10;
int* p = &a;
```

* `p` 保存的是 `a` 的地址
* `*p` 表示访问该地址的内容

---

### 2. 指针的基本操作

```cpp
*p = 20;   // 修改 a
```

内存关系：

```
p --> a --> 20
```

---

### 3. 指针与数组的关系（核心）

```cpp
int a[5];
int* p = a;     // 等价于 &a[0]
```

访问等价：

```cpp
a[2] == *(a + 2) == *(p + 2)
```

**数组下标本质是指针运算**

---

### 4. 指针运算

```cpp
p + 1   // 指向下一个元素（+ sizeof(type)）
```

⚠️ 指针只能在**同一数组范围内运算**

---

### 5. 空指针与野指针

```cpp
int* p = nullptr;   // C++ 推荐
```

* 空指针：不指向任何对象
* 野指针：指向非法地址（最危险）

---

### 6. 指针与 const（常考）

```cpp
const int* p;   // 指向 const
int* const p;   // 指针本身 const
```

记忆口诀：

> **const 靠谁，谁不能改**

---

## 2.3 结构体（struct）

### 1. 结构体的定义

结构体是 **不同类型数据的组合**。

```cpp
struct Student {
    int id;
    char name[20];
    float score;
};
```

---

### 2. 结构体变量

```cpp
Student s1;
s1.id = 1;
```

C++ 中可直接省略 `struct`。

---

### 3. 结构体数组

```cpp
Student cls[30];
```

用于表示“多个同类对象”。

---

### 4. 结构体指针

```cpp
Student* p = &s1;
p->id = 2;   // 等价于 (*p).id
```

---

### 5. 结构体内存对齐

* 成员按对齐规则存放
* 可能存在 **填充字节**

```cpp
sizeof(Student);
```

内存对齐影响：

* 性能
* 网络通信
* 文件存储

---

## 2.4 总结

* 数组：连续内存
* 指针：地址操作
* 结构体：数据聚合

> **数组 + 指针 = 底层访问能力**
> **结构体 = 复杂数据建模能力**

这三者合在一起，就是：

* 操作系统
* 驱动
* 嵌入式
* 高性能程序

---

# 三、函数（Functions）

## 3.1 基础语法

### 1. 函数的作用

* 拆分程序逻辑
* 提高复用性
* 降低复杂度
* 便于测试和维护

---

### 2. 函数定义结构

```cpp
返回类型 函数名(参数列表) {
    函数体
    return 返回值;
}
```

示例：

```cpp
int add(int a, int b) {
    return a + b;
}
```

---

### 3. 函数声明（原型）

```cpp
int add(int, int);
```

**先声明，后使用**（尤其在多文件工程中）

---

### 4. return 语句

* 返回值
* 结束函数执行

```cpp
return;
return 10;
```

---

## 3.2 传参（Parameter Passing）

### 1. 值传递（最基础）

```cpp
void func(int x) {
    x = 10;
}
```

* 传入的是副本
* 不影响实参

**安全，但效率低（大对象）**

---

### 2. 指针传递

```cpp
void func(int* p) {
    *p = 10;
}
```

* 可修改外部变量
* 需要判空

C 风格方式

---

### 3. 引用传递（C++ 推荐）

```cpp
void func(int& x) {
    x = 10;
}
```

* 语义清晰
* 无需解引用
* 不可为空

**90% 场景优先用引用**

---

### 4. const 引用（工程级重点）

```cpp
void print(const string& s);
```

* 防止修改
* 避免拷贝
* 提升性能

---

## 3.3 作用域（Scope）

### 1. 局部作用域

```cpp
void func() {
    int x = 10;
}
```

* 只在函数内部可见
* 生命周期随函数结束

---

### 2. 全局作用域

```cpp
int g = 10;
```

* 整个程序可见
* 工程中慎用

---

### 3. 块作用域

```cpp
if (true) {
    int x = 5;
}
```

---

### 4. 名字遮蔽（重要）

```cpp
int x = 10;

void func() {
    int x = 20;   // 遮蔽全局 x
}
```

---

## 3.4 嵌套调用（Function Call Chain）

### 1. 什么是嵌套调用

```cpp
func1(func2());
```

* 一个函数调用另一个函数
* 调用顺序由表达式决定

---

### 2. 调用栈（Call Stack）

```cpp
main → funcA → funcB
```

* 每次调用生成一个栈帧
* 局部变量存放在栈上

📌 **函数返回 = 栈帧销毁**

---

## 3.5 递归（Recursion）

### 1. 定义

函数**直接或间接调用自己**。

```cpp
int fact(int n) {
    if (n == 1) return 1;
    return n * fact(n - 1);
}
```

---

### 2. 递归的必要条件（必须满足）

1️⃣ 递归终止条件
2️⃣ 问题规模逐步减小

❌ 否则栈溢出

---

### 3. 递归 vs 循环

| 对比  | 递归 | 循环 |
| --- | -- | -- |
| 可读性 | 高  | 一般 |
| 性能  | 较低 | 高  |
| 栈消耗 | 有  | 无  |

注：**工程中优先循环**

---

## 3.6 参数进阶

### 1. 默认参数

### 2. 占位参数

### 3. 什么可以做参数

#### 变量与常量

#### 数组与指针

#### 结构体

#### 类与对象

#### 函数本身做参数

---

## 3.6 参数进阶

### 1. 默认参数（Default Argument）

#### 1️⃣ 基本概念

```cpp
void func(int a, int b = 10, int c = 20);
```

* 调用时可省略带默认值的参数
* **从右向左依次省略**

```cpp
func(1);        // b=10, c=20
func(1, 2);     // c=20
func(1, 2, 3);  // 全部显式传参
```

---

#### 2️⃣ 重要规则（必记）

**默认参数最好写在函数声明处**

```cpp
void func(int a, int b = 10);  // ✔
void func(int a, int b = 20) // 写在定义处，不推荐
{

}  
```

**默认参数是编译期行为**

* 调用处直接“补参数”
* 与函数体无关

---

#### 3️⃣ 工程注意点

* 默认参数属于 **接口设计**
* 改默认值 ≠ 改函数实现

---

### 2. 占位参数（Unused / Placeholder Parameter）

#### 1️⃣ 什么是占位参数

```cpp
void foo(int a, int) {
    // 第二个参数不使用
}
```

* 只占位置，不参与逻辑
* 常用于 **接口兼容 / 回调函数**

---

#### 2️⃣ 常见使用场景

##### ✔ 接口预留

```cpp
void handler(int event, void* context) {
    (void)context; // 避免未使用警告
}
```

##### ✔ 回调函数

```cpp
void callback(int, int) {
    // 按接口要求保留
}
```

---

### 3. 什么可以作为参数

#### 1️⃣ 变量与常量

```cpp
void foo(int a);
void foo(const int a);
```

* 默认是 **值传递**
* `const` 仅限制函数内修改

---

#### 2️⃣ 数组与指针

```cpp
void foo(int arr[]);
void foo(int* arr);
```

等价于：

```cpp
void foo(int* arr);
```

**数组作为参数会退化为指针**

---

#### 3️⃣ 结构体作为参数

```cpp
struct Point { int x, y; };

void foo(Point p);          // 值传递（拷贝）
void foo(const Point& p);   // 推荐
void foo(Point& p);         // 修改实参
```

---

#### 4️⃣ 类与对象作为参数

```cpp
class Person {};

void foo(Person p);           // 调用拷贝构造
void foo(const Person& p);    // ⭐ 最常用
void foo(Person& p);          // 会修改对象
```

**工程原则：类参数 90% 用 `const T&`**

---

#### 5️⃣ 函数本身作为参数

##### 函数指针

```cpp
int add(int a, int b);

void calc(int (*op)(int,int)) {
    op(1, 2);
}
```

##### 函数引用（更推荐）

```cpp
void calc(int (&op)(int,int)) {
    op(1, 2);
}
```

#### `std::function`（高层抽象）

```cpp
void calc(std::function<int(int,int)> op);
```

---

### 4. 参数传递方式（语义维度）

#### 1️⃣ 值传递

```cpp
void foo(int a);
```

* 拷贝一份
* 不影响实参
* 小对象适用

---

#### 2️⃣ 引用传递

```cpp
void foo(int& a);
```

* 不拷贝
* 修改会影响外部
* 表达“我要改你”

---

#### 3️⃣ const 引用（⭐⭐⭐）

```cpp
void foo(const int& a);
```

* 不拷贝
* 不可修改
* 可绑定临时对象

**最推荐**

---

#### 4️⃣ 指针传递

```cpp
void foo(int* p);
```

* 可修改
* 可为空
* 需要判空

---

### 5. 参数设计的工程原则（核心）

#### 一条黄金法则

> **能 const 引用，就别值传**

---

#### 设计选择速查表

| 场景        | 推荐方式        |
| --------- | ----------- |
| 基本类型（int） | 值传          |
| 小结构体      | 值传 / const& |
| 大对象 / 类   | const&      |
| 需要修改      | &           |
| 可为空       | *           |
| 回调函数      | 函数指针 / 引用   |

---

### 6. 性能与安全视角

#### 1️⃣ 性能

* 值传递可能触发拷贝构造
* 引用 / 指针零拷贝

---

#### 2️⃣ 生命周期

```cpp
const T& foo(); // ❌ 返回局部变量
```

**参数不能引用已销毁对象**

---

## 3.7 函数重载（Overloading）

### 1. 什么是函数重载

**同名函数，不同参数列表**

```cpp
int add(int a, int b);
double add(double a, double b);
```

---

### 2. 重载匹配规则（函数重载满足条件）

* 参数个数不同
* 参数类型不同
* 顺序不同
* 在同一个作用域下
* 函数名称相同

❌ **仅返回类型不同不构成重载**

---

### 3. 重载的意义

* 提升接口一致性
* 提高代码可读性

**体现 C++ 的类型系统能力**

---

## 3.7 总结

* 函数 = 抽象 + 复用
* 传参方式决定 **安全性与效率**
* 调用栈是理解递归和嵌套的关键
* 重载是 C++ 区别于 C 的重要特性

---

# 四、联合、枚举与类型别名

## 4.1 联合（union）

* **定义**：联合是一种特殊的数据结构，所有成员共享同一块内存区域。
* **特点**：

  * 所有成员共用一块内存，大小等于最大成员的大小。
  * 同一时刻只能存储其中一个成员的值。
* **用途**：

  * 节省内存。
  * 实现类型转换或不同类型数据的复用。
* **示例**：

  ```cpp
  union Data {
      int i;
      float f;
      char c;
  };
  Data d;
  d.i = 10;
  // 此时 d.f 和 d.c 的值不确定，访问前需谨慎
  ```
* **注意事项**：

  * 访问当前未赋值的成员是未定义行为。
  * C++11 后联合体可以有非平凡类型成员，但使用需注意。

---

## 4.2 枚举（enum）

* **定义**：枚举是一种自定义的整型常量集合。
* **传统枚举**：

  * 默认底层类型是 `int`，枚举成员默认从0开始递增。
  * 例如：

    ```cpp
    enum Color { RED, GREEN, BLUE };
    ```
* **强类型枚举（C++11 起）**：

  * 使用 `enum class`，具有作用域且不隐式转换为整型。
  * 例如：

    ```cpp
    enum class Color : char { RED = 'r', GREEN = 'g', BLUE = 'b' };
    ```
  * 优点：避免枚举名冲突，类型更安全。
* **枚举使用**：

  * 方便表示一组相关常量，增强代码可读性。
* **注意**：

  * 传统枚举成员会污染外部作用域，强类型枚举则不会。

---

## 4.3 类型别名（Type Aliases）

* **定义**：给已有类型取一个新的名字，方便使用或提高代码可读性。
* **方式一：`typedef`**

  * 经典写法，源自C语言。
  * 例如：

    ```cpp
    typedef unsigned int uint;
    typedef int* IntPtr;
    ```
* **方式二：`using`（C++11 引入）**

  * 现代写法，更直观且支持模板别名。
  * 例如：

    ```cpp
    using uint = unsigned int;
    template<typename T>
    using Vec = std::vector<T>;
    ```
* **区别**：

  * `using` 可以用于模板别名，`typedef` 不支持模板别名。
  * `using` 语法更清晰，现代C++推荐使用。
* **注意**：

  * 类型别名不会创建新类型，只是为已有类型取别名。

---

## 4.4 总结

| 概念   | 作用           | 关键特点                         |
| ---- | ------------ | ---------------------------- |
| 联合   | 多种类型共享同一内存空间 | 成员共享内存，只能存储一个成员的值            |
| 枚举   | 定义一组整型常量     | 传统枚举易污染作用域，强类型枚举更安全          |
| 类型别名 | 给类型起新名字      | `typedef`与`using`，后者支持模板且更灵活 |

---

# 五、类与对象（Class and Object）

C++中的类与对象是面向对象编程（OOP）的核心，封装数据和操作数据的方法，是组织代码的重要方式。

---

## 5.1 封装（Encapsulation）

* **定义**：将数据成员（属性）和成员函数（方法）绑定在一起，隐藏内部实现细节，只暴露接口给外部使用。
* **访问控制**：

  * `public`：公有成员，对外公开，类外可访问。
  * `private`：私有成员，只能在类内部访问，外部不可见。
  * `protected`：受保护成员，类内部和子类可访问，外部不可见。
* **好处**：

  * 隐藏实现细节，保证数据安全。
  * 控制访问权限，防止外部非法修改。
  * 使接口清晰，降低耦合度。

```cpp
class Person {
private:
    int age;  // 私有成员变量
public:
    void setAge(int a) {
        if (a > 0) age = a;
    }
    int getAge() {
        return age;
    }
};
```

---

## 5.2 构造函数与析构函数（Constructor & Destructor）

在C++中，构造函数和析构函数是编译器强制要求做的事情，如果我们不提供构造和析构函数，编译器会提供，当然编译器提供的这些都是空实现。

### 1. 构造函数：

  * 类对象创建时自动调用，只调用一次，用于初始化对象。
  * 函数名与类名相同，没有返回值，不写void。
  * 支持函数重载（可以有多个构造函数）。
  * **默认构造函数**：无参数的构造函数。
  * **带参构造函数**：带参数的构造函数。
  * **拷贝构造函数**：用一个同类型对象初始化另一个对象，形如`ClassName(const ClassName& obj)`。
  * **成员初始化列表**：构造函数初始化成员变量的推荐方式。

#### 语法
```text
类名(){}
```

```cpp
class Person {
public:
    int age;
    Person() { age = 0; }  // 默认构造函数
    Person(int a) : age(a) {}  // 带参构造函数，使用初始化列表
    Person(const Person& p) { age = p.age; }  // 拷贝构造函数
};
```

#### 构造函数的调用方式

假设我们有如下类：

```cpp
class Person {
public:
    int age;

    Person() {
        age = 0;
    }

    Person(int a) {
        age = a;
    }
};
```

---

##### 1️⃣ 括号法（最常用、最直观）

* 语法

```cpp
类名 对象名(参数);
```

* 示例

```cpp
Person p1;        // 调用无参构造函数
Person p2(10);    // 调用有参构造函数
```

* ⚠ 注意（经典陷阱）

```cpp
Person p();   // ❌ 不是创建对象
```

这行代码**被编译器当作函数声明**（返回 Person 的函数 p）

✔ 正确写法：

```cpp
Person p;
```

---

##### 2️⃣ 显示法（等号法 / 拷贝初始化）

* 语法

```cpp
类名 对象名 = 类名(参数);
```

* 示例

```cpp
Person p1 = Person(10);
Person p2 = Person();
```

* 本质

* **先构造一个临时对象**
* 再用它来初始化 `p1`
* 编译器通常会做 **返回值优化（RVO）**，不会真的产生多余拷贝

* 特点

* 看起来像赋值，其实是初始化
* 早期 C++ 教学中常见
* 可读性略差于括号法

---

##### 3️⃣ 隐式转换法（重点理解）

* 语法

```cpp
类名 对象名 = 参数;
```

* 示例

```cpp
Person p = 10;   // 等价于 Person p(10);
```

⚠ 潜在问题

隐式转换有时会导致：

* 代码可读性变差
* 意外构造对象
* 难以发现的 bug

---

explicit 关键字（和隐式转换强相关）

禁止隐式转换

```cpp
class Person {
public:
    explicit Person(int a) {
        age = a;
    }
    int age;
};
```

此时：

```cpp
Person p1(10);    // ✅ OK
Person p2 = 10;   // ❌ 编译错误（隐式转换被禁止）
```

---

##### 结论

* **单参数构造函数 + 未加 explicit → 允许隐式转换**
* **加 explicit → 只能显式调用**

👉 工程中强烈推荐：

> **单参数构造函数默认加 `explicit`**

---

##### 三种调用方式对比总结

| 调用方式 | 示例                       | 是否推荐  | 特点      |
| ---- | ------------------------ | ----- | ------- |
| 括号法  | `Person p(10);`          | ⭐⭐⭐⭐⭐ | 最常用、最清晰 |
| 显示法  | `Person p = Person(10);` | ⭐⭐⭐   | 可读性一般   |
| 隐式法  | `Person p = 10;`         | ⭐⭐    | 易出隐患，慎用 |

---

### 2. 析构函数：

  * 对象生命周期结束时自动调用，用于资源释放。
  * 名称为`~ClassName()`，无参数，无返回值。
  * 每个类只有一个析构函数，不支持重载。

```cpp
class Person {
public:
    ~Person() { 
        // 清理代码，如释放动态内存
    }
};
```

---

### 3. 初始化列表

---

## 5.3 const 成员与只读对象（const Members and Const Objects）

* **const成员函数**：

  * 在成员函数后加`const`，表示该函数不会修改对象的任何成员变量。
  * 只能调用其他const成员函数或访问成员变量，不能修改成员变量。

```cpp
class Person {
private:
    int age;
public:
    int getAge() const { return age; }  // 不修改成员变量，声明为const
    void setAge(int a) { age = a; }
};
```

* **只读对象（const对象）**：

  * 用`const`修饰的对象，只能调用const成员函数。
  * 防止对象状态被修改。

```cpp
const Person p;
p.getAge();  // 合法，getAge()是const成员函数
p.setAge(10);  // 错误，不能调用非const成员函数
```

* **const成员变量**：

  * 在类中用`const`修饰成员变量，必须在初始化列表中初始化。
  * 该成员变量对象创建后不可修改。

---

## 5.4 对象拷贝与赋值（Copy and Assignment）

### 1. 拷贝构造函数

  * 用于初始化新对象，如`ClassName obj2(obj1);`或`ClassName obj2 = obj1;`。
  * 默认实现为成员逐个拷贝（浅拷贝）。
  * 如果类中有指针成员，默认浅拷贝可能导致资源冲突（双重释放），需自定义深拷贝。

### 2. 赋值运算符（operator=）重载

  * 用于已经存在的对象赋值，如`obj2 = obj1;`。
  * 默认赋值运算符也是浅拷贝。
  * 自定义赋值运算符时，通常需要判断自赋值，释放旧资源，深拷贝新资源，返回`*this`。

```cpp
class Person {
    int* pAge;
public:
    Person(int age) { pAge = new int(age); }
    ~Person() { delete pAge; }
    
    // 拷贝构造函数
    Person(const Person& other) {
        pAge = new int(*other.pAge);
    }
    
    // 赋值运算符重载
    Person& operator=(const Person& other) {
        if (this == &other) return *this; // 自赋值保护
        delete pAge;
        pAge = new int(*other.pAge);
        return *this;
    }
};
```

### 3. 浅拷贝与深拷贝

  * 浅拷贝：拷贝指针地址，多个对象指向同一内存，容易出错。
  * 深拷贝：拷贝指针指向的内容，避免资源冲突。

---

## 5.5 友元（Friend）

### 1. 定义

  * 允许非成员函数或其他类访问该类的私有成员或保护成员。
  * 用`friend`关键字声明。

### 2. 类型

  * 友元函数
  * 友元类
  * 友元成员函数（另一个类的成员函数作为友元）

### 3. 使用场景

  * 需要打破封装实现特定操作，如运算符重载、接口设计等。

```cpp
class Box {
private:
    int width;
public:
    Box() : width(0) {}
    friend void printWidth(Box& b);  // 友元函数声明
};

void printWidth(Box& b) {
    std::cout << "Width: " << b.width << std::endl;  // 访问私有成员
}
```

---

## 5.6 继承（Inheritance）

### 1. 定义

  * 派生类（子类）继承基类（父类）的成员，支持代码复用和扩展。

### 2. 语法

  ```cpp
  class Base { ... };
  class Derived : access_specifier Base { ... };
  ```

---

### 3. 访问权限

  * `public`继承：基类的`public`成员在派生类中仍为`public`，`protected`成员为`protected`，`private`成员不可见。
  * `protected`继承：基类`public`和`protected`成员都变为`protected`。
  * `private`继承：基类`public`和`protected`成员都变为`private`。

### 4. 继承特性

  * 继承不继承基类的构造函数、析构函数、友元、私有成员。
  * 子类对象中包含父类对象部分。

### 5. 重载与隐藏

  * 派生类可以重载基类成员函数（同名不同参数）。
  * 派生类成员隐藏基类同名成员（不论参数）。

```cpp
class Animal {
public:
    void speak() { std::cout << "Animal speaks" << std::endl; }
};

class Dog : public Animal {
public:
    void speak() { std::cout << "Dog barks" << std::endl; }
};
```

---

## 5.7 多态（虚函数）（Polymorphism）

* **定义**：

  * 同一接口，不同实现，运行时决定调用哪个函数。

* **实现机制**：

  * 使用`virtual`关键字声明基类成员函数为虚函数。
  * 派生类重写虚函数，实现不同功能。
  * 通过基类指针或引用调用时，调用派生类版本（运行时绑定）。

* **特点**：

  * 支持动态绑定（运行时多态）。
  * 必须通过基类指针或引用访问才生效。
  * 虚函数表（vtable）和虚指针（vptr）实现。

* **纯虚函数与抽象类**：

  * 纯虚函数：`virtual void func() = 0;`，没有实现，必须在派生类重写。
  * 抽象类：包含至少一个纯虚函数的类，不能实例化，只能作为基类。

```cpp
class Animal {
public:
    virtual void speak() {
        std::cout << "Animal speaks" << std::endl;
    }
    virtual ~Animal() {}  // 虚析构函数，确保派生类正确析构
};

class Dog : public Animal {
public:
    void speak() override {  // 重写虚函数
        std::cout << "Dog barks" << std::endl;
    }
};

void makeSpeak(Animal& a) {
    a.speak();  // 运行时根据对象类型调用对应版本
}
```

---

## 5.8 运算符重载（Operator Overloading）

### 1. 什么是运算符重载

**定义**：
运算符重载是指 **为自定义类型（类）赋予运算符的含义**，使对象像内置类型一样使用运算符。

> 本质：
> **运算符重载 ≠ 新运算符**
> 而是 **函数调用的语法糖**

---

### 2. 为什么需要运算符重载

#### 没有重载时

```cpp
Point p3 = add(p1, p2);
```

#### 有重载时

```cpp
Point p3 = p1 + p2;
```

#### 好处

* 提升代码可读性
* 接近数学/领域模型表达
* 方便 STL / 泛型编程
* 符合 C++ 的类型扩展思想

---

### 3. 运算符重载的基本规则

#### ✔ 可以重载的运算符

```
+  -  *  /  %
== != < > <= >=
[] () -> 
<< >> 
++ --
= += -= *= /= 
```

#### ❌ 不可以重载的运算符

```
.   ::   ?:   sizeof   typeid
```

#### 通用规则

1. **至少有一个操作数是自定义类型**
2. **不能改变运算符的优先级和结合性**
3. **不能发明新运算符**
4. **不要滥用，语义必须自然**

---

### 4. 运算符重载的两种方式

#### 1️⃣ 成员函数方式

```cpp
class Point {
public:
    int x, y;

    Point(int x, int y) : x(x), y(y) {}

    Point operator+(const Point& p) {
        return Point(x + p.x, y + p.y);
    }
};
```

使用

```cpp
Point p3 = p1 + p2;  // 等价于 p1.operator+(p2)
```

---

#### 2️⃣ 全局函数（友元）方式

```cpp
class Point {
public:
    int x, y;
    Point(int x, int y) : x(x), y(y) {}

    friend Point operator+(const Point& a, const Point& b);
};

Point operator+(const Point& a, const Point& b) {
    return Point(a.x + b.x, a.y + b.y);
}
```

何时用友元？

* 左操作数不是当前类对象
* 需要访问私有成员
* 如：`<<`、`>>`

---

### 5. 常见运算符重载详解

#### 1️⃣ `+` 运算符

语义

* 不改变原对象
* 返回新对象

```cpp
Point operator+(const Point& p) const;
```

---

### 2️⃣ `=` 赋值运算符（非常重要）

#### 典型实现模板（必背）

```cpp
class Person {
    int* age;
public:
    Person(int a) {
        age = new int(a);
    }

    ~Person() {
        delete age;
    }

    Person& operator=(const Person& other) {
        if (this == &other) return *this;  // 自赋值保护

        delete age;
        age = new int(*other.age);
        return *this;
    }
};
```

#### 必须做的三件事

1. 判断自赋值
2. 释放旧资源
3. 深拷贝新资源

---

### 3️⃣ `==` / `!=` 比较运算符

```cpp
bool operator==(const Point& p) const {
    return x == p.x && y == p.y;
}

bool operator!=(const Point& p) const {
    return !(*this == p);
}
```

---

### 4️⃣ `[]` 下标运算符

```cpp
class Array {
    int data[10];
public:
    int& operator[](int index) {
        return data[index];
    }
};
```

✔ 返回 **引用**
✔ 支持左值赋值

```cpp
arr[3] = 100;
```

---

### 5️⃣ `<<` 输出运算符（必会）

```cpp
class Person {
    int age;
public:
    Person(int a) : age(a) {}

    friend ostream& operator<<(ostream& os, const Person& p);
};

ostream& operator<<(ostream& os, const Person& p) {
    os << "age=" << p.age;
    return os;
}
```

#### 必须这样写的原因

* 左操作数是 `ostream`
* 不能作为成员函数
* 返回引用支持链式输出

```cpp
cout << p << endl;
```

---

### 6️⃣ `++` 自增运算符（面试高频）

#### 前置++

```cpp
Point& operator++() {
    ++x;
    return *this;
}
```

#### 后置++

```cpp
Point operator++(int) {
    Point temp = *this;
    ++x;
    return temp;
}
```

| 类型   | 参数      | 返回值 |
| ---- | ------- | --- |
| 前置++ | 无       | 引用  |
| 后置++ | int（占位） | 值   |

---

## 5.9 const 与运算符重载（工程细节）

### 推荐写法

```cpp
Point operator+(const Point& p) const;
```

含义：

* 参数不被修改
* 当前对象不被修改
* 支持 const 对象参与运算

---

## 5.10 运算符重载的工程建议（非常重要）

### ✔ 推荐

* `+ - *`：返回新对象
* `= +=`：返回 `*this`
* `== !=`：只比较语义状态
* `<<`：友元函数

### ❌ 不推荐

* 改变运算符原有语义
* 过度重载（可读性下降）
* 把业务逻辑塞进运算符

---

## 5.11 速查表

| 运算符  | 推荐方式    | 返回值        |
| ---- | ------- | ---------- |
| `+`  | 成员 / 友元 | 新对象        |
| `=`  | 成员      | `*this`    |
| `==` | 成员      | `bool`     |
| `[]` | 成员      | 引用         |
| `<<` | 友元      | `ostream&` |
| `++` | 成员      | 前置引用 / 后置值 |

---

## 5.12 总结

| 主题      | 说明              | 关键点                            |
| ------- | --------------- | ------------------------------ |
| 封装      | 数据和操作绑定，访问控制    | `public`、`private`、`protected` |
| 构造与析构   | 对象初始化和资源释放      | 构造函数重载、析构函数、初始化列表              |
| const成员 | 不修改对象状态的函数及只读对象 | const成员函数、const对象              |
| 拷贝与赋值   | 对象复制，浅拷贝与深拷贝    | 拷贝构造函数、赋值运算符重载                 |
| 友元      | 允许非成员访问私有成员     | 友元函数、友元类                       |
| 继承      | 派生类继承基类，实现代码复用  | 公有继承、保护继承、私有继承                 |
| 多态      | 运行时动态调用函数       | 虚函数、纯虚函数、虚析构函数                 |

---

# 六、C++ 文件操作（File I/O）

C++ 的文件操作基于 **流（Stream）模型**，通过 **`<fstream>`** 头文件完成，核心思想是：

> **文件 = 特殊的输入输出流**

---

## 6.1 文件流的基本概念

### C++ 中的三大文件流类

| 类名         | 作用        | 方向      |
| ---------- | --------- | ------- |
| `ifstream` | 文件输入流     | 文件 → 程序 |
| `ofstream` | 文件输出流     | 程序 → 文件 |
| `fstream`  | 文件输入 + 输出 | 双向      |

```cpp
#include <fstream>
```

---

## 6.2 文件打开与关闭

### 1️⃣ 打开文件

#### 方法一：构造函数（常用）

```cpp
ofstream ofs("test.txt");
```

#### 方法二：`open()` 成员函数

```cpp
ofstream ofs;
ofs.open("test.txt");
```

---

### 2️⃣ 判断是否打开成功

```cpp
if (!ofs.is_open()) {
    cout << "open file failed" << endl;
}
```

---

### 3️⃣ 关闭文件

```cpp
ofs.close();
```

#### **原则**：

> 打开 → 使用 → 关闭（RAII 风格）

---

## 6.3 文件打开模式（重点）

### 常用打开模式

| 模式            | 说明          |
| ------------- | ----------- |
| `ios::in`     | 读文件         |
| `ios::out`    | 写文件（会清空原内容） |
| `ios::app`    | 追加写         |
| `ios::trunc`  | 清空文件        |
| `ios::binary` | 二进制方式       |

### 模式组合

```cpp
ofstream ofs("test.txt", ios::out | ios::app);
```

---

## 6.4 文本文件写操作

### 示例：写入文本文件

```cpp
ofstream ofs("data.txt");

ofs << "hello world" << endl;
ofs << 123 << " " << 3.14 << endl;

ofs.close();
```

✔ 操作方式和 `cout` 完全一致
✔ 本质：输出流

---

## 6.5 文本文件读操作（重点）

### 方式一：`>>` 运算符（按空白分隔）

```cpp
ifstream ifs("data.txt");
string s;
int x;

ifs >> s >> x;
```

遇到空格、换行会停止

---

### 方式二：`getline()`（最常用）

```cpp
string line;
while (getline(ifs, line)) {
    cout << line << endl;
}
```

✔ 一行一行读

✔ 不丢空格

✔ 推荐用于文本文件

---

### 方式三：`get()`（按字符读）

```cpp
char c;
while ((c = ifs.get()) != EOF) {
    cout << c;
}
```

✔ 精细控制
✔ 了解即可

---

## 6.6 二进制文件操作（工程重点）

### 为什么需要二进制文件

* 不做格式转换
* 读写效率高
* 常用于：

  * 结构体存储
  * 配置文件
  * 序列化数据

---

### 1️⃣ 二进制写文件

```cpp
class Person {
public:
    char name[20];
    int age;
};

Person p = {"Tom", 18};

ofstream ofs("person.bin", ios::binary | ios::out);
ofs.write((const char*)&p, sizeof(p));
ofs.close();
```

---

### 2️⃣ 二进制读文件

```cpp
Person p;

ifstream ifs("person.bin", ios::binary | ios::in);
ifs.read((char*)&p, sizeof(p));
ifs.close();

cout << p.name << " " << p.age << endl;
```

#### 注意：

* `write/read` 操作的是 **内存字节**
* 必须确保读写结构一致

---

## 6.7 文件读写位置指针（了解）

### 1️⃣ 获取位置

```cpp
ifs.tellg();  // 输入指针
ofs.tellp();  // 输出指针
```

---

### 2️⃣ 移动位置

```cpp
ifs.seekg(0, ios::beg);  // 从文件开头
ofs.seekp(10, ios::cur); // 相对当前位置
```

| 位置         | 含义   |
| ---------- | ---- |
| `ios::beg` | 文件开头 |
| `ios::cur` | 当前位置 |
| `ios::end` | 文件结尾 |

---

## 6.8 文件状态判断（工程技巧）

### 常用判断函数

```cpp
ifs.eof();   // 是否到达文件末尾
ifs.fail();  // 逻辑错误
ifs.bad();   // 严重错误
ifs.good();  // 状态正常
```

### 实际工程中常用：

```cpp
while (getline(ifs, line)) {
    ...
}
```

---

## 6.9 C 文件操作 vs C++ 文件操作

| 对比项 | C             | C++                |
| --- | ------------- | ------------------ |
| 接口  | `FILE*`       | 流对象                |
| 函数  | `fopen/fread` | `<< >> read write` |
| 风格  | 过程式           | 面向对象               |
| 安全性 | 低             | 高                  |
| 可读性 | 一般            | 好                  |

---

## 6.10 工程实践建议

### ✔ 推荐做法

* 文本文件：`ifstream / ofstream + getline`
* 二进制文件：`binary + read/write`
* 使用 RAII，避免忘记 `close()`
* 打开文件立即判断是否成功

### ❌ 不推荐

* 结构体含指针直接写入文件
* 混用文本和二进制模式
* 不检查文件打开状态

---

## 6.11 总结

> * 文件操作基于 **流**
> * 文本文件：`<< >> getline`
> * 二进制文件：`read / write`
> * 模式要选对
> * 打开成功一定要检查
> * 结构体二进制存储要谨慎

---

# 七、现代 C++ 基础（C++11+）

现代 C++ 的目标只有一句话：

> **写更安全的代码，少犯错，让编译器多干活**

---

## 7.1 `auto` 与类型推导

### 1️⃣ `auto` 是什么

`auto` 用于 **让编译器自动推导变量类型**，在编译期完成。

```cpp
auto x = 10;        // int
auto y = 3.14;      // double
auto s = "hello";   // const char*
```

`auto` **不是动态类型**，类型一旦推导就固定。

---

### 2️⃣ 使用场景（工程中非常常见）

#### （1）复杂类型简化

```cpp
std::vector<int>::iterator it = v.begin();
auto it = v.begin();
```

#### （2）模板 / 泛型代码

```cpp
template<typename T>
void func(T t) {
    auto x = t + 1;
}
```

---

### 3️⃣ 推导规则要点

```cpp
int a = 10;
int& r = a;
const int c = 20;

auto x1 = a;   // int
auto x2 = r;   // int（引用被忽略）
auto x3 = c;   // int（const被忽略）
```

**auto 会忽略顶层 const / 引用**

#### 保留方式

```cpp
auto& x = r;        // int&
const auto& y = c; // const int&
```

---

## 7.2 `nullptr` 与类型安全

### 1️⃣ 为什么需要 `nullptr`

C++98 中：

```cpp
#define NULL 0
```

问题：

```cpp
void f(int);
void f(char*);

f(NULL);  // ❌ 二义性
```

---

### 2️⃣ `nullptr` 的引入（C++11）

```cpp
int* p = nullptr;
```

特点：

* 类型是 `std::nullptr_t`
* 只能表示 **空指针**
* 不会被当作整数

```cpp
f(nullptr); // 明确调用 f(char*)
```

**现代 C++ 中：一律使用 `nullptr`**

---

## 7.3 `enum class`（强类型枚举）

### 1️⃣ 传统 `enum` 的问题

```cpp
enum Color { Red, Green };
enum Status { OK, Error };

Color c = Red;
if (c == OK) { }  // ❌ 竟然合法
```

---

### 2️⃣ `enum class`（C++11）

```cpp
enum class Color { Red, Green };
enum class Status { OK, Error };
```

特点：

* **强类型**
* 不会隐式转换为 int
* 枚举值有作用域

```cpp
Color c = Color::Red;
// if (c == Status::OK) ❌ 编译错误
```

---

### 3️⃣ 指定底层类型

```cpp
enum class ErrorCode : uint8_t {
    OK = 0,
    FAIL = 1
};
```

📌 工程中非常常见（协议 / 嵌入式）

---

## 7.4 `constexpr`（编译期常量）

### 1️⃣ `constexpr` 是什么

表示：

> **能在编译期求值的表达式 / 函数**

```cpp
constexpr int N = 10;
int arr[N];  // 合法
```

---

### 2️⃣ `constexpr` 函数

```cpp
constexpr int square(int x) {
    return x * x;
}
```

```cpp
int a = square(5);     // 编译期计算
int b = square(a);     // 运行期计算
```

#### 同一个函数：
**能编译期就编译期，不能就退化成运行期**

---

### 3️⃣ vs `const`

| 对比      | const | constexpr |
| ------- | ----- | --------- |
| 是否编译期   | 不一定   | 一定（条件满足）  |
| 可用于数组大小 | 不稳定   | ✔         |
| 表达语义    | 只读    | 编译期常量     |

---

## 7.5 范围 for（Range-based for）

### 1️⃣ 语法

```cpp
for (元素声明 : 容器) {
    ...
}
```

---

### 2️⃣ 示例

```cpp
std::vector<int> v = {1, 2, 3};

for (int x : v) {
    cout << x << endl;
}
```

---

### 3️⃣ 引用 + const（工程推荐）

```cpp
for (const auto& x : v) {
    cout << x << endl;
}
```

✔ 避免拷贝
✔ 不修改元素

---

### 4️⃣ 可修改版本

```cpp
for (auto& x : v) {
    x *= 2;
}
```

---

## 7.6 智能指针初识（重点）

### 1️⃣ 为什么需要智能指针

传统指针问题：

* 忘记 `delete`
* 重复释放
* 异常安全差

> 智能指针 = **RAII + 自动释放资源**

---

### 7.6.1 `unique_ptr`（独占所有权）

#### 特点

* **一个资源只能被一个指针拥有**
* 不可拷贝
* 可移动

```cpp
#include <memory>

std::unique_ptr<int> p = std::make_unique<int>(10);
```

❌ 拷贝

```cpp
auto p2 = p; // 编译错误
```

✔ 移动

```cpp
auto p2 = std::move(p);
```

#### 使用场景：

> 资源唯一所有者（文件、锁、对象）

---

### 7.6.2 `shared_ptr`（共享所有权）

#### 特点

* 引用计数
* 最后一个释放时销毁资源

```cpp
std::shared_ptr<int> p1 = std::make_shared<int>(10);
std::shared_ptr<int> p2 = p1;
```

```cpp
cout << p1.use_count();  // 引用计数
```

#### 使用场景：

> 多对象共享同一资源

---

### ⚠ 注意：循环引用问题

```cpp
class A {
public:
    std::shared_ptr<B> b;
};
class B {
public:
    std::shared_ptr<A> a;
};
```

👉 会导致 **内存泄漏**

✔ 解决方案：`weak_ptr`（后续章节）

---

## 7.7总结

| 特性         | 作用     |
| ---------- | ------ |
| auto       | 减少类型噪音 |
| nullptr    | 指针安全   |
| enum class | 强类型枚举  |
| constexpr  | 编译期计算  |
| 范围 for     | 简洁遍历   |
| unique_ptr | 独占资源   |
| shared_ptr | 共享资源   |

---

# 八、STL 核心工具（Standard Template Library）

STL 是 C++ 标准库中最重要的组成部分，其核心目标是：

> **数据结构与算法解耦，通过迭代器连接**

---

## 8.1 STL 设计思想（必须理解）

### 1️⃣ 三大核心组件

| 组件            | 作用   |
| ------------- | ---- |
| 容器（Container） | 存放数据 |
| 迭代器（Iterator） | 访问数据 |
| 算法（Algorithm） | 操作数据 |

**算法不关心容器类型，只关心迭代器**

---

### 2️⃣ 泛型编程思想

* 使用模板
* 类型在编译期确定
* 零运行时开销（相对）

```cpp
template<typename It>
void mySort(It begin, It end);
```

---

### 3️⃣ 设计原则总结

> * 高内聚，低耦合
> * 编译期多态（模板）
> * 性能优先
> * 通用 + 可复用

---

## 8.2 容器（重点）

### 8.2.1 `string`

#### 特点

* 动态字符串
* 自动管理内存
* 支持 `+`、`[]`

```cpp
string s = "hello";
s += " world";
```

📌 推荐替代 `char[]`

---

### 8.2.2 `vector`（最常用）

#### 特点

* 连续内存
* 支持随机访问
* 尾部插入高效

```cpp
vector<int> v = {1,2,3};
v.push_back(4);
```

#### 注意

* 扩容会导致迭代器失效
* 中间插入/删除代价高

---

### 8.2.3 `deque`

#### 特点

* 双端队列
* 头尾插入都高效
* 非连续内存

```cpp
deque<int> d;
d.push_front(1);
d.push_back(2);
```

---

### 8.2.4 `queue`（适配器）

#### 特点

* FIFO
* 不能随机访问
* 基于 `deque`

```cpp
queue<int> q;
q.push(1);
q.pop();
```

---

### 8.2.5 `list`

#### 特点

* 双向链表
* 任意位置插入/删除快
* 不支持随机访问

```cpp
list<int> l;
l.push_back(1);
```

迭代器不会失效（除被删除元素）

---

### 8.2.6 `map` / `set`

#### 特点

* 红黑树（有序）
* 查找/插入：O(log n)

```cpp
map<string,int> m;
m["Tom"] = 18;
```

* `map`：key-value
* `set`：只有 key

#### 无序版本

```cpp
unordered_map
unordered_set
```

✔ 哈希表
✔ 平均 O(1)

---

### 8.2.7 容器选型速查表

| 需求     | 推荐            |
| ------ | ------------- |
| 随机访问   | vector        |
| 频繁头尾插入 | deque         |
| 任意位置插删 | list          |
| 有序查找   | map / set     |
| 高性能查找  | unordered_map |

---

## 8.3 迭代器（Iterator）

### 1️⃣ 什么是迭代器

> **容器与算法之间的桥梁**

```cpp
vector<int>::iterator it = v.begin();
```

---

### 2️⃣ 迭代器分类（了解）

| 类型   | 支持操作   |
| ---- | ------ |
| 输入   | 读      |
| 输出   | 写      |
| 前向   | ++     |
| 双向   | ++ --  |
| 随机访问 | + - [] |

`vector` 支持随机访问

`list` 只支持双向

---

### 3️⃣ 迭代器失效（工程重点）

| 容器     | 情况         |
| ------ | ---------- |
| vector | 扩容 / erase |
| deque  | 头尾操作       |
| list   | 几乎不失效      |

---

## 8.4 常用算法（重点）

### 1️⃣ 查找类

```cpp
find(v.begin(), v.end(), 10);
```

---

### 2️⃣ 排序类

```cpp
sort(v.begin(), v.end());
```

只能用于 **随机访问迭代器**

---

### 3️⃣ 遍历类

```cpp
for_each(v.begin(), v.end(), func);
```

---

### 4️⃣ 统计 / 判断

```cpp
count(v.begin(), v.end(), 3);
any_of(v.begin(), v.end(), pred);
```

---

### 5️⃣ 修改类

```cpp
reverse(v.begin(), v.end());
remove(v.begin(), v.end(), 5);
```

`remove` 只是移动元素，不真的删除（erase-remove 惯用法）

```cpp
v.erase(remove(v.begin(), v.end(), 5), v.end());
```

---

## 8.5 Lambda 表达式（现代 C++ 核心）

### 1️⃣ Lambda 是什么

> **匿名函数对象**

---

### 2️⃣ 基本语法

```cpp
[capture](params) -> return_type {
    body
};
```

---

### 3️⃣ 示例

```cpp
auto add = [](int a, int b) {
    return a + b;
};
```

---

### 4️⃣ 捕获列表（重点）

```cpp
int x = 10;

auto f1 = [x]() { cout << x; };     // 值捕获
auto f2 = [&x]() { x++; };          // 引用捕获
auto f3 = [=]() {};                 // 捕获全部（值）
auto f4 = [&]() {};                 // 捕获全部（引用）
```

#### 工程中优先：

```cpp
[&] 或 [const auto&]
```

---

### 5️⃣ Lambda + STL 算法（黄金组合）

```cpp
sort(v.begin(), v.end(), [](int a, int b) {
    return a > b;
});
```

---

# 九、模板与泛型（Templates & Generic Programming）

模板是 C++ 实现**泛型编程（Generic Programming）**的核心机制，其本质是：

> **在编译期生成代码，而不是在运行期做类型判断**

---

## 9.1 函数模板（Function Template）

### 9.1.1 为什么需要函数模板

没有模板时：

```cpp
int add(int a, int b);
double add(double a, double b);
```

问题：

* 代码重复
* 可维护性差

---

### 9.1.2 函数模板的基本语法

```cpp
template<typename T>
T add(T a, T b) {
    return a + b;
}
```

使用：

```cpp
add<int>(1, 2);      // 显式指定类型
add(1, 2);           // 自动类型推导（推荐）
```

---

### 9.1.3 模板参数说明

```cpp
template<class T>
```

* `typename` 与 `class` **等价**
* 现代 C++ 推荐 `typename`

---

### 9.1.4 模板类型推导规则（重点）

```cpp
add(1, 2);      // T = int
add(1.0, 2.0);  // T = double
```

❌ 不允许隐式类型转换

```cpp
add(1, 2.5);  // 编译错误
```

✔ 解决方式

```cpp
add<double>(1, 2.5);
```

---

### 9.1.5 函数模板 vs 普通函数（非常重要）

```cpp
int add(int a, int b);  // 普通函数
```

规则：

> **如果普通函数和模板都能匹配，优先调用普通函数**

---

### 9.1.6 模板特化（了解 → 重要）

#### 全特化

```cpp
template<>
int add<int>(int a, int b) {
    return a + b + 100;
}
```

---

#### 偏特化（函数模板不支持）

* ❌ 函数模板 **不支持偏特化**
* ✔ 类模板支持偏特化（下一节）

---

### 9.1.7 模板的本质（必懂）

* 编译期代码生成
* 不是真正的函数
* 编译器为 **每个类型生成一份代码**

**代码膨胀** 是模板的代价

---

## 9.2 类模板（Class Template）

### 9.2.1 类模板基本语法

```cpp
template<typename T>
class Box {
public:
    T value;
    Box(T v) : value(v) {}
};
```

使用：

```cpp
Box<int> b1(10);
Box<string> b2("hello");
```

**类模板必须显式指定类型**

---

### 9.2.2 类模板成员函数定义位置（重点）

#### ✔ 推荐：全部写在头文件

```cpp
template<typename T>
class Box {
public:
    void show() {
        cout << value << endl;
    }
private:
    T value;
};
```

#### 原因：
模板在 **编译期实例化**，需要看到完整定义

---

### 9.2.3 类模板与继承

#### 派生类是普通类

```cpp
template<typename T>
class Base { };

class Derived : public Base<int> { };
```

---

#### 派生类也是模板

```cpp
template<typename T>
class Derived : public Base<T> { };
```

---

### 9.2.4 类模板默认参数

```cpp
template<typename T = int>
class MyClass { };
```

```cpp
MyClass<> a;        // T = int
MyClass<double> b;
```

---

### 9.2.5 类模板特化（重点）

#### 全特化

```cpp
template<>
class Box<char> {
public:
    void show() {
        cout << "char box" << endl;
    }
};
```

---

#### 偏特化（类模板独有）

```cpp
template<typename T>
class Box<T*> {
public:
    void show() {
        cout << "pointer box" << endl;
    }
};
```

STL 内部大量使用偏特化

---

### 9.2.6 类模板中的静态成员（易错点）

```cpp
template<typename T>
class Counter {
public:
    static int cnt;
};

template<typename T>
int Counter<T>::cnt = 0;
```

**每种类型一份静态成员**

---

## 9.3 工程实践建议

### ✔ 推荐

* 模板用于 **算法 / 容器 / 通用工具**
* 结合 `auto`、`constexpr`
* 类型推导优先

### ❌ 不推荐

* 模板嵌套过深
* 模板错误信息难以理解的写法
* 用模板替代一切继承

---

## 9.4 总结

| 对比项    | 函数模板 | 类模板 |
| ------ | ---- | --- |
| 是否自动推导 | ✔    | ❌   |
| 是否偏特化  | ❌    | ✔   |
| 使用频率   | 高    | 高   |
| 实例化时机  | 调用时  | 使用时 |

---

# 十、C++ 其他 —— 谓词与函数对象

> **核心一句话：**
> STL 算法 = 容器 + 迭代器 + **谓词 / 函数对象**

---

## 10.1 谓词（Predicate）

### 10.1.1 什么是谓词

**谓词 = 返回 `bool` 的可调用对象**

它可以是：

* 普通函数
* 函数对象（仿函数）
* Lambda 表达式

📌 STL 中常用于 **判断 / 过滤 / 比较**

---

### 10.1.2 一元谓词（Unary Predicate）

#### 定义

* **参数：1 个**
* **返回值：bool**

```cpp
bool isEven(int x) {
    return x % 2 == 0;
}
```

---

#### STL 使用示例

```cpp
vector<int> v = {1,2,3,4,5};

auto it = find_if(v.begin(), v.end(), isEven);
```

---

#### Lambda 形式（最常用）

```cpp
auto it = find_if(v.begin(), v.end(),
    [](int x) {
        return x > 3;
    }
);
```

---

#### 常见使用场景

| 算法          | 说明        |
| ----------- | --------- |
| `find_if`   | 查找满足条件的元素 |
| `count_if`  | 统计满足条件的个数 |
| `remove_if` | 按条件移除     |

---

### 10.1.3 二元谓词（Binary Predicate）

#### 定义

* **参数：2 个**
* **返回值：bool**

```cpp
bool cmp(int a, int b) {
    return a > b;   // 降序
}
```

---

#### STL 使用示例

```cpp
sort(v.begin(), v.end(), cmp);
```

---

#### Lambda 形式（工程首选）

```cpp
sort(v.begin(), v.end(),
    [](int a, int b) {
        return a > b;
    }
);
```

---

#### 常见使用场景

| 算法            | 用途   |
| ------------- | ---- |
| `sort`        | 排序   |
| `stable_sort` | 稳定排序 |
| `max_element` | 查最大  |
| `min_element` | 查最小  |

---

## 10.2 内建函数对象（Built-in Functors）

### 10.2.1 什么是函数对象（仿函数）

**重载了 `operator()` 的类对象**

```cpp
struct Add {
    int operator()(int a, int b) const {
        return a + b;
    }
};
```

使用：

```cpp
Add add;
cout << add(1, 2);  // 像函数一样使用
```

📌 STL 内部大量使用函数对象，而不是普通函数。

---

### 10.2.1 算术仿函数（Arithmetic Functors）

📌 头文件：

```cpp
#include <functional>
```

#### 常见算术仿函数

| 仿函数             | 含义 |
| --------------- | -- |
| `plus<T>`       | 加法 |
| `minus<T>`      | 减法 |
| `multiplies<T>` | 乘法 |
| `divides<T>`    | 除法 |
| `modulus<T>`    | 取模 |
| `negate<T>`     | 取负 |

---

#### 示例

```cpp
plus<int> add;
cout << add(10, 5);   // 15
```

---

### 10.2.2 关系仿函数（Relational Functors）

#### 常见关系仿函数

| 仿函数                | 含义 |
| ------------------ | -- |
| `equal_to<T>`      | 等于 |
| `not_equal_to<T>`  | 不等 |
| `greater<T>`       | 大于 |
| `less<T>`          | 小于 |
| `greater_equal<T>` | ≥  |
| `less_equal<T>`    | ≤  |

---

#### 示例（排序）

```cpp
sort(v.begin(), v.end(), greater<int>());
```

📌 等价于：

```cpp
[](int a, int b) { return a > b; }
```

---

### 10.2.3 逻辑仿函数（Logical Functors）

#### 常见逻辑仿函数

| 仿函数              | 含义 |
| ---------------- | -- |
| `logical_and<T>` | && |
| `logical_or<T>`  | || |
| `logical_not<T>` | !  |

---

#### 示例

```cpp
logical_not<bool> notOp;
cout << notOp(true);  // false
```

---

## 10.3 谓词 vs Lambda vs 仿函数（工程对比）

| 方式     | 可读性 | 灵活性 | 使用频率  |
| ------ | --- | --- | ----- |
| 普通函数   | 一般  | 低   | 低     |
| 仿函数    | 高   | 高   | 中     |
| Lambda | 非常高 | 非常高 | ⭐⭐⭐⭐⭐ |

**现代 C++：Lambda 是首选**

---

## 10.4 工程实践总结

### ✔ 推荐

* STL 算法 + Lambda
* 简单条件直接写 Lambda
* 复杂逻辑用命名仿函数

### ❌ 不推荐

* 滥用全局函数谓词
* 写语义不清的 Lambda
* 不理解比较规则就自定义排序

---

# 十一、工程实践（C++ Engineering Practice）

> **核心目标：**
> 把「能写代码」升级为「能维护、能构建、能调试的工程能力」

## 11.1 头文件设计（Header Design）

### 11.1.1 头文件的职责（必须牢记）

头文件 `.h / .hpp` 的作用：

> **声明接口，而不是实现逻辑**

#### 头文件中通常包含：

* 类声明
* 函数声明
* 模板定义
* 常量 / 枚举
* 内联函数（小函数）

#### 不应该包含：

* 复杂函数实现
* 全局变量定义
* `.cpp` 实现代码

---

### 11.1.2 头文件保护（必做）

#### 方式一：宏保护（通用）

```cpp
#ifndef PERSON_H
#define PERSON_H

class Person {
public:
    void speak();
};

#endif
```

---

#### 方式二：`#pragma once`（现代）

```cpp
#pragma once
```

📌 更简洁，但依赖编译器（现代编译器均支持）

---

### 11.1.3 头文件包含原则（工程重点）

#### ✔ 正确做法

```cpp
// person.h
#include <string>

class Person {
    std::string name;
};
```

#### ❌ 错误做法

```cpp
#include <iostream>  // ❌ 不必要
```

📌 原则：

> **头文件最小化依赖（Include What You Use）**

---

### 11.1.4 前向声明（降低耦合）

```cpp
class Engine;  // 前向声明

class Car {
    Engine* engine;  // 指针/引用即可
};
```

📌 好处：

* 减少编译时间
* 避免循环依赖

---

## 11.2 编译单元（Translation Unit）

### 11.2.1 什么是编译单元

> **一个 `.cpp` 文件 + 它所包含的所有头文件**

📌 每个 `.cpp`：

* 单独编译
* 生成一个 `.o` / `.obj` 文件

---

### 11.2.2 编译流程回顾

```text
.cpp → 预处理 → 编译 → 汇编 → .o → 链接 → 可执行文件
```

---

### 11.2.3 ODR（One Definition Rule）

> **一个符号在整个程序中只能有一个定义**

常见违规：

* 头文件中定义非 inline 函数
* 定义全局变量

#### 错误示例

```cpp
// header.h
int g = 10;  // ❌ 多重定义
```

#### 正确方式

```cpp
// header.h
extern int g;

// source.cpp
int g = 10;
```

---

### 11.2.4 inline 的工程意义

```cpp
inline int add(int a, int b) {
    return a + b;
}
```

📌 作用：

* 允许头文件中定义函数
* 避免多重定义
* 是否真的内联由编译器决定

---

## 11.3 CMake 基础（工程必备）

### 11.3.1 CMake 是什么

> **跨平台构建系统生成工具**

它生成：

* Makefile
* Ninja
* Visual Studio 工程

---

### 11.3.2 最小 CMake 示例（必会）

```cmake
cmake_minimum_required(VERSION 3.10)

project(my_app)

add_executable(my_app
    main.cpp
    person.cpp
)
```

---

### 11.3.3 添加头文件路径

```cmake
target_include_directories(my_app
    PRIVATE include
)
```

---

### 11.3.4 添加库

```cmake
add_library(mylib person.cpp)

target_link_libraries(my_app
    PRIVATE mylib
)
```

---

### 11.3.5 Debug / Release

```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
cmake -DCMAKE_BUILD_TYPE=Release ..
```

Debug：

* 启用调试符号
* 关闭优化

---

## 11.4 调试（GDB）

### 11.4.1 编译时开启调试信息

```bash
g++ -g main.cpp
```

没有 `-g`，gdb 基本没用

---

### 11.4.2 常用 gdb 命令（必背）

| 命令           | 作用       |
| ------------ | -------- |
| `gdb a.out`  | 启动调试     |
| `break main` | 设置断点     |
| `run`        | 运行       |
| `next`       | 单步（不进函数） |
| `step`       | 单步（进函数）  |
| `print x`    | 查看变量     |
| `backtrace`  | 查看调用栈    |
| `continue`   | 继续运行     |
| `quit`       | 退出       |

---

### 11.4.3 调试思想（比命令更重要）

调试不是“瞎点”，而是：

1. 复现问题
2. 缩小范围
3. 观察状态
4. 验证假设

---

## 11.5 总结

> * 头文件只声明接口
> * 一个 `.cpp` 就是一个编译单元
> * 遵守 ODR，避免多重定义
> * CMake 是现代 C++ 构建核心
> * 会用 gdb，才算能调工程

---