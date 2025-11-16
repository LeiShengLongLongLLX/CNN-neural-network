# 基于Linux系统上（Ubuntu）的RISC-V开发所需的IDE与工具链

- 主机开发平台：ubuntu 20.04
- 工具链：riscv64-unknown-elf-

## 一、交叉编译器

### 1. 以RISC-V为目标架构的编译器

> riscv64-unknown-elf-gcc

> riscv32-unknown-elf-gcc (疑似没有)

### 2.终端调试指令

> -nostdlib
 
> -fno-builtin
 
> -march=rv32g
 
> -mabi=ilp32
 
> -g
 
> -Wall

---

| 选项             | 作用              | 详细说明                                                                  |
| -------------- | --------------- | --------------------------------------------------------------------- |
| `-nostdlib`    | 不链接标准库          | 编译时不会自动链接 libc、libgcc 等标准库，通常用于裸机开发，因为裸机上没有操作系统或标准库。                  |
| `-fno-builtin` | 禁用内建函数          | 编译器默认会把一些常用函数（如 `memcpy`、`printf`）优化为内建指令，使用此选项可以避免这种优化。裸机开发时常用。      |
| `-march=rv32g` | 指定指令集架构         | `rv32g` 表示 32 位通用 RISC-V 指令集，包括整数指令（I）、乘除指令（M）、原子指令（A）、浮点指令（F/D）等扩展。  |
| `-mabi=ilp32`  | 指定 ABI（应用二进制接口） | `ilp32` 表示整数（int）、长整数（long）、指针（pointer）都是 32 位。ABI 决定函数调用约定、参数传递和栈布局。 |
| `-g`           | 生成调试信息          | 编译器会在输出的 ELF 文件中生成符号表和调试信息，方便用 gdb 调试。                                |
| `-Wall`        | 开启所有警告          | 提示潜在的代码问题，有助于写出更可靠的代码。                                                |



## 二、Binutils工具

### 2.1 汇编器（as）

> riscv64-unknown-elf-as

> riscv32-unknown-elf-as

#### 终端调试指令

假设有一个汇编源文件 `main.s`：

```asm
    .section .text
    .global _start
_start:
    li a0, 1      # 将 1 载入 a0
    li a1, 2      # 将 2 载入 a1
    add a2, a0, a1
```

在终端汇编：

```bash
# 对 RV32 架构
riscv32-unknown-elf-as -o main.o main.s

# 对 RV64 架构
riscv64-unknown-elf-as -o main.o main.s
```

解释：

* `-o main.o`：指定输出目标文件名。
* `main.s`：输入汇编源文件。

生成的 `main.o` 是 ELF 目标文件，接下来可用链接器生成可执行文件：

```bash
riscv32-unknown-elf-ld -o main.elf main.o
```

---

####  2. 常用调试选项

汇编器提供一些调试和信息输出选项，方便分析汇编结果：

| 选项          | 作用                                                  |
| ----------- | --------------------------------------------------- |
| `-g`        | 在目标文件中生成调试信息，可用于 gdb 调试                             |
| `-a=cdgst`  | 汇编详细信息输出（code, data, symbols, text），可查看每条汇编指令对应的机器码 |
| `--32`      | 强制使用 32 位指令模式（适用于 RV32）                             |
| `--64`      | 强制使用 64 位指令模式（适用于 RV64）                             |
| `-o <file>` | 指定输出文件名                                             |
| `-v`        | 输出版本信息及内部信息（有助于调试汇编器问题）                             |

示例：

```bash
# 汇编并输出详细信息
riscv32-unknown-elf-as -g -a=cdgst -o main.o main.s
```

---

#### 3. 与 GDB 配合调试

1. 汇编时生成调试信息：

```bash
riscv32-unknown-elf-as -g -o main.o main.s
riscv32-unknown-elf-ld -o main.elf main.o
```

2. 使用 GDB 调试：

```bash
riscv32-unknown-elf-gdb main.elf
(gdb) target sim   # 或使用 QEMU 模拟器
(gdb) break _start
(gdb) run
(gdb) step        # 单步执行
(gdb) info registers
```

这样可以看到寄存器值变化，验证汇编逻辑是否正确。

---

### 2.2 链接器（ld）

> riscv64-unknown-elf-ld

> riscv32-unknown-elf-ld

#### 终端调试指令

1. 基本链接命令

假设有目标文件 `main.o`：

```bash
# 链接生成 RV32 ELF 可执行文件
riscv32-unknown-elf-ld -o main.elf main.o

# 链接生成 RV64 ELF 可执行文件
riscv64-unknown-elf-ld -o main.elf main.o
```

* `-o main.elf`：指定输出可执行文件名
* `main.o`：输入目标文件
* 默认情况下，如果没有指定链接脚本，ld 会使用默认的内存布局，这在裸机开发中可能不准确，通常需要自定义 `.ld` 文件。

---

2. 常用调试和信息输出选项

| 选项                     | 作用                            |
| ---------------------- | ----------------------------- |
| `-Map=output.map`      | 生成映射文件，显示各个符号的地址和段信息，便于调试链接问题 |
| `-t`                   | 显示输入文件和输出文件的处理过程              |
| `-verbose`             | 输出详细链接信息，包括符号解析和段布局           |
| `-o <file>`            | 指定输出文件名                       |
| `--print-memory-usage` | 显示内存占用情况                      |
| `-r`                   | 生成可重定位目标文件，而不是最终可执行文件         |

---

##### 与汇编器和调试器配合示例

1. 汇编生成目标文件：

```bash
riscv32-unknown-elf-as -g -o main.o main.s
```

2. 链接生成 ELF 可执行文件：

```bash
riscv32-unknown-elf-ld -T link.ld -o main.elf main.o
```

* `-T link.ld`：指定链接脚本 `link.ld`，控制段的起始地址和布局。

3. 查看符号和段信息（调试链接结果）：

```bash
# 查看 ELF 符号表
riscv32-unknown-elf-nm main.elf

# 查看段信息
riscv32-unknown-elf-readelf -S main.elf
```

4. 使用 GDB 或 QEMU 调试：

```bash
riscv32-unknown-elf-gdb main.elf
(gdb) target sim   # 或使用 qemu-riscv32
(gdb) break _start
(gdb) run
(gdb) info files   # 查看链接的段地址
```

---

### 2.3 反汇编器（objdump）

> riscv64-unknown-elf-objdump

> riscv32-unknown-elf-objdump



## 三、QEMU模拟器

> qemu-system-riscv64 (系统模式)

> qemu-system-riscv32 (系统模式)

> qemu-user-riscv64 (用户模式) 

> qemu-user-riscv32 (用户模式) 

### 终端调试指令

#### 1. 直接运行 ELF 可执行文件

假设有 ELF 文件 `main.elf`：

```bash
# RV32 裸机
qemu-system-riscv32 -nographic -machine virt -kernel main.elf

# RV64 裸机
qemu-system-riscv64 -nographic -machine virt -kernel main.elf
```

#### 选项说明：

| 选项               | 作用                      |
| ---------------- | ----------------------- |
| `-nographic`     | 不启动图形界面，将串口输出映射到终端      |
| `-machine virt`  | 使用虚拟化硬件平台（适合裸机）         |
| `-kernel <file>` | 加载 ELF 可执行文件作为 CPU 启动程序 |
| `-s -S`          | 启用 GDB 远程调试（下文详细介绍）     |

---

#### 2. 与 GDB 远程调试结合

QEMU 可以作为 **GDB 远程服务器**，方便单步调试裸机程序：

步骤 1：启动 QEMU 并等待 GDB 连接

```bash
# RV32 示例
qemu-system-riscv32 -nographic -machine virt -kernel main.elf -S -s

# RV64 示例
qemu-system-riscv64 -nographic -machine virt -kernel main.elf -S -s
```

说明：

* `-S`：启动后暂停 CPU，不立即执行，等待 GDB 连接。
* `-s`：默认在 TCP 1234 端口开启 GDB 服务器。

---

步骤 2：使用 GDB 连接 QEMU

```bash
riscv32-unknown-elf-gdb main.elf  # 打开 GDB
(gdb) target remote :1234          # 连接 QEMU GDB 服务器
(gdb) break _start                 # 设置断点
(gdb) continue                     # 开始运行
(gdb) step                         # 单步执行
(gdb) info registers               # 查看寄存器
```

> 优点：
>
> * 可以在 QEMU 中运行程序，而不需要真实硬件
> * 可以单步调试、观察寄存器和内存
> * 可结合 `readelf`、`objdump` 分析 ELF 文件

---

#### 3. 常用 QEMU 调试选项

| 选项              | 作用                          |
| --------------- | --------------------------- |
| `-nographic`    | 串口输出到终端                     |
| `-serial stdio` | 等同于 `-nographic`，将串口输出映射到终端 |
| `-S`            | 启动后暂停 CPU，等待 GDB 连接         |
| `-s`            | 开启 GDB 远程调试端口（1234）         |
| `-d in_asm,cpu` | 输出汇编指令执行日志，方便调试 CPU 指令流     |
| `-D <file>`     | 将 QEMU 日志输出到指定文件            |
| `-machine virt` | 选择虚拟平台，适合裸机程序               |

---

#### 4. 典型裸机调试流程

1. 汇编/编译 → `.o` → 链接 → `.elf`
2. 启动 QEMU 并等待 GDB 连接：

```bash
qemu-system-riscv32 -nographic -machine virt -kernel main.elf -S -s
```

3. 打开 GDB 并连接：

```bash
riscv32-unknown-elf-gdb main.elf
(gdb) target remote :1234
```

4. 设置断点和单步调试：

```bash
(gdb) break _start
(gdb) continue
(gdb) step
(gdb) info registers
```

---

## 四、GDB调试器

> riscv64-unknown-elf-gdb (疑似不叫这个名字)

> riscv32-unknown-elf-gdb (疑似不叫这个名字)

> gdb-multiarch (用这个)

### 终端调试指令


---

## 五、IDE集成开发平台



