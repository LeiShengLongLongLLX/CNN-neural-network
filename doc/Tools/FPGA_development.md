# FPGA开发

# 硬件资源总概

## 一、PS 和 PL 总概

### 1.1 PS（Processing System，处理系统）

**PS 是 SoC FPGA 芯片中集成的“硬核处理器系统”**，通常在芯片出厂时已经固定，用户无法修改其内部结构。

典型组成：

* CPU：ARM Cortex-A9 / A53 / R5 等
* 存储控制器：DDR、OCM
* 外设：UART、ETH、USB、SD、I2C、SPI
* 中断控制器、定时器

> 本质：**一个完整的 MCU / CPU 系统**

---

### 1.2 PL（Programmable Logic，可编程逻辑）

**PL 是 FPGA 的可编程硬件逻辑部分**，由 LUT、FF、BRAM、DSP 等资源组成。

PL 中可以实现：

* Verilog / VHDL 自定义逻辑
* 各类 IP 核（FIFO、DMA、AXI 外设）
* 硬件加速器（DSP / AI / CNN）

> 本质：**可并行运行的硬件电路**

---

### 1.3 PS 与 PL 的核心区别

| 对比维度 | PS               | PL             |
| ---- | ---------------- | -------------- |
| 实现方式 | 硬核（芯片内固定）        | 可编程逻辑          |
| 开发语言 | C / C++ / Python | Verilog / VHDL |
| 执行方式 | 顺序执行             | 并行执行           |
| 延迟   | 高                | 极低             |
| 适合任务 | 控制、调度、系统管理       | 高速计算、实时处理      |

---

### 1.4 典型 SoC FPGA 架构（以 Zynq 为例）

```
┌──────────────┐        ┌─────────────────────┐
│      PS      │  AXI   │         PL          │
│              ├────────►  自定义逻辑 / IP    │
│  ARM CPU     │        │  加速器 / 外设       │
│  DDR / 外设  │        │                     │
└──────────────┘        └─────────────────────┘
```

说明：

* PS 和 PL 通过 **AXI 总线** 互联
* PS 负责系统控制
* PL 负责高速并行处理

---

### 1.5 PS 与 PL 的通信方式

#### 1. AXI-Lite

* 低速
* 用于寄存器配置
* PS → PL 控制

#### 2. AXI-Stream

* 无地址
* 高速数据流
* 常用于视频 / AI

#### 3. AXI-Full / HP

* 大数据传输
* 通常配合 DMA

#### 4. 中断

* PL → PS 事件通知

---

## 二、PL（Programmable Logic，可编程逻辑）部分

### 2.1 PL 的定义与定位

**PL（Programmable Logic）** 是 FPGA 或 SoC FPGA 中的 **可编程硬件逻辑区域**，由大量可配置的硬件资源组成，用户可以通过 **Verilog / VHDL** 描述电路结构，并下载到芯片中运行。

**本质特征**：

* 并行硬件
* 时钟驱动
* 实时响应
* 结构可定制

> PL ≠ 软件执行环境，而是“电路本身”。

---

### 2.2 PL 的资源构成总览（核心框架）

```
PL（Programmable Logic）
├─ 逻辑资源
│   ├─ LUT（Look-Up Table）
│   ├─ FF（Flip-Flop）
│   └─ Carry Chain（进位链）
├─ 存储资源
│   ├─ BRAM（Block RAM）
│   ├─ 分布式 RAM（LUT RAM）
│   └─ UltraRAM（部分高端器件）
├─ 运算资源
│   └─ DSP Slice
├─ 结构与组织资源
│   ├─ CLB / Slice
│   └─ Routing（互连）
├─ 时钟与复位资源
│   ├─ Global Clock Network
│   ├─ PLL / MMCM
│   └─ Clock Enable / Reset
├─ IO 与接口资源
│   ├─ IOB / IO Bank
│   ├─ IDELAY / ODELAY
│   └─ 高速收发器（GT）
└─ 专用硬核 / 专用模块
    ├─ DDR PHY / Controller
    ├─ PCIe / Ethernet Hard IP
    └─ XADC 等
```

---

### 2.3 逻辑资源（PL 的基础）

#### 1. LUT（Look up table 查找表）

**作用**：

* 实现组合逻辑
* 可配置为 ROM / 分布式 RAM / 移位寄存器

**特点**：

* FPGA 的最小组合逻辑单元
* 常见为 6-input LUT

---

#### 2. FF（Flip Flop 触发器）

**作用**：

* 存储状态
* 构成寄存器、流水线

**特点**：

* 与 LUT 紧密耦合
* 决定时序性能

---

#### 3. Carry Chain（进位链）

**作用**：

* 快速实现加法器、计数器、比较器

**意义**：

* 显著降低延迟
* 节省 LUT

---

### 2.4 存储资源

#### 4. BRAM（Block RAM）

**作用**：

* 大容量存储（IMEM / DMEM / FIFO / Cache）

**特点**：

* 固定大小（如 18Kb / 36Kb）
* 支持双端口

---

#### 5. 分布式 RAM（LUT RAM）

**作用**：

* 小容量高速 RAM

**适合场景**：

* FIFO
* 寄存器堆
* Cache Tag

---

#### 6. UltraRAM（部分器件）

**特点**：

* 超大容量
* 面向高端应用（AI / 视频）

---

### 2.5 运算资源

#### 7. DSP Slice

**作用**：

* 乘法
* MAC（乘加）
* 累加

**特点**：

* 高性能
* 低功耗
* AI / DSP 核心资源

---

### 2.6 结构与组织资源

#### 8. CLB（Configurable Logic Block）

**定义**：

* FPGA 中承载 LUT、FF 的物理逻辑单元

**说明**：

* CLB 不是独立资源类别
* 是逻辑资源的组织形式

---

#### 9. Routing（互连资源）

**作用**：

* 连接所有逻辑、存储、DSP 单元

**意义**：

* 决定布局布线是否成功
* 影响时序和可实现性

---

### 2.7 时钟与复位资源（性能关键）

#### 10. Clock Network

* 全局时钟
* 低 skew

#### 11. PLL / MMCM

* 倍频 / 分频
* 相位调整
* 抖动抑制

---

### 2.8 IO 与接口资源

#### 12. IOB（IO Block）

**作用**：

* FPGA 对外接口

**支持**：

* LVCMOS / LVDS
* DDR IO

---

#### 13. 高速收发器（GT）

**作用**：

* SerDes

**应用**：

* PCIe
* SATA
* 高速以太网

---

### 2.9 专用硬核 / 专用模块

**特点**：

* 位于 PL 区域
* 非 LUT 实现
* 性能高、资源占用低

示例：

* DDR 控制器
* PCIe Hard IP
* Ethernet MAC

---

### 2.10 PL 的设计特性总结

| 特性  | 说明        |
| --- | --------- |
| 并行性 | 所有逻辑同时运行  |
| 实时性 | 时钟到即响应    |
| 可定制 | 架构完全由用户决定 |
| 高性能 | 适合加速与实时计算 |

---

## 三、PS（Processing System 处理系统）部分

### 3.1 PS 的定义与定位

**PS（Processing System）** 是 SoC FPGA 芯片中 **预集成的硬核处理系统**，由 CPU、存储控制器和大量片上外设组成。

**核心特点**：

* 硬核实现（出厂固定）
* 运行软件（C / C++ / OS）
* 负责系统控制与管理

> 可以把 PS 看成：
> **“一颗完整的 MCU / 应用处理器，被集成进 FPGA 芯片中”**

---

### 3.2 PS 的资源构成总览（框架）

```
PS（Processing System）
├─ 处理器资源
│   ├─ 应用处理器（APU）
│   └─ 实时处理器（RPU，部分器件）
├─ 存储与存储控制
│   ├─ DDR Controller
│   ├─ OCM（片上存储）
│   ├─ Cache（L1 / L2）
|   └─ SD Controller / QSPI Controller / eMMC Controller 
├─ 片上外设
│   ├─ UART / SPI / I2C
│   ├─ Ethernet / USB / SD /
│   └─ GPIO / CAN
├─ 中断与定时
│   ├─ GIC
│   └─ Timers / Watchdog
├─ 系统与安全资源
│   ├─ BootROM
│   ├─ TrustZone / Secure Boot
│   └─ Reset / Clock Control
├─ PS–PL 接口资源
│   ├─ AXI GP / HP / ACP
│   └─ 中断接口
└─ 电源与时钟管理
    ├─ PLL
    └─ Power Domains
```

---

### 3.3 处理器资源（PS 的核心）

#### 1. 应用处理器（APU）

* ARM Cortex-A 系列（A9 / A53 / A72）
* 支持 MMU
* 可运行 Linux / Android

用途：

* 操作系统
* 网络通信
* 文件系统
* 高层业务逻辑

---

#### 2. 实时处理器（RPU，部分器件）

* ARM Cortex-R5 / R52
* 无 MMU（或轻量）
* 硬实时

用途：

* 实时控制
* 安全监控
* 协处理任务

---

### 3.4 存储与存储控制资源

#### 3. DDR 控制器

* 连接外部 DDR3 / DDR4 / LPDDR
* 高带宽
* 硬核实现

---

#### 4. OCM（On-Chip Memory）

* 片上 SRAM
* 容量较小
* 延迟低

常用于：

* 启动代码
* 关键数据

#### 5. 其它存储控制器

| 名称                 | 在哪        | 性质   |
| ------------------ | --------- | ---- |
| QSPI Controller    | **PS 内部** | 硬核外设 |
| SD/eMMC Controller | **PS 内部** | 硬核外设 |
| NAND Controller    | **PS 内部** | 硬核外设 |
| DDR Controller     | **PS 内部** | 硬核外设 |


---

#### 5. Cache 体系

* L1 Cache（指令 / 数据）
* L2 Cache（共享）

作用：

* 提升 CPU 性能

---

### 3.5 片上外设资源（非常丰富）

#### 6. 通用通信外设

* UART
* SPI
* I2C
* GPIO

---

#### 7. 高级外设

* Ethernet MAC
* USB OTG
* SD / eMMC
* CAN

> PS 外设 ≈ 高性能 MCU + 应用处理器

---

### 3.6 中断与定时资源

#### 8. 中断控制器（GIC）

* 管理 PS 内部中断
* 接收 PL 侧中断

---

#### 9. 定时器 / 看门狗

* System Timer
* Private Timer
* Watchdog

---

### 3.7 系统与安全资源

#### 10. BootROM

* 上电即执行
* 固定在芯片内

功能：

* 选择启动介质
* 加载 FSBL

---

#### 11. 安全机制（部分器件）

* Secure Boot
* TrustZone
* AES / RSA 加密

---

### 3.8 PS–PL 接口资源（协同关键）

#### 12. AXI 接口类型

| 接口      | 方向         | 用途        |
| ------- | ---------- | --------- |
| AXI GP  | PS → PL    | 控制寄存器     |
| AXI HP  | PL → DDR   | 高速数据      |
| AXI ACP | PL ↔ Cache | Cache 一致性 |

---

#### 13. PS–PL 中断

* PL 触发
* PS 响应

---

### 3.9 电源与时钟管理

#### 14. PS PLL / Clock Control

* CPU 时钟
* 外设时钟

#### 15. 电源域管理

* 支持低功耗
* 模块级控制

---

### 3.10 PS 与 PL 的分工总结

| 项目   | PS      | PL      |
| ---- | ------- | ------- |
| 处理方式 | 顺序执行    | 并行执行    |
| 开发方式 | 软件      | 硬件      |
| 典型任务 | 控制 / OS | 加速 / 实时 |

---

## 四、AXI

### 4.1 AXI 是什么？

**AXI（Advanced eXtensible Interface）** 是 ARM 提出的 **AMBA 总线规范之一**，是一种 **高性能、可扩展、面向 SoC 的片上总线协议**。

在 FPGA / SoC FPGA 中：

* **PS 与 PL 通信** 用 AXI
* **CPU ↔ 外设 / 加速器** 用 AXI
* **DMA / DDR / 高速数据通路** 用 AXI

> 一句话：**AXI 是 SoC 里“数据和控制流动的高速公路”**

---

#### AMBA家族总结表
| AMBA 代际 | 规范名称            | 英文全称                             | 中文直译       | 性能级别 | 典型用途                |
| ------- | --------------- | -------------------------------- | ---------- | ---- | ------------------- |
| AMBA 2  | **APB**         | Advanced Peripheral Bus          | 高级外设总线     | 低    | UART / GPIO / Timer |
| AMBA 2  | **AHB**         | Advanced High-performance Bus    | 高级高性能总线    | 中    | MCU 外设 / SRAM       |
| AMBA 3  | **AXI3**        | Advanced eXtensible Interface v3 | 高级可扩展接口 v3 | 高    | 早期 SoC 主干           |
| AMBA 4  | **AXI4**        | Advanced eXtensible Interface v4 | 高级可扩展接口 v4 | 高    | SoC / PS–PL         |
| AMBA 4  | **AXI4-Lite**   | AXI4 Lightweight                 | 轻量 AXI4 接口 | 中低   | 寄存器控制               |
| AMBA 4  | **AXI4-Stream** | AXI4 Streaming Interface         | AXI4 流接口   | 高    | 视频 / CNN / DSP      |
| AMBA 5  | **AXI5**        | Advanced eXtensible Interface v5 | 高级可扩展接口 v5 | 很高   | 新一代 SoC             |
| AMBA 5  | **CHI**         | Coherent Hub Interface           | 一致性枢纽接口    | 极高   | 多核 CPU / Cache      |

---

### 4.2 AXI 在 SoC 中的位置

```
CPU / PS
   │
   │  AXI
   ▼
互连（Interconnect / SmartConnect）
   │
   ├── AXI-Lite 外设（寄存器）
   ├── AXI-Full 加速器 / DMA
   └── AXI-Stream 数据通路
```

AXI 本身 **不是一根线**，而是一套 **通信规则 + 信号集合**。

---

### 4.3 AXI 的设计

#### 1. 设计目标
AXI 解决的问题：

* 高带宽
* 低延迟
* 支持并发
* 适合多主多从

关键设计思想：

* **读写分离**
* **地址与数据分离**
* **多通道并行**

---

#### 2. 五个独立通道

AXI 把一次访问拆成 **5 个独立通道**：

| 通道 | 名称             | 作用  |
| -- | -------------- | --- |
| AW | Write Address  | 写地址 |
| W  | Write Data     | 写数据 |
| B  | Write Response | 写响应 |
| AR | Read Address   | 读地址 |
| R  | Read Data      | 读数据 |

> **读和写完全独立、可并行**

---

#### 3. 握手机制（VALID / READY）

所有 AXI 通道都遵循：

```
VALID = 1  → 主设备数据有效
READY = 1  → 从设备准备好接收
VALID & READY = 1 → 传输完成
```

特点：

* 解耦发送方与接收方
* 支持不同速率模块互联

---

#### 4. AXI 的突发传输（Burst）

AXI 支持 **一次地址，多拍数据**：

* Burst Length
* Burst Type（INCR / FIXED / WRAP）

优势：

* 大幅提高 DDR / DMA 效率

---

### 4.4 AXI 的三大常用类型

#### 1️⃣ AXI4-Full（完整 AXI）

**特点**：

* 支持地址 + 数据
* 支持 Burst
* 高带宽

**应用**：

* DDR
* DMA
* 高速加速器

---

#### 2️⃣ AXI4-Lite（轻量 AXI）

**特点**：

* 无 Burst
* 低资源
* 简单寄存器访问

**应用**：

* 外设寄存器
* 控制接口

---

#### 3️⃣ AXI4-Stream（流接口）

**特点**：

* 无地址
* 连续数据流
* 高吞吐

**常用信号**：

* TVALID
* TREADY
* TDATA
* TLAST

**应用**：

* 视频
* CNN 数据流
* DSP Pipeline

---

### 4.5 AXI 主从设备（Master / Slave）与互联

#### 1. Master / Slave
| 角色     | 含义              |
| ------ | --------------- |
| Master | 发起访问（CPU / DMA） |
| Slave  | 响应访问（外设 / 存储）   |

在 Zynq 中：

* PS CPU：AXI Master
* PL 外设：AXI Slave

---

#### 2. AXI Interconnect（互连）

作用：

* 多 Master ↔ 多 Slave
* 仲裁 / 地址映射 / 数据转发

Xilinx 中常见：

* AXI Interconnect
* SmartConnect

---

### 4.6 AXI 与 PS–PL 的关系（你最关心的）

| 接口       | 方向      | 类型          |
| -------- | ------- | ----------- |
| PS → PL  | AXI GP  | Lite / Full |
| PL → DDR | AXI HP  | Full        |
| Cache 一致 | AXI ACP | Full        |

---

## 五、常见 FPGA 不同层次型号的资源配置

### 5.1 “资源配置层次”分类总概

FPGA 厂商型号众多，在**资源配置呈现明显的层次化规律**，工程上通常按：

* 教学 / 入门级
* 中低端通用型
* 中高端性能型
* 高端 / SoC FPGA

来分类

---

### 5.2 FPGA 资源总览

| 大类     | 典型资源                  |
| ------ | --------------------- |
| 逻辑资源   | LUT、FF、Logic Cell     |
| 存储资源   | BRAM、URAM（部分高端）       |
| 计算资源   | DSP Slice             |
| 时钟资源   | PLL、MMCM、Clock Buffer |
| I/O 资源 | GPIO、LVDS、SerDes      |
| 系统资源   | AXI、NoC、硬核控制器（SoC）    |

---

### 5.3 不同层次 FPGA 资源配置对比表

#### 1️⃣ 教学 / 入门级 FPGA（低端）

| 项目                | 典型配置范围            |
| ----------------- | ----------------- |
| LUT / Logic Cells | 几千 ~ 2 万          |
| FF                | 与 LUT 同量级         |
| BRAM              | 少量（几十 KB ~ 几百 KB） |
| DSP               | 0 ~ 少量（< 40）      |
| PLL               | 1 ~ 2             |
| 高速接口              | ❌                 |
| 典型应用              | 教学、基础实验、软核 MCU    |

示例：

* Xilinx Spartan-6 / Artix-7（小型号）
* Intel Cyclone IV / V（低配）

---

#### 2️⃣ 中低端通用 FPGA（主流工程）

| 项目         | 典型配置范围          |
| ---------- | --------------- |
| LUT        | 2 万 ~ 20 万      |
| FF         | 与 LUT 接近        |
| BRAM       | 几百 KB ~ 数 MB    |
| DSP        | 数十 ~ 数百         |
| PLL / MMCM | 多个              |
| 高速串行       | 少量（6–12 Gbps）   |
| 典型应用       | 工业控制、图像处理、AI 边缘 |

示例：

* Xilinx Artix-7 / Kintex-7（低型号）
* Intel Cyclone 10 / Arria 10（低配）

---

#### 3️⃣ 中高端性能型 FPGA

| 项目        | 典型配置范围        |
| --------- | ------------- |
| LUT       | 20 万 ~ 100 万+ |
| FF        | 与 LUT 同量级     |
| BRAM      | 数 MB ~ 十几 MB  |
| URAM      | ✔（部分）         |
| DSP       | 数百 ~ 数千       |
| 高速 SerDes | 10–28 Gbps    |
| 典型应用      | 通信、雷达、AI 加速   |

示例：

* Xilinx Kintex UltraScale
* Intel Arria 10 / Agilex（低配）

---

#### 4️⃣ 高端 FPGA / SoC FPGA

| 项目        | 典型配置范围         |
| --------- | -------------- |
| LUT       | 数十万 ~ 百万级      |
| FF        | 百万级            |
| BRAM      | 十几 MB          |
| URAM      | ✔ 大量           |
| DSP       | 上千             |
| 硬核 CPU    | ARM Cortex-A/R |
| DDR 控制器   | ✔              |
| 高速 SerDes | 28–112 Gbps    |
| 典型应用      | 数据中心、5G、自动驾驶   |

示例：

* Zynq-7000 / Zynq UltraScale+
* Intel Agilex SoC

---

### 5.4 存储资源随层次的变化规律

| 层次  | 片上存储特征            |
| --- | ----------------- |
| 低端  | 少量 BRAM           |
| 中端  | BRAM 为主           |
| 高端  | BRAM + URAM       |
| SoC | BRAM + URAM + DDR |

---

### 5.5 DSP 资源随层次变化

* 低端：几乎不用 DSP
* 中端：DSP 是性能核心
* 高端：DSP + AI 专用单元

---

### 5.6 总结

逻辑规模看 LUT，性能看 DSP，缓存看 BRAM/URAM，系统复杂度看是否带 PS。

---

# PL资源

## 逻辑
## 一、LUT

## 二、FF

## 存储

## 三、BRAM

## 四、UltraRAM(URAM)

## 五、LUT RAM 

## 四、DSP 

## 运算


## 五、

---

# PS资源




---
# 开发总概
## 一、FPGA开发流程

##