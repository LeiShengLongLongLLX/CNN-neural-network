# Xilinx（现 AMD）的FPGA开发

##  一、Vivado 工具链总概

Vivado 是 Xilinx（AMD）为 FPGA 与 SoC（如 Zynq、Versal 等）设计提供的**完整开发环境**。
它的工具链涵盖了从 **硬件电路设计 → 高层次综合 (HLS) → 系统软件开发 → 仿真验证 → 文档浏览** 的全过程。

---

### 1.1. **Vivado Design Suite**

**主要功能：**
FPGA 硬件设计的核心工具，用于 RTL 级开发（Verilog / VHDL），包括：

* **逻辑综合 (Synthesis)**：把 HDL 转成门级网表。
* **实现 (Implementation)**：布局布线、时序约束检查。
* **比特流生成 (Bitstream Generation)**：生成 `.bit` 或 `.bin` 文件，烧录到 FPGA。
* **IP Integrator**：图形化搭建系统（例如 Zynq SoC 的 PS+PL 架构）。
* **仿真与调试**：内置仿真器、ILA 调试。

> 生成的结果文件包括：
> * `.bit`（FPGA 配置文件）
> * `.hdf` / `.xsa`（系统描述文件，供 Vitis 使用）

---

### 1.2. **Vitis**

**主要功能：**
Vitis 是 **软件与系统级开发平台**，专为 Zynq / Versal 等嵌入式 SoC FPGA 设计。

* 支持 **C/C++、Python、OpenCL** 应用开发；
* 使用 Vivado 生成的 **硬件平台文件 (.xsa)**；
* 提供 **驱动程序与 BSP（Board Support Package）**；
* 可生成并部署可执行程序到 ARM 处理器上（PS 端）；
* 适合嵌入式 Linux 或裸机（Bare-metal）应用开发。

📌 简而言之：

> Vivado 做硬件 → Vitis 做软件。

---

### 1.3. **Vitis HLS (High-Level Synthesis)**

**主要功能：**
高层次综合工具，用于把 **C/C++ 或 SystemC** 转换成可综合的 RTL（Verilog/VHDL）。

特点：

* 支持 **算法级加速设计**；
* 自动插入流水线、并行化、接口协议（AXI 等）；
* 输出 IP 核，可导入到 Vivado 进行集成；
* 常用于加速计算模块（如图像处理、AI 推理）。

📌 举例：

> 你写一个 C 函数实现矩阵乘法 → Vitis HLS 自动转成硬件模块（IP） → Vivado 集成到设计中。

---

### 1.4. **Model Composer**

**主要功能：**
在 **MATLAB/Simulink** 环境中进行基于模型的 FPGA 设计。

* 图形化建模，不需直接写 HDL；
* 可与 Vitis HLS / Vivado 连接；
* 一键生成 FPGA 硬件模块；
* 支持与 MATLAB 数据流直接交互（仿真非常方便）。

📌 适合人群：

> 习惯 Simulink 流程的算法开发人员（例如信号处理、控制系统）。

---

### 1.5. **DocNav (Documentation Navigator)**

**主要功能：**
官方的 **文档导航工具**。

* 提供所有 AMD/Xilinx 工具的官方文档、用户指南、应用笔记；
* 支持离线与在线模式；
* 内置搜索、分类导航，非常适合查 Vivado/Vitis 的 UG（User Guide）编号文档。

📌 举例：

> 想查 “Vivado Constraints User Guide” → 打开 DocNav → 搜索 “UG903”。

---

### 1.6 工具链关系总结

| 工具                 | 作用层次    | 输入          | 输出              | 主要用途         |
| ------------------ | ------- | ----------- | --------------- | ------------ |
| **Vivado**         | 硬件设计    | HDL / IP    | Bitstream / XSA | 设计 FPGA 硬件系统 |
| **Vitis**          | 软件开发    | XSA 平台      | 可执行文件 (ELF)     | 编写并运行应用程序    |
| **Vitis HLS**      | 算法级硬件设计 | C/C++       | IP (RTL)        | 把算法转为硬件模块    |
| **Model Composer** | 模型级硬件设计 | Simulink 模型 | IP / HDL        | 图形化 FPGA 设计  |
| **DocNav**         | 文档浏览    | -           | -               | 查看官方技术资料     |

---

# Vitis





# Vitis HLS