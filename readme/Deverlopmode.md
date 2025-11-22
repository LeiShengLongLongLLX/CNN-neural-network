# 开发总述

## 一、方案总概

### 1.1 Win11 + VSCode + SSH 远程开发

#### 优点

* **Ubuntu 上直接编译 + 运行 + 调试**
* 代码编辑体验好（VSCode 自动补全、跳转、语法提示）
* `ctrl + S` 保存即可在 Ubuntu 立即生效
* 不需要来回同步、提交、push、pull
* 支持：

  * C/C++ 语法分析
  * Makefile/CMake 调试
  * 终端集成
  * 设置断点调试（gdb）
* 适合长期开发、频繁修改、调试

本质上：
**像本地开发一样开发 Ubuntu 上的代码。**

---

#### 使用场景

* 工程在 Ubuntu 上跑
* 需要不断调试运行（CNN、嵌入式部署、RISC-V 交叉编译等）
* 项目文件多、更新频繁

几乎是最完美体验。

---

### 1.2 GitHub 作为代码同步工具
#### 优点

* 项目版本管理很清晰
* 开发、提交记录完整
* 换机器、换系统都能同步
* 团队协作也能扩展

#### 缺点

* **每次同步需要：add → commit → push → pull**
* 调试不方便（必须切换到 Ubuntu 上运行）

这个方案非常适合：

* 代码已经成熟
* 主要是在写新功能
* 需要版本记录

---

#### 1.3 总结

```
代码在 Ubuntu 上 → 用 Win11 VSCode SSH 编辑
完成后 → git commit & push 到 GitHub
```

```
VSCode(SSH) 写代码  
     ↓
Ubuntu 本地编译运行  
     ↓
成功  
     ↓
git push → GitHub
```
---

## 二、github作代码同步

使用 GitHub 作为“中介仓库”
也就是：

```
Win11 ←→ GitHub ←→ Ubuntu
```

两个系统都从 GitHub 拉代码、推代码，就能永远同步。

---

### 2.1 同步流程图

```
Win11 VSCode 编辑代码
       ↓ git push
     GitHub 仓库
       ↓ git pull
Ubuntu 服务器获取最新代码
```

反过来也一样：

```
Ubuntu 改代码 → push → Win11 pull
```

---

### 2.2环境准备

#### Win11 上：

* 确保已经有 Git
* VSCode 也能用
* 开发完成后：

```bash
git add .
git commit -m "your update information"
git push origin main
```

---

#### Ubuntu 上：

第一次使用：

```bash
git clone https://github.com/你的仓库.git
```

以后只需要：

```bash
git pull
```

就能更新到 Win11 的最新修改。

---

### 2.3 典型协作流程

#### ① Win11 开发完提交

```bash
git add .
git commit -m "fixed my code"
git push
```

#### ② 跑到 Ubuntu 获取最新代码

```bash
git pull
make
./lenet
```

#### ③ 如果在 Ubuntu 修改了代码

```bash
git add .
git commit -m "bug fix"
git push
```

#### ④ 回到 Win11 更新

```bash
git pull
```

---

#### 多人/多设备同步的关键重点: **遵循 Git 正常流程**

永远保持：

```
先 pull → 再改 → 再 commit → 再 push
```

这样就不会出冲突。

---

### 2.4 常见问题说明

#### 如果两边都改了，没有先 pull 就 push

Git 会阻止：

```
! [rejected] main -> main (non-fast-forward)
```

解决方法很简单：

```bash
git pull
# git pull 会解决本地代码仓库和远程代码仓库的冲突
git push
```

---

### 2.5 总结

Win11 和 Ubuntu 都独立有仓库

流程：

```
两边修改 → 每次先 pull → 然后 push
```

流程：

### Win11

```bash
git pull
git add .
git commit -m "update"
git push
```

### Ubuntu

```bash
git pull
git add .
git commit -m "update"
git push
```

这样多端不会冲突。

---


