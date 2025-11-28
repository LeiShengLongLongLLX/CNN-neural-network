# Linux Operation System
## 1. System directory
- Linux系统没有盘符的概念
- "/"是根目录

---

### 1.1 根目录下的文件结构
### 核心目录功能简介

| 目录名 | 全称/含义 | 主要用途 |
| :--- | :--- | :--- |
| **`bin`** | **Binary** | 存放所有**用户**都可使用的基本命令（如 `ls`, `cp`, `cat`, `bash`）。通常是 `/usr/bin` 的符号链接。 |
| **`boot`** | **Boot** | 存放系统**启动**所需的文件，如内核镜像 (`vmlinuz`)、引导加载程序 (`grub2`) 等。 |
| **`dev`** | **Devices** | 存放所有的**设备文件**（如硬盘 `/dev/sda`、终端 `/dev/tty`、随机数发生器 `/dev/urandom`）。 |
| **`etc`** | **Etcetera** (扩展配置) | 存放系统的**全局配置文件**（如网络配置、软件配置、用户数据库）。**非常重要**。 |
| **`home`** | **Home** | 普通用户的**家目录**所在地。每个用户有一个子目录（如 `/home/alice`），用于存放个人文件和配置。 |
| **`lib`** | **Libraries** | 存放系统**最基础的共享库**和内核模块。通常是 `/usr/lib` 的符号链接。 |
| **`lib64`** | **Libraries 64-bit** | 存放64位系统的基础共享库。通常是 `/usr/lib64` 的符号链接。 |
| **`media`** | **Media** | 系统自动挂载**可移动媒体**（如 U 盘、光盘）的标准位置。 |
| **`mnt`** | **Mount** | 用于**临时手动挂载**文件系统（如另一个硬盘分区、网络共享）的目录。 |
| **`opt`** | **Optional** | 用于安装**附加的第三方大型软件**（通常是一个独立套件）。 |
| **`proc`** | **Process** | 一个**虚拟文件系统**，以文件形式提供系统进程和内核信息的接口（如 CPU 信息、内存使用）。**不在硬盘上，是内存中的映射**。 |
| **`root`** | **Root** | **系统管理员 (root)** 的家目录。不同于 `/`，这是 root 用户的个人工作目录。 |
| **`run`** | **Run** | 存放自系统启动以来**正在运行的进程**的临时数据（如 PID 文件、套接字文件）。 |
| **`sbin`** | **System Binary** | 存放供**系统管理员**使用的基本管理命令（如 `fdisk`, `fsck`, `ifconfig`）。通常是 `/usr/sbin` 的符号链接。 |
| **`srv`** | **Service** | 存放由系统**提供服务**的数据（如网站文件 `/srv/www/`、FTP 资源）。 |
| **`sys`** | **System** | 一个**虚拟文件系统**，用于与内核交互、配置硬件设备和驱动。**不在硬盘上**。 |
| **`tmp`** | **Temporary** | 供所有用户存放**临时文件**的目录。系统重启后，这里的文件可能会被清除。 |
| **`usr`** | **Unix System Resources** | 存放**用户级别的程序和数据**，是仅次于 `/` 的第二重要的目录。包含大量的应用程序、库、文档等。 |
| **`var`** | **Variable** | 存放经常**变化(Variable)** 的数据，如日志文件、缓存、数据库文件、邮件队列等。 |

---

### 1.2 记忆技巧与分类

可以把这些目录分为几大类来帮助记忆：

1.  **系统启动必备**：`boot`, `bin`, `sbin`, `lib`, `lib64`, `dev`, `etc`
2.  **用户空间**：`home`, `root`
3.  **可变数据**：`var`, `tmp`
4.  **静态资源**：`usr`, `opt`
5.  **挂载点**：`media`, `mnt`
6.  **虚拟文件系统（内核接口）**：`proc`, `sys`, `run`

---

## 2. 指令结构

### 2.1 Shell 提示符
 **Shell 提示符**是由系统显示的一行信息，用来提示你：系统已经准备就绪，正在等待你输入命令。

---

我们可以把它拆解成几个部分来理解：

### 1. 整体结构：`[用户名@主机名 当前目录]权限标识符`
*   **`[` 和 `]`**： 只是两个括号，用来把前面的信息括起来，使其在视觉上成为一个整体，更容易阅读。
*   **`lxlei`**： 这是当前登录系统的**用户名**。表示你正以用户 “lxlei” 的身份在操作。
*   **`@`**： 只是一个分隔符号，读作英文的 “at”，表示“在...”。
*   **`localhost`**： 这是你当前登录的计算机的**主机名**。`localhost` 是一个特殊的主机名，它总是指向当前这台机器本身。如果你的计算机有自己设定的名字（比如 `my-pc`, `server-01`），这里就会显示那个名字。
*   **`/`**： 这是你当前所在的**工作目录**（也叫当前目录）。一个单独的斜杠 `/` 代表的是Linux文件系统的**根目录**，是所有文件和目录的起点。如果你在 `/home/lxlei` 目录下，这里显示的就是 `lxlei` 或完整的 `/home/lxlei`。
*   **`$`**： 这是**命令提示符**本身，它告诉你后面可以开始输入命令了。这个符号也暗示了当前用户的**权限等级**：
    *   **`$`**： 表示当前登录的是**普通用户**。
    *   **`#`**： 如果这里显示的是井号，则表示当前登录的是**超级用户（root）**。root用户拥有系统的最高权限，可以执行任何操作，需要非常小心。

---

### 2. 总结与举例

所以，`[lxlei@localhost /]$` 整句的意思是：

> **“你好，lxlei。你当前正登录在名为 ‘localhost’ 的这台机器上，并且你正处于系统的根目录 (/) 下。你现在是一个普通用户（$），可以开始输入你的命令了。”**

---

### 2.2 指令结构

Linux 命令通常遵循一个标准的模式，可以分解为以下几个部分：

**`command [options] [arguments]`**

*   **`command`**：命令本身，表示要执行的操作。
*   **`[options]`**：选项（也叫“标志”或“开关”），用于修改命令的行为。通常以 `-` 或 `--` 开头。
*   **`[arguments]`**：参数（也叫“操作数”），指定命令作用的对象，通常是文件名、目录名或用户名等。

**`[ ]` 方括号表示该部分是可选的**，不是所有命令都需要选项和参数。

命令+选项＋参数

---

### 2.2.1. 命令 (Command)

这是指令的核心，告诉系统你要“做什么”。它通常是一个可执行程序，可能位于以下位置之一：

*   **内置命令 (Shell Builtin)**：内置于 Shell 本身（如 `bash` 中的 `cd`, `pwd`, `echo`）。
*   **外部程序**：存储在文件系统特定目录（如 `/bin`, `/usr/bin`, `/sbin`）下的独立程序（如 `ls`, `vim`, `python`）。

你可以用 `type` 命令来查看一个命令是内置的还是外部的：
```bash
type cd   # 输出: cd is a shell builtin
type ls   # 输出: ls is /usr/bin/ls
```

---

### 2.2.2. 选项 (Options / Flags)

选项用于“如何”执行命令，即修改命令的默认行为。

*   **短选项 (Short Options)**：通常以**单个连字符 `-`** 开头，后跟一个字母。
    *   可以组合使用：`-a -l -h` 可以写成 `-alh`。
    *   例子：`ls -l` 中的 `-l` 表示以长格式列表显示。

*   **长选项 (Long Options)**：通常以**双连字符 `--`** 开头，后跟一个完整的单词。
    *   可读性更强，但不能组合。
    *   例子：`ls --all` 中的 `--all` 表示显示所有文件（包括隐藏文件）。它等同于短选项 `-a`。

**示例：**
```bash
# 使用短选项
ls -l -a      # 以长格式显示所有文件（包括隐藏文件）
ls -la        # 与上一行等价，选项合并了

# 使用长选项
ls --all --human-readable # 显示所有文件并以易读格式（K, M, G）显示大小
```

---

### 2.2.3. 参数 (Arguments)

参数指定命令操作的对象，即“对什么执行”。

*   参数可以是**文件名**、**目录名**、**用户名**、**主机名**等，具体取决于命令。
*   有些命令需要固定数量的参数，有些可以接受任意数量，有些则可以不需要参数。

**示例：**
```bash
# 参数是文件名
cat file1.txt file2.txt  # 将 file1.txt 和 file2.txt 的内容连接输出
rm old_file.txt          # 删除名为 old_file.txt 的文件

# 参数是目录名
ls /home                 # 列出 /home 目录下的内容
cd Documents             # 切换到当前目录下的 Documents 目录

# 参数是用户名
chown alice file.txt     # 将 file.txt 的所有者改为用户 alice
```

---

### 综合实例分析

让我们分解一些常见的复杂命令：

**示例 1: `tar`**
```bash
tar -czvf backup.tar.gz /home/user/Documents
```
*   **命令**: `tar` (归档工具)
*   **选项**:
    *   `-c`：**c**reate，创建新的归档文件。
    *   `-z`：使用 g**z**ip 压缩。
    *   `-v`：**v**erbose，显示详细处理过程。
    *   `-f`：指定归档**f**ilename。
*   **参数**:
    *   `backup.tar.gz`：是 `-f` 选项的参数，指定生成的归档文件名。
    *   `/home/user/Documents`：是命令的主要参数，指定要被归档的目录。

**示例 2: `find`**
```bash
find /var/log -name "*.log" -type f -mtime +7 -exec rm {} \;
```
*   **命令**: `find` (查找文件)
*   **参数/选项**:
    *   `/var/log`：查找的起始路径。
    *   `-name "*.log"`：选项，按名称搜索（所有 .log 结尾的文件）。
    *   `-type f`：选项，只查找普通文件。
    *   `-mtime +7`：选项，查找7天前修改过的文件。
    *   `-exec rm {} \;`：选项，对找到的每个文件执行 `rm` 命令。

---

### 其他重要概念和符号

1.  **空格分隔**：命令、选项、参数之间必须用**空格**分隔。
2.  **`--` 终止选项解析**：双连字符 `--` 用于告诉命令“后面的内容不再是选项”，即使它以 `-` 开头。这在处理特殊文件名的场景下非常有用。
    ```bash
    rm -- -my-weird-file.txt # 删除一个以短横线开头的文件
    ```
3.  **引用参数**：如果参数包含空格或特殊字符，需要用**引号**（单引号 `'` 或双引号 `"`）将其括起来。
    ```bash
    echo "Hello World"  # 正确
    mkdir 'My Folder'   # 创建一个包含空格的目录
    ```

---

### 总结

Linux 指令的结构可以概括为：**命令 [选项] [参数]**。

*   **命令**：要执行的操作。
*   **选项**：以 `-` 或 `--` 开头，决定操作的方式。
*   **参数**：操作的目标对象。
   
----

## 3. 常用指令
### 3.1 文件操作
增删改查
好的，这是在 Linux 系统中对文件进行增、删、改、查的核心终端命令大全。我将它们分门别类，并附上常用示例。

---

### 一、查（View & Find）：查看和查找文件

这是最常用的操作。

#### 1. `ls` - **列出目录内容**
   - `ls`：列出当前目录下的文件和目录（不显示隐藏文件）。
   - `ls -l`：以**长格式**列出，显示详细信息（权限、所有者、大小、修改时间）。
   - `ls -a`：列出**所有**文件，包括隐藏文件（以 `.` 开头的文件）。
   - `ls -la`：`-l` 和 `-a` 的组合
   - `ls -alh`：`-l`和`-a`和`-h`的组合
   - `ls /path/to/dir`：列出指定目录的内容。

#### 2. `cat` - **查看文件全部内容**
   - `cat file.txt`：在终端中一次性显示 `file.txt` 的全部内容。适合查看**短小**的文件。

#### 3. `less` / `more` - **分页查看文件内容**
   - `less file.txt`：**推荐使用**。分页显示文件内容，支持上下滚动、搜索（按 `/` 然后输入关键词）。
     - **退出：** 按 `q` 键。
     - **搜索后跳转：** 按 `n`（下一个匹配项），`N`（上一个匹配项）。
   - `more file.txt`：功能类似 `less`，但功能更简单。

#### 4. `head` / `tail` - **查看文件开头或结尾**
   - `head -n 10 file.txt`：查看文件的前10行。
   - `tail -n 10 file.txt`：查看文件的最后10行。
   - `tail -f logfile.log`：**实时追踪**日志文件的新内容，非常适合监控日志。按 `Ctrl+C` 退出。

#### 5. `find` - **高级查找**
   - `find /home -name "*.txt"`：在 `/home` 目录下**查找所有扩展名为 `.txt` 的文件**。
   - `find . -name "config*" -type f`：在当前目录（`.`）下查找**文件名以 `config` 开头**的**普通文件（`-type f`）**。
   - `find / -type f -size +100M`：在整个系统（`/`）中查找**大于100MB**的文件。

#### 6. `grep` - **在文件内容中查找**
   - `grep "error" logfile.log`：在 `logfile.log` 文件中搜索包含关键词 `"error"` 的**行**。
   - `grep -r "function_name" /path/to/code/`：**递归搜索**（`-r`）`/path/to/code/` 目录下所有文件，查找包含 `"function_name"` 的文件。
   - `grep -i "warning" file.txt`：**忽略大小写**（`-i`）地搜索 `"warning"`。

#### 7. `which` / `whereis` - **查找命令的位置**
   - `which ls`：显示 `ls` 这个**命令的可执行文件**存放在哪个路径。
   - `whereis python`：显示 `python` 这个命令的**二进制文件、源码和手册页**的位置。

---

### 二、增（Create）：创建文件和目录

#### 1. `touch` - **创建空文件**或**更新文件时间戳**
   - `touch new_file.txt`：如果 `new_file.txt` 不存在，则创建一个新的空文件；如果存在，则更新它的最后修改时间。

#### 2. `mkdir` - **创建新目录**
   - `mkdir new_folder`：在当前目录下创建名为 `new_folder` 的目录。
   - `mkdir -p project/{src,docs,test}`：**一次性创建嵌套目录**。`-p` 
   - 参数允许创建多级目录。此命令会创建 `project/src`, `project/docs`, `project/test`。

#### 3. `echo` + `>` - **创建并写入内容**
   - `echo "Hello World" > hello.txt`：创建 `hello.txt` 文件，并写入 `"Hello World"`。**如果文件已存在，会覆盖原有内容**。
   - `echo "New line" >> hello.txt`：**追加**（`>>`）内容到 `hello.txt` 文件的末尾，不会覆盖原内容。

---

### 三、改（Modify）：修改、移动和复制

#### 1. `cp` - **复制文件或目录**
   - `cp file.txt file_backup.txt`：复制 `file.txt` 为 `file_backup.txt`。
   - `cp [源文件] [备份文件]`
   - 
   - `cp -r dir1/ dir2/`：**递归复制**（`-r`）整个 `dir1` 目录到 `dir2` 目录。

#### 2. `mv` - **移动或重命名文件/目录**
   - `mv old_name.txt new_name.txt`：**重命名**文件。
   - `mv file.txt /tmp/`：将 `file.txt` **移动**到 `/tmp/` 目录下。
   - `mv aa.txt aa`: 将 `aa.txt` **移动**到 `aa` 目录下。

### 四、删（Remove）：删除

#### 1. `rm` - **删除文件或目录**（⚠️ **危险！慎用！**）
   - `rm file.txt`：删除 `file.txt` 文件。
   - `rm -r folder_name/`：**递归删除**（`-r`）整个 `folder_name` 目录及其内部所有内容。
   - `rm -f log.txt`：**强制删除**（`-f`），不提示确认。`-f` 和 `-r` 结合非常危险。
   - `rm -i` 带提示的删除
   - `rm -rf` 强制性删除
   - **⚠️ 警告：** `rm -rf /` 或 `rm -rf *` 是毁灭性的命令，可能导致系统被擦除。

#### 2.rmdir - **删除目录**


#### 3. 文本编辑器
   - `nano file.txt`：简单易用的命令行文本编辑器，适合新手。
   - `vim file.txt` 或 `vi file.txt`：功能强大但学习曲线较陡的专业编辑器。

---

### 其他重要相关命令

#### 1. `pwd` - **显示当前工作目录**
   - 告诉你你现在在哪个目录里。

#### 2. `cd` - **切换目录**
   - `cd /path/to/dir`: 切换到绝对路径。
   - `cd relative/path`: 切换到相对路径	
   - `cd ..`：切换到上一级目录。
   - `cd ~` 或 `cd`: 直接回到当前用户的**家目录**（`/home/你的用户名`）。
   - `cd -`: 切换到上一个工作目录（来回切换）
   - `cd .`: 保持在当前目录（无实际效果）	
   - `$VAR`: 切换到环境变量指定的路径

#### 3. `file` - **查看文件类型**
   - `file picture.jpg`：告诉你 `picture.jpg` 是一个 JPEG image data...，而不是简单地看后缀名。

### 速查表

| 操作 | 命令 | 常用示例 |
| :--- | :--- | :--- |
| **列目录** | `ls` | `ls -la` |
| **看内容** | `cat`, `less` | `less file.log` |
| **找文件** | `find` | `find . -name "*.txt"` |
| **找内容** | `grep` | `grep -r "error" /var/log/` |
| **创建文件** | `touch` | `touch new_file` |
| **创建目录** | `mkdir` | `mkdir -p parent/child` |
| **复制** | `cp` | `cp -r old_dir/ new_dir/` |
| **移动/重命名** | `mv` | `mv old new` |
| **删除** | `rm` (**小心!**) | `rm -r bad_dir/` |
| **看当前位置** | `pwd` | `pwd` |
| **切换位置** | `cd` | `cd ~` |

在终端里操作，尤其是 `rm`（删除）和 `mv`（移动/重命名）时，一定要**确认好路径和文件名**再按回车！

---

### 3.2 搜索命令

### 一、按文件名搜索

### 1. `find` - 最强大、最常用的搜索命令
在指定目录及其子目录中递归搜索文件。

**基本语法：**
```bash
find [路径] [选项] [表达式]
```

### `find` **搜索**命令常见用法  

   - `find . -name "filename.txt"`：在当前目录及子目录中搜索名为 "filename.txt" 的文件
   - `find . -iname "filename.txt"`: 不区分大小写搜索


#### `find` - **查找文件和目录**

**基本语法：**
```bash
find [路径...] [表达式]
```

**常用用法：**

**1. 基本查找**
- `find`：查找当前目录及其所有子目录中的文件。
- `find /path/to/dir`：在指定目录中查找。
- `find . -name "filename"`：按**文件名**查找（精确匹配）。
- `find . -name "*.txt"`：使用通配符查找所有`.txt`文件。

**2. 按类型查找**
- `find . -type f`：只查找**普通文件**。
- `find . -type d`：只查找**目录**。
- `find . -type l`：只查找**符号链接**。

**3. 按时间查找**
- `find . -mtime -7`：查找**7天内**修改过的文件。
- `find . -mtime +30`：查找**30天前**修改过的文件。
- `find . -amin -60`：查找**60分钟内**访问过的文件。
- `find . -newer file.txt`：查找比`file.txt`更新的文件。

**4. 按大小查找**
- `find . -size +10M`：查找**大于10MB**的文件。
- `find . -size -1G`：查找**小于1GB**的文件。
- `find . -size 0`：查找**空文件**。

**5. 组合条件查找**
- `find . -name "*.log" -type f -mtime +30`：查找30天前的`.log`文件。
- `find . -size +100M -exec ls -lh {} \;`：查找大于100MB的文件并显示详细信息。
- `find . -name "temp*" -delete`：查找并删除以`temp`开头的文件。

**6. 执行操作**
- `find . -name "*.jpg" -exec cp {} /backup/ \;`：查找所有jpg文件并复制到backup目录。
- `find . -type f -empty -exec rm -f {} \;`：查找并删除所有空文件。
- `find . -perm 644`：查找权限为644的文件。

**7. 其他常用选项**
- `find . -maxdepth 2 -name "*.py"`：限制搜索深度为2层。
- `find . -user username`：按文件所有者查找。
- `find . -iname "readme"`：不区分大小写按文件名查找。

**示例组合：**
```bash
# 查找当前目录下7天内修改过的大于1MB的Python文件
find . -name "*.py" -type f -size +1M -mtime -7

# 查找并压缩所有日志文件
find /var/log -name "*.log" -exec gzip {} \;
```

---

## 二、按文件内容搜索

### 2. `grep` - 文本内容搜索工具
在文件内容中搜索特定的文本模式。

**基本语法：**
```bash
grep [选项] "搜索模式" [文件]
```

**常用示例：**
```bash
# 在文件中搜索包含 "error" 的行
grep "error" filename.log

# 递归搜索当前目录及子目录中的所有文件
grep -r "TODO" .

# 不区分大小写搜索
grep -i "error" file.txt

# 显示匹配行的行号
grep -n "pattern" file.txt

# 只显示匹配的文件名（不显示具体行）
grep -l "pattern" *.txt

# 反向搜索（显示不包含模式的行）
grep -v "success" file.log

# 使用正则表达式搜索
grep "^[0-9]" file.txt  # 以数字开头的行
grep "\.html$" file.txt # 以 .html 结尾的行

# 搜索多个模式
grep -e "error" -e "warning" file.log

# 结合 find 使用：在所有 .txt 文件中搜索
find . -name "*.txt" -exec grep "pattern" {} \;
```

### 3. `ack` / `rg` (ripgrep) - 更快的替代品
这些是 `grep` 的现代替代品，搜索速度更快，默认忽略版本控制目录。

```bash
# 如果已安装 ack
ack "pattern"

# 如果已安装 ripgrep
rg "pattern"
```

---

## 三、按命令历史搜索

### 4. `history` + `grep` - 搜索命令历史
```bash
# 搜索曾经使用过的命令
history | grep "ssh"

# 或者使用 Ctrl+R 进行反向搜索（在终端中按 Ctrl+R 然后输入关键词）
```

---

## 四、按程序名和文档搜索

### 5. `which` - 显示命令的完整路径
```bash
# 查看命令的位置
which python
which git
```

### 6. `whereis` - 查找二进制程序、源码和手册页
```bash
# 查找命令的相关文件
whereis python
whereis ls
```

### 7. `locate` / `mlocate` - 基于数据库的快速文件搜索
使用预建的数据库进行快速搜索，但需要定期更新数据库。

```bash
# 更新搜索数据库（需要sudo）
sudo updatedb

# 快速搜索文件
locate filename.txt

# 限制搜索结果数量
locate "*.jpg" | head -20

# 忽略大小写
locate -i "readme"
```

---

## 五、组合使用示例

### 实际应用场景

**场景1：在项目中搜索所有包含 "TODO" 的 Python 文件**
```bash
find . -name "*.py" -exec grep -l "TODO" {} \;
# 或者更简单的方式
grep -r "TODO" --include="*.py" .
```

**场景2：查找最近修改过的大文件**
```bash
find . -type f -size +10M -mtime -30 -exec ls -lh {} \;
```

**场景3：搜索并处理文件**
```bash
# 搜索所有 .log 文件并统计行数
find /var/log -name "*.log" -exec wc -l {} \;

# 搜索所有空目录
find . -type d -empty
```

**场景4：复杂的条件搜索**
```bash
# 搜索大于1MB且最近30天内修改过的图片文件
find . -type f \( -name "*.jpg" -o -name "*.png" \) -size +1M -mtime -30
```

---

## 六、实用技巧

### 1. 使用通配符
```bash
# 搜索当前目录下所有 .txt 文件
ls *.txt

# 递归搜索（需要启用 globstar）
shopt -s globstar
ls **/*.txt
```

### 2. 管道组合搜索
```bash
# 搜索包含特定内容的文件，然后统计数量
grep -r "error" . | wc -l

# 查找大文件并排序
find . -type f -size +100M -exec ls -lh {} \; | sort -k5 -hr
```

## 总结

| 命令 | 主要用途 | 特点 |
|------|----------|------|
| `find` | 按文件名、属性搜索 | 功能最强大，支持递归 |
| `grep` | 按文件内容搜索 | 文本搜索专家 |
| `locate` | 快速文件名搜索 | 速度快，但需要更新数据库 |
| `which`/`whereis` | 查找命令位置 | 定位可执行文件 |

根据具体需求选择合适的工具：
- **找文件**：用 `find` 或 `locate`
- **找内容**：用 `grep`
- **找命令**：用 `which` 或 `whereis`

---

### 3.3帮助命令
 `man` - **查看命令手册页**

**基本语法：**
```bash
man [选项] 命令名
```

**常用用法：**
- `man ls`：查看`ls`命令的完整手册。
- `man -k "keyword"`：搜索包含关键词的手册页。
- `man 5 passwd`：查看第5节（文件格式）的passwd手册（而不是第1节的passwd命令）。
- `man -a command`：显示所有匹配的手册节。

**手册节说明：**
- `1`：用户命令
- `2`：系统调用
- `3`：库函数
- `4`：特殊文件
- `5`：文件格式
- `6`：游戏
- `7`：杂项
- `8`：系统管理命令

**在man页面中的操作：**
- `空格`/`Page Down`：向下翻页
- `b`/`Page Up`：向上翻页
- `/keyword`：搜索关键词
- `n`：下一个匹配项
- `q`：退出man页面

---

#### `help` - **获取shell内置命令帮助**

**基本语法：**
```bash
help [选项] [命令名]
```

**常用用法：**
- `help`：显示所有shell内置命令列表。
- `help cd`：查看`cd`命令的帮助（因为cd是shell内置命令）。
- `help -d command`：显示命令的简短描述。
- `help -s command`：显示命令的语法概要。

**针对外部命令的帮助：**
- `ls --help`：大多数GNU工具使用`--help`选项。
- `command --help`：通用的帮助选项。

---

#### 两种命令的区别和使用场景：

**`man`命令：**
- 适用于**所有命令**（内置命令和外部命令）
- 提供**完整的官方文档**
- 内容详细，格式规范
- **示例：** `man grep`, `man find`, `man bash`

**`help`命令：**
- 主要适用于**shell内置命令**
- 内容简洁，快速参考
- **示例：** `help export`, `help alias`, `help echo`

**实用技巧：**
```bash
# 判断命令类型
type cd          # 显示：cd is a shell builtin（使用help）
type ls          # 显示：ls is /usr/bin/ls（使用man或--help）

# 组合使用
whatis command   # 显示命令的简短描述
which command    # 显示命令的路径
whereis command  # 显示命令的二进制、源码和手册页位置

# 快速帮助参考
man -f command   # 等同于whatis
apropos keyword  # 等同于man -k
```

**总结：**
- **内置命令** → 使用 `help`
- **外部命令** → 使用 `man` 或 `命令 --help`
- **不确定时** → 先用 `type` 判断命令类型

---

### 3.4压缩解压与打包指令


### 一、打包与压缩的区别

| 类型                  | 含义                     | 常见命令                        |
| ------------------- | ---------------------- | --------------------------- |
| **打包（Packaging）**   | 将多个文件或目录合并成一个文件（不压缩体积） | `tar`                       |
| **压缩（Compression）** | 减少文件体积（单文件或包）          | `gzip`、`bzip2`、`xz`、`zip` 等 |

> 打包常与压缩结合使用，例如：
> `tar` + `gzip` → `.tar.gz`
> `tar` + `bzip2` → `.tar.bz2`

---

## 🎯 二、常用打包命令 `tar`

### 1️⃣ 基本语法

```bash
tar [选项] [目标文件] [源文件或目录]
```

### 2️⃣ 常用选项

| 选项   | 含义                      |
| ---- | ----------------------- |
| `-c` | 创建新归档（create）           |
| `-x` | 解包（extract）             |
| `-t` | 查看内容（list）              |
| `-v` | 显示过程（verbose）           |
| `-f` | 指定文件名（file）             |
| `-z` | 用 gzip 压缩或解压（.tar.gz）   |
| `-j` | 用 bzip2 压缩或解压（.tar.bz2） |
| `-J` | 用 xz 压缩或解压（.tar.xz）     |
| `-C` | 指定解压目录                  |

---

### 3️⃣ 打包与解包示例

#### 🧱 仅打包（不压缩）

```bash
# 打包
tar -cvf backup.tar /home/user/
# 解包
tar -xvf backup.tar
```

#### 🌀 打包 + gzip 压缩（.tar.gz）

```bash
# 打包压缩
tar -zcvf backup.tar.gz /home/user/
# 解压缩
tar -zxvf backup.tar.gz
# 解压到指定目录
tar -zxvf backup.tar.gz -C /opt/data/
```

#### 🧩 打包 + bzip2 压缩（.tar.bz2）

```bash
# 打包压缩
tar -jcvf backup.tar.bz2 /home/user/
# 解压缩
tar -jxvf backup.tar.bz2
```

#### 🧮 打包 + xz 压缩（.tar.xz）

```bash
# 打包压缩
tar -Jcvf backup.tar.xz /home/user/
# 解压缩
tar -Jxvf backup.tar.xz
```

#### 🔍 查看压缩包内容

```bash
tar -tvf backup.tar.gz
```

---

## 📦 三、单独压缩命令（含解压）

### 1️⃣ gzip / gunzip

```bash
# 压缩
gzip file.txt          # → file.txt.gz
gzip -r dir/           # 递归压缩目录下所有文件
gzip -k file.txt       # 压缩后保留原文件

# 解压
gunzip file.txt.gz
gzip -d file.txt.gz    # 与 gunzip 等价
```

---

### 2️⃣ bzip2 / bunzip2

```bash
# 压缩
bzip2 file.txt         # → file.txt.bz2
bzip2 -k file.txt      # 保留原文件

# 解压
bunzip2 file.txt.bz2
bzip2 -d file.txt.bz2  # 与 bunzip2 等价
```

---

### 3️⃣ xz / unxz

```bash
# 压缩
xz file.txt            # → file.txt.xz
xz -k file.txt         # 保留原文件

# 解压
unxz file.txt.xz
xz -d file.txt.xz      # 与 unxz 等价
```

---

### 4️⃣ zip / unzip

```bash
# 压缩
zip archive.zip file1 file2 dir/     # 压缩多个文件或目录
zip -r archive.zip dir/              # 递归压缩目录

# 解压
unzip archive.zip                    # 解压到当前目录
unzip -d /target/ archive.zip        # 解压到指定目录
unzip -l archive.zip                 # 查看压缩包内容
```

---

### 5️⃣ 7z（需安装 p7zip）

```bash
# 压缩
7z a archive.7z dir/

# 解压
7z x archive.7z
```

---

## 四、常见格式与对应解压命令对照表

| 文件后缀       | 压缩方式  | 压缩命令                          | 解压命令                     |
| ---------- | ----- | ----------------------------- | ------------------------ |
| `.tar`     | 打包    | `tar -cvf file.tar dir/`      | `tar -xvf file.tar`      |
| `.tar.gz`  | gzip  | `tar -zcvf file.tar.gz dir/`  | `tar -zxvf file.tar.gz`  |
| `.tar.bz2` | bzip2 | `tar -jcvf file.tar.bz2 dir/` | `tar -jxvf file.tar.bz2` |
| `.tar.xz`  | xz    | `tar -Jcvf file.tar.xz dir/`  | `tar -Jxvf file.tar.xz`  |
| `.gz`      | gzip  | `gzip file`                   | `gunzip file.gz`         |
| `.bz2`     | bzip2 | `bzip2 file`                  | `bunzip2 file.bz2`       |
| `.xz`      | xz    | `xz file`                     | `unxz file.xz`           |
| `.zip`     | zip   | `zip archive.zip dir/`        | `unzip archive.zip`      |
| `.7z`      | 7zip  | `7z a archive.7z dir/`        | `7z x archive.7z`        |

---

## 五、实用技巧

| 场景             | 命令示例                                                 |
| -------------- | ---------------------------------------------------- |
| **压缩时排除某些文件**  | `tar -zcvf backup.tar.gz /home --exclude=/home/temp` |
| **仅查看压缩包内容**   | `tar -tvf file.tar.gz`                               |
| **解压到指定路径**    | `tar -zxvf file.tar.gz -C /usr/local`                |
| **最大压缩比（XZ）**  | `tar -Jcvf backup.tar.xz /data`                      |
| **分卷压缩（ZIP）**  | `zip -r -s 100m backup.zip /data`（每卷100MB）           |
| **在解压时自动创建目录** | `mkdir -p /data && tar -zxvf backup.tar.gz -C /data` |

---

## 六、命令速查表（打包 + 压缩 + 解压）

| 操作            | 命令                            |
| ------------- | ----------------------------- |
| 打包            | `tar -cvf file.tar dir/`      |
| 解包            | `tar -xvf file.tar`           |
| 打包并压缩 (gzip)  | `tar -zcvf file.tar.gz dir/`  |
| 解压 `.tar.gz`  | `tar -zxvf file.tar.gz`       |
| 打包并压缩 (bzip2) | `tar -jcvf file.tar.bz2 dir/` |
| 解压 `.tar.bz2` | `tar -jxvf file.tar.bz2`      |
| 打包并压缩 (xz)    | `tar -Jcvf file.tar.xz dir/`  |
| 解压 `.tar.xz`  | `tar -Jxvf file.tar.xz`       |
| 压缩单文件 (gzip)  | `gzip file.txt`               |
| 解压 `.gz`      | `gunzip file.gz`              |
| 压缩为 zip       | `zip -r file.zip dir/`        |
| 解压 `.zip`     | `unzip file.zip`              |

---

## 3.5 关机与重启

下面是整理好的 **Linux关机与重启命令笔记总结**，包括常用命令、参数区别、运行级别、定时与立即关机、以及图形/远程环境下的使用技巧。内容系统、简明，非常适合作为学习或复习笔记。

---

## ⚙️ Linux 关机与重启命令笔记总结

---

## 🧭 一、关机与重启的本质

在 Linux 中，“关机（shutdown）”和“重启（reboot）”都属于 **系统运行级别（runlevel）** 的切换。

* 运行级别 0：**关机**
* 运行级别 6：**重启**
* 运行级别 3：多用户命令行模式
* 运行级别 5：图形界面模式

因此，执行关机或重启命令实质是让系统进入相应的运行级别。

---

## 🛑 二、常用关机命令汇总

### 1️⃣ `shutdown` —— 最常用、安全的关机方式

```bash
shutdown [选项] [时间] [警告信息]
```

#### ✅ 常见用法

| 操作         | 命令示例                                 | 说明                 |
| ---------- | ------------------------------------ | ------------------ |
| 立即关机       | `shutdown -h now`                    | `-h` 表示 halt（停止系统） |
| 定时关机（1分钟后） | `shutdown -h +1`                     | “+1”表示1分钟后执行       |
| 指定时间关机     | `shutdown -h 22:00`                  | 每天晚上10点自动关机        |
| 立即重启       | `shutdown -r now`                    | `-r` 表示 reboot（重启） |
| 定时重启       | `shutdown -r +5`                     | 5分钟后重启             |
| 取消计划的关机任务  | `shutdown -c`                        | 取消正在等待的关机任务        |
| 广播关机警告     | `shutdown -h +3 "系统将在3分钟后关机，请保存工作！"` | 通知所有登录用户           |

> 💡 `shutdown` 会先向所有登录用户发送警告消息，再逐步终止进程、卸载文件系统，**是最安全的关机命令**。

---

### 2️⃣ `poweroff` —— 直接关闭电源

```bash
poweroff
```

* 效果等价于 `shutdown -h now`
* 立即关机并切断电源
* 通常用于嵌入式设备、服务器脚本中

---

### 3️⃣ `halt` —— 停止系统

```bash
halt
```

* 停止所有 CPU 功能，系统挂起
* 不一定断电（取决于 BIOS/电源管理设置）
* 相当于 `shutdown -H now`

---

### 4️⃣ `init` —— 切换运行级别

```bash
init 0   # 进入运行级别0 → 关机
init 6   # 进入运行级别6 → 重启
```

> ⚠️ `init` 是底层命令，适合脚本或老系统（SysVinit）。在 systemd 系统中建议使用 `systemctl`。

---

### 5️⃣ `systemctl` —— 现代系统（systemd）推荐方式

```bash
systemctl poweroff     # 关机
systemctl halt         # 停止系统
systemctl reboot       # 重启
systemctl suspend      # 挂起（休眠到内存）
systemctl hibernate    # 休眠到硬盘
systemctl hybrid-sleep # 混合休眠
```

> ✅ 适用于大多数现代 Linux 发行版（如 Ubuntu、CentOS 7+、Fedora、Debian 8+ 等）。

---

## 🔁 三、重启命令汇总

| 命令                 | 功能说明                          | 是否安全       |
| ------------------ | ----------------------------- | ---------- |
| `reboot`           | 立即重启系统（等价于 `shutdown -r now`） | ✅          |
| `shutdown -r now`  | 安全重启，先通知用户、再重启                | ✅          |
| `systemctl reboot` | 使用 systemd 管理重启               | ✅          |
| `init 6`           | 切换到运行级别 6（重启）                 | ⚠️（旧系统）    |
| `ctrl + alt + del` | 图形界面/终端快捷键重启（需配置）             | ⚠️（有时会被禁用） |

---

## 🕒 四、定时与延迟关机/重启

| 任务      | 示例                  | 说明         |
| ------- | ------------------- | ---------- |
| 10分钟后关机 | `shutdown -h +10`   | +10 表示10分钟 |
| 指定时间关机  | `shutdown -h 23:30` | 23:30 执行关机 |
| 5分钟后重启  | `shutdown -r +5`    | 5分钟后自动重启   |
| 取消计划任务  | `shutdown -c`       | 取消等待中的关机   |

> ⏰ 也可用 `at` 命令实现更灵活的定时任务：

```bash
echo "shutdown -h now" | at 23:00
```

---

## 🧍‍♂️ 五、权限与远程控制

### 1️⃣ 普通用户执行关机/重启

* 默认需要 **root 权限**
* 普通用户可通过以下方式执行：

  ```bash
  sudo shutdown -h now
  sudo reboot
  ```

### 2️⃣ 远程关机（SSH）

* 使用远程登录后直接执行命令：

  ```bash
  ssh user@server 'sudo shutdown -r now'
  ```
* 或者通过 Ansible、批量脚本实现多机控制。

---

## 🧠 六、常见区别与注意事项

| 命令                   | 特点      | 是否断电    | 是否推荐   |
| -------------------- | ------- | ------- | ------ |
| `halt`               | 停止CPU运行 | ❌ 不一定断电 | ⚠️ 老命令 |
| `poweroff`           | 停止并断电   | ✅       | ✅      |
| `shutdown -h now`    | 有序安全关机  | ✅       | ✅      |
| `reboot`             | 快速重启系统  | ✅       | ✅      |
| `systemctl poweroff` | 现代推荐方式  | ✅       | ✅      |
| `init 0` / `init 6`  | 旧式方式    | ✅       | ⚠️ 不推荐 |

---

## 🧩 七、命令速查表

| 操作          | 命令                                          |
| ----------- | ------------------------------------------- |
| 立即关机        | `shutdown -h now`                           |
| 定时关机（10分钟后） | `shutdown -h +10`                           |
| 指定时间关机      | `shutdown -h 22:00`                         |
| 立即重启        | `shutdown -r now`                           |
| 定时重启        | `shutdown -r +5`                            |
| 取消计划任务      | `shutdown -c`                               |
| 快速关机        | `poweroff`                                  |
| 快速重启        | `reboot`                                    |
| 切换到关机级别     | `init 0`                                    |
| 切换到重启级别     | `init 6`                                    |
| systemd关机   | `systemctl poweroff`                        |
| systemd重启   | `systemctl reboot`                          |
| 挂起休眠        | `systemctl suspend` / `systemctl hibernate` |

---

## 🧩 八、示例汇总

```bash
# 安全关机
sudo shutdown -h now

# 安全重启
sudo shutdown -r now

# 延迟5分钟重启并广播信息
sudo shutdown -r +5 "系统将在5分钟后重启，请保存数据！"

# 取消定时任务
sudo shutdown -c

# 现代系统直接关机
sudo systemctl poweroff

# 从命令行快速重启
sudo reboot
```

---

## 3.6 其他命令

下面是经过**扩充版的 Linux 系统用户与状态命令速查笔记**，包含了：
✅ 命令说明 ✅ 常见参数 ✅ 实际示例输出（示意） ✅ 实用小技巧

---

## 🧭 Linux 系统信息与状态查看命令速查笔记

## 一、查看用户登录信息

#### 1️⃣ `who` — 查看当前登录的用户信息

**功能说明：**
显示当前系统上已登录的用户、终端、登录时间、来源IP等。

**常见参数：**

| 参数   | 说明                     |
| ---- | ---------------------- |
| 无    | 显示基本登录信息               |
| `-a` | 显示所有可用信息（包括运行级别、启动时间等） |

**示例：**

```bash
$ who
lxlei   tty2         2025-10-16 10:15
root    pts/0        2025-10-16 10:17 (192.168.1.100)
```

---

#### 2️⃣ `w` — 查看登录用户及其活动信息

**功能说明：**
`w` 是 `who` 的增强版，会显示每个用户的运行状态、CPU占用、当前命令等。

**输出示例：**

```bash
$ w
 15:30:20 up  2:10,  3 users,  load average: 0.15, 0.05, 0.01
USER     TTY      FROM             IDLE   JCPU   PCPU WHAT
root     pts/0    192.168.1.100    00:01  0.10s  0.05s top
lxlei    tty2                      01:15  0.02s  0.02s bash
```

💡 **技巧：** 想快速看谁在登录、干什么，用 `w` 比 `who` 更全面。

---

#### 3️⃣ `lastlog` — 查看每个用户最后一次登录时间

**功能说明：**
从 `/var/log/lastlog` 文件中读取所有用户的最后登录记录。

**示例：**

```bash
$ lastlog
Username  Port  From             Latest
root      pts/0 192.168.1.100    Thu Oct 16 10:17:02 +0800 2025
lxlei     tty2                   Thu Oct 16 10:15:40 +0800 2025
```

💡 **技巧：** 如果某用户从未登录过，会显示 “**Never logged in**”。

---

### 二、系统资源与任务状态

#### 4️⃣ `df` — 查看磁盘使用情况

**功能说明：**
显示文件系统磁盘空间的使用情况。

**常见参数：**

| 参数   | 说明                         |
| ---- | -------------------------- |
| `-h` | 以人类可读的方式显示（自动换算为 KB/MB/GB） |
| `-T` | 显示文件系统类型                   |
| `-i` | 查看 inode 使用情况              |

**示例：**

```bash
$ df -h
Filesystem      Size  Used Avail Use% Mounted on
/dev/sda2        50G   20G   28G  42% /
tmpfs           3.9G  2.0M  3.9G   1% /run
```

💡 **技巧：** 检查磁盘空间不足问题首选命令。

---

#### 5️⃣ `top` — 实时查看系统任务

**功能说明：**
动态显示系统中各进程资源占用情况（CPU、内存、运行时间等）。

**常用快捷键：**

| 快捷键 | 作用             |
| --- | -------------- |
| `q` | 退出 top         |
| `P` | 按 CPU 使用率排序    |
| `M` | 按内存使用率排序       |
| `k` | 结束指定进程（输入 PID） |

**示例：**

```
top - 15:34:10 up  2:14,  3 users,  load average: 0.12, 0.03, 0.00
PID USER  PR NI VIRT RES SHR S %CPU %MEM TIME+ COMMAND
1650 root  20  0 162M 12M  8M S  0.5  0.3  0:05.03 sshd
1801 lxlei 20  0 320M 55M 21M S  0.2  1.4  0:10.21 bash
```

💡 **技巧：** 想看整体CPU/内存状态更直观可用 `htop`（需安装）。

---

#### 6️⃣ `free` — 查看内存使用情况

**功能说明：**
显示物理内存与交换分区（swap）的使用情况。

**常见参数：**

| 参数     | 说明          |
| ------ | ----------- |
| `-h`   | 人类可读格式显示    |
| `-s 2` | 每2秒自动刷新显示一次 |

**示例：**

```bash
$ free -h
              total        used        free      shared  buff/cache   available
Mem:           7.7G        2.3G        4.2G        200M        1.2G        5.0G
Swap:          2.0G          0B        2.0G
```

💡 **技巧：** “available” 比 “free” 更准确反映可用内存。

---

### 三、操作历史与输出显示

#### 7️⃣ `history` — 查看命令历史记录

**功能说明：**
显示当前用户执行过的命令历史。

**常见参数：**

| 参数          | 说明                 |
| ----------- | ------------------ |
| `history n` | 显示最近 n 条命令         |
| `!n`        | 执行历史编号为 n 的命令      |
| `!string`   | 执行最近以 string 开头的命令 |

**示例：**

```bash
$ history 5
  58  df -h
  59  free -h
  60  history 10
  61  top
  62  who
```

💡 **技巧：** 历史记录文件位于 `~/.bash_history`。

---

#### 8️⃣ `echo` — 显示内容或变量

**功能说明：**
用于输出字符串、变量内容，或写入文件。

**示例：**

```bash
$ echo "Hello Linux!"
Hello Linux!

$ echo $HOME
/home/lxlei
```

💡 **技巧：** 可配合重定向符 `>` 输出到文件：

```bash
echo "Backup Done" > result.log
```

---

### 四、文件内容查看命令

#### 9️⃣ `cat` — 显示文件内容

**功能说明：**
一次性输出整个文件内容。

**常见参数：**

| 参数   | 说明                |
| ---- | ----------------- |
| `-n` | 显示行号              |
| `-b` | 对非空行显示行号          |
| `-A` | 显示不可见字符（如制表符、换行符） |

**示例：**

```bash
$ cat -n /etc/hosts
     1  127.0.0.1   localhost
     2  192.168.1.10 server.local
```

---

#### 🔟 `tail` — 查看文件末尾内容

**功能说明：**
显示文件最后几行内容，常用于日志监控。

**常见参数：**

| 参数     | 说明               |
| ------ | ---------------- |
| `-n N` | 显示最后 N 行         |
| `-f`   | 实时跟踪文件更新（监控日志必备） |

**示例：**

```bash
$ tail -n 5 /var/log/messages
$ tail -f /var/log/syslog
```

💡 **技巧：** 按 `Ctrl + C` 退出 `tail -f`。

---

## 4. vi编辑器

### 4.1 vi 编辑器简介

**vi (Visual Editor)** 是 Linux/Unix 系统中最常用的文本编辑器之一，轻量、强大，几乎所有系统都内置。
虽然使用上略有门槛，但熟练后非常高效。

**三种工作模式：**

| 模式名称                     | 功能               | 进入方式                  | 退出方式                  |
| ------------------------ | ---------------- | --------------------- | --------------------- |
| **命令模式（Command Mode）**   | 控制光标移动、删除、复制、粘贴等 | 打开文件默认即进入             | 按 `i`、`a`、`o` 等进入编辑模式 |
| **插入模式（Insert Mode）**    | 输入文本内容           | 在命令模式下按 `i`、`a`、`o` 等 | 按 `Esc` 返回命令模式        |
| **底行模式（Last Line Mode）** | 保存、退出、搜索、替换等     | 命令模式下输入 `:`           | 按 `Enter` 执行命令后返回命令模式 |

---

### 4.2 基本文件操作命令

| 操作       | 命令            | 说明            |
| -------- | ------------- | ------------- |
| 打开文件     | `vi filename` | 打开或新建文件       |
| 保存文件     | `:w`          | 保存修改          |
| 保存并退出    | `:wq` 或 `ZZ`  | 写入文件并退出       |
| 不保存退出    | `:q!`         | 放弃修改直接退出      |
| 只退出（无修改） | `:q`          | 若有修改未保存则会提示错误 |
| 另存为      | `:w newfile`  | 将当前内容保存为新文件   |

---

### 4.3 光标移动命令（命令模式下）

| 命令         | 功能             |
| ---------- | -------------- |
| `h`        | 向左移动一个字符       |
| `l`        | 向右移动一个字符       |
| `j`        | 向下移动一行         |
| `k`        | 向上移动一行         |
| `0`（数字零）   | 移动到行首          |
| `$`        | 移动到行尾          |
| `gg`       | 跳转到文件开头        |
| `G`        | 跳转到文件末尾        |
| `:n`       | 跳转到第 n 行       |
| `Ctrl + f` | 向下翻页（forward）  |
| `Ctrl + b` | 向上翻页（backward） |

💡 **技巧：** 组合数字可移动多次，如 `5j` 表示向下移动 5 行。

---

### 4.4 文本编辑命令（命令模式）

| 操作     | 命令         | 说明              |
| ------ | ---------- | --------------- |
| 插入模式   | `i`        | 在光标前插入          |
| 追加模式   | `a`        | 在光标后插入          |
| 新起一行   | `o`        | 在当前行下方新建一行并进入编辑 |
| 删除字符   | `x`        | 删除光标所在字符        |
| 删除整行   | `dd`       | 删除当前行           |
| 删除多行   | `ndd`      | 删除 n 行（如 `3dd`） |
| 复制当前行  | `yy`       | 复制当前行           |
| 复制多行   | `nyy`      | 复制 n 行（如 `5yy`） |
| 粘贴     | `p`        | 在光标后粘贴          |
| 替换单个字符 | `r`        | 替换当前光标下的字符      |
| 撤销上一步  | `u`        | 撤销操作            |
| 重做上一步  | `Ctrl + r` | 撤销的恢复           |
| 合并两行   | `J`        | 将下一行并入当前行       |

---

### 4.5 查找与替换

#### 🔍 查找

| 命令         | 功能            |
| ---------- | ------------- |
| `/keyword` | 向下搜索“keyword” |
| `?keyword` | 向上搜索“keyword” |
| `n`        | 重复上一次搜索（同方向）  |
| `N`        | 反向重复上一次搜索     |

💡 搜索时大小写敏感，若希望不区分大小写可输入 `:set ic`。

#### 🔁 替换

| 命令               | 功能          |
| ---------------- | ----------- |
| `:s/old/new/`    | 替换当前行第一个匹配项 |
| `:s/old/new/g`   | 替换当前行所有匹配项  |
| `:%s/old/new/g`  | 替换整个文件所有匹配项 |
| `:%s/old/new/gc` | 全局替换并逐一确认   |

---

### 4.6 文件浏览与多文件操作

| 命令               | 功能                 |
| ---------------- | ------------------ |
| `:e filename`    | 打开另一个文件            |
| `:n`             | 打开下一个文件（若多个文件一起打开） |
| `:prev`          | 打开上一个文件            |
| `:r filename`    | 在光标处插入其他文件内容       |
| `:sp filename`   | 分屏打开文件（水平）         |
| `:vsp filename`  | 分屏打开文件（垂直）         |
| `Ctrl + w` + `w` | 在分屏间切换光标           |

---

### 4.7 显示与设置

| 命令            | 功能             |
| ------------- | -------------- |
| `:set nu`     | 显示行号           |
| `:set nonu`   | 取消行号           |
| `:set ic`     | 搜索时忽略大小写       |
| `:set noic`   | 搜索时区分大小写       |
| `:set paste`  | 进入粘贴模式（防止缩进错乱） |
| `:syntax on`  | 启用语法高亮         |
| `:syntax off` | 关闭语法高亮         |

---

### 4.8 组合操作示例

| 操作目标                  | 命令示例               | 说明            |
| --------------------- | ------------------ | ------------- |
| 删除第10行                | `:10d`             | 删除第10行内容      |
| 复制第20行到第25行           | `:20,25y`          | 复制指定行范围       |
| 替换第3到10行中的“abc”为“xyz” | `:3,10s/abc/xyz/g` | 区间替换          |
| 从当前行到文件末尾删除           | `:.,$d`            | 删除至文件结尾       |
| 全选并复制                 | `ggVGy`            | 快捷操作：从头到尾选中复制 |

---

### 4.9 退出技巧总结

| 操作        | 命令               |
| --------- | ---------------- |
| 保存并退出     | `:wq` 或 `ZZ`     |
| 强制退出（不保存） | `:q!`            |
| 保存但不退出    | `:w`             |
| 另存为新文件    | `:w newfile.txt` |

---

### 🔧 小结技巧

* 想快速进入编辑：`i` 或 `o`
* 想保存退出：`:wq`
* 想放弃修改：`:q!`
* 想撤销操作：`u`
* 想复制粘贴：`yy` → `p`
* 想搜索关键词：`/word`
* 想全局替换：`:%s/old/new/g`

---

## 5. 系统服务管理

### 5.1 systemctl 简介

**`systemctl`** 是 Linux 系统中基于 **systemd** 的核心管理工具，用于：

* 启动、停止、重启、启用或禁用系统服务；
* 管理系统运行级别（target）；
* 查看服务状态与日志；
* 控制系统关机、重启等操作。

> 🔹 systemctl 相当于旧系统中 `service` 与 `chkconfig` 命令的整合与升级版本。

---

### 5.2 基本语法

```bash
systemctl [command] [unit]
```

**示例：**

```bash
systemctl start nginx
systemctl status sshd
systemctl enable firewalld
```

---

### 5.3 常见“服务控制”子命令

| 操作           | 命令格式                                       | 说明            |
| ------------ | ------------------------------------------ | ------------- |
| 启动服务         | `systemctl start 服务名`                      | 立即启动服务        |
| 停止服务         | `systemctl stop 服务名`                       | 停止运行的服务       |
| 重启服务         | `systemctl restart 服务名`                    | 先停止再启动        |
| 重新加载配置       | `systemctl reload 服务名`                     | 不中断服务，仅重载配置   |
| 查看状态         | `systemctl status 服务名`                     | 查看运行状态、PID、日志 |
| 查看所有服务       | `systemctl list-units --type=service`      | 列出当前活动服务      |
| 查看所有（含未激活）服务 | `systemctl list-unit-files --type=service` | 查看全部服务启动配置    |

**示例：**

```bash
systemctl status sshd
systemctl restart network
```

---

### 5.4 开机启动项管理

| 操作       | 命令                            | 说明         |
| -------- | ----------------------------- | ---------- |
| 启用开机自启   | `systemctl enable 服务名`        | 设置服务开机自动启动 |
| 禁用开机自启   | `systemctl disable 服务名`       | 取消自动启动     |
| 检查开机启动状态 | `systemctl is-enabled 服务名`    | 显示是否启用     |
| 立即启用并启动  | `systemctl enable --now 服务名`  | 同时生效并启动    |
| 立即禁用并停止  | `systemctl disable --now 服务名` | 同时关闭并禁用    |

**示例：**

```bash
systemctl enable firewalld
systemctl disable httpd
```

---

### 5.5 系统级控制命令（替代传统 init 命令）

| 操作         | 命令                                       | 说明                    |
| ---------- | ---------------------------------------- | --------------------- |
| 重启系统       | `systemctl reboot`                       | 等同于 `reboot`          |
| 关机         | `systemctl poweroff`                     | 等同于 `shutdown -h now` |
| 挂起系统       | `systemctl suspend`                      | 进入内存挂起                |
| 休眠系统       | `systemctl hibernate`                    | 将内存写入硬盘再休眠            |
| 切换运行级别（目标） | `systemctl isolate multi-user.target`    | 切换到多用户模式              |
| 切换到图形界面    | `systemctl isolate graphical.target`     | 启动图形界面                |
| 查看当前运行目标   | `systemctl get-default`                  | 查看默认启动模式              |
| 设置默认目标     | `systemctl set-default graphical.target` | 设置系统默认运行级别            |

💡 运行目标（target）是 systemd 取代旧有运行级别（runlevel）的概念：

| 传统 runlevel | 对应 target         | 含义       |
| ----------- | ----------------- | -------- |
| 0           | poweroff.target   | 关机       |
| 1           | rescue.target     | 单用户模式    |
| 3           | multi-user.target | 命令行多用户模式 |
| 5           | graphical.target  | 图形界面模式   |
| 6           | reboot.target     | 重启       |

---

### 5.6 服务文件（Unit）管理

systemctl 管理的对象称为 **“Unit” 单元**，可包含服务、挂载点、套接字、目标等类型。

| 类型     | 后缀         | 说明            |
| ------ | ---------- | ------------- |
| 服务单元   | `.service` | 系统服务（最常用）     |
| 目标单元   | `.target`  | 运行级别控制        |
| 挂载单元   | `.mount`   | 文件系统挂载点       |
| 套接字单元  | `.socket`  | 网络监听套接字       |
| 定时任务单元 | `.timer`   | 定时任务（替代 cron） |

**示例：**

```bash
systemctl status sshd.service
systemctl restart nginx.service
```

💡 后缀 `.service` 可省略（systemctl 会自动识别）。

---

### 5.7 日志与调试命令

| 操作        | 命令                        | 说明           |
| --------- | ------------------------- | ------------ |
| 查看服务日志    | `journalctl -u 服务名`       | 查看指定服务日志     |
| 查看最近的系统日志 | `journalctl -xe`          | 查看系统错误信息     |
| 实时监控日志    | `journalctl -f -u 服务名`    | 类似 `tail -f` |
| 限制日志条数    | `journalctl -n 50 -u 服务名` | 显示最近50条日志    |
| 查看所有启动日志  | `journalctl -b`           | 当前启动的全部日志    |

**示例：**

```bash
journalctl -u nginx -n 20
journalctl -xe
```

---

### 5.8 常用诊断与辅助命令

| 操作              | 命令                        | 说明                       |
| --------------- | ------------------------- | ------------------------ |
| 检查服务是否运行        | `systemctl is-active 服务名` | 返回 `active` 或 `inactive` |
| 检查是否失败          | `systemctl is-failed 服务名` | 返回 `failed` 表示异常         |
| 列出失败的服务         | `systemctl --failed`      | 快速定位出错服务                 |
| 重新加载 systemd 配置 | `systemctl daemon-reload` | 修改 unit 文件后必须执行          |
| 屏蔽服务（禁止手动或自动启动） | `systemctl mask 服务名`      | 阻止服务被启动                  |
| 取消屏蔽            | `systemctl unmask 服务名`    | 恢复服务启动功能                 |

---

### 5.9 实用示例汇总

| 目标                   | 命令                                 | 说明     |
| -------------------- | ---------------------------------- | ------ |
| 启动防火墙服务              | `systemctl start firewalld`        | 立即启动   |
| 设置防火墙开机启动            | `systemctl enable firewalld`       | 永久启用   |
| 禁止 NetworkManager 服务 | `systemctl disable NetworkManager` | 关闭自启   |
| 查看 ssh 服务运行状态        | `systemctl status sshd`            | 查看详细状态 |
| 实时查看 nginx 日志        | `journalctl -f -u nginx`           | 动态输出日志 |
| 重载 nginx 配置          | `systemctl reload nginx`           | 无需中断服务 |
| 屏蔽 httpd 服务          | `systemctl mask httpd`             | 防止误启动  |

---

### 5.10 命令逻辑总结（思维导图式记忆）

```
systemctl
├── 服务控制：start | stop | restart | reload | status
├── 开机自启：enable | disable | is-enabled
├── 系统控制：reboot | poweroff | suspend | hibernate
├── 状态查看：list-units | list-unit-files
├── 运行级别：get-default | set-default | isolate
├── 日志管理：journalctl -u
└── 配置维护：daemon-reload | mask | unmask
```

---

### 5.11 常见问题与技巧

* **修改了配置文件但不生效？**
  → 执行 `systemctl daemon-reload` 让 systemd 重新读取配置。

* **服务崩溃了如何恢复？**
  → 可执行 `systemctl restart 服务名` 或检查日志 `journalctl -u 服务名`。

* **如何确认 systemd 正在使用？**

  ```bash
  ps -p 1 -o comm=
  ```

  输出结果若为 `systemd`，说明系统正在使用 systemd 管理。

---

## 6、进程管理

### 6.1 进程基础知识

**1️⃣ 什么是进程（Process）？**

> 进程是系统中正在运行的程序，是 CPU 资源分配和调度的基本单位。
> 每个进程都有一个唯一的 **PID（Process ID）**。

**2️⃣ 进程的状态（STAT列含义）：**

| 状态字母 | 含义                      |
| ---- | ----------------------- |
| R    | 运行中（Running）            |
| S    | 可中断睡眠（Sleeping）         |
| D    | 不可中断睡眠（I/O等待）           |
| T    | 停止（Stopped）             |
| Z    | 僵尸进程（Zombie）            |
| <    | 高优先级进程                  |
| N    | 低优先级进程                  |
| s    | 进程会话领导者（Session leader） |

---

### 6.2 查看进程信息的命令

---

#### 1️⃣ `ps` —— 查看当前系统的进程快照

**基本语法：**

```bash
ps [选项]
```

**常用选项组合：**

| 命令       | 含义                 |
| -------- | ------------------ |
| `ps`     | 查看当前终端下的进程         |
| `ps -a`  | 查看所有用户的终端进程        |
| `ps -u`  | 以用户为中心显示详细信息       |
| `ps -x`  | 显示无控制终端的进程（后台）     |
| `ps aux` | 显示所有进程的详细信息（常用）    |
| `ps -ef` | 以全格式显示所有进程（另一常用格式） |

**常见输出列说明：**

| 列名      | 含义     |
| ------- | ------ |
| USER    | 所属用户   |
| PID     | 进程号    |
| %CPU    | CPU占用率 |
| %MEM    | 内存占用率  |
| VSZ     | 虚拟内存大小 |
| RSS     | 实际占用内存 |
| STAT    | 进程状态   |
| START   | 启动时间   |
| COMMAND | 启动命令   |

**示例：**

```bash
$ ps aux | grep ssh
root     1025  0.0  0.1  14568  1234 ?  Ss   10:00   0:00 /usr/sbin/sshd
```

💡 **技巧：** `ps aux --sort=-%mem` 可以按内存使用排序。

---

#### 2️⃣ `top` —— 实时动态查看系统进程状态

**功能：**
显示系统运行状态、CPU/内存使用率、各进程资源占用情况。

**常用快捷键：**

| 按键  | 功能             |
| --- | -------------- |
| `q` | 退出             |
| `P` | 按 CPU 占用率排序    |
| `M` | 按内存占用率排序       |
| `T` | 按运行时间排序        |
| `k` | 杀死进程（输入 PID）   |
| `1` | 展示每个 CPU 的使用情况 |
| `h` | 帮助说明           |

**示例：**

```bash
$ top
top - 14:23:41 up 2 days,  3:10,  3 users,  load average: 0.23, 0.35, 0.31
```

💡 **进阶版：** 安装 `htop` 提供更直观的彩色界面（需手动安装）。

---

#### 3️⃣ `pstree` —— 以树状结构显示进程关系

**语法：**

```bash
pstree [选项]
```

**常用选项：**

| 参数   | 含义      |
| ---- | ------- |
| `-p` | 显示 PID  |
| `-u` | 显示用户    |
| `-a` | 显示完整命令行 |

**示例：**

```bash
pstree -p
systemd(1)─┬─NetworkManager(720)─┬─dhclient(900)
            ├─sshd(1025)─┬─bash(1200)───top(1320)
```

💡 **用途：** 可快速查看父子进程关系，判断服务层级结构。

---

#### 4️⃣ `pidof` —— 获取进程 PID

**示例：**

```bash
pidof sshd
```

输出：

```
1025
```

---

#### 5️⃣ `pgrep` —— 按名称查找进程 PID

**语法：**

```bash
pgrep 进程名
```

**示例：**

```bash
pgrep nginx
```

可配合 `pkill` 批量终止。

---

### 6.3 进程控制命令

---

#### 1️⃣ `kill` —— 根据 PID 终止进程

**语法：**

```bash
kill [-信号] PID
```

**常用信号：**

| 信号编号 | 名称      | 说明             |
| ---- | ------- | -------------- |
| 1    | SIGHUP  | 重新加载配置         |
| 9    | SIGKILL | 强制终止进程（无法被捕获）  |
| 15   | SIGTERM | 默认终止信号（可被程序处理） |

**示例：**

```bash
kill 1234        # 发送默认终止信号
kill -9 1234     # 强制杀死进程
```

💡 **技巧：** 若 PID 不确定，可配合 `ps` 或 `pgrep` 使用。

---

#### 2️⃣ `pkill` —— 按进程名杀死

```bash
pkill nginx
```

等价于：

```bash
kill $(pidof nginx)
```

---

#### 3️⃣ `killall` —— 按命令名结束所有同名进程

```bash
killall httpd
```

💡 与 `pkill` 类似，但 `killall` 匹配命令名而非模式。

---

### 6.4 后台与作业控制


#### 1️⃣ 后台运行与挂起

| 操作       | 命令          | 说明           |
| -------- | ----------- | ------------ |
| 后台运行命令   | `command &` | 在后台运行，不占用终端  |
| 查看后台任务   | `jobs`      | 显示当前会话的后台任务  |
| 重新切回前台   | `fg %n`     | 将第 n 个任务放回前台 |
| 挂起任务     | `Ctrl + Z`  | 暂停任务并挂起后台    |
| 恢复运行（后台） | `bg %n`     | 在后台恢复任务运行    |

**示例：**

```bash
sleep 100 &
jobs
fg %1
```

---

#### 2️⃣ 永久后台运行

| 命令                | 说明                   |
| ----------------- | -------------------- |
| `nohup command &` | 忽略挂起信号，即使注销仍继续运行     |
| `disown`          | 将任务从当前 shell 作业列表中移除 |

**示例：**

```bash
nohup python3 server.py &
disown %1
```

---

### 6.5 进程优先级控制

---

#### 1️⃣ `nice` —— 启动进程时指定优先级

**语法：**

```bash
nice -n <优先级> command
```

**说明：**

* 优先级范围：`-20`（最高）～ `19`（最低）
* 普通用户仅能调高数值（降低优先级）

**示例：**

```bash
nice -n 10 python3 run_task.py
```

---

#### 2️⃣ `renice` —— 修改已运行进程的优先级

**语法：**

```bash
renice <优先级> -p <PID>
```

**示例：**

```bash
renice -5 -p 1320
```

💡 优先级越低，进程获得 CPU 调度的机会越多。

---

### 6.6 进程监控与统计

---

#### 1️⃣ `pidstat` —— 按进程统计 CPU/内存使用（需安装 `sysstat`）

```bash
pidstat -u 2
```

→ 每2秒刷新一次CPU使用情况。

---

#### 2️⃣ `vmstat` —— 系统整体资源统计

```bash
vmstat 2 5
```

→ 每2秒输出一次，共输出5次系统整体运行指标。

---

### 6.7 实用命令汇总表

| 操作目标   | 命令示例                 | 说明          |       |
| ------ | -------------------- | ----------- | ----- |
| 查看所有进程 | `ps -ef`             | 全格式查看       |       |
| 查找特定进程 | `ps aux              | grep nginx` | 匹配关键字 |
| 查看进程树  | `pstree -p`          | 树状结构        |       |
| 实时查看进程 | `top`                | 动态刷新        |       |
| 杀死进程   | `kill -9 PID`        | 强制终止        |       |
| 按名称杀死  | `pkill httpd`        | 结束所有同名进程    |       |
| 查看后台任务 | `jobs`               | 显示作业列表      |       |
| 恢复后台任务 | `fg %1`              | 前台恢复        |       |
| 设置优先级  | `nice -n 10 command` | 启动时设定       |       |
| 修改优先级  | `renice -5 -p 1234`  | 动态调整        |       |

---

### 6.8 进程管理思维导图式总结

```
进程管理命令
├── 查看进程信息
│   ├─ ps / top / pstree / pgrep / pidof
│
├── 控制进程
│   ├─ kill / pkill / killall
│   ├─ jobs / fg / bg / nohup
│
├── 优先级调整
│   ├─ nice / renice
│
└── 监控与统计
    ├─ vmstat / pidstat / top
```

---

### 🧠 小结记忆口诀

> 查进程看 `ps`，动态看 `top`；
> 杀进程用 `kill`，批量用 `pkill`；
> 后台加 `&`，挂起用 `Ctrl+Z`；
> 优先级调 `nice`，监控别忘 `vmstat`！

---

## 7、内存和存储管理

### 7.1 内存管理基础知识

#### 1️⃣ 什么是内存管理？

> 内存管理是操作系统对 RAM 的分配、回收与调度机制。
> 在 Linux 中，一切皆文件，内存的使用状态、缓存、交换分区都可被查询。

#### 2️⃣ 内存相关概念：

| 名称                 | 说明                      |
| ------------------ | ----------------------- |
| **MemTotal**       | 总内存大小                   |
| **MemFree**        | 空闲内存                    |
| **Buffers/Cached** | 文件缓存，用于加速磁盘访问           |
| **Swap**           | 交换分区，用于临时扩展内存           |
| **Used**           | 实际已用内存（= 总内存 - 空闲 - 缓存） |

---

### 7.2 查看内存使用情况的命令


#### 1️⃣ `free` —— 显示系统内存使用状态

**语法：**

```bash
free [选项]
```

**常用选项：**

| 选项     | 含义               |
| ------ | ---------------- |
| `-h`   | 以人类可读方式（MB/GB）显示 |
| `-s N` | 每隔 N 秒刷新一次       |
| `-t`   | 显示物理内存 + 交换区总和   |

**示例：**

```bash
free -h
```

输出示例：

```
              total        used        free      shared  buff/cache   available
Mem:           15Gi       3.2Gi       8.4Gi       320Mi       3.5Gi        11Gi
Swap:          2.0Gi       0.0Gi       2.0Gi
```

💡 **技巧：**

* `used` 并不等于“真正占用”，应参考 `available` 才能反映系统可用内存。

---

#### 2️⃣ `vmstat` —— 系统整体性能与内存统计

**语法：**

```bash
vmstat [间隔秒数] [次数]
```

**示例：**

```bash
vmstat 2 5
```

输出字段说明：

| 字段      | 含义       |
| ------- | -------- |
| `r`     | 正在运行的进程数 |
| `b`     | 阻塞进程数    |
| `swpd`  | 使用的交换区   |
| `free`  | 空闲内存     |
| `buff`  | 缓冲区内存    |
| `cache` | 缓存内存     |

---

#### 3️⃣ `top` / `htop` —— 实时监控内存使用

* 在 `top` 中可直接查看 **%MEM** 和 **VIRT / RES / SHR** 等列。
* 按 **M** 键可按内存占用排序。

---

#### 4️⃣ `cat /proc/meminfo` —— 查看详细内存指标

**示例：**

```bash
cat /proc/meminfo
```

该文件包含系统内核维护的所有内存参数，适合深入分析。

---

#### 5️⃣ `sar -r` —— 查看历史内存使用情况（需安装 `sysstat`）

```bash
sar -r 2 5
```

→ 每2秒刷新一次，共输出5次内存使用记录。

---

### 7.3 交换分区（Swap）管理命令

#### 1️⃣ 查看交换区使用情况

```bash
swapon --show
free -h
```

#### 2️⃣ 启用交换区

```bash
sudo swapon /swapfile
```

#### 3️⃣ 禁用交换区

```bash
sudo swapoff /swapfile
```

#### 4️⃣ 创建新的交换文件

```bash
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

💡 **永久启用：**
将 `/swapfile none swap sw 0 0` 写入 `/etc/fstab`。

---

### 7.4 磁盘与存储管理命令

#### 1️⃣ `df` —— 查看文件系统磁盘使用情况

**语法：**

```bash
df [选项]
```

**常用选项：**

| 选项   | 说明            |
| ---- | ------------- |
| `-h` | 以人类可读格式显示     |
| `-T` | 显示文件系统类型      |
| `-i` | 显示 inode 使用情况 |

**示例：**

```bash
df -hT
```

输出示例：

```
Filesystem     Type  Size  Used Avail Use% Mounted on
/dev/sda2      ext4  200G   30G  160G  16% /
tmpfs          tmpfs 7.8G  1.2M  7.8G   1% /run
```

---

#### 2️⃣ `du` —— 查看目录或文件的磁盘占用量

**语法：**

```bash
du [选项] [路径]
```

**常用选项：**

| 选项              | 说明        |
| --------------- | --------- |
| `-h`            | 以人类可读格式显示 |
| `-s`            | 只统计总和     |
| `--max-depth=N` | 限制目录层级    |

**示例：**

```bash
du -sh /home
du -h --max-depth=1 /var/log
```

---

#### 3️⃣ `lsblk` —— 显示块设备信息（磁盘结构树）

**示例：**

```bash
lsblk
```

输出示例：

```
NAME   MAJ:MIN RM  SIZE RO TYPE MOUNTPOINTS
sda      8:0    0  512G  0 disk
├─sda1   8:1    0  512M  0 part /boot
└─sda2   8:2    0  511G  0 part /
```

💡 适合查看磁盘分区结构。

---

#### 4️⃣ `fdisk` / `parted` —— 分区工具

**查看磁盘分区信息：**

```bash
sudo fdisk -l
```

**交互式修改分区：**

```bash
sudo fdisk /dev/sda
```

💡 对 GPT 分区表可使用 `parted` 命令。

---

#### 5️⃣ `mount` / `umount` —— 挂载与卸载文件系统

**挂载命令：**

```bash
sudo mount /dev/sdb1 /mnt/usb
```

**卸载命令：**

```bash
sudo umount /mnt/usb
```

💡 永久挂载可编辑 `/etc/fstab` 文件。

---

#### 6️⃣ `blkid` —— 查看磁盘UUID与文件系统类型

**示例：**

```bash
sudo blkid
```

输出示例：

```
/dev/sda1: UUID="e2b0..." TYPE="ext4"
/dev/sdb1: UUID="F1A8-3D9A" TYPE="vfat"
```

---

#### 7️⃣ `df` + `du` 实战技巧

| 场景          | 命令                      |          |       |
| ----------- | ----------------------- | -------- | ----- |
| 查看系统磁盘总体情况  | `df -h`                 |          |       |
| 查找哪个文件夹占用最大 | `du -h --max-depth=1 /  | sort -hr | head` |
| 查看某个用户目录的空间 | `du -sh /home/username` |          |       |

---

### 7.5 磁盘健康与性能监测

#### 1️⃣ `iostat` —— 查看磁盘I/O统计（需安装 `sysstat`）

```bash
iostat -x 2 3
```

显示每个磁盘的读写速率、I/O利用率。

---

#### 2️⃣ `iotop` —— 实时查看进程I/O占用（需安装）

```bash
sudo iotop
```

可显示哪个进程在大量读写磁盘。

---

#### 3️⃣ `df -i` —— 查看 inode 使用情况

文件系统 inode 耗尽时会导致无法新建文件，即使磁盘仍有空间。

---

#### 4️⃣ `smartctl` —— 检测硬盘健康状态（需安装 smartmontools）

```bash
sudo smartctl -a /dev/sda
```

查看硬盘SMART信息、通电时间、坏道统计等。

---

### 7.6 文件系统管理命令

#### 1️⃣ `mkfs` —— 创建文件系统

**语法：**

```bash
sudo mkfs -t ext4 /dev/sdb1
```

常见类型：

* `ext4`：Linux常用文件系统
* `vfat`：兼容Windows的FAT32
* `xfs`：高性能日志文件系统

---

#### 2️⃣ `fsck` —— 检查并修复文件系统错误

```bash
sudo fsck -y /dev/sdb1
```

💡 建议在卸载状态下执行。

---

### 7.7 系统磁盘与内存汇总命令表

| 功能       | 命令                  | 说明          |
| -------- | ------------------- | ----------- |
| 查看内存使用   | `free -h`           | 内存占用与Swap状态 |
| 动态内存监控   | `vmstat 2`          | 实时刷新        |
| 查看详细内存信息 | `cat /proc/meminfo` | 查看内核级指标     |
| 查看磁盘使用   | `df -hT`            | 文件系统占用      |
| 查看目录大小   | `du -sh /path`      | 文件夹占用       |
| 查看分区结构   | `lsblk`             | 磁盘结构树       |
| 查看分区UUID | `blkid`             | 磁盘标识        |
| 管理挂载     | `mount / umount`    | 挂载与卸载       |
| 创建文件系统   | `mkfs -t ext4`      | 格式化分区       |
| 检查文件系统   | `fsck`              | 修复错误        |
| 查看I/O性能  | `iostat / iotop`    | 磁盘负载监测      |
| 查看Swap状态 | `swapon --show`     | 当前交换分区      |

---

### 7.8 快速记忆口诀

> 查内存看 `free`，监控用 `vmstat`；
> 看磁盘用 `df`，查目录配 `du`；
> 查分区用 `lsblk`，修系统靠 `fsck`；
> 挂载靠 `mount`，性能看 `iostat`！

---

## 8、用户管理

### 8.1 查看用户信息

| 指令              | 功能                   | 示例                |
| --------------- | -------------------- | ----------------- |
| `who`           | 查看当前登录的用户            | `who`             |
| `w`             | 查看当前登录用户及正在执行的命令     | `w`               |
| `id`            | 查看指定用户的 UID、GID 及所属组 | `id username`     |
| `finger`        | 查看用户详细信息（需安装）        | `finger username` |
| `getent passwd` | 查看系统所有用户             | `getent passwd`   |

---

### 8.2 添加用户

| 指令                     | 功能           | 示例                                      |
| ---------------------- | ------------ | --------------------------------------- |
| `useradd`              | 添加用户（仅创建账户）  | `sudo useradd testuser`                 |
| `useradd -m`           | 添加用户并创建主目录   | `sudo useradd -m testuser`              |
| `useradd -s /bin/bash` | 指定用户登录 shell | `sudo useradd -m -s /bin/bash testuser` |
| `passwd`               | 设置或修改用户密码    | `sudo passwd testuser`                  |

**备注：**

* 用户主目录默认在 `/home/用户名`。
* `-d /路径` 可以自定义主目录。

---

### 8.3 删除用户

| 指令                    | 功能           | 示例                         |
| --------------------- | ------------ | -------------------------- |
| `userdel username`    | 删除用户（不删除主目录） | `sudo userdel testuser`    |
| `userdel -r username` | 删除用户及其主目录    | `sudo userdel -r testuser` |

---

### 8.4 修改用户

| 指令                        | 功能         | 示例                                         |
| ------------------------- | ---------- | ------------------------------------------ |
| `usermod -l 新用户名 旧用户名`    | 修改用户名      | `sudo usermod -l newuser olduser`          |
| `usermod -d /新/目录 -m 用户名` | 修改主目录并移动文件 | `sudo usermod -d /home/newdir -m testuser` |
| `usermod -s /bin/zsh 用户名` | 修改登录 shell | `sudo usermod -s /bin/zsh testuser`        |
| `usermod -G 组名 用户名`       | 添加附加组      | `sudo usermod -aG sudo testuser`           |

**备注：** `-a` 参数非常重要，防止覆盖原有组。

---

## 9、用户组管理

### 9.1 查看组信息

| 指令               | 功能        | 示例                |
| ---------------- | --------- | ----------------- |
| `groups`         | 查看用户所属组   | `groups testuser` |
| `getent group`   | 查看系统所有组   | `getent group`    |
| `cat /etc/group` | 查看所有组及组成员 | `cat /etc/group`  |

---

### 9.2 添加/删除组

| 指令                    | 功能   | 示例                                 |
| --------------------- | ---- | ---------------------------------- |
| `groupadd 组名`         | 添加新组 | `sudo groupadd developers`         |
| `groupdel 组名`         | 删除组  | `sudo groupdel developers`         |
| `groupmod -n 新组名 旧组名` | 修改组名 | `sudo groupmod -n devs developers` |

---

### 9.3 用户与组的关联

| 指令                  | 功能      | 示例                               |
| ------------------- | ------- | -------------------------------- |
| `usermod -G 组名 用户名` | 修改用户附加组 | `sudo usermod -aG sudo testuser` |
| `gpasswd -a 用户名 组名` | 将用户加入组  | `sudo gpasswd -a testuser sudo`  |
| `gpasswd -d 用户名 组名` | 将用户移出组  | `sudo gpasswd -d testuser sudo`  |

---

### 9.3 权限与账号控制

| 指令                | 功能         | 示例                          |
| ----------------- | ---------- | --------------------------- |
| `chage -l 用户名`    | 查看密码过期信息   | `chage -l testuser`         |
| `chage -M 90 用户名` | 设置密码最大使用天数 | `sudo chage -M 90 testuser` |
| `usermod -L 用户名`  | 锁定账户（禁止登录） | `sudo usermod -L testuser`  |
| `usermod -U 用户名`  | 解锁账户       | `sudo usermod -U testuser`  |

---

### 9.4 常用快捷命令

* `whoami`：显示当前用户
* `su 用户名`：切换用户
* `sudo -i`：以 root 身份登录 shell
* `sudo command`：以 root 权限执行命令

---

## 10、文件管理

### 10.1 文件和目录操作

| 命令              | 功能          | 示例                                           |
| --------------- | ----------- | -------------------------------------------- |
| `ls`            | 列出目录内容      | `ls` / `ls -l` / `ls -a`                     |
| `cd`            | 切换目录        | `cd /home/lxlei` / `cd ..`                   |
| `pwd`           | 显示当前工作目录    | `pwd`                                        |
| `mkdir`         | 创建目录        | `mkdir mydir` / `mkdir -p a/b/c`             |
| `rmdir`         | 删除空目录       | `rmdir mydir`                                |
| `rm -r`         | 删除目录及其内容    | `rm -r mydir`                                |
| `touch`         | 创建空文件或更新时间戳 | `touch file.txt`                             |
| `cp`            | 复制文件或目录     | `cp file1.txt file2.txt` / `cp -r dir1 dir2` |
| `mv`            | 移动或重命名文件/目录 | `mv file.txt /tmp/` / `mv oldname newname`   |
| `cat`           | 查看文件内容      | `cat file.txt`                               |
| `more` / `less` | 分页查看文件      | `less file.txt`                              |
| `head`          | 查看文件开头内容    | `head -n 10 file.txt`                        |
| `tail`          | 查看文件结尾内容    | `tail -n 10 file.txt`                        |
| `stat`          | 查看文件详细信息    | `stat file.txt`                              |

---

### 10.2 文件搜索与查找

| 命令        | 功能                    | 示例                                                    |
| --------- | --------------------- | ----------------------------------------------------- |
| `find`    | 查找文件/目录               | `find /home -name "*.txt"`                            |
| `locate`  | 快速查找文件（需更新数据库）        | `locate file.txt` / `sudo updatedb`                   |
| `which`   | 查看命令路径                | `which python3`                                       |
| `whereis` | 查找命令二进制文件、源代码和 man 页面 | `whereis ls`                                          |
| `grep`    | 文件内容搜索                | `grep "pattern" file.txt` / `grep -r "pattern" /home` |

---

### 10.3 文件权限管理

| 命令      | 功能          | 示例                                        |
| ------- | ----------- | ----------------------------------------- |
| `chmod` | 修改文件权限      | `chmod 755 file.sh` / `chmod u+x file.sh` |
| `chown` | 修改文件所有者     | `chown lxlei file.txt`                    |
| `chgrp` | 修改文件所属组     | `chgrp users file.txt`                    |
| `umask` | 查看/设置默认权限掩码 | `umask` / `umask 022`                     |
| `ls -l` | 查看权限信息      | `ls -l file.txt`                          |

---

### 10.4 硬链接与软链接

| 命令         | 功能          | 示例                                      |
| ---------- | ----------- | --------------------------------------- |
| `ln`       | 创建硬链接       | `ln file1.txt file2.txt`                |
| `ln -s`    | 创建软链接（符号链接） | `ln -s /home/lxlei/file1.txt link1.txt` |
| `readlink` | 查看软链接指向     | `readlink link1.txt`                    |

---

### 10.5 文件压缩与解压

| 命令                  | 功能           | 示例                                                           |
| ------------------- | ------------ | ------------------------------------------------------------ |
| `tar`               | 打包和解包        | `tar -cvf archive.tar dir/` / `tar -xvf archive.tar`         |
| `tar + gzip`        | 打包并压缩        | `tar -czvf archive.tar.gz dir/` / `tar -xzvf archive.tar.gz` |
| `zip` / `unzip`     | 压缩/解压 zip 文件 | `zip -r archive.zip dir/` / `unzip archive.zip`              |
| `gzip` / `gunzip`   | 压缩/解压单文件     | `gzip file.txt` / `gunzip file.txt.gz`                       |
| `bzip2` / `bunzip2` | bz2 压缩/解压    | `bzip2 file.txt` / `bunzip2 file.txt.bz2`                    |
| `xz` / `unxz`       | xz 压缩/解压     | `xz file.txt` / `unxz file.txt.xz`                           |

---

### 10.6 磁盘和文件系统相关

| 命令                 | 功能                  | 示例                                     |
| ------------------ | ------------------- | -------------------------------------- |
| `df -h`            | 查看磁盘空间使用情况          | `df -h`                                |
| `du -h`            | 查看目录/文件大小           | `du -sh /home/lxlei`                   |
| `mount` / `umount` | 挂载/卸载文件系统           | `mount /dev/sdb1 /mnt` / `umount /mnt` |
| `stat`             | 查看文件详细信息（大小、权限、时间等） | `stat file.txt`                        |

---

### 10.7 其他常用文件管理命令

| 命令         | 功能            | 示例                                             |
| ---------- | ------------- | ---------------------------------------------- |
| `file`     | 查看文件类型        | `file file.txt`                                |
| `diff`     | 比较文件内容差异      | `diff file1.txt file2.txt`                     |
| `cmp`      | 按字节比较文件       | `cmp file1.txt file2.txt`                      |
| `touch`    | 更新文件时间戳或创建空文件 | `touch newfile.txt`                            |
| `basename` | 获取文件名         | `basename /home/lxlei/file.txt` → `file.txt`   |
| `dirname`  | 获取目录名         | `dirname /home/lxlei/file.txt` → `/home/lxlei` |

---

### **总结**

1. **文件操作**：`ls` / `cd` / `cp` / `mv` / `rm`
2. **查找搜索**：`find` / `locate` / `grep`
3. **权限管理**：`chmod` / `chown` / `chgrp`
4. **压缩解压**：`tar` / `gzip` / `zip`
5. **磁盘管理**：`df` / `du` / `mount`

---

## 11、文件权限的查看与修改

### 11.1 Linux 文件权限概念

在 Linux 中，每个文件和目录都有 **三类用户的权限**：

1. **用户 (User, u)**：文件的所有者
2. **组 (Group, g)**：文件所属组内的用户
3. **其他人 (Others, o)**：除了用户和组以外的所有人

每类用户都有 **三种基本权限**：

| 权限 | 符号  | 含义                           |
| -- | --- | ---------------------------- |
| 读  | `r` | 可以查看文件内容（文件）或列出目录内容（目录）      |
| 写  | `w` | 可以修改文件内容（文件）或在目录下创建/删除文件（目录） |
| 执行 | `x` | 可以执行文件（可执行程序或脚本）或进入目录（目录）    |

---

#### **文件权限显示格式**

使用 `ls -l` 查看文件权限：

```
-rwxr-xr-- 1 lxlei lxgroup 1024 2025-10-19 file.txt
```

解释：

```
-rwxr-xr--
| |   |  |     |       |
类型 权限 链接 数量 用户 组   大小  修改时间 文件名
```

1. **文件类型（首字符）**：

   * `-`：普通文件
   * `d`：目录
   * `l`：符号链接
   * `c`：字符设备
   * `b`：块设备

2. **权限字段（9个字符）**：

   ```
   rwx r-x r--
   u   g   o
   ```

   * 前三位 (`rwx`) → 用户权限
   * 中三位 (`r-x`) → 组权限
   * 后三位 (`r--`) → 其他人权限

---

#### **特殊权限**

| 权限         | 符号  | 含义                          |
| ---------- | --- | --------------------------- |
| Setuid     | `s` | 执行文件时以文件所有者权限运行             |
| Setgid     | `s` | 执行文件时以文件所属组权限运行，或目录中新建文件继承组 |
| Sticky bit | `t` | 目录中只有文件所有者可删除文件（常用于 `/tmp`） |

---

### 11.2 查看文件权限

**1. ls -l**

```bash
ls -l file.txt
ls -l /home/lxlei
```

输出示例：

```
-rw-r--r-- 1 lxlei lxgroup 1024 Oct 19 12:00 file.txt
drwxr-xr-x 2 lxlei lxgroup 4096 Oct 19 12:00 mydir
```

**2. stat**

```bash
stat file.txt
```

输出示例：

```
  File: file.txt
  Size: 1024       Blocks: 8      IO Block: 4096 regular file
Device: 801h/2049d Inode: 1234567 Links: 1
Access: (0644/-rw-r--r--)  Uid: (1000/lxlei)   Gid: (1000/lxgroup)
Access: 2025-10-19 12:00:00
Modify: 2025-10-19 12:00:00
Change: 2025-10-19 12:00:00
```

* `Access: (0644/-rw-r--r--)` → 权限的 **八进制数和符号形式**

**3. id / groups**

查看当前用户 ID 和所属组：

```bash
id lxlei
groups lxlei
```

---

### 11.3 修改文件权限

Linux 提供两种方式修改权限：**符号模式**和**数字模式（八进制）**。

---

#### 1. chmod 修改权限

**(1) 符号模式**

格式：

```bash
chmod [ugoa][+-=][rwx] 文件
```

* `u` → 用户
* `g` → 组
* `o` → 其他
* `a` → 全部用户
* `+` → 添加权限
* `-` → 去掉权限
* `=` → 设置精确权限

**示例：**

```bash
chmod u+x file.sh       # 给用户添加执行权限
chmod go-w file.txt     # 去掉组和其他人的写权限
chmod a=r file.txt      # 所有人只读权限
```

---

**(2) 数字模式**

* 权限用数字表示（八进制）：

  * `r` = 4
  * `w` = 2
  * `x` = 1
* 数字权限按 **用户-组-其他** 顺序组合

**示例：**

```bash
chmod 755 file.sh   # rwx r-x r-x
chmod 644 file.txt  # rw- r-- r--
chmod 700 secret.sh # rwx --- ---
```

---

#### 2. chown 修改文件所有者

```bash
sudo chown 用户 文件
sudo chown 用户:组 文件
```

**示例：**

```bash
sudo chown lxlei file.txt       # 修改文件所有者
sudo chown lxlei:lxgroup file.txt  # 修改所有者和所属组
```

---

#### 3. chgrp 修改文件所属组

```bash
sudo chgrp 组 文件
```

**示例：**

```bash
sudo chgrp developers file.txt
```

---

### 11.4 查看和验证权限

**示例操作**

```bash
touch test.txt              # 创建文件
ls -l test.txt              # 查看权限
chmod u+x test.txt          # 给用户添加执行权限
chmod g-w test.txt          # 去掉组写权限
chmod 644 test.txt          # 设置权限 rw-r--r--
sudo chown lxlei:users test.txt  # 修改所有者和组
ls -l test.txt
```

输出：

```
-rw-r--r-- 1 lxlei users 0 Oct 19 12:00 test.txt
```

---

### 11.5 常用快捷指令总结

| 命令       | 功能              |
| -------- | --------------- |
| `ls -l`  | 查看权限和文件信息       |
| `stat`   | 查看详细权限及元信息      |
| `chmod`  | 修改权限（符号模式/数字模式） |
| `chown`  | 修改文件所有者         |
| `chgrp`  | 修改文件所属组         |
| `id`     | 查看用户 UID/GID    |
| `groups` | 查看用户所属组         |

---

💡 **总结**

1. **理解权限三类用户和三种基本权限**，这是 Linux 文件管理基础
2. **`ls -l` + `stat`** 是查看权限最常用的方法
3. **修改权限**：`chmod`（符号/数字模式）
4. **修改所有者或组**：`chown`、`chgrp`

---

## 12、系统网络配置

本章涵盖：

> 网络基础概念 → 网络配置 → DNS → 静态 IP → 防火墙管理（firewalld）

---

### 12.1 网络基本概念

| 概念                                             | 说明                                                     |
| ---------------------------------------------- | ------------------------------------------------------ |
| **IP地址 (Internet Protocol Address)**           | 网络中每台设备的唯一标识；分为 IPv4（如 192.168.1.10）和 IPv6（如 fe80::1）。 |
| **子网掩码 (Subnet Mask)**                         | 用于区分网络地址和主机地址，例如 `255.255.255.0` 表示前 24 位为网络号。         |
| **网关 (Gateway)**                               | 数据出局的“出口”，通常是路由器的 IP 地址。                               |
| **DNS (Domain Name System)**                   | 将域名解析为 IP 地址，例如把 `www.baidu.com` 转换为 `220.181.38.148`。 |
| **MAC 地址**                                     | 网卡硬件地址，用于局域网通信。                                        |
| **DHCP (Dynamic Host Configuration Protocol)** | 自动分配 IP 地址的协议。                                         |
| **静态 IP**                                      | 手动配置固定 IP 地址，重启后不会变化。                                  |

---

### 12.2 查看网络信息

#### 1. 查看网络接口状态

```bash
ip addr
# 或
ifconfig    # 需安装 net-tools：sudo yum install net-tools
```

#### 2. 查看路由信息

```bash
ip route
# 默认路由（网关）示例：
# default via 192.168.1.1 dev eth0
```

#### 3. 查看 DNS 配置

```bash
cat /etc/resolv.conf
```

输出示例：

```
nameserver 8.8.8.8
nameserver 114.114.114.114
```

#### 4. 网络测试命令

| 命令                | 作用             |
| ----------------- | -------------- |
| `ping <IP/域名>`    | 测试网络连通性        |
| `traceroute <IP>` | 跟踪数据包路径        |
| `netstat -tulnp`  | 查看监听端口与服务      |
| `ss -tulnp`       | 替代 netstat，更现代 |
| `nslookup <域名>`   | DNS 解析测试       |
| `curl <网址>`       | 测试 HTTP 请求响应   |
| `wget <网址>`       | 下载文件、测试网络      |

---

### 12.3 网络配置（CentOS / Ubuntu）

Linux 的网络接口名常见为 `eth0`、`ens33`、`enp0s3` 等。
配置文件路径略有不同：

| 系统                   | 网络配置文件                                      |
| -------------------- | ------------------------------------------- |
| **CentOS 7+**        | `/etc/sysconfig/network-scripts/ifcfg-eth0` |
| **Ubuntu (Netplan)** | `/etc/netplan/*.yaml`                       |

---

#### 1️⃣ **查看当前网络接口**

```bash
nmcli device status
```

示例：

```
DEVICE  TYPE      STATE      CONNECTION
eth0    ethernet  connected  eth0
lo      loopback  unmanaged  --
```

---

#### 2️⃣ **设置静态 IP（CentOS 7）**

编辑对应接口配置文件（例如 `/etc/sysconfig/network-scripts/ifcfg-eth0`）：

```bash
sudo vi /etc/sysconfig/network-scripts/ifcfg-eth0
```

修改内容如下：

```bash
TYPE=Ethernet
BOOTPROTO=static      # 改为静态
IPADDR=192.168.1.100  # 设置IP地址
NETMASK=255.255.255.0 # 子网掩码
GATEWAY=192.168.1.1   # 网关
DNS1=8.8.8.8          # 主DNS
DNS2=114.114.114.114  # 备用DNS
ONBOOT=yes            # 开机启用
```

保存后重启网络：

```bash
sudo systemctl restart network
```

或：

```bash
sudo nmcli connection reload
sudo nmcli connection up eth0
```

---

#### 3️⃣ **设置静态 IP（Ubuntu 20.04+）**

Ubuntu 使用 **Netplan** 配置：

编辑文件 `/etc/netplan/01-netcfg.yaml`：

```yaml
network:
  version: 2
  renderer: networkd
  ethernets:
    enp0s3:
      dhcp4: no
      addresses: [192.168.1.100/24]
      gateway4: 192.168.1.1
      nameservers:
        addresses: [8.8.8.8, 114.114.114.114]
```

保存后应用：

```bash
sudo netplan apply
```

---

### 12.4 DNS 配置

**1. 临时修改 DNS（立即生效）**

```bash
sudo vi /etc/resolv.conf
```

添加：

```
nameserver 8.8.8.8
nameserver 114.114.114.114
```

> ⚠️ 注意：该文件在重启后可能被 NetworkManager 覆盖。

---

#### 2. 永久修改 DNS（CentOS）

编辑 `/etc/sysconfig/network-scripts/ifcfg-eth0`：

```bash
DNS1=8.8.8.8
DNS2=114.114.114.114
```

或者在 NetworkManager 中修改：

```bash
nmcli con mod eth0 ipv4.dns "8.8.8.8 114.114.114.114"
nmcli con up eth0
```

---

### 12.5 防火墙管理（firewalld）

CentOS 7+ 默认使用 **firewalld** 管理防火墙规则。

**1. 启动 / 停止 / 状态**

```bash
sudo systemctl start firewalld
sudo systemctl stop firewalld
sudo systemctl enable firewalld
sudo systemctl status firewalld
```

---

**2. 基本命令**

| 命令                                                | 说明        |
| ------------------------------------------------- | --------- |
| `firewall-cmd --state`                            | 查看防火墙是否运行 |
| `firewall-cmd --list-all`                         | 查看当前区域规则  |
| `firewall-cmd --zone=public --list-ports`         | 查看已开放端口   |
| `firewall-cmd --add-port=8080/tcp --permanent`    | 永久开放端口    |
| `firewall-cmd --remove-port=8080/tcp --permanent` | 永久关闭端口    |
| `firewall-cmd --reload`                           | 重新加载规则生效  |

---

### **3. 开放常见端口**

```bash
sudo firewall-cmd --zone=public --add-service=http --permanent
sudo firewall-cmd --zone=public --add-port=22/tcp --permanent   # SSH
sudo firewall-cmd --zone=public --add-port=80/tcp --permanent   # Web
sudo firewall-cmd --zone=public --add-port=443/tcp --permanent  # HTTPS
sudo firewall-cmd --reload
```

---

### 12.6 网络服务命令总结表

| 功能     | 命令                          |
| ------ | --------------------------- |
| 查看网络接口 | `ip addr` / `ifconfig`      |
| 查看路由   | `ip route`                  |
| 测试连通性  | `ping <IP>`                 |
| DNS 测试 | `nslookup <域名>`             |
| 修改网络配置 | `nmcli` / `vi ifcfg-xxx`    |
| 重启网络服务 | `systemctl restart network` |
| 修改防火墙  | `firewall-cmd`              |
| 查看端口占用 | `ss -tulnp`                 |
| 查看网络统计 | `netstat -anp`              |

---

### 12.7 总结

✅ 网络三要素：`IP + 子网掩码 + 网关`
✅ DNS 负责域名解析
✅ 静态 IP 通过配置文件或 Netplan 设置
✅ 防火墙主要用 `firewall-cmd` 管理端口规则
✅ 常见网络诊断命令：`ping`、`traceroute`、`nslookup`、`ss`

---

## 13、软件安装 

### 13.1 Linux 软件包管理概念

Linux 中的软件一般以 **包（package）** 的形式分发和安装。
不同发行版使用不同的包管理工具：

| 发行版                      | 包格式    | 管理工具                  |
| ------------------------ | ------ | --------------------- |
| CentOS / RedHat / Fedora | `.rpm` | `yum` / `dnf` / `rpm` |
| Ubuntu / Debian          | `.deb` | `apt` / `dpkg`        |

包管理器的主要功能包括：

* 安装、更新、删除软件
* 自动解决依赖关系
* 管理软件源（repositories）
* 查询已安装的软件信息

---

### 13.2 RedHat / CentOS 系列（YUM / DNF）

> **YUM (Yellowdog Updater, Modified)** 是 CentOS 7 及以前版本的默认包管理器
> **DNF (Dandified YUM)** 是 CentOS 8+、Fedora 默认包管理器，语法几乎一致。

---

#### **1️⃣ 软件安装**

```bash
sudo yum install 软件名
# 示例：
sudo yum install vim
sudo yum install net-tools
```

若系统较新：

```bash
sudo dnf install vim
```

---

#### **2️⃣ 卸载软件**

```bash
sudo yum remove 软件名
```

例如：

```bash
sudo yum remove httpd
```

---

#### **3️⃣ 更新软件**

```bash
sudo yum update
```

更新单个软件：

```bash
sudo yum update vim
```

---

#### **4️⃣ 搜索软件包**

```bash
yum search 软件名关键字
```

例如：

```bash
yum search nginx
```

---

#### **5️⃣ 查看软件包信息**

```bash
yum info 软件名
```

---

#### **6️⃣ 列出所有已安装软件**

```bash
yum list installed
```

---

#### **7️⃣ 清理缓存**

```bash
yum clean all
yum makecache
```

（常用于更新镜像源或解决“metadata expired”错误）

---

#### **8️⃣ 本地安装 .rpm 包**

```bash
sudo rpm -ivh package.rpm       # 安装
sudo rpm -Uvh package.rpm       # 升级
sudo rpm -e package_name        # 卸载
rpm -qa                         # 查询所有已安装包
```

> ⚠️ 注意：`rpm` 不会自动解决依赖关系，建议优先使用 `yum` 或 `dnf`。

---

### 13.3 Ubuntu / Debian 系列（APT / DPKG）

> **APT (Advanced Package Tool)** 是 Debian / Ubuntu 的包管理工具
> **DPKG** 是底层工具，APT 是它的高级前端。

---

#### **1️⃣ 安装软件**

```bash
sudo apt update              # 更新软件源索引
sudo apt install 软件名
```

示例：

```bash
sudo apt install vim
sudo apt install net-tools
```

---

#### **2️⃣ 卸载软件**

```bash
sudo apt remove 软件名          # 删除软件但保留配置文件
sudo apt purge 软件名           # 连同配置文件一并删除
```

---

#### **3️⃣ 更新系统**

```bash
sudo apt update               # 更新包索引
sudo apt upgrade              # 更新所有可升级软件
sudo apt full-upgrade         # 包括依赖调整的升级
```

---

#### **4️⃣ 搜索软件**

```bash
apt search 软件名
```

---

#### **5️⃣ 查看软件信息**

```bash
apt show 软件名
```

---

#### **6️⃣ 清理缓存**

```bash
sudo apt clean                # 清除缓存包文件
sudo apt autoclean            # 清除旧版本缓存
sudo apt autoremove           # 删除无用依赖包
```

---

#### **7️⃣ 本地安装 .deb 包**

```bash
sudo dpkg -i package.deb      # 安装
sudo dpkg -r package_name     # 卸载
dpkg -l                       # 列出已安装软件
```

若依赖未解决，可执行：

```bash
sudo apt --fix-broken install
```

---

### 13.4 软件源管理

软件源（Repository）是软件安装的来源。

| 文件位置                    | 系统              |
| ----------------------- | --------------- |
| `/etc/yum.repos.d/`     | CentOS / RHEL   |
| `/etc/apt/sources.list` | Ubuntu / Debian |

### **示例：CentOS 7 换成阿里云源**

```bash
sudo mv /etc/yum.repos.d/CentOS-Base.repo /etc/yum.repos.d/CentOS-Base.repo.bak
sudo curl -o /etc/yum.repos.d/CentOS-Base.repo http://mirrors.aliyun.com/repo/Centos-7.repo
sudo yum clean all && sudo yum makecache
```

### **示例：Ubuntu 换成阿里云源**

编辑：

```bash
sudo vi /etc/apt/sources.list
```

替换为：

```
deb http://mirrors.aliyun.com/ubuntu/ focal main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ focal-updates main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ focal-security main restricted universe multiverse
```

保存并更新：

```bash
sudo apt update
```

---

### 13.5 源外软件安装方法

| 方法                 | 说明                          |
| ------------------ | --------------------------- |
| **源码安装**           | 下载 `.tar.gz` 压缩包，手动编译安装。    |
| **Snap / Flatpak** | 通用包管理工具，可跨发行版安装。            |
| **wget + make 安装** | 常见于开源项目（例如 nginx、redis 源码）。 |

---

#### **源码安装步骤通用模板**

```bash
tar -zxvf software.tar.gz
cd software
./configure
make
sudo make install
```

安装完后可用：

```bash
which 软件名   # 查看安装路径
```

---

### 13.6 常用软件包命令速查表

| 功能    | CentOS 命令          | Ubuntu 命令         |
| ----- | ------------------ | ----------------- |
| 安装软件  | `yum install xxx`  | `apt install xxx` |
| 卸载软件  | `yum remove xxx`   | `apt remove xxx`  |
| 更新系统  | `yum update`       | `apt upgrade`     |
| 搜索包   | `yum search xxx`   | `apt search xxx`  |
| 查看信息  | `yum info xxx`     | `apt show xxx`    |
| 清理缓存  | `yum clean all`    | `apt clean`       |
| 本地包安装 | `rpm -ivh xxx.rpm` | `dpkg -i xxx.deb` |

---

### 13.7 实用技巧与建议

1. **优先使用包管理器**，不要手动删除文件。
2. 安装前可先搜索包名，避免输入错误（如 `net-look` ➜ 应为 `net-tools`）。
3. 若 `yum` 无法联网，先检查 `/etc/yum.repos.d/` 源文件是否有效。
4. 定期执行：

   ```bash
   sudo yum update    # 或 sudo apt update && sudo apt upgrade
   ```

   保持系统安全和最新。

---

## 14、CenOS7 Ubuntu两大系统软件管理

### 14.1 软件包管理基础概念

在 Linux 系统中，软件并不是像 Windows 那样双击安装包执行 `.exe` 文件，而是通过「软件包管理器」统一安装、卸载和更新的。
软件通常被打包成两种格式：

| 系统系列                                  | 包格式    | 包管理工具                              | 底层命令   |
| ------------------------------------- | ------ | ---------------------------------- | ------ |
| **Red Hat 系（CentOS / Fedora / RHEL）** | `.rpm` | `yum`（CentOS 7） / `dnf`（CentOS 8+） | `rpm`  |
| **Debian 系（Ubuntu / Debian / Mint）**  | `.deb` | `apt`（或 `apt-get`）                 | `dpkg` |

### ✅ 结构层级

```
Yum（CentOS 7）   → 基于 RPM 封装，自动解决依赖
Apt（Ubuntu）     → 基于 Dpkg 封装，自动解决依赖
```

`rpm` 和 `dpkg` 负责底层的安装操作；
`yum` 和 `apt` 则提供智能依赖解决、联网下载、版本管理等高级功能。

---

### 14.2 CentOS 7：YUM 与 RPM 管理系统

#### 1️⃣ 基本原理

* **YUM（Yellowdog Updater, Modified）** 是 CentOS 7 的默认包管理器。
* 它会从配置的 **仓库（repository）** 下载软件包（.rpm 文件）并自动解决依赖。
* 所有软件包元数据保存在 `/etc/yum.repos.d/*.repo` 中。

---

#### 2️⃣ 常用操作命令

| 操作      | 命令示例                         | 说明                  |
| ------- | ---------------------------- | ------------------- |
| 安装软件    | `sudo yum install httpd -y`  | 安装 Apache；`-y` 自动确认 |
| 卸载软件    | `sudo yum remove httpd`      | 删除软件包               |
| 更新软件    | `sudo yum update httpd`      | 更新指定软件              |
| 更新系统    | `sudo yum update -y`         | 更新全部软件              |
| 搜索软件    | `yum search nginx`           | 模糊查找软件包名            |
| 查看信息    | `yum info nginx`             | 显示详细包信息             |
| 查看已安装软件 | `yum list installed`         | 列出系统中所有包            |
| 清理缓存    | `yum clean all`              | 清理旧包缓存              |
| 查看依赖关系  | `repoquery --requires httpd` | 显示依赖列表              |

---

#### 3️⃣ 管理软件源（仓库）

仓库文件存放在：

```
/etc/yum.repos.d/
```

示例文件：`CentOS-Base.repo`

```ini
[base]
name=CentOS-7 - Base
baseurl=http://mirrors.aliyun.com/centos/7/os/x86_64/
enabled=1
gpgcheck=1
gpgkey=http://mirrors.aliyun.com/centos/RPM-GPG-KEY-CentOS-7
```

#### ➕ 添加新的软件源：

```bash
sudo yum-config-manager --add-repo http://mirrors.aliyun.com/repo/Centos-7.repo
```

#### 🚫 禁用或启用仓库：

```bash
sudo yum-config-manager --disable epel
sudo yum-config-manager --enable base
```

---

#### 4️⃣ RPM 底层命令（手动安装与查询）

| 操作     | 命令                      | 说明        |
| ------ | ----------------------- | --------- |
| 安装本地包  | `sudo rpm -ivh xxx.rpm` | 不会自动解决依赖  |
| 卸载包    | `sudo rpm -e 包名`        | 删除软件      |
| 查看包信息  | `rpm -qi 包名`            | 软件详情      |
| 查询文件来源 | `rpm -qf /usr/bin/vim`  | 查询文件属于哪个包 |
| 列出所有包  | `rpm -qa`               | 查看已安装包    |

> ⚠️ 注意：直接使用 `rpm` 安装时依赖问题需手动解决，建议使用 `yum`。

---

### 14.3 Ubuntu：APT 与 DPKG 管理系统

#### 1️⃣ 基本原理

* **APT（Advanced Package Tool）** 是 Ubuntu 默认的包管理系统。
* 它从 `/etc/apt/sources.list` 和 `/etc/apt/sources.list.d/*.list` 中读取源地址，自动下载 `.deb` 包并处理依赖。

---

#### 2️⃣ 常用操作命令

| 操作      | 命令示例                                   | 说明         |
| ------- | -------------------------------------- | ---------- |
| 更新软件源索引 | `sudo apt update`                      | 从源中获取最新包列表 |
| 安装软件    | `sudo apt install vim -y`              | 安装包        |
| 卸载软件    | `sudo apt remove vim`                  | 删除软件但保留配置  |
| 卸载含配置   | `sudo apt purge vim`                   | 完全删除含配置文件  |
| 升级已安装包  | `sudo apt upgrade`                     | 升级所有包      |
| 升级系统版本  | `sudo apt dist-upgrade`                | 更智能的系统更新   |
| 查找软件    | `apt search nginx`                     | 模糊搜索       |
| 查看包信息   | `apt show nginx`                       | 详细信息       |
| 清理缓存    | `sudo apt clean && sudo apt autoclean` | 清除下载缓存     |
| 删除无用包   | `sudo apt autoremove`                  | 清理依赖残留     |

---

#### 3️⃣ DPKG 底层命令

| 操作     | 命令                         | 说明        |
| ------ | -------------------------- | --------- |
| 安装本地包  | `sudo dpkg -i xxx.deb`     | 手动安装      |
| 修复依赖   | `sudo apt install -f`      | 自动解决缺失依赖  |
| 查看已安装包 | `dpkg -l`                  | 列出系统所有包   |
| 查看文件来源 | `dpkg -S /usr/bin/python3` | 查询文件属于哪个包 |
| 删除包    | `sudo dpkg -r 包名`          | 删除已安装包    |

---

#### 4️⃣ 管理软件源

文件位置：

```
/etc/apt/sources.list
/etc/apt/sources.list.d/
```

#### 示例源配置：

```bash
deb http://mirrors.aliyun.com/ubuntu/ focal main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ focal-updates main restricted universe multiverse
```

#### ➕ 添加新源：

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
```

---

### 14.4 软件更新与维护策略

| 操作目标   | CentOS 7             | Ubuntu                             |
| ------ | -------------------- | ---------------------------------- |
| 更新缓存   | `yum makecache fast` | `apt update`                       |
| 更新单包   | `yum update nginx`   | `apt install --only-upgrade nginx` |
| 更新系统   | `yum update -y`      | `apt upgrade -y`                   |
| 自动清理旧包 | `yum autoremove`     | `apt autoremove`                   |
| 清除缓存   | `yum clean all`      | `apt clean`                        |

---

### 14.5 常见错误与修复方法

| 问题     | 解决命令                                                         |
| ------ | ------------------------------------------------------------ |
| 依赖冲突   | `sudo yum deplist 包名` 或 `sudo apt install -f`                |
| 软件源失效  | 更新镜像源地址（推荐阿里云或清华源）                                           |
| 包下载失败  | 检查 `/etc/resolv.conf` 的 DNS 配置                               |
| 系统缓存损坏 | `yum clean all && yum makecache` / `apt clean && apt update` |

---

### 14.6 实用技巧

* 安装时强制跳过确认：

  ```bash
  yum install -y <包名>
  apt install -y <包名>
  ```

* 查看哪些包最近安装：

  ```bash
  yum history
  cat /var/log/apt/history.log
  ```

* 搜索命令所属包：

  ```bash
  yum provides /usr/bin/python3
  apt-file search /usr/bin/python3
  ```

---

### 14.7 总结对照表

| 功能    | CentOS 7 (Yum)       | Ubuntu (Apt)               |
| ----- | -------------------- | -------------------------- |
| 安装    | `yum install`        | `apt install`              |
| 卸载    | `yum remove`         | `apt remove` / `apt purge` |
| 搜索    | `yum search`         | `apt search`               |
| 查看信息  | `yum info`           | `apt show`                 |
| 更新    | `yum update`         | `apt upgrade`              |
| 清理缓存  | `yum clean all`      | `apt clean`                |
| 删除无用包 | `yum autoremove`     | `apt autoremove`           |
| 查看已安装 | `yum list installed` | `dpkg -l`                  |

---

### 14.8 实战建议

* **CentOS 运维建议：**

  * 推荐启用阿里云镜像源，提高速度；
  * 不要随意升级系统版本；
  * 生产环境固定软件版本以保持稳定。

* **Ubuntu 开发建议：**

  * 多用 `apt search` 查找包；
  * 经常执行 `apt update && apt upgrade`；
  * 使用 `ppa` 获取新版本软件。

---




