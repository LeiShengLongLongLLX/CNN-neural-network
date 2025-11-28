# Git介绍

以下是git，以及github，gitee的详细介绍和使用教程

## 一、总概
### 1.1 Git 是什么？

Git 是由 Linux 之父 **Linus** 编写的 **分布式版本控制系统（DVCS, Distributed Version Control System）**，用于管理文件（尤其是代码）的版本变更。

> * 主要用于代码管理，支持多人协作、版本追踪、分支管理等。
> * 与传统集中式版本控制系统（如 SVN）不同，Git 是分布式的，每个开发者本地都完整保存代码仓库。

---

### 1.2 Git 核心架构与原理

#### 1. 版本控制模型

* Git 不是记录文件变化的差异（diff），而是以**快照（Snapshot）**的形式保存文件状态。
* 每次提交时，Git 会保存所有文件的快照，未改动的文件只引用之前的版本，节省空间。

#### 2. Git 对象类型

* **Blob（Binary Large Object）**：文件内容。
* **Tree**：目录信息，指向文件 blob 和子目录 tree。
* **Commit**：提交对象，包含作者信息、提交时间、提交信息、指向 tree、指向父 commit。
* **Tag**：给 commit 打标签。

---

#### 3. Git 工作区四个区域

| 区域             | 作用                        |
| -------------- | ------------------------- |
| **工作区**        | 你编辑文件的地方，文件处于未跟踪或已修改状态    |
| **暂存区（Index）** | 临时存储将要提交的文件快照             |
| **本地仓库（.git）** | 保存所有提交的历史记录和对象            |
| **远程仓库**       | 远端的代码仓库，如 GitHub、GitLab 等 |

---

### 1.3 基本工作流程

```bash
# 1. 修改文件（工作区）
# 2. 将修改添加到暂存区
git add file1 file2

# 3. 提交暂存区内容到本地仓库
git commit -m "描述信息"

# 4. 将本地仓库提交推送到远程仓库
git push origin main
```

```sql
工作区 → 暂存区 → 本地仓库 → 推送 → 远程仓库
git add    git commit        git push
```

---

### 1.4 Git 常用命令详解

#### 1. 配置 Git 用户信息（必须）

```bash
git config --global user.name "你的名字"
git config --global user.email "你的邮箱"
```

---

#### 2. 创建和初始化仓库

* 创建新仓库

```bash
git init
```

* 克隆远程仓库到本地

```bash
git clone https://github.com/user/repo.git
```

---

#### 3. 文件状态查看

```bash
git status              # 查看工作区和暂存区状态
git diff                # 查看未暂存的文件更新了哪些内容
git diff --staged       # 查看暂存区和最后一次提交的差异
```

---

#### 4. 添加文件到暂存区

```bash
git add file.txt        # 添加单个文件
git add .               # 添加当前目录所有修改（新增、修改）
git add -p              # 交互式选择要添加的代码块（更细粒度）
```

---

#### 5. 提交修改

```bash
git commit -m "简短说明"                    # 简单提交
git commit -am "修改文件并提交"             # 直接添加跟踪文件并提交（不包含新文件）
git commit --amend                        # 修改上次提交（慎用）
```

---

#### 6. 查看提交历史

```bash
git log                                  # 查看详细提交记录
git log --oneline --graph --all         # 图形化简洁显示
git log -p                              # 查看每次提交具体改动
```

---

#### 7. 分支操作

```bash
git branch                              # 查看所有分支，当前分支会有星号
git branch new_feature                  # 创建新分支
git checkout new_feature                # 切换分支
git checkout -b hotfix                  # 创建并切换分支
git branch -d feature_branch            # 删除本地分支
```

---

#### 8. 合并分支

```bash
git checkout main                      # 切回主分支
git merge feature_branch               # 合并 feature_branch 分支
```

---

#### 9. 解决冲突

1. 执行合并命令时出现冲突，Git 会提示冲突文件。
2. 打开冲突文件，查找 `<<<<<<<`、`=======`、`>>>>>>>` 标记。
3. 手动修改为期望代码，保存。
4. 使用 `git add conflict_file` 标记冲突已解决。
5. 使用 `git commit` 完成合并提交。

---

#### 10. 远程仓库管理

```bash
git remote -v                          # 查看远程仓库地址
git remote add origin <url>            # 添加远程仓库
git push -u origin main                # 推送代码到远程并设置默认上游分支
git pull                             # 拉取远程仓库代码并合并到当前分支
```

---

#### 11. 回退操作

```bash
git reset --soft HEAD^                 # 回退到上一次提交，保留改动到暂存区
git reset --mixed HEAD^                # 回退并取消暂存改动，改动保留在工作区
git reset --hard HEAD^                 # 回退并丢弃所有改动（危险）
git checkout -- file.txt               # 恢复文件到最后一次提交状态，丢弃改动
```

---

#### 12. 标签（Tag）

```bash
git tag v1.0                          # 创建轻量标签
git tag -a v1.0 -m "版本1.0发布"      # 创建带注释标签
git push origin v1.0                  # 推送标签到远程
```

---

### 1.5 高级技巧

#### 1. 交互式暂存（`git add -p`）

* 将大文件修改拆成若干块，选择性提交，提升代码质量。

#### 2. 查看某文件历史改动

```bash
git log -p -- path/to/file
```

#### 3. 修改历史提交（交互式 rebase）

```bash
git rebase -i HEAD~3
```

* 修改、合并、删除最近 3 次提交。

#### 4. 忽略文件

* 创建 `.gitignore` 文件，列出不需要 Git 管理的文件/目录。

---

### 1.6 多人协作常用流程（Git Flow 简化版）

```
main/master（主分支）
    ↑
    ├── feature/xxx（新功能分支）
    │       ↑
    │       ├─ 修改代码 → 提交 → push
    │
    └── hotfix/xxx（热修复分支）
```

**步骤：**

1. 从 `main` 创建分支（`feature/login`）
2. 本地开发，提交并推送分支
3. 远程发起 Pull Request（PR）
4. 代码评审，修复问题，合并回 `main`
5. 关闭分支

---

### 1.7 常见问题及解决方案

| 问题              | 解决方案                          |
| --------------- | ----------------------------- |
| 文件误删或改错想恢复      | `git checkout -- file`        |
| 撤销暂存区文件         | `git restore --staged file`   |
| 不小心提交错误，想修改提交信息 | `git commit --amend`          |
| 回退到某个历史版本       | `git reset --hard commit_id`  |
| 代码冲突            | 手动修改冲突，`git add`，`git commit` |
| 远程仓库 push 被拒绝   | `git pull` 先合并远程改动后再 push     |

---

### 1.8 总结附录：常用命令速查表

| 功能     | 命令                          | 说明          |
| ------ | --------------------------- | ----------- |
| 查看状态   | `git status`                | 查看文件状态      |
| 添加文件   | `git add file`              | 添加文件到暂存区    |
| 提交     | `git commit -m "msg"`       | 提交暂存区内容     |
| 查看历史   | `git log`                   | 查看提交历史      |
| 创建分支   | `git branch new_branch`     | 创建新分支       |
| 切换分支   | `git checkout branch`       | 切换分支        |
| 合并分支   | `git merge branch`          | 合并指定分支到当前分支 |
| 查看远程   | `git remote -v`             | 查看远程仓库      |
| 添加远程   | `git remote add origin url` | 添加远程仓库      |
| 推送代码   | `git push origin branch`    | 推送分支到远程     |
| 拉取代码   | `git pull`                  | 拉取并合并远程分支   |
| 查看文件差异 | `git diff`                  | 查看未暂存改动     |
| 回退     | `git reset --hard commit`   | 回退到指定提交     |
| 恢复文件   | `git checkout -- file`      | 恢复文件到最近一次提交 |
| 删除分支   | `git branch -d branch`      | 删除本地分支      |

---

## 二、本地仓库`.git`

### 2.1 什么是 `.git` 目录？

* `.git` 是 **Git 仓库的核心文件夹**，位于项目根目录下（隐藏文件夹）。
* 它保存了所有版本控制相关数据，是 Git 追踪和管理代码历史的“数据库”和“配置中心”。
* 当你执行 `git init` 初始化仓库时，Git 会自动创建这个文件夹。

---

### 2.2 `.git` 目录的核心结构和作用

| 目录/文件名        | 类型 | 作用说明                                      |
| ------------- | -- | ----------------------------------------- |
| `objects/`    | 目录 | 存储 Git 所有对象（commit、tree、blob、tag）文件       |
| `refs/`       | 目录 | 存放指针，包含分支（heads）、标签（tags）指向的 commit 哈希    |
| `HEAD`        | 文件 | 当前所在分支指针，指向 refs/heads/xxx 分支或直接指向 commit |
| `config`      | 文件 | 仓库本地配置文件（类似 `.gitconfig`）                 |
| `index`       | 文件 | 暂存区索引，记录工作区和暂存区文件的映射                      |
| `logs/`       | 目录 | 记录分支和 HEAD 的变动日志                          |
| `hooks/`      | 目录 | Git 钩子脚本，可以定义自动化操作                        |
| `description` | 文件 | 用于 GitWeb 服务的仓库描述（一般不用修改）                 |
| `info/`       | 目录 | 存放一些全局忽略文件 `.git/info/exclude`            |

---

### 2.3 详细解读重要目录和文件

#### 2.3.1 `objects/`

* 存储所有 Git 对象，核心是**以 SHA-1 哈希命名的文件**。

* 包括三种主要对象：

  * **blob**：文件内容快照
  * **tree**：目录结构，包含指向 blob 和子 tree 的引用
  * **commit**：一次提交信息，指向对应的 tree 和父提交

* 存储方式：以对象哈希的前两位为目录名，后38位为文件名，例如哈希 `ab1234...` 存在 `.git/objects/ab/1234...`。

---

#### 2.3.2 `refs/`

* 存储分支、标签的引用。
* 结构：

  * `refs/heads/` 存放本地分支的最新 commit 哈希
  * `refs/tags/` 存放标签的 commit 哈希
  * `refs/remotes/` 存放远程分支信息

---

#### 2.3.3 `HEAD`

* 记录当前分支或当前检出(commit)的位置。
* 常见内容为：`ref: refs/heads/main` 表示当前在 `main` 分支。
* 也可能直接指向某个 commit 哈希（游离 HEAD 状态）。

---

#### 2.3.4 `index`

* 又称为**暂存区索引文件**。
* 记录了工作区文件和暂存区的对应关系，是 Git 实现高效差异比较和提交的关键。
* 它保存了文件名、文件哈希、文件权限、修改时间等元信息。

---

#### 2.3.5 `logs/`

* 记录 HEAD 和分支指针的历史变动，方便追踪分支操作。
* 例如：`logs/HEAD` 记录了 HEAD 位置的变更历史。
* 用于 `git reflog` 命令实现，可以恢复误删提交。

---

#### 2.3.6 `hooks/`

* 存放 Git 钩子脚本文件（shell 脚本等），可用于自动化工作流。
* 常见钩子有：`pre-commit`、`post-commit`、`pre-push` 等。
* 默认文件以 `.sample` 结尾，需手动重命名启用。

---

#### 2.3.7 `config`

* Git 仓库的本地配置文件（文本格式）。
* 存储仓库相关设置，如远程仓库地址、忽略规则、用户信息覆盖等。
* 优先级低于全局配置 `~/.gitconfig`。

---

### 2.4 Git 对象存储流程概述

* 当你执行 `git add file` 时：

  1. Git 计算文件内容的 SHA-1 哈希，生成 blob 对象，保存到 `objects/` 目录。
  2. Git 创建 tree 对象表示目录结构，也存入 `objects/`。
  3. 暂存区（index）记录这些对象信息。

* 当你执行 `git commit` 时：

  1. Git 创建 commit 对象，指向相应的 tree 对象和父 commit。
  2. 该 commit 对象也存储在 `objects/`。
  3. 更新 `refs/heads/<branch>`，让分支指向新 commit。
  4. 更新 `HEAD` 指向当前分支。

---

### 2.5 总结

| 项目              | 说明                             |
| --------------- | ------------------------------ |
| `.git/objects/` | Git 所有对象（blob/tree/commit）存储目录 |
| `.git/refs/`    | 分支、标签、远程分支指针                   |
| `.git/HEAD`     | 当前检出指针，指示当前分支或 commit          |
| `.git/index`    | 暂存区索引，存储已添加到暂存区的文件信息           |
| `.git/logs/`    | HEAD 和分支的操作日志                  |
| `.git/hooks/`   | Git 钩子脚本目录，可定制自动化流程            |
| `.git/config`   | 仓库级配置文件，记录远程仓库、分支策略等           |

---

### 2.6 常用命令辅助理解

| 命令                       | 作用              |
| ------------------------ | --------------- |
| `git cat-file -p <hash>` | 查看指定对象内容        |
| `git ls-files --stage`   | 查看暂存区文件信息       |
| `git reflog`             | 查看 HEAD 和分支操作日志 |
| `git config --list`      | 查看当前配置          |

---

## 三、分支Branch与团队开发

### 3.1 Git 分支（Branch）基础概念

* **分支本质**：Git 中的分支是指向某个特定提交（commit）的指针，是对代码历史中某一点的命名。
* **轻量级**：创建分支只是在 `.git/refs/heads/` 目录下新建一个文本文件，内容是 commit 的哈希值，操作几乎瞬间完成，节省存储。
* **斜杠命名规则**：分支名中允许包含 `/`，如 `feature/login`，但这只是名字的一部分，Git 并不会把它当成目录结构，所有分支都是平级的。IDE 或工具界面会把带斜杠的分支名以“目录”形式展示，仅是视觉层级方便管理。

---

### 3.2 分支命名规范与团队协作

* **常见命名格式**：

  * `feature/xxx`：开发新功能
  * `bugfix/xxx`：修复问题
  * `hotfix/xxx`：紧急修复
  * `release/xxx`：发布版本
  * `test/xxx`：测试分支
* **命名目的**：明确分支作用，方便管理和协作，避免混乱。
* **允许字符**：大小写字母、数字、下划线 `_`、中划线 `-`、斜杠 `/`、点 `.` 等。空格和特殊字符不允许。
* **团队约定**：多数团队有自己的命名规范和风格，建议遵守。

---

### 3.3 分支的历史和结构

* **commit 与父 commit**

  * 每个 commit 记录自己父提交 **（parent commit）** 的哈希，构成一条历史链。
  * 普通提交只有一个父 commit；合并提交（merge commit）可能有两个或多个父 commit。
  * 这使得 Git 的历史实际上是一个有向无环图（DAG），而非简单的链条。

* **分支的历史是指向某个 commit 的链条**

  * 新建分支时，分支指针指向当前 commit，继承历史。
  * 后续在该分支提交会产生新的 commit，分支历史开始分叉。
  * 不同分支的历史是共享的，直到分支开始分叉。

---

### 3.4 分支操作及工作流

* **创建分支**

  * 命令：`git checkout -b feature/login`
  * 创建新指针指向当前 commit，几乎零成本。

* **切换分支**

  * 命令：`git checkout 分支名`
  * 移动 HEAD 指向该分支。

* **合并分支**

  * 命令：`git merge feature/login`
  * 合并分支内容，可能产生合并提交，有两个父 commit。

* **回退操作**

  * 未提交改动，使用 `git checkout -- 文件名` 或 `git restore 文件名` 丢弃工作区修改。
  * 已暂存未提交，使用 `git reset 文件名` 取消暂存。
  * 已提交未推送，使用 `git reset --soft/mixed/hard HEAD^` 根据需求回退提交。
  * 已推送，推荐提交新的修复，避免强制推送带来的风险。

* **`git reflog`**

  * 记录 HEAD 的历史变动，能在误操作后恢复丢失的提交。

---

### 3.5 远程仓库与本地分支同步

* **`git fetch`**

  * 从远程仓库拉取最新提交到本地仓库，但不修改当前工作区代码。
  * 适合查看远程状态，之后决定如何合并。

* **`git pull`**

  * 等同于 `git fetch` + `git merge`，自动合并远程更新到当前分支。
  * 可能产生合并冲突或产生复杂历史。
  * 默认合并方式，有时不如手动 `fetch` + `rebase` 清晰。

* **推荐实践**

  * 复杂团队环境，优先使用 `git fetch`，观察远程更新后手动合并。
  * 使用 `git pull --rebase` 保持历史线性，避免合并提交杂乱。
  * 避免直接强制覆盖远程，保持团队协作稳定。

---

### 3.6 总结

| 主题     | 核心观点                                         |
| ------ | -------------------------------------------- |
| 分支本质   | 指向特定 commit 的指针，分支名带 `/` 只是命名，不是层级。          |
| 命名规范   | 使用 `feature/xxx`、`bugfix/xxx` 等前缀清晰标注目的。     |
| 分支历史   | 由 commit 与父 commit 构成 DAG，分支之间共享或分叉历史。       |
| 操作与回退  | 创建、切换、合并分支轻量高效，撤销改动有多种安全策略。                  |
| 远程同步   | `fetch` 安全拉取，`pull` 自动合并，建议使用带 **rebase** 的拉取方式。 |
| 团队协作建议 | 明确命名规则，合理使用分支，结合远程操作规范保证协作顺畅。                |

---

### 3.7 git分支命令常见总结表
| 操作内容          | 命令示例                                              | 说明                |
| ------------- | ------------------------------------------------- | ----------------- |
| 查看本地分支        | `git branch`                                      | 列出所有本地分支          |
| 查看远程分支        | `git branch -r`                                   | 列出远程仓库的分支         |
| 查看所有分支（本地+远程） | `git branch -a`                                   | 列出本地和远程所有分支       |
| 查看分支跟踪关系      | `git branch -vv`                                  | 显示分支及关联的远程分支      |
| 创建新分支         | `git branch branch-name`                          | 新建本地分支            |
| 创建并切换到新分支     | `git checkout -b branch-name`                     | 新建并切换到该分支         |
| 切换分支          | `git checkout branch-name`                        | 切换到已有分支           |
| 设置分支关联远程分支    | `git branch --set-upstream-to=origin/branch-name` | 建立本地分支与远程分支的跟踪关系  |
| 第一次推送并建立关联    | `git push --set-upstream origin branch-name`      | 推送本地分支到远程并建立关联    |
| 解除分支关联        | `git branch --unset-upstream`                     | 取消当前分支的远程跟踪       |
| 后续推送          | `git push`                                        | 推送当前分支，前提是已关联远程分支 |
| 删除本地分支        | `git branch -d branch-name`                       | 删除本地分支（安全）        |
| 强制删除本地分支      | `git branch -D branch-name`                       | 强制删除本地分支          |
| 删除远程分支        | `git push origin --delete branch-name`            | 删除远程仓库分支          |
