# Git介绍

以下是git，以及github，gitee的详细介绍和使用教程

## 一、总概
### 1.1 Git 是什么？

Git 是由 Linux 之父 **Linus** 编写的 **分布式版本控制系统（DVCS, Distributed Version Control System）**，用于管理文件（尤其是代码）的版本变更。

> * 主要用于代码管理，支持多人协作、版本追踪、分支管理等。
> * 与传统集中式版本控制系统（如 SVN）不同，Git 是分布式的，每个开发者本地都完整保存代码仓库。

---

### 1.2 Git 核心架构与原理

#### 1.2.1 版本控制模型

* Git 不是记录文件变化的差异（diff），而是以**快照（Snapshot）**的形式保存文件状态。
* 每次提交时，Git 会保存所有文件的快照，未改动的文件只引用之前的版本，节省空间。

#### 1.2.2 Git 对象类型

* **Blob（Binary Large Object）**：文件内容。
* **Tree**：目录信息，指向文件 blob 和子目录 tree。
* **Commit**：提交对象，包含作者信息、提交时间、提交信息、指向 tree、指向父 commit。
* **Tag**：给 commit 打标签。

---

#### 1.2.3 Git 工作区四个区域

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

#### 1.4.1 配置 Git 用户信息（必须）

```bash
git config --global user.name "你的名字"
git config --global user.email "你的邮箱"
```

---

#### 1.4.2 创建和初始化仓库

* 创建新仓库

```bash
git init
```

* 克隆远程仓库到本地

```bash
git clone https://github.com/user/repo.git
```

---

#### 1.4.3 文件状态查看

```bash
git status              # 查看工作区和暂存区状态
git diff                # 查看未暂存的文件更新了哪些内容
git diff --staged       # 查看暂存区和最后一次提交的差异
```

---

#### 1.4.4 添加文件到暂存区

```bash
git add file.txt        # 添加单个文件
git add .               # 添加当前目录所有修改（新增、修改）
git add -p              # 交互式选择要添加的代码块（更细粒度）
```

---

#### 1.4.5 提交修改

```bash
git commit -m "简短说明"                    # 简单提交
git commit -am "修改文件并提交"             # 直接添加跟踪文件并提交（不包含新文件）
git commit --amend                        # 修改上次提交（慎用）
```

---

#### 1.4.6 查看提交历史

```bash
git log                                  # 查看详细提交记录
git log --oneline --graph --all         # 图形化简洁显示
git log -p                              # 查看每次提交具体改动
```

---

#### 1.4.7 分支操作

```bash
git branch                              # 查看所有分支，当前分支会有星号
git branch new_feature                  # 创建新分支
git checkout new_feature                # 切换分支
git checkout -b hotfix                  # 创建并切换分支
git branch -d feature_branch            # 删除本地分支
```

---

#### 1.4.8 合并分支

```bash
git checkout main                      # 切回主分支
git merge feature_branch               # 合并 feature_branch 分支
```

---

#### 1.4.9 解决冲突

1. 执行合并命令时出现冲突，Git 会提示冲突文件。
2. 打开冲突文件，查找 `<<<<<<<`、`=======`、`>>>>>>>` 标记。
3. 手动修改为期望代码，保存。
4. 使用 `git add conflict_file` 标记冲突已解决。
5. 使用 `git commit` 完成合并提交。

---

#### 1.4.10 远程仓库管理

```bash
git remote -v                          # 查看远程仓库地址
git remote add origin <url>            # 添加远程仓库
git push -u origin main                # 推送代码到远程并设置默认上游分支
git pull                             # 拉取远程仓库代码并合并到当前分支
```

---

#### 1.4.11 回退操作

```bash
git reset --soft HEAD^                 # 回退到上一次提交，保留改动到暂存区
git reset --mixed HEAD^                # 回退并取消暂存改动，改动保留在工作区
git reset --hard HEAD^                 # 回退并丢弃所有改动（危险）
git checkout -- file.txt               # 恢复文件到最后一次提交状态，丢弃改动
```

---

#### 1.4.12 标签（Tag）

```bash
git tag v1.0                          # 创建轻量标签
git tag -a v1.0 -m "版本1.0发布"      # 创建带注释标签
git push origin v1.0                  # 推送标签到远程
```

---

### 1.5 高级技巧

#### 1.5.1 交互式暂存（`git add -p`）

* 将大文件修改拆成若干块，选择性提交，提升代码质量。

#### 1.5.2 查看某文件历史改动

```bash
git log -p -- path/to/file
```

#### 1.5.3 修改历史提交（交互式 rebase）

```bash
git rebase -i HEAD~3
```

* 修改、合并、删除最近 3 次提交。

#### 1.5.4 忽略文件

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


