# MindStudio Assistant

MindStudio Assistant 是一个命令行辅助工具，旨在简化和加速 MindStudio 中自定义算子的开发流程。它提供了一系列命令，可以帮助开发者自动生成算子工程骨架等工作，帮助开发者专注于核心的算子逻辑实现。

## 功能说明

- **项目脚手架生成**：通过 `assitant opgen` 命令，可以基于预设的模板快速创建一个新的算子工程。
- **项目编译**：     （待实现）通过 `assitant build` 命令，可以执行编译命令。
- **跨版本兼容**：   支持 Python 3.x 及以上版本。


## 快速上手 (Quick Start)

### 1. 环境要求

- Python 3.x

### 2. 安装

克隆本项目到您的本地，可以通过直接调用脚本使用。


### 3. 生成您的第一个算子工程

安装完成后，您就可以使用 `opgen` 命令来创建新的算子了。例如，创建一个名为 `my_op` 的 `math` 类算子：

```bash
python ./assistant.py opgen -t math -n my_op
```

执行完毕后，您会在 `./math/my_op/` 目录下看到一个完整的、可以直接开发的算子工程模板。


## 命令行用法

假设在ops-math根目录下执行。

### 主命令

通过 `--help` 查看所有可用的命令和选项：

```bash
> python .\scripts\mindstudio\assist.py --help

Usage: assistant.py [OPTIONS] COMMAND [ARGS]...

  MindStudio assistant main entry.

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  opgen  生产项目骨架
  build  构建项目
```

### `opgen` 命令

用于生成算子工程骨架。

**用法:**
```bash
python .\assistant.py opgen [OPTIONS]
```

**选项:**
- `-t`, `--op_type` (必需): 算子的分类，例如 `aclnn`。
- `-n`, `--op_name` (必需): 新算子的名称，推荐使用蛇形命名法 (snake_case)，例如 `my_op`。
- `-p`, `--output_path` (可选): 工程的输出根路径，默认为当前目录 (`.`)。

**示例:**
```bash
# 在当前目录下创建 math/my_op 工程
python .\assistant.py opgen -t math -n my_op

# 在指定路径 D:\projects 下创建 custom/new_op 工程
python .\assistant.py opgen -t custom -n new_op -p D:\projects
```

### `build` 命令

用于构建项目（当前此功能为占位符，待实现）。
