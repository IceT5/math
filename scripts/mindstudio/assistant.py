import argparse
import sys
import os

# 将当前脚本所在目录添加到模块搜索路径，以便导入同级目录下的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入命令模块
import opgen_standalone
import build_standalone

def main():
    """
    主函数，用于解析命令行参数并分发到对应的子命令。
    """
    parser = argparse.ArgumentParser(
        description="项目开发辅助工具",
        epilog="使用 'python assist.py <命令> --help' 获取特定命令的帮助信息。"
    )
    # 创建子命令解析器
    subparsers = parser.add_subparsers(dest='command', required=True, help='可用的命令')

    # 注册 opgen 命令
    opgen_standalone.register_parser(subparsers)

    # 注册 build 命令
    build_standalone.register_parser(subparsers)

    # 如果没有提供任何参数，则打印帮助信息
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    # 根据 'func' 属性执行选定的命令函数
    if hasattr(args, 'func'):
        args.func(args)
    else:
        # 此分支在 subparser 设置为 required=True 时通常不会被触发
        print(f"命令 '{args.command}' 的功能尚未实现。")

if __name__ == "__main__":
    main()