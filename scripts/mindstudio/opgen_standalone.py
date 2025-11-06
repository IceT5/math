import argparse
import os
import shutil
import sys
import re


class OpGenerator:
    """算子工程生成器"""

    def __init__(self, op_type, op_name, output_path):
        self.op_type = op_type
        self.op_name = op_name
        self.output_path = output_path
        self.template_name = "add_example"

        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.template_dir = os.path.abspath(os.path.join(self.script_dir, 'template', 'add'))
        self.dest_dir = os.path.abspath(os.path.join(self.output_path, self.op_type, self.op_name))

    def run(self):
        """执行生成流程"""
        try:
            self._validate_inputs()
            self._copy_template()
            self._rename_files()
            self._replace_content()
            print(f"成功为 {self.op_type}/{self.op_name} 创建算子工程！")
            print(f"工程路径: {self.dest_dir}")
        except (ValueError, FileExistsError, FileNotFoundError, OSError) as e:
            print(f"错误: {e}", file=sys.stderr)
            sys.exit(1)

    def _validate_inputs(self):
        """校验输入参数的有效性和安全性"""
        if not self.op_type or not self.op_name:
            raise ValueError("算子类型和算子名称均不能为空。")

        if not re.match(r"^[a-zA-Z0-9_]+$", self.op_type):
            raise ValueError(f"算子类型 '{self.op_type}' 包含无效字符。只允许字母、数字和下划线。")

        if not re.match(r"^[a-zA-Z0-9_]+$", self.op_name):
            raise ValueError(f"算子名称 '{self.op_name}' 包含无效字符。只允许字母、数字和下划线。")
        
        if os.path.exists(self.dest_dir):
            raise FileExistsError(f"目标目录 '{self.dest_dir}' 已存在。")

    def _copy_template(self):
        """复制模板文件到目标目录"""
        print(f"使用模板在 '{self.dest_dir}' 创建算子工程...")
        if not os.path.exists(self.template_dir):
            raise FileNotFoundError(f"找不到模板目录 '{self.template_dir}'。请确保 'template/add' 目录存在。")
        
        try:
            shutil.copytree(self.template_dir, self.dest_dir)
        except OSError as e:
            raise OSError(f"复制模板文件失败: {e}")

    def _rename_files(self):
        """重命名文件和目录中的占位符"""
        for root, dirs, files in os.walk(self.dest_dir, topdown=False):
            for name in files + dirs:
                if self.template_name in name:
                    old_path = os.path.join(root, name)
                    new_name = name.replace(self.template_name, self.op_name)
                    new_path = os.path.join(root, new_name)
                    try:
                        os.rename(old_path, new_path)
                    except OSError as e:
                        raise OSError(f"重命名 '{old_path}' 到 '{new_path}' 失败: {e}")

    def _replace_content(self):
        """替换文件内容中的占位符"""
        replacements = {
            self.template_name: self.op_name,
            _to_pascal_case(self.template_name): _to_pascal_case(self.op_name),
            _to_upper_case(self.template_name): _to_upper_case(self.op_name),
            f"{_to_upper_case(self.template_name)}_H": f"{_to_upper_case(self.op_name)}_H"
        }

        for root, _, files in os.walk(self.dest_dir):
            for filename in files:
                file_path = os.path.join(root, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    original_content = content
                    for placeholder, replacement in replacements.items():
                        content = content.replace(placeholder, replacement)

                    if content != original_content:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                except UnicodeDecodeError:
                    print(f"跳过二进制或非UTF-8编码文件: {file_path}")
                except Exception as e:
                    print(f"处理文件 '{file_path}' 时出错: {e}", file=sys.stderr)


def _to_pascal_case(s: str) -> str:
    """将 snake_case 或 mixedCase 转换为 PascalCase。"""
    parts = s.replace('-', '_').split('_')
    return "".join(p[0].upper() + p[1:] if p else '' for p in parts)


def _to_upper_case(s: str) -> str:
    """将 snake_case 转换为 UPPER_CASE。"""
    return s.upper()


def execute(args):
    """根据模板目录生成算子工程骨架。"""
    generator = OpGenerator(args.op_type, args.op_name, args.output_path)
    generator.run()


def register_parser(subparsers):
    """为 opgen 命令注册解析器。"""
    parser_opgen = subparsers.add_parser('opgen', help='生产项目骨架')
    parser_opgen.add_argument('--op_type', '-t', required=True, help='算子分类，例如 math')
    parser_opgen.add_argument('--op_name', '-n', required=True, help='新算子的名称，例如 asinh')
    parser_opgen.add_argument('--output_path', '-p', default='.', help='生成工程的根路径')
    parser_opgen.set_defaults(func=execute)


def main():
    """
    主函数，用于独立执行 opgen。
    """
    parser = argparse.ArgumentParser(description="生产项目骨架")
    parser.add_argument('--op_type', '-t', required=True, help='算子分类，例如 math')
    parser.add_argument('--op_name', '-n', required=True, help='新算子的名称，例如 asinh')
    parser.add_argument('--output_path', '-p', default='.', help='生成工程的根路径')
    
    args = parser.parse_args()
    
    print("准备创建算子工程...")
    execute(args)


if __name__ == "__main__":
    main()