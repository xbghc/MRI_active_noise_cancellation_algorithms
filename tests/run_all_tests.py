"""
主测试运行器
运行所有测试模块
"""

import os
import sys

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("运行所有测试")
    print("=" * 60)

    # 测试模块列表
    test_modules = [
        ("test_hyc_data_loader", "HycDataLoader测试"),
        ("test_comparison", "Comparison测试"),
        ("test_mat", "MAT模块测试"),
    ]

    for module_name, description in test_modules:
        print(f"\n{'=' * 40}")
        print(f"运行 {description}")
        print(f"{'=' * 40}")

        try:
            # 动态导入测试模块
            module = __import__(module_name)

            # 运行测试函数
            if hasattr(module, "test_hyc_data_loader") and hasattr(
                module, "test_data_consistency"
            ):
                module.test_hyc_data_loader()
                module.test_data_consistency()
            elif hasattr(module, "test_comparison"):
                module.test_comparison()
            elif hasattr(module, "test_mat_processing"):
                module.test_mat_processing()
            else:
                print(f"模块 {module_name} 没有找到测试函数")

        except Exception as e:
            print(f"运行 {description} 时出错: {e}")

    print(f"\n{'=' * 60}")
    print("所有测试完成")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run_all_tests()
