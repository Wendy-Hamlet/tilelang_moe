#!/usr/bin/env python3
"""
验证TileLang环境和源码是否正确配置
"""
import os
import sys
from pathlib import Path

def check_tilelang_installation():
    """检查TileLang是否通过pip安装"""
    print("🔍 检查TileLang安装...")
    try:
        import tilelang
        print(f"  ✓ TileLang已安装: {tilelang.__version__}")
        return True
    except ImportError:
        print("  ✗ TileLang未安装，请运行: pip install tilelang")
        return False

def check_tilelang_source():
    """检查TileLang源码是否下载"""
    print("\n🔍 检查TileLang源码...")
    source_path = Path("third_party/tilelang")
    
    if not source_path.exists():
        print(f"  ✗ 源码目录不存在: {source_path}")
        print("  请运行: git submodule add https://github.com/tile-ai/tilelang.git third_party/tilelang")
        return False
    
    # 检查关键目录
    key_dirs = [
        source_path / "examples",
        source_path / "tilelang",
        source_path / "docs"
    ]
    
    all_exist = True
    for dir_path in key_dirs:
        if dir_path.exists():
            print(f"  ✓ {dir_path.relative_to(source_path)}/")
        else:
            print(f"  ✗ {dir_path.relative_to(source_path)}/ 不存在")
            all_exist = False
    
    if all_exist:
        # 统计示例数量
        examples_dir = source_path / "examples"
        if examples_dir.exists():
            example_count = len([d for d in examples_dir.iterdir() if d.is_dir()])
            print(f"  ✓ 找到 {example_count} 个示例目录")
    
    return all_exist

def check_dependencies():
    """检查其他依赖"""
    print("\n🔍 检查其他依赖...")
    deps = {
        "torch": "PyTorch",
        "numpy": "NumPy",
        "pandas": "Pandas",
        "matplotlib": "Matplotlib"
    }
    
    all_installed = True
    for module, name in deps.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} 未安装")
            all_installed = False
    
    return all_installed

def list_available_examples():
    """列出可用的示例"""
    print("\n📚 可用的TileLang示例:")
    source_path = Path("third_party/tilelang")
    examples_dir = source_path / "examples"
    
    if not examples_dir.exists():
        print("  示例目录不存在")
        return
    
    # 列出一些重要的示例
    important_examples = [
        "gemm",
        "flash_attention",
        "deepseek_v32",
        "fusedmoe",
        "convolution",
        "linear_attention"
    ]
    
    for example in important_examples:
        example_path = examples_dir / example
        if example_path.exists():
            print(f"  ✓ {example}/")
        else:
            print(f"  - {example}/ (未找到)")
    
    print(f"\n  完整列表: third_party/tilelang/examples/")

def main():
    print("=" * 60)
    print("TileLang MoE Benchmark - 环境验证")
    print("=" * 60)
    
    # 检查
    has_tilelang = check_tilelang_installation()
    has_source = check_tilelang_source()
    has_deps = check_dependencies()
    
    if has_source:
        list_available_examples()
    
    # 总结
    print("\n" + "=" * 60)
    if has_tilelang and has_source and has_deps:
        print("✅ 环境配置完成！可以开始开发了。")
        print("\n接下来:")
        print("  1. python examples/tilelang_syntax_examples.py")
        print("  2. cd reference_impl && python test_example_fusedmoe.py")
        print("  3. python benchmarks/benchmark_moe.py")
    else:
        print("⚠️  环境配置不完整")
        if not has_tilelang:
            print("  → 安装TileLang: pip install tilelang")
        if not has_source:
            print("  → 下载源码: git submodule add https://github.com/tile-ai/tilelang.git third_party/tilelang")
        if not has_deps:
            print("  → 安装依赖: pip install -r requirements.txt")
    print("=" * 60)

if __name__ == "__main__":
    main()
