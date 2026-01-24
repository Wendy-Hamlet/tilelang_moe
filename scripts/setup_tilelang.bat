@echo off
REM TileLang源码下载和安装脚本 (Windows)

echo ================================================
echo TileLang MoE Benchmark - 环境配置
echo ================================================

REM 1. 下载tilelang源码
echo.
echo 📥 Step 1: 下载TileLang源码...
if exist "third_party\tilelang\.git" (
    echo ✓ TileLang源码已存在，更新中...
    cd third_party\tilelang
    git pull
    cd ..\..
) else (
    echo ✓ 正在下载TileLang源码...
    REM 如果submodule还未添加，先添加
    if not exist ".git\modules\third_party\tilelang" (
        git submodule add https://github.com/tile-ai/tilelang.git third_party/tilelang
    ) else (
        git submodule update --init --recursive
    )
)

REM 2. 安装tilelang (通过pip)
echo.
echo 📦 Step 2: 安装TileLang...
pip install tilelang

REM 3. 安装其他依赖
echo.
echo 📦 Step 3: 安装其他依赖...
pip install -r requirements.txt

REM 4. 验证安装
echo.
echo ✅ Step 4: 验证安装...
python -c "import tilelang; print(f'TileLang version: {tilelang.__version__}')"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

echo.
echo ================================================
echo ✓ 环境配置完成！
echo ================================================
echo.
echo 接下来：
echo   1. 查看示例: python examples\tilelang_syntax_examples.py
echo   2. 学习源码: dir third_party\tilelang\examples\
echo   3. 运行测试: cd reference_impl ^&^& python test_example_fusedmoe.py
echo.

pause
