# 如何下载TileLang源码

## 问题：为什么 `git submodule update --init --recursive` 没有作用？

因为首次需要先**添加submodule**，而不是更新。

## ✅ 正确的操作步骤

### 方法1: 使用自动脚本（推荐）

**Windows:**
```powershell
.\scripts\setup_tilelang.bat
```

**Linux/Mac:**
```bash
bash scripts/setup_tilelang.sh
```

### 方法2: 手动添加（如果你已经克隆了项目）

```bash
# 第一次需要添加submodule
git submodule add https://github.com/tile-ai/tilelang.git third_party/tilelang

# 之后可以用这个命令更新
git submodule update --init --recursive
```

### 方法3: 克隆时一起下载

```bash
# 克隆项目时使用 --recursive 参数
git clone --recursive https://github.com/your-username/tilelang_moe.git
```

## 📋 验证是否成功

运行验证脚本：
```bash
python scripts/verify_setup.py
```

或者手动检查：
```bash
# 检查目录是否存在
ls third_party/tilelang/

# 查看示例
ls third_party/tilelang/examples/

# 查看版本
cd third_party/tilelang
git log --oneline -5
```

## 🎯 现在已经完成

如果你已经运行了 `git submodule add https://github.com/tile-ai/tilelang.git third_party/tilelang`，
那么TileLang源码已经成功下载到 `third_party/tilelang/` 目录！

可以查看：
```bash
# Windows
dir third_party\tilelang\examples

# Linux/Mac  
ls third_party/tilelang/examples/
```

## 📚 接下来

1. **安装运行环境：**
   ```bash
   pip install tilelang
   pip install -r requirements.txt
   ```

2. **验证环境：**
   ```bash
   python scripts/verify_setup.py
   ```

3. **开始学习：**
   ```bash
   # 查看语法示例
   python examples/tilelang_syntax_examples.py
   
   # 查看官方GEMM示例
   cat third_party/tilelang/examples/gemm/example_gemm_tile_device.py
   
   # 查看MoE示例
   cat third_party/tilelang/examples/deepseek_v32/sparse_mla_fwd.py
   ```

## 常见问题

**Q: submodule和pip安装有什么区别？**
- `third_party/tilelang/` - 完整源码，用于学习和查看实现
- `pip install tilelang` - 编译好的包，用于运行

**Q: 需要从源码编译吗？**
- 不需要！用pip安装即可运行
- 源码仅用于学习参考

**Q: 如何更新源码？**
```bash
cd third_party/tilelang
git pull origin main
cd ../..
```
