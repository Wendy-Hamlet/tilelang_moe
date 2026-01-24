# 使用TileLang源码

## 为什么下载TileLang源码？

通过git submodule下载完整的TileLang源码有以下好处：

1. **学习实现细节** - 查看编译器、优化器的实现
2. **丰富的示例** - 访问所有官方示例代码
3. **调试便利** - 可以直接查看和修改源码
4. **离线使用** - 不依赖网络查看文档

## 快速使用

### Windows用户

```powershell
# 运行一键安装脚本
.\scripts\setup_tilelang.bat
```

### Linux/Mac用户

```bash
# 运行一键安装脚本
bash scripts/setup_tilelang.sh
```

### 手动安装

```bash
# 1. 下载tilelang源码
git submodule update --init --recursive

# 2. 安装tilelang
pip install tilelang

# 3. 安装其他依赖
pip install -r requirements.txt
```

## 探索TileLang源码

下载完成后，你可以浏览：

```bash
# 查看示例代码
ls third_party/tilelang/examples/

# MoE相关示例
cat third_party/tilelang/examples/deepseek_v32/sparse_mla_fwd.py

# GEMM优化示例
cat third_party/tilelang/examples/gemm/example_gemm_tile_device.py

# 其他高级示例
ls third_party/tilelang/examples/flash_attention/
ls third_party/tilelang/examples/linear_attention/
```

## 学习路径

1. **基础语法** - `examples/tilelang_syntax_examples.py`
2. **官方示例** - `third_party/tilelang/examples/`
3. **参考实现** - `reference_impl/example_fusedmoe_tilelang.py`
4. **源码实现** - `third_party/tilelang/tilelang/`

## 常用示例位置

| 示例类型 | 位置 |
|---------|------|
| GEMM优化 | `third_party/tilelang/examples/gemm/` |
| Flash Attention | `third_party/tilelang/examples/flash_attention/` |
| MoE实现 | `third_party/tilelang/examples/deepseek_v32/` |
| 卷积 | `third_party/tilelang/examples/convolution/` |
| 线性注意力 | `third_party/tilelang/examples/linear_attention/` |

## 更新源码

```bash
# 更新到最新版本
cd third_party/tilelang
git pull origin main
cd ../..
```

## 注意事项

- ⚠️ `third_party/tilelang/` 仅用于学习，运行时使用pip安装的版本
- 💡 如果要从源码编译tilelang，参考 `third_party/tilelang/docs/get_started/Installation.md`
- 📝 源码大小约 200MB，首次下载需要一些时间
