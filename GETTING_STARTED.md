# TileLang MoE Benchmark - 完整指南

## 📦 项目设置完成！

你现在拥有一个完整的MoE算子benchmark项目，包含：

```
tilelang_moe/
├── .gitmodules              ← Git submodule配置
├── scripts/
│   ├── setup_tilelang.bat   ← Windows一键安装脚本
│   └── setup_tilelang.sh    ← Linux/Mac一键安装脚本
├── third_party/tilelang/    ← TileLang完整源码（git submodule）
├── reference_impl/          ← 官方MoE参考实现
├── kernels/                 ← 你的TileLang实现
├── examples/                ← 语法示例
└── docs/                    ← 详细文档
```

## 🚀 快速开始（3步）

### Step 1: 下载并安装

**Windows用户：**
```powershell
# 在项目根目录运行
.\scripts\setup_tilelang.bat
```

**Linux/Mac用户：**
```bash
# 在项目根目录运行
bash scripts/setup_tilelang.sh
```

这个脚本会自动：
1. ✅ 下载TileLang完整源码到 `third_party/tilelang/`
2. ✅ 通过pip安装tilelang运行时
3. ✅ 安装所有其他依赖
4. ✅ 验证安装成功

### Step 2: 学习TileLang

```bash
# 1. 查看语法入门示例
python examples/tilelang_syntax_examples.py

# 2. 浏览官方示例（已下载到本地）
dir third_party/tilelang/examples/     # Windows
ls third_party/tilelang/examples/      # Linux/Mac

# 3. 运行参考实现
cd reference_impl
python test_example_fusedmoe.py
```

### Step 3: 开始开发

```bash
# 编辑你的kernel实现
code kernels/moe_gemm.py
code kernels/topk_gating.py

# 运行benchmark
python benchmarks/benchmark_moe.py
```

## 📚 学习路径

### 初学者路径 (第1-2天)

1. **理解基础语法**
   - 📄 阅读 `examples/tilelang_syntax_examples.py`
   - 📄 查看 `QUICKSTART.md`

2. **学习官方示例**
   - 📁 `third_party/tilelang/examples/gemm/` - GEMM优化
   - 📁 `third_party/tilelang/examples/flash_attention/` - Attention实现

3. **研究参考实现**
   - 📄 `reference_impl/example_fusedmoe_torch.py` - PyTorch版本
   - 📄 `reference_impl/example_fusedmoe_tilelang.py` - TileLang版本

### 进阶路径 (第3-5天)

4. **实现简单kernel**
   - 从简单的向量加法开始
   - 逐步实现矩阵乘法
   - 添加shared memory优化

5. **实现MoE算子**
   - TopK Gating
   - Expert GEMM
   - 完整MoE层

6. **性能优化**
   - 对比benchmark结果
   - 参考官方优化技术
   - 迭代优化

## 🔍 重要资源位置

### TileLang源码（已下载到本地）

| 内容 | 位置 |
|------|------|
| GEMM示例 | `third_party/tilelang/examples/gemm/` |
| Flash Attention | `third_party/tilelang/examples/flash_attention/` |
| DeepSeek MoE | `third_party/tilelang/examples/deepseek_v32/` |
| 文档 | `third_party/tilelang/docs/` |
| 核心代码 | `third_party/tilelang/tilelang/` |

### 项目文件

| 文件 | 用途 |
|------|------|
| `QUICKSTART.md` | 详细入门指南 |
| `docs/using_tilelang_source.md` | 如何使用TileLang源码 |
| `examples/tilelang_syntax_examples.py` | 语法快速入门 |
| `reference_impl/` | 完整的MoE参考实现 |

## 💡 开发技巧

### 查看TileLang示例

```bash
# 查看所有官方示例
ls third_party/tilelang/examples/

# 查看特定示例
cat third_party/tilelang/examples/gemm/example_gemm_tile_device.py

# 查看MoE相关实现
cat third_party/tilelang/examples/deepseek_v32/sparse_mla_fwd.py
```

### 运行官方测试

```bash
cd third_party/tilelang/examples/gemm
python example_gemm_tile_device.py
```

### 更新TileLang源码

```bash
cd third_party/tilelang
git pull origin main
cd ../..
```

## 🎯 开发建议

1. **先理解再实现** - 多看示例代码，理解优化思路
2. **从简单开始** - 先实现能跑的版本，再优化
3. **频繁测试** - 每次改动后运行测试确保正确性
4. **参考对比** - 遇到问题对照官方实现学习

## ❓ 常见问题

**Q: 为什么要下载完整源码？**  
A: 便于离线学习，查看实现细节，调试代码。运行时仍使用pip安装的稳定版本。

**Q: third_party/tilelang/ 占用多少空间？**  
A: 约200-300MB，包含完整源码、示例和文档。

**Q: 如何切换TileLang版本？**  
A: pip安装的版本和git源码独立。pip控制运行版本，git源码用于学习。

**Q: 可以修改third_party/tilelang/的代码吗？**  
A: 可以，但建议只用于学习和实验。生产代码使用pip安装的版本。

## 🎉 开始你的MoE优化之旅！

现在你已经准备好了：
- ✅ 完整的TileLang源码和示例
- ✅ 官方MoE参考实现
- ✅ 详细的文档和指南
- ✅ 自动化的配置脚本

开始编码吧！ 🚀
