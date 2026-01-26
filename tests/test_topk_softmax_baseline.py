"""
快速测试：验证PyTorch baseline的正确性和数学等价性
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kernels.topk_softmax_baseline import verify_equivalence

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("PyTorch TopK+Softmax 数学等价性验证")
    print("=" * 80)
    
    # DeepSeek-V2配置
    print("\n测试配置1: DeepSeek-V2")
    verify_equivalence(batch_size=64, num_experts=160, top_k=6)
    
    # DeepSeek-V3配置  
    print("\n测试配置2: DeepSeek-V3")
    verify_equivalence(batch_size=64, num_experts=256, top_k=8)
    
    print("\n✅ 所有测试完成！")
