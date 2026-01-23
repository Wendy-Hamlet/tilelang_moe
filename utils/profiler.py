"""
性能分析工具
用于测量kernel运行时间和吞吐量
"""
import torch
import time
from typing import Callable, Dict, List
import numpy as np


class Profiler:
    """性能分析器"""
    
    def __init__(self, warmup_iters: int = 10, test_iters: int = 100):
        self.warmup_iters = warmup_iters
        self.test_iters = test_iters
        
    def benchmark(self, func: Callable, *args, **kwargs) -> Dict:
        """
        测试函数性能
        
        Returns:
            包含平均时间、标准差等统计信息的字典
        """
        # Warmup
        for _ in range(self.warmup_iters):
            func(*args, **kwargs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(self.test_iters):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
        
        return {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'median_ms': np.median(times),
        }
