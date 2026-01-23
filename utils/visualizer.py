"""
结果可视化工具
生成性能对比图表
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path


def plot_performance_comparison(results: dict, save_path: str = None):
    """
    绘制性能对比图
    
    Args:
        results: {impl_name: {config: perf_data}}
        save_path: 保存路径
    """
    plt.figure(figsize=(12, 6))
    
    # TODO: 实现具体的可视化逻辑
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_performance_table(results: dict, save_path: str = None):
    """
    生成性能对比表格
    
    Args:
        results: benchmark结果
        save_path: 保存路径
    """
    df = pd.DataFrame(results)
    
    if save_path:
        df.to_csv(save_path, index=False)
    
    return df
