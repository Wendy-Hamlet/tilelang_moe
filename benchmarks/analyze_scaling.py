"""
Analyze TopK scaling behavior and predict crossover point
"""
import pandas as pd
import numpy as np
from pathlib import Path

def analyze_trend(csv_path):
    """Analyze scaling trend from benchmark results"""
    df = pd.read_csv(csv_path)
    
    print("=" * 80)
    print("TopK Scaling Analysis")
    print("=" * 80)
    
    # Calculate growth rates
    batch_sizes = df['batch'].values
    pytorch_times = df['pytorch_ms'].values
    tilelang_times = df['tilelang_ms'].values
    
    print("\n📊 Scaling Behavior:")
    print("-" * 80)
    print(f"{'Batch':<10} {'PyTorch(ms)':<15} {'TileLang(ms)':<15} {'Gap(ms)':<12} {'Speedup':<10}")
    print("-" * 80)
    
    for i, batch in enumerate(batch_sizes):
        gap = tilelang_times[i] - pytorch_times[i]
        speedup = pytorch_times[i] / tilelang_times[i]
        print(f"{batch:<10} {pytorch_times[i]:<15.6f} {tilelang_times[i]:<15.6f} {gap:<12.6f} {speedup:<10.3f}x")
    
    # Calculate growth factors
    pt_growth = pytorch_times[-1] / pytorch_times[0]
    tl_growth = tilelang_times[-1] / tilelang_times[0]
    
    print("\n📈 Growth Analysis (first → last):")
    print(f"  PyTorch:  {pytorch_times[0]:.6f}ms → {pytorch_times[-1]:.6f}ms (growth: {pt_growth:.2f}x)")
    print(f"  TileLang: {tilelang_times[0]:.6f}ms → {tilelang_times[-1]:.6f}ms (growth: {tl_growth:.2f}x)")
    
    # Find crossover point
    crossover_idx = None
    for i in range(len(batch_sizes)):
        if tilelang_times[i] < pytorch_times[i]:
            crossover_idx = i
            break
    
    if crossover_idx is not None:
        print(f"\n🎯 CROSSOVER DETECTED at batch={batch_sizes[crossover_idx]}!")
        print(f"   TileLang: {tilelang_times[crossover_idx]:.6f}ms")
        print(f"   PyTorch:  {pytorch_times[crossover_idx]:.6f}ms")
        print(f"   Speedup:  {pytorch_times[crossover_idx]/tilelang_times[crossover_idx]:.2f}x")
    else:
        # Linear extrapolation to predict crossover
        if len(batch_sizes) >= 3:
            # Fit linear models (log-scale for better fit)
            log_batch = np.log2(batch_sizes)
            
            # PyTorch: fit y = a*x + b
            pt_slope, pt_intercept = np.polyfit(log_batch, pytorch_times, 1)
            tl_slope, tl_intercept = np.polyfit(log_batch, tilelang_times, 1)
            
            # Find intersection: pt_slope*x + pt_intercept = tl_slope*x + tl_intercept
            if pt_slope != tl_slope:
                crossover_log = (tl_intercept - pt_intercept) / (pt_slope - tl_slope)
                crossover_batch = int(2 ** crossover_log)
                crossover_pt = pt_slope * crossover_log + pt_intercept
                crossover_tl = tl_slope * crossover_log + tl_intercept
                
                print(f"\n🔮 Predicted Crossover (linear extrapolation):")
                print(f"   Batch size: ~{crossover_batch:,}")
                print(f"   Estimated time: ~{crossover_pt:.3f}ms")
                print(f"   PyTorch slope: {pt_slope:.6f} ms/log2(batch)")
                print(f"   TileLang slope: {tl_slope:.6f} ms/log2(batch)")
                
                # Predict at common large batch sizes
                print(f"\n📊 Extrapolation to larger batches:")
                for test_batch in [16384, 32768, 65536, 131072]:
                    log_b = np.log2(test_batch)
                    pred_pt = pt_slope * log_b + pt_intercept
                    pred_tl = tl_slope * log_b + tl_intercept
                    speedup = pred_pt / pred_tl
                    status = "✅ TileLang faster" if pred_tl < pred_pt else "❌ PyTorch faster"
                    print(f"   batch={test_batch:>6}: PyTorch={pred_pt:.3f}ms, TileLang={pred_tl:.3f}ms, speedup={speedup:.2f}x {status}")
    
    # Analyze efficiency (time per token)
    print(f"\n⚡ Efficiency (time per token in nanoseconds):")
    print("-" * 80)
    print(f"{'Batch':<10} {'PyTorch(ns/token)':<20} {'TileLang(ns/token)':<20} {'Ratio':<10}")
    print("-" * 80)
    for i, batch in enumerate(batch_sizes):
        pt_per_token = (pytorch_times[i] * 1e6) / batch  # convert ms to ns
        tl_per_token = (tilelang_times[i] * 1e6) / batch
        ratio = pt_per_token / tl_per_token
        print(f"{batch:<10} {pt_per_token:<20.2f} {tl_per_token:<20.2f} {ratio:<10.3f}x")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    csv_path = Path("benchmarks/results/topk_benchmark_results.csv")
    if csv_path.exists():
        analyze_trend(csv_path)
    else:
        print(f"Error: {csv_path} not found. Run benchmark_topk.py first.")
