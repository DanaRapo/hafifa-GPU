import time
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from logic.computes.Basic_Channelizer import BasicChannelizer
from logic.computes.polyphase import PolyphaseChannelizer

def run_polyphase_benchmark():
    # 1. Initialize Polyphase Channelizer Config
    poly_config = PolyphaseChannelizer.Config(
        channel_bw_hz = 1000.0,
        fs_hz = 10000.0,
        filter_path = "/home/test3/Desktop/git/hafifa-GPU/channelizer/channelizer-python/signals/filter_from_designer.bin",
        decimation_factor = 10,
        grid_spacing_hz = 1000.0
    )
    poly_module = poly_config.create_logical_instance()
    poly_module.initialize()

    basic_config = BasicChannelizer.Config(
        channel_bw_hz=1000.0,
        fs_hz=10000,
        up_sample_factor=1
    )
    basic_module = basic_config.create_logical_instance()

    sizes_gb = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0] 
    basic_times = []
    cpu_times = []
    gpu_times = []

    for size in sizes_gb:
        num_samples = int((size * 1e9) / 8)
        # Create dummy complex data on CPU
        data_cpu = np.random.randn(num_samples).astype(np.complex64) + \
                   1j * np.random.randn(num_samples).astype(np.complex64)

        # --- Basic Channelizer Benchmark (Python Loops) ---
        start_basic = time.perf_counter()
        basic_module.run(data_cpu)
        end_basic = time.perf_counter()
        basic_duration = end_basic - start_basic
        basic_times.append(basic_duration)

        # --- CPU Benchmark ---
        start_cpu = time.perf_counter()
        poly_module.run(data_cpu)
        end_cpu = time.perf_counter()
        cpu_duration = end_cpu - start_cpu
        cpu_times.append(cpu_duration)

        # --- GPU Benchmark ---
        # included the transfer time to GPU (asarray) as part of the benchmark
        start_gpu = time.perf_counter()
        data_gpu = cp.asarray(data_cpu) # Move to VRAM
        res_gpu = poly_module.run(data_gpu)  # Run on GPU
        cp.cuda.Stream.null.synchronize() # Wait for GPU to finish!
        end_gpu = time.perf_counter()
        
        gpu_duration = end_gpu - start_gpu
        gpu_times.append(gpu_duration)
        
        print(f"Size: {size:>5} GB | CPU: {cpu_duration:.4f}s | GPU: {gpu_duration:.4f}s | Basic: {basic_duration:.4f}s")

    plt.figure(figsize=(12, 7), facecolor='w')
    
    plt.plot(sizes_gb, basic_times, 'o-r', linewidth=2, label='Basic Channelizer (Python Loops)')
    plt.plot(sizes_gb, cpu_times, 'o-b', linewidth=2, label='Polyphase CPU (NumPy Vectorized)')
    plt.plot(sizes_gb, gpu_times, 's-g', linewidth=2, label='Polyphase GPU (CuPy)')
    
    plt.xlabel('Data Size (GB)', fontsize=12)
    plt.ylabel('Execution Time (seconds)', fontsize=12)
    plt.title('Channelizer Architecture Comparison: Execution Time vs Data Size', fontsize=14)
    plt.grid(True, which="major", linestyle='--', alpha=0.7)
    plt.xticks(sizes_gb)
    plt.legend(fontsize=10)
    
    output_filename = "polyphase_performance_comparison.png"
    plt.savefig(output_filename)
    plt.show()
    print(f"\nBenchmark graph saved as {output_filename}")

if __name__ == "__main__":
    run_polyphase_benchmark()