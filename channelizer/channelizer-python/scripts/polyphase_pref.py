import time
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import pydantic
from logic.computes.Basic_Channelizer import BasicChannelizer
from logic.computes.polyphase import PolyphaseChannelizer

def run_polyphase_benchmark():
    # Common parameters
    fs = 10000.0
    bw = 1000.0
    decim = 10
    filter_path = "/home/test3/Desktop/git/hafifa-GPU/channelizer/channelizer-python/signals/filter_from_designer.bin"

    # 1. Initialize CPU Polyphase Instance
    cpu_poly_config = PolyphaseChannelizer.Config(
        channel_bw_hz = bw,
        fs_hz = fs,
        filter_path = filter_path,
        decimation_factor = decim,
        grid_spacing_hz = bw, # Added missing field if required
        use_gpu = False
    )
    cpu_poly_module = cpu_poly_config.create_logical_instance()
    cpu_poly_module.initialize()

    # 2. Initialize GPU Polyphase Instance
    gpu_poly_config = PolyphaseChannelizer.Config(
        channel_bw_hz = bw,
        fs_hz = fs,
        filter_path = filter_path,
        decimation_factor = decim,
        grid_spacing_hz = bw,
        use_gpu = True
    )
    gpu_poly_module = gpu_poly_config.create_logical_instance()
    gpu_poly_module.initialize()

    # 3. Initialize Basic Channelizer
    basic_config = BasicChannelizer.Config(
        channel_bw_hz=bw,
        fs_hz=fs,
        up_sample_factor=1
    )
    basic_module = basic_config.create_logical_instance()

    sizes_gb = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0] 
    basic_times = []
    cpu_times = []
    gpu_times = []

    for size in sizes_gb:
        num_samples = int((size * 1e9) / 8)
        data_cpu = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)).astype(np.complex64)

        # --- Basic Channelizer Benchmark ---
        start_basic = time.perf_counter()
        basic_module.run(data_cpu)
        basic_times.append(time.perf_counter() - start_basic)

        # --- Polyphase CPU Benchmark ---
        start_cpu = time.perf_counter()
        cpu_poly_module.run(data_cpu)
        cpu_times.append(time.perf_counter() - start_cpu)

        # --- Polyphase GPU Benchmark ---
        start_gpu = time.perf_counter()

        data_gpu = cp.asarray(data_cpu) 
        gpu_poly_module.run(data_gpu) 
        cp.cuda.Stream.null.synchronize() 
        gpu_times.append(time.perf_counter() - start_gpu)
        
        print(f"Size: {size:>5} GB | CPU: {cpu_times[-1]:.4f}s | GPU: {gpu_times[-1]:.4f}s | Basic: {basic_times[-1]:.4f}s")

    plt.figure(figsize=(12, 7), facecolor='w')
    plt.plot(sizes_gb, basic_times, 'o-r', linewidth=2, label='Basic Channelizer (Python Loops)')
    plt.plot(sizes_gb, cpu_times, 'o-b', linewidth=2, label='Polyphase CPU (NumPy)')
    plt.plot(sizes_gb, gpu_times, 's-g', linewidth=2, label='Polyphase GPU (CuPy + Transfer)')
    
    plt.xlabel('Data Size (GB)')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Performance Comparison: CPU vs GPU vs Basic')
    plt.grid(True, which="major", linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.savefig("polyphase_performance_comparison.png")
    plt.show()

if __name__ == "__main__":
    run_polyphase_benchmark()