import time
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

from logic.computes.polyphase import PolyphaseChannelizer

def run_polyphase_benchmark():
    # 1. Initialize Polyphase Channelizer Config
    config = PolyphaseChannelizer.Config(
        channel_bw_hz=1000.0,
        fs_hz=10000.0,
        filter_path="/home/test3/Desktop/git/hafifa-GPU/channelizer/channelizer-python/signals/filter_from_designer.bin",
        decimation_factor=10
    )
    module = config.create_logical_instance()
    module.initialize()

    sizes_gb = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0] 
    cpu_times = []
    gpu_times = []

    for size in sizes_gb:
        num_samples = int((size * 1e9) / 8)
        # Create dummy complex data on CPU
        data_cpu = np.random.randn(num_samples).astype(np.complex64) + \
                   1j * np.random.randn(num_samples).astype(np.complex64)
        
        # --- CPU Benchmark ---
        start_cpu = time.perf_counter()
        module.run(data_cpu) # Module detects numpy -> runs on CPU
        end_cpu = time.perf_counter()
        cpu_duration = end_cpu - start_cpu
        cpu_times.append(cpu_duration)

        # --- GPU Benchmark ---
        # included the transfer time to GPU (asarray) as part of the benchmark
        # because in real life, data often starts in the RAM.
        start_gpu = time.perf_counter()
        data_gpu = cp.asarray(data_cpu) # Move to VRAM
        res_gpu = module.run(data_gpu)  # Run on GPU
        cp.cuda.Stream.null.synchronize() # Wait for GPU to finish!
        end_gpu = time.perf_counter()
        
        gpu_duration = end_gpu - start_gpu
        gpu_times.append(gpu_duration)
        
        print(f"Size: {size:>5} GB | CPU: {cpu_duration:.4f}s | GPU: {gpu_duration:.4f}s")

    # 3. Plotting
    plt.figure(figsize=(10, 6), facecolor='w')
    plt.plot(sizes_gb, cpu_times, 'o-', color='blue', linewidth=2, label='CPU (NumPy)')
    plt.plot(sizes_gb, gpu_times, 's-', color='green', linewidth=2, label='GPU (CuPy)')
    
    plt.xscale('log') # Use log scale if sizes vary significantly
    plt.yscale('log')
    plt.xlabel('Data Size (GB)')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Polyphase Channelizer Performance: CPU vs GPU')
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    plt.legend()
    
    output_filename = "polyphase_performance_comparison.png"
    plt.savefig(output_filename)
    plt.show()
    print(f"\nBenchmark graph saved as {output_filename}")

if __name__ == "__main__":
    run_polyphase_benchmark()