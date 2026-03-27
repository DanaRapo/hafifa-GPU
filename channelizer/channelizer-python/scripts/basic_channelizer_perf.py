import time
import numpy as np
import matplotlib.pyplot as plt
from logic.computes.Basic_Channelizer import BasicChannelizer

def run_benchmark():
    # 1. Initialize Channelizer
    config = BasicChannelizer.Config(
        channel_bw_hz=1000.0,
        fs_hz=10000,
        up_sample_factor=1
    )
    module = config.create_logical_instance()

    # 2. Define data sizes in GB 
    sizes_gb = [0.01, 0.1, 0.25, 0.5, 1.0] 
    execution_times = []

    for size in sizes_gb:
        # Calculate number of complex64 samples
        num_samples = int((size * 1e9) / 8)
        
        # Create dummy complex data
        data = np.random.randn(num_samples).astype(np.complex64) + \
               1j * np.random.randn(num_samples).astype(np.complex64)
        
        # Measure execution time
        start = time.perf_counter()
        module.run(data)
        end = time.perf_counter()
        
        duration = end - start
        execution_times.append(duration)
        print(f"Processed {size} GB in {duration:.4f} seconds")

    plt.figure(figsize=(10, 6))
    plt.plot(sizes_gb, execution_times, 'o-r', linewidth=2, label='Python Runtime')
    plt.xlabel('Data Size (GB)')
    plt.ylabel('Time (seconds)')
    plt.title('Channelizer Scalability: Data Size vs Execution Time')
    plt.grid(True, linestyle='--')
    plt.legend()
    output_filename = "performance_graph.png"
    plt.savefig(output_filename)
    print(f"\nGraph saved successfully as {output_filename}")

if __name__ == "__main__":
    run_benchmark()