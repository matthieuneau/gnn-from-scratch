import time

import torch


def benchmark(device):
    # Create two large random matrices on the specified device.
    A = torch.randn(2000, 20000, device=device)
    B = torch.randn(20000, 2000, device=device)

    # Synchronize GPU (if applicable) before starting the timer.
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()

    # Perform 100 matrix multiplications.
    for _ in range(100):
        C = torch.mm(A, B)

    # Synchronize GPU (if applicable) after operations are complete.
    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.time()

    print(f"Time on {device}: {end - start:.4f} seconds")


if __name__ == "__main__":
    # Benchmark on CPU.
    cpu_device = torch.device("cpu")
    print("Benchmarking on CPU:")
    benchmark(cpu_device)

    # Benchmark on GPU if available.
    if torch.cuda.is_available():
        cuda_device = torch.device("cuda")
        print("Benchmarking on GPU:")
        benchmark(cuda_device)
    else:
        print("CUDA is not available on this machine.")
