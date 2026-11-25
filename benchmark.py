"""
benchmark.py - Measure Model Footprint, Memory Usage, Latency & FPS
--------------------------------------------------------------------
Usage:
    python benchmark.py
"""

import os
import sys
import time
import torch
import numpy as np

from model import CustomYOLOSeg


def measure_footprint():
    print("=" * 70)
    print("      CUSTOM YOLO-SEG BENCHMARK & MEMORY FOOTPRINT PROFILER        ")
    print("=" * 70)

    weights_file = "model.pt"
    if os.path.exists(weights_file):
        file_size_mb = os.path.getsize(weights_file) / (1024 * 1024)
    else:
        file_size_mb = 0.0

    # 1. Model Parameters
    model = CustomYOLOSeg(in_channels=3, num_classes=1)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    raw_params_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

    print(f"[Model Architecture Summary]")
    print(f" - Input Resolution            : 64 x 36 pixels (RGB, 3 channels)")
    print(f" - Output Resolution           : 64 x 36 logits -> 1280 x 720 binary mask")
    print(f" - Total Parameters            : {total_params:,} ({total_params/1e6:.2f}M)")
    print(f" - Trainable Parameters        : {trainable_params:,} (100% trained from scratch)")
    print(f" - Raw Float32 Parameter Size  : {raw_params_mb:.2f} MB")
    print(f" - Model File Size (model.pt)  : {file_size_mb:.2f} MB")
    print("-" * 70)

    # 2. GPU Benchmark
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[GPU Inference & Memory Profiling - {gpu_name}]")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        device = torch.device("cuda")
        model = model.to(device)
        model.eval()

        mem_after_load = torch.cuda.memory_allocated(device) / (1024 * 1024)

        # Batch Size = 1
        dummy_input = torch.randn(1, 3, 36, 64, device=device)

        # Warmup
        for _ in range(50):
            _ = model(dummy_input)
        torch.cuda.synchronize()

        # Benchmarking 200 runs
        runs = 200
        t0 = time.perf_counter()
        for _ in range(runs):
            _ = model(dummy_input)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        avg_latency_ms = ((t1 - t0) / runs) * 1000.0
        fps = 1000.0 / avg_latency_ms

        peak_vram = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        reserved_vram = torch.cuda.memory_reserved(device) / (1024 * 1024)

        print(f" - Static Model VRAM           : {mem_after_load:.2f} MB")
        print(f" - Peak Inference VRAM (B=1)   : {peak_vram:.2f} MB")
        print(f" - Total Reserved CUDA Memory  : {reserved_vram:.2f} MB")
        print(f" - Forward Pass Latency        : {avg_latency_ms:.3f} ms")
        print(f" - Pure Forward Throughput     : {fps:.1f} FPS")
    else:
        print("[GPU Inference] CUDA not available on this machine.")

    print("-" * 70)

    # 3. CPU Benchmark
    print(f"[CPU Inference Profiling]")
    model_cpu = CustomYOLOSeg(in_channels=3, num_classes=1).to("cpu")
    model_cpu.eval()
    dummy_cpu = torch.randn(1, 3, 36, 64)

    # Warmup
    for _ in range(20):
        _ = model_cpu(dummy_cpu)

    runs = 100
    t0 = time.perf_counter()
    for _ in range(runs):
        _ = model_cpu(dummy_cpu)
    t1 = time.perf_counter()

    avg_cpu_lat = ((t1 - t0) / runs) * 1000.0
    cpu_fps = 1000.0 / avg_cpu_lat

    print(f" - Forward Pass Latency (CPU)  : {avg_cpu_lat:.3f} ms")
    print(f" - Forward Throughput (CPU)    : {cpu_fps:.1f} FPS")
    print("=" * 70)


if __name__ == "__main__":
    measure_footprint()
