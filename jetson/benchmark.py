"""TensorRT Benchmark Suite for NVIDIA Jetson Orin Nano.

Measures inference latency, frames per second (FPS), percentile latencies (P50, P90, P99),
and GPU memory consumption for serialized TensorRT engines using synthetic input data.

NOTE ON POWER AND THERMAL MONITORING (tegrastats):
--------------------------------------------------
To capture real-time power dissipation, GPU core utilization, and thermal throttle stats
during benchmarking on NVIDIA Jetson Orin Nano, run tegrastats in a secondary terminal:

    # 1. Lock Jetson Orin Nano to maximum power profile and clocks:
    sudo nvpmodel -m 0          # Set 15W Mode (Max-N on Orin Nano)
    sudo jetson_clocks          # Pin clocks at max frequency

    # 2. Start tegrastats logging:
    sudo tegrastats --interval 1000 --logfile tegrastats_benchmark.csv

Key tegrastats fields to monitor:
    - RAM: Unified memory usage (MB / MB total)
    - GR3D_FREQ: Orin Ampere GPU utilization % and clock rate (MHz)
    - VDD_GPU_SOC / VDD_CPU_CV: Real-time power draw in mW
    - thermal-zone temperatures (GPU / CPU / SOC in Celsius)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark")

# Attempt importing TensorRT and PyCUDA if available
try:
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    import tensorrt as trt

    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    cuda = None
    trt = None


class TensorRTBenchmarkRunner:
    """Manages loading, memory allocation, and execution profiling for TensorRT engines."""

    def __init__(
        self,
        engine_path: str | Path,
        input_shape: Tuple[int, int, int, int] = (1, 3, 512, 512),
    ) -> None:
        """Initialize benchmark runner.

        Args:
            engine_path: Path to serialized TensorRT .engine plan file.
            input_shape: 4D input tensor shape (Batch, Channels, Height, Width).
        """
        self.engine_path = Path(engine_path)
        self.input_shape = input_shape
        self.engine = None
        self.context = None
        self.stream = None
        self.bindings: List[int] = []
        self.host_inputs: List[np.ndarray] = []
        self.cuda_inputs: List[int] = []
        self.host_outputs: List[np.ndarray] = []
        self.cuda_outputs: List[int] = []

        self._setup_engine()

    def _setup_engine(self) -> None:
        """Load TensorRT engine and allocate CUDA pinned host/device memory.

        TODO: When running on JetPack Orin Nano:
            1. trt_logger = trt.Logger(trt.Logger.WARNING)
            2. runtime = trt.Runtime(trt_logger)
            3. with open(self.engine_path, 'rb') as f:
                   self.engine = runtime.deserialize_cuda_engine(f.read())
            4. self.context = self.engine.create_execution_context()
            5. self.stream = cuda.Stream()
            6. Query engine input/output bindings and allocate page-locked buffers:
                   host_mem = cuda.pagelocked_empty(size, dtype)
                   device_mem = cuda.mem_alloc(host_mem.nbytes)
        """
        if not TRT_AVAILABLE:
            logger.warning(
                "TensorRT/PyCUDA not found in this environment. "
                "Running benchmark in mock/simulation mode."
            )
            return

        logger.info("Loading TensorRT engine: %s", self.engine_path)
        if not self.engine_path.exists():
            raise FileNotFoundError(f"Engine file not found: {self.engine_path}")

        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        with open(self.engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError("Failed to deserialize TensorRT CUDA engine.")

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # Allocate host and device memory buffers
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(device_mem)
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(device_mem)

        logger.info("Successfully allocated CUDA buffers for TRT engine.")

    def run_iteration(self, dummy_data: np.ndarray) -> None:
        """Execute one complete forward inference pass.

        Args:
            dummy_data: Input numpy array matching expected shape and type.
        """
        if not TRT_AVAILABLE or self.context is None:
            # Simulated inference sleep matching target ~30-40 FPS latency (~25ms)
            time.sleep(0.025)
            return

        # 1. Copy host input data to page-locked memory
        np.copyto(self.host_inputs[0], dummy_data.ravel())

        # 2. Transfer from Host to Device asynchronously
        cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)

        # 3. Execute inference kernel on GPU
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle,
        )

        # 4. Transfer output from Device to Host asynchronously
        cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)

        # 5. Synchronize CUDA stream
        self.stream.synchronize()

    def get_gpu_memory_usage(self) -> Tuple[float, float]:
        """Query current GPU unified memory usage (Free MB, Total MB).

        Returns:
            Tuple[float, float]: (Free memory in MB, Total memory in MB).
        """
        if TRT_AVAILABLE and cuda is not None:
            free_bytes, total_bytes = cuda.mem_get_info()
            return free_bytes / (1024**2), total_bytes / (1024**2)
        return 0.0, 0.0


# ==============================================================================
# Benchmark Orchestration
# ==============================================================================
def benchmark_engine(
    engine_path: str,
    input_size: Tuple[int, int] = (512, 512),
    batch_size: int = 1,
    num_frames: int = 500,
    warmup_frames: int = 50,
) -> None:
    """Execute benchmarking suite and report latency and throughput statistics.

    Args:
        engine_path: Path to serialized TensorRT .engine plan file.
        input_size: Spatial dimensions (Height, Width).
        batch_size: Batch size (default 1).
        num_frames: Number of timed inference iterations.
        warmup_frames: Number of un-timed warmup iterations.
    """
    height, width = input_size
    input_shape = (batch_size, 3, height, width)

    print("======================================================================")
    print("           NVIDIA Jetson Orin Nano - TensorRT Benchmark               ")
    print("======================================================================")
    print(f" Engine File        : {engine_path}")
    print(f" Input Shape (NCHW) : {input_shape}")
    print(f" Timed Iterations   : {num_frames}")
    print(f" Warmup Iterations  : {warmup_frames}")
    print("======================================================================")

    # Instantiate runner
    runner = TensorRTBenchmarkRunner(engine_path=engine_path, input_shape=input_shape)

    # Generate synthetic input tensor (float32, normal distribution)
    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # Initial GPU Memory measurement
    free_mem_start, total_mem = runner.get_gpu_memory_usage()

    # 1. Warm-up Phase
    logger.info("Executing %d warmup iterations...", warmup_frames)
    for _ in range(warmup_frames):
        runner.run_iteration(dummy_input)

    # 2. Timed Benchmark Phase
    logger.info("Running %d timed inference iterations...", num_frames)
    latencies_sec: List[float] = []

    total_start = time.perf_counter()
    for i in range(num_frames):
        t0 = time.perf_counter()
        runner.run_iteration(dummy_input)
        t1 = time.perf_counter()
        latencies_sec.append(t1 - t0)

        if (i + 1) % 100 == 0:
            logger.info("Progress: %d/%d iterations completed...", i + 1, num_frames)

    total_elapsed = time.perf_counter() - total_start
    free_mem_end, _ = runner.get_gpu_memory_usage()

    # 3. Compute Metrics
    latencies_ms = np.array(latencies_sec) * 1000.0

    avg_latency = float(np.mean(latencies_ms))
    min_latency = float(np.min(latencies_ms))
    max_latency = float(np.max(latencies_ms))
    std_latency = float(np.std(latencies_ms))
    p50_latency = float(np.percentile(latencies_ms, 50))
    p90_latency = float(np.percentile(latencies_ms, 90))
    p99_latency = float(np.percentile(latencies_ms, 99))
    fps = num_frames / total_elapsed

    # 4. Display Results
    print("\n======================================================================")
    print("                         BENCHMARK RESULTS                            ")
    print("======================================================================")
    print(f" Throughput (Avg FPS)   : {fps:.2f} frames/sec")
    print(f" Average Latency        : {avg_latency:.2f} ms (±{std_latency:.2f} ms)")
    print(f" Min Latency            : {min_latency:.2f} ms")
    print(f" Max Latency            : {max_latency:.2f} ms")
    print(" Latency Percentiles:")
    print(f"   - 50th Percentile (P50): {p50_latency:.2f} ms")
    print(f"   - 90th Percentile (P90): {p90_latency:.2f} ms")
    print(f"   - 99th Percentile (P99): {p99_latency:.2f} ms")

    if total_mem > 0:
        used_mem = total_mem - free_mem_end
        print(" GPU Memory Consumption:")
        print(f"   - Total Device Memory : {total_mem:.1f} MB")
        print(f"   - Used Memory         : {used_mem:.1f} MB ({used_mem / total_mem * 100:.1f}%)")
        print(f"   - Free Memory         : {free_mem_end:.1f} MB")
    else:
        print(" GPU Memory Consumption: (PyCUDA not active; query via tegrastats)")

    print("======================================================================")
    print(" POWER & THERMAL MONITORING ADVISORY:")
    print(" For hardware power draw (mW), thermal throttling, and GPU clock metrics:")
    print("   1. Open a new terminal on the Jetson Orin Nano.")
    print("   2. Run: sudo tegrastats --interval 1000")
    print("   3. Look for 'POM_5V_IN' (total power) and 'GR3D_FREQ' (GPU activity).")
    print("======================================================================\n")


# ==============================================================================
# CLI Entry Point
# ==============================================================================
def parse_args() -> argparse.Namespace:
    """Parse benchmark arguments."""
    parser = argparse.ArgumentParser(
        description="TensorRT Inference Benchmark for NVIDIA Jetson Orin Nano",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--engine",
        type=str,
        required=True,
        help="Path to serialized TensorRT .engine plan file",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=[512, 512],
        metavar=("HEIGHT", "WIDTH"),
        help="Model input resolution as: Height Width",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=500,
        help="Number of timed inference frames",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Number of warmup inference frames before timing",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Inference batch size",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    benchmark_engine(
        engine_path=args.engine,
        input_size=(args.input_size[0], args.input_size[1]),
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        warmup_frames=args.warmup,
    )
