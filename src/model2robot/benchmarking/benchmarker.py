
import time
import numpy as np

import os
import pynvml

class Benchmarker:

    def __init__(self, inference, warmup_iterations, benchmark_iterations):

        self.inference = inference
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations

        pynvml.nvmlInit()

        # TODO: make GPU selection configurable if needed
        self.gpu_id = 0
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)

    def run(self, dataloader):

        if not dataloader.drop_last:
            raise ValueError("Benchmark requires DataLoader with drop_last=True.")

        baseline_memory = self._get_gpu_memory()
        peak_memory = baseline_memory

        # Warm-up
        for i, images in enumerate(self._get_batches(dataloader)):
            if i == 0:
                self.inference.synchronize()
                start = time.perf_counter()
                self.inference.run(images)
                self.inference.synchronize()
                end = time.perf_counter()
                first_inference_latency = end - start
            else:
                self.inference.run(images)
            if i == self.warmup_iterations-1:
                break

        # Benchmark
        latencies = []

        for images in self._get_batches(dataloader):

            # Make sure previous inference has finished (redundant)
            self.inference.synchronize()
            start = time.perf_counter()
            self.inference.run(images)
            self.inference.synchronize()
            end = time.perf_counter()

            latencies.append(end - start)

            current_memory = self._get_gpu_memory()
            peak_memory = max(peak_memory, current_memory)

            if len(latencies) >= self.benchmark_iterations:
                break

        return self._calculate_stats(first_inference_latency,
                                     latencies,
                                     dataloader.batch_size,
                                     baseline_memory,
                                     peak_memory)

    def _get_batches(self, dataloader):
        # infinite generator
        while True:
            for images, _ in dataloader:
                yield images

    def _calculate_stats(self, first_inference_latency, latencies, batch_size, baseline_memory, peak_memory):

        latencies = np.asarray(latencies)

        return {
            "first_inference_latency": first_inference_latency,
            "mean_latency": np.mean(latencies),
            "median_latency": np.median(latencies),
            "std_latency": np.std(latencies),
            "min_latency": np.min(latencies),
            "max_latency": np.max(latencies),
            "throughput": batch_size / np.mean(latencies),
            "gpu_memory_baseline_mib": self._bytes_to_mib(baseline_memory),
            "gpu_memory_peak_mib": self._bytes_to_mib(peak_memory),
            "gpu_memory_additional_mib": self._bytes_to_mib(peak_memory - baseline_memory),
        }, latencies

    def _get_gpu_memory(self):
        try:
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
        except pynvml.NVMLError as e:
            raise RuntimeError("Failed to retrieve GPU memory usage.") from e
        return memory_info.used

    def _bytes_to_mib(self, value):
        return value / (1024 ** 2)

    def close(self):
        pynvml.nvmlShutdown()