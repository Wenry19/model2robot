
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_report(output_path, stats, latencies):
    save_raw_stats(output_path, stats)
    save_raw_latencies(output_path, latencies)
    plot_latencies(output_path, latencies)
    plot_latencies_distribution(output_path, latencies)

def save_raw_stats(output_path, stats):

    output_file = os.path.join(output_path, "benchmark_stats.json")

    with open(output_file, "w") as f:
        json.dump(stats, f, indent=4)

def save_raw_latencies(output_path, latencies):

    output_file = os.path.join(output_path, "latencies.csv")

    df = pd.DataFrame({"iteration": range(1, len(latencies) + 1),
                       "latency_seconds": latencies,
                       "latency_ms": latencies * 1000})

    df.to_csv(output_file, index=False)

def plot_latencies(output_path, latencies):

    plt.figure(figsize=(8, 5))

    plt.plot(np.arange(1, len(latencies) + 1),
             latencies * 1000)

    plt.xlabel("Iteration")
    plt.ylabel("Latency (ms)")
    plt.title("Inference Latency")

    plt.grid()
    plt.tight_layout()

    plt.savefig(os.path.join(output_path, "latency_curve.png"),
                dpi=300,
                bbox_inches="tight")

    plt.close()

def plot_latencies_distribution(output_path, latencies):

    plt.figure(figsize=(8, 5))

    plt.hist(latencies * 1000, bins=20)

    plt.xlabel("Latency (ms)")
    plt.ylabel("Frequency")
    plt.title("Inference Latency Distribution")

    plt.grid()
    plt.tight_layout()

    plt.savefig(os.path.join(output_path, "latency_distribution.png"),
                dpi=300,
                bbox_inches="tight")

    plt.close()
