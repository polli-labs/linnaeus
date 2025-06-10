# linnaeus/evaluation/results_logger.py

import csv
import os

import matplotlib.pyplot as plt


class ResultsLogger:
    def __init__(self, config):
        self.results_dir = config.ENV.OUTPUT.DIRS.LOGS
        os.makedirs(self.results_dir, exist_ok=True)

    def save_to_csv(self, results, filename="throughput_results.csv"):
        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, "w", newline="") as csvfile:
            fieldnames = results[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        print(f"CSV results saved to {filepath}")

    def save_summary(self, results, filename="throughput_summary.txt"):
        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, "w") as f:
            f.write("Throughput Test Summary\n")
            f.write("=======================\n\n")
            for result in results:
                f.write(f"Batch Size: {result['batch_size']}\n")
                f.write(f"Images/second: {result['imgs_per_sec']:.2f}\n")
                f.write(f"Memory Used (GB): {result['memory_used_gb']:.2f}\n")
                f.write(f"GPU Utilization: {result['gpu_utilization']}%\n")
                f.write("\n")
        print(f"Summary saved to {filepath}")

    def plot_results(self, results, filename="throughput_plot.png"):
        batch_sizes = [r["batch_size"] for r in results]
        imgs_per_sec = [r["imgs_per_sec"] for r in results]
        memory_used = [r["memory_used_gb"] for r in results]

        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.set_xlabel("Batch Size")
        ax1.set_ylabel("Images/Second", color="tab:blue")
        ax1.plot(batch_sizes, imgs_per_sec, color="tab:blue", marker="o")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        ax2 = ax1.twinx()
        ax2.set_ylabel("Memory Used (GB)", color="tab:orange")
        ax2.plot(batch_sizes, memory_used, color="tab:orange", marker="s")
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        plt.title("Throughput and Memory Usage vs Batch Size")
        fig.tight_layout()

        filepath = os.path.join(self.results_dir, filename)
        plt.savefig(filepath)
        print(f"Plot saved to {filepath}")
        plt.close()

    def log_results(self, results):
        self.save_to_csv(results)
        self.save_summary(results)
        self.plot_results(results)
