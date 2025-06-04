import matplotlib.pyplot as plt
import numpy as np
import re
import os

# Attempt to initialize mplfonts for Chinese font support
try:
    from mplfonts.bin.cli import init
    init()
    plt.rcParams['font.family'] = ['Source Han Sans CN', 'sans-serif'] # Add fallback
    plt.rcParams['axes.unicode_minus'] = False
except ImportError:
    print("mplfonts not found. Using default fonts. Chinese characters may not display correctly.")
    plt.rcParams['font.family'] = ['sans-serif'] # Default fallback
    plt.rcParams['axes.unicode_minus'] = False

LOG_FILE = "../log/mlp_forward_perf.log" # Path relative to src directory
REPORT_DIR = "../report"

def parse_log_file(log_file_path):
    """Parses the log file to extract performance timings."""
    timings = {
        "cpu_total_time_ms": None,
        "htod_time_ms": None,
        "matmul1_time_ms": None,
        "add_bias1_time_ms": None,
        "relu_time_ms": None,
        "matmul2_time_ms": None,
        "add_bias2_time_ms": None,
        "dtoh_time_ms": None,
        "total_kernel_time_ms": None, # DCU Kernel total
        "total_mlp_time_ms": None    # DCU MLP total (HtoD + Kernels + DtoH)
    }

    try:
        with open(log_file_path, 'r') as f:
            content = f.read()

            # Regex patterns to find the timings
            patterns = {
                "cpu_total_time_ms": r"CPU reference computation finished. Time: (\d+\.\d+) ms",
                "htod_time_ms": r"Time for HtoD copy: (\d+\.\d+) ms",
                "matmul1_time_ms": r"Time for MatMul1 Kernel: (\d+\.\d+) ms",
                "add_bias1_time_ms": r"Time for AddBias1 Kernel: (\d+\.\d+) ms",
                "relu_time_ms": r"Time for ReLU Kernel: (\d+\.\d+) ms",
                "matmul2_time_ms": r"Time for MatMul2 Kernel: (\d+\.\d+) ms",
                "add_bias2_time_ms": r"Time for AddBias2 Kernel: (\d+\.\d+) ms",
                "dtoh_time_ms": r"Time for DtoH copy: (\d+\.\d+) ms",
                "total_kernel_time_ms": r"Total Kernel Execution Time: (\d+\.\d+) ms",
                "total_mlp_time_ms": r"Total MLP Forward Time \(HtoD \+ Kernels \+ DtoH\): (\d+\.\d+) ms"
            }

            for key, pattern in patterns.items():
                match = re.search(pattern, content)
                if match:
                    timings[key] = float(match.group(1))
                else:
                    print(f"Warning: Could not find timing for '{key}' in log file.")

    except FileNotFoundError:
        print(f"Error: Log file not found at {log_file_path}")
        return None
    except Exception as e:
        print(f"Error parsing log file: {e}")
        return None
    
    # Validate that essential DCU timings were found for breakdown chart
    dcu_breakdown_keys = ["htod_time_ms", "matmul1_time_ms", "add_bias1_time_ms", "relu_time_ms", 
                            "matmul2_time_ms", "add_bias2_time_ms", "dtoh_time_ms"]
    if not all(timings[key] is not None for key in dcu_breakdown_keys):
        print("Error: Missing one or more DCU component timings needed for breakdown chart.")
        # If total_kernel_time or total_mlp_time is missing, other charts might also fail
        if timings["total_mlp_time_ms"] is None or timings["cpu_total_time_ms"] is None:
             print("Error: Critical timings missing for comparison chart.")
             return None # Can't make charts if critical data is missing

    return timings

def plot_mlp_performance_breakdown(timings, output_path):
    """Plots the breakdown of DCU MLP execution time."""
    labels = ['HtoD', 'MatMul1', 'AddBias1', 'ReLU', 'MatMul2', 'AddBias2', 'DtoH']
    values = [
        timings.get("htod_time_ms", 0),
        timings.get("matmul1_time_ms", 0),
        timings.get("add_bias1_time_ms", 0),
        timings.get("relu_time_ms", 0),
        timings.get("matmul2_time_ms", 0),
        timings.get("add_bias2_time_ms", 0),
        timings.get("dtoh_time_ms", 0)
    ]

    if not any(v is not None and v > 0 for v in values):
        print("Skipping breakdown plot: No valid data.")
        return
    
    # Filter out None values for plotting if any occurred despite earlier checks
    plot_labels = [l for l, v in zip(labels, values) if v is not None]
    plot_values = [v for v in values if v is not None]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(plot_labels, plot_values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'lightcoral', 'lightgreen', 'skyblue'])
    plt.ylabel('Time (ms)')
    plt.title('MLP Forward Pass on DCU: Performance Breakdown')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05 * max(plot_values, default=1), f'{yval:.2f}', ha='center', va='bottom')

    plt.savefig(os.path.join(output_path, 'mlp_performance_breakdown.png'))
    plt.savefig(os.path.join(output_path, 'mlp_performance_breakdown.pdf'))
    print(f"Saved MLP performance breakdown chart to {output_path}")
    plt.close()

def plot_cpu_vs_dcu_comparison(timings, output_path):
    """Plots the comparison of total CPU time vs. total DCU MLP time."""
    cpu_time = timings.get("cpu_total_time_ms")
    dcu_time = timings.get("total_mlp_time_ms")

    if cpu_time is None or dcu_time is None:
        print("Skipping CPU vs DCU comparison plot: Missing data.")
        return

    labels = ['CPU', 'DCU']
    values = [cpu_time, dcu_time]

    plt.figure(figsize=(6, 6))
    bars = plt.bar(labels, values, color=['salmon', 'lightseagreen'])
    plt.ylabel('Total Time (ms)')
    plt.title('MLP Forward Pass: CPU vs DCU Performance')
    plt.yscale('log') # Use log scale if times are very different
    plt.tight_layout()

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', ha='center', va='bottom')

    plt.savefig(os.path.join(output_path, 'mlp_cpu_vs_dcu_comparison.png'))
    plt.savefig(os.path.join(output_path, 'mlp_cpu_vs_dcu_comparison.pdf'))
    print(f"Saved CPU vs DCU comparison chart to {output_path}")
    plt.close()

if __name__ == "__main__":
    # Ensure report directory exists
    if not os.path.exists(REPORT_DIR):
        os.makedirs(REPORT_DIR)
        print(f"Created report directory: {REPORT_DIR}")

    parsed_timings = parse_log_file(LOG_FILE)

    if parsed_timings:
        print("Successfully parsed timings:", parsed_timings)
        plot_mlp_performance_breakdown(parsed_timings, REPORT_DIR)
        plot_cpu_vs_dcu_comparison(parsed_timings, REPORT_DIR)
    else:
        print("Failed to parse timings. Cannot generate plots.") 