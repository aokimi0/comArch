import re
import matplotlib.pyplot as plt
import numpy as np
import os

def parse_log_file(log_path):
    """Parses the MLP training log file."""
    epochs_data = []
    test_data = {
        "mse_normalized": None,
        "mse_denormalized": None,
        "total_test_time_ms": None,
        "avg_latency_ms_per_sample": None,
        "throughput_samples_per_sec": None,
        "predictions": []
    }

    epoch_pattern = re.compile(
        r"\[Epoch (?P<epoch>\d+)/\d+\] Avg Loss: (?P<loss>[\d.]+), Time: (?P<time>[\d.]+) ms "
        r"\(HtoD: (?P<htod>[\d.]+), Kernels: (?P<kernels>[\d.]+), DtoH: (?P<dtoh>[\d.]+)\)"
    )
    mse_norm_pattern = re.compile(r"Average Test MSE \(Normalized\): (?P<mse>[\d.]+)")
    mse_denorm_pattern = re.compile(r"Average Test MSE \(Denormalized\): (?P<mse>[\d.]+)")
    total_test_time_pattern = re.compile(r"Total Simulated Test/Inference Time: (?P<time>[\d.]+) ms for \d+ samples")
    avg_latency_pattern = re.compile(r"Avg Simulated Inference Latency per Sample: (?P<latency>[\d.]+)")
    throughput_pattern = re.compile(r"Simulated Inference Throughput: (?P<throughput>[\d.]+)")
    prediction_pattern = re.compile(r"Pred: (?P<pred>[\d.]+), Actual: (?P<actual>[\d.]+)")

    with open(log_path, 'r') as f:
        for line in f:
            match = epoch_pattern.search(line)
            if match:
                epochs_data.append({
                    "epoch": int(match.group("epoch")),
                    "loss": float(match.group("loss")),
                    "time_ms": float(match.group("time")),
                    "htod_ms": float(match.group("htod")),
                    "kernels_ms": float(match.group("kernels")),
                    "dtoh_ms": float(match.group("dtoh")),
                })
                continue
            
            match = mse_norm_pattern.search(line)
            if match:
                test_data["mse_normalized"] = float(match.group("mse"))
                continue

            match = mse_denorm_pattern.search(line)
            if match:
                test_data["mse_denormalized"] = float(match.group("mse"))
                continue

            match = total_test_time_pattern.search(line)
            if match:
                test_data["total_test_time_ms"] = float(match.group("time"))
                continue
            
            match = avg_latency_pattern.search(line)
            if match:
                test_data["avg_latency_ms_per_sample"] = float(match.group("latency"))
                continue

            match = throughput_pattern.search(line)
            if match:
                test_data["throughput_samples_per_sec"] = float(match.group("throughput"))
                continue

            match = prediction_pattern.search(line)
            if match:
                test_data["predictions"].append({
                    "predicted": float(match.group("pred")),
                    "actual": float(match.group("actual"))
                })
    return epochs_data, test_data

def plot_training_loss(epochs_data, output_path):
    """Plots training loss vs. epoch."""
    if not epochs_data:
        print("No epoch data to plot training loss.")
        return
    epochs = [e['epoch'] for e in epochs_data]
    losses = [e['loss'] for e in epochs_data]
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, marker='o', linestyle='-')
    plt.title('Training Loss (MSE) vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Average MSE Loss')
    plt.grid(True)
    plt.xticks(epochs)
    plt.savefig(os.path.join(output_path, 'lesson3_training_loss.png'))
    plt.close()

def plot_training_time_breakdown(epochs_data, output_path):
    """Plots training time breakdown vs. epoch."""
    if not epochs_data:
        print("No epoch data to plot training time breakdown.")
        return
    epochs = [e['epoch'] for e in epochs_data]
    htod_times = np.array([e['htod_ms'] for e in epochs_data])
    kernels_times = np.array([e['kernels_ms'] for e in epochs_data])
    dtoh_times = np.array([e['dtoh_ms'] for e in epochs_data]) # May be zero

    plt.figure(figsize=(12, 7))
    plt.stackplot(epochs, htod_times, kernels_times, dtoh_times, 
                  labels=['HtoD Time (ms)', 'Kernels Time (ms)', 'DtoH Time (ms)'],
                  colors=['#1f77b4', '#ff7f0e', '#2ca02c']) # Example colors
    # For line plots instead of stackplot:
    # plt.plot(epochs, htod_times, marker='.', linestyle='-', label='HtoD Time (ms)')
    # plt.plot(epochs, kernels_times, marker='.', linestyle='-', label='Kernels Time (ms)')
    # if np.any(dtoh_times): # Only plot DtoH if it's not all zeros
    #     plt.plot(epochs, dtoh_times, marker='.', linestyle='-', label='DtoH Time (ms)')

    plt.title('Simulated Training Time Breakdown vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Time (ms)')
    plt.xticks(epochs)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(os.path.join(output_path, 'lesson3_training_time_breakdown.png'))
    plt.close()

def plot_predictions_vs_actual(test_data, output_path):
    """Plots a sample of predictions vs. actual values."""
    if not test_data or not test_data["predictions"]:
        print("No prediction data to plot.")
        return
        
    predictions_list = test_data["predictions"]
    preds = [p['predicted'] for p in predictions_list]
    actuals = [p['actual'] for p in predictions_list]
    
    num_samples = len(preds)
    indices = np.arange(num_samples)

    plt.figure(figsize=(12, 7))
    bar_width = 0.35
    plt.bar(indices - bar_width/2, actuals, bar_width, label='Actual Values', color='skyblue')
    plt.bar(indices + bar_width/2, preds, bar_width, label='Predicted Values', color='orange')

    plt.xlabel('Sample Index')
    plt.ylabel('Bandwidth (Denormalized)')
    plt.title('Sample Test Predictions vs. Actual Values')
    plt.xticks(indices)
    plt.legend()
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'lesson3_predictions_vs_actual.png'))
    plt.close()

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_file_path = os.path.join(script_dir, '../log/mlp_train_perf.log')
    report_dir = os.path.join(script_dir, '../report')

    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
        
    if not os.path.exists(log_file_path):
        print(f"Error: Log file not found at {log_file_path}")
        return

    epochs_data, test_data = parse_log_file(log_file_path)

    if epochs_data:
        print(f"Parsed {len(epochs_data)} epochs of training data.")
    else:
        print("Warning: No training epoch data found in log.")
        
    if test_data["mse_normalized"] is not None:
         print(f"Parsed test data: MSE Norm={test_data['mse_normalized']:.4f}, MSE Denorm={test_data['mse_denormalized']:.4f}")
         print(f"  Total Test Time: {test_data['total_test_time_ms']:.3f} ms")
         print(f"  Avg Latency: {test_data['avg_latency_ms_per_sample']:.3f} ms/sample")
         print(f"  Throughput: {test_data['throughput_samples_per_sec']:.2f} samples/sec")
    else:
        print("Warning: No test performance data found in log.")

    plot_training_loss(epochs_data, report_dir)
    plot_training_time_breakdown(epochs_data, report_dir)
    plot_predictions_vs_actual(test_data, report_dir)

    print(f"Plots saved to {report_dir}")

if __name__ == '__main__':
    main() 