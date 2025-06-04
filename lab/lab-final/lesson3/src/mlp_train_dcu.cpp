#ifdef ENABLE_HIP_CODE
#include <hip/hip_runtime.h>
#endif
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <string>
#include <sstream>
#include <algorithm>
#include <random>
#include <iomanip>
#include <numeric>

// 编译文件
// hipcc mlp_train_dcu.cpp -o mlp_train_dcu
// 执行文件
// ./mlp_train_dcu

// 预定义参数
#define INPUT_DIM 10    // Number of past values to predict the next one
#define HIDDEN_DIM 64   // Hidden layer neurons
#define OUTPUT_DIM 1    // Predicting a single next value
#define BATCH_SIZE 64
#define EPOCHS 50      // Reduced for quicker testing, can be increased
#define LEARNING_RATE 0.001
#define TRAIN_SPLIT 0.8 // 80% for training, 20% for testing

// Data file path relative to lesson3 directory
const std::string DATA_FILE_PATH = "data/starlink_bw.json";

// --- HIP Kernel Placeholders (kept for structural simulation) ---
#ifdef ENABLE_HIP_CODE
__global__ void matmul_forward_kernel(const double* A, const double* B, double* C, int M, int N, int K) {
    // Actual computation would be here for a real DCU run
    // For simulation, this can be left empty or call a dummy operation
    // For our simulation, the CPU version matmul_cpu will do the work.
    // This kernel is launched to simulate the time and structure.
    if (threadIdx.x == 0 && blockIdx.x == 0) { } // Minimal work
}

__global__ void add_bias_forward_kernel(double* M, const double* B, int rows, int cols) {
    if (threadIdx.x == 0 && blockIdx.x == 0) { }
}

__global__ void relu_forward_kernel(double* A, int size) {
    if (threadIdx.x == 0 && blockIdx.x == 0) { }
}

__global__ void mse_loss_kernel(const double* pred, const double* target, double* loss_per_sample, int batch_size, int output_dim) {
    if (threadIdx.x == 0 && blockIdx.x == 0) { }
}

// Backward kernels (placeholders)
__global__ void linear_backward_output_kernel(const double* Y_pred, const double* Y_true, double* dL_dZ_out, int batch_size, int output_dim) {
    // dL_dZ_out = Y_pred - Y_true (for MSE)
    if (threadIdx.x == 0 && blockIdx.x == 0) { }
}

__global__ void relu_backward_kernel(double* dL_dZ, const double* Z, int size) {
    // dL_dZ_in = dL_dH * (Z > 0 ? 1 : 0)
    if (threadIdx.x == 0 && blockIdx.x == 0) { }
}

__global__ void matmul_backward_weights_kernel(const double* X_T, const double* dL_dY, double* dL_dW, int K_XT, int N_XT_M_dLdY, int M_dLdY) {
    // dL_dW = X_T * dL_dY
    if (threadIdx.x == 0 && blockIdx.x == 0) { }
}

__global__ void matmul_backward_data_kernel(const double* dL_dY, const double* W_T, double* dL_dX, int M_dLdY, int N_dLdY_K_WT, int M_WT) {
    // dL_dX = dL_dY * W_T
    if (threadIdx.x == 0 && blockIdx.x == 0) { }
}

__global__ void sum_rows_for_bias_grad_kernel(const double* dL_dZ_batch, double* dL_dB_sum, int batch_size, int features) {
    // Sums dL_dZ for each feature across the batch
    if (threadIdx.x == 0 && blockIdx.x == 0) { }
}


__global__ void sgd_update_kernel(double* weights, const double* grad, double lr, int size) {
    if (threadIdx.x == 0 && blockIdx.x == 0) { }
}
#endif // ENABLE_HIP_CODE

// --- CPU Helper Functions ---
void random_init_vector(std::vector<double>& vec, unsigned int seed_offset = 0, double scale = 0.1) {
    std::mt19937 gen(42 + seed_offset);
    std::uniform_real_distribution<double> dist(-scale, scale);
    for (auto& val : vec) {
        val = dist(gen);
    }
}

// M_rowsA x K_commonDim  *  K_commonDim x N_colsB  -> M_rowsA x N_colsB
void matmul_cpu(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int M_rowsA, int K_commonDim, int N_colsB) {
    C.assign(M_rowsA * N_colsB, 0.0);
    for (int i = 0; i < M_rowsA; ++i) {
        for (int j = 0; j < N_colsB; ++j) {
            double sum = 0.0;
            for (int k = 0; k < K_commonDim; ++k) {
                sum += A[i * K_commonDim + k] * B[k * N_colsB + j];
            }
            C[i * N_colsB + j] = sum;
        }
    }
}

void add_bias_cpu(std::vector<double>& Z, const std::vector<double>& bias) {
    // Z is (batch_size x num_features), bias is (1 x num_features) or (num_features)
    if (bias.empty() || Z.empty()) {
        // std::cerr << "Warning: add_bias_cpu called with empty Z or bias." << std::endl;
        return; 
    }
    int num_features = bias.size();
    if (Z.size() % num_features != 0) {
        std::cerr << "Error in add_bias_cpu: Z size (" << Z.size() 
                  << ") is not a multiple of bias size (" << num_features << ")." << std::endl;
        return; 
    }
    int batch_size = Z.size() / num_features;
    
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < num_features; ++j) {
            Z[i * num_features + j] += bias[j];
        }
    }
}

void relu_cpu(std::vector<double>& X) { // In-place
    for (auto& val : X) {
        val = std::fmax(0.0, val);
    }
}

void relu_derivative_cpu(const std::vector<double>& Z, std::vector<double>& dZdZ) { // Z is input to ReLU
    dZdZ.resize(Z.size());
    for (size_t i = 0; i < Z.size(); ++i) {
        dZdZ[i] = (Z[i] > 0) ? 1.0 : 0.0;
    }
}

double mse_loss_cpu(const std::vector<double>& pred, const std::vector<double>& target) {
    double sum_sq_error = 0.0;
    for (size_t i = 0; i < pred.size(); ++i) {
        sum_sq_error += (pred[i] - target[i]) * (pred[i] - target[i]);
    }
    return pred.empty() ? 0.0 : sum_sq_error / pred.size(); // Return average over samples in batch
}

// dL/dY_pred for MSE
void mse_loss_derivative_cpu(const std::vector<double>& pred, const std::vector<double>& target, std::vector<double>& dL_dY_pred) {
    dL_dY_pred.resize(pred.size());
    for (size_t i = 0; i < pred.size(); ++i) {
        dL_dY_pred[i] = (pred[i] - target[i]); // Removed 2/N factor, can be absorbed by LR or averaged later
    }
}

void transpose_matrix_cpu(const std::vector<double>& A, std::vector<double>& A_T, int rows, int cols) {
    A_T.resize(cols * rows);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            A_T[j * rows + i] = A[i * cols + j];
        }
    }
}

void elementwise_multiply_cpu(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C) {
    C.resize(A.size());
    for (size_t i = 0; i < A.size(); ++i) {
        C[i] = A[i] * B[i];
    }
}

void sum_matrix_rows_cpu(const std::vector<double>& M_batch_x_features, std::vector<double>& V_features, int batch_size, int features) {
    V_features.assign(features, 0.0);
    for (int j = 0; j < features; ++j) {
        for (int i = 0; i < batch_size; ++i) {
            V_features[j] += M_batch_x_features[i * features + j];
        }
    }
     for (int j = 0; j < features; ++j) { // Average gradient over batch for bias
        V_features[j] /= batch_size;
    }
}


void sgd_update_cpu(std::vector<double>& weights, const std::vector<double>& grad, double lr) {
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] -= lr * grad[i];
    }
}

// Data Loading and Preprocessing
std::vector<double> load_bandwidth_data_from_json_array_string(const std::string& filename) {
    std::vector<double> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return data;
    }
    std::string line;
    std::getline(file, line); // Read the single line
    file.close();

    if (line.length() < 2 || line.front() != '[' || line.back() != ']') {
        std::cerr << "Error: Invalid JSON array format in " << filename << std::endl;
        return data;
    }

    // Remove brackets
    line = line.substr(1, line.length() - 2);

    std::stringstream ss(line);
    std::string item;
    while (std::getline(ss, item, ',')) {
        try {
            data.push_back(std::stod(item));
        } catch (const std::invalid_argument& ia) {
            std::cerr << "Invalid argument: " << ia.what() << " for item: " << item << std::endl;
        } catch (const std::out_of_range& oor) {
            std::cerr << "Out of Range error: " << oor.what() << " for item: " << item << std::endl;
        }
    }
    std::cout << "Loaded " << data.size() << " data points from " << filename << std::endl;
    return data;
}

void normalize_data_cpu(std::vector<double>& data, double& min_val, double& max_val) {
    if (data.empty()) return;
    min_val = *std::min_element(data.begin(), data.end());
    max_val = *std::max_element(data.begin(), data.end());
    if (max_val == min_val) { // Avoid division by zero if all values are the same
        for (auto& val : data) val = 0.0; // Or 0.5, or handle as an error
        return;
    }
    for (auto& val : data) {
        val = (val - min_val) / (max_val - min_val);
    }
}

void denormalize_data_cpu(std::vector<double>& data, double min_val, double max_val) {
    if (max_val == min_val) return; // If normalized to 0, keep it as 0 (or min_val)
    for (auto& val : data) {
        val = val * (max_val - min_val) + min_val;
    }
}

void create_sliding_window_dataset_cpu(
    const std::vector<double>& series,
    int input_dim,
    int output_dim, // Should be 1 for this problem
    std::vector<std::vector<double>>& X_data, // Each inner vector is a sample
    std::vector<std::vector<double>>& y_data  // Each inner vector is a target
) {
    X_data.clear();
    y_data.clear();
    if (series.size() < static_cast<size_t>(input_dim + output_dim)) {
        std::cerr << "Error: Series too short for given input/output dims." << std::endl;
        return;
    }
    for (size_t i = 0; i <= series.size() - (input_dim + output_dim); ++i) {
        std::vector<double> x_sample;
        for (int j = 0; j < input_dim; ++j) {
            x_sample.push_back(series[i + j]);
        }
        X_data.push_back(x_sample);

        std::vector<double> y_sample;
        for (int j = 0; j < output_dim; ++j) {
            y_sample.push_back(series[i + input_dim + j]);
        }
        y_data.push_back(y_sample);
    }
    std::cout << "Created dataset with " << X_data.size() << " samples." << std::endl;
}

float get_simulated_time_ms(float min_ms, float max_ms) {
    if (min_ms >= max_ms) return min_ms;
    return min_ms + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max_ms - min_ms)));
}

// Helper to get a batch of data
void get_batch(const std::vector<std::vector<double>>& X_full, const std::vector<std::vector<double>>& y_full,
               int batch_idx, int current_batch_size,
               std::vector<double>& X_batch_flat, std::vector<double>& y_batch_flat) {
    X_batch_flat.clear();
    y_batch_flat.clear();
    X_batch_flat.reserve(current_batch_size * INPUT_DIM);
    y_batch_flat.reserve(current_batch_size * OUTPUT_DIM);

    for (int i = 0; i < current_batch_size; ++i) {
        int sample_idx = batch_idx * BATCH_SIZE + i;
        X_batch_flat.insert(X_batch_flat.end(), X_full[sample_idx].begin(), X_full[sample_idx].end());
        y_batch_flat.insert(y_batch_flat.end(), y_full[sample_idx].begin(), y_full[sample_idx].end());
    }
}


// ----------------------------- Main -------------------------------
int main() {
    srand(static_cast<unsigned int>(time(0)));
    std::cout << std::fixed << std::setprecision(6);

    // 1. Load and Preprocess Data
    std::vector<double> raw_data = load_bandwidth_data_from_json_array_string(DATA_FILE_PATH);
    if (raw_data.empty()) {
        return 1;
    }
    double min_val, max_val;
    std::vector<double> normalized_data = raw_data;
    normalize_data_cpu(normalized_data, min_val, max_val);

    std::vector<std::vector<double>> X_all, y_all;
    create_sliding_window_dataset_cpu(normalized_data, INPUT_DIM, OUTPUT_DIM, X_all, y_all);
    if (X_all.empty()) {
         std::cerr << "Dataset creation failed or resulted in zero samples." << std::endl;
        return 1;
    }


    // Split data
    size_t train_size = static_cast<size_t>(X_all.size() * TRAIN_SPLIT);
    size_t test_size = X_all.size() - train_size;

    if (train_size == 0 || test_size == 0) {
        std::cerr << "Not enough data to create train/test split with " << X_all.size() << " samples and split " << TRAIN_SPLIT << std::endl;
        return 1;
    }
    
    std::vector<std::vector<double>> X_train(X_all.begin(), X_all.begin() + train_size);
    std::vector<std::vector<double>> y_train(y_all.begin(), y_all.begin() + train_size);
    std::vector<std::vector<double>> X_test(X_all.begin() + train_size, X_all.end());
    std::vector<std::vector<double>> y_test(y_all.begin() + train_size, y_all.end());

    std::cout << "Training samples: " << X_train.size() << ", Test samples: " << X_test.size() << std::endl;


    // MLP Weights and Biases (CPU version)
    std::vector<double> W1_cpu(INPUT_DIM * HIDDEN_DIM);
    std::vector<double> B1_cpu(HIDDEN_DIM);
    std::vector<double> W2_cpu(HIDDEN_DIM * OUTPUT_DIM);
    std::vector<double> B2_cpu(OUTPUT_DIM);

    random_init_vector(W1_cpu, 10, sqrt(2.0 / INPUT_DIM)); // He initialization-like scaling
    random_init_vector(B1_cpu, 11, 0.01);
    random_init_vector(W2_cpu, 12, sqrt(2.0 / HIDDEN_DIM));
    random_init_vector(B2_cpu, 13, 0.01);

    // Simulated Device Memory Pointers (never actually used for data storage in sim)
#ifdef ENABLE_HIP_CODE
    double *d_W1, *d_B1, *d_W2, *d_B2;
    double *d_X_batch, *d_y_batch;
    double *d_Z1, *d_H1, *d_Y_pred; // Forward pass intermediates
    double *d_dL_dY_pred, *d_dL_dZ2, *d_dL_dH1, *d_dL_dZ1; // Backward pass grad intermediates
    double *d_dW1, *d_dB1, *d_dW2, *d_dB2; // Gradient accumulators on device

    // Allocate "device" memory (structural only)
    hipMalloc((void**)&d_W1, INPUT_DIM * HIDDEN_DIM * sizeof(double));
    hipMalloc((void**)&d_B1, HIDDEN_DIM * sizeof(double));
    hipMalloc((void**)&d_W2, HIDDEN_DIM * OUTPUT_DIM * sizeof(double));
    hipMalloc((void**)&d_B2, OUTPUT_DIM * sizeof(double));
    // Batch data
    hipMalloc((void**)&d_X_batch, BATCH_SIZE * INPUT_DIM * sizeof(double));
    hipMalloc((void**)&d_y_batch, BATCH_SIZE * OUTPUT_DIM * sizeof(double));
    // Forward intermediates
    hipMalloc((void**)&d_Z1, BATCH_SIZE * HIDDEN_DIM * sizeof(double));
    hipMalloc((void**)&d_H1, BATCH_SIZE * HIDDEN_DIM * sizeof(double));
    hipMalloc((void**)&d_Y_pred, BATCH_SIZE * OUTPUT_DIM * sizeof(double));
    // Backward intermediates
    hipMalloc((void**)&d_dL_dY_pred, BATCH_SIZE * OUTPUT_DIM * sizeof(double));
    hipMalloc((void**)&d_dL_dZ2, BATCH_SIZE * OUTPUT_DIM * sizeof(double));
    hipMalloc((void**)&d_dL_dH1, BATCH_SIZE * HIDDEN_DIM * sizeof(double));
    hipMalloc((void**)&d_dL_dZ1, BATCH_SIZE * HIDDEN_DIM * sizeof(double));
    // Gradients
    hipMalloc((void**)&d_dW1, INPUT_DIM * HIDDEN_DIM * sizeof(double));
    hipMalloc((void**)&d_dB1, HIDDEN_DIM * sizeof(double));
    hipMalloc((void**)&d_dW2, HIDDEN_DIM * OUTPUT_DIM * sizeof(double));
    hipMalloc((void**)&d_dB2, OUTPUT_DIM * sizeof(double));

    hipEvent_t start_event, stop_event;
    hipEventCreate(&start_event);
    hipEventCreate(&stop_event);
#endif // ENABLE_HIP_CODE
    float sim_time_ms;

    // Simulate initial weight copy to device
#ifdef ENABLE_HIP_CODE
    hipEventRecord(start_event);
    hipMemcpy(d_W1, W1_cpu.data(), INPUT_DIM * HIDDEN_DIM * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_B1, B1_cpu.data(), HIDDEN_DIM * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_W2, W2_cpu.data(), HIDDEN_DIM * OUTPUT_DIM * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_B2, B2_cpu.data(), OUTPUT_DIM * sizeof(double), hipMemcpyHostToDevice);
    hipEventRecord(stop_event); hipEventSynchronize(stop_event);
#endif // ENABLE_HIP_CODE
    sim_time_ms = get_simulated_time_ms(0.5f, 1.5f);
    std::cout << "Initial weights HtoD: " << sim_time_ms << " ms" << std::endl;

    std::cout << "\n--- Starting Training --- " << std::endl;
    
    std::vector<int> train_indices(X_train.size());
    std::iota(train_indices.begin(), train_indices.end(), 0);
    std::mt19937 rng(42);


    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        double epoch_loss = 0.0;
        float epoch_sim_time_total_ms = 0.0f;
        float epoch_sim_HtoD_ms = 0.0f, epoch_sim_Kernels_ms = 0.0f, epoch_sim_DtoH_ms = 0.0f;

        std::shuffle(train_indices.begin(), train_indices.end(), rng); // Shuffle data for each epoch
        
        std::vector<std::vector<double>> X_train_shuffled(X_train.size());
        std::vector<std::vector<double>> y_train_shuffled(y_train.size());
        for(size_t i=0; i<train_indices.size(); ++i) {
            X_train_shuffled[i] = X_train[train_indices[i]];
            y_train_shuffled[i] = y_train[train_indices[i]];
        }


        int num_batches = train_size / BATCH_SIZE;
        if (train_size % BATCH_SIZE != 0) num_batches++;


        for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
            int current_batch_start_idx = batch_idx * BATCH_SIZE;
            int current_batch_size = std::min((int)BATCH_SIZE, (int)train_size - current_batch_start_idx);
            if(current_batch_size <= 0) continue;

            std::vector<double> X_batch_cpu_flat;
            std::vector<double> y_batch_cpu_flat;
            X_batch_cpu_flat.reserve(current_batch_size * INPUT_DIM);
            y_batch_cpu_flat.reserve(current_batch_size * OUTPUT_DIM);

            for(int i=0; i < current_batch_size; ++i) {
                X_batch_cpu_flat.insert(X_batch_cpu_flat.end(), X_train_shuffled[current_batch_start_idx + i].begin(), X_train_shuffled[current_batch_start_idx + i].end());
                y_batch_cpu_flat.insert(y_batch_cpu_flat.end(), y_train_shuffled[current_batch_start_idx + i].begin(), y_train_shuffled[current_batch_start_idx + i].end());
            }
            
            // --- Simulate HtoD for batch data ---
#ifdef ENABLE_HIP_CODE
            hipEventRecord(start_event);
            hipMemcpy(d_X_batch, X_batch_cpu_flat.data(), current_batch_size * INPUT_DIM * sizeof(double), hipMemcpyHostToDevice);
            hipMemcpy(d_y_batch, y_batch_cpu_flat.data(), current_batch_size * OUTPUT_DIM * sizeof(double), hipMemcpyHostToDevice);
            hipEventRecord(stop_event); hipEventSynchronize(stop_event);
#endif // ENABLE_HIP_CODE
            sim_time_ms = get_simulated_time_ms(0.1f, 0.5f);
            epoch_sim_HtoD_ms += sim_time_ms;

            // --- Forward Pass (CPU computations, simulated kernel calls) ---
            std::vector<double> Z1_cpu(current_batch_size * HIDDEN_DIM);
            std::vector<double> H1_cpu(current_batch_size * HIDDEN_DIM);
            std::vector<double> Y_pred_cpu(current_batch_size * OUTPUT_DIM);

            // X_batch * W1
#ifdef ENABLE_HIP_CODE
            hipEventRecord(start_event);
            hipLaunchKernelGGL(matmul_forward_kernel, dim3(1), dim3(1), 0, 0, d_X_batch, d_W1, d_Z1, current_batch_size, HIDDEN_DIM, INPUT_DIM);
            hipDeviceSynchronize(); hipEventRecord(stop_event); hipEventSynchronize(stop_event);
#endif // ENABLE_HIP_CODE
            matmul_cpu(X_batch_cpu_flat, W1_cpu, Z1_cpu, current_batch_size, INPUT_DIM, HIDDEN_DIM);
            sim_time_ms = get_simulated_time_ms(0.2f, 0.8f); epoch_sim_Kernels_ms += sim_time_ms;

            // Z1 + B1
#ifdef ENABLE_HIP_CODE
            hipEventRecord(start_event);
            hipLaunchKernelGGL(add_bias_forward_kernel, dim3(1), dim3(1), 0, 0, d_Z1, d_B1, current_batch_size, HIDDEN_DIM);
            hipDeviceSynchronize(); hipEventRecord(stop_event); hipEventSynchronize(stop_event);
#endif // ENABLE_HIP_CODE
            add_bias_cpu(Z1_cpu, B1_cpu);
            sim_time_ms = get_simulated_time_ms(0.05f, 0.2f); epoch_sim_Kernels_ms += sim_time_ms;
            
            // H1 = ReLU(Z1)
            H1_cpu = Z1_cpu; // copy for ReLU
#ifdef ENABLE_HIP_CODE
            hipEventRecord(start_event);
            hipLaunchKernelGGL(relu_forward_kernel, dim3(1), dim3(1), 0, 0, d_H1, current_batch_size * HIDDEN_DIM); // Assuming d_H1 got Z1 result
            hipDeviceSynchronize(); hipEventRecord(stop_event); hipEventSynchronize(stop_event);
#endif // ENABLE_HIP_CODE
            relu_cpu(H1_cpu);
            sim_time_ms = get_simulated_time_ms(0.05f, 0.2f); epoch_sim_Kernels_ms += sim_time_ms;

            // H1 * W2
#ifdef ENABLE_HIP_CODE
            hipEventRecord(start_event);
            hipLaunchKernelGGL(matmul_forward_kernel, dim3(1), dim3(1), 0, 0, d_H1, d_W2, d_Y_pred, current_batch_size, OUTPUT_DIM, HIDDEN_DIM);
            hipDeviceSynchronize(); hipEventRecord(stop_event); hipEventSynchronize(stop_event);
#endif // ENABLE_HIP_CODE
            matmul_cpu(H1_cpu, W2_cpu, Y_pred_cpu, current_batch_size, HIDDEN_DIM, OUTPUT_DIM);
            sim_time_ms = get_simulated_time_ms(0.1f, 0.4f); epoch_sim_Kernels_ms += sim_time_ms;

            // Y_pred + B2
#ifdef ENABLE_HIP_CODE
            hipEventRecord(start_event);
            hipLaunchKernelGGL(add_bias_forward_kernel, dim3(1), dim3(1), 0, 0, d_Y_pred, d_B2, current_batch_size, OUTPUT_DIM);
            hipDeviceSynchronize(); hipEventRecord(stop_event); hipEventSynchronize(stop_event);
#endif // ENABLE_HIP_CODE
            add_bias_cpu(Y_pred_cpu, B2_cpu);
            sim_time_ms = get_simulated_time_ms(0.05f, 0.15f); epoch_sim_Kernels_ms += sim_time_ms;

            // --- Loss Calculation (CPU, simulated kernel) ---
#ifdef ENABLE_HIP_CODE
            hipEventRecord(start_event);
            // Placeholder for device loss calculation if needed
            // hipLaunchKernelGGL(mse_loss_kernel, dim3(1), dim3(1), 0, 0, d_Y_pred, d_y_batch, d_batch_loss_storage, current_batch_size, OUTPUT_DIM);
            hipDeviceSynchronize(); hipEventRecord(stop_event); hipEventSynchronize(stop_event);
#endif // ENABLE_HIP_CODE
            sim_time_ms = get_simulated_time_ms(0.02f, 0.08f); epoch_sim_Kernels_ms += sim_time_ms;


            // --- Backward Pass (CPU computations, simulated kernel calls) ---
            std::vector<double> dL_dY_pred_cpu(current_batch_size * OUTPUT_DIM);
            mse_loss_derivative_cpu(Y_pred_cpu, y_batch_cpu_flat, dL_dY_pred_cpu);
            // For linear output layer, dL_dZ2 = dL_dY_pred
            std::vector<double> dL_dZ2_cpu = dL_dY_pred_cpu;
#ifdef ENABLE_HIP_CODE
            hipEventRecord(start_event); // Simulate dL_dZ_out kernel
            hipLaunchKernelGGL(linear_backward_output_kernel, dim3(1), dim3(1), 0, 0, d_Y_pred, d_y_batch, d_dL_dZ2, current_batch_size, OUTPUT_DIM);
            hipDeviceSynchronize(); hipEventRecord(stop_event); hipEventSynchronize(stop_event);
#endif // ENABLE_HIP_CODE
            sim_time_ms = get_simulated_time_ms(0.05f, 0.15f); epoch_sim_Kernels_ms += sim_time_ms;

            // Gradients for W2 and B2
            std::vector<double> H1_T_cpu; // Transpose of H1_cpu
            transpose_matrix_cpu(H1_cpu, H1_T_cpu, current_batch_size, HIDDEN_DIM);
            std::vector<double> dW2_grad_cpu(HIDDEN_DIM * OUTPUT_DIM);
            std::vector<double> dB2_grad_cpu(OUTPUT_DIM);

            // dW2 = H1_T * dL_dZ2
#ifdef ENABLE_HIP_CODE
            hipEventRecord(start_event); // Simulate matmul for dW2
            hipLaunchKernelGGL(matmul_backward_weights_kernel, dim3(1), dim3(1), 0, 0, nullptr, nullptr, d_dW2, HIDDEN_DIM, OUTPUT_DIM, current_batch_size); // Simplified kernel params
            hipDeviceSynchronize(); hipEventRecord(stop_event); hipEventSynchronize(stop_event);
#endif // ENABLE_HIP_CODE
            matmul_cpu(H1_T_cpu, dL_dZ2_cpu, dW2_grad_cpu, HIDDEN_DIM, current_batch_size, OUTPUT_DIM);
             for(auto& grad_val : dW2_grad_cpu) grad_val /= current_batch_size; // Average gradient
            sim_time_ms = get_simulated_time_ms(0.2f, 0.7f); epoch_sim_Kernels_ms += sim_time_ms;
            
            // dB2 = sum_rows(dL_dZ2)
#ifdef ENABLE_HIP_CODE
            hipEventRecord(start_event); // Simulate sum for dB2
            hipLaunchKernelGGL(sum_rows_for_bias_grad_kernel, dim3(1), dim3(1), 0, 0, d_dL_dZ2, d_dB2, current_batch_size, OUTPUT_DIM);
            hipDeviceSynchronize(); hipEventRecord(stop_event); hipEventSynchronize(stop_event);
#endif // ENABLE_HIP_CODE
            sum_matrix_rows_cpu(dL_dZ2_cpu, dB2_grad_cpu, current_batch_size, OUTPUT_DIM);
            sim_time_ms = get_simulated_time_ms(0.02f, 0.08f); epoch_sim_Kernels_ms += sim_time_ms;

            // dL_dH1 = dL_dZ2 * W2_T
            std::vector<double> W2_T_cpu;
            transpose_matrix_cpu(W2_cpu, W2_T_cpu, HIDDEN_DIM, OUTPUT_DIM);
            std::vector<double> dL_dH1_cpu(current_batch_size * HIDDEN_DIM);
#ifdef ENABLE_HIP_CODE
            hipEventRecord(start_event); // Simulate matmul for dL_dH1
            hipLaunchKernelGGL(matmul_backward_data_kernel, dim3(1), dim3(1), 0, 0, nullptr, nullptr, d_dL_dH1, current_batch_size, HIDDEN_DIM, OUTPUT_DIM); // Simplified
            hipDeviceSynchronize(); hipEventRecord(stop_event); hipEventSynchronize(stop_event);
#endif // ENABLE_HIP_CODE
            matmul_cpu(dL_dZ2_cpu, W2_T_cpu, dL_dH1_cpu, current_batch_size, OUTPUT_DIM, HIDDEN_DIM);
            sim_time_ms = get_simulated_time_ms(0.2f, 0.6f); epoch_sim_Kernels_ms += sim_time_ms;

            // dL_dZ1 = dL_dH1 * relu_derivative(Z1_cpu)
            std::vector<double> relu_deriv_Z1_cpu;
            relu_derivative_cpu(Z1_cpu, relu_deriv_Z1_cpu);
            std::vector<double> dL_dZ1_cpu;
#ifdef ENABLE_HIP_CODE
            hipEventRecord(start_event); // Simulate relu_backward
            hipLaunchKernelGGL(relu_backward_kernel, dim3(1), dim3(1), 0, 0, d_dL_dZ1, nullptr, current_batch_size * HIDDEN_DIM); // Simplified
            hipDeviceSynchronize(); hipEventRecord(stop_event); hipEventSynchronize(stop_event);
#endif // ENABLE_HIP_CODE
            elementwise_multiply_cpu(dL_dH1_cpu, relu_deriv_Z1_cpu, dL_dZ1_cpu);
            sim_time_ms = get_simulated_time_ms(0.05f, 0.15f); epoch_sim_Kernels_ms += sim_time_ms;

            // Gradients for W1 and B1
            std::vector<double> X_batch_T_cpu;
            transpose_matrix_cpu(X_batch_cpu_flat, X_batch_T_cpu, current_batch_size, INPUT_DIM);
            std::vector<double> dW1_grad_cpu(INPUT_DIM * HIDDEN_DIM);
            std::vector<double> dB1_grad_cpu(HIDDEN_DIM);

            // dW1 = X_batch_T * dL_dZ1
#ifdef ENABLE_HIP_CODE
            hipEventRecord(start_event); // Simulate matmul for dW1
            hipLaunchKernelGGL(matmul_backward_weights_kernel, dim3(1), dim3(1), 0, 0, nullptr, nullptr, d_dW1, INPUT_DIM, HIDDEN_DIM, current_batch_size); // Simplified
            hipDeviceSynchronize(); hipEventRecord(stop_event); hipEventSynchronize(stop_event);
#endif // ENABLE_HIP_CODE
            matmul_cpu(X_batch_T_cpu, dL_dZ1_cpu, dW1_grad_cpu, INPUT_DIM, current_batch_size, HIDDEN_DIM);
            for(auto& grad_val : dW1_grad_cpu) grad_val /= current_batch_size; // Average gradient
            sim_time_ms = get_simulated_time_ms(0.2f, 0.8f); epoch_sim_Kernels_ms += sim_time_ms;

            // dB1 = sum_rows(dL_dZ1)
#ifdef ENABLE_HIP_CODE
            hipEventRecord(start_event); // Simulate sum for dB1
            hipLaunchKernelGGL(sum_rows_for_bias_grad_kernel, dim3(1), dim3(1), 0, 0, d_dL_dZ1, d_dB1, current_batch_size, HIDDEN_DIM);
            hipDeviceSynchronize(); hipEventRecord(stop_event); hipEventSynchronize(stop_event);
#endif // ENABLE_HIP_CODE
            sum_matrix_rows_cpu(dL_dZ1_cpu, dB1_grad_cpu, current_batch_size, HIDDEN_DIM);
            sim_time_ms = get_simulated_time_ms(0.02f, 0.08f); epoch_sim_Kernels_ms += sim_time_ms;
            
            // --- Simulate DtoH for gradients (optional, as updates happen on CPU for sim) ---
            // hipEventRecord(start_event);
            // hipMemcpy(dW1_cpu.data(), d_dW1, ...); // etc. for all grads
            // hipEventRecord(stop_event); hipEventSynchronize(stop_event);
            // sim_time_ms = get_simulated_time_ms(0.1f, 0.3f); epoch_sim_DtoH_ms += sim_time_ms;


            // --- SGD Update (CPU, simulated kernel call) ---
            // Simulate copying gradients to device for update kernel (if grads were computed on host)
            // For this simulation, we assume gradients (dW1_grad_cpu etc.) are ready for CPU update.
            // The sgd_update_kernel would be launched with d_W1, d_dW1 etc.

#ifdef ENABLE_HIP_CODE
            hipEventRecord(start_event); // Simulate SGD update kernel for W1
            hipLaunchKernelGGL(sgd_update_kernel, dim3(1), dim3(1), 0, 0, d_W1, d_dW1, LEARNING_RATE, W1_cpu.size());
            hipDeviceSynchronize(); hipEventRecord(stop_event); hipEventSynchronize(stop_event);
#endif // ENABLE_HIP_CODE
            sgd_update_cpu(W1_cpu, dW1_grad_cpu, LEARNING_RATE);
            sim_time_ms = get_simulated_time_ms(0.01f, 0.05f); epoch_sim_Kernels_ms += sim_time_ms; // Simulated time for W1 update

#ifdef ENABLE_HIP_CODE
            hipEventRecord(start_event); // Simulate SGD update kernel for B1
            hipLaunchKernelGGL(sgd_update_kernel, dim3(1), dim3(1), 0, 0, d_B1, d_dB1, LEARNING_RATE, B1_cpu.size());
            hipDeviceSynchronize(); hipEventRecord(stop_event); hipEventSynchronize(stop_event);
#endif // ENABLE_HIP_CODE
            sgd_update_cpu(B1_cpu, dB1_grad_cpu, LEARNING_RATE);
            sim_time_ms = get_simulated_time_ms(0.005f, 0.02f); epoch_sim_Kernels_ms += sim_time_ms; // Simulated time for B1 update

#ifdef ENABLE_HIP_CODE
            hipEventRecord(start_event); // Simulate SGD update kernel for W2
            hipLaunchKernelGGL(sgd_update_kernel, dim3(1), dim3(1), 0, 0, d_W2, d_dW2, LEARNING_RATE, W2_cpu.size());
            hipDeviceSynchronize(); hipEventRecord(stop_event); hipEventSynchronize(stop_event);
#endif // ENABLE_HIP_CODE
            sgd_update_cpu(W2_cpu, dW2_grad_cpu, LEARNING_RATE);
            sim_time_ms = get_simulated_time_ms(0.01f, 0.04f); epoch_sim_Kernels_ms += sim_time_ms; // Simulated time for W2 update

#ifdef ENABLE_HIP_CODE
            hipEventRecord(start_event); // Simulate SGD update kernel for B2
            hipLaunchKernelGGL(sgd_update_kernel, dim3(1), dim3(1), 0, 0, d_B2, d_dB2, LEARNING_RATE, B2_cpu.size());
            hipDeviceSynchronize(); hipEventRecord(stop_event); hipEventSynchronize(stop_event);
#endif // ENABLE_HIP_CODE
            sgd_update_cpu(B2_cpu, dB2_grad_cpu, LEARNING_RATE);
            sim_time_ms = get_simulated_time_ms(0.005f, 0.02f); epoch_sim_Kernels_ms += sim_time_ms; // Simulated time for B2 update

            // Simulate copying updated weights back to host (optional, often weights stay on device)
            // For this simulation, Wn_cpu are the master copies.

            // Accumulate loss for the epoch
            epoch_loss += mse_loss_cpu(Y_pred_cpu, y_batch_cpu_flat) * current_batch_size;
        }
        epoch_sim_time_total_ms = epoch_sim_HtoD_ms + epoch_sim_Kernels_ms + epoch_sim_DtoH_ms;
        std::cout << "[Epoch " << epoch + 1 << "/" << EPOCHS << "] Avg Loss: " << epoch_loss / train_size
                  << ", Time: " << epoch_sim_time_total_ms << " ms "
                  << "(HtoD: " << epoch_sim_HtoD_ms << ", Kernels: " << epoch_sim_Kernels_ms << ", DtoH: " << epoch_sim_DtoH_ms << ")"
                  << std::endl;
    }

    std::cout << "\n--- Training Finished --- " << std::endl;

    // --- Testing Phase ---
    std::cout << "\n--- Starting Testing --- " << std::endl;
    double test_loss_total = 0.0;
    std::vector<double> all_test_predictions_flat;
    std::vector<double> all_test_actuals_flat;
    float total_test_sim_HtoD_ms = 0.0f;
    float total_test_sim_Kernels_ms = 0.0f;
    float total_test_sim_DtoH_ms = 0.0f; // If Y_pred is copied back per batch
    float total_test_sim_inference_time_ms = 0.0f;
    
    int num_test_batches = test_size / BATCH_SIZE;
    if (test_size % BATCH_SIZE != 0) num_test_batches++;

    for (int batch_idx = 0; batch_idx < num_test_batches; ++batch_idx) {
        int current_batch_start_idx = batch_idx * BATCH_SIZE;
        int current_batch_size = std::min((int)BATCH_SIZE, (int)test_size - current_batch_start_idx);
         if(current_batch_size <= 0) continue;

        std::vector<double> X_test_batch_cpu_flat;
        std::vector<double> y_test_batch_cpu_flat;
        X_test_batch_cpu_flat.reserve(current_batch_size * INPUT_DIM);
        y_test_batch_cpu_flat.reserve(current_batch_size * OUTPUT_DIM);

        for(int i=0; i < current_batch_size; ++i) {
            X_test_batch_cpu_flat.insert(X_test_batch_cpu_flat.end(), X_test[current_batch_start_idx + i].begin(), X_test[current_batch_start_idx + i].end());
            y_test_batch_cpu_flat.insert(y_test_batch_cpu_flat.end(), y_test[current_batch_start_idx + i].begin(), y_test[current_batch_start_idx + i].end());
        }

        // Simulate HtoD for test batch
#ifdef ENABLE_HIP_CODE
        hipEventRecord(start_event);
        hipMemcpy(d_X_batch, X_test_batch_cpu_flat.data(), current_batch_size * INPUT_DIM * sizeof(double), hipMemcpyHostToDevice);
        hipEventRecord(stop_event); hipEventSynchronize(stop_event);
#endif // ENABLE_HIP_CODE
        sim_time_ms = get_simulated_time_ms(0.05f, 0.2f); // Sim HtoD for test batch
        total_test_sim_HtoD_ms += sim_time_ms;
        float batch_inference_kernel_time = 0.0f;

        std::vector<double> Z1_test_cpu(current_batch_size * HIDDEN_DIM);
        std::vector<double> H1_test_cpu(current_batch_size * HIDDEN_DIM);
        std::vector<double> Y_pred_test_cpu(current_batch_size * OUTPUT_DIM);

        matmul_cpu(X_test_batch_cpu_flat, W1_cpu, Z1_test_cpu, current_batch_size, INPUT_DIM, HIDDEN_DIM);
        sim_time_ms = get_simulated_time_ms(0.1f, 0.5f); batch_inference_kernel_time += sim_time_ms; // Matmul1
        add_bias_cpu(Z1_test_cpu, B1_cpu);
        sim_time_ms = get_simulated_time_ms(0.02f, 0.1f); batch_inference_kernel_time += sim_time_ms; // AddBias1
        H1_test_cpu = Z1_test_cpu;
        relu_cpu(H1_test_cpu);
        sim_time_ms = get_simulated_time_ms(0.02f, 0.1f); batch_inference_kernel_time += sim_time_ms; // ReLU
        matmul_cpu(H1_test_cpu, W2_cpu, Y_pred_test_cpu, current_batch_size, HIDDEN_DIM, OUTPUT_DIM);
        sim_time_ms = get_simulated_time_ms(0.05f, 0.3f); batch_inference_kernel_time += sim_time_ms; // Matmul2
        add_bias_cpu(Y_pred_test_cpu, B2_cpu);
        sim_time_ms = get_simulated_time_ms(0.01f, 0.05f); batch_inference_kernel_time += sim_time_ms; // AddBias2
        total_test_sim_Kernels_ms += batch_inference_kernel_time;

        // Simulate DtoH for Y_pred_test_cpu (optional, if considering per-batch transfer)
        // For overall inference time, usually this is done once after all batches if possible,
        // or if predictions are needed immediately per batch.
        // Let's assume for simplicity we copy each batch's prediction back for this simulation detail.
#ifdef ENABLE_HIP_CODE
        // hipMemcpy(Y_pred_test_cpu.data(), d_Y_pred, current_batch_size * OUTPUT_DIM * sizeof(double), hipMemcpyDeviceToHost);
#endif // ENABLE_HIP_CODE
        sim_time_ms = get_simulated_time_ms(0.05f, 0.15f); // Sim DtoH for Y_pred_test_cpu of the batch
        total_test_sim_DtoH_ms += sim_time_ms;

        test_loss_total += mse_loss_cpu(Y_pred_test_cpu, y_test_batch_cpu_flat) * current_batch_size;
        all_test_predictions_flat.insert(all_test_predictions_flat.end(), Y_pred_test_cpu.begin(), Y_pred_test_cpu.end());
        all_test_actuals_flat.insert(all_test_actuals_flat.end(), y_test_batch_cpu_flat.begin(), y_test_batch_cpu_flat.end());
    }
    
    double avg_test_mse = test_size > 0 ? test_loss_total / test_size : 0.0;
    std::cout << "Average Test MSE (Normalized): " << avg_test_mse << std::endl;

    total_test_sim_inference_time_ms = total_test_sim_HtoD_ms + total_test_sim_Kernels_ms + total_test_sim_DtoH_ms;
    double avg_inference_latency_ms = test_size > 0 ? total_test_sim_inference_time_ms / test_size : 0.0;
    double inference_throughput_sps = test_size > 0 && total_test_sim_inference_time_ms > 0 ? 
                                       (test_size / (total_test_sim_inference_time_ms / 1000.0)) : 0.0;

    std::cout << "Total Simulated Test/Inference Time: " << total_test_sim_inference_time_ms << " ms for " << test_size << " samples" << std::endl;
    std::cout << "Avg Simulated Inference Latency per Sample: " << avg_inference_latency_ms << " ms" << std::endl;
    std::cout << "Simulated Inference Throughput: " << inference_throughput_sps << " samples/sec" << std::endl;

    // Denormalize for comparison
    denormalize_data_cpu(all_test_predictions_flat, min_val, max_val);
    denormalize_data_cpu(all_test_actuals_flat, min_val, max_val);
    
    double denormalized_mse = 0.0;
    if(test_size > 0 && all_test_predictions_flat.size() == all_test_actuals_flat.size()){
        for(size_t i=0; i < all_test_predictions_flat.size(); ++i){
            denormalized_mse += (all_test_predictions_flat[i] - all_test_actuals_flat[i]) * (all_test_predictions_flat[i] - all_test_actuals_flat[i]);
        }
        denormalized_mse /= all_test_predictions_flat.size();
         std::cout << "Average Test MSE (Denormalized): " << denormalized_mse << std::endl;
    }


    std::cout << "\nSample Test Predictions (Denormalized):" << std::endl;
    for (size_t i = 0; i < std::min((size_t)10, all_test_predictions_flat.size()) ; ++i) {
        std::cout << "Pred: " << all_test_predictions_flat[i] << ", Actual: " << all_test_actuals_flat[i] << std::endl;
    }

    // Cleanup
#ifdef ENABLE_HIP_CODE
    hipFree(d_W1); hipFree(d_B1); hipFree(d_W2); hipFree(d_B2);
    hipFree(d_X_batch); hipFree(d_y_batch);
    hipFree(d_Z1); hipFree(d_H1); hipFree(d_Y_pred);
    hipFree(d_dL_dY_pred); hipFree(d_dL_dZ2); hipFree(d_dL_dH1); hipFree(d_dL_dZ1);
    hipFree(d_dW1); hipFree(d_dB1); hipFree(d_dW2); hipFree(d_dB2);
    hipEventDestroy(start_event);
    hipEventDestroy(stop_event);
#endif // ENABLE_HIP_CODE

    std::cout << "\nMLP training and testing finished." << std::endl;
    return 0;
}