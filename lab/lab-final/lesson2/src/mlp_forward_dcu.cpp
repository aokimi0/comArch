#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib> // For rand(), srand()
#include <cmath>   // For fmax, abs
#include <random>  // For std::mt19937, std::uniform_real_distribution
#include <ctime>   // For time()
#include <iomanip> // For std::fixed, std::setprecision
#include <numeric> // For std::accumulate (optional)
#include <algorithm> // For std::min
#include <chrono> // For timing CPU part

// 编译文件
// hipcc mlp_forward_dcu.cpp -o mlp_forward_dcu
// 执行文件
// ./mlp_forward_dcu

#define BATCH 1024
#define I 10 // Input features
#define H 20 // Hidden layer neurons
#define O 5  // Output layer neurons

// HIP_CHECK macro for error handling
#define HIP_CHECK(command) \
    do { \
        hipError_t error = command; \
        if (error != hipSuccess) { \
            std::cerr << "HIP Error: " << hipGetErrorString(error) \
                      << " at line " << __LINE__ \
                      << " in file " << __FILE__ << std::endl; \
            exit(error); \
        } \
    } while (0)

// HIP Kernels (remains the same structurally)
__global__ void matmul_kernel(const double* A, const double* B, double* C, int M_rowsA, int N_colsB, int K_commonDim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M_rowsA && col < N_colsB) {
        double sum = 0.0;
        for (int k = 0; k < K_commonDim; ++k) {
            sum += A[row * K_commonDim + k] * B[k * N_colsB + col];
        }
        C[row * N_colsB + col] = sum;
    }
}

__global__ void add_bias_kernel(double* matrix_M, const double* bias_B, int rows_M, int cols_M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows_M && col < cols_M) {
        matrix_M[row * cols_M + col] += bias_B[col];
    }
}

__global__ void relu_kernel(double* A, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        A[idx] = fmax(0.0, A[idx]);
    }
}

// CPU Helper Functions
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

void add_bias_cpu(std::vector<double>& matrix_M, const std::vector<double>& bias_B, int rows_M, int cols_M) {
    for (int i = 0; i < rows_M; ++i) {
        for (int j = 0; j < cols_M; ++j) {
            matrix_M[i * cols_M + j] += bias_B[j];
        }
    }
}

void relu_cpu(const std::vector<double>& A, std::vector<double>& Result, int size) {
    Result.resize(size);
    for (int i = 0; i < size; ++i) {
        Result[i] = std::fmax(0.0, A[i]);
    }
}

// Function to print a small part of a vector for debugging
void print_vector_sample(const std::string& name, const std::vector<double>& vec, int rows, int cols, int sample_size = 5) {
    std::cout << name << " sample (first " << sample_size << " elements of first " << std::min(rows, sample_size) << " rows):" << std::endl;
    for (int i = 0; i < std::min(rows, sample_size); ++i) {
        for (int j = 0; j < std::min(cols, sample_size); ++j) {
            std::cout << std::fixed << std::setprecision(3) << vec[i * cols + j] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void random_init_vector(std::vector<double>& vec, unsigned int seed_offset = 0) {
    std::mt19937 gen(42 + seed_offset); // Consistent seed + offset for different vectors
    std::uniform_real_distribution<double> dist(-1.0, 1.0); // Smaller range for weights/biases often better
    for (auto& val : vec) {
        val = dist(gen);
    }
}

bool validate_vectors(const std::vector<double>& ref, const std::vector<double>& test, double tol = 1e-6) {
    if (ref.size() != test.size()) {
        std::cerr << "Validation failed: Size mismatch. Ref size: " << ref.size() << ", Test size: " << test.size() << std::endl;
        return false;
    }
    for (size_t i = 0; i < ref.size(); ++i) {
        if (std::abs(ref[i] - test[i]) > tol) {
            std::cerr << "Validation failed at index " << i << ". Ref: " << ref[i] << ", Test: " << test[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    // srand(static_cast<unsigned int>(time(0))); // Seed for rand() - No longer needed for simulation
    std::cout << std::fixed << std::setprecision(3); // Output precision for timings

    std::vector<double> h_X(BATCH * I), h_W1(I * H), h_B1(H), h_W2(H * O), h_B2(O);
    std::vector<double> h_Y_dcu(BATCH * O); // Renamed from h_Y_simulated_dcu
    std::vector<double> h_Y_cpu_ref(BATCH * O);   // To store CPU reference result

    // Initialize host data with consistent seeds
    random_init_vector(h_X, 0);
    random_init_vector(h_W1, 1);
    random_init_vector(h_B1, 2);
    random_init_vector(h_W2, 3);
    random_init_vector(h_B2, 4);

    // --- CPU Reference Computation ---
    std::cout << "Performing CPU reference computation..." << std::endl;
    std::vector<double> Z1_cpu(BATCH * H), H1_cpu(BATCH * H);
    std::vector<double> Z2_cpu(BATCH * O);

    auto cpu_start_time = std::chrono::high_resolution_clock::now();

    matmul_cpu(h_X, h_W1, Z1_cpu, BATCH, I, H);
    add_bias_cpu(Z1_cpu, h_B1, BATCH, H);
    relu_cpu(Z1_cpu, H1_cpu, BATCH * H);

    matmul_cpu(H1_cpu, h_W2, Z2_cpu, BATCH, H, O);
    add_bias_cpu(Z2_cpu, h_B2, BATCH, O);
    h_Y_cpu_ref = Z2_cpu; // Final CPU reference output

    auto cpu_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration_ms = cpu_end_time - cpu_start_time;
    std::cout << "CPU reference computation finished. Time: " << cpu_duration_ms.count() << " ms" << std::endl << std::endl;

    // --- Actual DCU Path ---
    std::cout << "--- MLP Forward Pass DCU --- " << std::endl;

    double *d_X, *d_W1, *d_B1, *d_H, *d_W2, *d_B2, *d_Y;
    hipEvent_t start_event, stop_event;
    // hipEventCreate(&start_event); // Create events once
    // hipEventCreate(&stop_event);
    HIP_CHECK(hipEventCreate(&start_event));
    HIP_CHECK(hipEventCreate(&stop_event));

    float memcpy_htod_time = 0.0f, memcpy_dtoh_time = 0.0f;
    float matmul1_time = 0.0f, add_bias1_time = 0.0f, relu_time = 0.0f;
    float matmul2_time = 0.0f, add_bias2_time = 0.0f;
    float total_kernel_time = 0.0f; // Renamed
    float total_mlp_time = 0.0f;    // Renamed

    // Allocate device memory
    HIP_CHECK(hipMalloc((void**)&d_X,  static_cast<size_t>(BATCH) * I * sizeof(double)));
    HIP_CHECK(hipMalloc((void**)&d_W1, static_cast<size_t>(I) * H * sizeof(double)));
    HIP_CHECK(hipMalloc((void**)&d_B1, static_cast<size_t>(H) * sizeof(double)));
    HIP_CHECK(hipMalloc((void**)&d_H,  static_cast<size_t>(BATCH) * H * sizeof(double))); // Intermediate result for H1
    HIP_CHECK(hipMalloc((void**)&d_W2, static_cast<size_t>(H) * O * sizeof(double)));
    HIP_CHECK(hipMalloc((void**)&d_B2, static_cast<size_t>(O) * sizeof(double)));
    HIP_CHECK(hipMalloc((void**)&d_Y,  static_cast<size_t>(BATCH) * O * sizeof(double)));

    // Copy host data to device
    HIP_CHECK(hipEventRecord(start_event, 0));
    HIP_CHECK(hipMemcpy(d_X,  h_X.data(),  static_cast<size_t>(BATCH) * I * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_W1, h_W1.data(), static_cast<size_t>(I) * H * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B1, h_B1.data(), static_cast<size_t>(H) * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_W2, h_W2.data(), static_cast<size_t>(H) * O * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B2, h_B2.data(), static_cast<size_t>(O) * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipEventRecord(stop_event, 0));
    HIP_CHECK(hipEventSynchronize(stop_event));
    HIP_CHECK(hipEventElapsedTime(&memcpy_htod_time, start_event, stop_event));
    std::cout << "Time for HtoD copy: " << memcpy_htod_time << " ms" << std::endl;

    // Define thread block and grid dimensions
    dim3 threadsPerBlock(16, 16); // For matmul and add_bias
    dim3 threadsPerBlock_relu(256); // For relu (1D kernel)

    // Layer 1: H1 = ReLU(X * W1 + B1)
    // 1. X * W1 -> d_H
    dim3 numBlocks_matmul1((H + threadsPerBlock.x - 1) / threadsPerBlock.x, (BATCH + threadsPerBlock.y - 1) / threadsPerBlock.y);
    HIP_CHECK(hipEventRecord(start_event, 0));
    hipLaunchKernelGGL(matmul_kernel, numBlocks_matmul1, threadsPerBlock, 0, 0, d_X, d_W1, d_H, BATCH, H, I);
    HIP_CHECK(hipDeviceSynchronize()); // Ensure kernel completion before stopping timer
    HIP_CHECK(hipEventRecord(stop_event, 0));
    HIP_CHECK(hipEventSynchronize(stop_event));
    HIP_CHECK(hipEventElapsedTime(&matmul1_time, start_event, stop_event));
    std::cout << "Time for MatMul1 Kernel: " << matmul1_time << " ms" << std::endl;
    total_kernel_time += matmul1_time;

    // 2. d_H + B1 -> d_H (in-place)
    // Grid dim for add_bias should match the matrix it's operating on (d_H: BATCH x H)
    dim3 numBlocks_add_bias1((H + threadsPerBlock.x -1) / threadsPerBlock.x, (BATCH + threadsPerBlock.y -1) / threadsPerBlock.y);
    HIP_CHECK(hipEventRecord(start_event, 0));
    hipLaunchKernelGGL(add_bias_kernel, numBlocks_add_bias1, threadsPerBlock, 0, 0, d_H, d_B1, BATCH, H);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipEventRecord(stop_event, 0));
    HIP_CHECK(hipEventSynchronize(stop_event));
    HIP_CHECK(hipEventElapsedTime(&add_bias1_time, start_event, stop_event));
    std::cout << "Time for AddBias1 Kernel: " << add_bias1_time << " ms" << std::endl;
    total_kernel_time += add_bias1_time;

    // 3. ReLU(d_H) -> d_H (in-place)
    // Grid dim for relu should match the total elements in d_H
    dim3 numBlocks_relu1((BATCH * H + threadsPerBlock_relu.x - 1) / threadsPerBlock_relu.x);
    HIP_CHECK(hipEventRecord(start_event, 0));
    hipLaunchKernelGGL(relu_kernel, numBlocks_relu1, threadsPerBlock_relu, 0, 0, d_H, BATCH * H);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipEventRecord(stop_event, 0));
    HIP_CHECK(hipEventSynchronize(stop_event));
    HIP_CHECK(hipEventElapsedTime(&relu_time, start_event, stop_event));
    std::cout << "Time for ReLU Kernel: " << relu_time << " ms" << std::endl;
    total_kernel_time += relu_time;

    // Layer 2: Y = H1 * W2 + B2
    // 1. d_H * W2 -> d_Y
    dim3 numBlocks_matmul2((O + threadsPerBlock.x - 1) / threadsPerBlock.x, (BATCH + threadsPerBlock.y - 1) / threadsPerBlock.y);
    HIP_CHECK(hipEventRecord(start_event, 0));
    hipLaunchKernelGGL(matmul_kernel, numBlocks_matmul2, threadsPerBlock, 0, 0, d_H, d_W2, d_Y, BATCH, O, H);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipEventRecord(stop_event, 0));
    HIP_CHECK(hipEventSynchronize(stop_event));
    HIP_CHECK(hipEventElapsedTime(&matmul2_time, start_event, stop_event));
    std::cout << "Time for MatMul2 Kernel: " << matmul2_time << " ms" << std::endl;
    total_kernel_time += matmul2_time;

    // 2. d_Y + B2 -> d_Y (in-place)
    // Grid dim for add_bias should match the matrix it's operating on (d_Y: BATCH x O)
    dim3 numBlocks_add_bias2((O + threadsPerBlock.x -1) / threadsPerBlock.x, (BATCH + threadsPerBlock.y -1) / threadsPerBlock.y);
    HIP_CHECK(hipEventRecord(start_event, 0));
    hipLaunchKernelGGL(add_bias_kernel, numBlocks_add_bias2, threadsPerBlock, 0, 0, d_Y, d_B2, BATCH, O);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipEventRecord(stop_event, 0));
    HIP_CHECK(hipEventSynchronize(stop_event));
    HIP_CHECK(hipEventElapsedTime(&add_bias2_time, start_event, stop_event));
    std::cout << "Time for AddBias2 Kernel: " << add_bias2_time << " ms" << std::endl;
    total_kernel_time += add_bias2_time;

    // Copy result Y back to host
    HIP_CHECK(hipEventRecord(start_event, 0));
    HIP_CHECK(hipMemcpy(h_Y_dcu.data(), d_Y, static_cast<size_t>(BATCH) * O * sizeof(double), hipMemcpyDeviceToHost));
    HIP_CHECK(hipEventRecord(stop_event, 0)); // Record event after data is copied
    HIP_CHECK(hipEventSynchronize(stop_event)); // Wait for copy to complete
    HIP_CHECK(hipEventElapsedTime(&memcpy_dtoh_time, start_event, stop_event));
    std::cout << "Time for DtoH copy: " << memcpy_dtoh_time << " ms" << std::endl;

    // REMOVED: h_Y_simulated_dcu = h_Y_cpu_ref;

    std::cout << "Total Kernel Execution Time: " << total_kernel_time << " ms" << std::endl;
    total_mlp_time = memcpy_htod_time + total_kernel_time + memcpy_dtoh_time;
    std::cout << "Total MLP Forward Time (HtoD + Kernels + DtoH): " << total_mlp_time << " ms" << std::endl;
    
    std::cout << std::endl << "Validating DCU result against CPU reference..." << std::endl;
    // print_vector_sample("CPU Y_ref", h_Y_cpu_ref, BATCH, O);
    // print_vector_sample("DCU Y", h_Y_dcu, BATCH, O);
    if (validate_vectors(h_Y_cpu_ref, h_Y_dcu)) { // Use h_Y_dcu for validation
        std::cout << "Validation PASSED!" << std::endl;
    } else {
        std::cout << "VALIDATION FAILED!" << std::endl;
    }

    // Free device memory
    HIP_CHECK(hipFree(d_X));
    HIP_CHECK(hipFree(d_W1));
    HIP_CHECK(hipFree(d_B1));
    HIP_CHECK(hipFree(d_H));
    HIP_CHECK(hipFree(d_W2));
    HIP_CHECK(hipFree(d_B2));
    HIP_CHECK(hipFree(d_Y));

    HIP_CHECK(hipEventDestroy(start_event));
    HIP_CHECK(hipEventDestroy(stop_event));

    return 0;
}