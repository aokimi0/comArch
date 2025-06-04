#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <cmath> // For fabs
#include <stdexcept> // For std::runtime_error

#include <hip/hip_runtime.h>

// Helper macro for HIP API error checking
#define HIP_CHECK(command) \
    do { \
        hipError_t error = command; \
        if (error != hipSuccess) { \
            std::cerr << "HIP Error: " << hipGetErrorString(error) \
                      << " at line " << __LINE__ \
                      << " in file " << __FILE__ << std::endl; \
            /*exit(error);*/ \
            throw std::runtime_error(std::string("HIP Error: ") + hipGetErrorString(error)); \
        } \
    } while (0)


// Matrix dimensions (can be overridden by command line args)
int N_DEFAULT_DCU = 1024;
int M_DEFAULT_DCU = 2048;
int P_DEFAULT_DCU = 512;

// --- Helper Functions (Host-side) ---
void initialize_matrix_host(std::vector<double>& matrix, int rows, int cols, bool random_fill = true, double val = 0.0) {
    matrix.assign(static_cast<size_t>(rows) * cols, val);
    if (random_fill) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> distrib(-2.0, 2.0);
        for (size_t i = 0; i < matrix.size(); ++i) {
            matrix[i] = distrib(gen);
        }
    }
}

void print_matrix_host(const std::string& name, const std::vector<double>& matrix, int rows, int cols) {
    std::cout << name << " (" << rows << "x" << cols << "):\n";
    if (matrix.empty() || rows == 0 || cols == 0) {
        std::cout << "  [empty or zero-dimension]\n";
        return;
    }
    const int max_print_rows = 8;
    const int max_print_cols = 8;
    for (int i = 0; i < std::min(rows, max_print_rows); ++i) {
        for (int j = 0; j < std::min(cols, max_print_cols); ++j) {
            std::cout << std::fixed << std::setprecision(2) << std::setw(8) << matrix[static_cast<size_t>(i) * cols + j] << " ";
        }
        if (cols > max_print_cols) std::cout << "...";
        std::cout << "\n";
    }
    if (rows > max_print_rows) std::cout << "...\n";
    std::cout << "\n";
}

bool verify_matrices_host(const std::vector<double>& mat1, const std::vector<double>& mat2, int rows, int cols, double tolerance = 1e-6) {
    if (mat1.size() != static_cast<size_t>(rows * cols) || mat2.size() != static_cast<size_t>(rows * cols)) {
        std::cerr << "Verification failed: Matrix dimension mismatch. mat1: " << mat1.size() << ", mat2: " << mat2.size() << ", expected: " << static_cast<size_t>(rows)*cols << "\n";
        return false;
    }
    double max_diff = 0.0;
    int errors = 0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            size_t index = static_cast<size_t>(i) * cols + j;
            double diff = std::fabs(mat1[index] - mat2[index]);
            if (diff > max_diff) max_diff = diff;
            if (diff > tolerance) {
                if(errors < 5) { // Print first few errors
                     std::cerr << "Verification failed: Difference at (" << i << "," << j << ") is " << diff
                               << " (val1: " << mat1[index] << ", val2: " << mat2[index] << ")\n";
                }
                errors++;
            }
        }
    }
    if (errors > 0) {
        std::cout << "Matrices differ. Total errors: " << errors << ". Max difference: " << std::fixed << std::setprecision(10) << max_diff << " (tolerance: " << tolerance << ").\n";
        return false;
    }
    std::cout << "Matrices are identical (max difference: " << std::fixed << std::setprecision(10) << max_diff << ", tolerance: " << tolerance << ").\n";
    return true;
}

// CPU Baseline for verification (can be simple)
void matrix_multiply_baseline_host(
    const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C,
    int N, int M, int P) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < P; ++j) {
            double sum = 0.0;
            for (int k = 0; k < M; ++k) {
                sum += A[static_cast<size_t>(i) * M + k] * B[static_cast<size_t>(k) * P + j];
            }
            C[static_cast<size_t>(i) * P + j] = sum;
        }
    }
}

// --- HIP Kernel ---
__global__ void matmul_kernel_dcu(const double* A, const double* B, double* C, int N, int M, int P) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < P) {
        double sum = 0.0;
        for (int k = 0; k < M; ++k) {
            sum += A[static_cast<size_t>(row) * M + k] * B[static_cast<size_t>(k) * P + col];
        }
        C[static_cast<size_t>(row) * P + col] = sum;
    }
}

// --- Main DCU Logic ---
void run_dcu_matmul(int N, int M, int P) {
    std::cout << "\n--- DCU Matrix Multiplication ---" << std::endl;
    std::cout << "Matrix dimensions: N=" << N << ", M=" << M << ", P=" << P << std::endl;

    std::vector<double> h_A(static_cast<size_t>(N) * M);
    std::vector<double> h_B(static_cast<size_t>(M) * P);
    std::vector<double> h_C_dcu(static_cast<size_t>(N) * P);
    std::vector<double> h_C_baseline(static_cast<size_t>(N) * P);

    initialize_matrix_host(h_A, N, M, true);
    initialize_matrix_host(h_B, M, P, true);
    initialize_matrix_host(h_C_dcu, N, P, false, 0.0); // Initialize with zeros

    // print_matrix_host("Host A (DCU)", h_A, N, M);
    // print_matrix_host("Host B (DCU)", h_B, M, P);

    std::cout << "Computing baseline on host for DCU verification..." << std::endl;
    auto baseline_start_time = std::chrono::high_resolution_clock::now();
    matrix_multiply_baseline_host(h_A, h_B, h_C_baseline, N, M, P);
    auto baseline_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> baseline_elapsed_ms = baseline_end_time - baseline_start_time;
    std::cout << "Host baseline computed in " << baseline_elapsed_ms.count() << " ms." << std::endl;

    double *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    hipEvent_t start_event = nullptr, stop_event = nullptr;

    try {
        size_t size_A = static_cast<size_t>(N) * M * sizeof(double);
        size_t size_B = static_cast<size_t>(M) * P * sizeof(double);
        size_t size_C = static_cast<size_t>(N) * P * sizeof(double);

        HIP_CHECK(hipMalloc((void**)&d_A, size_A));
        HIP_CHECK(hipMalloc((void**)&d_B, size_B));
        HIP_CHECK(hipMalloc((void**)&d_C, size_C));

        std::cout << "Copying data from Host to Device..." << std::endl;
        HIP_CHECK(hipMemcpy(d_A, h_A.data(), size_A, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_B, h_B.data(), size_B, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_C, h_C_dcu.data(), size_C, hipMemcpyHostToDevice)); 

        const int THREADS_PER_BLOCK_X = 16;
        const int THREADS_PER_BLOCK_Y = 16;
        dim3 threadsPerBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
        dim3 numBlocks((P + THREADS_PER_BLOCK_X - 1) / THREADS_PER_BLOCK_X,
                       (N + THREADS_PER_BLOCK_Y - 1) / THREADS_PER_BLOCK_Y);

        std::cout << "Launching HIP kernel... (Blocks: " << numBlocks.x << "x" << numBlocks.y 
                  << ", Threads/Block: " << threadsPerBlock.x << "x" << threadsPerBlock.y << ")" << std::endl;
        
        HIP_CHECK(hipEventCreate(&start_event));
        HIP_CHECK(hipEventCreate(&stop_event));

        HIP_CHECK(hipEventRecord(start_event, 0));
        hipLaunchKernelGGL(matmul_kernel_dcu, numBlocks, threadsPerBlock, 0, 0, d_A, d_B, d_C, N, M, P);
        HIP_CHECK(hipGetLastError()); 
        HIP_CHECK(hipEventRecord(stop_event, 0));
        HIP_CHECK(hipEventSynchronize(stop_event));

        float milliseconds = 0;
        HIP_CHECK(hipEventElapsedTime(&milliseconds, start_event, stop_event));
        std::cout << "DCU Matmul Time (Kernel Execution): " << std::fixed << std::setprecision(4) << milliseconds << " ms" << std::endl;

        std::cout << "Copying results from Device to Host..." << std::endl;
        HIP_CHECK(hipMemcpy(h_C_dcu.data(), d_C, size_C, hipMemcpyDeviceToHost));

        std::cout << "Verifying DCU result against host baseline..." << std::endl;
        if (!verify_matrices_host(h_C_dcu, h_C_baseline, N, P)) {
            std::cout << "Warning: DCU VERIFICATION FAILED against host baseline.\n";
        } else {
            std::cout << "DCU VERIFICATION PASSED.\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Exception during DCU operations: " << e.what() << std::endl;
    }

    // Cleanup
    if (start_event) hipEventDestroy(start_event);
    if (stop_event) hipEventDestroy(stop_event);
    if (d_A) hipFree(d_A);
    if (d_B) hipFree(d_B);
    if (d_C) hipFree(d_C);

    std::cout << "--- DCU Matrix Multiplication Finished ---" << std::endl;
}

int main(int argc, char* argv[]) {
    int N = N_DEFAULT_DCU;
    int M = M_DEFAULT_DCU;
    int P = P_DEFAULT_DCU;

    if (argc > 1) N = std::stoi(argv[1]);
    if (argc > 2) M = std::stoi(argv[2]);
    if (argc > 3) P = std::stoi(argv[3]);

    if (N <=0 || M <= 0 || P <= 0) { 
        std::cerr << "Error: Matrix dimensions N, M, P must be positive. Got N="<< N <<", M="<< M <<", P="<<P<< std::endl;
        return 1;
    }

    try {
        run_dcu_matmul(N, M, P);
    } catch (const std::exception& e) {
        std::cerr << "An exception occurred in main: " << e.what() << std::endl;
        return 1;
    }

    return 0;
} 