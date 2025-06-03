#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <mpi.h>

// 编译执行方式参考：
// 编译， 也可以使用g++，但使用MPI时需使用mpic
// mpic++ -fopenmp -o outputfile sourcefile.cpp

// 运行 baseline
// ./outputfile baseline

// 运行 OpenMP
// ./outputfile openmp

// 运行 子块并行优化
// ./outputfile block

// 运行 MPI（假设 4 个进程）
// mpirun -np 4 ./outputfile mpi

// 运行 MPI（假设 4 个进程）
// ./outputfile other


// 初始化矩阵（以一维数组形式表示），用于随机填充浮点数
void init_matrix(std::vector<double>& mat, int rows, int cols) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-100.0, 100.0);
    for (int i = 0; i < rows * cols; ++i)
        mat[i] = dist(gen);
}

// 验证计算优化后的矩阵计算和baseline实现是否结果一致，可以设计其他验证方法，来验证计算的正确性和性能
bool validate(const std::vector<double>& A, const std::vector<double>& B, int rows, int cols, double tol = 1e-6) {
    for (int i = 0; i < rows * cols; ++i)
        if (std::abs(A[i] - B[i]) > tol) return false;
    return true;
}

// 基础的矩阵乘法baseline实现（使用一维数组）
void matmul_baseline(const std::vector<double>& A,
                     const std::vector<double>& B,
                     std::vector<double>& C, int N, int M, int P) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < P; ++j) {
            C[i * P + j] = 0;
            for (int k = 0; k < M; ++k)
                C[i * P + j] += A[i * M + k] * B[k * P + j];
        }
}

// 方式1: 利用OpenMP进行多线程并发的编程 （主要修改函数）
void matmul_openmp(const std::vector<double>& A,
                   const std::vector<double>& B,
                   std::vector<double>& C, int N, int M, int P) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < P; ++j) {
            double sum = 0.0;
            for (int k = 0; k < M; ++k) {
                sum += A[i * M + k] * B[k * P + j];
            }
            C[i * P + j] = sum;
        }
    }
}

// 方式2: 利用子块并行思想，进行缓存友好型的并行优化方法 （主要修改函数)
void matmul_block_tiling(const std::vector<double>& A,
                         const std::vector<double>& B,
                         std::vector<double>& C, int N, int M, int P, int block_size = 64) {
    #pragma omp parallel for collapse(2)
    for (int i_block = 0; i_block < N; i_block += block_size) {
        for (int j_block = 0; j_block < P; j_block += block_size) {
            for (int k_block = 0; k_block < M; k_block += block_size) {
                for (int i = i_block; i < std::min(i_block + block_size, N); ++i) {
                    for (int j = j_block; j < std::min(j_block + block_size, P); ++j) {
                        // C[i * P + j] 必须在最内层循环之外初始化或在 k_block == 0 时初始化
                        if (k_block == 0) {
                            C[i * P + j] = 0;
                        }
                        double sum_val = 0.0; // 用于累加一个块内的乘积
                        for (int k = k_block; k < std::min(k_block + block_size, M); ++k) {
                           sum_val += A[i * M + k] * B[k * P + j];
                        }
                        C[i * P + j] += sum_val; // 将块的累加结果加到C
                    }
                }
            }
        }
    }
}

// 方式3: 利用MPI消息传递，实现多进程并行优化 （主要修改函数）
void matmul_mpi(int N, int M, int P) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<double> A_local, B_global(M * P), C_local;
    std::vector<double> A_global, C_global;

    int rows_per_process = N / size;
    int remaining_rows = N % size;

    if (rank == 0) {
        A_global.resize(N * M);
        C_global.resize(N * P);
        init_matrix(A_global, N, M);
        init_matrix(B_global, M, P); // B 矩阵在所有进程中都需要，但这里只在主进程初始化并通过广播发送
    }

    // 广播 B 矩阵给所有进程
    MPI_Bcast(B_global.data(), M * P, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 计算每个进程负责的行数和偏移量
    std::vector<int> sendcounts(size);
    std::vector<int> displs(size);
    int current_displ = 0;
    for (int i = 0; i < size; ++i) {
        sendcounts[i] = (rows_per_process + (i < remaining_rows ? 1 : 0)) * M;
        displs[i] = current_displ;
        current_displ += sendcounts[i];
    }

    int local_N = rows_per_process + (rank < remaining_rows ? 1 : 0);
    A_local.resize(local_N * M);

    // 分发 A 矩阵的行到各个进程
    MPI_Scatterv(A_global.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                 A_local.data(), local_N * M, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    C_local.resize(local_N * P, 0.0);
    // 各个进程进行局部矩阵乘法
    for (int i = 0; i < local_N; ++i) {
        for (int j = 0; j < P; ++j) {
            for (int k = 0; k < M; ++k) {
                C_local[i * P + j] += A_local[i * M + k] * B_global[k * P + j];
            }
        }
    }

    // 收集结果到主进程
    // 更新 recvcounts 和 recv_displs 用于 C 矩阵
    current_displ = 0;
    for (int i = 0; i < size; ++i) {
        sendcounts[i] = (rows_per_process + (i < remaining_rows ? 1 : 0)) * P; // P 列
        displs[i] = current_displ;
        current_displ += sendcounts[i];
    }

    MPI_Gatherv(C_local.data(), local_N * P, MPI_DOUBLE,
                C_global.data(), sendcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // 可选：验证结果，这里仅打印完成信息
        std::vector<double> C_ref(N * P);
        // 需要 A_global 和 B_global 来计算 C_ref
        // matmul_baseline(A_global, B_global, C_ref, N, M, P); // matmul_baseline 需要在这里可用或重新获取 A_global 和 B_global
        // std::cout << "[MPI] Validation with baseline would go here." << std::endl;
        std::cout << "[MPI] Done. Master collected results." << std::endl;
         // 可以在此将 C_global 与 C_ref 进行比较验证
         // 例如，可以用之前定义的 validate 函数，但这需要 C_ref 被正确计算
    }
}

// 方式4: 其他方式 （主要修改函数）
void matmul_other(const std::vector<double>& A,
                  const std::vector<double>& B,
                  std::vector<double>& C, int N, int M, int P) {
    std::cout << "Other methods..." << std::endl;
}

int main(int argc, char** argv) {
    const int N = 1024, M = 2048, P = 512;
    std::string mode = argc >= 2 ? argv[1] : "baseline";

    if (mode == "mpi") {
        MPI_Init(&argc, &argv);
        matmul_mpi(N, M, P);
        MPI_Finalize();
        return 0;
    }

    std::vector<double> A(N * M);
    std::vector<double> B(M * P);
    std::vector<double> C(N * P, 0);
    std::vector<double> C_ref(N * P, 0);

    init_matrix(A, N, M);
    init_matrix(B, M, P);
    matmul_baseline(A, B, C_ref, N, M, P);

    if (mode == "baseline") {
        std::cout << "[Baseline] Done.\n";
    } else if (mode == "openmp") {
        matmul_openmp(A, B, C, N, M, P);
        std::cout << "[OpenMP] Valid: " << validate(C, C_ref, N, P) << std::endl;
    } else if (mode == "block") {
        matmul_block_tiling(A, B, C, N, M, P);
        std::cout << "[Block Parallel] Valid: " << validate(C, C_ref, N, P) << std::endl;
    } else if (mode == "other") {
        matmul_other(A, B, C, N, M, P);
        std::cout << "[Other] Valid: " << validate(C, C_ref, N, P) << std::endl;
    } else {
        std::cerr << "Usage: ./main [baseline|openmp|block|mpi]" << std::endl;
    }
        // 需额外增加性能评测代码或其他工具进行评测
    return 0;
}