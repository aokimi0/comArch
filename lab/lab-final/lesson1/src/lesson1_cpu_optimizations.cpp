#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <cmath> // For fabs
#include <string>
#include <stdexcept> // For std::runtime_error

#ifdef _OPENMP
#include <omp.h>
#endif

// Conditional MPI compilation
#ifdef ENABLE_MPI
#include <mpi.h>
#define USE_MPI
#endif

// Matrix dimensions
const int N_DEFAULT = 1024;
const int M_DEFAULT = 2048;
const int P_DEFAULT = 512;
const int BLOCK_SIZE_DEFAULT = 64;

// --- Helper Functions ---

// Initialize matrix with random double values or zeros
void initialize_matrix(std::vector<double>& matrix, int rows, int cols, bool random_fill = true, double val = 0.0) {
    matrix.assign(static_cast<size_t>(rows) * cols, val);
    if (random_fill) {
        // Consistent seed for reproducibility during verification across runs if needed
        // std::mt19937 gen(42);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> distrib(-2.0, 2.0); // Smaller range for easier debugging
        for (size_t i = 0; i < matrix.size(); ++i) {
            matrix[i] = distrib(gen);
        }
    }
}

// Print matrix (for debugging small matrices)
void print_matrix(const std::string& name, const std::vector<double>& matrix, int rows, int cols) {
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

// Verify if two matrices are element-wise equal within a tolerance
bool verify_matrices(const std::vector<double>& mat1, const std::vector<double>& mat2, int rows, int cols, double tolerance = 1e-6) {
    if (mat1.size() != static_cast<size_t>(rows * cols) || mat2.size() != static_cast<size_t>(rows * cols)) {
        std::cerr << "Verification failed: Matrix dimension mismatch for verification. mat1: " << mat1.size() << ", mat2: " << mat2.size() << ", expected: " << static_cast<size_t>(rows)*cols << "\n";
        return false;
    }
    double max_diff = 0.0;
    int errors = 0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            size_t index = static_cast<size_t>(i) * cols + j;
            double diff = std::fabs(mat1[index] - mat2[index]);
            if (diff > max_diff) {
                max_diff = diff;
            }
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

// --- Baseline Matrix Multiplication ---
void matrix_multiply_baseline(
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

// --- OpenMP Optimized Matrix Multiplication ---
void matrix_multiply_openmp(
    const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C,
    int N, int M, int P) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) collapse(2) // Using static schedule, collapse for outer loops
#endif
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

// --- Blocked Matrix Multiplication (with OpenMP) ---
void matrix_multiply_blocked_openmp(
    const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C,
    int N, int M, int P, int block_size) {
    if (block_size <= 0) {
        throw std::runtime_error("Block size must be positive.");
    }
    // Initialize C to zeros, as accumulation happens across k0 blocks
    for(size_t i = 0; i < static_cast<size_t>(N)*P; ++i) C[i] = 0.0;

#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int i0 = 0; i0 < N; i0 += block_size) {
        for (int j0 = 0; j0 < P; j0 += block_size) {
            for (int k0 = 0; k0 < M; k0 += block_size) {
                for (int i = i0; i < std::min(i0 + block_size, N); ++i) {
                    for (int j = j0; j < std::min(j0 + block_size, P); ++j) {
                        double sum_val = 0.0; 
                        for (int k = k0; k < std::min(k0 + block_size, M); ++k) {
                           sum_val += A[static_cast<size_t>(i) * M + k] * B[static_cast<size_t>(k) * P + j];
                        }
                        C[static_cast<size_t>(i) * P + j] += sum_val;
                    }
                }
            }
        }
    }
}


// --- MPI Optimized Matrix Multiplication ---
void matrix_multiply_mpi(
    const std::vector<double>& A_global_const, 
    const std::vector<double>& B_global_const, 
    std::vector<double>& C_global,       
    int N, int M, int P, int world_rank, int world_size) {
#ifdef USE_MPI
    if (world_size == 1) {
        if (world_rank == 0) {
            // std::cout << "MPI: Running with 1 process, using OpenMP version for computation.\n";
            matrix_multiply_openmp(A_global_const, B_global_const, C_global, N, M, P);
        }
        return;
    }

    std::vector<double> B_local(static_cast<size_t>(M) * P); 
    if (world_rank == 0) {
        B_local = B_global_const;
    }
    MPI_Bcast(B_local.data(), M * P, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<int> counts_A(world_size);       
    std::vector<int> displs_A(world_size);       
    std::vector<int> row_counts_local(world_size);   
    std::vector<int> counts_C(world_size);       
    std::vector<int> displs_C(world_size);       

    int current_row_displ = 0;
    for (int i = 0; i < world_size; ++i) {
        row_counts_local[i] = N / world_size + (i < N % world_size ? 1 : 0);
        counts_A[i] = row_counts_local[i] * M; 
        displs_A[i] = current_row_displ * M; 
        
        counts_C[i] = row_counts_local[i] * P;
        displs_C[i] = current_row_displ * P;

        current_row_displ += row_counts_local[i];
    }

    int local_N = row_counts_local[world_rank];
    std::vector<double> A_local(static_cast<size_t>(local_N) * M);

    MPI_Scatterv(world_rank == 0 ? A_global_const.data() : nullptr,
                 counts_A.data(), displs_A.data(), MPI_DOUBLE,
                 A_local.data(), counts_A[world_rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<double> C_local(static_cast<size_t>(local_N) * P, 0.0);

#ifdef _OPENMP // Enable hybrid MPI + OpenMP
    #pragma omp parallel for schedule(static) collapse(2)
#endif
    for (int i = 0; i < local_N; ++i) {
        for (int j = 0; j < P; ++j) {
            double sum = 0.0;
            for (int k = 0; k < M; ++k) {
                sum += A_local[static_cast<size_t>(i) * M + k] * B_local[static_cast<size_t>(k) * P + j];
            }
            C_local[static_cast<size_t>(i) * P + j] = sum;
        }
    }
    
    MPI_Gatherv(C_local.data(), counts_C[world_rank], MPI_DOUBLE,
                world_rank == 0 ? C_global.data() : nullptr, 
                counts_C.data(), displs_C.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

#else
    if(world_rank == 0) {
      std::cerr << "MPI version called, but USE_MPI not defined. Falling back to OpenMP.\n";
      matrix_multiply_openmp(A_global_const, B_global_const, C_global, N, M, P);
    }
#endif
}


int main(int argc, char* argv[]) {
    std::string version_to_run = "baseline";
    if (argc > 1) {
        version_to_run = argv[1];
    }

    int N = N_DEFAULT;
    int M = M_DEFAULT;
    int P = P_DEFAULT;
    int block_size = BLOCK_SIZE_DEFAULT;

    if (argc > 2) N = std::stoi(argv[2]);
    if (argc > 3) M = std::stoi(argv[3]);
    if (argc > 4) P = std::stoi(argv[4]);
    if (argc > 5 && version_to_run == "block") block_size = std::stoi(argv[5]);

    int world_rank = 0;
    int world_size = 1;

#ifdef USE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
#else
    if (version_to_run == "mpi") {
        std::cerr << "Error: MPI version requested but not compiled with USE_MPI defined.\n";
        return 1;
    }
#endif

    std::vector<double> A_host, B_host, C_result_host, C_baseline_verify_host;
    
    if (world_rank == 0) {
        if (N <=0 || M <= 0 || P <= 0) { 
            std::cerr << "Error: Matrix dimensions N, M, P must be positive. Got N="<< N <<", M="<< M <<", P="<<P<< std::endl;
#ifdef USE_MPI
            MPI_Abort(MPI_COMM_WORLD, 1); 
#endif
            return 1;
        }
        std::cout << "Process 0: Initializing matrices. Target N, M, P: " 
                  << N << ", " << M << ", " << P << ".\n";
        if (version_to_run == "block") std::cout << "Block size: " << block_size << "\n";

        A_host.resize(static_cast<size_t>(N) * M);
        B_host.resize(static_cast<size_t>(M) * P);
        C_result_host.assign(static_cast<size_t>(N) * P, 0.0);
        C_baseline_verify_host.resize(static_cast<size_t>(N) * P);

        initialize_matrix(A_host, N, M, true);
        initialize_matrix(B_host, M, P, true);
        
        // print_matrix("A_host (init)", A_host, N, M);
        // print_matrix("B_host (init)", B_host, M, P);

        std::cout << "Process 0: Computing baseline for verification...\n";
        auto baseline_start_time = std::chrono::high_resolution_clock::now();
        matrix_multiply_baseline(A_host, B_host, C_baseline_verify_host, N, M, P);
        auto baseline_end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> baseline_elapsed_ms = baseline_end_time - baseline_start_time;
        std::cout << "Process 0: Baseline for verification computed in " << baseline_elapsed_ms.count() << " ms.\n";
        // print_matrix("C_baseline_verify_host (main)", C_baseline_verify_host, N, P);
    }

    // For MPI, B needs to be broadcasted, A needs to be scattered. C will be gathered.
    // Ensure non-rank-0 processes have B_host allocated if matrix_multiply_mpi expects it (it doesn't, it uses B_local)

    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_chrono;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_time_chrono;
    double mpi_wtime_start = 0.0, mpi_wtime_end = 0.0;

    if (world_rank == 0) {
        std::cout << "\nProcess 0: Starting computation for version: " << version_to_run << "\n";
    }

#ifdef USE_MPI
    if (world_size > 1 || version_to_run == "mpi") MPI_Barrier(MPI_COMM_WORLD);
    mpi_wtime_start = MPI_Wtime();
#endif
    start_time_chrono = std::chrono::high_resolution_clock::now();

    if (version_to_run == "baseline") {
        if (world_rank == 0) matrix_multiply_baseline(A_host, B_host, C_result_host, N, M, P);
    } else if (version_to_run == "openmp") {
        if (world_rank == 0) matrix_multiply_openmp(A_host, B_host, C_result_host, N, M, P);
    } else if (version_to_run == "block") {
        if (world_rank == 0) matrix_multiply_blocked_openmp(A_host, B_host, C_result_host, N, M, P, block_size);
    } else if (version_to_run == "mpi") {
#ifdef USE_MPI
        // A_host, B_host, C_result_host are already prepared on rank 0.
        // matrix_multiply_mpi will handle scattering A, broadcasting B, and gathering C.
        matrix_multiply_mpi(A_host, B_host, C_result_host, N, M, P, world_rank, world_size);
#else
        // This case should have been caught earlier.
#endif
    } else {
        if (world_rank == 0) std::cerr << "Error: Unknown version '" << version_to_run << "'. Supported: baseline, openmp, block, mpi.\n";
#ifdef USE_MPI
        MPI_Abort(MPI_COMM_WORLD, 1);
#endif
        return 1;
    }

#ifdef USE_MPI
    if (world_size > 1 || version_to_run == "mpi") MPI_Barrier(MPI_COMM_WORLD); 
    mpi_wtime_end = MPI_Wtime();
#endif
    end_time_chrono = std::chrono::high_resolution_clock::now();
    
    if (world_rank == 0) {
        std::chrono::duration<double, std::milli> elapsed_ms_chrono = end_time_chrono - start_time_chrono;
        double elapsed_ms_mpi_wtime = (mpi_wtime_end - mpi_wtime_start) * 1000.0;

        double reported_time_ms;
        if (version_to_run == "mpi" && world_size > 1) {
            reported_time_ms = elapsed_ms_mpi_wtime;
        } else {
            // For non-MPI versions or MPI with 1 proc, chrono is more reliable for single-node timing
            reported_time_ms = elapsed_ms_chrono.count();
        }
        std::cout << version_to_run << " Matmul Time: " << std::fixed << std::setprecision(4) << reported_time_ms << " ms\n";

        std::cout << "Validating " << version_to_run << " result against pre-computed baseline...\n";
        if (!verify_matrices(C_result_host, C_baseline_verify_host, N, P)) {
            std::cout << "Warning: " << version_to_run << " VERIFICATION FAILED against baseline.\n";
        } else {
            std::cout << version_to_run << " VERIFICATION PASSED.\n";
        }
        // print_matrix("C_result_host (" + version_to_run + ")", C_result_host, N, P);

    }

#ifdef USE_MPI
    MPI_Finalize();
#endif

    return 0;
} 