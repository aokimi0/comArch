import numpy as np
import time
import sys

# 朴素算法 - 列优先访问
def col_access(matrix, vector):
    n = len(matrix)
    result = np.zeros(n)
    for j in range(n):  # 列优先访问
        for i in range(n):
            result[i] += matrix[i][j] * vector[j]
    return result

# 缓存优化 - 行优先访问
def row_access(matrix, vector):
    n = len(matrix)
    result = np.zeros(n)
    for i in range(n):  # 行优先访问
        sum_val = 0.0
        for j in range(n):
            sum_val += matrix[i][j] * vector[j]
        result[i] = sum_val
    return result

# NumPy原生实现
def numpy_matmul(matrix, vector):
    return np.matmul(matrix, vector)

# Python循环展开 - 模拟展开10次的优化
def unroll10(matrix, vector):
    n = len(matrix)
    result = np.zeros(n)
    for i in range(n):
        sum_val = 0.0
        j = 0
        # 每次处理10个元素
        while j <= n - 10:
            sum_val += (matrix[i][j] * vector[j] +
                       matrix[i][j+1] * vector[j+1] +
                       matrix[i][j+2] * vector[j+2] +
                       matrix[i][j+3] * vector[j+3] +
                       matrix[i][j+4] * vector[j+4] +
                       matrix[i][j+5] * vector[j+5] +
                       matrix[i][j+6] * vector[j+6] +
                       matrix[i][j+7] * vector[j+7] +
                       matrix[i][j+8] * vector[j+8] +
                       matrix[i][j+9] * vector[j+9])
            j += 10
        # 处理剩余元素
        while j < n:
            sum_val += matrix[i][j] * vector[j]
            j += 1
        result[i] = sum_val
    return result

def main():
    # 默认矩阵大小
    n = 1000
    
    # 解析命令行参数
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    
    print(f"Matrix size: {n}x{n}")
    
    # 创建随机矩阵和向量
    matrix = np.random.random((n, n))
    vector = np.random.random(n)
    
    # 测试列优先访问
    start_time = time.time()
    result_col = col_access(matrix, vector)
    col_time = (time.time() - start_time) * 1000
    print(f"Python Column-major access time: {col_time:.2f} ms")
    
    # 测试行优先访问
    start_time = time.time()
    result_row = row_access(matrix, vector)
    row_time = (time.time() - start_time) * 1000
    print(f"Python Row-major access time: {row_time:.2f} ms")
    
    # 测试展开优化
    start_time = time.time()
    result_unroll = unroll10(matrix, vector)
    unroll_time = (time.time() - start_time) * 1000
    print(f"Python Unroll10 time: {unroll_time:.2f} ms")
    
    # 测试NumPy原生实现
    start_time = time.time()
    result_numpy = numpy_matmul(matrix, vector)
    numpy_time = (time.time() - start_time) * 1000
    print(f"NumPy native implementation time: {numpy_time:.2f} ms")
    
    # 计算加速比
    col_speedup = col_time / row_time
    print(f"Row vs Column speedup: {col_speedup:.2f}x")
    
    numpy_speedup = col_time / numpy_time
    print(f"NumPy vs Column speedup: {numpy_speedup:.2f}x")
    
    # 验证结果正确性
    tolerance = 1e-10
    row_correct = np.allclose(result_col, result_row, rtol=tolerance)
    unroll_correct = np.allclose(result_col, result_unroll, rtol=tolerance)
    numpy_correct = np.allclose(result_col, result_numpy, rtol=tolerance)
    
    print(f"Row access result correct: {row_correct}")
    print(f"Unroll10 result correct: {unroll_correct}")
    print(f"NumPy result correct: {numpy_correct}")

if __name__ == "__main__":
    main() 