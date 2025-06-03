import numpy as np
import time
import sys

# 朴素算法：简单循环累加
def naive_sum(array):
    sum_val = 0.0
    for i in range(len(array)):
        sum_val += array[i]
    return sum_val

# 双链路算法：使用两个独立的累加器增加指令级并行度
def dual_path_sum(array):
    sum1 = 0.0
    sum2 = 0.0
    n = len(array)
    
    # 使用两个累加器
    for i in range(0, n, 2):
        sum1 += array[i]
        if i + 1 < n:  # 防止越界
            sum2 += array[i + 1]
    
    return sum1 + sum2

# 递归算法：分治策略
def recursive_sum_helper(array, start, end):
    if end - start <= 1:
        return array[start] if start < end else 0.0
    
    mid = start + (end - start) // 2
    return recursive_sum_helper(array, start, mid) + recursive_sum_helper(array, mid, end)

def recursive_sum(array):
    return recursive_sum_helper(array, 0, len(array))

# NumPy原生实现
def numpy_sum(array):
    return np.sum(array)

def main():
    # 默认数组大小 2^24 = 16,777,216
    n = 1 << 24
    
    # 从命令行参数获取数组大小
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    
    print(f"数组大小: {n} 元素")
    
    # 创建随机数组
    array = np.random.uniform(-1.0, 1.0, n)
    
    # 测试朴素算法性能
    start_time = time.time()
    naive_result = naive_sum(array)
    naive_time = (time.time() - start_time) * 1000
    print(f"Python naive sum time: {naive_time:.2f} ms")
    
    # 测试双链路算法性能
    start_time = time.time()
    dual_result = dual_path_sum(array)
    dual_time = (time.time() - start_time) * 1000
    print(f"Python dual path sum time: {dual_time:.2f} ms")
    
    # 对于大数组，递归深度可能超出限制，所以这里使用更小的数组进行递归测试
    if n > (1 << 18):
        test_n = 1 << 18  # 使用更小的数组大小进行递归测试
        test_array = array[:test_n]
        print(f"递归算法使用的缩减数组大小: {test_n} 元素")
    else:
        test_array = array
        test_n = n
    
    # 测试递归算法性能
    start_time = time.time()
    recursive_result = recursive_sum(test_array)
    recursive_time = (time.time() - start_time) * 1000
    recursive_time = recursive_time * (n / test_n) if test_n != n else recursive_time  # 估算完整数组所需时间
    print(f"Python recursive sum time (estimated): {recursive_time:.2f} ms")
    
    # 测试NumPy原生实现性能
    start_time = time.time()
    numpy_result = numpy_sum(array)
    numpy_time = (time.time() - start_time) * 1000
    print(f"NumPy native sum time: {numpy_time:.2f} ms")
    
    # 计算加速比
    dual_speedup = naive_time / dual_time if dual_time > 0 else 0
    recursive_speedup = naive_time / recursive_time if recursive_time > 0 else 0
    numpy_speedup = naive_time / numpy_time if numpy_time > 0 else 0
    
    print(f"Python dual path speedup: {dual_speedup:.2f}x")
    print(f"Python recursive speedup: {recursive_speedup:.2f}x")
    print(f"NumPy vs Python speedup: {numpy_speedup:.2f}x")
    
    # 验证结果
    tolerance = 1e-10
    dual_correct = abs(naive_result - dual_result) < tolerance
    numpy_correct = abs(naive_result - numpy_result) < tolerance
    recursive_correct = test_n == n and abs(naive_result - recursive_result) < tolerance
    
    print(f"Python dual path result correct: {dual_correct}")
    print(f"Python recursive result correct: {recursive_correct}")
    print(f"NumPy result correct: {numpy_correct}")

if __name__ == "__main__":
    main() 