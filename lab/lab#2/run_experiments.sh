#!/bin/bash

# 创建必要的目录
mkdir -p bin
mkdir -p results
mkdir -p results/language_comparison
mkdir -p fig
mkdir -p cachegrind_logs

echo "===== 编译C++程序 ====="

# 编译程序 - 使用优化级别O3
g++ -O3 -o bin/matrix_vector_cpp src/matrix_vector.cpp
g++ -O3 -o bin/sum_array_cpp src/sum_array.cpp

echo "===== 运行矩阵-向量乘法C++实验 ====="

# 矩阵大小列表
matrix_sizes=(1000 2000 4000)

# 创建结果文件
echo "size,col_access,row_access,unroll5,unroll10,unroll15,unroll20" > results/matrix_vector_results.csv

for size in "${matrix_sizes[@]}"; do
    echo "测试矩阵大小: $size x $size"
    
    # 运行所有算法
    result=$(bin/matrix_vector_cpp $size 0)
    
    # 提取结果
    col_time=$(echo "$result" | grep "Col access time" | awk '{print $4}')
    row_time=$(echo "$result" | grep "Row access time" | awk '{print $4}')
    unroll5_time=$(echo "$result" | grep "Unroll5 time" | awk '{print $3}')
    unroll10_time=$(echo "$result" | grep "Unroll10 time" | awk '{print $3}')
    unroll15_time=$(echo "$result" | grep "Unroll15 time" | awk '{print $3}')
    unroll20_time=$(echo "$result" | grep "Unroll20 time" | awk '{print $3}')
    
    # 将结果写入CSV
    echo "$size,$col_time,$row_time,$unroll5_time,$unroll10_time,$unroll15_time,$unroll20_time" >> results/matrix_vector_results.csv
    
    echo "完成矩阵大小 $size"
done

echo "===== 运行数组求和C++实验 ====="

# 数组大小列表（以2的幂表示）
array_sizes=(18 22 24)  # 2^18, 2^22, 2^24

# 创建结果文件
echo "size,naive,dual_path,recursive" > results/sum_array_results.csv

for power in "${array_sizes[@]}"; do
    size=$((1 << $power))
    echo "测试数组大小: 2^$power = $size"
    
    # 运行测试
    result=$(bin/sum_array_cpp $size)
    
    # 提取结果
    naive_time=$(echo "$result" | grep "Naive sum time" | awk '{print $4}')
    dual_time=$(echo "$result" | grep "Dual path sum time" | awk '{print $5}')
    recursive_time=$(echo "$result" | grep "Recursive sum time" | awk '{print $4}')
    
    # 将结果写入CSV
    echo "$size,$naive_time,$dual_time,$recursive_time" >> results/sum_array_results.csv
    
    echo "完成数组大小 2^$power"
done

echo "===== 运行语言对比实验 ====="

# 创建CSV文件
echo "size,cpp_col,cpp_row,cpp_unroll10,py_col,py_row,py_unroll10,py_numpy" > results/language_comparison/matrix_vector_comparison.csv

for size in "${matrix_sizes[@]}"; do
    echo "测试矩阵大小: $size x $size"
    
    # 运行C++版本
    cpp_col=$(bin/matrix_vector_cpp $size 1 | grep "Col access time" | awk '{print $4}')
    cpp_row=$(bin/matrix_vector_cpp $size 2 | grep "Row access time" | awk '{print $4}')
    cpp_unroll=$(bin/matrix_vector_cpp $size 4 | grep "Unroll10 time" | awk '{print $3}')
    
    # 运行Python版本
    py_results=$(python3 src/matrix_vector.py $size)
    py_col=$(echo "$py_results" | grep "Python Column-major access time" | awk '{print $5}')
    py_row=$(echo "$py_results" | grep "Python Row-major access time" | awk '{print $5}')
    py_unroll=$(echo "$py_results" | grep "Python Unroll10 time" | awk '{print $4}')
    py_numpy=$(echo "$py_results" | grep "NumPy native implementation time" | awk '{print $5}')
    
    # 输出结果到CSV文件
    echo "$size,$cpp_col,$cpp_row,$cpp_unroll,$py_col,$py_row,$py_unroll,$py_numpy" >> results/language_comparison/matrix_vector_comparison.csv
    
    echo "完成矩阵大小 $size"
done

echo "运行C++和Python数组求和对比..."

# 数组大小列表（以元素数量表示）
sum_sizes=(262144 4194304 16777216)  # 2^18, 2^22, 2^24

echo "size,cpp_naive,cpp_dual,cpp_recursive,py_naive,py_dual,py_recursive,py_numpy" > results/language_comparison/sum_array_comparison.csv

for size in "${sum_sizes[@]}"; do
    echo "测试数组大小: $size"
    
    # 运行C++版本
    cpp_results=$(bin/sum_array_cpp $size)
    cpp_naive=$(echo "$cpp_results" | grep "Naive sum time" | awk '{print $4}')
    cpp_dual=$(echo "$cpp_results" | grep "Dual path sum time" | awk '{print $5}')
    cpp_recursive=$(echo "$cpp_results" | grep "Recursive sum time" | awk '{print $4}')
    
    # 运行Python版本
    py_results=$(python3 src/sum_array.py $size)
    py_naive=$(echo "$py_results" | grep "Python naive sum time" | awk '{print $5}')
    py_dual=$(echo "$py_results" | grep "Python dual path sum time" | awk '{print $6}')
    py_recursive=$(echo "$py_results" | grep "Python recursive sum time" | awk '{print $6}')
    py_numpy=$(echo "$py_results" | grep "NumPy native sum time" | awk '{print $5}')
    
    # 输出结果到CSV文件
    echo "$size,$cpp_naive,$cpp_dual,$cpp_recursive,$py_naive,$py_dual,$py_recursive,$py_numpy" >> results/language_comparison/sum_array_comparison.csv
    
    echo "完成数组大小 $size"
done

echo "===== 生成图表 ====="

# 生成基本性能图表
echo "生成基本性能图表..."
python3 src/plot_matrix_vector_performance.py
python3 src/plot_access_patterns.py
python3 src/plot_loop_unrolling.py

# 生成语言对比图表
echo "生成语言对比图表..."
python3 src/plot_language_comparison.py

echo "===== 实验完成 ====="
echo "结果已保存在 results/ 目录下"
echo "图表已保存在 fig/ 目录下" 