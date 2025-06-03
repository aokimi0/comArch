#!/bin/bash

# 创建结果目录
mkdir -p results/language_comparison

echo "===== 编译C++程序 ====="
g++ -O3 -o bin/matrix_vector_cpp src/matrix_vector.cpp
g++ -O3 -o bin/sum_array_cpp src/sum_array.cpp

echo "===== 运行矩阵-向量乘法语言对比实验 ====="

# 矩阵大小列表
matrix_sizes=(1000 2000 4000)

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

echo "===== 运行数组求和语言对比实验 ====="

# 数组大小列表
array_sizes=(262144 4194304 16777216)

echo "size,cpp_naive,cpp_dual,cpp_recursive,py_naive,py_dual,py_recursive,py_numpy" > results/language_comparison/sum_array_comparison.csv

for size in "${array_sizes[@]}"; do
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

echo "实验完成，结果保存在 results/language_comparison/ 目录下" 