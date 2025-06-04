#!/bin/bash

LOG_FILE="log/mlp_forward_perf.log"
SRC_DIR="src"
DCU_SRC_FILE="mlp_forward_dcu.cpp"
DCU_EXE_NAME="mlp_forward_dcu"

# 清理以前的日志和可执行文件
rm -f $LOG_FILE $DCU_EXE_NAME

# 创建日志目录
mkdir -p log

echo "Starting Lesson 2 MLP Forward Pass Performance Test..." | tee -a $LOG_FILE
echo "Timestamp: $(date)" | tee -a $LOG_FILE
echo "=======================================" | tee -a $LOG_FILE

# 编译 DCU 加速版本
echo "" | tee -a $LOG_FILE
echo "--- MLP Forward Pass DCU Version ---" | tee -a $LOG_FILE
echo "Compiling Lesson 2 MLP Forward DCU ($DCU_SRC_FILE)..." | tee -a $LOG_FILE

if hipcc -O3 "$SRC_DIR/$DCU_SRC_FILE" -o $DCU_EXE_NAME; then
    echo "DCU Compilation successful: $DCU_EXE_NAME" | tee -a $LOG_FILE
    echo "Running MLP Forward DCU Version..." | tee -a $LOG_FILE
    echo "Command: ./$DCU_EXE_NAME" >> $LOG_FILE
    (time ./$DCU_EXE_NAME) >> $LOG_FILE 2>&1
else
    echo "DCU Compilation failed or hipcc is not functional." | tee -a $LOG_FILE
    echo "MLP Forward DCU run cannot be performed." | tee -a $LOG_FILE
fi

echo "=======================================" | tee -a $LOG_FILE
echo "Lesson 2 MLP Forward Pass Test Finished." | tee -a $LOG_FILE
echo "Log saved to $LOG_FILE" | tee -a $LOG_FILE

# 清理可执行文件
rm -f $DCU_EXE_NAME

echo "run.sh script for Lesson 2 finished." 