#!/bin/bash

# 定义相关路径和文件名
LESSON_DIR="lesson3"
SRC_DIR="src"
LOG_DIR="log"
REPORT_DIR="report"

CPP_SRC_FILE="mlp_train_dcu.cpp"
EXECUTABLE_NAME="mlp_train_dcu"
LOG_FILE="$LOG_DIR/mlp_train_perf.log"

# 进入 lesson3 目录 (如果脚本不是从 lesson3 内部运行)
# cd "$LESSON_DIR" || exit

# 清理旧的日志和可执行文件
rm -f "$EXECUTABLE_NAME" "$LOG_FILE"

# 创建日志目录 (如果不存在)
mkdir -p "$LOG_DIR"

echo "=====================================================" | tee -a "$LOG_FILE"
echo "Lesson 3: MLP Training & Inference Performance Test" | tee -a "$LOG_FILE"
echo "Timestamp: $(date)" | tee -a "$LOG_FILE"
echo "=====================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# 检查 hipcc 是否可用
if ! command -v hipcc &> /dev/null
then
    echo "[ERROR] hipcc command not found. Cannot compile HIP code." | tee -a "$LOG_FILE"
    echo "Please ensure HIP SDK is installed and hipcc is in your PATH." | tee -a "$LOG_FILE"
    # 即使 hipcc 不可用，由于代码主要依赖CPU执行并模拟计时，这里可以选择继续尝试编译（如果g++可以处理）
    # 或者直接报错退出。当前选择报错并提示，但允许用户修改脚本以适应纯CPU编译（移除-DENABLE_HIP_CODE）。
    echo "Attempting to compile with g++ as a fallback for CPU part if main is not guarded..." | tee -a "$LOG_FILE"
    COMPILER="g++"
    COMPILE_FLAGS="-std=c++17 -O3"
    # The -DENABLE_HIP_CODE should ideally not be used if hipcc is not found and kernels are not guarded for g++
else
    COMPILER="hipcc"
    # ENABLE_HIP_CODE 宏用于激活代码中与HIP相关的部分（例如空的核函数调用和hipMalloc/hipFree模拟）
    COMPILE_FLAGS="-std=c++17 -DENABLE_HIP_CODE -O3"
fi

echo "--- Compiling MLP Training Code ($CPP_SRC_FILE) --- " | tee -a "$LOG_FILE"
echo "Compiler: $COMPILER" | tee -a "$LOG_FILE"
echo "Compile flags: $COMPILE_FLAGS" | tee -a "$LOG_FILE"
echo "Source: $SRC_DIR/$CPP_SRC_FILE" | tee -a "$LOG_FILE"
echo "Output: $EXECUTABLE_NAME" | tee -a "$LOG_FILE"

# 编译源代码
if $COMPILER $SRC_DIR/$CPP_SRC_FILE -o $EXECUTABLE_NAME $COMPILE_FLAGS; then
    echo "Compilation successful." | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    echo "--- Running MLP Training & Testing --- " | tee -a "$LOG_FILE"
    echo "Executing: ./$EXECUTABLE_NAME" | tee -a "$LOG_FILE"
    echo "Output will be saved to: $LOG_FILE" | tee -a "$LOG_FILE"
    echo "-------------------------------------------" | tee -a "$LOG_FILE"
    
    # 执行程序并将所有输出追加到日志文件
    # 使用 time 命令记录整个执行的实际时间 (可选)
    # (time ./$EXECUTABLE_NAME) >> "$LOG_FILE" 2>&1
    ./$EXECUTABLE_NAME >> "$LOG_FILE" 2>&1
    
    EXEC_STATUS=$?
    if [ $EXEC_STATUS -eq 0 ]; then
        echo "-------------------------------------------" | tee -a "$LOG_FILE"
        echo "Execution finished successfully." | tee -a "$LOG_FILE"
    else
        echo "-------------------------------------------" | tee -a "$LOG_FILE"
        echo "[ERROR] Execution failed with status: $EXEC_STATUS" | tee -a "$LOG_FILE"
    fi
else
    echo "[ERROR] Compilation failed." | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "=====================================================" | tee -a "$LOG_FILE"
echo "Lesson 3 MLP Test Script Finished." | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"

# 清理可执行文件 (可选, 根据需要保留)
# rm -f "$EXECUTABLE_NAME"

echo "run.sh script for Lesson 3 finished." 