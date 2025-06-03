#!/bin/bash

# Get the directory of the script itself and cd into it
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit

# Define paths relative to the script's directory (lesson3)
SRC_DIR="src"
LOG_DIR="log"
REPORT_DIR="report"
DATA_DIR="data"

SOURCE_FILE="$SRC_DIR/mlp_train_dcu.cpp"
HIP_EXECUTABLE="$SRC_DIR/mlp_train_dcu_hip_executable"
CPU_EXECUTABLE="$SRC_DIR/mlp_train_dcu_cpu_executable"
LOG_FILE="$LOG_DIR/mlp_train_perf.log"

# Create necessary directories if they don't exist
mkdir -p "$SRC_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$REPORT_DIR"
mkdir -p "$DATA_DIR"

# Clean up old log and executables
rm -f "$LOG_FILE"
rm -f "$HIP_EXECUTABLE"
rm -f "$CPU_EXECUTABLE"

echo "Current working directory: $(pwd)"

# Check if data file exists
if [ ! -f "$DATA_DIR/starlink_bw.json" ]; then
    echo "Error: Data file $DATA_DIR/starlink_bw.json not found in $(pwd)/$DATA_DIR/" >&2
    echo "Please ensure the data file is present before running." >&2
    exit 1 # Exit if data file is crucial and not found
fi

echo "Attempting to compile with hipcc (ENABLE_HIP_CODE defined)..."
# Note: ROCM_PATH might be needed if hipcc is not in default PATH
# Example: ROCM_PATH=${ROCM_PATH:-/opt/rocm}
# "$ROCM_PATH/bin/hipcc" ...
/opt/rocm/bin/hipcc -DENABLE_HIP_CODE "$SOURCE_FILE" -o "$HIP_EXECUTABLE" -Wall -O2

EXECUTABLE_TO_RUN=""

if [ -f "$HIP_EXECUTABLE" ]; then
    echo "Compilation with hipcc successful: $HIP_EXECUTABLE"
    EXECUTABLE_TO_RUN="$HIP_EXECUTABLE"
else
    echo "hipcc compilation failed. Output was:"
    # Try to show hipcc error if any was redirected, or just a message
    echo "(hipcc error message might have been complex or not captured if it hung)"
    echo "Attempting to compile with g++ for CPU fallback (ENABLE_HIP_CODE NOT defined)..."
    g++ -std=c++17 "$SOURCE_FILE" -o "$CPU_EXECUTABLE" -Wall -O2 -lm
    if [ -f "$CPU_EXECUTABLE" ]; then
        echo "Compilation with g++ successful: $CPU_EXECUTABLE"
        EXECUTABLE_TO_RUN="$CPU_EXECUTABLE"
    else
        echo "g++ compilation also failed. Cannot run the program." >&2
        # Create a dummy log file indicating failure, so report generation doesn't entirely break
        echo "FATAL: Compilation failed for both hipcc and g++." > "$LOG_FILE"
        echo "Source file: $SOURCE_FILE" >> "$LOG_FILE"
        echo "Please check compiler errors and environment." >> "$LOG_FILE"
        echo "Script finished with errors." >&2
        exit 1
    fi
fi

if [ -n "$EXECUTABLE_TO_RUN" ]; then
    echo "Running MLP training and testing (output to $LOG_FILE)..."
    # Run the executable. Its CWD will be lesson3/
    # The C++ code will look for data/starlink_bw.json which is lesson3/data/starlink_bw.json
    "$EXECUTABLE_TO_RUN" > "$LOG_FILE" 2>&1
    if [ $? -eq 0 ]; then
        echo "Execution finished successfully. Log saved to $(pwd)/$LOG_FILE"
    else
        echo "Execution failed. Check $LOG_FILE for error messages." >&2
        # Log might still be useful
    fi
else
    echo "No executable was built. Cannot run." >&2
fi

echo "Script finished." 