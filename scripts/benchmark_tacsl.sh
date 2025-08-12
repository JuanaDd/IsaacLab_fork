#!/bin/bash

# Check for debug parameter
DEBUG_MODE=false

# Generate timestamp for unique result filename
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_FILE="./tactile_record/benchmark/res_txt/res-${TIMESTAMP}.txt"

if [[ "$1" == "--debug" ]]; then
    DEBUG_MODE=true
    RESULT_FILE="./tactile_record/benchmark/res_txt/res_debug-${TIMESTAMP}.txt"
    echo "Debug mode enabled"
fi

# Configuration parameters
TACSL_EXAMPLE="./tacsl/tacsl_example.py"
BASE_ARGS="--headless --enable_cameras --indenter nut"

# Set different parameters based on debug mode
if [[ "$DEBUG_MODE" == "true" ]]; then
    NUM_ENVS_ARRAY=(4)
    MAX_RUNS=2
else
    NUM_ENVS_ARRAY=(1 4 16 64 256 512)
    MAX_RUNS=3
fi

WAIT_TIME=5

# Function to run a single configuration
run_single_config() {
    local num_envs=$1
    local run_count=0
    
    echo "========================================"
    echo "Starting test with num_envs=$num_envs, running $MAX_RUNS times"
    echo "========================================"
    
    while [ $run_count -lt $MAX_RUNS ]; do
        echo "=== num_envs=$num_envs, run $((run_count + 1))/$MAX_RUNS of tacsl_example.py ==="
        
        # Create temporary file to save output
        local temp_output=$(mktemp)
        echo "Temporary output file: $temp_output"
        
        # Start process and redirect output to temporary file (force unbuffered output)
        PYTHONUNBUFFERED=1 python $TACSL_EXAMPLE $BASE_ARGS --num_envs $num_envs > "$temp_output" 2>&1 &
        local pid=$!
        
        echo "Process started, PID: $pid"
        
        # Get baseline GPU memory usage
        local baseline_memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        echo "Baseline GPU memory usage: ${baseline_memory} MiB"
        
        # Monitor process output and GPU memory usage
        local found_stopping=false
        local max_memory=$baseline_memory
        while kill -0 $pid 2>/dev/null; do
            # Check if output file contains "Stopping replicator"
            if grep -q "Stopping benchmarking" "$temp_output" 2>/dev/null; then
                echo "Detected 'Stopping benchmarking' output, stopping process..."
                kill -2 $pid
                found_stopping=true
                break
            fi
            
            # Monitor GPU memory usage
            local current_memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
            if [ "$current_memory" -gt "$max_memory" ]; then
                max_memory=$current_memory
            fi
            
            sleep 0.5
        done
        
        # Wait for process to end
        wait $pid 2>/dev/null
        
        # Calculate GPU memory usage
        local used_memory=$((max_memory - baseline_memory))
        echo "=== GPU Memory Usage ==="
        echo "Baseline memory: ${baseline_memory} MiB"
        echo "Peak memory: ${max_memory} MiB"
        echo "Program usage: ${used_memory} MiB ($(python3 -c "print(f'{$used_memory * 1024 / 1000 / 1000:.2f}')")GB)"
        echo "========================"
        
        # Display output content (optional)
        echo "--- Process Output ---"
        cat "$temp_output"
        echo "--- End of Output ---"
        
        # Extract and save timing summary to result file
        echo "Extracting timing summary data..."
        timing_summary=$(grep -E "^\{'call_count':" "$temp_output" | tail -1)
        if [ -n "$timing_summary" ]; then
            # Add GPU memory information to timing summary
            enhanced_summary=$(echo "$timing_summary" | sed "s/}$/, 'gpu_memory_baseline_mib': $baseline_memory, 'gpu_memory_peak_mib': $max_memory, 'gpu_memory_used_mib': $used_memory}/")
            echo "$enhanced_summary" >> "$RESULT_FILE"
            echo "Timing summary (with GPU memory info) appended to $RESULT_FILE"
        else
            echo "No timing summary data found"
        fi
        
        # Clean up temporary files
        rm -f "$temp_output"
        
        # Wait for process to completely finish
        wait $pid 2>/dev/null
        echo "Process has ended"
        
        run_count=$((run_count + 1))
        
        if [ $run_count -lt $MAX_RUNS ]; then
            echo "Waiting $WAIT_TIME seconds before starting next run..."
            sleep $WAIT_TIME
        fi
    done
    
    echo "=== All runs for num_envs=$num_envs completed, total of $MAX_RUNS runs ==="
    echo ""
}

# Main function: iterate through all num_envs configurations
main() {
    echo "Starting batch testing, num_envs configurations: ${NUM_ENVS_ARRAY[*]}"
    echo "Each configuration runs $MAX_RUNS times"
    echo "Results will be saved to: $RESULT_FILE"
    echo ""
    
    # Add timestamp to results file at start
    echo "# Batch testing start time: $(date)" >> "$RESULT_FILE"
    
    for num_envs in "${NUM_ENVS_ARRAY[@]}"; do
        run_single_config $num_envs
    done
    
    echo "========================================" 
    echo "All configuration testing completed!"
    echo "Results saved to: $RESULT_FILE"
    echo "========================================" 
    
    # Add timestamp to results file at end
    echo "# Batch testing end time: $(date)" >> "$RESULT_FILE"
    echo "" >> "$RESULT_FILE"
}

# Execute main function
main

