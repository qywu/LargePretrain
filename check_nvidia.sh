#!/bin/bash

MEMORY_MINIMUM=100
# Delay the running
sleep 600s

while true; do
    # Get GPU UTIL
    GPU_UTIL=$(nvidia-smi --format=csv --query-gpu=memory.used)
    read -r -d '\n' -a tmp_array <<<$GPU_UTIL
    GPU_UTIL=${tmp_array[2]}

    if [ $GPU_UTIL -lt $MEMORY_MINIMUM ]; then
        # reboot if gpu is not used
        reboot
    else
        echo 'Running OK'
    fi

    sleep 5s
done
