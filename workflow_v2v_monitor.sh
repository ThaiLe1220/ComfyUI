#!/bin/bash

while true; do
    python3 workflow_v2v.py
    exit_code=$?
    
    if [ $exit_code -ne 0 ]; then
        echo "workflow_v2v.py failed with exit code $exit_code. Restarting..."
        sleep 1
    else
        echo "workflow_v2v.py completed successfully. Exiting."
        break
    fi
done