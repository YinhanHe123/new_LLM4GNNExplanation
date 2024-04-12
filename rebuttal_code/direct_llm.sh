#!/bin/bash

# Define session name
SESSION_NAME="direct_llm"
tmux has-session -t $SESSION_NAME 2>/dev/null
if [ $? -eq 0 ]; then
  echo "Session $SESSION_NAME already exists. Killing it."
  tmux kill-session -t $SESSION_NAME
fi
# Start a new tmux session
tmux new-session -d -s $SESSION_NAME

# Array of datasets
datasets=('AIDS' 'ClinTox' 'Tox21' 'BBBP' 'Mutagenicity')

# Array of models
models=('gpt-3.5-turbo' 'gpt-4')

# Array of device numbers
devices=(0 1 2 3 4 5)

# Pane counter to keep track of which pane we're working on
pane_counter=0
source activate explain_gnn;
# Loop over each dataset
for dataset in "${datasets[@]}"; do
    # Loop over each model
    for model in "${models[@]}"; do
        if [ $pane_counter -ne 0 ]; then
            # Split window vertically for additional commands
            tmux split-window -v -t $SESSION_NAME
            tmux select-layout -t $SESSION_NAME tiled > /dev/null
        fi

        # Randomly select a device from 0 to 5
        device=${devices[$RANDOM % ${#devices[@]}]}

        # Form the command with the current dataset, model, and selected device
        cmd="python run_rebuttal.py --dataset $dataset --model $model --device $device"

        # Send the command to the pane
        tmux send-keys -t $pane_counter "$cmd" C-m
        
        # Increment pane counter
        pane_counter=$((pane_counter + 1))
    done
done

# Ensure all panes are evenly distributed
tmux select-layout -t $SESSION_NAME tiled

# Attach to the session
tmux attach -t $SESSION_NAME
