#!/bin/bash
session_name="debug_np_epochs"
# Check if the tmux session already exists, and delete it if it does
tmux has-session -t $session_name 2>/dev/null
if [ $? -eq 0 ]; then
  echo "Session $session_name already exists. Killing it."
  tmux kill-session -t $session_name
fi
# Start a new tmux session and name it Tox21_experiments
tmux new-session -d -s debug_np_epochs

# Base command without --exp_pretrain_epochs and --device
base_command="source activate llm_gnn && python main.py --dataset Tox21 -epe 100 --exp_data_percent 0.2 --exp_train_epochs 1 --exp_feedback_times 1 --exp_pretrain_lr 1e-2"

# Pre-training epochs to iterate over
pretrain_epochs=(100 300 500)

# GPU devices
devices=(0 1 2 3 4 5)

# Counter for devices array
device_counter=0

# Iterate over pretrain_epochs
for epoch in "${pretrain_epochs[@]}"; do
    # Determine the current device
    device="${devices[device_counter]}"

    # Command with -at np
    command_np="${base_command} -at np --exp_pretrain_epochs ${epoch} --device ${device}"
    tmux split-window -h
    tmux select-layout tiled > /dev/null
    tmux send-keys "$command_np" C-m

    # Increase device_counter and reset if necessary
    ((device_counter++))
    if [ "$device_counter" -ge "${#devices[@]}" ]; then device_counter=0; fi

    # Determine the next device for the next command
    device="${devices[device_counter]}"

    # Command without -at np
    command_no_np="${base_command} --exp_pretrain_epochs ${epoch} --device ${device}"
    tmux split-window -h
    tmux select-layout tiled > /dev/null
    tmux send-keys "$command_no_np" C-m

    # Increase device_counter for the next iteration
    ((device_counter++))
    if [ "$device_counter" -ge "${#devices[@]}" ]; then device_counter=0; fi
done

# After setup, kill the first pane which was empty initially
tmux select-pane -t 0
tmux kill-pane

# Attach to the session
tmux attach-session -t debug_np_epochs
