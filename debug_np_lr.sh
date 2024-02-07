#!/bin/bash

session_name="debug_np_lr"
# Check if the tmux session already exists, and delete it if it does
tmux has-session -t $session_name 2>/dev/null
if [ $? -eq 0 ]; then
  echo "Session $session_name already exists. Killing it."
  tmux kill-session -t $session_name
fi

# Start a new tmux session and name it Tox21_experiments
tmux new-session -d -s debug_np_lr

# Define the base command without the device and learning rate
base_command="source activate llm_gnn && python /home/nee7ne/research_code/new_LLM4GNNExplanation/main.py --dataset Tox21 -epe 100 --exp_data_percent 0.2 --exp_train_epochs 1 --exp_feedback_times 1"

# First command variant
first_command_variant="-at np"

# Define learning rates and devices to iterate over
learning_rates=("1e-2" "1e-3" "1e-4" "1e-5")
devices=(0 1 2 3 4 5)

# Loop through each device
for device in "${devices[@]}"; do
    # For each device, loop through each learning rate
    for lr in "${learning_rates[@]}"; do
        # Prepare the command with the specific device and learning rate
        command_with_device_lr="${base_command} --exp_pretrain_lr ${lr} --device ${device}"

        # Run first variant of the command in a new pane
        tmux split-window -h
        tmux select-layout tiled > /dev/null
        tmux send-keys "${command_with_device_lr} ${first_command_variant}" C-m

        # Split window again for the second command without the variant
        tmux split-window -h
        tmux select-layout tiled > /dev/null

        # Run the base command without variant in the new last pane
        tmux send-keys "${command_with_device_lr}" C-m
    done
done

# After setup, kill the first pane which was empty initially
tmux select-pane -t 0
tmux kill-pane

# Attach to the session
tmux attach-session -t debug_np_lr
