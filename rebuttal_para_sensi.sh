#!/bin/bash

# Define session name
session_name="para_sensi"

# Check if the tmux session already exists, and delete it if it does
tmux has-session -t $session_name 2>/dev/null
if [ $? -eq 0 ]; then
  echo "Session $session_name already exists. Killing it."
  tmux kill-session -t $session_name
fi

# Start a new tmux session with the name main_exps without attaching to it
tmux new-session -d -s $session_name

# Define datasets
datasets=("AIDS" "ClinTox")

# Define GPU devices
devices=(0 3 5)

# Define the pairs of positive numbers
pairs=(
    "0.96 0.19"
    "0.89 0.45"
    "0.71 0.71"
    "0.45 0.89"
    "0.19 0.96"
)

# Counter for devices array to assign GPUs in round-robin
device_counter=0

# Loop through each dataset
for dataset in "${datasets[@]}"; do
  # Loop through each pair of (a, b)
  for pair in "${pairs[@]}"; do
    # Determine the current device
    device=${devices[device_counter]}

    # Split the pair into two variables
    read -r a b <<< "$pair"

    # Command to run, including the new parameters
    command="source activate explain_gnn && CUDA_VISIBLE_DEVICES=$device python main.py --dataset $dataset --exp_m_mu $a --exp_c_mu $b"

    # Send the command to a new window or pane in tmux
    if [[ "$dataset" == "${datasets[0]}" ]] && [[ "$pair" == "${pairs[0]}" ]]; then
      tmux send-keys -t $session_name "$command" C-m
    else
      tmux split-window -h -t $session_name
      tmux send-keys -t $session_name "$command" C-m
    fi

    # Adjust layout to ensure even spacing
    tmux select-layout -t $session_name tiled

    # Increment device counter and reset if it exceeds the number of devices
    ((device_counter++))
    if [ "$device_counter" -ge "${#devices[@]}" ]; then
      device_counter=0
    fi
  done
done

# Attach to the session
tmux attach-session -t $session_name
