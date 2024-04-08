#!/bin/bash

# Define session name
session_name="AIDS_grid"

# Check if the tmux session already exists, and delete it if it does
tmux has-session -t $session_name 2>/dev/null
if [ $? -eq 0 ]; then
  echo "Session $session_name already exists. Killing it."
  tmux kill-session -t $session_name
fi

# Start a new tmux session with the name main_exps without attaching to it
tmux new-session -d -s $session_name

# Define datasets
datasets=("AIDS")

# Define GPU devices
devices=(2 3 4)

# Counter for devices array to assign GPUs in round-robin
device_counter=0

# Loop through each dataset
for dataset in "${datasets[@]}"; do
  # Determine the current device
  device=${devices[device_counter]}

  # Command to run
  command="source activate llm_gnn && python main.py --dataset $dataset --device $device --exp_feedback_times 1"


  # First dataset uses the initial pane, subsequent datasets get new panes
  if [ "$dataset" == "${datasets[0]}" ]; then
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

# Kill the initial pane that was automatically created by tmux
# tmux kill-pane -t $session_name.0

# Attach to the session
tmux attach-session -t $session_name
