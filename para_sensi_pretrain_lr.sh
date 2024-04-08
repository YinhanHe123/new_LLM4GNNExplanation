# # we research on the parameter sensitivity, (1) number of feebacks (2) Pretrain epochs
# #!/bin/bash

# # Define session name
# SESSION_NAME="para_sensi_pretrain_lr"

# tmux has-session -t $SESSION_NAME 2>/dev/null
# if [ $? -eq 0 ]; then
#   echo "Session $SESSION_NAME already exists. Killing it."
#   tmux kill-session -t $SESSION_NAME
# fi

# # Start a new tmux session
# tmux new-session -d -s $SESSION_NAME

# # Array of exp_pretrain_epochs values
# EPOCHS_VALUES=(10 50 100 200)

# # Create panes and run commands
# for i in "${!EPOCHS_VALUES[@]}"; do
#     if [ $i -ne 0 ]; then
#         # Split window horizontally for additional commands, but adjust to split vertically if preferred
#         tmux split-window -v -t $SESSION_NAME
#         tmux select-layout -t $SESSION_NAME tiled > /dev/null
#     fi
#     # Pick a random device number between 0 and 5
#     DEVICE=$((RANDOM % 6))
#     # Form the command with the current exp_pretrain_epochs value and random device
#     CMD="source activate llm_gnn && python main.py --dataset ClinTox --exp_pretrain_epochs ${EPOCHS_VALUES[$i]} --device $DEVICE"
#     # Send the command to the pane
#     tmux send-keys -t ${i} "$CMD" C-m
# done

# # Ensure all panes are evenly distributed
# tmux select-layout -t $SESSION_NAME tiled

# # Attach to the session
# tmux attach -t $SESSION_NAME



#!/bin/bash

# Define session name
SESSION_NAME="para_sensi_pretrain_lr"

tmux has-session -t $SESSION_NAME 2>/dev/null
if [ $? -eq 0 ]; then
  echo "Session $SESSION_NAME already exists. Killing it."
  tmux kill-session -t $SESSION_NAME
fi

# Start a new tmux session
tmux new-session -d -s $SESSION_NAME

# Array of datasets
DATASETS=("ClinTox" "BBBP")

# Array of exp_pretrain_epochs values
EPOCHS_VALUES=(10 50 100 200)

# Pane counter to manage the pane indexing
PANE_COUNTER=0

for DATASET in "${DATASETS[@]}"; do
  for i in "${!EPOCHS_VALUES[@]}"; do
      if [ $PANE_COUNTER -ne 0 ]; then
          # Split window vertically for additional commands, adjust if needed
          tmux split-window -v -t $SESSION_NAME
          tmux select-layout -t $SESSION_NAME tiled > /dev/null
      fi
      # Pick a random device number between 0 and 5
      DEVICE=$((RANDOM % 6))
      # Form the command with the current dataset, exp_pretrain_epochs value, and random device
      CMD="source activate llm_gnn && python main.py --dataset $DATASET --exp_pretrain_epochs ${EPOCHS_VALUES[$i]} --device $DEVICE"
      # Send the command to the pane
      tmux send-keys -t ${PANE_COUNTER} "$CMD" C-m
      
      # Increment pane counter for next command
      PANE_COUNTER=$((PANE_COUNTER + 1))
  done
done

# Ensure all panes are evenly distributed
tmux select-layout -t $SESSION_NAME tiled

# Attach to the session
tmux attach -t $SESSION_NAME

