# #!/bin/bash

# # Start a new tmux session
# SESSION_NAME="ablation"

# # Check if the tmux session already exists, and delete it if it does
# tmux has-session -t $SESSION_NAME 2>/dev/null
# if [ $? -eq 0 ]; then
#   echo "Session $SESSION_NAME already exists. Killing it."
#   tmux kill-session -t $SESSION_NAME
# fi
# tmux new-session -d -s $SESSION_NAME

# # Array of ablation types including an empty option
# ABLA_TYPES=("" "--ablation_type np" "--ablation_type nf" "--ablation_type nt")

# # Create panes and run commands
# for i in {0..3}; do
#     if [ $i -ne 0 ]; then
#         # Split window horizontally for additional commands
#         tmux split-window -h -t $SESSION_NAME
#         tmux select-layout -t $SESSION_NAME tiled > /dev/null
#     fi
#     # Pick a random device number between 0 and 5
#     DEVICE=$((RANDOM % 6))
#     # Form the command
#     CMD="source activate llm_gnn && python main.py --dataset ClinTox ${ABLA_TYPES[$i]} --device $DEVICE"
#     # Send the command to the pane
#     tmux send-keys -t ${i} "$CMD" C-m
# done

# # Attach to the session
# tmux attach -t $SESSION_NAME

#!/bin/bash

# Start a new tmux session
SESSION_NAME="ablation"

# Check if the tmux session already exists, and delete it if it does
tmux has-session -t $SESSION_NAME 2>/dev/null
if [ $? -eq 0 ]; then
  echo "Session $SESSION_NAME already exists. Killing it."
  tmux kill-session -t $SESSION_NAME
fi
tmux new-session -d -s $SESSION_NAME

# Array of datasets
DATASETS=("ClinTox" "BBBP" "Mutagenicity")

# Array of ablation types including an empty option
# ABLA_TYPES=("" "--ablation_type np" "--ablation_type nf" "--ablation_type nt")
ABLA_TYPES=("--ablation_type nf")
# Counter for panes
PANE_COUNTER=0

for DATASET in "${DATASETS[@]}"; do
  for ABLA_TYPE in "${ABLA_TYPES[@]}"; do
    if [ $PANE_COUNTER -ne 0 ]; then
        # Split window horizontally for additional commands, but after the first command, start splitting vertically to manage space
        if [ $PANE_COUNTER -lt 4 ]; then
          tmux split-window -h -t $SESSION_NAME
        else
          tmux split-window -v -t $SESSION_NAME
        fi
        tmux select-layout -t $SESSION_NAME tiled > /dev/null
    fi
    # Pick a random device number between 0 and 5
    DEVICE=$((RANDOM % 6))
    # Form the command with the current dataset and ablation type
    CMD="source activate llm_gnn && python main.py --dataset $DATASET ${ABLA_TYPE} --device $DEVICE"
    # Send the command to the pane
    tmux send-keys -t ${PANE_COUNTER} "$CMD" C-m
    # Increment pane counter
    PANE_COUNTER=$((PANE_COUNTER + 1))
  done
done

# Attach to the session
tmux attach -t $SESSION_NAME

