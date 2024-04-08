# #!/bin/bash

# # Define session name
# SESSION_NAME="para_sensi_feedback_num"
# tmux has-session -t $SESSION_NAME 2>/dev/null
# if [ $? -eq 0 ]; then
#   echo "Session $SESSION_NAME already exists. Killing it."
#   tmux kill-session -t $SESSION_NAME
# fi
# # Start a new tmux session
# tmux new-session -d -s $SESSION_NAME

# # Array of exp_feedback_times values
# FEEDBACK_TIMES=(1 2 3 4 5)

# # Device numbers
# DEVICES=(0 1 2 3 4 5)

# # Keep track of the last used device to attempt not to reuse devices
# LAST_USED_DEVICE=-1

# # Create panes and run commands
# for i in "${!FEEDBACK_TIMES[@]}"; do
#     if [ $i -ne 0 ]; then
#         # Split window horizontally for additional commands, adjust if needed
#         tmux split-window -v -t $SESSION_NAME
#         tmux select-layout -t $SESSION_NAME tiled > /dev/null
#     fi
    
#     # Assign devices in a round-robin fashion
#     DEVICE_INDEX=$(( (i + LAST_USED_DEVICE + 1) % ${#DEVICES[@]} ))
#     DEVICE=${DEVICES[$DEVICE_INDEX]}
#     LAST_USED_DEVICE=$DEVICE_INDEX

#     # Form the command with the current exp_feedback_times value and selected device
#     CMD="source activate llm_gnn && python main.py --dataset ClinTox --exp_feedback_times ${FEEDBACK_TIMES[$i]} --device $DEVICE"
#     # Send the command to the pane
#     tmux send-keys -t ${i} "$CMD" C-m
# done

# # Ensure all panes are evenly distributed
# tmux select-layout -t $SESSION_NAME tiled

# # Attach to the session
# tmux attach -t $SESSION_NAME

#!/bin/bash

# # Define session name
# SESSION_NAME="para_sensi_feedback_num"
# tmux has-session -t $SESSION_NAME 2>/dev/null
# if [ $? -eq 0 ]; then
#   echo "Session $SESSION_NAME already exists. Killing it."
#   tmux kill-session -t $SESSION_NAME
# fi
# # Start a new tmux session
# tmux new-session -d -s $SESSION_NAME

# # Array of datasets
# DATASETS=("ClinTox" "BBBP")

# # Array of exp_feedback_times values
# FEEDBACK_TIMES=(1 2 3 4 5)

# # Device numbers
# DEVICES=(0 1 2 3 4 5)

# # Keep track of the last used device to attempt not to reuse devices
# LAST_USED_DEVICE=-1

# # Pane counter to keep track of which pane we're working on
# PANE_COUNTER=0

# for DATASET in "${DATASETS[@]}"; do
#   for i in "${!FEEDBACK_TIMES[@]}"; do
#       if [ $PANE_COUNTER -ne 0 ]; then
#           # Split window vertically for additional commands, adjust if needed
#           tmux split-window -v -t $SESSION_NAME
#           tmux select-layout -t $SESSION_NAME tiled > /dev/null
#       fi
      
#       # Assign devices in a round-robin fashion
#       DEVICE_INDEX=$(( (PANE_COUNTER + LAST_USED_DEVICE + 1) % ${#DEVICES[@]} ))
#       DEVICE=${DEVICES[$DEVICE_INDEX]}
#       LAST_USED_DEVICE=$DEVICE_INDEX

#       # Form the command with the current dataset, exp_feedback_times value, and selected device
#       CMD="source activate llm_gnn && python main.py --dataset $DATASET --exp_feedback_times ${FEEDBACK_TIMES[$i]} --device $DEVICE"
#       # Send the command to the pane
#       tmux send-keys -t ${PANE_COUNTER} "$CMD" C-m
      
#       # Increment pane counter
#       PANE_COUNTER=$((PANE_COUNTER + 1))
#   done
# done

# # Ensure all panes are evenly distributed
# tmux select-layout -t $SESSION_NAME tiled

# # Attach to the session
# tmux attach -t $SESSION_NAME


#!/bin/bash

# Define session name
SESSION_NAME="sub_feedback_num14"
tmux has-session -t $SESSION_NAME 2>/dev/null
if [ $? -eq 0 ]; then
  echo "Session $SESSION_NAME already exists. Killing it."
  tmux kill-session -t $SESSION_NAME
fi
# Start a new tmux session
tmux new-session -d -s $SESSION_NAME

# Array of datasets
DATASETS=("BBBP")

# Array of exp_feedback_times values
FEEDBACK_TIMES=(1 4)

# Device numbers limited to 0, 3, and 5
DEVICES=(0 3 4)

# Keep track of the last used device to attempt not to reuse devices
LAST_USED_DEVICE=-1

# Pane counter to keep track of which pane we're working on
PANE_COUNTER=0

for DATASET in "${DATASETS[@]}"; do
  for i in "${!FEEDBACK_TIMES[@]}"; do
      if [ $PANE_COUNTER -ne 0 ]; then
          # Split window vertically for additional commands, adjust if needed
          tmux split-window -v -t $SESSION_NAME
          tmux select-layout -t $SESSION_NAME tiled > /dev/null
      fi
      
      # Assign devices in a round-robin fashion from the modified DEVICES array
      DEVICE_INDEX=$(( (PANE_COUNTER + LAST_USED_DEVICE + 1) % ${#DEVICES[@]} ))
      DEVICE=${DEVICES[$DEVICE_INDEX]}
      LAST_USED_DEVICE=$DEVICE_INDEX

      # Form the command with the current dataset, exp_feedback_times value, and selected device
      CMD="source activate llm_gnn && python main.py --dataset $DATASET --exp_feedback_times ${FEEDBACK_TIMES[$i]} --device $DEVICE"
      # Send the command to the pane
      tmux send-keys -t ${PANE_COUNTER} "$CMD" C-m
      
      # Increment pane counter
      PANE_COUNTER=$((PANE_COUNTER + 1))
  done
done

# Ensure all panes are evenly distributed
tmux select-layout -t $SESSION_NAME tiled

# Attach to the session
tmux attach -t $SESSION_NAME
