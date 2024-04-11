SESSION_NAME="encoder_analysis"

tmux has-session -t $SESSION_NAME 2>/dev/null
if [ $? -eq 0 ]; then
  echo "Session $SESSION_NAME already exists. Killing it."
  tmux kill-session -t $SESSION_NAME
fi

# Start a new tmux session
tmux new-session -d -s $SESSION_NAME

# Array of datasets
DATASETS=("ClinTox" "BBBP" "AIDS")

# Array of exp_pretrain_epochs values
ENCODERS=("Bert" "DeBERTa" "Electra")

# Pane counter to manage the pane indexing
PANE_COUNTER=0

for DATASET in "${DATASETS[@]}"; do
  for ENCODER in "${ENCODERS[@]}"; do
      if [ $PANE_COUNTER -ne 0 ]; then
          # Split window vertically for additional commands, adjust if needed
          tmux split-window -v -t $SESSION_NAME
          tmux select-layout -t $SESSION_NAME tiled > /dev/null
      fi
      # Pick a random device number between 0 and 5
      DEVICE=$((RANDOM % 6))
      # Form the command with the current dataset, exp_pretrain_epochs value, and random device
      CMD="conda activate graph && python main.py --dataset $DATASET --exp_pretrain_epochs 100 --device $DEVICE -encoder $ENCODER"
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