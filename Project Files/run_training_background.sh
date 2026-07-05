#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Run R-LLaVA training as a detached background process
# Works WITH tmux (preferred) or falls back to nohup if tmux unavailable
# Survives SSH disconnects, browser closes, and Jupyter kernel issues
# ═══════════════════════════════════════════════════════════════════════════

# Auto-detect home directory — works regardless of username
HOME_DIR="$HOME"

NOTEBOOK="$HOME_DIR/dgx_rllava_stage1_and_2.ipynb"
OUTPUT_NB="$HOME_DIR/dgx_rllava_output.ipynb"
LOG_FILE="$HOME_DIR/training.log"
PID_FILE="$HOME_DIR/training.pid"
VENV_PYTHON="$HOME_DIR/nutriderm_env/bin/python"
KERNEL_NAME="nutriderm"

echo "═══════════════════════════════════════════════════════════"
echo " Config check"
echo "═══════════════════════════════════════════════════════════"
echo "  Home dir   : $HOME_DIR"
echo "  Notebook   : $NOTEBOOK"
echo "  Venv python: $VENV_PYTHON"
echo ""

# Verify required files exist before starting
if [ ! -f "$NOTEBOOK" ]; then
    echo "ERROR: Notebook not found at $NOTEBOOK"
    echo "Upload dgx_rllava_stage1_and_2.ipynb to $HOME_DIR first."
    exit 1
fi

if [ ! -f "$VENV_PYTHON" ]; then
    echo "ERROR: Venv python not found at $VENV_PYTHON"
    echo "Check your venv location — update VENV_PYTHON in this script if different."
    exit 1
fi

CMD="$VENV_PYTHON -m jupyter nbconvert --to notebook --execute \
   --ExecutePreprocessor.timeout=-1 \
   --ExecutePreprocessor.kernel_name=$KERNEL_NAME \
   --output $OUTPUT_NB \
   $NOTEBOOK"

echo "═══════════════════════════════════════════════════════════"

if command -v tmux &> /dev/null; then
    echo " Starting training in detached tmux session"
    echo "═══════════════════════════════════════════════════════════"
    tmux kill-session -t rllava_train 2>/dev/null
    tmux new-session -d -s rllava_train \
      "$CMD 2>&1 | tee $LOG_FILE"
    sleep 3
    if tmux has-session -t rllava_train 2>/dev/null; then
        echo "Training started in tmux session: rllava_train"
        echo ""
        echo "  Monitor : tail -f $LOG_FILE"
        echo "  Attach  : tmux attach -t rllava_train"
        echo "  Detach  : Ctrl+B then D"
        echo "  Stop    : tmux kill-session -t rllava_train"
    else
        echo "tmux session failed to start — check tmux installation"
    fi
else
    echo " tmux not found — using nohup instead"
    echo "═══════════════════════════════════════════════════════════"
    if [ -f "$PID_FILE" ]; then
        OLD_PID=$(cat "$PID_FILE")
        kill -9 "$OLD_PID" 2>/dev/null
    fi
    nohup $CMD > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    sleep 2
    if ps -p $(cat "$PID_FILE") > /dev/null 2>&1; then
        echo "Training started with PID: $(cat $PID_FILE)"
        echo ""
        echo "  Monitor      : tail -f $LOG_FILE"
        echo "  Check alive  : ps -p \$(cat $PID_FILE)"
        echo "  Stop         : kill -9 \$(cat $PID_FILE)"
    else
        echo "FAILED to start. Check $LOG_FILE for errors:"
        tail -30 "$LOG_FILE"
    fi
fi

echo ""
echo "You can now close SSH / browser — training continues running."
echo "Output notebook (with all cell results) will be saved to:"
echo "  $OUTPUT_NB"
