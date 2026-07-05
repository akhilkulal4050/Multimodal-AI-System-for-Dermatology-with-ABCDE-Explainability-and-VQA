#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Run R-LLaVA training as a detached background process
# Works WITH tmux (preferred) or falls back to nohup if tmux unavailable
# Survives SSH disconnects, browser closes, and Jupyter kernel issues
# ═══════════════════════════════════════════════════════════════════════════

# ── Paths — set explicitly since venv and files are in different locations ──
FILES_DIR="/home/user_name"
VENV_DIR="/root/nutriderm_env"

NOTEBOOK="$FILES_DIR/dgx_rllava_stage1_and_2.ipynb"
OUTPUT_NB="$FILES_DIR/dgx_rllava_output.ipynb"
LOG_FILE="$FILES_DIR/training.log"
PID_FILE="$FILES_DIR/training.pid"
VENV_PYTHON="$VENV_DIR/bin/python"
KERNEL_NAME="nutriderm"

echo "═══════════════════════════════════════════════════════════"
echo " Config check"
echo "═══════════════════════════════════════════════════════════"
echo "  Files dir  : $FILES_DIR"
echo "  Venv dir   : $VENV_DIR"
echo "  Notebook   : $NOTEBOOK"
echo "  Venv python: $VENV_PYTHON"
echo ""

# Verify required files exist before starting
if [ ! -f "$NOTEBOOK" ]; then
    echo "ERROR: Notebook not found at $NOTEBOOK"
    echo "Upload dgx_rllava_stage1_and_2.ipynb to $FILES_DIR first."
    exit 1
fi

if [ ! -f "$VENV_PYTHON" ]; then
    echo "ERROR: Venv python not found at $VENV_PYTHON"
    echo "Check VENV_DIR at the top of this script."
    exit 1
fi

# Verify the kernel is registered for this venv
if ! "$VENV_PYTHON" -m jupyter kernelspec list 2>/dev/null | grep -q "$KERNEL_NAME"; then
    echo "WARNING: Kernel '$KERNEL_NAME' not found in registered kernels."
    echo "Registered kernels:"
    "$VENV_PYTHON" -m jupyter kernelspec list 2>/dev/null
    echo ""
    echo "Register it with:"
    echo "  $VENV_PYTHON -m ipykernel install --user --name $KERNEL_NAME --display-name 'NutriDerm (venv)'"
    echo ""
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
