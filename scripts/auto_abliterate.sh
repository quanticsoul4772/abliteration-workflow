#!/bin/bash
# Fully Automated Abliteration E2E Script
# Runs completely unattended, deploys to HuggingFace

set -e  # Exit on error

# Load configuration from .env if available
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Configuration (can be overridden by environment variables)
MODEL="${MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
OUTPUT_REPO="${OUTPUT_REPO:-rawcell/qwen-7b-abliterated}"
GPU_TIER="${GPU:-H200}"
NUM_GPUS="${NUM_GPUS:-1}"
TRIALS="${TRIALS:-200}"
DISK_SIZE="${DISK:-200}"
MAX_RUNTIME="${MAX_RUNTIME:-14400}"  # 4 hours max by default

# Validate required environment variables
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN not set. Add to .env file or export HF_TOKEN=..."
    exit 1
fi

if [ -z "$VAST_API_KEY" ]; then
    echo "ERROR: VAST_API_KEY not set. Add to .env file or export VAST_API_KEY=..."
    exit 1
fi

echo "╔════════════════════════════════════════════════════════════╗"
echo "║        AUTOMATED ABLITERATION - END TO END                ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Configuration:"
echo "  Model:      $MODEL"
echo "  Output:     https://huggingface.co/$OUTPUT_REPO"
echo "  GPU:        $NUM_GPUS x $GPU_TIER"
echo "  Trials:     $TRIALS"
echo "  Disk:       ${DISK_SIZE}GB"
echo ""

# Step 1: Create instance
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1/5: Creating GPU instance..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if ! uv run bruno-vast create $GPU_TIER $NUM_GPUS --disk $DISK_SIZE; then
    echo "ERROR: Failed to create instance"
    exit 1
fi

echo "✓ Instance created"
echo ""

# Step 2: Wait for instance to be ready
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2/5: Waiting for instance to be ready..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

sleep 60

# Wait for running status
MAX_WAIT=300  # 5 minutes
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if uv run bruno-vast list | grep -q "running"; then
        echo "✓ Instance running"
        break
    fi
    sleep 10
    WAITED=$((WAITED + 10))
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "ERROR: Instance did not start within 5 minutes"
    exit 1
fi

echo ""

# Step 3: Setup bruno
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3/5: Installing bruno on instance..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

uv run bruno-vast setup

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to setup bruno"
    exit 1
fi

echo "✓ Bruno installed"
sleep 5
echo ""

# Step 4: Start abliteration with auto-upload
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 4/5: Starting abliteration..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "This will take approximately 1.5-2 hours."
echo "The process will:"
echo "  1. Download model (~3 min)"
echo "  2. Extract refusal directions (~6 min)"
echo "  3. Run $TRIALS optimization trials (~50 min)"
echo "  4. Upload to HuggingFace (~4 min)"
echo ""
echo "You can monitor with: bruno-vast watch"
echo ""

# Start training in background with nohup
uv run bruno-vast exec "export HF_TOKEN=$HF_TOKEN && \
export HF_HOME=/workspace/.cache/huggingface && \
cd /workspace && \
nohup bruno \
  --model $MODEL \
  --cache-weights true \
  --compile \
  --n-trials $TRIALS \
  --storage sqlite:////workspace/bruno_study.db \
  --study-name auto-abliteration \
  --auto-select true \
  --auto-select-path /workspace/models \
  --hf-upload $OUTPUT_REPO \
  --unhelpfulness-prompts.config en \
  > /workspace/bruno.log 2>&1 &
echo \"Started PID: \$!\" && \
sleep 2 && \
tail -20 /workspace/bruno.log"

echo ""
echo "✓ Training started in background"
echo ""

# Step 5: Monitor and wait for completion
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 5/5: Monitoring progress..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Monitor loop
START_TIME=$(date +%s)
CHECK_INTERVAL=300  # 5 minutes

while true; do
    # Check timeout first
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    ELAPSED_MIN=$((ELAPSED / 60))

    if [ $ELAPSED -ge $MAX_RUNTIME ]; then
        echo ""
        echo "⚠ WARNING: Maximum runtime reached (${MAX_RUNTIME}s / $((MAX_RUNTIME/3600))h)"
        echo "  Training may still be running. Check manually: uv run bruno-vast watch"
        break
    fi

    # Check if bruno process is still running (robust pattern matching)
    # Use pgrep for more reliable process detection
    PROCESS_CHECK=$(uv run bruno-vast exec "pgrep -f 'bruno.*--model' >/dev/null 2>&1 && echo 'RUNNING' || echo 'NOTRUNNING'" 2>/dev/null || echo 'NOTRUNNING')

    if echo "$PROCESS_CHECK" | grep -q "NOTRUNNING"; then
        echo ""
        echo "✓ Training process completed"
        break
    fi

    # Get trial progress
    TRIAL_COUNT=$(uv run bruno-vast exec "grep -c 'Trial' /workspace/bruno.log 2>/dev/null || echo 0" 2>/dev/null || echo "?")

    # Show progress
    echo "[$(date +%H:%M:%S)] Elapsed: ${ELAPSED_MIN} min | Trials: ${TRIAL_COUNT}/${TRIALS}"

    # Sleep before next check
    sleep $CHECK_INTERVAL
done

echo ""

# Check results
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Checking results..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check if model was saved
MODEL_CHECK=$(uv run bruno-vast exec "ls -d /workspace/models/*abliterated* 2>/dev/null || echo 'NOTFOUND'")

if echo "$MODEL_CHECK" | grep -q "NOTFOUND"; then
    echo "⚠ WARNING: Model not found in /workspace/models/"
    echo "Checking logs for errors..."
    uv run bruno-vast exec "tail -50 /workspace/bruno.log"
    echo ""
    echo "ERROR: Training may have failed. Check logs manually."
    exit 1
fi

echo "✓ Model saved locally: $MODEL_CHECK"

# Check HuggingFace upload
echo ""
echo "Verifying HuggingFace upload..."
sleep 15  # Give HF time to process

# Retry up to 3 times with backoff
UPLOAD_SUCCESS=false
for i in 1 2 3; do
    if curl -s --max-time 30 "https://huggingface.co/api/models/$OUTPUT_REPO" 2>/dev/null | grep -q "modelId"; then
        UPLOAD_SUCCESS=true
        break
    fi
    echo "  Retry $i/3..."
    sleep $((i * 10))
done

if [ "$UPLOAD_SUCCESS" = true ]; then
    echo "✓ Upload successful!"
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║                    SUCCESS!                                ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Model deployed: https://huggingface.co/$OUTPUT_REPO"
    echo ""
    echo "You can now:"
    echo "  1. Use from phone via HuggingFace inference widget"
    echo "  2. Access via API: InferenceClient('$OUTPUT_REPO')"
    echo "  3. Download locally: huggingface-cli download $OUTPUT_REPO"
    echo ""
else
    echo "⚠ WARNING: HuggingFace upload verification failed"
    echo "Model may still be processing or upload failed"
    echo "Check manually: https://huggingface.co/$OUTPUT_REPO"
    echo ""
fi

# Destroy instance
echo "Destroying instance to stop billing..."
uv run bruno-vast stop

if [ $? -eq 0 ]; then
    echo "✓ Instance destroyed"
else
    echo "⚠ Failed to destroy instance automatically"
    echo "Destroy manually at: https://cloud.vast.ai/instances/"
fi

# Final summary
TOTAL_TIME=$((($(date +%s) - START_TIME) / 60))
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "COMPLETE - Total time: $TOTAL_TIME minutes"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
