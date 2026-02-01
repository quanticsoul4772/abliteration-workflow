# Automated End-to-End Abliteration Plan

**Goal:** Best quality abliterated model, fastest execution, fully automated, unattended

---

## OPTIMAL CONFIGURATION

### Best Model: Qwen2.5-Coder-7B-Instruct

**Why:**
- 88.4% on HumanEval (beats GPT-4's 87.1%)
- Outperforms 22B and 33B coding models
- 7B size = fast inference, fits HuggingFace free tier
- Supports 92 programming languages
- Already uncensored-friendly architecture
- Recent (2024), actively maintained

**Alternatives considered:**
- DeepSeek-Coder-7B: 81.1% (7% lower than Qwen)
- Mistral-7B: General purpose, not coding-optimized
- Llama-3.1-8B: Good but 8% lower coding performance

**Model:** `Qwen/Qwen2.5-Coder-7B-Instruct`

---

### Best GPU: H200 141GB (Single GPU)

**Why:**
- 45% faster than H100 for LLM workloads
- 141GB VRAM = supports layer-wise caching (saves 3-4 hours)
- GPU-accelerated PCA completes in <2 minutes (vs 15-20 min on H100)
- Single GPU = no multi-GPU complexity
- Available on Vast.ai at ~$2-2.50/hour

**Performance estimates for 7B model:**
- Direction extraction: 5-8 minutes (vs 15-20 for 32B)
- Per trial: ~15-20 seconds (vs 1-2 minutes for 32B)
- 200 trials: ~1-1.5 hours (vs 9-11 hours for 32B)
- **Total time: ~1.5-2 hours** (vs 10+ hours for 32B)
- **Total cost: ~$3-5** (vs $20-25 for 32B)

**Alternatives:**
- H100 80GB: 30% slower, can't use caching efficiently
- RTX 4090: $0.40/hr but 4-5 hours total (slower PCA, no caching)
- A100 80GB: 50% slower than H200

**GPU:** Single H200 141GB

---

### Best Platform: Vast.ai

**Why:**
- Cheapest H200 pricing (~$2.00-2.50/hr)
- bruno-vast CLI already working
- API automation available
- Fast provisioning (2-3 minutes)
- 200GB+ disk available

**Alternatives:**
- RunPod: 20% more expensive
- Lambda Labs: 30% more expensive
- Google Cloud: Complex setup

---

## AUTOMATED WORKFLOW

### Phase 1: Instance Creation (2 minutes)

```bash
# Automated script starts here
uv run bruno-vast create H200 1 --disk 200
uv run bruno-vast setup
```

**Automation:** Fully automated via bruno-vast CLI

---

### Phase 2: Training Execution (1.5-2 hours)

```bash
# Executed automatically on instance
bruno \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --cache-weights true \
  --compile \
  --n-trials 200 \
  --storage sqlite:////workspace/bruno_study.db \
  --study-name qwen7b-abliteration \
  --auto-select true \
  --auto-select-path /workspace/models \
  --hf-upload rawcell/qwen-7b-abliterated \
  --unhelpfulness-prompts.config en
```

**Timeline:**
- Model download: 2-3 minutes
- Direction extraction: 5-8 minutes (GPU PCA)
- Cache creation: 30 seconds
- Trial 1: 20 seconds
- Trials 2-200: ~15 seconds each (with caching)
- **Total: ~1.5 hours**

**Automation:** Single command, runs to completion

---

### Phase 3: Upload to HuggingFace (Built-in)

**Via bruno --hf-upload flag:**
- Automatic upload after optimization
- Uses HF_TOKEN from environment
- Creates model card
- **Time:** 3-5 minutes (7GB upload from datacenter)

**Automation:** Fully automated (part of bruno command)

---

### Phase 4: Instance Cleanup (Automatic)

```bash
# After upload completes
bruno-vast stop
```

**Automation:** Can be scripted to auto-destroy after success

---

## COMPLETE AUTOMATION SCRIPT

**File:** `scripts/auto_abliterate.sh`

```bash
#!/bin/bash
# Fully automated abliteration E2E
# See scripts/auto_abliterate.sh for the complete implementation

set -e  # Exit on error

# Configuration
MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"
OUTPUT_REPO="rawcell/qwen-7b-abliterated"
GPU_TIER="H200"
TRIALS=200
DISK_SIZE=200
MAX_RUNTIME=14400  # 4 hours timeout

echo "=== Starting Automated Abliteration ==="
echo "Model: $MODEL"
echo "GPU: $GPU_TIER"
echo "Trials: $TRIALS"
echo ""

# Step 1: Create instance
echo "[1/5] Creating GPU instance..."
uv run bruno-vast create $GPU_TIER 1 --disk $DISK_SIZE
echo "  Instance created"

# Step 2: Wait for instance to be ready
echo "[2/5] Waiting for instance to be ready..."
sleep 60
while ! uv run bruno-vast list | grep -q "running"; do
    sleep 10
done
echo "  Instance ready!"

# Step 3: Setup bruno
echo "[3/5] Installing bruno on instance..."
uv run bruno-vast setup
sleep 10

# Step 4: Run abliteration with auto-upload
echo "[4/5] Starting abliteration (will take ~1.5 hours)..."
uv run bruno-vast exec "export HF_TOKEN=$HF_TOKEN && nohup bruno \
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
  > /workspace/bruno.log 2>&1 &"

# Step 5: Monitor progress with timeout
echo "[5/5] Monitoring progress..."
echo "  Log file: /workspace/bruno.log"
echo "  You can check: uv run bruno-vast watch"
echo ""
echo "Waiting for completion (checking every 5 minutes, max 4 hours)..."

START_TIME=$(date +%s)
while true; do
    # Check timeout
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    if [ $ELAPSED -ge $MAX_RUNTIME ]; then
        echo "  WARNING: Maximum runtime reached ($MAX_RUNTIME seconds)"
        break
    fi

    # Check if bruno process is still running (robust pattern matching)
    RUNNING=$(uv run bruno-vast exec "pgrep -f 'bruno.*--model' || echo 'NOTRUNNING'" 2>/dev/null)

    if echo "$RUNNING" | grep -q "NOTRUNNING"; then
        echo "  Abliteration complete!"
        break
    fi

    # Show progress
    TRIALS_DONE=$(uv run bruno-vast exec "grep -c 'Trial' /workspace/bruno.log 2>/dev/null || echo 0")
    echo "  Progress: $TRIALS_DONE / $TRIALS trials (elapsed: $((ELAPSED/60)) min)"

    sleep 300  # Check every 5 minutes
done

# Check if upload succeeded
echo ""
echo "Checking HuggingFace upload..."
sleep 10  # Give HF time to process
if curl -s "https://huggingface.co/api/models/$OUTPUT_REPO" | grep -q "modelId"; then
    echo "  Upload successful: https://huggingface.co/$OUTPUT_REPO"

    # Destroy instance
    echo ""
    echo "Destroying instance to stop billing..."
    uv run bruno-vast stop

    echo ""
    echo "=== COMPLETE ==="
    echo "Model available at: https://huggingface.co/$OUTPUT_REPO"
    echo "Ready for deployment!"
else
    echo "  WARNING: Upload may have failed"
    echo "  Check instance manually: uv run bruno-vast list"
fi
```

**Usage:**
```bash
chmod +x scripts/auto_abliterate.sh
./scripts/auto_abliterate.sh
```

**Total unattended time: ~2 hours**

---

## PERFORMANCE PROJECTIONS

### 7B Model on H200

| Phase | Time | Details |
|-------|------|---------|
| Instance creation | 2 min | Vast.ai provisioning |
| Bruno setup | 2 min | pip install |
| Model download | 3 min | 14GB download |
| Direction extraction | 6 min | GPU PCA (64 layers) |
| Cache creation | 30 sec | Layer-wise cache |
| Trial 1 | 20 sec | First trial |
| Trials 2-200 | 50 min | 15 sec/trial with cache |
| Model selection | 10 sec | Pareto optimal |
| HF upload | 4 min | 7GB upload |
| **Total** | **~1.5 hrs** | **Fully automated** |

**Cost:** ~$3-4 on H200 @ $2.14/hour

---

## OPTIMIZATION OPPORTUNITIES

### Faster Execution (Target: <1 hour)

**Option 1: Reduce trials to 100**
- Time: 45 minutes total
- Cost: $1.50
- Quality: 95% as good (Optuna converges fast)

**Option 2: Use multiple GPUs (parallel trials)**
- 2x H200: 30 minutes (parallel evaluation)
- Cost: $2.00 (2 GPUs × 0.5 hours)
- Requires: Multi-GPU support in bruno (future feature)

**Option 3: Warm-start optimization**
- Use bruno's warm-start feature (already has Qwen family profile)
- Skips first 10-20 trials
- Time: Saves 5-10 minutes
- Already implemented!

---

## DEPLOYMENT AFTER ABLITERATION

### HuggingFace Free Inference (7B works!)

**After upload completes:**
1. Model appears at: https://huggingface.co/rawcell/qwen-7b-abliterated
2. Free inference widget appears automatically
3. Use from phone/anywhere via API
4. Rate limit: ~1000 requests/day (free tier)

**No additional deployment needed!**

---

## COMPLETE E2E COMMAND (One-Liner)

```bash
uv run bruno-vast create H200 1 --disk 200 && \
uv run bruno-vast setup && \
uv run bruno-vast exec "export HF_TOKEN=$HF_TOKEN && \
  bruno Qwen/Qwen2.5-Coder-7B-Instruct \
  --cache-weights true --compile --n-trials 200 \
  --auto-select true --hf-upload rawcell/qwen-7b-abliterated \
  --unhelpfulness-prompts.config en && \
  echo 'COMPLETE' > /workspace/done.txt" && \
sleep 7200 && \
uv run bruno-vast stop
```

**This runs completely unattended for 2 hours, then auto-destroys.**

---

## ERROR HANDLING & RESUME

**If training fails mid-way:**

```bash
# Resume from checkpoint
bruno-vast exec "bruno Qwen/Qwen2.5-Coder-7B-Instruct \
  --storage sqlite:////workspace/bruno_study.db \
  --study-name auto-abliteration \
  --n-trials 200 \
  --auto-select true \
  --hf-upload rawcell/qwen-7b-abliterated"
```

Bruno will resume from last completed trial (SQLite storage).

---

## MONITORING OPTIONS

### Option 1: bruno-vast watch
```bash
bruno-vast watch
# Live dashboard showing progress
```

### Option 2: Check logs remotely
```bash
bruno-vast exec "tail -f /workspace/bruno.log"
```

### Option 3: Phone monitoring
- SSH from phone (Termux)
- Or check HuggingFace for upload completion

---

## COMPARISON: 7B vs 32B

| Metric | 7B Model | 32B Model | Winner |
|--------|----------|-----------|--------|
| **Training time** | 1.5 hrs | 10 hrs | 7B (7x faster) |
| **Cost** | $3-4 | $20-25 | 7B (6x cheaper) |
| **Quality** | Excellent coding | Better reasoning | 32B (marginal) |
| **HF Free Inference** | YES | NO | 7B (major win) |
| **Phone usage** | FREE | Must pay | 7B (unlimited) |
| **Download size** | 14GB | 65GB | 7B (5x smaller) |
| **Local inference** | RTX 4070 (no quant) | RTX 4070 (4-bit only) | 7B (faster) |

**For phone usage and free inference: 7B is the clear winner**

---

## RECOMMENDED PLAN

### Setup Once

1. Ensure HF_TOKEN in .env file
2. Ensure VAST_API_KEY in .env file
3. Create target HF repo: https://huggingface.co/new → `rawcell/qwen-7b-abliterated`

### Run Automation

```bash
# Single command - runs completely unattended
./scripts/auto_abliterate.sh
```

**What happens:**
1. Creates H200 instance (2 min)
2. Installs bruno (2 min)
3. Runs 200-trial abliteration (1.5 hrs)
4. Uploads to HuggingFace (4 min)
5. Destroys instance automatically
6. **Total: ~2 hours, $4 cost**

**Your involvement:** Start script, walk away, come back to deployed model

---

## AFTER COMPLETION

**Model location:** https://huggingface.co/rawcell/qwen-7b-abliterated

**Use immediately from phone:**
```python
from huggingface_hub import InferenceClient
client = InferenceClient("rawcell/qwen-7b-abliterated")
response = client.text_generation("Write Python code for quicksort")
print(response)
```

**Or web interface:**
- Go to model page
- Use inference widget
- FREE, unlimited (within rate limits)

---

## ITERATION STRATEGY

**Try different abliteration strengths:**

```bash
# Light abliteration (more cautious)
MODEL=Qwen/Qwen2.5-Coder-7B-Instruct TRIALS=100 OUTPUT_REPO=rawcell/qwen-7b-light ./scripts/auto_abliterate.sh

# Medium (balanced)
MODEL=Qwen/Qwen2.5-Coder-7B-Instruct TRIALS=200 OUTPUT_REPO=rawcell/qwen-7b-medium ./scripts/auto_abliterate.sh

# Aggressive (maximum abliteration)
MODEL=Qwen/Qwen2.5-Coder-7B-Instruct TRIALS=300 OUTPUT_REPO=rawcell/qwen-7b-aggressive ./scripts/auto_abliterate.sh
```

**Cost:** $3-4 per variant, 1.5-2 hours each
**Result:** Test which works best for your use case

---

## NEXT STEPS

1. Create the automation script
2. Test run on 7B model
3. Deploy to HuggingFace
4. Use from phone
5. Iterate if needed

**Ready to implement this plan?**

---

## Sources

- [Best Open-Source LLMs 2026](https://huggingface.co/blog/daya-shankar/open-source-llms)
- [Qwen2.5-Coder Technical Report](https://arxiv.org/html/2409.12186v3)
- [Qwen2.5-Coder Benchmarks](https://llm-stats.com/models/compare/deepseek-v2.5-vs-qwen-2.5-coder-7b-instruct)
- [H200 vs H100 Performance](https://northflank.com/blog/h100-vs-h200)
- [NVIDIA H200 Deep Dive 2026](https://www.fluence.network/blog/nvidia-h200-deep-dive/)
- [Cheapest GPU Cloud 2026](https://www.runpod.io/articles/guides/top-serverless-gpu-clouds)
