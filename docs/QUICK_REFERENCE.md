# Bruno Quick Reference Card

Essential commands and flags for daily use.

## Common Commands

```bash
# Build & deploy
uv build
scp -P <PORT> dist/bruno_ai-*.whl root@<HOST>:/workspace/
uv run bruno-vast exec "pip install /workspace/bruno_ai-*.whl --force-reinstall"

# Instance management
uv run bruno-vast create <GPU_TYPE> <COUNT>
uv run bruno-vast list
uv run bruno-vast start
uv run bruno-vast stop
uv run bruno-vast terminate

# Monitoring
uv run bruno-vast watch          # Live terminal dashboard
uv run bruno-vast progress        # Quick status
uv run bruno-vast exec "tail -f /workspace/bruno.log"  # Raw logs

# Gradio Web Dashboard (NEW!)
uv run bruno-vast exec "cd /workspace && tmux new-session -d -s monitor 'python monitor_app.py --storage sqlite:///moonlight_reabliteration.db --study STUDY_NAME --share --port 7860'"
uv run bruno-vast exec "tmux capture-pane -t monitor -p | grep gradio.live"  # Get public URL

# Download results
uv run bruno-vast models
uv run bruno-vast download <MODEL_NAME>
```

## Run Commands by Model Size

### 7B Models
```bash
bruno --model Qwen/Qwen2.5-7B-Instruct \
  --auto-select true \
  --n-trials 50 \
  --compile
```

### 32B Models
```bash
bruno --model Qwen/Qwen2.5-Coder-32B-Instruct \
  --auto-select true \
  --cache-weights true \
  --unhelpfulness-prompts.config en \
  --storage sqlite:///study.db \
  --study-name qwen32b \
  --n-trials 200
```

### 70B Models
```bash
bruno --model Qwen/Qwen2.5-70B-Instruct \
  --auto-select true \
  --cache-weights false \
  --unhelpfulness-prompts.config en \
  --batch-size 1 \
  --max-batch-size 16 \
  --storage sqlite:///study.db \
  --study-name qwen70b \
  --n-trials 200
```

## Critical Flags

| Flag | When to Use |
|------|-------------|
| `--cache-weights true` | **Default for all models** (v1.2.0+) |
| `--cache-weights false` | **Only for H100 80GB with 32B models** |
| `--unhelpfulness-prompts.config en` | **Required for 32B+** (C4 dataset) |
| `--auto-select true` | Recommended (auto-saves best trial) |
| `--storage sqlite:///study.db` | Recommended (resume support) |
| `--compile` | Optional (1.5-2x inference speedup) |

## Model Size Guidelines

**v1.2.0+ with Layer-Wise Caching:**

| Size | GPU | Disk | Cache Weights | C4 Config |
|------|-----|------|---------------|-----------|
| 7B | 24GB | 100GB | ✅ Yes | ❌ No |
| 13B | 24GB | 150GB | ✅ Yes | ❌ No |
| 32B | **H200 141GB** | **200GB** | ✅ Yes | ✅ Yes |
| 32B | H100 80GB | **200GB** | ❌ No | ✅ Yes |
| 70B | 80GB+ | **300GB** | ❌ No | ✅ Yes |

**Note:** v1.1.0+ streams C4 on-demand (~0GB overhead). Network required during dataset loading.

**Note:** v1.2.0+ uses layer-wise caching (55-75% less memory), enabling caching for 32B models on H200.

## Troubleshooting

```bash
# Process not running?
uv run bruno-vast exec "ps aux | grep 'bruno --model'"

# Check errors
uv run bruno-vast exec "tail -100 /workspace/bruno.log | grep -i error"

# GPU OOM?
uv run bruno-vast exec "dmesg | grep -i 'out of memory'"

# Clear cache after reinstall
uv run bruno-vast exec "find /usr/local/lib/python3.*/dist-packages/bruno -name '*.pyc' -delete"

# Restart fresh
uv run bruno-vast exec "pkill -f bruno && rm /workspace/bruno.log"
```

## Performance Stats (H200)

| Operation | Time | Notes |
|-----------|------|-------|
| Model download (32B) | ~5 min | One-time, cached |
| PCA extraction (32B) | **~5 min** | GPU optimized (was 4-6 hrs) |
| Single trial (32B) | ~2 min | With layer-wise caching |
| 200 trials (32B) | ~9-11 hrs | Total cost: ~$19-24 |

## Cost Quick Math

```
Cost = GPU_rate × (setup_time + trial_time × n_trials)

Example (32B on H200 @ $2.14/hr with caching):
= $2.14 × (0.5hr + 0.033hr × 200)
= $2.14 × 7.1hr
= $15.19

Example (32B on H200 @ $2.14/hr without caching):
= $2.14 × (0.5hr + 0.05hr × 200)
= $2.14 × 10.5hr
= $22.47
```

## Git Workflow

```bash
# Update docs
git add CLAUDE.md docs/
git commit -m "Document <change>"

# Push to fork (abliteration-workflow)
git push fork master

# NEVER push to main bruno repo
# git push origin master  # ❌ WRONG
```

## Gradio Monitor Dashboard

Real-time web dashboard for monitoring abliteration progress.

### CRITICAL: The Correct Way to Launch Gradio with Share URL

**Why this is tricky:** Gradio prints the share URL to stdout, but background processes (`nohup`, `&`) don't capture it properly. The ONLY reliable way is to run in tmux and capture the pane output.

**Step 1: Kill any existing monitor processes first**
```bash
uv run bruno-vast exec "pkill -f monitor_app; pkill -f frpc; tmux kill-session -t monitor 2>/dev/null; sleep 3"
```

**Step 2: Start monitor in tmux with share enabled**
```bash
uv run bruno-vast exec "cd /workspace && tmux new-session -d -s monitor 'python monitor_app.py --storage sqlite:///YOUR_STUDY.db --study YOUR_STUDY_NAME --target-trials 300 --share --port 7860 2>&1'"
```

**Step 3: Wait 30 seconds for Gradio to initialize the tunnel, then capture the URL**
```bash
uv run bruno-vast exec "sleep 30 && tmux capture-pane -t monitor -p -S -200 | grep gradio.live"
```

**One-liner (recommended):**
```bash
uv run bruno-vast exec "pkill -f monitor_app; pkill -f frpc; tmux kill-session -t monitor 2>/dev/null; sleep 2; cd /workspace && tmux new-session -d -s monitor 'python monitor_app.py --storage sqlite:///YOUR_STUDY.db --study YOUR_STUDY_NAME --target-trials 300 --share --port 7860 2>&1' && sleep 30 && tmux capture-pane -t monitor -p -S -200 | grep gradio.live"
```

### Common Mistakes (DON'T DO THESE)

```bash
# WRONG: nohup loses the share URL output
nohup python monitor_app.py --share > log.txt 2>&1 &

# WRONG: Background process loses stdout
python monitor_app.py --share &

# WRONG: Trying to grep frpc process args (unreliable)
ps aux | grep frpc  # The URL format changes!

# WRONG: Not waiting long enough (need 30 seconds)
sleep 10 && tmux capture-pane...  # Too short!
```

### Why 30 Seconds?

Gradio share URL setup requires:
1. Local server startup (~2s)
2. API call to gradio.app (~3s)
3. FRPC tunnel establishment (~10-20s)
4. URL printed to stdout (~1s)

Total: ~20-25 seconds minimum. Use 30 seconds to be safe.

**Features:**
- Real-time trial progress visualization
- Interactive Plotly charts (optimization history, Pareto front)
- Parameter importance analysis
- Trial comparison and timeline
- Auto-refresh every 30 seconds

## Emergency Stops

```bash
# Stop instance (KILLS process)
uv run bruno-vast stop

# Kill process only (keeps instance)
uv run bruno-vast exec "pkill -f 'bruno --model'"

# Download before stopping
uv run bruno-vast models
uv run bruno-vast download <MODEL>
uv run bruno-vast stop
```

## Successful Abliteration Results

### Qwen2.5-Coder-32B Trial 173 (Best Result)
- **Date:** 2026-02-01
- **GPU:** H200 141GB on Vast.ai
- **Result:** **0 refusals, KL=0.26**
- **Output:** `Qwen2.5-Coder-32B-trial173` (65GB)
