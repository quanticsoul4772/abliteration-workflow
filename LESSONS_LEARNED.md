# Lessons Learned

## Multi-GPU Issues

### GPU 0 overloaded, others idle
`device_map="auto"` fills GPU 0 first. Use `device_map="balanced"` for even distribution.

### Meta device errors during trials
`cache_weights=true` fails with `device_map="balanced"`. Set `cache_weights=false` for multi-GPU.

### OOM during trial evaluation
`iterative_rounds > 0` stores hidden states for all layers. Set `iterative_rounds=0` for 32B+ on 24GB GPUs.

### Slow residual extraction
`batch_size=1` is 4x slower than needed. Use `batch_size=4` for 32B models.

### PCA extraction hangs on large models
Resolved in v1.0.1+ with GPU-accelerated PCA. On older versions, set `use_pca_extraction=false`.

### Multiple processes competing for GPU
Always check before starting:
```bash
ps aux | grep bruno
nvidia-smi | grep python
pkill -9 -f bruno  # if needed
```

## Configuration by Setup

### Single GPU (A100/H200)
```toml
device_map = "auto"
cache_weights = true
iterative_rounds = 1
batch_size = 0
n_trials = 200
storage = "sqlite:///bruno_study.db"
study_name = "experiment1"
auto_select = true
auto_select_path = "/workspace/models/model-bruno"
```

### Multi-GPU (4x RTX 4090)
```toml
device_map = "balanced"
cache_weights = false
iterative_rounds = 0
batch_size = 4
max_batch_size = 16
n_trials = 200
storage = "sqlite:///bruno_study.db"
study_name = "experiment1"
auto_select = true
auto_select_path = "/workspace/models/model-bruno"
```

### 7B on single GPU
```toml
device_map = "auto"
cache_weights = true
iterative_rounds = 2
batch_size = 0
n_trials = 50
auto_select = true
```

## 32B Model GPU Selection

| GPU | cache_weights | Reload Time | 200 Trials |
|-----|---------------|-------------|------------|
| H200 141GB | yes | 10-15s | ~9-11 hrs |
| H100 80GB | no | 60-120s | ~13-15 hrs |
| 4x RTX 4090 | no | 30s | ~30-35 hrs |

H200 fits model weights (~64GB) plus layer-wise cache (~28GB). H100 does not.

## Pre-Flight Checklist (32B Models)

```bash
# 1. Check GPUs
nvidia-smi --query-gpu=name,memory.total --format=csv

# 2. Verify nothing running
ps aux | grep python
nvidia-smi | grep python

# 3. Start
cd /workspace && nohup bruno > bruno.log 2>&1 &

# 4. Verify GPU distribution after model loads (~5 min)
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
# Multi-GPU: ~15-17GB each. Single GPU: ~64GB.
```

## Successful Results

### Moonlight-16B-A3B-Instruct-Bruno
- **HuggingFace:** [rawcell/Moonlight-16B-A3B-Instruct-bruno](https://huggingface.co/rawcell/Moonlight-16B-A3B-Instruct-bruno)
- **Refusal reduction:** 100% to 41% (59% reduction)
- **Benchmarks vs previous:** MMLU 48.7% (+0.7%), HellaSwag 58.0% (+2.0%), GSM8K 55.0% (+4.0%)

### Qwen2.5-Coder-32B Trial 173
- **GPU:** H200 141GB
- **Result:** 0 refusals, KL=0.26 (trial 173 of 200)
