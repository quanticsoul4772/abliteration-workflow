# Quick Reference

## Instance Management

```bash
uv run bruno-vast create <GPU_TYPE> <COUNT>
uv run bruno-vast list
uv run bruno-vast setup
uv run bruno-vast watch
uv run bruno-vast stop
uv run bruno-vast terminate
```

## Run by Model Size

```bash
# 7B
bruno Qwen/Qwen2.5-7B-Instruct --auto-select --n-trials 50 --compile

# 32B
bruno Qwen/Qwen2.5-Coder-32B-Instruct \
  --auto-select --cache-weights true --unhelpfulness-prompts.config en \
  --storage sqlite:///study.db --study-name qwen32b --n-trials 200

# 70B
bruno Qwen/Qwen2.5-70B-Instruct \
  --auto-select --cache-weights false --unhelpfulness-prompts.config en \
  --batch-size 1 --storage sqlite:///study.db --study-name qwen70b --n-trials 200
```

## Monitoring

```bash
# Live dashboard
uv run bruno-vast watch

# Tail logs
uv run bruno-vast exec "tail -f /workspace/bruno.log"

# Trial count
uv run bruno-vast exec "cd /workspace && python3 -c \"import optuna; s=optuna.load_study(study_name='STUDY', storage='sqlite:///DB.db'); print(len([t for t in s.trials if t.state.name=='COMPLETE']))\""
```

## Checking Best Trials

`values[1]` is a normalized ratio, not the refusal count. Use `user_attrs['refusals']`:

```python
import optuna
study = optuna.load_study(study_name='STUDY', storage='sqlite:///DB.db')
trials = [(t.number, t.values[0], t.user_attrs.get("refusals", 999))
          for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
trials.sort(key=lambda x: (x[2], x[1]))
for num, kl, ref in trials[:10]:
    print(f'Trial {num}: KL={kl:.4f}, Refusals={ref}')
```

## Gradio Monitor

```bash
# Start in tmux (required to capture share URL)
uv run bruno-vast exec "pkill -f monitor_app; tmux kill-session -t monitor 2>/dev/null; sleep 2; \
  cd /workspace && tmux new-session -d -s monitor \
  'python monitor_app.py --storage sqlite:///DB.db --study STUDY --share --port 7860 2>&1'"

# Get share URL (wait 30s for tunnel)
uv run bruno-vast exec "sleep 30 && tmux capture-pane -t monitor -p -S -200 | grep gradio.live"
```

## Download & Stop

```bash
uv run bruno-vast models
uv run bruno-vast download <MODEL>
uv run bruno-vast stop
```

## Build & Deploy

```bash
uv build
scp -P <PORT> dist/bruno_ai-*.whl root@<HOST>:/workspace/
uv run bruno-vast exec "pip install /workspace/bruno_ai-*.whl --force-reinstall"
```

## Benchmark Comparison

```bash
python scripts/benchmark_compare.py \
  --model-a Qwen/Qwen2.5-7B-Instruct \
  --model-b ./models/Qwen2.5-7B-Instruct-bruno \
  --output results.json

# With 4-bit quantization for limited VRAM
python scripts/benchmark_compare.py \
  --model-a original --model-b abliterated --quantize-4bit
```

## Troubleshooting

```bash
# Check if running
uv run bruno-vast exec "ps aux | grep 'bruno --model'"

# Check errors
uv run bruno-vast exec "tail -100 /workspace/bruno.log | grep -i error"

# GPU OOM
uv run bruno-vast exec "dmesg | grep -i 'out of memory'"

# Clear cache after reinstall
uv run bruno-vast exec "find /usr/local/lib/python3.*/dist-packages/bruno -name '*.pyc' -delete"

# Kill and restart
uv run bruno-vast exec "pkill -f bruno && rm /workspace/bruno.log"
```

## Critical Flags

| Flag | When to Use |
|------|-------------|
| `--cache-weights true` | Default for all models (v1.2.0+) |
| `--cache-weights false` | H100 80GB with 32B models |
| `--unhelpfulness-prompts.config en` | Required for 32B+ (C4 dataset) |
| `--auto-select` | Auto-save best trial |
| `--storage sqlite:///study.db` | Resume support |
| `--compile` | torch.compile() speedup |

## Performance Stats (H200, 32B model)

| Operation | Time |
|-----------|------|
| Model download | ~5 min (one-time) |
| PCA extraction | ~5 min |
| Single trial | ~2 min (with caching) |
| 200 trials | ~9-11 hrs |

## Cost Estimation

```
Cost = GPU_rate * (setup_time + trial_time * n_trials)

Example: 32B on H200 @ $2.14/hr, 200 trials
= $2.14 * (0.5 + 0.033 * 200) = ~$15
```

## Git

```bash
git push fork master   # Always push to fork
```
