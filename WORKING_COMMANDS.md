# WORKING COMMANDS - PROVEN TO WORK

**CRITICAL: READ THIS BEFORE DOING ANYTHING**

## The ONLY Working Method for Vast.ai Operations

The `heretic-vast` CLI (Python) works. The PowerShell scripts have issues.

### Working Commands (use these!)

```bash
# List instances
uv run heretic-vast list

# Create instance (4x RTX 4090 for 32B model)
uv run heretic-vast create RTX_4090 4

# Setup heretic on instance
uv run heretic-vast setup

# Run abliteration with RESUME SUPPORT
uv run heretic-vast exec "export HF_HOME=/workspace/.cache/huggingface && cd /workspace && nohup heretic --model Qwen/Qwen2.5-Coder-32B-Instruct --auto-select true --auto-select-path /workspace/models --storage sqlite:////workspace/heretic_study.db --study-name qwen32b-abliteration > /workspace/heretic.log 2>&1 &"

# Monitor progress
uv run heretic-vast watch
uv run heretic-vast progress

# Check status
uv run heretic-vast status

# View logs
uv run heretic-vast exec "tail -100 /workspace/heretic.log"

# Stop instance (save money)
uv run heretic-vast stop

# Start stopped instance
uv run heretic-vast start

# Download model when complete
uv run heretic-vast download /workspace/models
```

## Resume Support (CRITICAL!)

The `--storage sqlite:////workspace/heretic_study.db --study-name qwen32b-abliteration` flags enable resume:

- SQLite database persists on `/workspace` 
- If training stops, restart instance and run same command
- Optuna automatically detects existing study and continues

## DO NOT USE

- `.\runpod.ps1` commands - have syntax issues
- `.\start-abliteration.ps1` - depends on broken runpod.ps1

## Instance Requirements for Qwen2.5-Coder-32B

- **GPUs**: 4x RTX 4090 (96GB total VRAM)
- **Cost**: ~$1.40-1.60/hr
- **Runtime**: ~5-6 hours for 100 trials
- **Total cost**: ~$8-10

## Full Training Workflow

```bash
# 1. Create instance
uv run heretic-vast create RTX_4090 4

# 2. Wait for instance to be ready, then setup
uv run heretic-vast setup

# 3. Start training with resume support
uv run heretic-vast exec "export HF_HOME=/workspace/.cache/huggingface && cd /workspace && nohup heretic --model Qwen/Qwen2.5-Coder-32B-Instruct --auto-select true --auto-select-path /workspace/models --storage sqlite:////workspace/heretic_study.db --study-name qwen32b-abliteration > /workspace/heretic.log 2>&1 &"

# 4. Monitor (in another terminal or periodically)
uv run heretic-vast watch

# 5. When complete, download model
uv run heretic-vast download /workspace/models

# 6. Stop instance
uv run heretic-vast stop
```

## If Training Stops

```bash
# 1. Start the instance
uv run heretic-vast start

# 2. Wait for it to be ready
uv run heretic-vast list

# 3. Re-run the SAME command - it will RESUME automatically
uv run heretic-vast exec "export HF_HOME=/workspace/.cache/huggingface && cd /workspace && nohup heretic --model Qwen/Qwen2.5-Coder-32B-Instruct --auto-select true --auto-select-path /workspace/models --storage sqlite:////workspace/heretic_study.db --study-name qwen32b-abliteration > /workspace/heretic.log 2>&1 &"
```

## Troubleshooting

### Instance stuck in "loading"
Wait 30-60 seconds and retry. Run `uv run heretic-vast list` to check status.

### SSH connection fails
The instance may not be fully ready. Wait and retry.

### heretic-vast command not found
Make sure you're in the project directory and use `uv run heretic-vast`.
