# Heretic Abliteration Runbook

**Last Updated:** This document contains PROVEN working commands and procedures.

---

## ⚠️ CRITICAL: READ THIS FIRST

### Repository Structure
- **Primary repo:** `fork` → https://github.com/quanticsoul4772/abliteration-workflow
- **Secondary repo:** `heretic-fork` → https://github.com/quanticsoul4772/heretic
- **Always push to `fork`** (abliteration-workflow) - this is the main working repo

### Resume Support (FIXED)
Training now has **persistent storage** via SQLite. If training stops, it RESUMES from last trial.

**The fix:** Added `--storage sqlite:////workspace/heretic_study.db --study-name qwen32b-abliteration` to the heretic command.

---

## Quick Start: Qwen2.5-Coder-32B Training

### One Command (Recommended)
```powershell
cd C:\Development\Projects\heretic
.\start-abliteration.ps1
```

### Step-by-Step (If One Command Fails)
```powershell
# 1. Create 4x RTX 4090 instance (~$1.50/hr, ~2 min)
.\runpod.ps1 vast-create-pod RTX_4090 4

# 2. Install heretic (~3 min)
.\runpod.ps1 vast-setup

# 3. Start abliteration with RESUME SUPPORT (~5-6 hours)
.\runpod.ps1 vast-exec "export HF_HOME=/workspace/.cache/huggingface && cd /workspace && nohup heretic --model Qwen/Qwen2.5-Coder-32B-Instruct --auto-select true --auto-select-path /workspace/models --storage sqlite:////workspace/heretic_study.db --study-name qwen32b-abliteration > /workspace/heretic.log 2>&1 &"

# 4. Monitor progress
.\runpod.ps1 vast-watch
```

---

## Monitoring Commands

```powershell
# Live dashboard (updates every 30s) - RECOMMENDED
.\runpod.ps1 vast-watch

# Quick status check
.\runpod.ps1 vast-progress

# Check GPU status
.\runpod.ps1 vast-status

# View live logs
.\runpod.ps1 vast-exec "tail -f /workspace/heretic.log"

# List instances
.\runpod.ps1 vast-list
```

---

## Resume After Interruption

If training stops for ANY reason (crash, preemption, network issue):

```powershell
# 1. Check if instance is still running
.\runpod.ps1 vast-list

# 2. If stopped, restart it
.\runpod.ps1 vast-start

# 3. Wait for instance to start (~30 seconds), then re-run heretic
#    IT WILL AUTOMATICALLY RESUME from the SQLite database
.\runpod.ps1 vast-exec "export HF_HOME=/workspace/.cache/huggingface && cd /workspace && nohup heretic --model Qwen/Qwen2.5-Coder-32B-Instruct --auto-select true --auto-select-path /workspace/models --storage sqlite:////workspace/heretic_study.db --study-name qwen32b-abliteration > /workspace/heretic.log 2>&1 &"

# 4. Verify it's running and resumed
.\runpod.ps1 vast-progress
```

**Why it resumes:** Optuna stores all completed trials in `/workspace/heretic_study.db`. When heretic starts with the same `--study-name`, it loads existing trials and continues.

---

## After Training Completes

```powershell
# 1. Check for completed model
.\runpod.ps1 vast-models

# 2. Download model to local machine (~60GB, may take 30+ min)
.\runpod.ps1 vast-download-model

# 3. STOP THE INSTANCE (stop billing!)
.\runpod.ps1 vast-stop
```

---

## Configuration Details

### Training Config (config-qwen32b.toml)
- **Trials:** 100
- **Batch size:** 8
- **Max response length:** 150
- **Prune trials:** true
- **Startup trials:** 25
- **Training data:** 500 harmless + 300 harmful prompts
- **Eval data:** 200 harmless + 150 harmful prompts

### Hardware
- **GPU:** 4x RTX 4090 (96GB total VRAM)
- **Cost:** ~$1.40-1.60/hr
- **Total estimated cost:** ~$8-10 for full run
- **Runtime:** ~5-6 hours

### Critical Flags for Resume Support
```
--storage sqlite:////workspace/heretic_study.db
--study-name qwen32b-abliteration
```

---

## Troubleshooting

### "No instance ID found"
```powershell
.\runpod.ps1 vast-list  # Check if instance exists
.\runpod.ps1 vast-create-pod RTX_4090 4  # Create new if needed
```

### SSH Connection Issues
```powershell
# Wait 30-60 seconds after instance starts, then:
.\runpod.ps1 vast-list  # Get updated SSH info
```

### "heretic command not found"
```powershell
.\runpod.ps1 vast-setup  # Reinstall heretic
```

### Training Not Resuming
Check that the SQLite database exists:
```powershell
.\runpod.ps1 vast-exec "ls -la /workspace/*.db"
```

If it doesn't exist, the previous run may have crashed before any trials completed.

### GPU Out of Memory
Reduce batch size in the config:
```powershell
.\runpod.ps1 vast-exec "cat > /workspace/config.toml << 'EOF'
batch_size = 4
max_batch_size = 4
EOF"
```

---

## Git Workflow

### Push Changes
```powershell
git add -A
git commit -m "Your message"
git push fork master  # Push to abliteration-workflow (PRIMARY)
```

### If Git Lock File Blocks Push
```powershell
# Kill stuck git processes
powershell -Command "Get-Process git*, git-remote* -ErrorAction SilentlyContinue | Stop-Process -Force"

# Then retry push
git push fork master
```

---

## Files Reference

| File | Purpose |
|------|---------|
| `start-abliteration.ps1` | One-command quick start for Qwen32B training |
| `runpod.ps1` | All cloud GPU commands (Vast.ai and RunPod) |
| `config-qwen32b.toml` | Optimized config for Qwen2.5-Coder-32B |
| `.env` | API keys (VAST_API_KEY, HF_TOKEN) |
| `WORKFLOW.md` | Detailed workflow documentation |
| `CLAUDE.md` | AI assistant instructions |

---

## Key Learnings (Don't Forget!)

1. **Always use `--storage` and `--study-name`** for resume support
2. **Push to `fork` remote** (abliteration-workflow), not heretic-fork
3. **SQLite database persists on /workspace** - survives instance restarts
4. **Kill zombie git processes** if push hangs
5. **4x RTX 4090 = 96GB VRAM** - sufficient for 32B models
6. **Vast.ai is 50% cheaper** than RunPod
7. **Monitor with `vast-watch`** for live dashboard
8. **Stop instance when done** - billing continues until stopped!

---

## ⛔ CRITICAL: DO NOT DO THESE THINGS ⛔

### NEVER judge instance status by uptime
- Vast.ai shared GPUs show ACCUMULATED uptime from all users
- An instance showing "47 hours" might have just started YOUR session
- **ONLY check if heretic process is running, NOT the uptime**

### NEVER create a new instance without explicit permission
- If user says "wait" - WAIT
- If instance is "loading" - WAIT
- If you're not sure - ASK
- **DO NOT assume an instance is stuck**

### NEVER take destructive actions without asking
- Don't stop instances
- Don't terminate instances  
- Don't create new instances
- **ALWAYS ASK FIRST**

### When user says STOP - STOP IMMEDIATELY
- Don't spawn more agents
- Don't make more tool calls
- Don't try to "fix" anything
- Just STOP and LISTEN
