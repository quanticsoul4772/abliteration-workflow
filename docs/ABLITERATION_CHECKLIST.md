# Abliteration Checklist

**Purpose:** A reusable checklist for every abliteration run to prevent mistakes and ensure consistency.

**Usage:** Copy this checklist for each new model. Check off items as you complete them.

---

## ⛔ MY FAILURE PATTERNS AND HARD RULES ⛔

**These are specific behaviors I (the AI assistant) repeatedly fail at. These rules exist to constrain me.**

### Failure Pattern 1: Asking Instead of Checking

**What I do wrong:** Ask the user for information that I could look up myself (e.g., "What's your HF token?" when it's already in `vastai show env-vars`).

**HARD RULE:** Before asking the user for ANY information, I MUST:
1. Run a command to check if I already have it
2. Show the command output
3. Only THEN ask if the information is missing

**Format I must use:**
```
I checked: [command I ran]
Result: [output]
I still need: [what's missing]
```

**If I ask without showing I checked first, call me out.**

---

### Failure Pattern 2: Empty Filler Words

**What I do wrong:** Generate phrases that sound helpful but do nothing.

**BANNED PHRASES (if I use these, I'm probably not doing real work):**
- "Your call"
- "Let me know"
- "When you're ready"
- "I understand"
- "That makes sense"
- "Feel free to"
- "I'm here when you need me"
- "I apologize" / "I'm sorry" / "My apologies" / "Apologies for" (more than once per session)
- "That's a great question"
- "Happy to help"
- "That's fair"
- "I didn't mean to"
- "I hear you"

**What I should say instead:** The actual next action, or nothing.

---

### Failure Pattern 3: Claiming Without Proving

**What I do wrong:** Say "I did X" without showing the output.

**HARD RULE:** Every action I claim to take MUST include:
1. The exact command I ran
2. The actual output (not summarized)
3. Verification that it worked

**Format I must use:**
```
Command: [exact command]
Output: [actual output, not "it worked"]
Verification: [how I confirmed it worked]
```

**If I say "done" without proof, call me out.**

---

### Failure Pattern 4: "Fixing" Instead of Stopping

**What I do wrong:** When something fails, I try workarounds instead of stopping and asking.

**HARD RULE:** When ANY error occurs:
1. STOP immediately
2. Show the exact error
3. List options (do NOT pick one)
4. Wait for user to choose

**Trigger phrases that mean STOP:**
- Any error message in output
- User says "stop", "wait", "hold on"
- Anything unexpected happens
- I'm about to try something "creative"

**What I must NOT do:**
- Try a different approach without asking
- Disable a feature to work around an error
- Make assumptions about what the user wants
- Say "let me try..." and then do it

---

### Failure Pattern 5: Big Leaps Instead of Small Steps

**What I do wrong:** Try to do too much at once, making it easy to screw up.

**HARD RULE:** One action at a time. After each action:
1. Show what I did
2. Show the result
3. State the next action
4. Wait for confirmation before ANY action that costs money, modifies files, or can't be undone

**Actions that ALWAYS require confirmation:**
- Creating instances
- Running commands on remote instances
- Modifying files
- Stopping/terminating instances
- Uploading anything
- Installing packages

**Actions I can do without asking:**
- Reading local files
- Listing directories
- Checking status (vastai list, show env-vars)
- Searching code

---

### Failure Pattern 6: Not Reading My Own Documentation

**What I do wrong:** Have documentation (CLAUDE.md, WORKFLOW.md, this checklist) and ignore it.

**HARD RULE:** At the start of EVERY abliteration session:
1. I MUST read this checklist file
2. I MUST read CLAUDE.md
3. I MUST quote the specific section that applies to prove I read it

**If I make a mistake that's documented in these files, the user should say: "It's in the docs. Read them."**

---

### Failure Pattern 7: Impatience with Cloud Instances

**What I do wrong:** Terminate instances too quickly, assume connectivity failure after 1-2 minutes, redeploy instead of waiting.

**HARD RULE:** Instance startup takes 5-10 minutes, not 1-2 minutes.
1. After creating instance, wait at least 5 minutes before first SSH attempt
2. If SSH fails, wait 2 more minutes and retry
3. Only consider recreating after 15+ minutes of no connectivity
4. NEVER terminate an instance just because SSH failed once

**What I must NOT do:**
- Say "connection failed, creating new instance"
- Assume the instance is broken after 2 minutes
- Terminate and recreate without explicit permission

---

### How to Hold Me Accountable

**When I slip, use these exact phrases:**

| My Failure | What to Say |
|------------|-------------|
| Asked without checking | "Did you check first? Show me." |
| Used filler words | "That's filler. What's the action?" |
| Claimed without proof | "Show me the output." |
| Tried to fix without asking | "Stop. What are the options?" |
| Did too much at once | "One step. What did you do?" |
| Ignored documentation | "It's in the docs." |
| Terminated instance too quickly | "Did you wait 10 minutes?" |
| Apologized again | "You already apologized. Move on." |

---

## 🔴 SESSION START PROTOCOL

**At the start of EVERY abliteration session, I MUST do these in order:**

```
1. Read this file (docs/ABLITERATION_CHECKLIST.md)
2. Read CLAUDE.md - quote the section that applies
3. Run: vastai show env-vars (check what's configured)
4. Run: vastai list (check for existing instances)
5. Report what I found with actual output
6. State the first action I propose
7. Wait for permission
```

**Example of correct session start:**
```
I read ABLITERATION_CHECKLIST.md and CLAUDE.md.

Relevant from CLAUDE.md: "NEVER create a new instance without explicit permission"

I checked:
- vastai show env-vars: HF_TOKEN, HF_HOME, HF_TRUST_REMOTE_CODE all set
- vastai list: No running instances

First action: Search for A100 40GB GPUs with 150GB disk.
Permission to proceed?
```

---

## 🟡 PRE-FLIGHT CHECKS

### Pre-Session Checks (Must complete before ANY action)

```
□ 1. Read CLAUDE.md completely (not skimmed)
□ 2. Read WORKFLOW.md for bruno-vast commands
□ 3. Understand what the user ACTUALLY wants
□ 4. Check what already exists (local files, previous runs, downloaded models)
□ 5. Verify local hardware can't do this first (RTX 4070, 8GB VRAM)
□ 6. Confirm cloud GPU is actually necessary
```

### Permission Checks (I will NOT proceed without explicit "yes")

| Action | Ask Permission? | Why |
|--------|----------------|-----|
| Create GPU instance | ✅ YES | Costs money |
| Run any destructive command (pkill, rm, stop) | ✅ YES | Irreversible |
| Start abliteration (200 trials) | ✅ YES | Commits to 6+ hours |
| Disable any feature | ✅ YES | Ruined Model 1 |
| Upload to HuggingFace | ✅ YES | Public action |
| Stop/terminate instance | ✅ YES | Kills processes |

---

## 🟡 MODEL PREPARATION

### Model Research Checklist

```
□ Model name: ____________________
□ Architecture type: □ Dense  □ MoE  □ Multimodal
□ Total parameters: ______ B
□ Active parameters (if MoE): ______ B
□ Required transformers version: ______
□ trust_remote_code required: □ Yes  □ No
□ Gated model (requires HF auth): □ Yes  □ No
□ Context length: ______ tokens
```

### MoE Compatibility Check (if MoE architecture)

Bruno supports these MoE patterns:
```
□ layer.mlp.experts (Qwen3 style)
□ layer.block_sparse_moe.experts (Phi-MoE style)
□ layer.moe.experts (Granite MoE style)
□ layer.feed_forward.experts (gpt-oss style)
```

**If model uses different pattern:** May need bruno modification. Test with 2 trials first!

---

## 🟢 RESOURCE CALCULATION

### VRAM Requirements

| Precision | Formula | My Model |
|-----------|---------|----------|
| BF16/FP16 | params × 2 bytes | ______ GB |
| 8-bit | params × 1 byte | ______ GB |
| 4-bit | params × 0.5 bytes | ______ GB |

### Disk Requirements

| Component | Size | My Model |
|-----------|------|----------|
| Model weights | params × 2 | ______ GB |
| Output model | params × 2 | ______ GB |
| HuggingFace cache | ~5GB | 5 GB |
| Optuna database | ~1GB | 1 GB |
| C4 dataset (streaming) | ~0GB | 0 GB |
| Working space | ~10GB | 10 GB |
| **MINIMUM TOTAL** | | ______ GB |
| **RECOMMENDED (+50%)** | | ______ GB |

### GPU Selection

| Model Size | Minimum GPU | Recommended GPU | Notes |
|------------|-------------|-----------------|-------|
| 7B | RTX 4090 (24GB) | A6000 (48GB) | |
| 13B | A100 40GB | A100 80GB | |
| 16B MoE | A100 40GB | A100 80GB | |
| 32B | H100 80GB | **H200 141GB** | ⚠️ See warning below |
| 70B | H200 141GB | 2x H100 | |

**⚠️ CRITICAL: H100 80GB vs H200 141GB for 32B Models**

| GPU | VRAM | 32B Model + Cache | Result |
|-----|------|-------------------|--------|
| H100 80GB | 80GB | ~92GB needed | ❌ **OOM - must use --cache-weights false** |
| H200 141GB | 141GB | ~92GB needed | ✅ **Works with caching (recommended)** |

**If using H100 80GB for 32B:** Must disable caching (`--cache-weights false`), which adds 3-4 hours to training time.

**My selection:** ____________________

---

## 🔵 INSTANCE CREATION

### Search Commands

```bash
# Check availability (run these, report results, wait for permission)
vastai search offers "gpu_name=<GPU> disk_space>=<DISK> rentable=true" --order dph_total
```

### Before Creating Instance

```
□ Searched for available GPUs
□ Reported options to user with prices
□ Received explicit permission to create
□ Disk size calculated: ______ GB
□ GPU tier confirmed: ______
```

### Instance Creation Command

**Option 1: Use Bruno Abliteration Template (RECOMMENDED)**
```bash
# Uses our custom template with bruno pre-installed
# Template ID: 337981
# Template Hash: 7ea2682501dc881b1097c18f096f7c63
vastai launch instance -g A100_SXM4 -n 1 --template_hash 7ea2682501dc881b1097c18f096f7c63 -d 150

# Or create from offer ID:
vastai create instance <OFFER_ID> --template_hash 7ea2682501dc881b1097c18f096f7c63 --disk 150
```

**Option 2: Use Vast.ai PyTorch Template (Fallback)**
```bash
# Uses the official vastai/pytorch template - requires manual bruno install
vastai launch instance -g A100_SXM4 -n 1 -i vastai/pytorch -d 150 --ssh --direct
```

**Option 3: Direct instance creation (Manual)**
```bash
# ONLY RUN AFTER USER SAYS "YES"
vastai create instance <OFFER_ID> \
  --disk <DISK_SIZE> \
  --image pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel \
  --onstart-cmd "pip install huggingface_hub accelerate && pip install git+https://github.com/quanticsoul4772/bruno.git"
```

**Bruno Abliteration Template (ID: 337981) Benefits:**
- ✅ Bruno pre-installed on startup
- ✅ HuggingFace Hub + Accelerate included
- ✅ `HF_HOME` and `HF_TRUST_REMOTE_CODE` pre-configured
- ✅ SSH + Direct connection enabled
- ✅ 40GB+ VRAM, 150GB+ disk, CUDA 12.1+ filters
- ✅ Public template - works for anyone

### After Creation - BE PATIENT!

**⚠️ PATIENCE RULES:**
- Instance startup can take **5-10 minutes** (not 1-2 minutes)
- SSH may not be available immediately even if instance shows "running"
- Network connectivity varies - don't assume failure too quickly
- **DO NOT terminate and recreate** - just wait longer
- If SSH fails, wait 2 more minutes and try again
- Only consider recreating after 15+ minutes of no connectivity

**Startup Timeline (typical):**
| Phase | Time | What's Happening |
|-------|------|------------------|
| Instance creation | 0-1 min | Vast.ai allocating resources |
| Container pulling | 1-5 min | Docker image downloading |
| onstart script | 2-5 min | pip install, setup commands |
| SSH ready | 5-10 min | Full connectivity available |

```
□ Created instance - note the time: ______
□ Waited at least 5 minutes before first SSH attempt
□ If SSH fails, waited 2 more minutes and retried
□ Verified instance is running: uv run bruno-vast list
□ Verified disk space: uv run bruno-vast exec "df -h /workspace"
□ Verified bruno installed: uv run bruno-vast exec "which bruno"
□ If still failing after 15 minutes, THEN consider recreating
```

---

## 🟣 ABLITERATION EXECUTION

### Environment Setup

```
□ export HF_HOME=/workspace/.cache/huggingface
□ export HF_TOKEN=hf_...
□ export HF_TRUST_REMOTE_CODE=1  (for models with custom code)
```

### ⚠️ COMPATIBILITY TEST (REQUIRED)

**ALWAYS run 2-trial test before committing to 200 trials!**

```bash
# Run in tmux to survive SSH disconnect
tmux new-session -d -s bruno-test '
export HF_HOME=/workspace/.cache/huggingface
export HF_TRUST_REMOTE_CODE=1
bruno \
  --model <MODEL> \
  --auto-select true \
  --auto-select-path /workspace/models \
  --storage sqlite:////workspace/bruno_test.db \
  --study-name test-2trials \
  --n-trials 2 \
  --cache-weights true \
  --unhelpfulness-prompts.dataset allenai/c4 \
  --unhelpfulness-prompts.config en \
  --unhelpfulness-prompts.split "train[:200]" \
  --unhelpfulness-prompts.column text \
  2>&1 | tee /workspace/bruno_test.log
'
```

### Test Verification Checklist

```
□ Model loads without errors
□ "Extracting refusal directions" completes
□ "Trial 0 finished" appears in logs
□ "Trial 1 finished" appears in logs
□ No KeyError or AttributeError for MoE layers
□ MoE experts detected (if MoE model)
□ No OOM errors
```

**If test fails:** STOP. Report error to user. Do not try workarounds.

### Full Training (AFTER test passes, AFTER user permission)

```bash
# ONLY RUN AFTER:
# 1. Compatibility test passed
# 2. User explicitly approved

tmux new-session -d -s bruno '
export HF_HOME=/workspace/.cache/huggingface
export HF_TRUST_REMOTE_CODE=1
bruno \
  --model <MODEL> \
  --auto-select true \
  --auto-select-path /workspace/models \
  --storage sqlite:////workspace/bruno_study.db \
  --study-name <MODEL_NAME>-abliteration \
  --n-trials 200 \
  --cache-weights true \
  --unhelpfulness-prompts.dataset allenai/c4 \
  --unhelpfulness-prompts.config en \
  --unhelpfulness-prompts.split "train[:200]" \
  --unhelpfulness-prompts.column text \
  2>&1 | tee /workspace/bruno.log
'
```

### Monitoring Commands

```bash
# Live dashboard
uv run bruno-vast watch

# Check progress
uv run bruno-vast progress

# View logs
uv run bruno-vast exec "tail -100 /workspace/bruno.log"

# Attach to tmux session
uv run bruno-vast exec "tmux attach -t bruno"
# Detach: Ctrl+B then D
```

---

## 🟤 POST-TRAINING

### Training Completion Checks

```
□ Logs show "Best trial" message
□ Logs show "Saved" message
□ Model exists: ls -la /workspace/models/
□ Model size correct: du -sh /workspace/models/*
□ No errors in last 100 lines of log
```

### Download Checklist

```
□ Verified training complete
□ Verified model exists on instance
□ Checked local disk space (need model size + buffer)
□ Download started: rsync or bruno-vast download
   ⚠️ Windows users: rsync requires WSL - run from WSL terminal, NOT PowerShell!
□ Download completed successfully
□ Verified local files match remote size
```

### Local Testing (BEFORE upload)

```
□ Model loads locally without errors
□ Tested with restricted prompt
□ Response is helpful (no refusal)
□ No excessive moralizing
□ No disclaimers or warnings
```

**If model still refuses:** STOP. Abliteration did not work. Do NOT upload.

---

## 🔵 HUGGINGFACE UPLOAD

### Pre-Upload Fixes

```
□ Fixed tokenizer_config.json (extra_special_tokens: {} not [])
□ Created README.md model card
□ Created handler.py (if using inference endpoints)
□ Created requirements.txt
```

### Upload Checklist

```
□ Received permission to upload
□ Repository name confirmed with user
□ Uploaded successfully
□ Model card displays correctly
□ Files all present on HuggingFace
```

---

## 🟠 CLEANUP

### Before Stopping Instance

```
□ Verified model downloaded to local machine
□ Verified local model loads correctly
□ Received explicit permission to stop
□ Understand stopping KILLS any running process
```

### Cleanup Commands

```bash
# ONLY AFTER user confirms download complete
uv run bruno-vast stop

# Or terminate completely
uv run bruno-vast terminate <INSTANCE_ID>
```

---

## ⛔ THINGS I WILL NOT DO

| Action | Why |
|--------|-----|
| Disable `--orthogonalize-directions` | Ruined Model 1 - model still moralizes |
| Disable any feature without permission | User must explicitly approve |
| Use 100GB disk for 32B models | Caused data loss on Model 2 |
| Skip the 2-trial compatibility test | Could waste 6+ hours on broken config |
| Proceed without explicit "yes" | Multiple past mistakes from assuming |
| Try workarounds when errors occur | Made things worse every time |
| Run commands without tmux | SSH disconnect kills process |
| Upload before local testing | Could publish broken model |
| Use H100 80GB with --cache-weights true for 32B | OOM - 92GB needed > 80GB available |
| Terminate instance because SSH fails once | Instance may still be starting up |
| Assume connectivity failure after 2 minutes | Wait at least 10 minutes before giving up |
| Redeploy without waiting for first instance | Wastes money and time |

---

## 📊 RUN LOG

Fill this in for each abliteration run:

```
Model: ____________________
Date Started: ____________________
GPU Used: ____________________
Vast.ai Instance ID: ____________________
Instance Cost/hr: $______

Compatibility Test:
  Started: ____________________
  Completed: ____________________
  Result: □ Pass  □ Fail

Full Training:
  Started: ____________________
  Completed: ____________________
  Total Trials: ______
  Best Trial: ______
  Final KL Divergence: ______
  Final Refusals: ______

Download:
  Started: ____________________
  Completed: ____________________
  Local Path: ____________________

Upload:
  Repository: ____________________
  Completed: ____________________

Total Cost: $______
Total Time: ______ hours

Notes:
____________________
____________________
____________________
```

---

## 📜 ABLITERATION HISTORY

| # | Model | Date | Status | GPU | Cost | Notes |
|---|-------|------|--------|-----|------|-------|
| 1 | Qwen2.5-7B-Instruct | Feb 2026 | ⚠️ Partial | 4x RTX 4090 | ~$15 | orthogonalization disabled |
| 2 | Qwen2.5-Coder-32B-Instruct | Feb 2026 | ✅ Success | H200 | ~$25 | Trial 173, 0 refusals |
| 3 | Moonlight-16B-A3B-Instruct | Feb 2026 | 🔄 Pending | TBD | TBD | First MoE model |

---

## 🔧 QUICK REFERENCE: CLI FLAGS

**Required for all runs:**
```bash
--model <MODEL>
--auto-select true                    # NOT just --auto-select
--auto-select-path /workspace/models
--storage sqlite:////workspace/bruno_study.db
--study-name <NAME>
```

**Required for C4 dataset (pass ALL four):**
```bash
--unhelpfulness-prompts.dataset allenai/c4
--unhelpfulness-prompts.config en
--unhelpfulness-prompts.split "train[:200]"
--unhelpfulness-prompts.column text
```

**Memory settings:**
```bash
--cache-weights true    # Default, works on H200 for 32B (v1.2.0+)
--cache-weights false   # Fallback for H100 80GB with 32B models
```

**Optional optimizations:**
```bash
--compile              # 1.5-2x inference speedup
--n-trials 200         # Default, adjust as needed
```

---

## 🔧 OOM TROUBLESHOOTING GUIDE

**CRITICAL: Diagnose software issues FIRST before blaming hardware!**

**Note:** All diagnostic commands below run on the **remote Vast.ai instance** via `bruno-vast exec`, not your local Windows machine.

When you encounter an OOM error, follow this diagnostic process in order:

### Step 1: Identify the EXACT Error

```bash
# Check the full error message
uv run bruno-vast exec "tail -200 /workspace/bruno.log | grep -A 10 -i 'error\|oom\|memory'"

# Check system OOM killer
uv run bruno-vast exec "dmesg | tail -50 | grep -i oom"
```

**Common OOM Error Types:**

| Error Message | Likely Cause | Software Fix First |
|---------------|--------------|--------------------|
| `torch.cuda.OutOfMemoryError` | GPU VRAM exhausted | Reduce batch_size, disable caching |
| `CUDA out of memory. Tried to allocate X GiB` | Single allocation too large | Reduce batch_size to 1 |
| `RuntimeError: out of memory` | General memory issue | Check for memory leaks |
| `Killed` (no error message) | System OOM killer | Check system RAM, not just GPU |
| `No space left on device` | **DISK**, not memory! | Clear cache, increase disk |
| `MemoryError` during download | System RAM exhausted | Set `HF_HOME` to disk with more space |

### Step 2: Check Current Memory State

```bash
# GPU memory breakdown
uv run bruno-vast exec "nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv"

# System RAM
uv run bruno-vast exec "free -h"

# Disk space (often confused with memory!)
uv run bruno-vast exec "df -h /workspace"

# Python process memory
uv run bruno-vast exec "ps aux --sort=-%mem | head -10"
```

### Step 3: Identify the Culprit Phase

**OOM can occur in different phases - each has different fixes:**

| Phase | When It Happens | Memory Usage | Fix |
|-------|-----------------|--------------|-----|
| **Model Loading** | First 1-2 minutes | Model size × 2 | Use `--cache-weights false` |
| **Batch Size Detection** | "Trying batch size X" | Batch × sequence length | Set explicit `--batch-size 4` |
| **Direction Extraction** | "Extracting refusal directions" | Residuals in memory | Reduce split size (e.g., `train[:100]`) |
| **Weight Caching** | "Caching abliterable weights" | +28GB for 32B model | Use `--cache-weights false` |
| **Trial Evaluation** | "Running trial X" | Generation memory | Reduce `--refusal-check-tokens` |
| **Model Saving** | "Saving model" | Full model copy | Clear cache first |

### Step 4: Software Fixes (Try These BEFORE Hardware)

**Fix 1: Disable weight caching (biggest memory saver)**
```bash
bruno --model MODEL --cache-weights false
# Saves ~28GB for 32B models, ~6GB for 7B models
```

**Fix 2: Reduce batch size**
```bash
bruno --model MODEL --batch-size 1 --max-batch-size 4
```

**Fix 3: Reduce token generation**
```bash
bruno --model MODEL --refusal-check-tokens 20
```

**Fix 4: Clear GPU cache between operations**
```python
# bruno already does this, but if you're debugging:
import torch; torch.cuda.empty_cache()
import gc; gc.collect()
```

**Fix 5: Check for memory leaks (multiple trials)**
```bash
# If OOM happens on trial 50 but not trial 1, suspect memory leak
uv run bruno-vast exec "watch -n 5 nvidia-smi"
# Bruno clears cache between trials, but leaks can occur.
# If you see gradual memory increase, report issue at GitHub.
```

**Fix 6: Multi-GPU memory imbalance**
```bash
# Check if GPU 0 is overloaded while others are empty
uv run bruno-vast exec "nvidia-smi"
# If GPU 0: 23GB, GPU 1-3: <1GB → device_map issue
# Fix: Use device_map="balanced" not "auto" in config
```

### Step 5: Memory Calculations

**Before blaming GPU size, calculate expected usage:**

| Component | Formula | 7B Model | 32B Model |
|-----------|---------|----------|------------|
| Model weights (BF16) | params × 2 bytes | 14 GB | 64 GB |
| Layer-wise cache | ~43% of model | 6 GB | 28 GB |
| KV cache (per batch) | layers × heads × seq × 2 × 2 | 0.5 GB | 2 GB |
| Residuals (extraction) | prompts × layers × hidden × 4 | 2 GB | 8 GB |
| Working memory | ~10% of model | 1.5 GB | 6 GB |
| **TOTAL (with caching)** | | **24 GB** | **108 GB** |
| **TOTAL (without caching)** | | **18 GB** | **80 GB** |

**GPU Selection Based on Calculation:**

| Model Size | With Caching | Without Caching | Recommended GPU |
|------------|--------------|-----------------|------------------|
| 7B | 24 GB | 18 GB | RTX 4090 (24 GB) |
| 13B | 45 GB | 32 GB | A100 40GB |
| 32B | 108 GB | 80 GB | H200 (141 GB) |
| 70B | 200 GB+ | 160 GB | 2× H100 |

### Step 6: Bruno's Built-in OOM Recovery

Bruno has automatic OOM recovery (see `src/bruno/main.py`):

1. **Catches `torch.cuda.OutOfMemoryError`** during evaluation
2. **Clears GPU cache** with `empty_cache()`
3. **Reloads model** to reset state
4. **Reduces batch size** by half automatically
5. **Retries up to 3 times** (`MAX_OOM_RETRIES`)
6. **Preserves progress** via Optuna storage (resume supported)

**If bruno's recovery triggers:**
```
[red]GPU OOM detected (attempt 1/3)[/]
[yellow]GPU Memory: 138.5/141.0 GB used[/]
[yellow]Reducing batch_size to 2 and retrying...[/]
```

**This is EXPECTED behavior**, not a failure. Wait for it to recover.

### Step 7: When to ACTUALLY Blame Hardware

**Only blame hardware if ALL of these are true:**

1. ✅ You've tried `--cache-weights false`
2. ✅ You've tried `--batch-size 1`
3. ✅ Memory calculation shows GPU is too small
4. ✅ OOM happens on Trial 1 (not a leak)
5. ✅ Model loads but fails on first operation

**Hardware upgrade needed when:**
- 32B model on H100 80GB with caching → Need H200 141GB
- 70B model on single H200 → Need 2× H100 or 8× A100

### Step 8: Diagnostic Commands Quick Reference

```bash
# Full diagnostic dump
uv run bruno-vast exec "
echo '=== GPU ===' && nvidia-smi
echo '=== RAM ===' && free -h
echo '=== Disk ===' && df -h /workspace
echo '=== Processes ===' && ps aux --sort=-%mem | head -5
echo '=== Python Memory ===' && python3 -c 'import torch; print(torch.cuda.memory_summary() if torch.cuda.is_available() else "No CUDA")'
"

# Memory over time (run in separate terminal)
uv run bruno-vast exec "while true; do nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader; sleep 10; done"
```

---

## 📝 OPERATOR PROMPTS (for user to paste)

### Session Start
```
You are helping me abliterate a model. Follow docs/ABLITERATION_CHECKLIST.md exactly.
Do NOT create instances, run commands, or disable features without explicit permission.
Stop and ask at checkpoints. Report errors and wait for instructions.
```

### Before Cloud Action
```
STOP. Before running this command:
1. What will this cost in time and money?
2. Is this reversible?
3. Have you confirmed prerequisites?
Wait for my "yes" before proceeding.
```

### When Things Go Wrong
```
STOP. Do not try to fix this on your own.
1. What exactly happened?
2. What do the logs say?
3. What are our options?
Wait for my decision.
```

### Emergency Stop
```
STOP. Do not spawn agents. Do not run commands. Just acknowledge.
```
