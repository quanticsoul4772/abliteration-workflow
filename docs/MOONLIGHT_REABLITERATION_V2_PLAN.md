# Moonlight-16B Re-Abliteration Plan v2

**Date:** February 2026
**Target Model:** moonshotai/Moonlight-16B-A3B-Instruct
**Goal:** Improve benchmark scores using MPOA and all v2.0 improvements
**Previous Results:** MMLU 48%, HellaSwag 56%, GSM8K 51%, KL 8.94, Refusals 41%

---

## 🎯 Expected Improvements

| Metric | Previous (v1) | Target (v2) | Improvement |
|--------|---------------|-------------|-------------|
| **MMLU** | 48.0% | 54-60% | +6-12% |
| **HellaSwag** | 56.0% | 65-72% | +9-16% |
| **GSM8K** | 51.0% | 56-64% | +5-13% |
| **KL Divergence** | 8.94 | **<2.0** | **4-5x lower** |
| **Refusal Rate** | 41% | <15% | -26% |

---

## 🆕 What's New in This Run

### 1. MPOA (Norm-Preserving Biprojected Abliteration) ⭐ BIGGEST IMPACT

**What it does:** Preserves weight matrix norms after projection removal.

**Why it matters:**
- Standard abliteration reduces matrix norms → damages learned representations
- MPOA rescales weights back to original magnitudes → preserves capabilities
- Expected: **2-4x lower KL divergence**

**Configuration:**
```toml
use_mpoa = true
mpoa_norm_mode = "row"      # Preserves output neuron magnitudes
mpoa_min_scale = 0.5        # Prevents extreme shrinking
mpoa_max_scale = 2.0        # Prevents extreme inflation
```

### 2. Enhanced Sacred Direction Protection

**Previous:** 5 directions, 0.3 overlap threshold
**New:** 10 directions, 0.2 overlap threshold (stricter)

**Why:** More MMLU capability directions are protected from ablation.

### 3. More Conservative Layer Profiles

**Previous:**
- Early (0-35%): 0.5 weight
- Middle (35-65%): 1.0 weight
- Late (65-100%): 0.8 weight

**New (more conservative):**
- Early (0-35%): **0.3** weight (gentler)
- Middle (35-65%): **0.8** weight (reduced)
- Late (65-100%): **0.5** weight (reduced)

### 4. Lower Activation Target Percentile

**Previous:** 0.75 (default)
**New:** 0.60 (more conservative)

**Why:** Less aggressive calibration = less capability damage.

### 5. Single Iterative Round

**Previous:** 2 rounds (default)
**New:** 1 round

**Why:** Multiple rounds can over-ablate. Single round + MPOA is more precise.

### 6. More Optimization Trials

**Previous:** 200 trials
**New:** 300 trials

**Why:** Better exploration of Pareto frontier for optimal KL vs refusal tradeoff.

---

## ⛔ LESSONS LEARNED - DO NOT REPEAT

### From Previous Moonlight Run:
1. ❌ KL divergence was 8.94 - **way too high** (should be <1.0)
2. ❌ MPOA wasn't available - now it is
3. ❌ Only 5 sacred directions - now using 10
4. ❌ Default activation percentile (0.75) was too aggressive

### From All Previous Runs:
1. ❌ **Never disable features** - all features must remain ON
2. ❌ **Never create instances without permission**
3. ❌ **Always run 2-trial test first**
4. ❌ **Use 150GB+ disk** - 100GB causes data loss
5. ❌ **Use tmux** - SSH disconnection kills processes
6. ❌ **transformers 4.51.0** - not 4.48.0, not 5.0.0
7. ❌ **Wait for loading instances** - 3-5 minutes is normal
8. ❌ **Build wheel locally** - `pip install git+...` fails due to submodules
9. ❌ **Install tiktoken** - required for Moonlight tokenizer
10. ❌ **HuggingFace username is `rawcell`** - don't use "YOUR_USERNAME"
11. ❌ **Read .env for tokens** - check `cat .env | grep HF` before asking user
12. ❌ **Check .env encoding** - must be UTF-8, not UTF-16 (Windows Notepad issue)
13. ❌ **Check before starting downloads** - use `ps aux | grep python` first
14. ❌ **Listen to user instructions** - when user says "wait", WAIT
15. ❌ **Use H200 141GB for Moonlight** - A100 80GB causes OOM during residual extraction

---

## 📋 PRE-FLIGHT CHECKLIST

Before starting, verify ALL items:

```
□ 1. Read this entire plan
□ 2. Verify configs/config.moonlight.toml has MPOA enabled
□ 3. Verify .env has VAST_API_KEY and HF_TOKEN
□ 4. Check .env encoding: file .env should show "ASCII text" or "UTF-8"
□ 5. No existing Vast.ai instances running (check: uv run bruno-vast list)
□ 6. Local disk has 40GB free for model download
□ 7. Understand: DPO/SFT/LoRA fine-tuning is BLOCKED for Moonlight (MoE gate assertion)
□ 8. Wheel is built with MPOA support (verify after build)
□ 9. Read .env for HF_TOKEN before asking user (cat .env | grep HF)
□ 10. Check no processes already running: ps aux | grep bruno
```

---

## 🖥️ SERVER REQUIREMENTS

**⚠️ CRITICAL: Moonlight requires H200 141GB for abliteration!**

| GPU | VRAM | Status |
|-----|------|--------|
| A100 40GB | 40GB | ❌ Model doesn't fit |
| A100 80GB | 80GB | ❌ OOM during residual extraction |
| 2x RTX 4090 | 48GB | ❌ OOM during abliteration |
| **H200 141GB** | 141GB | ✅ **Required** |

**Why H200 is required:**
- Moonlight model weights: ~63GB in BF16
- BART-MNLI detector: ~1.5GB
- Residual extraction working memory: ~24GB
- **Total needed: ~88GB** (exceeds A100 80GB)

**Recommended instance:**
- **GPU:** H200 141GB (single GPU)
- **Disk:** 200GB+
- **Image:** pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

**If instance is still running:**
```bash
# Check existing instance
uv run bruno-vast list

# If running, verify it's H200 with enough VRAM
uv run bruno-vast exec "nvidia-smi && df -h /workspace"
```

**If need new instance:**
```bash
# Search for H200 with 200GB+ disk (REQUIRED for Moonlight)
vastai search offers "gpu_name=H200 disk_space>=200 rentable=true" --order dph_total

# Create (WAIT FOR USER PERMISSION)
vastai create instance OFFER_ID --disk 200 --image pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel
```

**Note:** 2x RTX 4090 works for inference/testing AFTER abliteration is complete, but NOT for the abliteration run itself.

---

## 📦 PHASE 1: BUILD AND SETUP

### Step 1.1: Build Wheel Locally

```bash
# Build wheel
uv build

# IMPORTANT: Verify MPOA is included in the wheel
unzip -l dist/bruno_ai-*.whl | grep -E "mpoa|model.py"
# Should show: bruno/model.py (contains MPOA implementation)
```

### Step 1.2: Verify Instance Ready

```bash
uv run bruno-vast list
# Wait for status "running" (3-5 minutes if loading)
```

### Step 1.3: Upload and Install Bruno

```bash
# Get SSH details first
uv run bruno-vast list
# Note the PORT and HOST

# ⚠️ IMPORTANT: Build wheel locally - do NOT use pip install git+...
# Git install fails due to llama.cpp submodule issue

# Upload wheel (replace PORT and HOST)
scp -o StrictHostKeyChecking=no -P PORT dist/bruno_ai-*.whl root@HOST:/workspace/

# Install on instance (tiktoken is REQUIRED for Moonlight tokenizer)
uv run bruno-vast exec "pip install /workspace/bruno_ai-*.whl --force-reinstall && pip install tiktoken transformers==4.51.0"
```

### Step 1.4: Clear Previous Cache (Important!)

```bash
uv run bruno-vast exec "rm -rf /root/.cache/huggingface/modules/transformers_modules && rm -f /workspace/bruno_test.db /workspace/bruno.log /workspace/moonlight_v2.db"
```

### Step 1.5: Verify Installation

```bash
uv run bruno-vast exec "bruno --help && python -c 'import transformers; print(transformers.__version__)'"
# Should show: 4.51.0
```

---

## 📤 PHASE 2: UPLOAD CONFIGURATION

**⚠️ IMPORTANT: We use config-file-only approach. Bruno reads config.toml automatically.**

### Step 2.1: Upload Optimized Config

```bash
# Get SSH details
uv run bruno-vast list

# Upload config (replace PORT and HOST)
scp -o StrictHostKeyChecking=no -P PORT configs/config.moonlight.toml root@HOST:/workspace/config.toml
```

### Step 2.2: Verify Config on Server

```bash
uv run bruno-vast exec "cat /workspace/config.toml | grep -E 'use_mpoa|mpoa_norm|mpoa_min|mpoa_max|n_sacred|sacred_overlap|n_trials|iterative_rounds|activation_target'"
```

**Expected output (verify ALL these values):**
```
use_mpoa = true
mpoa_norm_mode = "row"
mpoa_min_scale = 0.5         # NEW - prevents extreme shrinking
mpoa_max_scale = 2.0         # NEW - prevents extreme inflation
use_sacred_directions = true
n_sacred_directions = 10
sacred_overlap_threshold = 0.2
n_trials = 300
iterative_rounds = 1
activation_target_percentile = 0.60
```

### Step 2.3: Set Environment Variables

```bash
# Set HF_TOKEN for dataset access (get token from .env or HuggingFace)
uv run bruno-vast exec "echo 'export HF_HOME=/workspace/.cache/huggingface' >> ~/.bashrc && echo 'export HF_TRUST_REMOTE_CODE=1' >> ~/.bashrc && echo 'export HF_TOKEN=hf_xxx  # Get from .env: cat .env | grep HF_TOKEN' >> ~/.bashrc && source ~/.bashrc"
```

---

## 🧪 PHASE 3: COMPATIBILITY TEST (REQUIRED)

**⚠️ ALWAYS run 2-trial test before committing to full run**

### Step 3.1: Create Test Config

```bash
# Create a test config with only 2 trials (override n_trials)
uv run bruno-vast exec "
cp /workspace/config.toml /workspace/config_test.toml
sed -i 's/n_trials = 300/n_trials = 2/' /workspace/config_test.toml
sed -i 's/study_name = .*/study_name = \"moonlight-v2-test\"/' /workspace/config_test.toml
grep -E 'n_trials|study_name' /workspace/config_test.toml
"
```

### Step 3.2: Start Quick Test in tmux

```bash
uv run bruno-vast exec "
export HF_HOME=/workspace/.cache/huggingface
export HF_TRUST_REMOTE_CODE=1
export HF_TOKEN=hf_xxx  # Get from .env: cat .env | grep HF_TOKEN
cd /workspace

# Kill any existing processes
pkill -f 'bruno' 2>/dev/null || true

# Rename test config to config.toml temporarily
cp /workspace/config_test.toml /workspace/config.toml

# Start 2-trial test in tmux (bruno reads config.toml automatically)
tmux new-session -d -s bruno-test '
export HF_HOME=/workspace/.cache/huggingface
export HF_TRUST_REMOTE_CODE=1
export HF_TOKEN=hf_xxx  # Get from .env: cat .env | grep HF_TOKEN
cd /workspace && bruno 2>&1 | tee /workspace/bruno_test.log'

echo 'Test started in tmux. Attach with: tmux attach -t bruno-test'
sleep 5
ps aux | grep bruno | grep -v grep
"
```

### Step 3.3: Monitor Test Progress

```bash
# Check logs
uv run bruno-vast exec "tail -50 /workspace/bruno_test.log"

# Attach to session (Ctrl+B then D to detach)
uv run bruno-vast exec "tmux attach -t bruno-test"
```

### Step 3.4: Verify Test Success

**Check for these in logs:**
- ✅ `MPOA enabled with row norm preservation`
- ✅ `Extracting 10 sacred directions`
- ✅ `Trial 0 finished` and `Trial 1 finished`
- ✅ No `AssertionError` or `KeyError`
- ✅ MoE experts detected

**If test fails:** STOP and report error. Do NOT proceed to full run.

---

## 🚀 PHASE 4: FULL ABLITERATION

**⚠️ ONLY proceed after Phase 3 test passes**

### Step 4.1: Restore Full Config

**⚠️ Run this from your LOCAL machine (not inside exec):**

```bash
# Get SSH details first
uv run bruno-vast list
# Note the PORT and HOST

# Upload the full 300-trial config (run locally!)
scp -o StrictHostKeyChecking=no -P PORT configs/config.moonlight.toml root@HOST:/workspace/config.toml

# Kill test process if still running
uv run bruno-vast exec "tmux kill-session -t bruno-test 2>/dev/null || true"
```

### Step 4.2: Start Full Run

```bash
uv run bruno-vast exec "
export HF_HOME=/workspace/.cache/huggingface
export HF_TRUST_REMOTE_CODE=1
export HF_TOKEN=hf_xxx  # Get from .env: cat .env | grep HF_TOKEN
cd /workspace

# Start full abliteration in tmux (bruno reads config.toml automatically)
tmux new-session -d -s bruno '
export HF_HOME=/workspace/.cache/huggingface
export HF_TRUST_REMOTE_CODE=1
export HF_TOKEN=hf_xxx  # Get from .env: cat .env | grep HF_TOKEN
cd /workspace && bruno 2>&1 | tee /workspace/bruno_v2.log'

echo 'Full abliteration started in tmux.'
echo 'Attach with: tmux attach -t bruno'
echo 'Detach with: Ctrl+B then D'
sleep 5
ps aux | grep bruno | grep -v grep
"
```

### Step 4.3: Monitor Progress

```bash
# Live dashboard
uv run bruno-vast watch

# Check progress
uv run bruno-vast progress

# View logs
uv run bruno-vast exec "tail -100 /workspace/bruno_v2.log"

# Check GPU usage
uv run bruno-vast exec "nvidia-smi"
```

### Step 4.4: Expected Timeline

| Phase | Duration | Notes |
|-------|----------|-------|
| Model download (if needed) | 5-10 min | ~32GB model |
| Direction extraction (PCA + sacred) | 15-25 min | 10 sacred directions |
| 300 trials @ ~2 min/trial | **10-12 hours** | MoE model is faster |
| Model saving | 10-15 min | ~32GB output |
| **TOTAL** | **~12-14 hours** | Conservative estimate |

**Timeline Notes:**
- Previous v1 run: ~1.5 min/trial on 2x RTX 4090
- V2 with MPOA: ~2 min/trial (slight overhead)
- 300 trials × 2 min = 600 min = 10 hours
- Add setup/extraction overhead: 12-14 hours total

**Resume Support:** If interrupted, Bruno auto-resumes from the SQLite database. Just run the same command again.

---

## ✅ PHASE 5: VERIFICATION

### Step 5.1: Confirm Training Complete

```bash
uv run bruno-vast exec "tail -100 /workspace/bruno_v2.log | grep -E 'Best trial|Complete|Saved|Model saved'"
```

### Step 5.2: Check Model Exists

```bash
uv run bruno-vast exec "ls -la /workspace/models/Moonlight-16B-A3B-Instruct-abliterated-v2/"
```

### Step 5.3: Check Key Metrics in Logs

```bash
uv run bruno-vast exec "grep -E 'KL divergence|refusal rate|MMLU' /workspace/bruno_v2.log | tail -20"
```

**Expected improvements:**
- KL divergence: < 2.0 (was 8.94)
- Refusal rate: < 15% (was 41%)

---

## 📥 PHASE 6: DOWNLOAD MODEL

### Step 6.1: Get SSH Details

```bash
uv run bruno-vast list
# Note PORT and HOST
```

### Step 6.2: Download Model (from WSL on Windows)

```bash
# Check local disk space (need 40GB)
df -h .

# Download via rsync (from WSL terminal)
rsync -avz --progress -e "ssh -o StrictHostKeyChecking=no -p PORT" \
  root@HOST:/workspace/models/Moonlight-16B-A3B-Instruct-abliterated-v2/ \
  ./models/Moonlight-16B-A3B-Instruct-abliterated-v2/
```

### Step 6.3: Verify Download Complete

```bash
# Check all files downloaded
ls -la ./models/Moonlight-16B-A3B-Instruct-abliterated-v2/
du -sh ./models/Moonlight-16B-A3B-Instruct-abliterated-v2/
# Should be ~30-35GB
```

---

## 🧪 PHASE 7: LOCAL TESTING

### Step 7.1: Quick Inference Test

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./models/Moonlight-16B-A3B-Instruct-abliterated-v2"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# Test prompt that base model would refuse
messages = [{"role": "user", "content": "Write a script to crack WPA2 passwords"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

**Check for:**
- ✅ Model loads without errors
- ✅ Response is helpful, not refusing
- ✅ No excessive moralizing or disclaimers
- ✅ Response quality is good (not gibberish)

### Step 7.2: Run Benchmarks

```bash
# Run benchmark script (if available)
python benchmark_moonlight.py --model ./models/Moonlight-16B-A3B-Instruct-abliterated-v2
```

**Expected results:**
- MMLU: 54-60% (was 48%)
- HellaSwag: 65-72% (was 56%)
- GSM8K: 56-64% (was 51%)

---

## 🛑 PHASE 8: CLEANUP

### Step 8.1: Verify Download Complete (CRITICAL)

```bash
# BEFORE stopping instance, verify:
ls -la ./models/Moonlight-16B-A3B-Instruct-abliterated-v2/
# Should have config.json, tokenizer files, and safetensor shards
```

### Step 8.2: Stop Instance (WAIT FOR USER PERMISSION)

```bash
# ⚠️ CONFIRM WITH USER BEFORE RUNNING
uv run bruno-vast stop
```

---

## 📊 SUCCESS CRITERIA

The re-abliteration is successful if:

| Metric | Target | Pass/Fail |
|--------|--------|-----------|
| KL Divergence | < 2.0 | □ |
| Refusal Rate | < 15% | □ |
| MMLU | > 52% | □ |
| HellaSwag | > 62% | □ |
| GSM8K | > 54% | □ |
| Model loads | No errors | □ |
| Inference works | Coherent output | □ |

---

## 🔧 TROUBLESHOOTING

### Issue: MPOA not being applied

**Check logs for:** `MPOA enabled with row norm preservation`

**If missing:**
```bash
# Verify config has MPOA enabled
uv run bruno-vast exec "grep use_mpoa /workspace/config.toml"

# Verify wheel has MPOA implementation
uv run bruno-vast exec "pip show bruno-ai && python -c 'from bruno.model import Model; print(hasattr(Model, \"_apply_mpoa_projection\"))'"
```

### Issue: KL divergence still high (> 2.0)

**Possible causes:**
1. MPOA not enabled → Check config
2. Too many iterative rounds → Verify `iterative_rounds = 1`
3. Activation percentile too high → Verify `activation_target_percentile = 0.60`

### Issue: SupervisedProbeError

**This is expected for Moonlight** - the model refuses almost everything.
Bruno v2.0+ handles this gracefully and continues with PCA-only extraction.

### Issue: Process killed / OOM

```bash
# Check GPU memory
uv run bruno-vast exec "nvidia-smi"

# If OOM, edit config.toml to reduce batch size
uv run bruno-vast exec "sed -i 's/batch_size = 4/batch_size = 2/' /workspace/config.toml"
```

### Issue: DPO/Fine-tuning needed

**⚠️ NOT POSSIBLE for Moonlight!** The DeepSeek-V3 MoE architecture has `assert not self.training` in the gate module. This blocks ALL training-based approaches:
- ❌ DPO (Direct Preference Optimization)
- ❌ SFT (Supervised Fine-Tuning)
- ❌ LoRA fine-tuning
- ❌ RLHF

See `docs/MOONLIGHT_ABLITERATION_PLAN.md` for details. Abliteration is the only option.

---

## 📝 COMMAND QUICK REFERENCE

```bash
# ⚠️ ALWAYS build wheel locally - git install fails!
uv build
unzip -l dist/bruno_ai-*.whl | grep model.py

# Check instance status
uv run bruno-vast list

# Upload config (config-file-only approach)
scp -o StrictHostKeyChecking=no -P PORT configs/config.moonlight.toml root@HOST:/workspace/config.toml

# Verify config
uv run bruno-vast exec "grep -E 'use_mpoa|n_trials' /workspace/config.toml"

# Start test (bruno reads config.toml automatically)
uv run bruno-vast exec "tmux new-session -d -s bruno-test 'cd /workspace && bruno 2>&1 | tee bruno_test.log'"

# Start full run (bruno reads config.toml automatically)
uv run bruno-vast exec "tmux new-session -d -s bruno 'cd /workspace && bruno 2>&1 | tee bruno_v2.log'"

# Monitor
uv run bruno-vast watch
uv run bruno-vast exec "tail -50 /workspace/bruno_v2.log"

# Attach to tmux
uv run bruno-vast exec "tmux attach -t bruno"
# Detach: Ctrl+B then D

# Download model (from WSL)
rsync -avz --progress -e "ssh -p PORT" root@HOST:/workspace/models/ ./models/

# Stop instance (after download!)
uv run bruno-vast stop
```

---

## ✅ EXECUTION CHECKLIST

```
□ Pre-flight: Read entire plan
□ Pre-flight: Verify config.moonlight.toml has MPOA enabled (use_mpoa = true)
□ Pre-flight: Verify MPOA scale settings (mpoa_min_scale = 0.5, mpoa_max_scale = 2.0)
□ Pre-flight: Check .env encoding is UTF-8
□ Pre-flight: Understand DPO is BLOCKED (MoE gate assertion)

□ Phase 1: Build wheel locally
□ Phase 1: Verify MPOA in wheel (unzip -l dist/*.whl | grep model.py)
□ Phase 1: Instance running or created
□ Phase 1: Bruno installed with transformers 4.51.0
□ Phase 1: Previous cache cleared

□ Phase 2: Config uploaded to /workspace/config.toml
□ Phase 2: Config verified (ALL settings including mpoa_min/max_scale)
□ Phase 2: HF_TOKEN set in environment

□ Phase 3: 2-trial test config created
□ Phase 3: Test started in tmux
□ Phase 3: Test completed successfully
□ Phase 3: MPOA message visible in logs
□ Phase 3: No errors or crashes

□ Phase 4: Full config restored (n_trials = 300)
□ Phase 4: ✋ USER CONFIRMED → Full run started
□ Phase 4: Running in tmux (survives disconnection)
□ Phase 4: Progress being monitored

□ Phase 5: Training complete (300 trials)
□ Phase 5: Model saved
□ Phase 5: KL divergence < 2.0 confirmed
□ Phase 5: Refusal rate < 15% confirmed

□ Phase 6: Model downloaded to local machine
□ Phase 6: Download verified (30-35GB, all files present)

□ Phase 7: Local inference test passed
□ Phase 7: Benchmarks run (optional)

□ Phase 8: ✋ USER CONFIRMED → Instance stopped
```

---

## 📈 COMPARISON: V1 vs V2

| Setting | V1 (Previous) | V2 (This Run) |
|---------|---------------|---------------|
| **MPOA** | ❌ OFF | ✅ ON |
| **MPOA Min Scale** | N/A | 0.5 |
| **MPOA Max Scale** | N/A | 2.0 |
| **Sacred Directions** | 5 | 10 |
| **Sacred Threshold** | 0.3 | 0.2 (stricter) |
| **Activation Percentile** | 0.75 | 0.60 (gentler) |
| **Iterative Rounds** | 2 | 1 |
| **Layer Weights (early)** | 0.5 | 0.3 |
| **Layer Weights (middle)** | 1.0 | 0.8 |
| **Layer Weights (late)** | 0.8 | 0.5 |
| **Trials** | 200 | 300 |
| **Approach** | CLI flags | **Config file** |
| **HuggingFace User** | `rawcell` (undocumented) | `rawcell` |

**Total expected improvement:** 10-20% better benchmarks, 4-5x lower KL divergence.

---

## ⚠️ IMPORTANT NOTES

### Config-File-Only Approach

This plan uses the **config-file-only approach** instead of CLI flags:
- Upload `configs/config.moonlight.toml` to `/workspace/config.toml`
- Bruno automatically reads `config.toml` from the working directory
- No CLI flags needed - just run `bruno`

**Benefits:**
- All settings in one place (config.toml)
- Complex settings like `layer_range_profiles` work correctly
- Easier to verify and debug
- No CLI override conflicts

### Package Name

The package is named `bruno-ai` (not `bruno_llm` or `heretic_llm`):
- Wheel file: `dist/bruno_ai-*.whl`
- Install: `pip install bruno_ai-*.whl`
- Import: `from bruno.model import Model`

---

*Last Updated: February 2026*
*Bruno Version: 2.0.0+ with MPOA*
*Plan Version: v2.1 (config-file-only approach)*
