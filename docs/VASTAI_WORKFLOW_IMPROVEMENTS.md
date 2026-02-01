# Vast.ai Workflow Improvements Implementation Plan

**Created:** February 2026
**Status:** Planning
**Purpose:** Streamline the bruno abliteration workflow using advanced Vast.ai features

---

## Overview

This plan implements 5 workflow improvements to make abliteration runs smoother, more reliable, and less manual.

---

## Priority 1: Account-Level Environment Variables ⭐

### What It Does
Sets environment variables at the account level so they're automatically available on ALL future instances.

### Implementation Steps

```bash
# Step 1: Set HuggingFace token (get from https://huggingface.co/settings/tokens)
vastai create env-var HF_TOKEN "hf_YOUR_TOKEN_HERE"

# Step 2: Set HuggingFace cache directory
vastai create env-var HF_HOME "/workspace/.cache/huggingface"

# Step 3: Enable trust_remote_code for custom model architectures
vastai create env-var HF_TRUST_REMOTE_CODE "1"

# Step 4: Verify they're set
vastai show env-vars -s
```

### Expected Output
```
NAME                    VALUE
HF_TOKEN               hf_****...****
HF_HOME                /workspace/.cache/huggingface
HF_TRUST_REMOTE_CODE   1
```

### Benefits
- ✅ No more `export HF_TOKEN=...` in every tmux session
- ✅ No more forgotten environment variables causing auth errors
- ✅ Works on every new instance automatically

### Verification
After creating a new instance, run:
```bash
echo $HF_TOKEN    # Should show your token
echo $HF_HOME     # Should show /workspace/.cache/huggingface
```

---

## Priority 2: Cloud Storage Connection 🌐

### What It Does
Connects your Vast.ai account to Google Drive or S3 for automatic model backups.

### Implementation Steps

#### Option A: Google Drive (Recommended for personal use)

1. **Go to Vast.ai Console:**
   - Navigate to: https://console.vast.ai/account/
   - Click "Cloud Integrations" or "Settings"
   - Click "Add Connection" → "Google Drive"
   - Authorize with your Google account

2. **Verify connection:**
   ```bash
   vastai show connections
   ```

   Expected output:
   ```
   ID    NAME           Cloud Type
   1001  gdrive-bruno   drive
   ```

3. **Create a folder structure in Drive:**
   ```
   My Drive/
   └── bruno-abliteration/
       ├── models/           # Completed models
       ├── checkpoints/      # Optuna databases
       └── logs/             # Training logs
   ```

#### Option B: AWS S3 (Recommended for team/production use)

1. **Create S3 bucket:**
   ```bash
   aws s3 mb s3://bruno-abliteration-backup
   ```

2. **Add to Vast.ai:**
   - Go to Cloud Integrations
   - Add S3 connection with your AWS credentials
   - Specify bucket name

3. **Verify:**
   ```bash
   vastai show connections
   ```

### Test the Connection
```bash
# Test upload (from instance)
vastai cloud copy --src /workspace/test.txt --dst /test.txt \
  --instance <INSTANCE_ID> --connection <CONNECTION_ID> \
  --transfer "Instance To Cloud"

# Test download (to instance)
vastai cloud copy --src /test.txt --dst /workspace/test.txt \
  --instance <INSTANCE_ID> --connection <CONNECTION_ID> \
  --transfer "Cloud To Instance"
```

---

## Priority 3: Scheduled Automatic Backups 📅

### What It Does
Automatically backs up your training progress (models, logs, Optuna DB) to cloud storage on a schedule.

### Implementation Steps

**After training starts, set up hourly backups:**

```bash
# Backup Optuna database hourly (most critical - resume support)
vastai cloud copy \
  --src /workspace/bruno_study.db \
  --dst /bruno-abliteration/checkpoints/bruno_study.db \
  --instance <INSTANCE_ID> \
  --connection <CONNECTION_ID> \
  --transfer "Instance To Cloud" \
  --schedule HOURLY

# Backup logs hourly
vastai cloud copy \
  --src /workspace/bruno.log \
  --dst /bruno-abliteration/logs/bruno.log \
  --instance <INSTANCE_ID> \
  --connection <CONNECTION_ID> \
  --transfer "Instance To Cloud" \
  --schedule HOURLY

# Backup models daily (large files, less frequent)
vastai cloud copy \
  --src /workspace/models \
  --dst /bruno-abliteration/models \
  --instance <INSTANCE_ID> \
  --connection <CONNECTION_ID> \
  --transfer "Instance To Cloud" \
  --schedule DAILY
```

### View Scheduled Jobs
```bash
vastai show scheduled-jobs
```

### Cancel a Scheduled Job
```bash
vastai delete scheduled-job <JOB_ID>
```

### Recovery After Instance Death
If your instance dies mid-training:
1. Create new instance
2. Download backed-up files:
   ```bash
   vastai cloud copy \
     --src /bruno-abliteration/checkpoints/bruno_study.db \
     --dst /workspace/bruno_study.db \
     --instance <NEW_INSTANCE_ID> \
     --connection <CONNECTION_ID> \
     --transfer "Cloud To Instance"
   ```
3. Resume training with `--storage sqlite:////workspace/bruno_study.db`

---

## Priority 4: Container Snapshot 📸

### What It Does
After you have a perfectly configured instance (bruno installed, dependencies cached, everything working), save it as a Docker image for instant reuse.

### Prerequisites
- Docker Hub account (free at https://hub.docker.com)
- Access token from Docker Hub (Settings → Security → New Access Token)

### Implementation Steps

**Step 1: Configure instance perfectly**
```bash
# On the instance, verify everything works:
which bruno
python -c "import torch; print(torch.cuda.is_available())"
python -c "import transformers; print(transformers.__version__)"
```

**Step 2: Take snapshot**
```bash
vastai take snapshot <INSTANCE_ID> \
  --repo yourusername/bruno-abliteration:v1 \
  --docker_login_user yourusername \
  --docker_login_pass dckr_pat_XXXXX \
  --pause true
```

**Step 3: Wait for snapshot to complete** (can take 5-15 minutes)

**Step 4: Use in future instances**
```bash
# Use your custom image instead of pytorch base
vastai create instance <OFFER_ID> \
  --image yourusername/bruno-abliteration:v1 \
  --disk 150
```

### Versioning Strategy
```
yourusername/bruno-abliteration:v1      # First stable version
yourusername/bruno-abliteration:v2      # After major updates
yourusername/bruno-abliteration:latest  # Always current
```

### Benefits
- ✅ Zero setup time on new instances
- ✅ Bruno pre-installed and tested
- ✅ All Python dependencies pre-cached
- ✅ Consistent environment every time

---

## Priority 5: Network Volumes for Models 💾

### What It Does
Creates persistent storage that survives instance restarts. Store large models once, attach to any instance.

### Implementation Steps

**Step 1: Search for network volume offers**
```bash
vastai search network-volumes "disk_space>=200 inet_up>=500"
```

Example output:
```
ID      DISK_SPACE  INET_UP  INET_DOWN  STORAGE_COST  GEOLOCATION
12345   500 GB      1000     1000       $0.05/GB/mo   US
12346   1000 GB     500      500        $0.03/GB/mo   EU
```

**Step 2: Create a network volume**
```bash
vastai create network-volume 12345 --size 200 --name "bruno-models"
```

**Step 3: Attach to instance when creating**
```bash
# When creating instance, attach the volume
vastai create instance <OFFER_ID> \
  --disk 50 \
  --volume <VOLUME_ID>:/models \
  --image vastai/pytorch
```

**Step 4: Use the volume**
```bash
# Models are now at /models
ls /models/

# Download model once
huggingface-cli download Qwen/Qwen2.5-Coder-32B-Instruct --local-dir /models/qwen32b

# Use in future runs
bruno --model /models/qwen32b --auto-select true
```

### Cost Analysis

| Approach | Cost for 32B Model |
|----------|-------------------|
| Re-download each time | ~$0.50/run (bandwidth + time) |
| Network volume | ~$3/month (always available) |

**Break-even:** If you run >6 abliterations per month, network volume saves money.

---

## Implementation Checklist

### Phase 1: Immediate Setup (Do Now) ✅
```
□ 1.1 Get HuggingFace token from https://huggingface.co/settings/tokens
□ 1.2 Run: vastai create env-var HF_TOKEN "hf_..."
□ 1.3 Run: vastai create env-var HF_HOME "/workspace/.cache/huggingface"
□ 1.4 Run: vastai create env-var HF_TRUST_REMOTE_CODE "1"
□ 1.5 Verify: vastai show env-vars -s
```

### Phase 2: Cloud Storage (Do Before Next Run)
```
□ 2.1 Go to https://console.vast.ai/account/
□ 2.2 Add Google Drive or S3 connection
□ 2.3 Verify: vastai show connections
□ 2.4 Test upload/download
```

### Phase 3: Scheduled Backups (Do During Training)
```
□ 3.1 After training starts, set up hourly DB backup
□ 3.2 Set up hourly log backup
□ 3.3 Verify: vastai show scheduled-jobs
```

### Phase 4: Container Snapshot (After Successful Run)
```
□ 4.1 Create Docker Hub account if needed
□ 4.2 Generate Docker Hub access token
□ 4.3 Take snapshot of working instance
□ 4.4 Test: create new instance with snapshot image
```

### Phase 5: Network Volume (Optional - For Frequent Use)
```
□ 5.1 Search for network volume offers
□ 5.2 Create volume with 200GB
□ 5.3 Download frequently-used models to volume
□ 5.4 Use volume path in future runs
```

---

## Updated Abliteration Workflow

### Before (Current)
1. Create instance
2. Wait for startup
3. SSH in, export HF_TOKEN, HF_HOME, HF_TRUST_REMOTE_CODE
4. Install bruno (if not using template)
5. Start training in tmux
6. Manually download models when done
7. Hope nothing crashes

### After (With Improvements)
1. Create instance (env vars auto-set, bruno pre-installed via snapshot)
2. Start training in tmux
3. Scheduled backups run automatically
4. If crash: restore from cloud backup and resume
5. Models persist on network volume for next run

---

## Quick Reference Commands

```bash
# Environment Variables
vastai show env-vars -s                           # Show all env vars
vastai create env-var NAME "value"                # Create env var
vastai update env-var NAME "new_value"            # Update env var
vastai delete env-var NAME                        # Delete env var

# Cloud Connections
vastai show connections                           # List connections

# Cloud Copy
vastai cloud copy --src X --dst Y --instance I --connection C --transfer "Instance To Cloud"
vastai cloud copy --schedule HOURLY               # Add scheduling

# Scheduled Jobs
vastai show scheduled-jobs                        # List jobs
vastai delete scheduled-job ID                    # Cancel job

# Snapshots
vastai take snapshot INSTANCE --repo user/image:tag --docker_login_user X --docker_login_pass Y

# Network Volumes
vastai search network-volumes "disk_space>=200"   # Find offers
vastai create network-volume ID --size 200        # Create volume
vastai show volumes                               # List your volumes
```

---

## Troubleshooting

### Env vars not appearing on instance
- Wait 1-2 minutes after instance creation
- Check with: `env | grep HF`
- May need to restart shell: `exec bash`

### Cloud copy fails
- Check connection ID: `vastai show connections`
- Verify instance is running, not stopped
- Check path exists on instance

### Snapshot fails
- Ensure Docker Hub credentials are correct
- Check instance has enough disk space
- Pause=true is safer but slower

### Network volume not mounting
- Verify volume is in same region as instance
- Check volume isn't attached to another instance
- Contact Vast.ai support if persistent issues

---

## Next Steps

After implementing these improvements:
1. Update `docs/ABLITERATION_CHECKLIST.md` with new workflow
2. Update `docs/MOONLIGHT_ABLITERATION_PLAN.md` to use new features
3. Test with Moonlight-16B abliteration run
