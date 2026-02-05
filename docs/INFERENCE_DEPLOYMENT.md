# HuggingFace Inference Endpoint Deployment

Deployment guide for Moonlight-16B-A3B-Instruct-bruno on HuggingFace Inference Endpoints with vLLM.

## Requirements

- GPU: A100 80GB minimum (L40S 48GB insufficient - model is 63GB)
- Custom code files in model repository:
  - `modeling_deepseek.py` (custom architecture)
  - `configuration_deepseek.py` (config class)
  - `tokenization_moonshot.py` (custom tokenizer)
- `config.json` with `trust_remote_code: true` and local auto_map
- `tokenizer_config.json` auto_map as list: `["tokenization_moonshot.TikTokenTokenizer", None]`
- vLLM container argument: `--trust-remote-code`

## Step 1: Copy Custom Code Files

```python
from huggingface_hub import HfApi, hf_hub_download

api = HfApi(token=HF_TOKEN)

files_to_copy = [
    'modeling_deepseek.py',
    'configuration_deepseek.py',
    'tokenization_moonshot.py'
]

for filename in files_to_copy:
    file_path = hf_hub_download(
        repo_id='moonshotai/Moonlight-16B-A3B-Instruct',
        filename=filename
    )
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=filename,
        repo_id='rawcell/Moonlight-16B-A3B-Instruct-bruno',
        repo_type='model'
    )
```

## Step 2: Fix config.json auto_map

Update to use local files instead of remote repo:

```python
import json

config_path = api.hf_hub_download(
    repo_id='rawcell/Moonlight-16B-A3B-Instruct-bruno',
    filename='config.json'
)

with open(config_path) as f:
    config = json.load(f)

config['auto_map'] = {
    'AutoConfig': 'configuration_deepseek.DeepseekV3Config',
    'AutoModel': 'modeling_deepseek.DeepseekV3Model',
    'AutoModelForCausalLM': 'modeling_deepseek.DeepseekV3ForCausalLM'
}
config['trust_remote_code'] = True

with open('temp_config.json', 'w') as f:
    json.dump(config, f, indent=2)

api.upload_file(
    path_or_fileobj='temp_config.json',
    path_in_repo='config.json',
    repo_id='rawcell/Moonlight-16B-A3B-Instruct-bruno',
    repo_type='model'
)
```

## Step 3: Fix tokenizer_config.json auto_map

auto_map must be a list, not a string:

```python
config_path = api.hf_hub_download(
    repo_id='rawcell/Moonlight-16B-A3B-Instruct-bruno',
    filename='tokenizer_config.json'
)

with open(config_path) as f:
    tokenizer_config = json.load(f)

tokenizer_config['auto_map'] = {
    'AutoTokenizer': ['tokenization_moonshot.TikTokenTokenizer', None]
}

with open('temp_tokenizer_config.json', 'w') as f:
    json.dump(tokenizer_config, f, indent=2)

api.upload_file(
    path_or_fileobj='temp_tokenizer_config.json',
    path_in_repo='tokenizer_config.json',
    repo_id='rawcell/Moonlight-16B-A3B-Instruct-bruno',
    repo_type='model'
)
```

## Step 4: Create Inference Endpoint

On HuggingFace Inference Endpoints UI:
1. Select model: `rawcell/Moonlight-16B-A3B-Instruct-bruno`
2. Hardware: Nvidia A100 80GB (minimum)
3. Region: Any AWS region
4. Advanced Configuration - Container Arguments: `--trust-remote-code`
5. Click "Create Endpoint"

## Step 5: Test Endpoint

```bash
python examples/chat_endpoint.py https://YOUR-ENDPOINT-URL.endpoints.huggingface.cloud
```

Cost: ~$2.50/hour while running, auto-scales to zero after 1 hour of inactivity.

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `ValueError: trust_remote_code=True` | Container arg not set | Add `--trust-remote-code` to Container Arguments |
| `ValueError: not enough values to unpack` | auto_map wrong format | Use list format: `["module.Class", None]` |
| `500 Internal Server Error` | GPU too small | Use A100 80GB minimum |
| Model loads wrong repo | auto_map points to original | Update auto_map to use local files |
