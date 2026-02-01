# HuggingFace Inference Endpoint Deployment Guide

This guide documents the complete process for deploying an abliterated model to HuggingFace Inference Endpoints, based on the successful deployment of `rawcell/Qwen2.5-Coder-7B-Instruct-bruno`.

---

## Overview

After running bruno abliteration on a model, you can deploy it to HuggingFace for cloud inference. This eliminates the need for local GPU resources when using the model.

**Deployed Model:** https://huggingface.co/rawcell/Qwen2.5-Coder-7B-Instruct-bruno

---

## Prerequisites

1. **Abliterated model** - Either local or already on HuggingFace
2. **HuggingFace account** with write access token
3. **huggingface_hub** Python package installed

```bash
pip install huggingface_hub
```

---

## Step 1: Upload Model to HuggingFace

If your model is local, upload it:

```python
from huggingface_hub import HfApi

api = HfApi(token="hf_YOUR_TOKEN")

# Create repository
api.create_repo(repo_id="your-username/model-name", private=False)

# Upload model files
api.upload_folder(
    folder_path="./models/your-model",
    repo_id="your-username/model-name",
    commit_message="Upload abliterated model"
)
```

---

## Step 2: Create Custom Handler (handler.py)

HuggingFace Inference Endpoints require a custom handler for non-standard models.

**Upload this file as `handler.py` in your model repository:**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class EndpointHandler:
    def __init__(self, path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def __call__(self, data):
        inputs = data.get("inputs", data)
        parameters = data.get("parameters", {})

        # Handle both string prompts and chat message format
        if isinstance(inputs, list):
            # Chat format: [{"role": "user", "content": "..."}]
            prompt = self.tokenizer.apply_chat_template(
                inputs,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Simple string prompt
            prompt = inputs

        # Tokenize
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)

        # Generate
        max_new_tokens = parameters.get("max_new_tokens", 512)
        temperature = parameters.get("temperature", 0.7)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return [{"generated_text": generated}]
```

> **⚠️ Important Response Format Note:** When sending chat messages (list format), the endpoint returns `generated_text` as a **list of message dicts**, not a plain string:
> ```python
> # Response format for chat messages:
> [{"generated_text": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "The actual response"}]}]
>
> # Response format for plain string prompts:
> [{"generated_text": "The actual response as a string"}]
> ```
> Your client code must handle both formats. See `examples/chat_app.py` `EndpointManager` class for a reference implementation.

**Upload command:**
```python
from huggingface_hub import HfApi

handler_code = '''...'''  # The handler code above

with open('handler.py', 'w') as f:
    f.write(handler_code)

api = HfApi(token="hf_YOUR_TOKEN")
api.upload_file(
    path_or_fileobj='handler.py',
    path_in_repo='handler.py',
    repo_id='your-username/model-name',
    commit_message='Add custom inference handler'
)
```

---

## Step 3: Create Requirements File (requirements.txt)

Specify the dependencies for the inference endpoint:

```
torch>=2.0.0
transformers>=4.40.0
accelerate>=0.27.0
```

**Upload:**
```python
api.upload_file(
    path_or_fileobj='requirements.txt',
    path_in_repo='requirements.txt',
    repo_id='your-username/model-name',
    commit_message='Add requirements.txt'
)
```

---

## Step 4: Create Model Card (README.md)

The model card provides metadata that helps HuggingFace recognize the model type:

```markdown
---
license: apache-2.0
language:
- en
base_model: Qwen/Qwen2.5-Coder-7B-Instruct
tags:
- text-generation
- conversational
- qwen2
- abliterated
- uncensored
pipeline_tag: text-generation
library_name: transformers
inference:
  parameters:
    max_new_tokens: 512
    temperature: 0.7
---

# Qwen2.5-Coder-7B-Instruct-Bruno (Abliterated)

This is an abliterated version of [Qwen/Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) with reduced refusals.

## Model Details

- **Base Model:** Qwen/Qwen2.5-Coder-7B-Instruct
- **Modification:** Abliteration (refusal direction removal)
- **Architecture:** Qwen2ForCausalLM
- **Parameters:** 7B
- **Context Length:** 32,768 tokens

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "your-username/model-name"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

messages = [{"role": "user", "content": "Hello!"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Disclaimer

This model has been modified to reduce refusals. Use responsibly.
```

---

## Step 5: Fix Tokenizer Configuration (CRITICAL!)

**⚠️ bruno saves `tokenizer_config.json` with `extra_special_tokens` as a list `[]` instead of a dict `{}`.**

This causes the error:
```
AttributeError: 'list' object has no attribute 'keys'
```

**Fix the tokenizer_config.json:**

```python
import json
from huggingface_hub import HfApi, hf_hub_download

token = 'hf_YOUR_TOKEN'
repo_id = 'your-username/model-name'

# Download existing config
path = hf_hub_download(repo_id=repo_id, filename='tokenizer_config.json', token=token)
with open(path, 'r', encoding='utf-8') as f:
    config = json.load(f)

# Fix extra_special_tokens: list -> dict
if 'extra_special_tokens' in config and isinstance(config['extra_special_tokens'], list):
    config['extra_special_tokens'] = {}
    print('Fixed extra_special_tokens: list -> dict')

# Save locally
with open('tokenizer_config_fixed.json', 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=2)

# Upload fix
api = HfApi(token=token)
api.upload_file(
    path_or_fileobj='tokenizer_config_fixed.json',
    path_in_repo='tokenizer_config.json',
    repo_id=repo_id,
    commit_message='Fix extra_special_tokens: list to dict'
)
```

---

## Step 6: Add Chat Template (If Missing)

Qwen models need a chat template in `tokenizer_config.json`. If missing, add it.

**Option 1: Copy from reference file**

The complete Qwen chat template is in `tokenizer_config_fixed.json` in the project root. Copy the `chat_template` field from that file.

**Option 2: Use the full template below**

```python
import json
from huggingface_hub import HfApi, hf_hub_download

# The complete Qwen chat template (Jinja2 format)
chat_template = """{%- if tools %}
    {{- '<|im_start|>system\\n' }}
    {%- if messages[0]['role'] == 'system' %}
        {{- messages[0]['content'] }}
    {%- else %}
        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}
    {%- endif %}
    {{- "\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}
    {%- for tool in tools %}
        {{- "\\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}
{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}
    {%- else %}
        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}
        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role }}
        {%- if message.content %}
            {{- '\\n' + message.content }}
        {%- endif %}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- '\\n<tool_call>\\n{"name": "' }}
            {{- tool_call.name }}
            {{- '", "arguments": ' }}
            {{- tool_call.arguments | tojson }}
            {{- '}\\n</tool_call>' }}
        {%- endfor %}
        {{- '<|im_end|>\\n' }}
    {%- elif message.role == "tool" %}
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\\n<tool_response>\\n' }}
        {{- message.content }}
        {{- '\\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
{%- endif %}"""

token = 'hf_YOUR_TOKEN'
repo_id = 'your-username/model-name'

# Download existing config
path = hf_hub_download(repo_id=repo_id, filename='tokenizer_config.json', token=token)
with open(path, 'r', encoding='utf-8') as f:
    config = json.load(f)

# Add chat template
config['chat_template'] = chat_template

# Save and upload
with open('tokenizer_config_updated.json', 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=2)

api = HfApi(token=token)
api.upload_file(
    path_or_fileobj='tokenizer_config_updated.json',
    path_in_repo='tokenizer_config.json',
    repo_id=repo_id,
    commit_message='Add Qwen chat template'
)
```

---

## Step 7: Create Inference Endpoint

1. Go to https://huggingface.co/your-username/model-name
2. Click "Deploy" → "Inference Endpoints"
3. Configure:
   - **Instance type:** CPU or GPU (GPU recommended for 7B+)
   - **Region:** Choose closest to your users
   - **Security:** Public or Private
4. Click "Create Endpoint"
5. Wait for status to change to "Running" (~2-5 minutes)

---

## Step 8: Test the Endpoint

Your endpoint URL will look like: `https://<random-id>.<region>.aws.endpoints.huggingface.cloud`

Example: `https://o1g59zji7ahyimsa.us-east-1.aws.endpoints.huggingface.cloud`

### cURL

```bash
curl https://YOUR_ENDPOINT_URL \
  -X POST \
  -H "Authorization: Bearer hf_YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"inputs": "Write a Python function", "parameters": {"max_new_tokens": 512}}'
```

### Python

```python
import requests

API_URL = "https://YOUR_ENDPOINT_URL"
headers = {"Authorization": "Bearer hf_YOUR_TOKEN"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

result = query({
    "inputs": [{"role": "user", "content": "Hello!"}],
    "parameters": {"max_new_tokens": 512}
})
print(result)
```

### HuggingFace Hub Client

```python
from huggingface_hub import InferenceClient

client = InferenceClient(
    "https://YOUR_ENDPOINT_URL",
    token="hf_YOUR_TOKEN"
)

response = client.text_generation(
    "Write a Python function",
    max_new_tokens=512
)
print(response)
```

---

## Step 9: Use with Chat App

The chat app (`examples/chat_app.py`) supports both local models and HuggingFace endpoints.

**The endpoint is already configured** in `examples/chat_app.py`:

```python
# Already configured in chat_app.py:
HF_ENDPOINTS: dict[str, dict[str, str]] = {
    "Qwen2.5-Coder-7B-Bruno (Cloud)": {
        "url": "https://o1g59zji7ahyimsa.us-east-1.aws.endpoints.huggingface.cloud",
        "token": "",  # Empty for public endpoints
    },
}
```

**To add additional endpoints**, edit the `HF_ENDPOINTS` dictionary in `examples/chat_app.py`.

Run the chat app:
```bash
python examples/chat_app.py
```

Select the cloud endpoint from the dropdown to chat without using local GPU.

> **Note:** The `EndpointManager` class in `chat_app.py` correctly handles both response formats (string and list of messages). Use it as a reference if implementing your own client.

---

## Common Issues & Solutions

### Issue: "This model is not part of our Model Catalog"

**Cause:** Missing metadata in model card.

**Fix:** Add proper YAML frontmatter with `pipeline_tag: text-generation` and `library_name: transformers`.

---

### Issue: "401 Unauthorized"

**Cause:** Endpoint is private and requires authentication.

**Fix:** Either:
1. Change endpoint to "Public" in settings
2. Include `Authorization: Bearer hf_TOKEN` header in requests

---

### Issue: "AttributeError: 'list' object has no attribute 'keys'"

**Cause:** `extra_special_tokens` in tokenizer_config.json is a list instead of dict.

**Fix:** See Step 5 above.

---

### Issue: Model still refuses/moralizes

**Cause:** Abliteration was run with `--orthogonalize-directions false` or insufficient trials.

**Fix:** Re-run abliteration with all features enabled:
```bash
bruno --model MODEL \
  --orthogonalize-directions true \
  --n-trials 200 \
  --cache-weights true
```

---

## Files Uploaded to HuggingFace

| File | Purpose |
|------|---------|
| `handler.py` | Custom inference handler for Endpoints |
| `requirements.txt` | Python dependencies |
| `README.md` | Model card with metadata |
| `tokenizer_config.json` | Fixed tokenizer config with chat template |
| `config.json` | Model architecture config (from abliteration) |
| `model.safetensors` / `model-*.safetensors` | Model weights |

---

## Cost Considerations

*Prices are approximate as of February 2026. Check [HuggingFace pricing](https://huggingface.co/pricing) for current rates.*

| Instance Type | Cost/Hour | Best For |
|---------------|-----------|----------|
| CPU (16 vCPU, 32GB) | ~$0.54 | Testing, low traffic |
| GPU (T4) | ~$0.60 | 7B models |
| GPU (A10G) | ~$1.50 | 7B-14B models |
| GPU (A100) | ~$4.00 | 32B+ models |

**Scale-to-zero:** Endpoints automatically stop when idle (configurable, default 1 hour). No billing when stopped.

---

## References

- [HuggingFace Inference Endpoints Documentation](https://huggingface.co/docs/inference-endpoints)
- [Custom Handler Documentation](https://huggingface.co/docs/inference-endpoints/guides/custom_handler)
- [bruno Documentation](../CLAUDE.md)
