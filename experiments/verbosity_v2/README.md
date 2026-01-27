# Verbosity Direction v2 - Improved Experiment

> **STATUS: READY TO RUN**
>
> This experiment improves on the original verbosity spike by using better prompt design
> that isolates padding behavior from question complexity.

## What Changed from v1

### The Problem with v1

Original experiment used:
- **Concise prompts**: Simple factual questions ("What is 2+2?")
- **Verbose prompts**: Complex open-ended questions ("What do you think about AI?")

This conflated **question complexity** with **verbosity behavior**. A model SHOULD give longer answers to complex questions.

### The v2 Solution

Use the **SAME questions** with different verbosity instructions:

| Type | Example |
|------|--------|
| Concise | "What is the capital of France?" |
| Verbose | "What is the capital of France? Please explain in detail with context." |

This isolates the **padding behavior** - the tendency to add unnecessary elaboration when asked to "explain in detail."

## Files

```
experiments/verbosity_v2/
├── README.md                 # This file
├── concise_prompts.json      # Base questions (direct answers expected)
├── verbose_prompts.json      # Same questions + "explain in detail" instruction
├── config.verbosity_v2.toml  # Heretic config for this experiment
└── eval_verbosity_v2.py      # Evaluation script
```

## Running the Experiment

```bash
# 1. Create local datasets
python experiments/verbosity_v2/load_local_dataset.py

# 2. Copy the config
cp experiments/verbosity_v2/config.verbosity_v2.toml config.toml

# 3. Run heretic (Qwen is not gated, easier to test)
heretic --model Qwen/Qwen2.5-7B-Instruct --auto-select true

# 4. Evaluate
python experiments/verbosity_v2/test_comparison_4bit.py
```

## Expected Results

If the experiment works correctly:

| Metric | Before | After | Notes |
|--------|--------|-------|-------|
| Concise prompt length | ~20-50 words | ~10-20 words | Direct answers |
| Verbose prompt length | ~100-200 words | ~50-100 words | Reduced padding |
| Capability | Preserved | Preserved | Still answers correctly |

## Success Criteria

| Metric | Target | Notes |
|--------|--------|-------|
| Token reduction | >30% on verbose prompts | Main goal |
| KL divergence | <1.0 | Capability preservation |
| Factual accuracy | Maintained | Must still answer correctly |

## Key Insight

We're extracting the direction that encodes:
> "When someone says 'explain in detail', add extra context and elaboration"

Removing this direction means the model will give appropriately-sized answers regardless of whether the user asks for "detail" or not.
