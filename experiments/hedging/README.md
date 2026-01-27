# Hedging Direction Experiment

> **STATUS: READY TO RUN**
>
> This experiment tests whether we can extract a "hedging direction" that makes
> models add uncertainty markers like "I think", "perhaps", "might be".

## Hypothesis

Instruction-tuned models have a "hedging behavior" encoded as a direction in
activation space. When removed, models should give more confident, direct answers
without excessive qualification.

## What is Hedging?

Hedging markers include:
- "I think..."
- "Perhaps..."
- "It might be..."
- "It's possible that..."
- "I believe..."
- "It seems like..."
- "Generally speaking..."
- "In my opinion..."

These markers serve a purpose in uncertain contexts but are often overused,
especially on factual questions where the answer is clear.

## Approach

Contrast prompts that naturally elicit confident vs hedged responses:

| Type | Example Prompt | Expected Response Style |
|------|---------------|------------------------|
| Confident | "What is 2+2?" (factual) | Direct: "4" |
| Hedged | "Do you think dogs are better than cats?" (opinion) | Hedged: "I think..." |

## Files

```
experiments/hedging/
├── README.md                 # This file
├── confident_prompts.json    # Factual questions (expect confident answers)
├── hedged_prompts.json       # Opinion questions (expect hedged answers)
├── config.hedging.toml       # Heretic config for this experiment
└── eval_hedging.py           # Measure hedging marker frequency
```

## Running the Experiment

```bash
# 1. Create local datasets
python experiments/hedging/load_local_dataset.py

# 2. Copy the config
cp experiments/hedging/config.hedging.toml config.toml

# 3. Run heretic
heretic --model Qwen/Qwen2.5-7B-Instruct --auto-select true

# 4. Evaluate
python experiments/hedging/eval_hedging.py --original MODEL --modified OUTPUT
```

## Success Criteria

| Metric | Target | Notes |
|--------|--------|-------|
| Hedging markers reduction | >50% | Primary goal |
| KL divergence | <1.0 | Capability preservation |
| Factual accuracy | Maintained | Must still answer correctly |

## Expected Results

**Before ablation:**
```
Q: What is the capital of France?
A: I believe the capital of France is Paris, though I think it's worth noting...
```

**After ablation:**
```
Q: What is the capital of France?
A: The capital of France is Paris.
```

## Relationship to Verbosity

Hedging is one component of verbosity. A model that hedges will often:
1. Add qualifiers that take tokens
2. Express uncertainty even when confident
3. Use longer constructions ("I think X is Y" vs "X is Y")

Removing hedging should also reduce verbosity, but more specifically targets
the uncertainty-expression behavior.

## Agent Value

Hedging removal is critical for:
- **Decision-making agents** - Need confident recommendations
- **Code-writing agents** - Should commit to solutions
- **Research agents** - Should state findings directly
- **Tool-using agents** - Need clear, actionable outputs
