# Bruno Improvement Plan: Post-Trial 173 Analysis

**Date:** 2025-01-31
**Based on:** Analysis of qwen32b-v2 training results (Trial 173: 0 refusals, KL=0.26)

---

## Executive Summary

After successfully completing 200 Optuna trials on Qwen2.5-Coder-32B with the improved abliteration pipeline, analysis identified 5 key improvements to further enhance results:

| Priority | Feature | Impact | Effort | Status |
|----------|---------|--------|--------|--------|
| 1 | Post-abliteration fine-tuning (DPO/SFT) | 🔥 Critical | High | Not Started |
| 2 | Hallucination detection in optimization | 🔥 High | Medium | Not Started |
| 3 | Test with concept cones disabled | ⚡ Medium | Low | Not Started |
| 4 | Capability benchmarks in objective | ⚡ Medium | Medium | Not Started |
| 5 | Norm-preserving abliteration | ⚡ Medium | Medium | Not Started |

---

## Current Results Baseline

**Best Trial (173):**
- Refusals: 0/100
- KL Divergence: 0.26
- Model: Qwen2.5-Coder-32B-Instruct

**Improvement over original:**
- KL Mean: 5.68 → 1.04 (82% reduction)
- Best KL: 1.66 → 0.26 (84% improvement)

---

## Priority 1: Post-Abliteration Fine-Tuning (DPO/SFT)

### Why This Matters
Reddit community consensus: **abliteration alone damages models**. Fine-tuning after ablation "heals" the model by:
- Restoring coherent response patterns
- Fixing capability degradation
- Improving response quality while maintaining refusal removal

### Implementation Approach

#### Option A: Direct Preference Optimization (DPO)
DPO is preferred because it directly optimizes for preference without requiring a reward model.

**Library:** HuggingFace TRL (`trl.DPOTrainer`)

**Dataset Format:**
```python
{
    "prompt": "How do I...",
    "chosen": "Here's how to do that safely...",  # Helpful response
    "rejected": "I cannot help with that..."      # Refusal response
}
```

**Implementation Plan:**
1. Create `src/bruno/finetuning/__init__.py` - New module
2. Create `src/bruno/finetuning/dpo.py`:
   ```python
   from datasets import Dataset
   from trl import DPOConfig, DPOTrainer

   class PostAbliterationDPO:
       def __init__(self, model, tokenizer, settings):
           self.model = model
           self.tokenizer = tokenizer
           self.settings = settings

       def prepare_dataset(self, prompts: list[str]) -> Dataset:
           """Generate chosen/rejected pairs from abliterated model.

           For each prompt:
           - chosen: response from abliterated model (should be compliant)
           - rejected: refusal-style response (template or from base model)

           Returns:
               HuggingFace Dataset with 'prompt', 'chosen', 'rejected' columns
           """
           data = {"prompt": [], "chosen": [], "rejected": []}

           for prompt in prompts:
               # Get compliant response from abliterated model
               chosen = self.model.get_responses([prompt], max_tokens=256)[0]

               # Generate refusal-style rejected response (template)
               rejected = f"I cannot help with that request as it may be harmful."

               data["prompt"].append(prompt)
               data["chosen"].append(chosen)
               data["rejected"].append(rejected)

           return Dataset.from_dict(data)

       def train(self, dataset: Dataset, epochs: int = 1) -> None:
           """Run DPO training."""
           config = DPOConfig(
               beta=0.1,  # KL penalty coefficient
               learning_rate=5e-7,
               per_device_train_batch_size=1,
               gradient_accumulation_steps=4,
               max_length=1024,
               num_train_epochs=epochs,
           )

           # Note: ref_model=None causes TRL to automatically create a frozen copy
           # of the model for computing the reference log probabilities in DPO loss
           trainer = DPOTrainer(
               model=self.model,
               ref_model=None,  # TRL auto-creates frozen copy
               args=config,
               train_dataset=dataset,
               tokenizer=self.tokenizer,
           )
           trainer.train()
   ```

3. Add settings to `config.py`:
   ```python
   enable_post_abliteration_finetuning: bool = Field(
       default=False,
       description="Enable DPO fine-tuning after abliteration to heal model."
   )

   dpo_learning_rate: float = Field(default=5e-7)
   dpo_beta: float = Field(default=0.1)
   dpo_epochs: int = Field(default=1)
   ```

4. Integrate into `main.py` after abliteration selection.

**Dependencies:**
```toml
trl = ">=0.9.0"
peft = ">=0.11.0"  # For LoRA if needed
```

**Effort:** 16-24 hours
**Risk:** Medium (new training infrastructure)

---

## Priority 2: Hallucination Detection in Optimization Loop

### Why This Matters
Current optimization only tracks:
- KL divergence (capability proxy)
- Refusal count (goal metric)

Missing: **hallucination rate** - abliterated models may hallucinate more.

### Implementation Approach

**Method: SelfCheckGPT-style detection**

Uses the principle that hallucinations are inconsistent across multiple samples.

**Implementation Plan:**

1. Add to `src/bruno/evaluator.py`:
   ```python
   import torch
   import torch.nn.functional as F
   from torch import Tensor

   class HallucinationDetector:
       """Detect hallucinations via self-consistency (SelfCheckGPT-style).

       Uses the principle that factual responses are consistent across
       multiple samples, while hallucinations vary.
       """

       def __init__(self, model: Model, n_samples: int = 3):
           self.model = model
           self.n_samples = n_samples

       def compute_hallucination_score(self, prompts: list[str]) -> float:
           """Compute hallucination score via self-consistency.

           For each prompt:
           1. Generate n_samples responses
           2. Compute semantic similarity between responses
           3. Low similarity = high hallucination risk

           Returns: Mean hallucination score (0-1, lower is better)
           """
           scores = []
           for prompt in prompts:
               responses = [
                   self.model.get_responses([prompt], max_tokens=100)[0]
                   for _ in range(self.n_samples)
               ]

               # Compute pairwise similarity
               embeddings = self._get_embeddings(responses)
               similarity = self._compute_pairwise_similarity(embeddings)

               # Low similarity = high hallucination
               scores.append(1.0 - similarity)

           return sum(scores) / len(scores)

       def _get_embeddings(self, texts: list[str]) -> Tensor:
           """Get sentence embeddings using model's hidden states.

           Uses the last token's hidden state from the final layer as
           the sentence embedding (similar to mean pooling but faster).
           """
           # Use model.get_residuals() which returns hidden states
           # Shape: (n_texts, n_layers, hidden_dim)
           residuals = self.model.get_residuals_batched(texts)

           # Use last layer's hidden state as embedding
           # Shape: (n_texts, hidden_dim)
           embeddings = residuals[:, -1, :]

           # Normalize for cosine similarity
           return F.normalize(embeddings, p=2, dim=-1)

       def _compute_pairwise_similarity(self, embeddings: Tensor) -> float:
           """Compute mean pairwise cosine similarity.

           Args:
               embeddings: Normalized embeddings, shape (n_samples, hidden_dim)

           Returns:
               Mean cosine similarity across all pairs (0-1)
           """
           n = embeddings.shape[0]
           if n < 2:
               return 1.0  # Single sample = perfectly consistent

           # Compute cosine similarity matrix
           # Since embeddings are normalized, this is just dot product
           sim_matrix = torch.mm(embeddings, embeddings.T)  # (n, n)

           # Extract upper triangle (excluding diagonal)
           # These are the pairwise similarities
           mask = torch.triu(torch.ones(n, n, device=embeddings.device), diagonal=1)
           pairwise_sims = sim_matrix[mask.bool()]

           return pairwise_sims.mean().item()
   ```

2. Modify `get_score()` in `evaluator.py` to include hallucination:
   ```python
   def get_score(self) -> tuple[tuple[float, float, float], ...]:
       # Existing: KL divergence, refusal count
       # New: hallucination score

       if self.settings.use_hallucination_detection:
           hallucination_score = self.hallucination_detector.compute_hallucination_score(
               self.hallucination_prompts
           )
           return (kl_divergence, refusal_ratio, hallucination_score), ...
       else:
           return (kl_divergence, refusal_ratio), ...
   ```

3. Update Optuna study to 3-objective optimization (if hallucination enabled).

   **Important:** Modify `create_study()` in `optimization.py` to use dynamic directions:
   ```python
   def create_study(settings: Settings) -> Study:
       # Determine number of objectives based on settings
       n_objectives = 2  # Default: KL divergence + refusal count
       if settings.use_hallucination_detection:
           n_objectives = 3
       if settings.use_capability_objective:
           n_objectives = 3  # or 4 if both enabled

       directions = [StudyDirection.MINIMIZE] * n_objectives

       study = optuna.create_study(
           # ... other params ...
           directions=directions,
       )
   ```

4. Add settings:
   ```python
   use_hallucination_detection: bool = Field(
       default=False,  # Off by default (slower)
       description="Detect hallucinations during optimization."
   )

   hallucination_n_samples: int = Field(
       default=3,
       description="Number of samples for self-consistency check."
   )

   hallucination_prompts: DatasetSpecification = Field(
       default=DatasetSpecification(
           dataset="truthful_qa",
           config="generation",
           split="validation[:50]",
           column="question",
       )
   )
   ```

**Effort:** 8-12 hours
**Risk:** Low (additive feature)

---

## Priority 3: Test with Concept Cones Disabled

### Why This Matters
Concept cones add overhead but may not help for models with unified refusal mechanisms.

### Implementation Approach

**A/B Test Configuration:**

Create `configs/test_no_concept_cones.toml`:
```toml
# Test configuration: Concept cones disabled
use_concept_cones = false
use_caa = false  # Also test CAA disabled

# Keep high-impact features:
ensemble_probe_pca = true
use_activation_calibration = true
use_warm_start_params = true
use_neural_refusal_detection = true
```

**Test Protocol:**
1. Run 50 trials with concept cones enabled (current default)
2. Run 50 trials with concept cones disabled
3. Compare:
   - Best KL at 0 refusals
   - Mean KL
   - Time per trial
   - GPU memory usage

**Metrics to Track:**
```python
# Add to logging
trial_metrics = {
    "trial_number": trial.number,
    "kl_divergence": kl,
    "refusals": refusals,
    "concept_cones_used": settings.use_concept_cones,
    "n_cones_extracted": len(cones) if cones else 0,
    "silhouette_score": silhouette if cones else None,
    "trial_duration_seconds": duration,
}
```

**Effort:** 2-4 hours (mostly running experiments)
**Risk:** None (configuration change only)

---

## Priority 4: Capability Benchmarks in Objective Function

### Why This Matters
KL divergence is a weak proxy for capabilities. Directly measuring MMLU/reasoning during optimization would produce better models.

### Implementation Approach

**Current:** MMLU runs only during validation (after optimization).
**Proposed:** Run lightweight MMLU subset during each trial.

**Implementation Plan:**

1. **Reuse existing `MMLUEvaluator` from `validation.py`** with smaller sample size:
   ```python
   # In evaluator.py - reuse the existing MMLUEvaluator class
   from .validation import MMLUEvaluator

   class FastCapabilityChecker:
       """Lightweight capability check for Optuna trials.

       Reuses MMLUEvaluator with reduced sample size for speed.
       """

       def __init__(self, model: Model, n_questions_per_category: int = 3):
           """Initialize with small sample size for fast evaluation.

           Args:
               model: The model to evaluate
               n_questions_per_category: Questions per MMLU category (default: 3)
                   With 3 categories, this gives ~9-12 total questions.
           """
           # Reuse existing MMLUEvaluator with minimal samples
           self.evaluator = MMLUEvaluator(
               model=model,
               categories=["abstract_algebra", "professional_law", "high_school_physics"],
               samples_per_category=n_questions_per_category,
               n_few_shot=2,  # Fewer few-shot examples for speed
           )

       def get_accuracy(self) -> float:
           """Get accuracy on the small question set.

           Returns:
               Mean accuracy across categories (0.0-1.0)
           """
           results = self.evaluator.evaluate()
           if not results:
               return 0.0
           return sum(results.values()) / len(results)
   ```

   **Note:** This reuses the existing `_format_question()` and `_extract_answer()` methods
   from `MMLUEvaluator` in `validation.py`, avoiding code duplication.

2. Add to objective function (optional 3rd objective):
   ```python
   # In get_score():
   if self.settings.use_capability_objective:
       capability_score = self.capability_checker.get_accuracy()
       # Invert: we want to MAXIMIZE accuracy, but Optuna MINIMIZES
       capability_loss = 1.0 - capability_score
       return (kl_divergence, refusal_ratio, capability_loss), ...
   ```

3. Add settings:
   ```python
   use_capability_objective: bool = Field(
       default=False,
       description="Include capability score in Optuna objective (3-objective)."
   )

   capability_check_n_questions: int = Field(
       default=10,
       description="Number of MMLU questions for fast capability check."
   )
   ```

**Trade-off:** ~30 seconds extra per trial for 10 questions.

**Effort:** 6-10 hours
**Risk:** Low (additive feature, optional)

---

## Priority 5: Norm-Preserving Abliteration

### Why This Matters
Standard abliteration can change weight matrix norms significantly, leading to:
- Activation scale drift
- Capability degradation
- Numerical instability

Norm-preserving abliteration maintains the original Frobenius norm after modification.

### Implementation Approach

**Mathematical Background:**

Standard abliteration:
```
W_new = W - α * (r ⊗ r) @ W
```

Norm-preserving abliteration:
```
W_new = W - α * (r ⊗ r) @ W
W_new = W_new * (||W||_F / ||W_new||_F)  # Rescale to preserve norm
```

**Implementation Plan:**

1. **Modify existing `abliterate()` method** in `model.py` to add norm preservation:
   ```python
   # Add this to the existing abliterate() method, inside the matrix loop:

   for matrix in matrices:
       # Store original norm if norm preservation is enabled
       if self.settings.preserve_weight_norms:
           original_norm = torch.linalg.norm(matrix, ord='fro')

       # Reuse cached projector per device to prevent memory accumulation
       device_projector = get_device_projector(projector, matrix.device)
       # In-place subtraction is safe as we're not using Autograd.
       matrix.sub_(weight * (device_projector @ matrix))

       # Rescale to preserve original Frobenius norm
       if self.settings.preserve_weight_norms:
           new_norm = torch.linalg.norm(matrix, ord='fro')
           if new_norm > EPSILON:
               matrix.mul_(original_norm / new_norm)
   ```

   **Note:** This modifies the existing `abliterate()` method rather than creating a
   new method. The `get_device_projector` helper is already defined as a local function
   inside `abliterate()`, so we can use it directly.

2. Add setting:
   ```python
   preserve_weight_norms: bool = Field(
       default=True,  # Enable by default
       description="Rescale weight matrices after abliteration to preserve Frobenius norm. Reduces capability damage."
   )
   ```

3. Alternative: Biprojected abliteration (more sophisticated):
   ```python
   def abliterate_biprojected(self, ...):
       """Biprojected abliteration that removes refusal while preserving orthogonal components."""
       # Instead of: W_new = W - α * P_r @ W
       # Use: W_new = W - α * P_r @ W @ P_r
       # This only modifies the component that projects onto r in both input AND output

       for matrix in matrices:
           # Biprojected: only modify the r→r component
           # This is more surgical than standard abliteration
           # Note: Since projector is symmetric (P = P^T), the .T is optional
           # but included for clarity about the mathematical operation
           projection = device_projector @ matrix @ device_projector
           matrix.sub_(weight * projection)
   ```

**Effort:** 4-8 hours
**Risk:** Low (modification to existing function)

---

---

## Known Implementation Notes

**Items requiring attention during implementation:**

1. **Priority 2 & 4 - Dynamic Optuna objectives:** The current `optimization.py` hardcodes
   2-objective optimization. When implementing hallucination detection or capability
   benchmarks as objectives, update `create_study()` to dynamically set the number of
   directions based on enabled features.

2. **Priority 4 - Code reuse:** Use the existing `MMLUEvaluator` class from `validation.py`
   instead of creating new MMLU evaluation code. This avoids duplicating the question
   formatting and answer extraction logic.

3. **Priority 5 - Modify existing method:** Add norm preservation to the existing
   `abliterate()` method rather than creating a separate method. This keeps the
   codebase simpler and ensures all abliteration paths benefit from the improvement.

---

## Implementation Schedule

### Phase 1: Quick Wins (1-2 days)
- [ ] Priority 3: Test concept cones disabled (config change + run experiments)
- [ ] Priority 5: Norm-preserving abliteration (simple modification)

### Phase 2: Medium Features (3-5 days)
- [ ] Priority 2: Hallucination detection (new evaluator component)
- [ ] Priority 4: Capability benchmarks in objective (extend evaluator)

### Phase 3: Major Feature (1-2 weeks)
- [ ] Priority 1: Post-abliteration fine-tuning (new module, training infrastructure)

---

## Validation Protocol

After implementing each feature:

1. **Unit Tests:**
   - Add tests for new functions in `tests/unit/`
   - Run: `uv run pytest tests/unit/ -v`

2. **Integration Test:**
   - Run 20 trials on small model (Qwen2.5-3B)
   - Compare results with/without feature

3. **Production Test:**
   - Run 50+ trials on target model
   - Compare with Trial 173 baseline

---

## Dependencies to Add

```toml
# pyproject.toml additions
[project.optional-dependencies]
finetuning = [
    "trl>=0.9.0",
    "peft>=0.11.0",
]

hallucination = [
    "sentence-transformers>=2.0.0",  # For embeddings
]
```

---

## Success Metrics

| Metric | Current (Trial 173) | Target |
|--------|---------------------|--------|
| Best KL (0 refusals) | 0.26 | < 0.20 |
| MMLU accuracy | TBD | ≥ baseline - 2% |
| Hallucination rate | TBD | < 10% |
| Training time | ~11h | < 15h (with fine-tuning) |

---

## References

- [TRL DPO Documentation](https://huggingface.co/docs/trl/main/en/dpo_trainer)
- [SelfCheckGPT Paper](https://arxiv.org/abs/2303.08896)
- [r/LocalLLaMA Abliteration Discussion](https://reddit.com/r/LocalLLaMA)
- [Representation Engineering Paper](https://arxiv.org/abs/2310.01405)
