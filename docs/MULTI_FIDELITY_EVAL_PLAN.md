# Multi-Fidelity Evaluation Plan

## Problem

Each Optuna trial evaluates all 104 bad prompts (~3 min/trial). Early trials explore randomly and often find poor configurations. We waste compute evaluating obviously bad trials with full precision.

## Solution: Multi-Fidelity Evaluation

Use fewer prompts for early trials (fast screening), more prompts for later trials (accurate selection). This is NOT pruning - all trials complete, but early ones use less compute.

**Key insight:** Multi-fidelity works with multi-objective optimization (unlike HyperbandPruner/BOHB which require single-objective).

---

## Design

### Fidelity Schedule

| Trial Range | Prompt Count | Purpose |
|-------------|--------------|---------|
| 1-60 (30%) | 30 prompts | Fast exploration - quickly reject bad configs |
| 61-140 (40%) | 60 prompts | Medium fidelity - better signal for TPE |
| 141-200 (30%) | 104 prompts | Full evaluation - accurate final selection |

**Rationale:**
- First 30% of trials: TPE is still random (n_startup_trials=30), so accurate evaluation is wasted
- Middle 40%: TPE is learning, needs better signal but not full precision
- Last 30%: TPE has converged, need accurate evaluation for Pareto front

### Prompt Selection Strategy

**Option A: Random subset (simpler)**
```python
def get_fidelity_prompts(self, trial_index: int, n_trials: int) -> list[str]:
    """Get subset of bad prompts based on trial progress."""
    progress = trial_index / n_trials

    if progress < 0.3:
        n_prompts = 30
    elif progress < 0.7:
        n_prompts = 60
    else:
        n_prompts = len(self.bad_prompts)

    # Use deterministic seed for reproducibility
    rng = random.Random(42)
    return rng.sample(self.bad_prompts, min(n_prompts, len(self.bad_prompts)))
```

**Option B: Stratified subset (better)**
```python
def get_fidelity_prompts(self, trial_index: int, n_trials: int) -> list[str]:
    """Get stratified subset ensuring diverse prompt coverage."""
    progress = trial_index / n_trials

    if progress < 0.3:
        n_prompts = 30
    elif progress < 0.7:
        n_prompts = 60
    else:
        return self.bad_prompts  # Full set

    # Always include the prompts that triggered refusals in baseline
    # This ensures we're testing the "hard" cases
    # Remaining slots filled with random selection
    return self._get_stratified_sample(n_prompts)
```

**Recommendation:** Start with Option A (simpler), can upgrade to Option B later if needed.

---

## Implementation

### Phase 1: Config Settings

Add to `src/bruno/config.py`:

```python
# Multi-Fidelity Evaluation Settings
use_multi_fidelity: bool = Field(
    default=True,
    description="Use multi-fidelity evaluation: fewer prompts for early trials, full evaluation for late trials. Provides ~40-50% faster early exploration without pruning.",
)

fidelity_schedule: list[tuple[float, int]] = Field(
    default=[(0.3, 30), (0.7, 60), (1.0, -1)],  # -1 = full
    description="Fidelity schedule as list of (progress_threshold, n_prompts) tuples. -1 means use all prompts.",
)

fidelity_min_prompts: int = Field(
    default=20,
    description="Minimum number of prompts for any fidelity level. Must have enough for signal.",
)
```

### Phase 2: Evaluator Changes

Modify `src/bruno/evaluator.py`:

```python
class Evaluator:
    def __init__(self, settings: Settings, model: Model):
        # ... existing init ...

        # Multi-fidelity: pre-compute stratified samples
        self._fidelity_samples: dict[int, list[str]] = {}
        if settings.use_multi_fidelity:
            self._precompute_fidelity_samples(settings.fidelity_schedule)

    def _precompute_fidelity_samples(self, schedule: list[tuple[float, int]]) -> None:
        """Pre-compute prompt samples for each fidelity level."""
        rng = random.Random(42)  # Deterministic for reproducibility

        for _, n_prompts in schedule:
            if n_prompts == -1 or n_prompts >= len(self.bad_prompts):
                self._fidelity_samples[n_prompts] = self.bad_prompts
            else:
                self._fidelity_samples[n_prompts] = rng.sample(self.bad_prompts, n_prompts)

    def get_fidelity_prompts(self, trial_index: int, n_trials: int) -> list[str]:
        """Get prompts for current fidelity level based on trial progress."""
        if not self.settings.use_multi_fidelity:
            return self.bad_prompts

        progress = trial_index / n_trials

        # Find appropriate fidelity level
        for threshold, n_prompts in self.settings.fidelity_schedule:
            if progress < threshold:
                if n_prompts == -1:
                    return self.bad_prompts
                return self._fidelity_samples.get(n_prompts, self.bad_prompts)

        return self.bad_prompts  # Default to full

    def count_refusals_fidelity(
        self,
        trial_index: int,
        n_trials: int,
        use_neural: bool | None = None
    ) -> tuple[int, int]:
        """Count refusals with fidelity-aware prompt selection.

        Returns:
            Tuple of (refusals, total_prompts) to allow proper ratio calculation
        """
        prompts = self.get_fidelity_prompts(trial_index, n_trials)

        responses = self.model.get_responses_batched(
            prompts,
            max_tokens=self.settings.refusal_check_tokens,
        )

        # ... existing refusal counting logic ...

        return refusal_count, len(prompts)

    def get_score_fidelity(
        self,
        trial_index: int,
        n_trials: int
    ) -> tuple[tuple[float, float], float, int, int]:
        """Get score with fidelity-aware evaluation.

        Returns:
            Tuple of (score, kl_divergence, refusals, total_prompts)
        """
        # KL divergence always uses full good_prompts (capability metric)
        # Only refusal counting uses fidelity

        with ThreadPoolExecutor(max_workers=2) as executor:
            kl_future = executor.submit(self._compute_kl_divergence)
            refusal_future = executor.submit(
                self.count_refusals_fidelity, trial_index, n_trials
            )

            kl_divergence = kl_future.result()
            refusals, total_prompts = refusal_future.result()

        # Normalize refusals to ratio for fair comparison across fidelities
        refusal_ratio = refusals / total_prompts if total_prompts > 0 else 0.0
        base_ratio = self.base_refusals / len(self.bad_prompts) if self.base_refusals > 0 else 1.0

        score = (
            (kl_divergence / self.settings.kl_divergence_scale),
            refusal_ratio / base_ratio if base_ratio > 0 else refusal_ratio,
        )

        return score, kl_divergence, refusals, total_prompts
```

### Phase 3: Main Loop Changes

Modify `src/bruno/main.py` objective function:

```python
def objective(trial: Trial) -> tuple[float, float]:
    nonlocal trial_index
    trial_index += 1

    # ... existing parameter sampling ...

    print("* Evaluating...")

    # Use fidelity-aware evaluation
    if settings.use_multi_fidelity:
        score, kl_divergence, refusals, total_prompts = evaluator.get_score_fidelity(
            trial_index, settings.n_trials
        )
        print(f"  * Fidelity: {total_prompts}/{len(evaluator.bad_prompts)} prompts")
    else:
        score, kl_divergence, refusals = evaluator.get_score()
        total_prompts = len(evaluator.bad_prompts)

    print(f"  * KL divergence: [bold]{kl_divergence:.2f}[/]")
    print(f"  * Refusals: [bold]{refusals}[/]/{total_prompts}")

    # Store for user display (convert ratio back to count for final display)
    trial.set_user_attr("kl_divergence", kl_divergence)
    trial.set_user_attr("refusals", refusals)
    trial.set_user_attr("total_prompts", total_prompts)
    trial.set_user_attr("fidelity", total_prompts / len(evaluator.bad_prompts))

    return score
```

### Phase 4: Validation Re-evaluation

After optimization, re-evaluate Pareto-optimal trials with full fidelity:

```python
# After optimization loop, before presenting choices
if settings.use_multi_fidelity:
    print()
    print("Re-evaluating Pareto-optimal trials with full fidelity...")

    for trial in best_trials:
        # Skip if already evaluated at full fidelity
        if trial.user_attrs.get("fidelity", 0) >= 1.0:
            continue

        # Restore and re-evaluate
        parameters = {
            k: AbliterationParameters(**v)
            for k, v in trial.user_attrs["parameters"].items()
        }
        restore_and_abliterate_trial(parameters, trial.user_attrs["direction_index"])

        # Full evaluation
        _, kl_divergence, refusals = evaluator.get_score()

        # Update trial attributes
        trial.set_user_attr("kl_divergence_full", kl_divergence)
        trial.set_user_attr("refusals_full", refusals)
        trial.set_user_attr("fidelity", 1.0)

    # Re-sort by full-fidelity metrics
    best_trials = sorted(
        best_trials,
        key=lambda t: (
            t.user_attrs.get("refusals_full", t.user_attrs["refusals"]),
            t.user_attrs.get("kl_divergence_full", t.user_attrs["kl_divergence"]),
        ),
    )
```

---

## Expected Benefits

### Speed Improvement

| Phase | Trials | Prompts | Time/Trial | Total |
|-------|--------|---------|------------|-------|
| Exploration (30%) | 60 | 30 | ~1 min | 60 min |
| Learning (40%) | 80 | 60 | ~2 min | 160 min |
| Selection (30%) | 60 | 104 | ~3 min | 180 min |
| **Total** | 200 | - | - | **400 min** |

vs. Current: 200 trials × 3 min = **600 min**

**Speedup: ~33% faster** (10 hours → 6.7 hours)

### Quality Impact

- Early trials: Slightly noisier signal, but TPE is random anyway
- Middle trials: Moderate noise, but still good enough for TPE learning
- Late trials: Full precision for accurate Pareto front
- Re-evaluation: Pareto trials get full evaluation before user selection

**Risk:** Some trials may look good at low fidelity but fail at full fidelity. Mitigated by re-evaluating Pareto-optimal trials.

---

## Configuration Options

```toml
# config.toml

# Enable multi-fidelity evaluation
use_multi_fidelity = true

# Custom schedule (optional - defaults are good)
# Format: [(progress_threshold, n_prompts), ...]
# fidelity_schedule = [[0.3, 30], [0.7, 60], [1.0, -1]]

# Minimum prompts (safety floor)
fidelity_min_prompts = 20
```

---

## Testing Plan

1. **Unit tests for fidelity calculation:**
   - `test_get_fidelity_prompts_early_trial` - returns 30 prompts
   - `test_get_fidelity_prompts_middle_trial` - returns 60 prompts
   - `test_get_fidelity_prompts_late_trial` - returns full prompts
   - `test_fidelity_samples_reproducible` - same seed → same samples

2. **Integration test:**
   - Run 10-trial optimization with multi-fidelity
   - Verify trial metadata includes fidelity info
   - Verify Pareto trials get re-evaluated

3. **Comparison test:**
   - Run same model with and without multi-fidelity
   - Compare final Pareto front quality
   - Measure wall-clock time difference

---

## Rollout Plan

1. **Phase 1:** Implement config settings (low risk)
2. **Phase 2:** Implement evaluator changes with feature flag (medium risk)
3. **Phase 3:** Implement main loop changes (medium risk)
4. **Phase 4:** Add re-evaluation logic (low risk)
5. **Phase 5:** Test on small model (7B) with 50 trials
6. **Phase 6:** Enable by default after validation

---

## Open Questions

1. **Should KL divergence also use fidelity?**
   - Current plan: No, KL uses full good_prompts always
   - Rationale: KL is a capability metric, needs consistent measurement
   - Alternative: Could use fidelity for both, but complicates comparison

2. **Should we re-evaluate ALL Pareto trials or just top-N?**
   - Current plan: All Pareto trials (usually 5-20)
   - Alternative: Only top 5 by refusal count
   - Tradeoff: More re-evaluation = more accurate, but slower

3. **Should fidelity be configurable per-trial by Optuna?**
   - Current plan: No, fixed schedule based on progress
   - Alternative: Let Optuna suggest fidelity (like Hyperband)
   - Tradeoff: More flexible, but adds complexity

---

## Files to Modify

1. `src/bruno/config.py` - Add multi-fidelity settings
2. `src/bruno/evaluator.py` - Add fidelity-aware methods
3. `src/bruno/main.py` - Update objective and post-optimization
4. `tests/unit/test_evaluator.py` - Add fidelity tests

**Estimated effort:** 4-6 hours implementation + 2 hours testing
