# MoE Gate Abliteration Implementation Plan

**Date:** February 2026
**Goal:** Improve MoE abliteration from 25% to 75%+ success rate
**Timeline:** 6-10 days

---

## Overview

Current MoE abliteration only targets expert weights, but the gate detects harmful content BEFORE routing. This plan implements gate-level abliteration to address the root cause.

---

## Theoretical Validation: Why Output-Space Directions Work for Gate Ablation

A key question was raised during code review: "The plan uses output-space refusal directions (from residuals) to abliterate the gate's input projection. Is this valid?"

**Answer: Yes, this approach is mathematically valid.**

### The Same Vector Space

Both the refusal direction and the gate's input are in the **same `hidden_dim` vector space**:

1. **Refusal directions**: Computed from hidden states (residuals) at each layer, shape `[n_layers, hidden_dim]`
2. **Gate input**: The hidden state (post-attention) at that layer, shape `[hidden_dim]`

These are the same vector space - the residual stream that carries information between transformer layers.

### Data Flow at Each MoE Layer

```
Input → Attention → [post_attn_hidden_state] → Gate & Experts → Output
                            ↑
                    Gate sees THIS (hidden_dim)
                    Refusal direction computed from hidden states (hidden_dim)
```

The gate performs a linear projection:
```python
affinity_scores = gate_weight @ hidden_state  # [n_experts, hidden_dim] @ [hidden_dim]
```

By orthogonalizing the gate weights against the refusal direction, we prevent the gate from "detecting" that direction in its input. This is the same mathematical operation used for expert ablation.

### Why It Works

The gate's job is to detect patterns in the hidden state to make routing decisions. If harmful content is encoded as a direction in `hidden_dim` space, the gate has learned to recognize that direction. By removing that direction from the gate's projection matrix, we blind the gate to refusal-related features.

### Supporting Research

- MoE gating networks receive the **same hidden state representation** as the experts - it's the token's hidden state at that layer
- Research on activation steering shows that directions computed at one layer can be applied across layers effectively
- "GateBreaker" research demonstrates that modifying gating activations can change model behavior, validating the approach of targeting gates

### Minor Subtlety (Not a Blocker)

The refusal direction is typically computed from post-MoE hidden states, but the gate sees pre-MoE (post-attention) hidden states. However:
- We're computing a **direction** (normalized), not absolute magnitudes
- The residual stream maintains coherent representations across the layer
- The refusal direction captures the relevant feature axis regardless of exact position in the forward pass

### Potential Limitations

Even though the math is valid, gate ablation might have limited impact if:
1. The gate has learned **multiple** refusal-detection directions (we only ablate one)
2. The refusal routing behavior is redundant across many experts
3. The gate's bias terms (`e_score_correction`) encode refusal behavior separately from the linear projection

The two-stage approach (gate first, then re-track and ablate experts) addresses these limitations by accounting for routing changes after gate modification.

### Optional Enhancement (Future Work)

For extra rigor in future iterations, we could:
- Hook into post-attention (pre-MoE) activations specifically to capture exactly what the gate sees
- Extract gate-specific directions by comparing gate outputs (expert affinity scores) on harmful vs safe prompts

These enhancements are likely unnecessary for practical purposes but could provide marginal improvements.

---

## Implementation Phases

### Phase 1: Infrastructure (1-2 days)

Add routing health metrics to enable monitoring and abort conditions.

#### 1.1 Add Routing Health Metrics

**File:** `src/bruno/model.py`

```python
def compute_routing_entropy(self, prompts: list[str]) -> dict[int, float]:
    """Compute entropy of expert routing distribution per layer.

    Higher entropy = more uniform routing (healthy)
    Lower entropy = routing collapse (unhealthy)

    Returns:
        Dict mapping layer_idx -> entropy (0.0 to log(n_experts))
    """
    activations = self.track_expert_activations(prompts)
    if activations is None:
        return {}

    entropies = {}
    for layer_idx, counts in activations.layer_expert_counts.items():
        total = sum(counts.values())
        if total == 0:
            entropies[layer_idx] = 0.0
            continue

        probs = [count / total for count in counts.values() if count > 0]
        entropy = -sum(p * math.log(p) for p in probs)
        entropies[layer_idx] = entropy

    return entropies


def check_routing_health(self, prompts: list[str], min_entropy: float = 0.5) -> bool:
    """Check if routing is healthy (not collapsed).

    Returns True if mean entropy > min_entropy.
    """
    entropies = self.compute_routing_entropy(prompts)
    if not entropies:
        return True  # Non-MoE model or no entropy data

    # Guard against empty dict after filtering
    if len(entropies) == 0:
        logger.warning("No entropy values computed, assuming healthy routing")
        return True

    mean_entropy = sum(entropies.values()) / len(entropies)

    moe_config = self.get_moe_config()
    if moe_config is None or moe_config[0] <= 1:
        logger.warning("Invalid MoE config for entropy normalization")
        return True

    max_possible = math.log(moe_config[0])
    if max_possible <= 0:
        logger.warning("Max possible entropy is non-positive, assuming healthy routing")
        return True

    normalized_entropy = mean_entropy / max_possible

    logger.debug(
        "Routing health check",
        mean_entropy=mean_entropy,
        max_possible=max_possible,
        normalized_entropy=normalized_entropy,
        threshold=min_entropy,
    )

    return normalized_entropy > min_entropy
```

#### 1.2 Add Gate Weight Caching

The existing `layer_weights_cache` only caches expert and attention weights. Gate weights must also be cached to enable reset between trials.

**File:** `src/bruno/model.py` - Modify `_create_layer_weights_cache()`

```python
def _create_layer_weights_cache(self) -> dict[int, dict[str, list[Tensor]]]:
    # ... existing code ...

    # Add gate weight caching for MoE models
    if hasattr(layer.mlp, 'gate') and hasattr(layer.mlp.gate, 'weight'):
        cache[layer_idx]['gate.weight'] = [layer.mlp.gate.weight.clone().detach()]

        # Also cache bias if present
        if hasattr(layer.mlp.gate, 'e_score_correction'):
            cache[layer_idx]['gate.bias'] = [layer.mlp.gate.e_score_correction.clone().detach()]
```

---

### Phase 2: Gate Abliteration (2-3 days)

Implement the core gate abliteration functionality.

#### 2.1 Add Gate Abliteration Method

**File:** `src/bruno/model.py`

```python
def abliterate_gate(
    self,
    layer_idx: int,
    refusal_direction: Tensor,
    strength: float = 1.0,
) -> bool:
    """Abliterate the gate's ability to detect refusal-related content.

    The gate projects inputs to expert affinity scores. By removing the
    refusal direction from the gate weights, we prevent the gate from
    "seeing" refusal-triggering patterns in the input.

    Args:
        layer_idx: Which layer's gate to modify
        refusal_direction: Direction to remove (hidden_dim,)
        strength: Ablation strength (0.0 to 1.0)

    Returns:
        True if gate was abliterated, False if layer has no gate
    """
    layer = self.get_layers()[layer_idx]

    if not hasattr(layer.mlp, 'gate'):
        logger.debug(f"Layer {layer_idx} has no gate attribute, skipping")
        return False

    gate = layer.mlp.gate
    if not hasattr(gate, 'weight'):
        logger.debug(f"Layer {layer_idx} gate has no weight attribute, skipping")
        return False

    # Gate weight: [n_experts, hidden_dim]
    gate_weight = gate.weight.data

    # Normalize direction
    direction = refusal_direction.to(gate_weight.device)
    direction = direction / (direction.norm() + 1e-8)

    # Create projector
    projector = torch.outer(direction, direction).to(gate_weight.dtype)

    # Remove refusal direction from gate's input detection
    # W' = W - strength * (W @ projector)
    gate.weight.data = gate_weight - strength * (gate_weight @ projector)

    logger.debug(f"Abliterated gate at layer {layer_idx} with strength {strength}")
    return True


def abliterate_gates(
    self,
    refusal_directions: Tensor,
    strength: float = 1.0,
    layer_profiles: list["LayerRangeProfile"] | None = None,
) -> int:
    """Abliterate gates across all MoE layers.

    Args:
        refusal_directions: Per-layer directions (n_layers+1, hidden_dim)
        strength: Base ablation strength
        layer_profiles: Optional per-layer weight profiles

    Returns:
        Number of gates abliterated
    """
    num_layers = len(self.get_layers())
    gates_ablated = 0

    for layer_idx in range(num_layers):
        # Get layer-specific direction (+1 for embedding offset)
        direction = refusal_directions[layer_idx + 1]

        # Apply layer profile multiplier
        layer_multiplier = self.get_layer_multiplier(
            layer_idx, num_layers, layer_profiles
        )
        effective_strength = strength * layer_multiplier

        if self.abliterate_gate(layer_idx, direction, effective_strength):
            gates_ablated += 1

    return gates_ablated
```

#### 2.2 Add Config Settings

**File:** `src/bruno/config.py`

```python
# MoE Gate Abliteration Settings
use_gate_abliteration: bool = Field(
    default=True,
    description="Abliterate MoE gate weights to prevent refusal detection at routing level. Only applies to MoE models.",
)

gate_abliteration_strength: float = Field(
    default=0.3,
    description="Strength of gate abliteration (0.0 to 1.0). Lower values are safer but may be less effective. This is a starting point - Optuna will optimize.",
)

gate_abliteration_optuna: bool = Field(
    default=True,
    description="Let Optuna optimize gate_abliteration_strength during trials.",
)

moe_top_k_experts: int = Field(
    default=8,
    description="Maximum number of experts to target per layer during MoE abliteration. Set to 0 for no limit.",
)

@field_validator("gate_abliteration_strength")
@classmethod
def validate_gate_abliteration_strength(cls, v: float) -> float:
    """Ensure gate_abliteration_strength is in valid range."""
    if not 0.0 <= v <= 1.0:
        raise ValueError("gate_abliteration_strength must be between 0.0 and 1.0")
    return v
```

#### 2.3 Add Optuna Parameter

**File:** `src/bruno/main.py` (in `objective` function)

```python
# Add to trial parameter suggestions (only for MoE models)
if model.is_moe_model() and settings.gate_abliteration_optuna:
    gate_strength = trial.suggest_float("gate_abliteration_strength", 0.0, 0.5)
else:
    gate_strength = settings.gate_abliteration_strength
```

#### 2.4 Integrate into Abliteration Flow

**File:** `src/bruno/main.py` (in abliteration section)

Note: This single-pass gate ablation is used when `use_two_stage_moe_abliteration=False`.
When `use_two_stage_moe_abliteration=True`, use the two-stage method from Phase 4 instead.

```python
# MoE abliteration path selection
if model.is_moe_model():
    if settings.use_two_stage_moe_abliteration:
        # Two-stage approach: gate first, re-track, then experts (Phase 4)
        # This is the recommended path for MoE models
        logger.info("Using two-stage MoE abliteration")
        moe_stats = model.abliterate_moe_two_stage(
            refusal_directions,
            bad_prompts,
            parameters,
            gate_strength=gate_strength,
            expert_threshold=settings.moe_expert_activation_threshold,
            layer_profiles=layer_profiles,
            use_mpoa=settings.use_mpoa,
            mpoa_norm_mode=settings.mpoa_norm_mode,
        )
        logger.info(
            "Two-stage MoE abliteration complete",
            gates_ablated=moe_stats.get('gates_ablated', 0),
            experts_ablated=moe_stats.get('experts_ablated', 0),
        )
    else:
        # Single-pass approach: gate ablation only (legacy/simpler path)
        if settings.use_gate_abliteration:
            gates_ablated = model.abliterate_gates(
                refusal_directions,
                strength=gate_strength,
                layer_profiles=layer_profiles,
            )
            logger.info(f"Abliterated {gates_ablated} MoE gates (single-pass mode)")

            # Verify routing health - abort trial if routing collapsed
            if not model.check_routing_health(bad_prompts[:20]):
                logger.warning("Routing collapsed after gate ablation, aborting trial")
                raise optuna.TrialPruned("Routing collapsed after gate ablation")

        # Then do standard MoE targeted abliteration
        targeted_experts = model.setup_moe_targeting(bad_prompts)
        if targeted_experts:
            model.abliterate_moe_targeted(
                refusal_directions,
                parameters,
                targeted_experts,
                layer_profiles=layer_profiles,
                use_mpoa=settings.use_mpoa,
                mpoa_norm_mode=settings.mpoa_norm_mode,
            )
else:
    # Dense model abliteration (unchanged)
    model.abliterate(
        refusal_directions,
        direction_index,
        parameters,
        layer_profiles=layer_profiles,
        use_mpoa=settings.use_mpoa,
        mpoa_norm_mode=settings.mpoa_norm_mode,
    )
```

---

### Phase 3: Bias Manipulation (1-2 days)

Modify expert selection biases to prefer compliant experts.

#### 3.1 Add Bias Modification Method

**File:** `src/bruno/model.py`

```python
def modify_expert_biases(
    self,
    expert_scores: dict[int, dict[int, float]],
    bias_delta: float = 0.3,
) -> int:
    """Modify gate bias terms to prefer compliant experts.

    Args:
        expert_scores: layer_idx -> expert_idx -> compliance score (-1 to 1)
            Negative = refusing, Positive = compliant
        bias_delta: Maximum bias adjustment

    Returns:
        Number of biases modified
    """
    biases_modified = 0
    layers_without_bias = 0

    for layer_idx, scores in expert_scores.items():
        layer = self.get_layers()[layer_idx]

        if not hasattr(layer.mlp, 'gate'):
            continue

        gate = layer.mlp.gate
        if not hasattr(gate, 'e_score_correction'):
            layers_without_bias += 1
            continue

        bias = gate.e_score_correction.data

        for exp_idx, score in scores.items():
            if exp_idx < len(bias):
                # score > 0 means compliant, increase bias
                # score < 0 means refusing, decrease bias
                adjustment = score * bias_delta
                bias[exp_idx] += adjustment
                biases_modified += 1

    if layers_without_bias > 0:
        logger.warning(
            f"Bias manipulation skipped for {layers_without_bias} layers "
            f"(no e_score_correction attribute)"
        )

    return biases_modified


def compute_expert_compliance_scores(
    self,
    bad_prompts: list[str],
    evaluator: "Evaluator",
) -> dict[int, dict[int, float]]:
    """Compute compliance scores for each expert based on per-prompt activation correlation.

    For each expert, correlate its activation frequency with refusal outcomes
    across individual prompts. Experts that activate more on prompts that
    result in refusals get negative scores.

    Returns:
        layer_idx -> expert_idx -> score (-1 to 1)
        Returns empty dict if tracking fails or no data available.
    """
    logger.info("Computing expert compliance scores")

    # Track per-prompt expert activations
    per_prompt_activations = self._track_expert_activations_per_prompt(bad_prompts)
    if per_prompt_activations is None:
        logger.warning("Could not track per-prompt activations, returning empty scores")
        return {}

    if not per_prompt_activations:
        logger.warning("No activation data captured, returning empty scores")
        return {}

    # Generate responses and check for refusals
    logger.debug(f"Generating responses for {len(bad_prompts)} prompts")
    responses = self.get_responses_batched(bad_prompts)
    refusal_mask = [evaluator.is_refusal(r) for r in responses]

    n_refusals = sum(refusal_mask)
    logger.debug(f"Refusal detection: {n_refusals}/{len(bad_prompts)} prompts refused")

    scores: dict[int, dict[int, float]] = {}
    for layer_idx in per_prompt_activations.keys():
        layer_scores: dict[int, float] = {}

        # per_prompt_activations[layer_idx] is dict: prompt_idx -> set of expert_idx
        prompt_experts = per_prompt_activations[layer_idx]

        # For each expert, compute correlation with refusal outcomes
        all_experts: set[int] = set()
        for experts in prompt_experts.values():
            all_experts.update(experts)

        for exp_idx in all_experts:
            # Count: how often does this expert activate on refusing vs compliant prompts?
            refusal_activations = 0
            comply_activations = 0

            for prompt_idx, experts in prompt_experts.items():
                if exp_idx in experts:
                    if refusal_mask[prompt_idx]:
                        refusal_activations += 1
                    else:
                        comply_activations += 1

            total = refusal_activations + comply_activations
            if total > 0:
                # Score: -1 if only activates on refusals, +1 if only on compliance
                compliance_score = (comply_activations - refusal_activations) / total
            else:
                compliance_score = 0.0

            layer_scores[exp_idx] = compliance_score

        scores[layer_idx] = layer_scores

    # Log summary statistics
    total_experts = sum(len(s) for s in scores.values())
    avg_score = sum(s for layer in scores.values() for s in layer.values()) / max(total_experts, 1)
    logger.info(
        f"Computed compliance scores for {total_experts} experts "
        f"across {len(scores)} layers (avg score: {avg_score:.3f})"
    )

    return scores


def _track_expert_activations_per_prompt(
    self,
    prompts: list[str],
) -> dict[int, dict[int, set[int]]] | None:
    """Track which experts activate for each individual prompt.

    Returns:
        layer_idx -> prompt_idx -> set of expert indices that activated
        Returns None if model is not MoE or tracking fails.
    """
    if not self.is_moe_model():
        logger.debug("Not an MoE model, skipping per-prompt activation tracking")
        return None

    # Implementation similar to track_expert_activations but preserves
    # per-prompt granularity instead of aggregating counts
    #
    # NOTE: The hook implementation below handles several common MoE output formats,
    # but may need architecture-specific customization when implementing for real models.
    # The exact output format of DeepSeek/Moonlight gates should be verified against
    # the actual model before production use.

    per_prompt_activations: dict[int, dict[int, set[int]]] = {}

    # Register hooks to capture expert selections per prompt
    hooks = []
    captured_experts: dict[int, set[int]] = {}  # layer_idx -> expert indices
    missing_router_indices_warned = False  # Track if we've warned about unrecognized output format

    def make_hook(layer_idx: int):
        def hook(module, input, output):
            nonlocal missing_router_indices_warned
            # Capture which experts were selected
            # Implementation depends on specific MoE architecture
            # For DeepSeek/Moonlight: check router output indices
            if hasattr(output, 'router_indices'):
                indices = output.router_indices.flatten().tolist()
                captured_experts.setdefault(layer_idx, set()).update(indices)
            elif hasattr(output, 'topk_indices'):
                # Alternative attribute name used by some MoE implementations
                indices = output.topk_indices.flatten().tolist()
                captured_experts.setdefault(layer_idx, set()).update(indices)
            elif isinstance(output, tuple) and len(output) >= 2:
                # Some gates return (scores, indices) tuple
                potential_indices = output[1]
                if hasattr(potential_indices, 'flatten'):
                    indices = potential_indices.flatten().tolist()
                    captured_experts.setdefault(layer_idx, set()).update(indices)
            else:
                # Gate output format not recognized - may need architecture-specific handling
                if not missing_router_indices_warned:
                    logger.warning(
                        f"Gate output at layer {layer_idx} has no recognized routing indices attribute. "
                        f"Output type: {type(output)}. Per-prompt tracking may be incomplete. "
                        f"Consider implementing architecture-specific index extraction."
                    )
                    missing_router_indices_warned = True
        return hook

    try:
        # Register hooks on each MoE layer's gate
        for layer_idx, layer in enumerate(self.get_layers()):
            if hasattr(layer.mlp, 'gate'):
                hook = layer.mlp.gate.register_forward_hook(make_hook(layer_idx))
                hooks.append(hook)

        # Process each prompt individually
        for prompt_idx, prompt in enumerate(prompts):
            captured_experts.clear()

            # Forward pass to capture expert activations
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                self.model(**inputs)

            # Store per-prompt results
            for layer_idx, experts in captured_experts.items():
                if layer_idx not in per_prompt_activations:
                    per_prompt_activations[layer_idx] = {}
                per_prompt_activations[layer_idx][prompt_idx] = experts.copy()

        logger.debug(
            f"Tracked per-prompt activations for {len(prompts)} prompts "
            f"across {len(per_prompt_activations)} layers"
        )
        return per_prompt_activations

    except Exception as e:
        logger.warning(f"Failed to track per-prompt expert activations: {e}")
        return None
    finally:
        # Clean up hooks
        for hook in hooks:
            hook.remove()
```

#### 3.2 Add Config Settings

**File:** `src/bruno/config.py`

```python
use_bias_manipulation: bool = Field(
    default=True,
    description="Modify MoE gate bias terms to prefer compliant experts. Only applies to MoE models with e_score_correction bias.",
)

bias_manipulation_delta: float = Field(
    default=0.3,
    description="Maximum bias adjustment for expert preference. Higher values steer routing more aggressively.",
)

@field_validator("bias_manipulation_delta")
@classmethod
def validate_bias_manipulation_delta(cls, v: float) -> float:
    """Ensure bias_manipulation_delta is in valid range."""
    if not 0.0 <= v <= 1.0:
        raise ValueError("bias_manipulation_delta must be between 0.0 and 1.0")
    return v
```

---

### Phase 4: Two-Stage Abliteration (1-2 days)

Implement the gate-first, re-track, experts-second flow.

#### 4.1 Add Two-Stage Method

**File:** `src/bruno/model.py`

```python
def abliterate_moe_two_stage(
    self,
    refusal_directions: Tensor,
    bad_prompts: list[str],
    parameters: dict[str, AbliterationParameters],
    gate_strength: float = 0.3,
    expert_threshold: float = 0.1,
    layer_profiles: list["LayerRangeProfile"] | None = None,
    use_mpoa: bool = False,
    mpoa_norm_mode: str = "row",
) -> dict[str, int]:
    """Two-stage MoE abliteration: gate first, then re-track and abliterate experts.

    Stage 1: Abliterate gates to disrupt refusal detection
    Stage 2: Re-track expert activations (routing patterns changed!)
    Stage 3: Abliterate remaining high-activation experts

    This addresses the whack-a-mole problem where abliterating one expert
    causes routing to shift to another refusing expert.

    Args:
        refusal_directions: Per-layer refusal directions
        bad_prompts: Prompts that trigger refusals
        parameters: Abliteration parameters per component
        gate_strength: Strength for gate abliteration
        expert_threshold: Activation threshold for expert targeting
        layer_profiles: Optional per-layer weight profiles
        use_mpoa: Use MPOA norm preservation
        mpoa_norm_mode: MPOA norm mode

    Returns:
        Statistics dict with counts of abliterated components
    """
    stats = {
        'gates_ablated': 0,
        'experts_ablated': 0,
        'shared_experts_ablated': 0,
        'attn_ablated': 0,
    }

    # Stage 1: Gate abliteration
    logger.info("Stage 1: Abliterating MoE gates")
    stats['gates_ablated'] = self.abliterate_gates(
        refusal_directions,
        strength=gate_strength,
        layer_profiles=layer_profiles,
    )
    logger.debug(f"Gates ablated: {stats['gates_ablated']}")

    # Check routing health after gate ablation
    if not self.check_routing_health(bad_prompts[:20]):
        logger.warning("Routing collapsed after gate ablation")
        stats['routing_collapsed'] = True
        return stats

    # Stage 2: Re-track activations with modified routing
    logger.info("Stage 2: Re-tracking expert activations after gate modification")
    new_activations = self.track_expert_activations(bad_prompts)

    if new_activations is None:
        logger.warning("Could not re-track activations after gate ablation")
        return stats

    # Get newly targeted experts
    targeted_experts = self.get_moe_targeted_experts(
        new_activations,
        threshold=expert_threshold,
        top_k=self.settings.moe_top_k_experts,
    )
    logger.debug(
        "Targeted experts after re-tracking",
        n_layers_with_targets=len(targeted_experts),
        total_experts_targeted=sum(len(e) for e in targeted_experts.values()),
    )

    # Stage 3: Targeted expert abliteration
    logger.info("Stage 3: Abliterating targeted experts")
    expert_stats = self.abliterate_moe_targeted(
        refusal_directions,
        parameters,
        targeted_experts,
        layer_profiles=layer_profiles,
        use_mpoa=use_mpoa,
        mpoa_norm_mode=mpoa_norm_mode,
    )

    stats.update(expert_stats)

    return stats
```

#### 4.2 Add Config Settings

**File:** `src/bruno/config.py`

```python
use_two_stage_moe_abliteration: bool = Field(
    default=True,
    description="Use two-stage MoE abliteration: gate first, then re-track activations and abliterate experts. More thorough than single-pass. When False, uses simpler single-pass gate ablation followed by standard expert targeting.",
)
```

#### 4.3 Path Selection Logic

The abliteration flow in `main.py` selects between two paths based on `use_two_stage_moe_abliteration`:

| Setting | Path | Description |
|---------|------|-------------|
| `use_two_stage_moe_abliteration=True` | Two-stage | Gate ablation -> Re-track activations -> Expert ablation. More thorough, accounts for routing changes. |
| `use_two_stage_moe_abliteration=False` | Single-pass | Gate ablation (if enabled) -> Standard expert targeting. Simpler, faster, but may miss rerouted refusals. |

The two-stage approach is recommended (and default) because gate ablation changes routing patterns, so expert targeting needs to be re-computed after gate modification.

---

### Phase 5: Validation Run (1 day)

Run 50-100 trial optimization to compare against baseline.

#### 5.1 Validation Command

```bash
# Quick validation (10 trials)
uv run bruno-vast exec "cd /workspace && bruno \
  --model moonshotai/Moonlight-16B-A3B-Instruct \
  --n-trials 10 \
  --use-gate-abliteration true \
  --use-two-stage-moe-abliteration true \
  --auto-select true \
  --storage sqlite:///gate_test.db \
  --study-name gate_validation"

# Full validation (100 trials)
uv run bruno-vast exec "cd /workspace && bruno \
  --model moonshotai/Moonlight-16B-A3B-Instruct \
  --n-trials 100 \
  --use-gate-abliteration true \
  --use-two-stage-moe-abliteration true \
  --storage sqlite:///gate_full.db \
  --study-name gate_full_validation"
```

#### 5.2 Success Criteria

| Metric | Baseline (Current) | Target | Pass Criteria |
|--------|-------------------|--------|---------------|
| Best refusals | 78/104 (25% success) | <50/104 | >50% improvement |
| KL divergence | 0.29 | <0.5 | Not degraded >50% |
| Routing entropy | - | >0.5 normalized | No collapse |

---

## Risk Mitigation

### Risk 1: Routing Collapse

**Risk:** Gate abliteration could cause all tokens to route to same experts.

**Mitigation:**
- Monitor routing entropy before/after gate ablation
- Add `check_routing_health()` that aborts if entropy < threshold
- Cap `gate_abliteration_strength` at 0.5 by default

### Risk 2: Model Degradation

**Risk:** Aggressive gate modification could damage general capabilities.

**Mitigation:**
- Use MPOA for gate ablation (norm preservation)
- Start with conservative strength (0.3)
- Let Optuna find optimal strength
- Monitor KL divergence and perplexity

### Risk 3: Backward Compatibility

**Risk:** Changes could break dense model abliteration.

**Mitigation:**
- All new code gated with `is_moe_model()` check
- Gate abliteration only activates for MoE models
- Existing dense model tests should pass unchanged

### Risk 4: Missing e_score_correction

**Risk:** Some MoE models may not have the `e_score_correction` bias term.

**Mitigation:**
- Check for attribute existence before modification
- Log warning when bias manipulation is skipped
- Bias manipulation is optional - gate abliteration can work without it

### Risk 5: No Rollback Strategy

**Risk:** If gate abliteration causes routing collapse mid-trial, there's no recovery.

**Mitigation:**
- Check routing health immediately after gate ablation
- Return penalty values to Optuna if routing collapses (trial is pruned)
- Gate weights are cached and restored between trials via `reload_model()`

---

## Testing Strategy

### Unit Tests

```python
# tests/unit/test_model_moe.py

def test_abliterate_gate_basic():
    """Test gate ablation modifies weights correctly."""

def test_abliterate_gate_non_moe():
    """Test gate ablation returns False for non-MoE models."""

def test_routing_entropy_computation():
    """Test routing entropy is computed correctly."""

def test_check_routing_health():
    """Test routing health check works."""

def test_two_stage_abliteration():
    """Test two-stage abliteration runs without error."""
```

### Integration Tests

```python
# tests/integration/test_moe_abliteration.py

def test_gate_abliteration_reduces_refusals():
    """Test that gate ablation reduces refusal rate on Moonlight."""

def test_gate_abliteration_preserves_routing():
    """Test that routing doesn't collapse after gate ablation."""
```

---

## Timeline

| Day | Phase | Tasks |
|-----|-------|-------|
| 1 | Infrastructure | Routing metrics, validation script skeleton |
| 2 | Gate Ablation | `abliterate_gate()`, `abliterate_gates()` |
| 3 | Gate Ablation | Config, Optuna integration, main.py integration |
| 4 | Bias Manipulation | Expert compliance scoring, bias modification |
| 5 | Two-Stage | `abliterate_moe_two_stage()`, config |
| 6 | Testing | Unit tests, integration tests |
| 7-8 | Validation | 100-trial run, compare to baseline |
| 9-10 | Refinement | Tune based on results, documentation |

---

## References

- `docs/MOE_ABLITERATION_RESEARCH.md` - Background research
- `src/bruno/model.py` - Current implementation
- `src/bruno/config.py` - Configuration settings
- DeepSeek-V3 Technical Report - Architecture details
