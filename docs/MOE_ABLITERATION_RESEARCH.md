# MoE Abliteration Research: Challenges and New Approaches

**Date:** February 2026
**Status:** Active Research
**Current Results:** 25% success rate (78/104 refusals) on Moonlight-16B

---

## Executive Summary

MoE (Mixture of Experts) models present unique challenges for abliteration that don't exist in dense models. While dense models like Qwen2.5-Coder-32B achieve 0 refusals with standard abliteration, MoE models like Moonlight-16B only achieve ~25% success rate using the same techniques.

This document analyzes why, and proposes new approaches to improve MoE abliteration.

---

## Current Implementation

### What We Do Now

1. **Router-aware targeting**: Track which experts activate on refusal prompts
2. **Shared expert abliteration**: Always abliterate shared experts (always active for every token)
3. **Top-K expert targeting**: Only abliterate experts with high activation frequency on refusal prompts
4. **Standard orthogonalization**: Remove refusal direction from expert `down_proj` weights

### Current Results (Moonlight-16B v3 Run)

| Metric | Value |
|--------|-------|
| Total trials | 95/300 |
| Best refusals | 78/104 (25% success) |
| Best KL divergence | 0.293 |
| Experts abliterated per trial | 211-850 |
| Shared experts abliterated | 4-16 per trial |
| Attention layers abliterated | 3-22 per trial |

### Comparison: Dense vs MoE

| Model | Architecture | Best Refusals | Success Rate |
|-------|--------------|---------------|--------------|
| Qwen2.5-Coder-32B | Dense | 0/104 | 100% |
| Moonlight-16B | MoE (DeepSeek-V3) | 78/104 | 25% |

---

## DeepSeek V3 / Moonlight Architecture

Understanding the architecture is critical to improving abliteration.

### Key Components

```
Input Token
    │
    ▼
┌─────────────────┐
│  Gate/Router    │ ◄── Sigmoid gating (not softmax)
│  (gate.weight)  │     Computes affinity scores
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐ ┌───────────────┐
│Shared │ │ Top-K Routed  │
│Expert │ │   Experts     │
│(1)    │ │ (8 of 64)     │
└───┬───┘ └───────┬───────┘
    │             │
    └──────┬──────┘
           │
           ▼
      Combined Output
```

### Architecture Details

| Component | Description |
|-----------|-------------|
| **Gating** | Sigmoid activation (not softmax), allows fine-grained control |
| **Bias terms** | `e_score_correction` - adjusts routing preferences per expert |
| **Shared experts** | 1 per layer, processes ALL tokens |
| **Routed experts** | 64 per layer, top-8 selected per token |
| **Selection** | Gate projects input to expert affinities, top-K selected |

### Code Locations

```python
# Gate/router weight
layer.mlp.gate.weight  # [n_experts, hidden_dim]

# Expert selection bias (load balancing)
layer.mlp.gate.e_score_correction  # [n_experts]

# Shared expert
layer.mlp.shared_experts.down_proj  # Always active

# Routed experts
layer.mlp.experts[i].down_proj  # Only active when selected
```

---

## Why MoE Abliteration is Harder

### The Core Problem

**In MoE models, refusal behavior is encoded in TWO places:**
1. The experts themselves (what we currently target)
2. The gating/routing mechanism (what we're missing)

### The Whack-a-Mole Problem

Current approach:
1. Track which experts activate on harmful prompts
2. Abliterate those experts
3. **Problem:** The gate simply routes to OTHER experts that also refuse

The gate has learned to recognize "harmful" content and makes routing decisions based on it. When you abliterate Expert A, it routes to Expert B which also refuses.

### Why Dense Models Succeed

In dense models:
- Every weight matrix processes every token
- Refusal direction is concentrated in specific layers
- Removing the direction removes the capability globally

### Why MoE Models Struggle

In MoE models:
- Gate detects harmful content BEFORE routing
- Multiple experts have learned refusal patterns
- Dynamic rerouting bypasses abliterated experts
- Refusal is distributed across the expert pool

---

## New Approaches (Prioritized)

### 1. Gate-Level Abliteration (Highest Priority)

**Concept:** The gate projection matrix is where harmful content gets "detected." Abliterate the gate's ability to recognize refusal-triggering inputs.

**Implementation:**

```python
def abliterate_gate(self, layer_idx: int, refusal_direction: Tensor) -> None:
    """Abliterate the gate's ability to detect refusal-related content.

    The gate projects inputs to expert affinity scores. If the gate can
    detect refusal-related content in the input, it can route accordingly.
    By removing the refusal direction from the gate weights, we prevent
    this detection.
    """
    layer = self.get_layers()[layer_idx]
    gate = layer.mlp.gate

    # Gate weight: [n_experts, hidden_dim] - projects input to expert affinities
    gate_weight = gate.weight.data

    # Create projector to remove refusal direction
    direction = refusal_direction / refusal_direction.norm()
    projector = torch.outer(direction, direction).to(gate_weight.device)

    # Remove the refusal-detecting component from gate
    # This prevents the gate from "seeing" the refusal direction in inputs
    gate.weight.data = gate_weight - gate_weight @ projector
```

**Why this should work:** The gate sees input BEFORE routing. If it can't detect harmful content, it can't make routing decisions based on it.

### 2. Two-Stage Abliteration (High Priority)

**Concept:** Abliterate gate first, then re-track activations and abliterate experts.

**Implementation:**

```python
def abliterate_moe_two_stage(
    self,
    refusal_directions: Tensor,
    bad_prompts: list[str],
    threshold: float = 0.1,
) -> dict[str, int]:
    """Two-stage MoE abliteration: gate first, then experts.

    Stage 1: Abliterate all gates to remove refusal detection
    Stage 2: Re-track expert activations (routing is now different!)
    Stage 3: Abliterate remaining problematic experts
    """
    stats = {'gates_ablated': 0, 'experts_ablated': 0}

    # Stage 1: Abliterate all gates
    for layer_idx in range(len(self.get_layers())):
        self.abliterate_gate(layer_idx, refusal_directions[layer_idx + 1])
        stats['gates_ablated'] += 1

    # Stage 2: Re-track activations with new routing
    new_activations = self.track_expert_activations(bad_prompts)

    # Stage 3: Abliterate remaining high-activation experts
    targeted_experts = self.get_moe_targeted_experts(new_activations, threshold)

    for layer_idx, experts in targeted_experts.items():
        for exp_idx in experts:
            self.abliterate_expert(layer_idx, exp_idx, refusal_directions[layer_idx + 1])
            stats['experts_ablated'] += 1

    return stats
```

**Why this should work:** Gate abliteration changes routing patterns, so expert targeting must happen AFTER gate changes.

### 3. Expert Bias Manipulation (High Priority)

**Concept:** Modify the `e_score_correction` bias terms to prefer compliant experts.

**Implementation:**

```python
def identify_expert_compliance(
    self,
    prompts: list[str],
    tokenizer,
) -> dict[int, dict[int, str]]:
    """Classify each expert as Refusing/Neutral/Compliant.

    Force-route each prompt through each expert individually and
    measure the output for refusal markers.
    """
    layer_classifications = {}

    for layer_idx, layer in enumerate(self.get_layers()):
        if not hasattr(layer.mlp, 'experts'):
            continue

        expert_scores = {}
        for exp_idx in range(len(layer.mlp.experts)):
            # Force route through this expert only
            refusal_count = self._evaluate_expert_outputs(
                layer_idx, exp_idx, prompts, tokenizer
            )

            # Classify
            if refusal_count > len(prompts) * 0.7:
                expert_scores[exp_idx] = 'refusing'
            elif refusal_count < len(prompts) * 0.3:
                expert_scores[exp_idx] = 'compliant'
            else:
                expert_scores[exp_idx] = 'neutral'

        layer_classifications[layer_idx] = expert_scores

    return layer_classifications


def modify_expert_biases(
    self,
    classifications: dict[int, dict[int, str]],
    bias_delta: float = 0.5,
) -> None:
    """Modify bias terms to prefer compliant experts."""

    for layer_idx, expert_classes in classifications.items():
        layer = self.get_layers()[layer_idx]

        if not hasattr(layer.mlp.gate, 'e_score_correction'):
            continue

        bias = layer.mlp.gate.e_score_correction

        for exp_idx, classification in expert_classes.items():
            if classification == 'compliant':
                bias.data[exp_idx] += bias_delta  # Increase affinity
            elif classification == 'refusing':
                bias.data[exp_idx] -= bias_delta  # Decrease affinity
```

**Why this should work:** Changes routing preferences without modifying expert capabilities. The gate will naturally prefer compliant experts.

### 4. Universal Expert Direction (Medium-High Priority)

**Concept:** Find the COMMON refusal direction across ALL experts, then abliterate uniformly.

**Implementation:**

```python
def compute_universal_expert_direction(
    self,
    good_prompts: list[str],
    bad_prompts: list[str],
) -> dict[int, Tensor]:
    """Compute a universal refusal direction per layer across all experts.

    Instead of per-expert directions, we find what's COMMON across experts.
    This prevents whack-a-mole - we remove the same direction everywhere.
    """
    universal_directions = {}

    for layer_idx, layer in enumerate(self.get_layers()):
        if not hasattr(layer.mlp, 'experts'):
            continue

        # Collect per-expert directions
        expert_directions = []

        for exp_idx, expert in enumerate(layer.mlp.experts):
            # Get expert's output on good vs bad prompts
            good_outputs = self._get_expert_outputs(layer_idx, exp_idx, good_prompts)
            bad_outputs = self._get_expert_outputs(layer_idx, exp_idx, bad_prompts)

            # Compute direction for this expert
            direction = bad_outputs.mean(dim=0) - good_outputs.mean(dim=0)
            direction = direction / direction.norm()
            expert_directions.append(direction)

        # Stack and find principal component (common direction)
        stacked = torch.stack(expert_directions)
        # SVD to find dominant direction
        U, S, V = torch.svd(stacked)
        universal_directions[layer_idx] = V[:, 0]  # First principal component

    return universal_directions


def abliterate_universal(
    self,
    universal_directions: dict[int, Tensor],
) -> None:
    """Abliterate the universal direction from ALL experts simultaneously."""

    for layer_idx, direction in universal_directions.items():
        layer = self.get_layers()[layer_idx]
        projector = torch.outer(direction, direction)

        # Abliterate from ALL experts
        for expert in layer.mlp.experts:
            expert.down_proj.weight.data -= expert.down_proj.weight.data @ projector

        # Also from shared expert
        if hasattr(layer.mlp, 'shared_experts'):
            shared = layer.mlp.shared_experts
            shared.down_proj.weight.data -= shared.down_proj.weight.data @ projector
```

**Why this should work:** Prevents rerouting to other refusing experts - you're removing the same direction everywhere simultaneously.

### 5. Pre-Router Direction Removal (Medium Priority)

**Concept:** Find the direction that predicts refusal BEFORE routing, and remove it from the gate's input space.

**Implementation:**

```python
def abliterate_pre_router(
    self,
    good_prompts: list[str],
    bad_prompts: list[str],
) -> None:
    """Remove refusal-predictive direction from gate input space.

    The gate receives hidden states as input. If we can find what makes
    hidden states 'look harmful' to the gate, we can remove that signal.
    """
    # Get hidden states right before MoE layers
    good_hidden = self._get_pre_moe_hidden_states(good_prompts)
    bad_hidden = self._get_pre_moe_hidden_states(bad_prompts)

    for layer_idx in range(len(self.get_layers())):
        # Direction that distinguishes good from bad before routing
        pre_router_direction = (
            bad_hidden[layer_idx].mean(dim=0) -
            good_hidden[layer_idx].mean(dim=0)
        )
        pre_router_direction = pre_router_direction / pre_router_direction.norm()

        # Abliterate from preceding layer's output projection
        # This removes the signal before it reaches the gate
        prev_layer = self.get_layers()[layer_idx - 1] if layer_idx > 0 else None
        if prev_layer and hasattr(prev_layer.self_attn, 'o_proj'):
            self._abliterate_weight(
                prev_layer.self_attn.o_proj.weight,
                pre_router_direction
            )
```

**Why this should work:** Removes the "signal" that triggers refusal detection before it even reaches the routing mechanism.

---

## Experimental Plan

### Phase 1: Gate Abliteration (Week 1)

1. Implement `abliterate_gate()` method
2. Test on Moonlight-16B with gate-only abliteration
3. Measure: refusal rate, KL divergence, benchmark scores
4. Compare to current expert-only approach

### Phase 2: Two-Stage Abliteration (Week 1-2)

1. Implement two-stage process (gate → re-track → experts)
2. Test with varying gate abliteration strengths
3. Find optimal combination of gate + expert abliteration

### Phase 3: Bias Manipulation (Week 2)

1. Implement expert compliance classification
2. Implement bias modification
3. Test alone and combined with gate abliteration
4. Measure routing changes

### Phase 4: Universal Direction (Week 2-3)

1. Implement universal direction computation
2. Test universal vs targeted abliteration
3. Measure coverage across expert pool

### Phase 5: Combined Approach (Week 3-4)

1. Combine most effective techniques
2. Tune hyperparameters via Optuna
3. Benchmark against dense model results
4. Document best practices for MoE

---

## Success Metrics

| Metric | Current | Target | Stretch |
|--------|---------|--------|---------|
| Refusal rate | 75% | 25% | 5% |
| KL divergence | 0.29 | <0.5 | <0.3 |
| MMLU | 48% | 45%+ | 50%+ |
| HellaSwag | 56% | 54%+ | 58%+ |

---

## Key Insights

### The Paradigm Shift

**Old paradigm:** "Which experts process harmful prompts? Abliterate those."

**New paradigm:** "The gate detects harmful content BEFORE routing. Abliterate the gate's detection capability first, then handle residual refusal in experts."

### Why We're Stuck at 25%

We've been treating symptoms (expert refusal) without addressing the root cause (gate detection and rerouting). The gate learns to recognize harmful content patterns and routes accordingly. Even after abliterating targeted experts, the gate routes to other experts that also refuse.

### The Path Forward

1. **Gate abliteration** - Prevent harmful content detection
2. **Bias manipulation** - Steer routing toward compliant experts
3. **Universal direction** - Remove refusal capability everywhere
4. **Iterative refinement** - Re-track and re-abliterate after each change

---

## References

### Papers
- DeepSeek-V3 Technical Report (arXiv 2412.19437)
- DeepSeekMoE: Towards Ultimate Expert Specialization (arXiv 2401.06066)
- On DeepSeekMoE: Statistical Benefits of Shared Experts and Normalized Sigmoid Gating (arXiv 2505.10860)

### Code
- `src/bruno/model.py` - Current MoE abliteration implementation
- `src/bruno/config.py` - MoE configuration settings
- `docs/MOONLIGHT_ABLITERATION_PLAN.md` - Previous abliteration attempts

### Related Work
- Abliteration for dense models (original heretic/bruno)
- Activation engineering research
- Expert specialization studies

---

## Appendix: Current MoE Code Analysis

### `track_expert_activations()` - What We Track

```python
# Hooks on gate to capture expert selection
def gate_hook(module, input, output):
    topk_idx = output[0]  # Expert indices selected
    # Count how many times each expert is selected
```

**Limitation:** Only tracks INPUT routing, not OUTPUT behavior.

### `abliterate_moe_targeted()` - What We Modify

```python
# For each targeted expert
expert.down_proj.weight.data = W - W @ projector

# For shared expert
shared_experts.down_proj.weight.data = W - W @ projector

# For attention
attn.o_proj.weight.data = W - W @ projector
```

**Limitation:** Doesn't touch gate weights or bias terms.

### Missing Components

| Component | Currently Modified | Proposed |
|-----------|-------------------|----------|
| Expert down_proj | Yes | Yes |
| Shared expert | Yes | Yes |
| Attention o_proj | Yes | Yes |
| Gate weight | **No** | **Yes** |
| Gate bias (e_score_correction) | **No** | **Yes** |
| Expert selection logic | **No** | **Yes** |

---

*Last Updated: February 2026*
*Status: Active Research - Awaiting Implementation*
