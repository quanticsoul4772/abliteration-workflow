# Bruno/Heretic Technical Debt & Silent Failures

**Created:** 2025-01-31
**Priority:** CRITICAL - These issues caused a production run to waste hours with incorrect settings

---

## Executive Summary

A production abliteration run on Moonlight-16B-A3B-Instruct ran for 123+ trials with **incorrect configuration** despite having correct values in `config.toml`. The following issues were identified:

| Setting | Config Value | Actual Value | Root Cause |
|---------|--------------|--------------|------------|
| `use_mpoa` | `true` | `false` | TOML not loaded / default override |
| `use_concept_cones` | `false` | `true` | Default is `True` in code |
| `activation_target_percentile` | `0.60` | `0.75` | Default is `0.75` in code |
| `direction_weights[1]` | `0.3` | `0.00` | Eigenvalue weights override |

---

## Critical Issues

### 1. TOML File Silent Non-Loading ✅ FIXED

**File:** `src/bruno/config.py`, `src/bruno/config_verify.py`

**Problem:** The `TomlConfigSettingsSource` looks for `config.toml` in the **current working directory**. If running from a different directory (e.g., `/workspace` on cloud instances), the file is not found and **all Field defaults are used silently**.

**Solution Implemented:**
1. ✅ Added `log_config_status()` in `config_verify.py` - logs whether config.toml was found
2. ✅ Added `verify_config_was_parsed()` - detects when TOML values weren't loaded
3. ✅ Both called at startup in `main.py`
4. ⬜ `--config` CLI flag deferred (low priority)

---

### 2. Dangerous Default Values ✅ FIXED

**File:** `src/bruno/config.py`

Several features had defaults that contradicted safe/conservative behavior:

| Field | Old Default | New Default | Status |
|-------|-------------|-------------|--------|
| `use_concept_cones` | `True` | `False` | ✅ Fixed |
| `use_caa` | `True` | `False` | ✅ Fixed |
| `activation_target_percentile` | `0.75` | `0.75` | Kept (user can override) |

**Solution Implemented:** Changed defaults in `config.py` to be opt-in for experimental features.

---

### 3. Eigenvalue Weights Override User Settings ✅ FIXED

**File:** `src/bruno/main.py` (line ~349)

**Problem:**
```python
direction_weights = direction_result.direction_weights or settings.direction_weights
```

When `use_eigenvalue_weights=True` (the default), `direction_result.direction_weights` contains computed values from PCA eigenvalues. The user's `direction_weights = [1.0, 0.3]` was **never used**.

**Solution Implemented:**
```python
if settings.use_eigenvalue_weights and direction_result.direction_weights is not None:
    direction_weights = direction_result.direction_weights
else:
    direction_weights = settings.direction_weights
```

Now user-specified weights are respected when `use_eigenvalue_weights=False`.

---

### 4. No Configuration Verification Logging ✅ FIXED

**File:** `src/bruno/main.py`, `src/bruno/config_verify.py`

**Problem:** Zero logging to confirm configuration was loaded correctly.

**Solution Implemented:**
1. ✅ `log_config_status()` - Logs config file path at startup
2. ✅ `log_effective_settings()` - Logs key settings (features enabled/disabled)
3. ✅ `verify_config_was_parsed()` - Detects when TOML values weren't loaded
4. ✅ `bruno show-config` CLI command to dump effective settings

---

### 5. No Feature Application Confirmation ✅ FIXED

**File:** `src/bruno/main.py`, `src/bruno/model.py`

**Problem:** When MPOA, CAA, or concept cones are applied, there was no confirmation logging.

**Solution Implemented:**
- ✅ Added `[bold green]MPOA enabled[/bold green]` / `[dim]MPOA disabled[/dim]` in main.py
- ✅ Added `[bold green]CAA enabled[/bold green]` / `[dim]CAA disabled[/dim]` in main.py
- ✅ Added `[bold green]Concept Cones enabled[/bold green]` / `[dim]Concept Cones disabled[/dim]` in main.py
- ✅ Feature status is now always visible at startup for all three features

---

### 6. Activation Calibration Silent Fallback ✅ FIXED

**File:** `src/bruno/model.py`

**Problem:** `compute_calibrated_weight()` had a silent fallback when mean projection was near zero.

**Solution Implemented:**
```python
if stats.mean_projection > EPSILON:
    calibration_factor = target_value / stats.mean_projection
else:
    logger.warning("Activation calibration fallback triggered - mean projection near zero")
    calibration_factor = 1.0
```

Now logs a warning when the fallback is triggered.

---

## Medium Priority Issues

### 7. Pydantic Settings Priority Confusion ✅ DOCUMENTED

**File:** `src/bruno/config.py`, `README.md`

**Problem:** CLI arguments always override TOML, which is confusing when users expect TOML to be the source of truth.

**Solution Implemented:**
- ✅ Documented in README.md "Configuration Priority" section
- ✅ Added note about CLI overriding TOML in README
- ✅ Added `bruno show-config` command to verify effective settings

---

### 8. C4 Dataset Config Not Auto-Detected ✅ FIXED

**File:** `src/bruno/utils.py`

**Problem:** The `unhelpfulness_prompts` dataset uses C4 which requires a config name ("en"), but this was often forgotten.

**Solution Implemented:**
- ✅ Improved error message in `utils.py` with clear fix hint
- ✅ Error now shows: "C4 dataset requires a config parameter. Add: --unhelpfulness-prompts.config en"
- ✅ Uses exact match for C4 detection (`dataset in ("allenai/c4", "c4")`) to avoid false positives

---

### 9. No Validation of Sacred Direction Overlap

**File:** `src/bruno/phases/direction_extraction.py`

**Problem:** `sacred_overlap_threshold` is checked but only logs a warning. High overlap (e.g., 0.9) may indicate the ablation will severely damage capabilities, but the run continues anyway.

**Fix Required:** Make high overlap a fatal error (or add `--force` flag to continue).

---

### 10. Circuit Ablation GQA Check Too Late ✅ FIXED

**File:** `src/bruno/main.py`

**Problem:** GQA model check happened at runtime inside `restore_and_abliterate_trial()`. The run discovered it can't use circuit ablation only after direction extraction was complete.

**Solution Implemented:**
- ✅ Early GQA detection immediately after model load (line ~210-216)
- ✅ Fails fast with clear error message and solutions
- ✅ Prevents wasting hours of direction extraction on incompatible models

---

## Low Priority / Code Quality

### 11. Inconsistent use_mpoa Parameter Threading

**File:** `src/bruno/model.py`

**Problem:** `use_mpoa` is passed through 6+ method calls before being used. Any new method in the chain must remember to add the parameter.

**Fix:** Consider a configuration object or class attribute.

---

### 12. Magic Numbers in Constants

**File:** `src/bruno/constants.py`

Many values like `NEAR_ZERO = 1e-8`, `EPSILON = 1e-7` are defined but not consistently used. Some methods use hardcoded values instead.

---

### 13. Test Coverage for Configuration Loading ✅ FIXED

**File:** `tests/integration/test_config_loading.py`

**Problem:** Tests mocked CLI parsing and chdir to temp directory, so they never tested real TOML loading behavior.

**Solution Implemented:**
- ✅ Added `tests/integration/test_config_loading.py` with 13 comprehensive tests
- ✅ Tests for `log_config_status()`, `verify_config_was_parsed()`, `log_effective_settings()`
- ✅ Tests for direction_weights eigenvalue override fix
- ✅ Uses pytest fixtures for MockSettings classes

---

## Recommended Fixes (Priority Order)

### Phase 1: Critical (Do Now) ✅ COMPLETED
1. ✅ Add config load verification logging in `main.py`
2. ✅ Fix default values for `use_concept_cones` and `use_caa`
3. ✅ Add feature application logging in `model.py`
4. ✅ Fix eigenvalue weights override logic

### Phase 2: Important ✅ COMPLETED
5. ✅ Add `--show-config` flag (cli.py)
6. ✅ Improve C4 error message (utils.py)
7. ✅ Early GQA detection for circuit ablation (main.py)
8. ✅ Add type hints to config_verify.py
9. ✅ Verify TOML was actually parsed (config_verify.py)
10. ✅ Add CAA/Concept Cones status logging (main.py)

### Phase 3: Nice to Have ✅ COMPLETED
11. ✅ Add integration test for TOML loading (tests/integration/test_config_loading.py)
12. ✅ Document settings priority in README
13. ⬜ Refactor use_mpoa threading (deferred - low priority)

---

## Testing Checklist

After fixes are applied, verify:

- [x] Running without config.toml shows warning
- [x] Running with config.toml shows "Configuration loaded from: config.toml"
- [x] `use_concept_cones=false` in TOML results in no concept cone logging
- [x] `use_mpoa=true` in TOML results in "MPOA enabled" logging
- [x] User-specified `direction_weights` are used when `use_eigenvalue_weights=false`
- [x] `bruno show-config` displays all effective settings
- [x] Early GQA detection prevents circuit ablation on GQA models
- [x] verify_config_was_parsed() warns when TOML values aren't loaded
- [x] Integration tests pass for config loading
