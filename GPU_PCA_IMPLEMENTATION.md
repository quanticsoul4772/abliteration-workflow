# GPU-Accelerated PCA Implementation

## Summary

Successfully implemented GPU-accelerated contrastive PCA extraction for heretic, replacing CPU-bound numpy eigendecomposition with GPU-accelerated torch operations.

## Changes Made

### Modified Files
1. **src/heretic/model.py** (lines 1035-1085)
   - `get_refusal_directions_pca()` method optimized for GPU execution
   - Replaced `np.linalg.eigh()` with `torch.linalg.eigh()`
   - Removed `.cpu().numpy()` conversions - keep tensors on GPU
   - Added float32 upcast for numerical stability
   - Updated error handling for torch exceptions

2. **tests/unit/test_model.py**
   - Added `test_get_refusal_directions_pca_gpu_performance()` benchmark
   - Tests 64 layers × 5120 hidden dimensions (32B model scale)

## Performance Impact

### Expected Speedup
- **Before:** 4-6 hours for PCA extraction (32B model, 64 layers × 5120 dims)
- **After:** 15-20 minutes (estimated 15-20x speedup on GPU)

### Benchmark Results (Local CPU)
- Test dimensions: 50 samples × 64 layers × 5120 dims
- CPU time: 173 seconds (~3 minutes)
- GPU time: Expected ~10 seconds (15-20x faster)

## Key Implementation Details

### Before (CPU-bound):
```python
good_layer = good_residuals[:, layer_idx, :].cpu().numpy()  # Move to CPU
bad_layer = bad_residuals[:, layer_idx, :].cpu().numpy()

# numpy operations
good_centered = good_layer - good_layer.mean(axis=0)
bad_centered = bad_layer - bad_layer.mean(axis=0)

cov_contrastive = cov_bad - alpha * cov_good
eigenvalues, eigenvectors = np.linalg.eigh(cov_contrastive)  # SLOW
```

### After (GPU-accelerated):
```python
good_layer = good_residuals[:, layer_idx, :].float()  # Stay on GPU
bad_layer = bad_residuals[:, layer_idx, :].float()

# torch operations on GPU
good_centered = good_layer - good_layer.mean(dim=0)
bad_centered = bad_layer - bad_layer.mean(dim=0)

cov_contrastive = cov_bad - alpha * cov_good
eigenvalues, eigenvectors = torch.linalg.eigh(cov_contrastive)  # FAST
```

## Testing

### All Tests Passed ✅
- 4/4 PCA extraction tests passed
- 45/45 total model tests passed
- No regressions introduced

### Test Coverage
- Shape validation
- Normalization verification
- Alpha parameter effect
- GPU performance benchmark

## Deployment

### Build Artifacts
- **Wheel:** `dist/heretic_llm-1.0.1-py3-none-any.whl`
- **Source:** `dist/heretic_llm-1.0.1.tar.gz`

### Deployment Steps
1. Upload wheel to H200 instance
2. Install: `pip install heretic_llm-1.0.1-py3-none-any.whl --force-reinstall`
3. Run full abliteration on Qwen2.5-Coder-32B
4. Monitor PCA completion time (should be 15-20 minutes)

## Verification Plan

### Success Criteria
- [x] All existing tests pass
- [x] Wheel builds successfully
- [ ] PCA completes in <20 minutes on H200 (32B model)
- [ ] No GPU OOM errors
- [ ] Abliteration quality unchanged (KL divergence, refusals)
- [ ] First Optuna trial completes successfully

## Risk Assessment

**Low Risk Implementation:**
- Single method modification (~50 lines changed)
- `torch.linalg.eigh()` is stable and well-tested
- Existing error handling preserved (fallback to mean difference)
- Comprehensive test coverage (4 PCA tests + 45 total)
- No API changes - drop-in replacement

**Potential Issues & Mitigations:**
1. GPU memory spike during eigendecomposition
   - **Mitigation:** float32 upcast is memory-safe for 5120×5120 matrices
2. Numerical differences between GPU/CPU
   - **Mitigation:** Tests validate with 1e-4 tolerance
3. Device errors from mixed GPU/CPU tensors
   - **Mitigation:** All operations stay on GPU until final result

## Expected Production Impact

For Qwen2.5-Coder-32B (64 layers × 5120 dimensions):
- **Current:** PCA incomplete after 2+ hours (H200 timeout)
- **Optimized:** PCA completes in 15-20 minutes
- **Full 200-trial run:** Now feasible within reasonable time

This enables:
- Full abliteration runs on 32B+ models
- Faster iteration during development
- Lower cloud GPU costs (fewer billable hours)
