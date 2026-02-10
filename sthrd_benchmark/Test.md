# ST-HRD Benchmark Test Suite

## Purpose
This test suite validates the scientific correctness and implementation integrity of the **ST-HRD (Smooth Time-varying High-dimensional Regret Dynamics)** benchmark for online convex optimization. It ensures:

- ✅ **Scientific invariants** (e.g., `L(θ,θ) = 0` for all landscapes)
- ✅ **Protocol A enforcement** (temporal consistency between optimizer and environment)
- ✅ **Information barrier integrity** (no leakage of optimal points to optimizers)
- ✅ **Theoretical metric correctness** (Lyapunov, dynamic regret, tracking error)
- ✅ **Full reproducibility** (identical results across independent runs with same seeds)
- ✅ **Export fidelity** (complete reproducibility signatures in JSON exports)

## Critical Prerequisites
**Before running tests**, apply these minimal fixes to core code:

### 1. `metrics.py` → `MetricsCollection` class
```python
def get_current_values(self) -> Dict[str, float]:
    """Return current values of all metrics for BenchmarkRunner integration."""
    return {
        metric.name: metric._current_value
        for metric in self.metrics
        if metric._current_value is not None
    }
```

### 2. `benchmark.py` → `BenchmarkRunner.run()` (line ~585)
```python
# REPLACE:
self.metrics_history = {name: [] for name in self.metrics.metric_names}

# WITH:
self.metrics_history = {m.name: [] for m in self.metrics.metrics}
```

### 3. `oracle.py` → `OfflineOracle.start_step()`
Remove `super().start_step()` call to prevent `_cached_theta` overwrite from dummy environment. Directly set:
```python
self._cached_theta = self.recorded_thetas[t].copy()
```

### 4. `benchmark.py` → `BenchmarkRunner.run()` (first lines)
```python
if seed is not None:
    np.random.seed(seed)   # Fix legacy np.random.* calls
    random.seed(seed)      # Fix Python's random module
```

> ⚠️ **Without these fixes**, tests will fail with:
> - `AttributeError: 'MetricsCollection' object has no attribute 'get_current_values'`
> - `OfflineOracle` returning zero trajectories instead of recorded sequences
> - Non-reproducible results despite identical seeds

## Running Tests
```bash
# Install dependencies
pip install pytest numpy

# Run all tests
pytest test_algorithms.py -v

# Expected output (after applying fixes):
# 16 passed in X.XXs
```

## Test Coverage Summary
| Category | Tests | Scientific Basis |
|----------|-------|------------------|
| **Landscape Invariants** | 2 | `L(θ,θ)=0`, `∇L(θ,θ)=0` (regret validity) |
| **Protocol A** | 2 | Temporal locking, consistent `θ_t` per step |
| **Information Barrier** | 2 | No `θ_t` leakage, blind-value mode |
| **Lyapunov Metrics** | 2 | Correct `p=ρ+1` norm, drift normalization |
| **Dynamic Regret** | 1 | Consistent `f_t(θ_t)` evaluation |
| **Benchmark Runner** | 2 | End-to-end execution, batch aggregation |
| **Offline Oracle** | 1 | Deterministic replay of recorded trajectories |
| **Error Handling** | 2 | `ρ` validation, protocol violation detection |
| **Reproducibility** | 2 | JSON export fidelity, independent run consistency |

## Why These Tests Matter
These tests enforce **scientific rigor** required for publication-quality benchmarks:
- Prevents invalid regret calculations (via `L(θ,θ)=0` invariant)
- Eliminates lookahead cheating 
- Enables fair algorithm comparison (via drift-normalized metrics)
- Guarantees reproducibility 

