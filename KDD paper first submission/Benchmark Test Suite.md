#

## Purpose
This test suite validates the correctness and reliability of the  benchmark for online convex optimization. The tests ensure that:

- All mathematical invariants required by the theory are satisfied
- The interaction between optimizers and dynamic environments follows strict temporal consistency
- No information about optimal points leaks to the optimizer during execution
- Metrics are computed according to their theoretical definitions
- Results are fully reproducible across independent runs
- Exported results contain complete information for verification

## Running Tests
To execute the test suite:

```bash
# Install dependencies
pip install pytest numpy

# Run all tests
pytest test_algorithms.py -v
```

The tests should pass completely with the current implementation.

## Test Coverage

The test suite includes comprehensive validation of all core components:

| **Component**                     | **Tested Functionality**                                                                 | **Scientific Purpose**                                                                 |
|----------------------------------|------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| **Landscape Invariants**         | `L(θ, θ) = 0` for all landscapes<br>`∇L(θ, θ) = 0`                                       | Ensures valid dynamic regret (`f_t(θ_t) = 0`) and first-order optimality               |
| **Temporal Consistency**         | Identical `θ_t` across multiple queries in same step<br>Environment locking during step   | Enforces non-anticipative interaction (Protocol A); prevents lookahead                 |
| **Information Barrier**          | No leakage of `θ_t` in `Observation`<br>Immutable query points<br>Blind-value mode       | Guarantees optimizer learns only through oracle feedback, not direct access to optimum  |
| **Lyapunov Metrics**             | Correct `p = ρ + 1` norm<br>Drift-invariant normalization<br>Asymptotic bound estimation | Validates theoretical stabilization guarantees under Hölder-smooth dynamics            |
| **Dynamic Regret**               | `optimum_value` computed with same `θ_t` as `f_t(x_t)`                                   | Ensures mathematically correct regret calculation                                     |
| **Benchmark Runner**             | End-to-end execution<br>Trajectory recording<br>Metric aggregation                       | Validates full pipeline from setup to result collection                               |
| **Batch Runner**                 | Multi-seed execution<br>Statistical aggregation (mean/std)                               | Enables significance testing across random seeds                                      |
| **Offline Oracle**               | Deterministic replay of recorded `θ_t` sequence                                          | Provides reproducible environment evolution for fair algorithm comparison              |
| **Error Handling**               | Consistent `ρ` validation<br>Protocol violation detection                                | Prevents misconfiguration that would invalidate theoretical claims                    |
| **Reproducibility & Export**     | JSON/CSV export with full signatures<br>Ground truth inclusion<br>Seed-based reproducibility | Meets scientific publishing standards (Pineau et al., 2021)                           |

This coverage ensures the benchmark adheres to theoretical foundations while maintaining engineering robustness for research use.
