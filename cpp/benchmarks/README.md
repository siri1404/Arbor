# Arbor Benchmarks

Production-grade performance benchmarks for the Arbor quantitative trading engine using [Google Benchmark](https://github.com/google/benchmark).

## Quick Start

### Run All Benchmarks

```bash
cd cpp
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
cd build

# Run all benchmarks
./orderbook_benchmark
./options_benchmark
./lockfree_benchmark
./montecarlo_benchmark
./production_benchmark
```

### Analyze Results

```bash
cd cpp/benchmarks

# View summary
python3 analyze_benchmarks.py results/orderbook_results_2026_02_04.json

# Compare with baseline
python3 analyze_benchmarks.py results/orderbook_results_2026_02_04.json \
  --compare results/orderbook_results_2026_01_28.json --threshold 0.10

# Generate reports
python3 analyze_benchmarks.py results/orderbook_results_2026_02_04.json \
  --markdown --html
```

## Benchmark Suites

### 1. Order Book Benchmarks (`orderbook_benchmark.cpp`)

Tests matching engine performance with realistic order flow.

**Scenarios:**
- `Insert/1000` - Insert 1,000 orders
- `Insert/10000` - Insert 10,000 orders
- `Insert/100000` - Insert 100,000 orders
- `BestBid/1000` - Query best bid (1,000 orders in book)
- `BestAsk/1000` - Query best ask
- `Match/1000` - Execute trade matching
- `CancelOrder/1000` - Cancel order from book
- `FullDepth/1000` - Get complete market depth

**Expected Performance:**
```
Insert (1000):     1,250 µs
BestBid (cached):    18.75 µs  [O(1) lookup]
Match:               185 µs
```

### 2. Options Pricing Benchmarks (`options_benchmark.cpp`)

Comprehensive pricing model performance testing.

**Models:**
- Black-Scholes (European options)
- Black-Scholes Greeks (Delta, Gamma, Vega, Theta, Rho)
- Heston (stochastic volatility)
- Merton Jump-Diffusion
- Asian options
- Barrier options
- Lookback options
- Digital options

**Expected Performance:**
```
Black-Scholes:      3.25 µs  [307k prices/sec]
Heston:           125.78 µs  [79.5k prices/sec]
Asian Arithmetic: 1,523 µs   [657 prices/sec]
```

### 3. Lock-Free Benchmarks (`lockfree_benchmark.cpp`)

Concurrent data structure performance with contention modeling.

**Tests:**
- SPSC (Single Producer, Single Consumer) queue
- MPSC (Multiple Producer, Single Consumer) queue
- MPMC (Multiple Producer, Multiple Consumer) queue
- Mutex-based queue (baseline for comparison)
- Varying thread counts (1, 2, 4, 8, 10 threads)

**Expected Performance:**
```
SPSC Enqueue:         0.85 µs  [1,176k ops/sec]
MPMC (10 threads):   87.34 µs  [11.4k ops/sec]
Mutex Baseline:      245.67 µs  [4.1k ops/sec]
Lock-Free Advantage:  2-200x faster
```

### 4. Monte Carlo Benchmarks (`montecarlo_benchmark.cpp`)

Simulation performance with variance reduction techniques.

**Techniques:**
- Vanilla Monte Carlo
- Antithetic variates
- Control variates
- Importance sampling
- Sobol sequences (quasi-random)
- SIMD vectorization (AVX2)

**Path Counts:**
- 100,000 paths
- 1,000,000 paths

**Expected Performance:**
```
Vanilla (100k):      8,450 µs  [118 evals/sec]
SIMD Vectorized:     2,146 µs  [466 evals/sec]  [3.9x faster]
Sobol (100k):        3,890 µs  [257 evals/sec]  [2.2x faster]
```

### 5. Production Benchmarks (`production_benchmark.cpp`)

End-to-end workflows simulating real trading scenarios.

**Scenarios:**
- Market snapshot (100, 1,000 symbols)
- Portfolio rebalancing (1,000 positions)
- VaR/CVaR calculation (100k samples)
- Volatility smile calibration (Heston)
- Greeks hedging (100 positions)
- Full valuation cycle
- End-of-day settlement (10k trades)
- HFT trade execution

**Expected Performance:**
```
Market Snapshot (100):     2,341 µs  [427/sec]
Portfolio Rebalance:      45,670 µs  [22/sec]
VaR Calculation:         156,340 µs  [6/sec]
Full Valuation Cycle:  1,234,560 µs  [1/sec]
HFT Trade Execution:       185.67 µs [5,385/sec]
```

## Output Formats

### JSON Output (for CI/CD comparison)

```bash
./orderbook_benchmark --benchmark_format=json --benchmark_out=results.json
```

**Structure:**
```json
{
  "context": {
    "date": "2026-02-04T14:32:15Z",
    "host_name": "ubuntu-latest",
    "build_type": "Release"
  },
  "benchmarks": [
    {
      "name": "OrderBook/Insert/1000",
      "iterations": 1000,
      "real_time": 1250.5,
      "cpu_time": 1248.3,
      "time_unit": "us",
      "items_per_second": 800.32
    }
  ]
}
```

### Console Output

```bash
./orderbook_benchmark
```

Produces human-readable table with ASCII formatting.

## Performance Regression Detection

The CI/CD pipeline automatically:

1. **Runs benchmarks** on every commit
2. **Compares with baseline** (main branch results)
3. **Detects regressions** >10% slowdown
4. **Generates reports** with detailed analysis
5. **Comments on PRs** with performance impact

### Threshold Policy

| Regression | Action | Acceptable |
|-----------|--------|-----------|
| <5% | Info only | ✅ Yes |
| 5-10% | Explanation required | ⚠️ Case-by-case |
| >10% | Blocks merge | ❌ No |

### Historical Results

See [RESULTS.md](RESULTS.md) for:
- Current benchmark results (2026-02-04)
- Historical data (last 7 days)
- Performance trend analysis
- Optimization recommendations

## Result Files

```
benchmarks/
├── results/
│   ├── orderbook_results_2026_02_04.json
│   ├── options_results_2026_02_04.json
│   ├── lockfree_results_2026_02_04.json
│   ├── montecarlo_results_2026_02_04.json
│   ├── production_results_2026_02_04.json
│   ├── orderbook_results_2026_02_01.json
│   └── orderbook_results_2026_01_28.json
├── analyze_benchmarks.py
├── RESULTS.md
└── README.md (this file)
```

## Tools & Scripts

### `analyze_benchmarks.py`

Command-line tool for analyzing and comparing benchmark results.

**Usage:**
```bash
# Print summary
python3 analyze_benchmarks.py results/orderbook_results_2026_02_04.json

# Compare with baseline
python3 analyze_benchmarks.py results/orderbook_results_2026_02_04.json \
  --compare results/orderbook_results_2026_01_28.json

# Generate markdown report
python3 analyze_benchmarks.py results/orderbook_results_2026_02_04.json --markdown

# Generate HTML report
python3 analyze_benchmarks.py results/orderbook_results_2026_02_04.json --html
```

## Best Practices

### For Development

1. **Run locally before pushing:**
   ```bash
   cmake --build build
   cd build && ./orderbook_benchmark
   ```

2. **Check key metrics:**
   - Is throughput acceptable?
   - Any unexpected slowdowns?
   - Memory usage reasonable?

3. **Document performance changes:**
   - If you expect regression, explain in commit message
   - Link to PR discussion if it's accepted trade-off

### For Optimization

1. **Profile first:**
   ```bash
   # With perf (Linux)
   perf record -g ./orderbook_benchmark
   perf report
   ```

2. **Benchmark after:**
   ```bash
   ./orderbook_benchmark --benchmark_format=json --benchmark_out=new.json
   python3 analyze_benchmarks.py new.json --compare old.json
   ```

3. **Verify improvements are real:**
   - Run multiple times (variance is normal)
   - Check cold vs warm cache behavior
   - Profile to confirm bottleneck is addressed

## Compiler Optimization Flags

Benchmarks use `-O3 -march=native` by default:

- `-O3`: Maximum optimization
- `-march=native`: Use CPU-specific instructions (AVX2, SSE4.2, etc.)
- `-DNDEBUG`: Remove assertions (Release builds only)

For comparison, rebuild with different flags:
```bash
cmake -B build-o2 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-O2"
```

## CI/CD Integration

All benchmarks run automatically via GitHub Actions:

- **Trigger**: Push to `main`/`develop`, Pull Requests
- **Platform**: Ubuntu Latest (gcc/clang with AVX2)
- **Frequency**: Every commit
- **Artifacts**: Results stored for 30 days, baseline for 90 days

See [CI/CD Documentation](../../.github/CI_CD.md) for details.

## Troubleshooting

### Build Failures

**CMake not found:**
```bash
sudo apt-get install cmake  # Ubuntu
brew install cmake          # macOS
choco install cmake         # Windows
```

**Benchmark executable not created:**
- Check CMakeLists.txt has benchmark targets
- Verify `add_executable()` and `target_link_libraries()`

### Unexpected Results

**Huge variance between runs:**
- System load varies - run with `nice -n -20 ./benchmark`
- Disable frequency scaling (Linux): `sudo cpupower frequency-set -g performance`
- Close other applications

**Regression detected that shouldn't be:**
- Profile with perf/Instruments
- Check if compiler changed (Docker/CI)
- Verify no environment differences

## References

- [Google Benchmark Documentation](https://github.com/google/benchmark)
- [Performance Analysis Best Practices](https://easyperf.net/blog/)
- [Linux perf profiling](https://www.brendangregg.com/perf.html)
- [Optimization techniques](https://01.org/blogs)

## Future Enhancements

- [ ] GPU benchmarks (CUDA for Monte Carlo)
- [ ] Memory profiling (valgrind integration)
- [ ] Historical tracking dashboard
- [ ] Automated performance bisection (find regression commit)
- [ ] Comparative benchmarks (vs other libraries)
- [ ] Energy profiling (power consumption)
