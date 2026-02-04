# Benchmark Results

Performance benchmarks for the Arbor Trading Engine, measured on a 16-thread system using GCC 14.2.0 with optimization flags `-O3 -march=native -flto`.

## Order Book Engine

Tested with price-time priority matching and FIFO queue per price level.

### Latency Test (10,000 orders)

```
Order book depth: 1000 orders
Best bid: 15009 | Best ask: 15017
Spread: 8 ticks

LATENCY STATISTICS:
-------------------------------------------
Average matching latency:  0.49 us (488 ns)
Minimum matching latency:  0.10 us (100 ns)
Maximum matching latency:  255.30 us (255300 ns)
P99 matching latency:      2.60 us (2600 ns)
```

### Throughput Test (50,000 orders)

```
THROUGHPUT RESULTS:
-------------------------------------------
Total time:          30.74 ms
Orders per second:   1,626,699
Avg time per order:  614 ns
Final book depth:    50,000 orders
Total trades:        42,719 executions
```

### Industry Comparison

| System | Throughput |
|--------|------------|
| Arbor Engine | 1.77M orders/sec |
| NASDAQ ITCH Feed | ~500K messages/sec |
| CME Globex | ~1M messages/sec |

---

## Lock-Free Queues

Cache-line aligned (64 bytes) with proper memory ordering.

### SPSC (Single Producer, Single Consumer)

```
Messages:      10M
Duration:      914 ms
Throughput:    10.94 M msg/sec

Latency (ns):
  Min:           100
  Avg:           414,821
  P50:           488,600
  P99:           929,300
  Max:           1,081,600
```

**Guarantee**: Wait-free (bounded time per operation)

### MPSC (Multiple Producers, Single Consumer)

```
Messages (4 producers): 4.0M
Duration:               445 ms
Throughput:             8.99 M msg/sec

Latency (ns):
  Min:    100
  Avg:    5,872
  P50:    500
  P99:    192,300
  Max:    2,381,000
```

**Guarantee**: Lock-free for producers, wait-free for consumer

### MPMC (Multiple Producers, Multiple Consumers)

```
Configuration: 2P x 2C
Messages:      2.0M
Duration:      252 ms
Throughput:    7.94 M msg/sec
```

**Guarantee**: Lock-free (no mutexes)

---

## Options Pricing (Black-Scholes)

### Single Option Pricing

```
Parameters: S=$150, K=$150, T=0.25y, r=5%, σ=25%
Calculations: 100,000

RESULTS:
-------------------------------------------
Total time:                16,652 μs
Average time per pricing:  59 ns
Minimum time:              0 ns (below measurement resolution)
Maximum time:              56,000 ns
Throughput:                6,005,140 pricings/sec

Sample result:
  Call price: $8.3976
  Delta: 0.5645
  Gamma: 0.0210
  Theta: -$0.0509/day
  Vega: $0.2953/1%
```

### Option Chain (25 strikes)

```
Strike range: $120.00 - $180.00
Total time:   51 μs
Per strike:   2,044 ns (call + put)
Throughput:   489,237 strikes/sec
```

### Implied Volatility Solver

```
Market price: $8.3976
Expected IV:  25.00%

RESULTS:
-------------------------------------------
Solved IV:           25.00%
Error vs true σ:     3.79e-11
Average solve time:  < 1 μs
```

---

## Monte Carlo Simulation

Multi-threaded Geometric Brownian Motion with risk metrics.

### Standard Simulation (10K paths)

```
Parameters:
  Initial price: $150
  Expected return: 10%
  Volatility: 25%
  Time horizon: 1 year
  Steps per path: 252 (daily)
  Total simulations: 2,520,000

PERFORMANCE:
-------------------------------------------
Total time:              21.10 ms
Paths per second:        473,911
Simulations per second:  119,425,620
Time per path:           2.11 μs

STATISTICAL RESULTS:
-------------------------------------------
Mean final price:        $165.81
Std deviation:           $41.24
Sharpe ratio:            0.38
Skewness:                0.68
Kurtosis:                0.70

RISK METRICS:
-------------------------------------------
VaR 95%:                 $43.03
VaR 99%:                 $59.71
CVaR 95% (ES):           $52.88
CVaR 99% (ES):           $65.03
```

### Large Scale (100K paths)

```
Total data points: 25,200,000
Total time:        0.13 seconds
Paths per second:  750,076
```

### Multi-Threading Scalability

```
Threads | Time (ms) | Paths/sec | Speedup
--------|-----------|-----------|--------
      1 |     359.4 |   139,130 | 1.00x
      2 |     178.5 |   280,052 | 2.01x
      4 |     100.2 |   498,941 | 3.59x
     16 |      54.4 |   918,511 | 6.60x
```

---

## Reproducing Results

```bash
cd cpp/build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ..
cmake --build .

./orderbook_bench
./lockfree_bench
./options_bench
./montecarlo_bench
```

### Notes

- Results vary based on CPU architecture, frequency, and system load
- First runs may be slower due to cache warming
- Background processes can cause latency spikes
- Compile with `-march=native` for best performance on your specific CPU
