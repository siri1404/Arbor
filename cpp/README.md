# Arbor Trading Engine

A high-performance C++ trading infrastructure library featuring sub-microsecond order book matching, lock-free data structures, options pricing, and Monte Carlo simulation.

## Features

- **Order Book Engine**: Price-time priority matching with O(1) best bid/ask lookups
- **Lock-Free Queues**: SPSC, MPSC, and MPMC implementations for inter-thread communication
- **Options Pricing**: Black-Scholes model with full Greeks calculation
- **Monte Carlo Engine**: Multi-threaded GBM simulation with VaR/CVaR risk metrics
- **Risk Management**: Pre-trade risk checks with position and order limits

## Performance

Benchmarks run on a 16-thread system with GCC 14.2.0 (`-O3 -march=native -flto`):

### Order Book

| Metric | Value |
|--------|-------|
| Average matching latency | 488 ns |
| P99 latency | 2.6 μs |
| Throughput | 1.77M orders/sec |

### Lock-Free Queues

| Queue Type | Throughput | Guarantee |
|------------|------------|-----------|
| SPSC | 10.9M msg/sec | Wait-free |
| MPSC | 9.0M msg/sec | Lock-free |
| MPMC | 7.9M msg/sec | Lock-free |

### Options Pricing

| Metric | Value |
|--------|-------|
| Black-Scholes pricing | 59 ns average |
| Throughput | 6M pricings/sec |

### Monte Carlo

| Metric | Value |
|--------|-------|
| Simulation throughput | 119M sims/sec |
| 10K paths (252 steps) | 21 ms |
| Multi-thread scaling | Near-linear |

## Building

### Requirements

- C++20 compatible compiler (GCC 11+, Clang 13+, MSVC 2022)
- CMake 3.16+
- Ninja (recommended) or Make

### Quick Start

```bash
mkdir build && cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

### Running Benchmarks

```bash
./orderbook_bench
./options_bench
./montecarlo_bench
./lockfree_bench
```

## Usage

### Order Book

```cpp
#include "orderbook.hpp"

using namespace arbor::orderbook;

LimitOrderBook book("AAPL", 1);  // symbol, tick size
std::vector<Trade> trades;

// Add orders
uint64_t order_id = book.add_order(Side::BUY, OrderType::LIMIT, 15000, 100, trades);

// Query market data
uint64_t best_bid = book.best_bid();
uint64_t best_ask = book.best_ask();
uint64_t spread = book.spread();

// Get latency statistics
const auto& stats = book.get_latency_stats();
std::cout << "P99 latency: " << stats.p99_ns() << " ns\n";
```

### Lock-Free Queues

```cpp
#include "lockfree_queue.hpp"

using namespace arbor::lockfree;

// Single producer, single consumer (wait-free)
SPSCQueue<Message, 8192> spsc_queue;
spsc_queue.push(msg);
auto result = spsc_queue.pop();  // returns std::optional

// Multiple producers, single consumer
MPSCQueue<Message, 8192> mpsc_queue;
mpsc_queue.push(msg);  // Thread-safe

// Multiple producers, multiple consumers
MPMCQueue<Message, 8192> mpmc_queue;
```

### Options Pricing

```cpp
#include "options_pricing.hpp"

using namespace arbor::options;

// Price a call option
auto result = BlackScholesPricer::price(
    150.0,   // spot
    150.0,   // strike
    0.25,    // time to expiry (years)
    0.05,    // risk-free rate
    0.25,    // volatility
    OptionType::CALL
);

std::cout << "Price: $" << result.price << "\n";
std::cout << "Delta: " << result.greeks.delta << "\n";
std::cout << "Gamma: " << result.greeks.gamma << "\n";

// Calculate implied volatility
double iv = BlackScholesPricer::implied_volatility(
    market_price, S, K, T, r, OptionType::CALL
);
```

### Monte Carlo

```cpp
#include "monte_carlo.hpp"

using namespace arbor::montecarlo;

SimulationParams params{
    .S0 = 150.0,
    .mu = 0.10,
    .sigma = 0.25,
    .T = 1.0,
    .dt = 1.0 / 252.0,
    .num_paths = 10000,
    .num_steps = 252,
    .seed = 42
};

MonteCarloEngine engine;  // Uses all available cores
auto result = engine.simulate_gbm(params);

std::cout << "Mean: $" << result.stats.mean_final_price << "\n";
std::cout << "VaR 99%: $" << result.var.var_99 << "\n";
std::cout << "CVaR 99%: $" << result.var.cvar_99 << "\n";
```

## Architecture

```
cpp/
├── include/
│   ├── orderbook.hpp       # Limit order book with price-time priority
│   ├── lockfree_queue.hpp  # SPSC, MPSC, MPMC queues
│   ├── options_pricing.hpp # Black-Scholes with Greeks
│   ├── monte_carlo.hpp     # GBM simulation engine
│   └── risk_manager.hpp    # Pre-trade risk checks
├── src/
│   └── orderbook.cpp       # Order book implementation
├── benchmarks/
│   ├── orderbook_benchmark.cpp
│   ├── lockfree_benchmark.cpp
│   ├── options_benchmark.cpp
│   └── montecarlo_benchmark.cpp
└── tests/
    └── *_test.cpp          # Unit tests (Google Test)
```

## Design Principles

1. **Zero allocation in hot paths**: Pre-allocated memory pools avoid heap allocations during order processing
2. **Cache-line awareness**: Data structures aligned to 64 bytes to prevent false sharing
3. **Lock-free where possible**: Atomic operations with proper memory ordering
4. **Minimal branching**: Predictable code paths for better CPU pipeline utilization

## License

MIT License - see LICENSE file for details.
