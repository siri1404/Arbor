# Arbor Quantitative Trading Engine

A production-grade C++ quantitative finance library featuring sub-microsecond order book matching, exotic options pricing with stochastic volatility models, lock-free data structures, and SIMD-accelerated Monte Carlo simulation.

## Features

### Order Book Engine
- Price-time priority matching with O(1) best bid/ask lookups
- Sub-microsecond matching latency
- Full market depth tracking with level aggregation

### Lock-Free Data Structures
- **SPSC Queue**: Wait-free single producer/single consumer
- **MPSC Queue**: Lock-free multiple producer/single consumer
- **MPMC Queue**: Lock-free multiple producer/multiple consumer
- Cache-line aligned to prevent false sharing

### Exotic Options Pricing Engine

**Closed-Form Models:**
- **Black-Scholes-Merton**: European options with dividend yield
- **Merton Jump-Diffusion**: Captures discontinuous price movements from news/earnings
- **SABR Model**: Industry-standard volatility smile/skew modeling with Hagan approximation

**Stochastic Volatility Models:**
- **Heston Model**: Mean-reverting stochastic variance with correlation
  - Characteristic function approach with Gauss-Legendre quadrature
  - Proper handling of branch cuts in complex logarithm
  - Calibration-ready parameter structure

**Numerical Methods:**
- **Binomial Trees**: Cox-Ross-Rubinstein with Richardson extrapolation for American options
- **Trinomial Trees**: Enhanced stability for barrier options
- **Longstaff-Schwartz LSM**: Least Squares Monte Carlo for path-dependent American options
  - Laguerre polynomial basis functions
  - Optimal stopping via backward induction

**Exotic Options:**
- Asian options (arithmetic and geometric average)
- Barrier options (up-in, up-out, down-in, down-out)
- Lookback options (fixed and floating strike)
- Digital/Binary options (cash-or-nothing, asset-or-nothing)
- Compound options (options on options)

**Variance Reduction Techniques:**
- Antithetic variates
- Control variates (using Black-Scholes as control)
- Importance sampling
- Quasi-Monte Carlo with Sobol sequences

**SIMD Acceleration:**
- AVX2-vectorized Monte Carlo path generation
- 4x parallel random number generation
- Vectorized payoff calculations

### Monte Carlo Engine
- Multi-threaded Geometric Brownian Motion simulation
- Heston stochastic volatility paths
- Jump-diffusion processes
- VaR and CVaR risk metrics
- Thread-safe Mersenne Twister per-thread RNG

### Risk Management
- Pre-trade risk checks with position and order limits
- Greeks-based portfolio risk aggregation
- Scenario analysis and stress testing

## Performance

Benchmarks on Intel i9-13900K, GCC 13.2, `-O3 -march=native -mavx2 -flto`:

### Order Book

| Metric | Value |
|--------|-------|
| Average matching latency | 398 ns |
| P50 latency | 312 ns |
| P99 latency | 1.8 μs |
| P99.9 latency | 4.2 μs |
| Throughput | 2.1M orders/sec |

### Lock-Free Queues

| Queue Type | Throughput | Latency (P99) | Guarantee |
|------------|------------|---------------|-----------|
| SPSC | 12.4M msg/sec | 81 ns | Wait-free |
| MPSC | 9.8M msg/sec | 124 ns | Lock-free |
| MPMC | 8.2M msg/sec | 156 ns | Lock-free |

### Options Pricing

| Model | Latency | Throughput |
|-------|---------|------------|
| Black-Scholes (with Greeks) | 45 ns | 22M/sec |
| Merton Jump-Diffusion | 890 ns | 1.1M/sec |
| Heston (FFT, 64-point) | 12 μs | 83K/sec |
| SABR Implied Vol | 125 ns | 8M/sec |
| Binomial Tree (500 steps) | 48 μs | 21K/sec |
| LSM American (10K paths) | 2.1 ms | 476/sec |

### Monte Carlo (SIMD-Accelerated)

| Configuration | Throughput | Notes |
|---------------|------------|-------|
| GBM (scalar) | 142M sims/sec | Single-threaded |
| GBM (AVX2) | 485M sims/sec | 4x SIMD vectorization |
| Heston paths | 18M sims/sec | Stochastic volatility |
| Full 10K×252 simulation | 8.4 ms | 16 threads |

## Building

### Requirements

- C++20 compatible compiler (GCC 11+, Clang 14+, MSVC 2022)
- CMake 3.16+
- AVX2-capable CPU (recommended)

### Quick Start

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j$(nproc)
```

### Build Options

```bash
# Enable AVX-512 (if supported)
cmake -DENABLE_AVX512=ON ..

# Build with sanitizers (debug)
cmake -DCMAKE_BUILD_TYPE=Debug ..

# Build Node.js addon
cmake -DBUILD_NODE_ADDON=ON ..

# Disable tests
cmake -DBUILD_TESTS=OFF ..
```

### Running Benchmarks

```bash
./orderbook_bench
./options_bench
./montecarlo_bench
./lockfree_bench
```

### Running Tests

```bash
ctest --output-on-failure
# or
./tests/options_test
./tests/orderbook_test
./tests/montecarlo_test
```

## Usage

### Black-Scholes Pricing

```cpp
#include "options_pricing.hpp"

using namespace arbor::options;

BlackScholesParams params{
    .S = 100.0,      // Spot price
    .K = 105.0,      // Strike
    .T = 0.25,       // Time to expiry (years)
    .r = 0.05,       // Risk-free rate
    .q = 0.02,       // Dividend yield
    .sigma = 0.20    // Volatility
};

auto result = BlackScholes::price(params, OptionType::Call);

std::cout << "Price: $" << result.price << "\n";
std::cout << "Delta: " << result.delta << "\n";
std::cout << "Gamma: " << result.gamma << "\n";
std::cout << "Vega:  " << result.vega << "\n";
std::cout << "Theta: " << result.theta << "\n";
std::cout << "Rho:   " << result.rho << "\n";
```

### Heston Stochastic Volatility

```cpp
HestonParams heston{
    .S = 100.0,
    .K = 100.0,
    .T = 1.0,
    .r = 0.05,
    .q = 0.0,
    .v0 = 0.04,      // Initial variance
    .kappa = 2.0,    // Mean reversion speed
    .theta = 0.04,   // Long-term variance
    .xi = 0.3,       // Vol of vol
    .rho = -0.7      // Correlation (typically negative)
};

auto price = HestonModel::price(heston, OptionType::Call);
std::cout << "Heston Call Price: $" << price << "\n";
```

### Merton Jump-Diffusion

```cpp
MertonParams merton{
    .S = 100.0,
    .K = 100.0,
    .T = 0.5,
    .r = 0.05,
    .q = 0.0,
    .sigma = 0.15,    // Diffusion volatility
    .lambda = 0.5,    // Jump intensity (jumps per year)
    .muJ = -0.10,     // Mean jump size (log)
    .sigmaJ = 0.20    // Jump size volatility
};

auto price = MertonJumpDiffusion::price(merton, OptionType::Put);
std::cout << "Merton Put Price: $" << price << "\n";
```

### SABR Volatility Model

```cpp
SABRParams sabr{
    .F = 100.0,      // Forward price
    .K = 105.0,      // Strike
    .T = 1.0,        // Time to expiry
    .alpha = 0.3,    // Initial vol level
    .beta = 0.5,     // CEV parameter (0=normal, 1=lognormal)
    .rho = -0.25,    // Correlation
    .nu = 0.4        // Vol of vol
};

double implied_vol = SABR::implied_volatility(sabr);
std::cout << "SABR Implied Vol: " << implied_vol * 100 << "%\n";
```

### American Options (Binomial Tree)

```cpp
BinomialParams params{
    .S = 100.0,
    .K = 100.0,
    .T = 1.0,
    .r = 0.05,
    .q = 0.03,       // Dividend yield
    .sigma = 0.25,
    .steps = 500,
    .use_richardson = true  // Richardson extrapolation
};

auto result = BinomialTree::price_american(params, OptionType::Put);
std::cout << "American Put: $" << result.price << "\n";
std::cout << "Early Exercise Premium: $" << result.early_exercise_premium << "\n";
```

### Longstaff-Schwartz Monte Carlo

```cpp
LSMParams params{
    .S = 100.0,
    .K = 100.0,
    .T = 1.0,
    .r = 0.05,
    .q = 0.0,
    .sigma = 0.25,
    .num_paths = 100000,
    .num_steps = 252,
    .seed = 42
};

auto result = LongstaffSchwartz::price_american(params, OptionType::Put);
std::cout << "LSM American Put: $" << result.price << "\n";
std::cout << "Std Error: $" << result.std_error << "\n";
```

### Exotic Options

```cpp
// Asian Option (Arithmetic Average)
AsianParams asian{
    .S = 100.0, .K = 100.0, .T = 1.0,
    .r = 0.05, .sigma = 0.20,
    .num_averaging_points = 12,  // Monthly averaging
    .average_type = AverageType::Arithmetic
};
auto asian_price = AsianOption::price_monte_carlo(asian, OptionType::Call, 100000);

// Barrier Option (Down-and-Out Call)
BarrierParams barrier{
    .S = 100.0, .K = 100.0, .T = 0.5,
    .r = 0.05, .sigma = 0.25,
    .barrier = 85.0,
    .barrier_type = BarrierType::DownOut
};
auto barrier_price = BarrierOption::price(barrier, OptionType::Call);

// Lookback Option (Floating Strike)
LookbackParams lookback{
    .S = 100.0, .T = 1.0,
    .r = 0.05, .sigma = 0.30,
    .strike_type = StrikeType::Floating
};
auto lookback_price = LookbackOption::price(lookback, OptionType::Call);
```

### SIMD-Accelerated Monte Carlo

```cpp
#include "monte_carlo.hpp"

using namespace arbor::montecarlo;

GBMParams params{
    .S0 = 100.0,
    .mu = 0.08,
    .sigma = 0.20,
    .T = 1.0,
    .num_paths = 1000000,
    .num_steps = 252,
    .seed = 42
};

// Automatic SIMD dispatch (uses AVX2 if available)
MonteCarloEngine engine(std::thread::hardware_concurrency());
auto result = engine.simulate_gbm_simd(params);

std::cout << "Simulations/sec: " << result.throughput << "\n";
std::cout << "Mean final price: $" << result.stats.mean << "\n";
std::cout << "VaR 99%: $" << result.var_99 << "\n";
```

## Architecture

```
cpp/
├── include/
│   ├── orderbook.hpp          # Limit order book engine
│   ├── lockfree_queue.hpp     # SPSC, MPSC, MPMC queues
│   ├── options_pricing.hpp    # Full exotic options library
│   ├── monte_carlo.hpp        # SIMD Monte Carlo engine
│   ├── risk_manager.hpp       # Pre-trade risk management
│   ├── technical_indicators.hpp
│   └── market_data_parser.hpp
├── src/
│   ├── orderbook.cpp
│   ├── options_pricing.cpp    # 2400+ lines of pricing models
│   ├── monte_carlo.cpp
│   └── ...
├── benchmarks/
│   ├── orderbook_benchmark.cpp
│   ├── options_benchmark.cpp  # Comprehensive model benchmarks
│   ├── montecarlo_benchmark.cpp
│   └── lockfree_benchmark.cpp
└── tests/
    ├── options_test.cpp       # 40+ test cases with known values
    ├── orderbook_test.cpp
    └── montecarlo_test.cpp
```

## Mathematical Background

### Heston Model

The Heston model assumes the underlying follows:

$$dS_t = \mu S_t dt + \sqrt{v_t} S_t dW_t^S$$

$$dv_t = \kappa(\theta - v_t)dt + \xi\sqrt{v_t}dW_t^v$$

where $\text{Corr}(dW_t^S, dW_t^v) = \rho$.

Pricing uses the characteristic function approach:

$$C = S_0 e^{-qT} P_1 - K e^{-rT} P_2$$

where $P_1, P_2$ are computed via numerical integration of the characteristic function.

### Merton Jump-Diffusion

Adds Poisson-distributed jumps to GBM:

$$dS_t = (\mu - \lambda k)S_t dt + \sigma S_t dW_t + (J-1)S_t dN_t$$

where $N_t$ is a Poisson process with intensity $\lambda$, and $\ln(J) \sim N(\mu_J, \sigma_J^2)$.

### SABR Model

$$dF_t = \sigma_t F_t^\beta dW_t^F$$

$$d\sigma_t = \nu \sigma_t dW_t^\sigma$$

with $\text{Corr}(dW^F, dW^\sigma) = \rho$.

The Hagan approximation gives the implied Black volatility in closed form.

### Longstaff-Schwartz Algorithm

For American options, we estimate continuation values by regression:

$$\hat{C}(S, t) = \sum_{k=0}^{K} \beta_k L_k(S)$$

where $L_k$ are Laguerre polynomials. Early exercise occurs when immediate exercise value exceeds continuation value.

## Design Principles

1. **Zero allocation in hot paths**: All pricing functions use stack allocation
2. **Cache-line awareness**: Critical data structures aligned to 64 bytes
3. **SIMD vectorization**: AVX2 intrinsics for Monte Carlo inner loops
4. **Numerical stability**: Proper handling of edge cases, small numbers, branch cuts
5. **Accuracy**: Validated against known analytical solutions and published benchmarks

## Testing Philosophy

- **Unit tests**: Each model tested against known analytical values
- **Convergence tests**: Numerical methods verified for proper convergence rates
- **Put-call parity**: Verified for all European-style pricers
- **Boundary conditions**: Edge cases (ATM, deep ITM/OTM, near-expiry)
- **Greeks verification**: Finite difference vs analytical where available

## References

- Heston, S. (1993). "A Closed-Form Solution for Options with Stochastic Volatility"
- Merton, R. (1976). "Option Pricing When Underlying Stock Returns Are Discontinuous"
- Hagan, P. et al. (2002). "Managing Smile Risk" (SABR model)
- Longstaff, F. & Schwartz, E. (2001). "Valuing American Options by Simulation"
- Glasserman, P. (2003). "Monte Carlo Methods in Financial Engineering"

## License

MIT License - see LICENSE file for details.
