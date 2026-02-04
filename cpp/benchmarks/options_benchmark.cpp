#include "../include/options_pricing.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

using namespace arbor::options;

// Benchmark utilities
struct BenchmarkStats {
    double mean_ns;
    double std_dev_ns;
    double min_ns;
    double max_ns;
    double p50_ns;
    double p95_ns;
    double p99_ns;
    int num_samples;
};

BenchmarkStats compute_stats(std::vector<int64_t>& times) {
    std::sort(times.begin(), times.end());
    
    const int n = times.size();
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double mean = sum / n;
    
    double sq_sum = 0.0;
    for (auto t : times) {
        sq_sum += (t - mean) * (t - mean);
    }
    double std_dev = std::sqrt(sq_sum / n);
    
    return BenchmarkStats{
        mean,
        std_dev,
        static_cast<double>(times.front()),
        static_cast<double>(times.back()),
        static_cast<double>(times[n / 2]),
        static_cast<double>(times[static_cast<int>(n * 0.95)]),
        static_cast<double>(times[static_cast<int>(n * 0.99)]),
        n
    };
}

void print_stats(const std::string& name, const BenchmarkStats& stats) {
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "  " << std::left << std::setw(30) << name << " | "
              << std::setw(8) << stats.mean_ns << " ns (mean) | "
              << std::setw(8) << stats.p50_ns << " ns (p50) | "
              << std::setw(8) << stats.p99_ns << " ns (p99) | "
              << std::setw(10) << (1e9 / stats.mean_ns) << " ops/sec\n";
}

// ============================================================================
// BLACK-SCHOLES BENCHMARK
// ============================================================================

void benchmark_black_scholes() {
    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "                BLACK-SCHOLES BENCHMARK                      \n";
    std::cout << "============================================================\n\n";
    
    const int NUM_WARMUP = 10000;
    const int NUM_ITERATIONS = 100000;
    
    // Parameters
    const double S = 100.0, K = 100.0, T = 0.25, r = 0.05, sigma = 0.25;
    
    // Warmup
    for (int i = 0; i < NUM_WARMUP; ++i) {
        volatile auto result = BlackScholesPricer::price(S, K, T, r, sigma, OptionType::CALL);
    }
    
    // Single option pricing
    std::vector<int64_t> single_times(NUM_ITERATIONS);
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        auto start = std::chrono::steady_clock::now();
        auto result = BlackScholesPricer::price(S, K, T, r, sigma, OptionType::CALL);
        auto end = std::chrono::steady_clock::now();
        single_times[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    
    auto single_stats = compute_stats(single_times);
    print_stats("Single Option (Call)", single_stats);
    
    // Put pricing
    std::vector<int64_t> put_times(NUM_ITERATIONS);
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        auto start = std::chrono::steady_clock::now();
        auto result = BlackScholesPricer::price(S, K, T, r, sigma, OptionType::PUT);
        auto end = std::chrono::steady_clock::now();
        put_times[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    
    auto put_stats = compute_stats(put_times);
    print_stats("Single Option (Put)", put_stats);
    
    // Full Greeks calculation
    std::vector<int64_t> greeks_times(NUM_ITERATIONS);
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        auto start = std::chrono::steady_clock::now();
        auto greeks = BlackScholesPricer::compute_all_greeks(S, K, T, r, sigma, OptionType::CALL);
        auto end = std::chrono::steady_clock::now();
        greeks_times[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    
    auto greeks_stats = compute_stats(greeks_times);
    print_stats("Full Greeks (9 sensitivities)", greeks_stats);
    
    // Option chain (17 strikes)
    std::vector<double> strikes;
    for (double k = 80; k <= 120; k += 2.5) strikes.push_back(k);
    
    const int CHAIN_ITERATIONS = 10000;
    std::vector<int64_t> chain_times(CHAIN_ITERATIONS);
    for (int i = 0; i < CHAIN_ITERATIONS; ++i) {
        auto start = std::chrono::steady_clock::now();
        auto chain = BlackScholesPricer::option_chain(S, strikes, T, r, sigma);
        auto end = std::chrono::steady_clock::now();
        chain_times[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    
    auto chain_stats = compute_stats(chain_times);
    print_stats("Option Chain (17 strikes, C+P)", chain_stats);
    
    // Implied volatility
    auto market_result = BlackScholesPricer::price(S, K, T, r, sigma, OptionType::CALL);
    const int IV_ITERATIONS = 50000;
    std::vector<int64_t> iv_times(IV_ITERATIONS);
    for (int i = 0; i < IV_ITERATIONS; ++i) {
        auto start = std::chrono::steady_clock::now();
        auto iv = BlackScholesPricer::implied_volatility(market_result.price, S, K, T, r, OptionType::CALL);
        auto end = std::chrono::steady_clock::now();
        iv_times[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    
    auto iv_stats = compute_stats(iv_times);
    print_stats("Implied Volatility (Newton)", iv_stats);
    
    std::cout << "\n  Sample Result: Call Price = $" << std::setprecision(4) << market_result.price
              << ", Delta = " << market_result.greeks.delta
              << ", Gamma = " << market_result.greeks.gamma << "\n";
}

// ============================================================================
// MERTON JUMP-DIFFUSION BENCHMARK
// ============================================================================

void benchmark_jump_diffusion() {
    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "            MERTON JUMP-DIFFUSION BENCHMARK                  \n";
    std::cout << "============================================================\n\n";
    
    const int NUM_ITERATIONS = 10000;
    
    const double S = 100.0, K = 100.0, T = 1.0, r = 0.05, sigma = 0.20;
    JumpDiffusionParams jp{0.5, -0.1, 0.2};
    
    // Warmup
    for (int i = 0; i < 1000; ++i) {
        volatile auto result = MertonJumpDiffusion::price(S, K, T, r, sigma, jp, OptionType::CALL);
    }
    
    // Series solution (default 50 terms)
    std::vector<int64_t> series_times(NUM_ITERATIONS);
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        auto start = std::chrono::steady_clock::now();
        auto result = MertonJumpDiffusion::price(S, K, T, r, sigma, jp, OptionType::CALL, 50);
        auto end = std::chrono::steady_clock::now();
        series_times[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    
    auto series_stats = compute_stats(series_times);
    print_stats("Analytical (50 terms)", series_stats);
    
    // High precision (100 terms)
    std::vector<int64_t> hp_times(NUM_ITERATIONS);
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        auto start = std::chrono::steady_clock::now();
        auto result = MertonJumpDiffusion::price(S, K, T, r, sigma, jp, OptionType::CALL, 100);
        auto end = std::chrono::steady_clock::now();
        hp_times[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    
    auto hp_stats = compute_stats(hp_times);
    print_stats("Analytical (100 terms)", hp_stats);
    
    // Greeks (finite difference)
    const int GREEKS_ITERATIONS = 5000;
    std::vector<int64_t> greeks_times(GREEKS_ITERATIONS);
    for (int i = 0; i < GREEKS_ITERATIONS; ++i) {
        auto start = std::chrono::steady_clock::now();
        auto greeks = MertonJumpDiffusion::compute_greeks(S, K, T, r, sigma, jp, OptionType::CALL);
        auto end = std::chrono::steady_clock::now();
        greeks_times[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    
    auto greeks_stats = compute_stats(greeks_times);
    print_stats("Greeks (finite diff)", greeks_stats);
    
    auto result = MertonJumpDiffusion::price(S, K, T, r, sigma, jp, OptionType::CALL);
    auto bs_result = BlackScholesPricer::price(S, K, T, r, sigma, OptionType::CALL);
    
    std::cout << "\n  Jump-Diffusion Price: $" << std::setprecision(4) << result.price
              << " vs BS Price: $" << bs_result.price
              << " (diff: $" << (result.price - bs_result.price) << ")\n";
}

// ============================================================================
// HESTON STOCHASTIC VOLATILITY BENCHMARK
// ============================================================================

void benchmark_heston() {
    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "         HESTON STOCHASTIC VOLATILITY BENCHMARK              \n";
    std::cout << "============================================================\n\n";
    
    const int NUM_ITERATIONS = 5000;
    
    const double S = 100.0, K = 100.0, T = 1.0, r = 0.05;
    HestonParams params{0.04, 2.0, 0.04, 0.3, -0.7};
    
    // Warmup
    for (int i = 0; i < 500; ++i) {
        volatile auto result = HestonModel::price(S, K, T, r, params, OptionType::CALL);
    }
    
    // Quadrature pricing
    std::vector<int64_t> quad_times(NUM_ITERATIONS);
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        auto start = std::chrono::steady_clock::now();
        auto result = HestonModel::price_quadrature(S, K, T, r, params, OptionType::CALL, 1000);
        auto end = std::chrono::steady_clock::now();
        quad_times[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    
    auto quad_stats = compute_stats(quad_times);
    print_stats("Quadrature (1000 pts)", quad_stats);
    
    // Greeks
    const int GREEKS_ITERATIONS = 1000;
    std::vector<int64_t> greeks_times(GREEKS_ITERATIONS);
    for (int i = 0; i < GREEKS_ITERATIONS; ++i) {
        auto start = std::chrono::steady_clock::now();
        auto greeks = HestonModel::compute_greeks(S, K, T, r, params, OptionType::CALL);
        auto end = std::chrono::steady_clock::now();
        greeks_times[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    
    auto greeks_stats = compute_stats(greeks_times);
    print_stats("Greeks (finite diff)", greeks_stats);
    
    // Characteristic function evaluation
    const int CF_ITERATIONS = 50000;
    std::vector<int64_t> cf_times(CF_ITERATIONS);
    for (int i = 0; i < CF_ITERATIONS; ++i) {
        std::complex<double> u(1.0, 0.5);
        auto start = std::chrono::steady_clock::now();
        auto cf = HestonModel::characteristic_function(u, S, T, r, params);
        auto end = std::chrono::steady_clock::now();
        cf_times[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    
    auto cf_stats = compute_stats(cf_times);
    print_stats("Characteristic Func", cf_stats);
    
    auto result = HestonModel::price(S, K, T, r, params, OptionType::CALL);
    std::cout << "\n  Heston Price: $" << std::setprecision(4) << result.price
              << ", Feller satisfied: " << (params.satisfies_feller() ? "Yes" : "No") << "\n";
}

// ============================================================================
// SABR MODEL BENCHMARK
// ============================================================================

void benchmark_sabr() {
    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "                    SABR MODEL BENCHMARK                      \n";
    std::cout << "============================================================\n\n";
    
    const int NUM_ITERATIONS = 100000;
    
    const double F = 100.0, T = 1.0;
    SABRParams params{0.3, 0.5, -0.3, 0.4};
    
    // Implied vol calculation
    std::vector<int64_t> iv_times(NUM_ITERATIONS);
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        auto start = std::chrono::steady_clock::now();
        auto vol = SABRModel::implied_volatility(F, F * 1.1, T, params);
        auto end = std::chrono::steady_clock::now();
        iv_times[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    
    auto iv_stats = compute_stats(iv_times);
    print_stats("Implied Vol (Hagan)", iv_stats);
    
    // Full pricing
    const int PRICE_ITERATIONS = 50000;
    std::vector<int64_t> price_times(PRICE_ITERATIONS);
    for (int i = 0; i < PRICE_ITERATIONS; ++i) {
        auto start = std::chrono::steady_clock::now();
        auto result = SABRModel::price(F, F * 1.1, T, 0.05, params, OptionType::CALL);
        auto end = std::chrono::steady_clock::now();
        price_times[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    
    auto price_stats = compute_stats(price_times);
    print_stats("Full Pricing", price_stats);
    
    // Vol surface (multiple strikes)
    std::vector<double> strikes = {80, 85, 90, 95, 100, 105, 110, 115, 120};
    const int SURFACE_ITERATIONS = 10000;
    std::vector<int64_t> surface_times(SURFACE_ITERATIONS);
    for (int i = 0; i < SURFACE_ITERATIONS; ++i) {
        auto start = std::chrono::steady_clock::now();
        for (double K : strikes) {
            volatile auto vol = SABRModel::implied_volatility(F, K, T, params);
        }
        auto end = std::chrono::steady_clock::now();
        surface_times[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    
    auto surface_stats = compute_stats(surface_times);
    print_stats("Vol Smile (9 strikes)", surface_stats);
    
    double atm_vol = SABRModel::implied_volatility(F, F, T, params);
    double otm_vol = SABRModel::implied_volatility(F, F * 1.2, T, params);
    std::cout << "\n  ATM Vol: " << std::setprecision(2) << (atm_vol * 100) << "%"
              << ", 20% OTM Call Vol: " << (otm_vol * 100) << "%\n";
}

// ============================================================================
// AMERICAN OPTIONS BENCHMARK
// ============================================================================

void benchmark_american() {
    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "              AMERICAN OPTIONS BENCHMARK                      \n";
    std::cout << "============================================================\n\n";
    
    const double S = 100.0, K = 100.0, T = 1.0, r = 0.05, sigma = 0.25;
    
    // Binomial tree (100 steps)
    const int TREE_ITERATIONS = 5000;
    std::vector<int64_t> tree100_times(TREE_ITERATIONS);
    for (int i = 0; i < TREE_ITERATIONS; ++i) {
        auto start = std::chrono::steady_clock::now();
        auto result = AmericanOptionPricer::price_binomial(S, K, T, r, sigma, OptionType::PUT, {100, false, false});
        auto end = std::chrono::steady_clock::now();
        tree100_times[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    
    auto tree100_stats = compute_stats(tree100_times);
    print_stats("Binomial (100 steps)", tree100_stats);
    
    // Binomial with Richardson
    std::vector<int64_t> tree_rich_times(TREE_ITERATIONS);
    for (int i = 0; i < TREE_ITERATIONS; ++i) {
        auto start = std::chrono::steady_clock::now();
        auto result = AmericanOptionPricer::price_binomial(S, K, T, r, sigma, OptionType::PUT, {100, true, true});
        auto end = std::chrono::steady_clock::now();
        tree_rich_times[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    
    auto tree_rich_stats = compute_stats(tree_rich_times);
    print_stats("Binomial + Richardson", tree_rich_stats);
    
    // High precision (500 steps)
    const int HP_ITERATIONS = 1000;
    std::vector<int64_t> tree500_times(HP_ITERATIONS);
    for (int i = 0; i < HP_ITERATIONS; ++i) {
        auto start = std::chrono::steady_clock::now();
        auto result = AmericanOptionPricer::price_binomial(S, K, T, r, sigma, OptionType::PUT, {500, true, true});
        auto end = std::chrono::steady_clock::now();
        tree500_times[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    
    auto tree500_stats = compute_stats(tree500_times);
    print_stats("Binomial (500 steps)", tree500_stats);
    
    // Trinomial
    std::vector<int64_t> tri_times(TREE_ITERATIONS);
    for (int i = 0; i < TREE_ITERATIONS; ++i) {
        auto start = std::chrono::steady_clock::now();
        auto result = AmericanOptionPricer::price_trinomial(S, K, T, r, sigma, OptionType::PUT, 100);
        auto end = std::chrono::steady_clock::now();
        tri_times[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    
    auto tri_stats = compute_stats(tri_times);
    print_stats("Trinomial (100 steps)", tri_stats);
    
    auto american = AmericanOptionPricer::price_binomial(S, K, T, r, sigma, OptionType::PUT, {200, true, true});
    auto european = BlackScholesPricer::price(S, K, T, r, sigma, OptionType::PUT);
    double premium = american.price - european.price;
    
    std::cout << "\n  American Put: $" << std::setprecision(4) << american.price
              << ", European Put: $" << european.price
              << ", Early Exercise Premium: $" << premium << "\n";
}

// ============================================================================
// LONGSTAFF-SCHWARTZ LSM BENCHMARK
// ============================================================================

void benchmark_lsm() {
    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "          LONGSTAFF-SCHWARTZ LSM BENCHMARK                    \n";
    std::cout << "============================================================\n\n";
    
    const double S = 100.0, K = 100.0, T = 1.0, r = 0.05, sigma = 0.25;
    
    // 10k paths
    LSMConfig config_10k{10000, 50, 3, 42, true, true};
    const int LSM_ITERATIONS = 100;
    
    std::vector<int64_t> lsm_10k_times(LSM_ITERATIONS);
    for (int i = 0; i < LSM_ITERATIONS; ++i) {
        LSMConfig config{10000, 50, 3, static_cast<uint64_t>(42 + i), true, true};
        auto start = std::chrono::steady_clock::now();
        auto result = LongstaffSchwartzPricer::price(S, K, T, r, sigma, OptionType::PUT, config);
        auto end = std::chrono::steady_clock::now();
        lsm_10k_times[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    
    auto lsm_10k_stats = compute_stats(lsm_10k_times);
    std::cout << "  " << std::left << std::setw(30) << "LSM (10k paths)" << " | "
              << std::fixed << std::setprecision(2)
              << std::setw(8) << (lsm_10k_stats.mean_ns / 1e6) << " ms (mean) | "
              << std::setw(8) << (lsm_10k_stats.p99_ns / 1e6) << " ms (p99)\n";
    
    // 50k paths
    std::vector<int64_t> lsm_50k_times(LSM_ITERATIONS);
    for (int i = 0; i < LSM_ITERATIONS; ++i) {
        LSMConfig config{50000, 50, 3, static_cast<uint64_t>(42 + i), true, true};
        auto start = std::chrono::steady_clock::now();
        auto result = LongstaffSchwartzPricer::price(S, K, T, r, sigma, OptionType::PUT, config);
        auto end = std::chrono::steady_clock::now();
        lsm_50k_times[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    
    auto lsm_50k_stats = compute_stats(lsm_50k_times);
    std::cout << "  " << std::left << std::setw(30) << "LSM (50k paths)" << " | "
              << std::setw(8) << (lsm_50k_stats.mean_ns / 1e6) << " ms (mean) | "
              << std::setw(8) << (lsm_50k_stats.p99_ns / 1e6) << " ms (p99)\n";
    
    // Heston LSM
    HestonParams hp{0.04, 2.0, 0.04, 0.3, -0.7};
    const int HESTON_ITERATIONS = 50;
    
    std::vector<int64_t> heston_times(HESTON_ITERATIONS);
    for (int i = 0; i < HESTON_ITERATIONS; ++i) {
        LSMConfig config{20000, 50, 3, static_cast<uint64_t>(42 + i), true, false};
        auto start = std::chrono::steady_clock::now();
        auto result = LongstaffSchwartzPricer::price_heston(S, K, T, r, hp, OptionType::PUT, config);
        auto end = std::chrono::steady_clock::now();
        heston_times[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    
    auto heston_stats = compute_stats(heston_times);
    std::cout << "  " << std::left << std::setw(30) << "LSM Heston (20k paths)" << " | "
              << std::setw(8) << (heston_stats.mean_ns / 1e6) << " ms (mean) | "
              << std::setw(8) << (heston_stats.p99_ns / 1e6) << " ms (p99)\n";
    
    auto result = LongstaffSchwartzPricer::price(S, K, T, r, sigma, OptionType::PUT, 
                                                  {50000, 50, 3, 42, true, true});
    auto binomial = AmericanOptionPricer::price_binomial(S, K, T, r, sigma, OptionType::PUT);
    
    std::cout << "\n  LSM Price: $" << std::setprecision(4) << result.price
              << " (SE: " << result.std_error << ")"
              << " vs Binomial: $" << binomial.price << "\n";
}

// ============================================================================
// MONTE CARLO ENGINE BENCHMARK
// ============================================================================

void benchmark_monte_carlo() {
    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "              MONTE CARLO ENGINE BENCHMARK                    \n";
    std::cout << "============================================================\n\n";
    
    const double S = 100.0, K = 100.0, T = 1.0, r = 0.05, sigma = 0.25;
    
    // European MC (100k paths)
    const int MC_ITERATIONS = 100;
    
    std::vector<int64_t> mc_100k_times(MC_ITERATIONS);
    for (int i = 0; i < MC_ITERATIONS; ++i) {
        auto start = std::chrono::steady_clock::now();
        auto result = MonteCarloEngine::price_european(S, K, T, r, sigma, OptionType::CALL, 100000, 42 + i, true, true);
        auto end = std::chrono::steady_clock::now();
        mc_100k_times[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    
    auto mc_100k_stats = compute_stats(mc_100k_times);
    std::cout << "  " << std::left << std::setw(30) << "European (100k paths)" << " | "
              << std::fixed << std::setprecision(2)
              << std::setw(8) << (mc_100k_stats.mean_ns / 1e6) << " ms (mean) | "
              << std::setw(10) << (100000.0 / (mc_100k_stats.mean_ns / 1e9)) << " paths/sec\n";
    
    // European MC (1M paths)
    std::vector<int64_t> mc_1m_times(MC_ITERATIONS);
    for (int i = 0; i < MC_ITERATIONS; ++i) {
        auto start = std::chrono::steady_clock::now();
        auto result = MonteCarloEngine::price_european(S, K, T, r, sigma, OptionType::CALL, 1000000, 42 + i, true, true);
        auto end = std::chrono::steady_clock::now();
        mc_1m_times[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    
    auto mc_1m_stats = compute_stats(mc_1m_times);
    std::cout << "  " << std::left << std::setw(30) << "European (1M paths)" << " | "
              << std::setw(8) << (mc_1m_stats.mean_ns / 1e6) << " ms (mean) | "
              << std::setw(10) << (1000000.0 / (mc_1m_stats.mean_ns / 1e9)) << " paths/sec\n";
    
    // Heston MC
    HestonParams hp{0.04, 2.0, 0.04, 0.3, -0.7};
    
    std::vector<int64_t> heston_mc_times(MC_ITERATIONS);
    for (int i = 0; i < MC_ITERATIONS; ++i) {
        auto start = std::chrono::steady_clock::now();
        auto result = MonteCarloEngine::price_european_heston(S, K, T, r, hp, OptionType::CALL, 100000, 252, 42 + i);
        auto end = std::chrono::steady_clock::now();
        heston_mc_times[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    
    auto heston_mc_stats = compute_stats(heston_mc_times);
    std::cout << "  " << std::left << std::setw(30) << "Heston MC (100k paths)" << " | "
              << std::setw(8) << (heston_mc_stats.mean_ns / 1e6) << " ms (mean) | "
              << std::setw(10) << (100000.0 / (heston_mc_stats.mean_ns / 1e9)) << " paths/sec\n";
    
    // Jump-Diffusion MC
    JumpDiffusionParams jp{0.5, -0.1, 0.2};
    
    std::vector<int64_t> jd_mc_times(MC_ITERATIONS);
    for (int i = 0; i < MC_ITERATIONS; ++i) {
        auto start = std::chrono::steady_clock::now();
        auto result = MonteCarloEngine::price_jump_diffusion(S, K, T, r, sigma, jp, OptionType::CALL, 100000, 252, 42 + i);
        auto end = std::chrono::steady_clock::now();
        jd_mc_times[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    
    auto jd_mc_stats = compute_stats(jd_mc_times);
    std::cout << "  " << std::left << std::setw(30) << "Jump-Diffusion MC (100k)" << " | "
              << std::setw(8) << (jd_mc_stats.mean_ns / 1e6) << " ms (mean) | "
              << std::setw(10) << (100000.0 / (jd_mc_stats.mean_ns / 1e9)) << " paths/sec\n";
    
#ifdef __AVX2__
    // SIMD Monte Carlo
    std::vector<int64_t> simd_times(MC_ITERATIONS);
    for (int i = 0; i < MC_ITERATIONS; ++i) {
        auto start = std::chrono::steady_clock::now();
        auto result = SIMDMonteCarloEngine::price_european_avx2(S, K, T, r, sigma, OptionType::CALL, 1000000, 42 + i);
        auto end = std::chrono::steady_clock::now();
        simd_times[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    
    auto simd_stats = compute_stats(simd_times);
    std::cout << "  " << std::left << std::setw(30) << "SIMD AVX2 (1M paths)" << " | "
              << std::setw(8) << (simd_stats.mean_ns / 1e6) << " ms (mean) | "
              << std::setw(10) << (1000000.0 / (simd_stats.mean_ns / 1e9)) << " paths/sec\n";
#endif
}

// ============================================================================
// EXOTIC OPTIONS BENCHMARK
// ============================================================================

void benchmark_exotics() {
    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "               EXOTIC OPTIONS BENCHMARK                       \n";
    std::cout << "============================================================\n\n";
    
    const double S = 100.0, K = 100.0, T = 1.0, r = 0.05, sigma = 0.25;
    
    // Barrier options - analytical
    BarrierParams barrier{BarrierType::DOWN_AND_OUT, 80.0, 0.0};
    const int BARRIER_ITERATIONS = 50000;
    
    std::vector<int64_t> barrier_anal_times(BARRIER_ITERATIONS);
    for (int i = 0; i < BARRIER_ITERATIONS; ++i) {
        auto start = std::chrono::steady_clock::now();
        auto result = BarrierOptionPricer::price_analytical(S, K, T, r, sigma, barrier, OptionType::CALL);
        auto end = std::chrono::steady_clock::now();
        barrier_anal_times[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    
    auto barrier_anal_stats = compute_stats(barrier_anal_times);
    print_stats("Barrier (analytical)", barrier_anal_stats);
    
    // Barrier MC
    const int BARRIER_MC_ITERATIONS = 100;
    std::vector<int64_t> barrier_mc_times(BARRIER_MC_ITERATIONS);
    for (int i = 0; i < BARRIER_MC_ITERATIONS; ++i) {
        auto start = std::chrono::steady_clock::now();
        auto result = BarrierOptionPricer::price_monte_carlo(S, K, T, r, sigma, barrier, OptionType::CALL, 252, 100000, 42 + i);
        auto end = std::chrono::steady_clock::now();
        barrier_mc_times[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    
    auto barrier_mc_stats = compute_stats(barrier_mc_times);
    std::cout << "  " << std::left << std::setw(30) << "Barrier MC (100k paths)" << " | "
              << std::fixed << std::setprecision(2)
              << std::setw(8) << (barrier_mc_stats.mean_ns / 1e6) << " ms (mean)\n";
    
    // Asian options
    AsianParams asian_params{AveragingType::GEOMETRIC, 12, false};
    const int ASIAN_ITERATIONS = 50000;
    
    std::vector<int64_t> asian_geo_times(ASIAN_ITERATIONS);
    for (int i = 0; i < ASIAN_ITERATIONS; ++i) {
        auto start = std::chrono::steady_clock::now();
        auto result = AsianOptionPricer::price_geometric(S, K, T, r, sigma, asian_params, OptionType::CALL);
        auto end = std::chrono::steady_clock::now();
        asian_geo_times[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    
    auto asian_geo_stats = compute_stats(asian_geo_times);
    print_stats("Asian Geometric (analytical)", asian_geo_stats);
    
    // Asian arithmetic MC
    AsianParams arith_params{AveragingType::ARITHMETIC, 12, false};
    const int ASIAN_MC_ITERATIONS = 100;
    
    std::vector<int64_t> asian_arith_times(ASIAN_MC_ITERATIONS);
    for (int i = 0; i < ASIAN_MC_ITERATIONS; ++i) {
        auto start = std::chrono::steady_clock::now();
        auto result = AsianOptionPricer::price_arithmetic(S, K, T, r, sigma, arith_params, OptionType::CALL, 100000, 42 + i);
        auto end = std::chrono::steady_clock::now();
        asian_arith_times[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    
    auto asian_arith_stats = compute_stats(asian_arith_times);
    std::cout << "  " << std::left << std::setw(30) << "Asian Arithmetic MC (100k)" << " | "
              << std::setw(8) << (asian_arith_stats.mean_ns / 1e6) << " ms (mean)\n";
    
    // Turnbull-Wakeman
    std::vector<int64_t> tw_times(ASIAN_ITERATIONS);
    for (int i = 0; i < ASIAN_ITERATIONS; ++i) {
        auto start = std::chrono::steady_clock::now();
        auto result = AsianOptionPricer::price_turnbull_wakeman(S, K, T, r, sigma, arith_params, OptionType::CALL);
        auto end = std::chrono::steady_clock::now();
        tw_times[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    
    auto tw_stats = compute_stats(tw_times);
    print_stats("Turnbull-Wakeman approx", tw_stats);
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "============================================================\n";
    std::cout << "      ARBOR EXOTIC OPTIONS PRICING ENGINE - BENCHMARK        \n";
    std::cout << "============================================================\n";
    std::cout << "\n";
    std::cout << "Models implemented:\n";
    std::cout << "  - Black-Scholes with extended Greeks (9 sensitivities)\n";
    std::cout << "  - Merton Jump-Diffusion (analytical series)\n";
    std::cout << "  - Heston Stochastic Volatility (characteristic function)\n";
    std::cout << "  - SABR Vol Surface (Hagan approximation)\n";
    std::cout << "  - American Options (Binomial/Trinomial + Richardson)\n";
    std::cout << "  - Longstaff-Schwartz LSM (Laguerre basis)\n";
    std::cout << "  - Barrier Options (analytical + Monte Carlo)\n";
    std::cout << "  - Asian Options (geometric + arithmetic + Turnbull-Wakeman)\n";
#ifdef __AVX2__
    std::cout << "  - SIMD Monte Carlo (AVX2 vectorized)\n";
#endif
    std::cout << "\n";
    
    benchmark_black_scholes();
    benchmark_jump_diffusion();
    benchmark_heston();
    benchmark_sabr();
    benchmark_american();
    benchmark_lsm();
    benchmark_monte_carlo();
    benchmark_exotics();
    
    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "                   BENCHMARK COMPLETE                         \n";
    std::cout << "============================================================\n";
    
    return 0;
}
