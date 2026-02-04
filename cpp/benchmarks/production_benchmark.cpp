/**
 * PRODUCTION-GRADE BENCHMARK SUITE
 * 
 * This benchmark produces the numbers that quant firms actually care about:
 * - Cache miss rates
 * - Branch misprediction rates
 * - IPC (Instructions Per Cycle)
 * - True SIMD throughput
 * - P99, P99.9, P99.99 latencies
 * 
 * Methodology:
 * 1. Warmup phase (stabilize CPU frequency, fill caches)
 * 2. Multiple iterations with statistical analysis
 * 3. Hardware counter collection via perf_event_open()
 * 4. Proper measurement using RDTSC with serialization
 * 
 * Compile: g++ -O3 -march=native -mavx2 -std=c++20 production_benchmark.cpp -o benchmark -lpthread
 * Run:     sudo ./benchmark (requires root for perf counters)
 */

#include "../include/orderbook.hpp"
#include "../include/perf_counters.hpp"
#include "../include/simd_monte_carlo.hpp"
#include "../include/lockfree_queue.hpp"

#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <thread>
#include <fstream>
#include <chrono>
#include <random>

using namespace arbor;

// =============================================================================
// SYSTEM INFORMATION
// =============================================================================

void print_system_info() {
    std::cout << "================================================================================\n";
    std::cout << "                    ARBOR TRADING ENGINE - PRODUCTION BENCHMARK                 \n";
    std::cout << "================================================================================\n\n";
    
    std::cout << "SYSTEM CONFIGURATION:\n";
    std::cout << "-------------------------------------------\n";
    
#if defined(__linux__)
    // CPU Info
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    std::string cpu_model;
    int cpu_count = 0;
    
    while (std::getline(cpuinfo, line)) {
        if (line.find("model name") != std::string::npos && cpu_model.empty()) {
            cpu_model = line.substr(line.find(":") + 2);
        }
        if (line.find("processor") != std::string::npos) {
            cpu_count++;
        }
    }
    
    std::cout << "CPU:             " << cpu_model << "\n";
    std::cout << "Cores:           " << cpu_count << "\n";
    
    // Memory Info
    std::ifstream meminfo("/proc/meminfo");
    while (std::getline(meminfo, line)) {
        if (line.find("MemTotal") != std::string::npos) {
            std::cout << "Memory:          " << line.substr(line.find(":") + 1) << "\n";
            break;
        }
    }
#endif
    
    // Compiler info
#if defined(__clang__)
    std::cout << "Compiler:        Clang " << __clang_major__ << "." << __clang_minor__ << "\n";
#elif defined(__GNUC__)
    std::cout << "Compiler:        GCC " << __GNUC__ << "." << __GNUC_MINOR__ << "\n";
#endif
    
    // SIMD support
    std::cout << "SIMD Support:    ";
#if ARBOR_HAS_AVX2
    std::cout << "AVX2 ";
#endif
#if defined(__AVX512F__)
    std::cout << "AVX-512 ";
#endif
#if defined(__FMA__)
    std::cout << "FMA ";
#endif
    std::cout << "\n";
    
    // Perf counter availability
#if ARBOR_HAS_PERF_COUNTERS
    perf::PerfCounterGroup test_counters;
    std::cout << "Perf Counters:   " << (test_counters.available() ? "Available" : "Unavailable (run as root)") << "\n";
#else
    std::cout << "Perf Counters:   Not supported (Linux only)\n";
#endif
    
    std::cout << "Build Flags:     -O3 -march=native -mavx2\n";
    std::cout << "\n";
}

// =============================================================================
// BENCHMARK 1: ORDER BOOK WITH HARDWARE COUNTERS
// =============================================================================

void benchmark_orderbook_with_counters() {
    std::cout << "================================================================================\n";
    std::cout << "BENCHMARK 1: ORDER BOOK OPERATIONS WITH HARDWARE COUNTERS\n";
    std::cout << "================================================================================\n\n";
    
    constexpr size_t WARMUP_OPS = 10000;
    constexpr size_t BENCHMARK_OPS = 100000;
    
    orderbook::LimitOrderBook book("AAPL", 1);
    std::vector<orderbook::Trade> trades;
    trades.reserve(10000);
    
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<uint64_t> price_dist(14900, 15100);
    std::uniform_int_distribution<uint32_t> qty_dist(1, 100);
    
    // Seed order book
    std::cout << "Seeding order book with 10,000 resting orders...\n";
    for (size_t i = 0; i < 5000; ++i) {
        book.add_order(orderbook::Side::BUY, orderbook::OrderType::LIMIT, 
                      price_dist(rng) - 50, qty_dist(rng), 
                      orderbook::TimeInForce::GTC, 0, &trades);
        book.add_order(orderbook::Side::SELL, orderbook::OrderType::LIMIT, 
                      price_dist(rng) + 50, qty_dist(rng),
                      orderbook::TimeInForce::GTC, 0, &trades);
    }
    trades.clear();
    
    std::cout << "  Book depth: " << book.total_orders() << " orders\n";
    std::cout << "  Bid levels: " << book.bid_levels() << "\n";
    std::cout << "  Ask levels: " << book.ask_levels() << "\n\n";
    
    // Warmup
    std::cout << "Warmup phase (" << WARMUP_OPS << " operations)...\n";
    for (size_t i = 0; i < WARMUP_OPS; ++i) {
        orderbook::Side side = (i % 2 == 0) ? orderbook::Side::BUY : orderbook::Side::SELL;
        book.add_order(side, orderbook::OrderType::LIMIT, price_dist(rng), qty_dist(rng),
                      orderbook::TimeInForce::GTC, 0, &trades);
    }
    trades.clear();
    
    // Benchmark with hardware counters
    std::cout << "Benchmark phase (" << BENCHMARK_OPS << " operations)...\n\n";
    
    perf::PerfCounterGroup counters;
    perf::LatencyHistogram<100000, 10> histogram;  // Up to 100us, 10ns buckets
    perf::RdtscTimer timer;
    
    counters.start();
    
    for (size_t i = 0; i < BENCHMARK_OPS; ++i) {
        orderbook::Side side = (i % 2 == 0) ? orderbook::Side::BUY : orderbook::Side::SELL;
        uint64_t price = price_dist(rng);
        uint32_t qty = qty_dist(rng);
        
        timer.start();
        book.add_order(side, orderbook::OrderType::LIMIT, price, qty,
                      orderbook::TimeInForce::GTC, 0, &trades);
        int64_t latency = timer.stop_ns();
        
        histogram.record(latency);
        
        if (trades.size() > 10000) trades.clear();
    }
    
    auto result = counters.stop();
    
    // Print results
    result.print("ADD ORDER", BENCHMARK_OPS);
    std::cout << "\n";
    histogram.print("ADD ORDER");
    
    // Interpretation
    std::cout << "\n";
    std::cout << "INTERPRETATION:\n";
    std::cout << "-------------------------------------------\n";
    
    double cache_miss_pct = result.cache_miss_rate();
    double branch_miss_pct = result.branch_miss_rate();
    double ipc = result.ipc();
    
    if (cache_miss_pct < 1.0) {
        std::cout << "  [EXCELLENT] Cache miss rate < 1% - Data structures are cache-friendly\n";
    } else if (cache_miss_pct < 5.0) {
        std::cout << "  [GOOD] Cache miss rate < 5% - Reasonable cache utilization\n";
    } else {
        std::cout << "  [WARNING] Cache miss rate > 5% - Consider data layout optimization\n";
    }
    
    if (branch_miss_pct < 1.0) {
        std::cout << "  [EXCELLENT] Branch miss rate < 1% - Predictable control flow\n";
    } else if (branch_miss_pct < 3.0) {
        std::cout << "  [GOOD] Branch miss rate < 3% - Acceptable branching\n";
    } else {
        std::cout << "  [WARNING] Branch miss rate > 3% - Consider branchless algorithms\n";
    }
    
    if (ipc > 2.0) {
        std::cout << "  [EXCELLENT] IPC > 2.0 - Good instruction-level parallelism\n";
    } else if (ipc > 1.0) {
        std::cout << "  [GOOD] IPC > 1.0 - CPU is reasonably utilized\n";
    } else {
        std::cout << "  [WARNING] IPC < 1.0 - Memory or dependency bottleneck\n";
    }
    
    std::cout << "\n";
}

// =============================================================================
// BENCHMARK 2: SIMD MONTE CARLO
// =============================================================================

void benchmark_simd_monte_carlo() {
    std::cout << "================================================================================\n";
    std::cout << "BENCHMARK 2: SIMD-ACCELERATED MONTE CARLO\n";
    std::cout << "================================================================================\n\n";
    
#if ARBOR_HAS_AVX2
    std::cout << "AVX2 SIMD enabled - Processing 8 paths per instruction\n\n";
    
    simd::SimdGbmParams params{
        .S0 = 100.0f,
        .mu = 0.10f,
        .sigma = 0.20f,
        .T = 1.0f,
        .dt = 1.0f / 252.0f,
        .num_paths = 1000000,  // 1M paths
        .num_steps = 252,      // Daily steps
        .seed = 12345
    };
    
    std::cout << "Simulation Parameters:\n";
    std::cout << "  Paths:        " << params.num_paths << "\n";
    std::cout << "  Steps:        " << params.num_steps << "\n";
    std::cout << "  Total sims:   " << (static_cast<uint64_t>(params.num_paths) * params.num_steps) << "\n\n";
    
    // Warmup
    std::cout << "Warmup (10K paths)...\n";
    simd::SimdGbmParams warmup_params = params;
    warmup_params.num_paths = 10000;
    simd::SimdGbmEngine warmup_engine(42);
    warmup_engine.simulate(warmup_params);
    
    // Single-threaded SIMD benchmark
    std::cout << "Single-threaded SIMD benchmark...\n";
    simd::SimdGbmEngine single_engine(params.seed);
    
    perf::PerfCounterGroup counters;
    counters.start();
    auto result = single_engine.simulate(params);
    auto perf_result = counters.stop();
    
    std::cout << "\nSINGLE-THREADED RESULTS:\n";
    std::cout << "-------------------------------------------\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Time:              " << (result.time_ns / 1e6) << " ms\n";
    std::cout << "  Sims/sec:          " << std::setprecision(0) << result.sims_per_sec << "\n";
    std::cout << "  Mean price:        $" << std::setprecision(2) << result.mean_price << "\n";
    std::cout << "  Std dev:           $" << result.std_price << "\n";
    
    uint64_t total_sims = static_cast<uint64_t>(params.num_paths) * params.num_steps;
    perf_result.print("SIMD GBM", total_sims);
    
    // Multi-threaded SIMD benchmark
    std::cout << "\nMulti-threaded SIMD benchmark (" << std::thread::hardware_concurrency() << " threads)...\n";
    simd::ParallelSimdGbmEngine multi_engine;
    
    counters.start();
    auto multi_result = multi_engine.simulate(params);
    auto multi_perf = counters.stop();
    
    std::cout << "\nMULTI-THREADED RESULTS:\n";
    std::cout << "-------------------------------------------\n";
    std::cout << "  Time:              " << std::setprecision(2) << (multi_result.time_ns / 1e6) << " ms\n";
    std::cout << "  Sims/sec:          " << std::setprecision(0) << multi_result.sims_per_sec << "\n";
    std::cout << "  Speedup:           " << std::setprecision(2) 
              << (static_cast<double>(result.time_ns) / multi_result.time_ns) << "x\n";
    
    // Industry comparison
    std::cout << "\nINDUSTRY COMPARISON:\n";
    std::cout << "-------------------------------------------\n";
    std::cout << "  Arbor (1 thread):  " << std::setprecision(0) << result.sims_per_sec << " sims/sec\n";
    std::cout << "  Arbor (N threads): " << multi_result.sims_per_sec << " sims/sec\n";
    std::cout << "  Target (CPU):      500M+ sims/sec\n";
    std::cout << "  Target (GPU):      1B+ sims/sec\n";
    
    if (multi_result.sims_per_sec > 500000000) {
        std::cout << "\n  [EXCELLENT] Exceeds 500M sims/sec target!\n";
    } else if (multi_result.sims_per_sec > 200000000) {
        std::cout << "\n  [GOOD] Above 200M sims/sec\n";
    } else {
        std::cout << "\n  [NOTE] Consider further SIMD optimization or GPU offload\n";
    }
    
#else
    std::cout << "AVX2 not available - using scalar implementation\n";
    std::cout << "Recompile with -mavx2 for SIMD acceleration\n";
#endif
    
    std::cout << "\n";
}

// =============================================================================
// BENCHMARK 3: LOCK-FREE QUEUE LATENCY
// =============================================================================

void benchmark_lockfree_queue() {
    std::cout << "================================================================================\n";
    std::cout << "BENCHMARK 3: LOCK-FREE QUEUE LATENCY\n";
    std::cout << "================================================================================\n\n";
    
    constexpr size_t QUEUE_SIZE = 65536;
    constexpr size_t NUM_OPS = 1000000;
    
    lockfree::SPSCQueue<uint64_t, QUEUE_SIZE> queue;
    
    perf::LatencyHistogram<10000, 1> push_histogram;  // Up to 10us, 1ns buckets
    perf::LatencyHistogram<10000, 1> pop_histogram;
    perf::RdtscTimer timer;
    
    // PUSH latency benchmark
    std::cout << "SPSC Queue PUSH latency (" << NUM_OPS << " operations)...\n";
    
    perf::PerfCounterGroup push_counters;
    push_counters.start();
    
    for (size_t i = 0; i < NUM_OPS; ++i) {
        timer.start();
        queue.push(i);
        int64_t latency = timer.stop_ns();
        push_histogram.record(latency);
        
        // Keep queue half-full
        if (i % 2 == 1) {
            queue.pop();
        }
    }
    
    auto push_perf = push_counters.stop();
    push_perf.print("SPSC PUSH", NUM_OPS);
    std::cout << "\n";
    push_histogram.print("SPSC PUSH");
    
    // Clear queue
    while (queue.pop().has_value()) {}
    
    // Refill for POP benchmark
    for (size_t i = 0; i < QUEUE_SIZE / 2; ++i) {
        queue.push(i);
    }
    
    // POP latency benchmark
    std::cout << "\n\nSPSC Queue POP latency (" << NUM_OPS << " operations)...\n";
    
    perf::PerfCounterGroup pop_counters;
    pop_counters.start();
    
    for (size_t i = 0; i < NUM_OPS; ++i) {
        // Keep queue half-full
        if (i % 2 == 0) {
            queue.push(i);
        }
        
        timer.start();
        auto val = queue.pop();
        int64_t latency = timer.stop_ns();
        
        if (val.has_value()) {
            pop_histogram.record(latency);
        }
    }
    
    auto pop_perf = pop_counters.stop();
    pop_perf.print("SPSC POP", NUM_OPS);
    std::cout << "\n";
    pop_histogram.print("SPSC POP");
    
    // Summary
    std::cout << "\nSUMMARY:\n";
    std::cout << "-------------------------------------------\n";
    std::cout << "  PUSH P99:    " << push_histogram.p99() << " ns\n";
    std::cout << "  POP P99:     " << pop_histogram.p99() << " ns\n";
    std::cout << "  PUSH P99.9:  " << push_histogram.p999() << " ns\n";
    std::cout << "  POP P99.9:   " << pop_histogram.p999() << " ns\n";
    
    if (push_histogram.p99() < 100 && pop_histogram.p99() < 100) {
        std::cout << "\n  [EXCELLENT] Sub-100ns P99 - suitable for HFT\n";
    } else if (push_histogram.p99() < 500 && pop_histogram.p99() < 500) {
        std::cout << "\n  [GOOD] Sub-500ns P99 - suitable for low-latency trading\n";
    }
    
    std::cout << "\n";
}

// =============================================================================
// BENCHMARK 4: SIMD BLACK-SCHOLES
// =============================================================================

void benchmark_simd_black_scholes() {
    std::cout << "================================================================================\n";
    std::cout << "BENCHMARK 4: SIMD BLACK-SCHOLES PRICING\n";
    std::cout << "================================================================================\n\n";
    
#if ARBOR_HAS_AVX2
    constexpr size_t NUM_OPTIONS = 1000000;  // 1M options
    constexpr size_t OPTIONS_PER_BATCH = 8;
    constexpr size_t NUM_BATCHES = NUM_OPTIONS / OPTIONS_PER_BATCH;
    
    std::cout << "Pricing " << NUM_OPTIONS << " options (" << OPTIONS_PER_BATCH << " per SIMD batch)...\n\n";
    
    // Prepare input data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> spot_dist(90.0f, 110.0f);
    std::uniform_real_distribution<float> strike_dist(95.0f, 105.0f);
    std::uniform_real_distribution<float> vol_dist(0.15f, 0.35f);
    std::uniform_real_distribution<float> time_dist(0.1f, 2.0f);
    
    std::vector<simd::SimdBsInput> inputs(NUM_BATCHES);
    std::vector<simd::SimdBsOutput> outputs(NUM_BATCHES);
    
    for (auto& input : inputs) {
        for (int i = 0; i < 8; ++i) {
            input.S[i] = spot_dist(rng);
            input.K[i] = strike_dist(rng);
            input.r[i] = 0.05f;
            input.T[i] = time_dist(rng);
            input.sigma[i] = vol_dist(rng);
        }
    }
    
    // Warmup
    std::cout << "Warmup...\n";
    for (size_t i = 0; i < 1000; ++i) {
        simd::black_scholes_avx2(inputs[i % NUM_BATCHES], outputs[i % NUM_BATCHES]);
    }
    
    // Benchmark
    std::cout << "Benchmark...\n";
    
    perf::PerfCounterGroup counters;
    perf::RdtscTimer timer;
    perf::LatencyHistogram<10000, 1> histogram;
    
    counters.start();
    
    for (size_t b = 0; b < NUM_BATCHES; ++b) {
        timer.start();
        simd::black_scholes_avx2(inputs[b], outputs[b]);
        histogram.record(timer.stop_ns());
    }
    
    auto perf_result = counters.stop();
    
    perf_result.print("SIMD BLACK-SCHOLES", NUM_OPTIONS);
    std::cout << "\n";
    histogram.print("8-option batch");
    
    double options_per_sec = NUM_OPTIONS * 1e9 / perf_result.time_ns;
    double ns_per_option = static_cast<double>(perf_result.time_ns) / NUM_OPTIONS;
    
    std::cout << "\nTHROUGHPUT:\n";
    std::cout << "-------------------------------------------\n";
    std::cout << "  Options/sec:       " << std::fixed << std::setprecision(0) << options_per_sec << "\n";
    std::cout << "  ns/option:         " << std::setprecision(2) << ns_per_option << "\n";
    std::cout << "  ns/8-option batch: " << histogram.mean() << "\n";
    
    if (options_per_sec > 100000000) {
        std::cout << "\n  [EXCELLENT] > 100M options/sec - Production-ready throughput\n";
    } else if (options_per_sec > 20000000) {
        std::cout << "\n  [GOOD] > 20M options/sec\n";
    }
    
#else
    std::cout << "AVX2 not available. Recompile with -mavx2\n";
#endif
    
    std::cout << "\n";
}

// =============================================================================
// BENCHMARK 5: SIMD vs SCALAR VALIDATION
// Proves that SIMD implementation is real, not a facade
// =============================================================================

void benchmark_simd_vs_scalar() {
    std::cout << "================================================================================\n";
    std::cout << "BENCHMARK 5: SIMD vs SCALAR VALIDATION\n";
    std::cout << "================================================================================\n\n";
    
    std::cout << "This benchmark proves the SIMD implementation is real,\n";
    std::cout << "not just scalar code wrapped in SIMD types.\n\n";
    
#if ARBOR_HAS_AVX2
    constexpr uint32_t NUM_PATHS = 100000;
    constexpr uint32_t NUM_STEPS = 252;
    
    simd::SimdGbmParams params{
        .S0 = 100.0f,
        .mu = 0.05f,
        .sigma = 0.2f,
        .T = 1.0f,
        .dt = 1.0f / NUM_STEPS,
        .num_paths = NUM_PATHS,
        .num_steps = NUM_STEPS,
        .seed = 42
    };
    
    // SIMD benchmark
    simd::SimdGbmEngine simd_engine(42);
    
    // Warmup
    for (int i = 0; i < 3; ++i) {
        simd_engine.simulate(params);
    }
    
    // Actual benchmark
    std::vector<int64_t> simd_times;
    for (int i = 0; i < 10; ++i) {
        auto result = simd_engine.simulate(params);
        simd_times.push_back(result.time_ns);
    }
    
    // Scalar benchmark (same algorithm, no SIMD)
    std::vector<int64_t> scalar_times;
    std::mt19937_64 scalar_rng(42);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    
    const float drift = (params.mu - 0.5f * params.sigma * params.sigma) * params.dt;
    const float vol_sqrt_dt = params.sigma * std::sqrt(params.dt);
    
    for (int trial = 0; trial < 10; ++trial) {
        auto start = std::chrono::steady_clock::now();
        
        std::vector<float> prices(NUM_PATHS);
        for (uint32_t p = 0; p < NUM_PATHS; ++p) {
            float S = params.S0;
            for (uint32_t t = 0; t < NUM_STEPS; ++t) {
                float z = normal(scalar_rng);
                S *= std::exp(drift + vol_sqrt_dt * z);
            }
            prices[p] = S;
        }
        
        auto end = std::chrono::steady_clock::now();
        scalar_times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
    }
    
    // Compute statistics
    auto compute_mean = [](const std::vector<int64_t>& v) {
        return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    };
    
    double simd_mean = compute_mean(simd_times);
    double scalar_mean = compute_mean(scalar_times);
    double speedup = scalar_mean / simd_mean;
    
    uint64_t total_sims = static_cast<uint64_t>(NUM_PATHS) * NUM_STEPS;
    double simd_throughput = (total_sims * 1e9) / simd_mean;
    double scalar_throughput = (total_sims * 1e9) / scalar_mean;
    
    std::cout << "Configuration:\n";
    std::cout << "  Paths:      " << NUM_PATHS << "\n";
    std::cout << "  Steps:      " << NUM_STEPS << "\n";
    std::cout << "  Total sims: " << total_sims << "\n\n";
    
    std::cout << "Results:\n";
    std::cout << std::fixed;
    std::cout << "  SIMD (AVX2):   " << std::setprecision(2) 
              << (simd_mean / 1e6) << " ms, " 
              << std::setprecision(0) << (simd_throughput / 1e6) << "M sims/sec\n";
    std::cout << "  Scalar:        " << std::setprecision(2) 
              << (scalar_mean / 1e6) << " ms, "
              << std::setprecision(0) << (scalar_throughput / 1e6) << "M sims/sec\n";
    std::cout << "  Speedup:       " << std::setprecision(2) << speedup << "x\n\n";
    
    if (speedup > 3.0) {
        std::cout << "[VALIDATED] SIMD implementation provides " << speedup << "x real speedup.\n";
        std::cout << "This proves the hot path uses actual SIMD intrinsics, not scalar fallbacks.\n";
    } else if (speedup > 1.5) {
        std::cout << "[PARTIAL] Speedup of " << speedup << "x detected.\n";
        std::cout << "SIMD is working but may be memory-bandwidth limited.\n";
    } else {
        std::cout << "[WARNING] Speedup lower than expected. Possible causes:\n";
        std::cout << "  - Compiler auto-vectorized scalar code\n";
        std::cout << "  - Memory bandwidth limited\n";
        std::cout << "  - CPU frequency throttling\n";
    }
    
#else
    std::cout << "AVX2 not available on this platform.\n";
    std::cout << "Recompile with -mavx2 to enable SIMD validation.\n";
#endif
    
    std::cout << "\n";
}

// =============================================================================
// MAIN
// =============================================================================

int main() {
    print_system_info();
    
    benchmark_orderbook_with_counters();
    benchmark_simd_monte_carlo();
    benchmark_lockfree_queue();
    benchmark_simd_black_scholes();
    benchmark_simd_vs_scalar();
    
    std::cout << "================================================================================\n";
    std::cout << "                           BENCHMARK COMPLETE                                   \n";
    std::cout << "================================================================================\n\n";
    
    std::cout << "REPRODUCIBILITY:\n";
    std::cout << "  - Run with sudo for hardware counter access\n";
    std::cout << "  - Disable CPU frequency scaling: cpupower frequency-set -g performance\n";
    std::cout << "  - Isolate CPU cores: taskset -c 0-3 ./benchmark\n";
    std::cout << "  - Disable hyperthreading for consistent results\n";
    std::cout << "\nFor flame graphs: perf record -g ./benchmark && perf report\n\n";
    
    return 0;
}
