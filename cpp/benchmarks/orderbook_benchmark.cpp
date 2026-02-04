/**
 * Production-Grade Order Book Benchmark Suite
 * 
 * Methodology (Following SPEC CPU and MLPerf standards):
 * 1. Warmup phase: Allow JIT, caches, and branch predictors to stabilize
 * 2. Multiple iterations: Statistical significance via repeated measurements
 * 3. Cache priming: Pre-touch all data structures before timing
 * 4. CPU pinning recommendation: Isolate from OS scheduler jitter
 * 5. Statistical analysis: Mean, stddev, percentiles, confidence intervals
 * 
 * Hardware profiling commands (run separately):
 *   perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses ./benchmark
 *   perf record -g ./benchmark && perf report
 */

#include "../include/orderbook.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <fstream>

#if defined(__linux__)
#include <sys/utsname.h>
#include <unistd.h>
#include <fstream>
#endif

using namespace arbor::orderbook;

// =============================================================================
// SYSTEM INFORMATION COLLECTOR
// =============================================================================

struct SystemInfo {
    std::string cpu_model;
    std::string os_version;
    std::string compiler_version;
    int num_cores;
    size_t l1_cache_size;
    size_t l2_cache_size;
    size_t l3_cache_size;
};

SystemInfo get_system_info() {
    SystemInfo info;
    
#if defined(__linux__)
    // CPU model
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
        if (line.find("model name") != std::string::npos) {
            info.cpu_model = line.substr(line.find(":") + 2);
            break;
        }
    }
    
    // OS version
    struct utsname uts;
    if (uname(&uts) == 0) {
        info.os_version = std::string(uts.sysname) + " " + uts.release;
    }
    
    // Core count
    info.num_cores = sysconf(_SC_NPROCESSORS_ONLN);
    
    // Cache sizes (from sysfs)
    auto read_cache = [](const std::string& path) -> size_t {
        std::ifstream f(path);
        std::string s;
        if (std::getline(f, s)) {
            size_t val = std::stoull(s);
            if (s.back() == 'K') val *= 1024;
            else if (s.back() == 'M') val *= 1024 * 1024;
            return val;
        }
        return 0;
    };
    
    info.l1_cache_size = read_cache("/sys/devices/system/cpu/cpu0/cache/index0/size");
    info.l2_cache_size = read_cache("/sys/devices/system/cpu/cpu0/cache/index2/size");
    info.l3_cache_size = read_cache("/sys/devices/system/cpu/cpu0/cache/index3/size");
#else
    info.cpu_model = "Unknown (non-Linux)";
    info.os_version = "Unknown";
    info.num_cores = std::thread::hardware_concurrency();
#endif
    
    // Compiler version
#if defined(__clang__)
    info.compiler_version = "Clang " + std::to_string(__clang_major__) + "." + 
                           std::to_string(__clang_minor__);
#elif defined(__GNUC__)
    info.compiler_version = "GCC " + std::to_string(__GNUC__) + "." + 
                           std::to_string(__GNUC_MINOR__);
#else
    info.compiler_version = "Unknown";
#endif
    
    return info;
}

void print_system_info(const SystemInfo& info) {
    std::cout << "HARDWARE CONFIGURATION:\n";
    std::cout << "-------------------------------------------\n";
    std::cout << "CPU:         " << info.cpu_model << "\n";
    std::cout << "Cores:       " << info.num_cores << "\n";
    std::cout << "L1 Cache:    " << (info.l1_cache_size / 1024) << " KB\n";
    std::cout << "L2 Cache:    " << (info.l2_cache_size / 1024) << " KB\n";
    std::cout << "L3 Cache:    " << (info.l3_cache_size / 1024 / 1024) << " MB\n";
    std::cout << "OS:          " << info.os_version << "\n";
    std::cout << "Compiler:    " << info.compiler_version << "\n";
    std::cout << "Build flags: -O3 -march=native -DNDEBUG\n";
    std::cout << "\n";
}

// =============================================================================
// STATISTICAL ANALYSIS
// =============================================================================

struct BenchmarkStats {
    double mean;
    double stddev;
    double min_val;
    double max_val;
    double p50;
    double p90;
    double p99;
    double p999;
    double ci95_lower;  // 95% confidence interval
    double ci95_upper;
    size_t sample_count;
};

BenchmarkStats compute_stats(std::vector<int64_t>& samples) {
    BenchmarkStats stats{};
    stats.sample_count = samples.size();
    
    if (samples.empty()) return stats;
    
    // Sort for percentiles
    std::sort(samples.begin(), samples.end());
    
    // Min/Max
    stats.min_val = static_cast<double>(samples.front());
    stats.max_val = static_cast<double>(samples.back());
    
    // Mean
    double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    stats.mean = sum / samples.size();
    
    // Standard deviation
    double sq_sum = 0;
    for (auto v : samples) {
        double diff = static_cast<double>(v) - stats.mean;
        sq_sum += diff * diff;
    }
    stats.stddev = std::sqrt(sq_sum / samples.size());
    
    // Percentiles
    auto percentile = [&samples](double p) {
        size_t idx = static_cast<size_t>(samples.size() * p);
        if (idx >= samples.size()) idx = samples.size() - 1;
        return static_cast<double>(samples[idx]);
    };
    
    stats.p50 = percentile(0.50);
    stats.p90 = percentile(0.90);
    stats.p99 = percentile(0.99);
    stats.p999 = percentile(0.999);
    
    // 95% confidence interval (assuming normal distribution)
    double se = stats.stddev / std::sqrt(samples.size());
    stats.ci95_lower = stats.mean - 1.96 * se;
    stats.ci95_upper = stats.mean + 1.96 * se;
    
    return stats;
}

void print_stats(const std::string& name, const BenchmarkStats& stats, const std::string& unit = "ns") {
    std::cout << name << ":\n";
    std::cout << "  Mean:        " << std::fixed << std::setprecision(2) << stats.mean << " " << unit << "\n";
    std::cout << "  Stddev:      " << stats.stddev << " " << unit << "\n";
    std::cout << "  Min:         " << stats.min_val << " " << unit << "\n";
    std::cout << "  Max:         " << stats.max_val << " " << unit << "\n";
    std::cout << "  P50:         " << stats.p50 << " " << unit << "\n";
    std::cout << "  P90:         " << stats.p90 << " " << unit << "\n";
    std::cout << "  P99:         " << stats.p99 << " " << unit << "\n";
    std::cout << "  P99.9:       " << stats.p999 << " " << unit << "\n";
    std::cout << "  95% CI:      [" << stats.ci95_lower << ", " << stats.ci95_upper << "] " << unit << "\n";
    std::cout << "  Samples:     " << stats.sample_count << "\n";
}

// =============================================================================
// BENCHMARK: ADD ORDER LATENCY
// =============================================================================

void benchmark_add_order_latency() {
    std::cout << "\n=== BENCHMARK: ADD ORDER LATENCY ===\n\n";
    
    constexpr int WARMUP_ITERATIONS = 1000;
    constexpr int BENCHMARK_ITERATIONS = 100000;
    
    LimitOrderBook book("AAPL", 1);
    std::vector<Trade> trades;
    trades.reserve(1000);
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<uint64_t> price_dist(14900, 15100);
    std::uniform_int_distribution<uint32_t> qty_dist(1, 100);
    
    // Pre-seed book with realistic depth
    std::cout << "Phase 1: Seeding order book with 10,000 resting orders...\n";
    for (int i = 0; i < 5000; ++i) {
        book.add_order(Side::BUY, OrderType::LIMIT, price_dist(rng) - 50, qty_dist(rng), TimeInForce::GTC, 0, &trades);
        book.add_order(Side::SELL, OrderType::LIMIT, price_dist(rng) + 50, qty_dist(rng), TimeInForce::GTC, 0, &trades);
    }
    trades.clear();
    
    std::cout << "  Book depth: " << book.total_orders() << " orders, "
              << book.bid_levels() << " bid levels, " << book.ask_levels() << " ask levels\n";
    std::cout << "  Best bid: " << book.best_bid() << " | Best ask: " << book.best_ask() << "\n\n";
    
    // Warmup phase - stabilize caches and branch predictors
    std::cout << "Phase 2: Warmup (" << WARMUP_ITERATIONS << " iterations)...\n";
    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        Side side = (i % 2 == 0) ? Side::BUY : Side::SELL;
        uint64_t price = price_dist(rng) + (side == Side::BUY ? -100 : 100);
        book.add_order(side, OrderType::LIMIT, price, qty_dist(rng), TimeInForce::GTC, 0, &trades);
    }
    trades.clear();
    book.clear();
    
    // Re-seed after warmup
    for (int i = 0; i < 5000; ++i) {
        book.add_order(Side::BUY, OrderType::LIMIT, price_dist(rng) - 50, qty_dist(rng), TimeInForce::GTC, 0, &trades);
        book.add_order(Side::SELL, OrderType::LIMIT, price_dist(rng) + 50, qty_dist(rng), TimeInForce::GTC, 0, &trades);
    }
    trades.clear();
    
    // Benchmark phase - individual order latencies
    std::cout << "Phase 3: Benchmark (" << BENCHMARK_ITERATIONS << " orders)...\n";
    std::vector<int64_t> latencies;
    latencies.reserve(BENCHMARK_ITERATIONS);
    
    for (int i = 0; i < BENCHMARK_ITERATIONS; ++i) {
        Side side = (i % 2 == 0) ? Side::BUY : Side::SELL;
        uint64_t price = price_dist(rng) + (side == Side::BUY ? -100 : 100);
        uint32_t qty = qty_dist(rng);
        
        const auto start = std::chrono::steady_clock::now();
        book.add_order(side, OrderType::LIMIT, price, qty, TimeInForce::GTC, 0, &trades);
        const auto end = std::chrono::steady_clock::now();
        
        latencies.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
        
        // Periodically clear trades to avoid memory growth
        if (trades.size() > 10000) trades.clear();
    }
    
    // Statistical analysis
    auto stats = compute_stats(latencies);
    
    std::cout << "\nRESULTS: ADD ORDER LATENCY\n";
    std::cout << "-------------------------------------------\n";
    print_stats("Per-order latency", stats);
    
    double throughput = 1e9 / stats.mean;
    std::cout << "\nTHROUGHPUT: " << std::fixed << std::setprecision(0) << throughput << " orders/sec\n";
    
    // Quality assessment
    std::cout << "\nASSESSMENT:\n";
    if (stats.p99 < 1000) {
        std::cout << "  [EXCELLENT] P99 < 1us - Suitable for HFT market making\n";
    } else if (stats.p99 < 10000) {
        std::cout << "  [GOOD] P99 < 10us - Suitable for quantitative trading\n";
    } else {
        std::cout << "  [ACCEPTABLE] P99 > 10us - Suitable for general trading\n";
    }
}

// =============================================================================
// BENCHMARK: CANCEL ORDER LATENCY (Critical for HFT)
// =============================================================================

void benchmark_cancel_latency() {
    std::cout << "\n=== BENCHMARK: CANCEL ORDER LATENCY ===\n\n";
    
    constexpr int BENCHMARK_ITERATIONS = 50000;
    
    LimitOrderBook book("AAPL", 1);
    std::vector<Trade> trades;
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<uint64_t> price_dist(14900, 15100);
    std::uniform_int_distribution<uint32_t> qty_dist(1, 100);
    
    // Create orders and store their IDs
    std::cout << "Creating " << BENCHMARK_ITERATIONS << " orders to cancel...\n";
    std::vector<uint64_t> order_ids;
    order_ids.reserve(BENCHMARK_ITERATIONS);
    
    for (int i = 0; i < BENCHMARK_ITERATIONS; ++i) {
        Side side = (i % 2 == 0) ? Side::BUY : Side::SELL;
        uint64_t price = price_dist(rng) + (side == Side::BUY ? -200 : 200);  // Far from spread
        uint64_t oid = book.add_order(side, OrderType::LIMIT, price, qty_dist(rng), TimeInForce::GTC, 0, &trades);
        if (oid > 0) order_ids.push_back(oid);
    }
    
    std::cout << "  Created " << order_ids.size() << " resting orders\n\n";
    
    // Shuffle to randomize access pattern (worst case for cache)
    std::shuffle(order_ids.begin(), order_ids.end(), rng);
    
    // Benchmark cancellations
    std::cout << "Benchmarking cancel operations (randomized order)...\n";
    std::vector<int64_t> latencies;
    latencies.reserve(order_ids.size());
    
    for (uint64_t oid : order_ids) {
        const auto start = std::chrono::steady_clock::now();
        book.cancel_order(oid);
        const auto end = std::chrono::steady_clock::now();
        
        latencies.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
    }
    
    auto stats = compute_stats(latencies);
    
    std::cout << "\nRESULTS: CANCEL ORDER LATENCY\n";
    std::cout << "-------------------------------------------\n";
    print_stats("Per-cancel latency", stats);
    
    std::cout << "\nNOTE: Cancel latency is critical for HFT risk management.\n";
    std::cout << "      Sub-microsecond P99 enables fast position unwinding.\n";
}

// =============================================================================
// BENCHMARK: SUSTAINED THROUGHPUT
// =============================================================================

void benchmark_throughput() {
    std::cout << "\n=== BENCHMARK: SUSTAINED THROUGHPUT ===\n\n";
    
    constexpr int NUM_ITERATIONS = 5;
    constexpr int ORDERS_PER_ITERATION = 100000;
    
    std::vector<double> throughputs;
    
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        LimitOrderBook book("SPY", 1);
        std::vector<Trade> trades;
        trades.reserve(50000);
        std::mt19937_64 rng(123 + iter);
        std::uniform_int_distribution<uint64_t> price_dist(40000, 41000);
        std::uniform_int_distribution<uint32_t> qty_dist(100, 500);
        
        const auto start = std::chrono::steady_clock::now();
        
        for (int i = 0; i < ORDERS_PER_ITERATION; ++i) {
            Side side = (i % 2 == 0) ? Side::BUY : Side::SELL;
            OrderType type = (i % 10 == 0) ? OrderType::MARKET : OrderType::LIMIT;
            uint64_t price = price_dist(rng);
            
            book.add_order(side, type, price, qty_dist(rng), TimeInForce::GTC, 0, &trades);
            
            // Periodic cleanup to simulate steady state
            if (trades.size() > 10000) trades.clear();
        }
        
        const auto end = std::chrono::steady_clock::now();
        const int64_t total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        double orders_per_sec = (ORDERS_PER_ITERATION * 1e9) / total_ns;
        throughputs.push_back(orders_per_sec);
        
        std::cout << "  Iteration " << (iter + 1) << ": " << std::fixed << std::setprecision(0) 
                  << orders_per_sec << " orders/sec\n";
    }
    
    // Statistics across iterations
    double mean = std::accumulate(throughputs.begin(), throughputs.end(), 0.0) / throughputs.size();
    double sq_sum = 0;
    for (double t : throughputs) {
        sq_sum += (t - mean) * (t - mean);
    }
    double stddev = std::sqrt(sq_sum / throughputs.size());
    
    std::cout << "\nTHROUGHPUT SUMMARY:\n";
    std::cout << "-------------------------------------------\n";
    std::cout << "Mean:     " << std::fixed << std::setprecision(0) << mean << " orders/sec\n";
    std::cout << "Stddev:   " << stddev << " orders/sec\n";
    std::cout << "CV:       " << std::setprecision(2) << (stddev/mean * 100) << "%\n";
    
    std::cout << "\nINDUSTRY COMPARISON:\n";
    std::cout << "-------------------------------------------\n";
    std::cout << "Arbor Engine:            " << std::setprecision(0) << mean << " orders/sec\n";
    std::cout << "NASDAQ ITCH Feed:        ~500,000 messages/sec\n";
    std::cout << "CME Globex:              ~1,000,000 messages/sec\n";
    std::cout << "Top-tier HFT systems:    ~5,000,000+ orders/sec\n";
}

int main() {
    std::cout << "============================================================\n";
    std::cout << "          ARBOR ORDER BOOK ENGINE - BENCHMARK SUITE         \n";
    std::cout << "    Production-Grade Performance Validation (v2.0)          \n";
    std::cout << "============================================================\n\n";
    
    // Print system information for reproducibility
    auto sysinfo = get_system_info();
    print_system_info(sysinfo);
    
    // Run benchmark suite
    benchmark_add_order_latency();
    benchmark_cancel_latency();
    benchmark_throughput();
    
    // Summary
    std::cout << "\n============================================================\n";
    std::cout << "                    BENCHMARK COMPLETE                       \n";
    std::cout << "============================================================\n\n";
    
    std::cout << "REPRODUCIBILITY CHECKLIST:\n";
    std::cout << "  [ ] Compiled with: -O3 -march=native -DNDEBUG\n";
    std::cout << "  [ ] CPU frequency scaling disabled (performance governor)\n";
    std::cout << "  [ ] CPU isolation (isolcpus) or taskset used\n";
    std::cout << "  [ ] System idle (no background load)\n";
    std::cout << "  [ ] Results averaged over multiple runs\n\n";
    
    std::cout << "For profiling, run:\n";
    std::cout << "  perf stat -e cycles,instructions,cache-references,cache-misses ./benchmark\n";
    std::cout << "  perf record -g ./benchmark && perf report\n\n";
    
    return 0;
}
