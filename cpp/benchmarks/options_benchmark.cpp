#include "../include/options_pricing.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>

using namespace arbor::options;

void benchmark_single_pricing() {
    std::cout << "\n=== SINGLE OPTION PRICING BENCHMARK ===\n\n";
    
    const double S = 150.0;
    const double K = 150.0;
    const double T = 0.25;  // 3 months
    const double r = 0.05;
    const double sigma = 0.25;
    
    std::cout << "Parameters: S=$150, K=$150, T=0.25y, r=5%, σ=25%\n";
    std::cout << "Running 100,000 option pricing calculations...\n\n";
    
    const int NUM_CALCS = 100000;
    std::vector<PricingResult> results;
    results.reserve(NUM_CALCS);
    
    const auto start = std::chrono::steady_clock::now();
    
    for (int i = 0; i < NUM_CALCS; ++i) {
        results.push_back(BlackScholesPricer::price(S, K, T, r, sigma, OptionType::CALL));
    }
    
    const auto end = std::chrono::steady_clock::now();
    const int64_t total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    
    // Calculate statistics
    int64_t min_ns = INT64_MAX;
    int64_t max_ns = 0;
    int64_t sum_ns = 0;
    
    for (const auto& result : results) {
        min_ns = std::min(min_ns, result.calc_time_ns);
        max_ns = std::max(max_ns, result.calc_time_ns);
        sum_ns += result.calc_time_ns;
    }
    
    const double avg_ns = static_cast<double>(sum_ns) / NUM_CALCS;
    
    std::cout << "📊 RESULTS:\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << std::fixed << std::setprecision(0);
    std::cout << "Total calculations:        " << NUM_CALCS << "\n";
    std::cout << "Total time:                " << total_ns / 1000 << " μs\n";
    std::cout << "Average time per pricing:  " << avg_ns << " ns\n";
    std::cout << "Minimum time:              " << min_ns << " ns\n";
    std::cout << "Maximum time:              " << max_ns << " ns\n";
    std::cout << "Throughput:                " << std::setprecision(0) 
              << (NUM_CALCS * 1e9) / total_ns << " pricings/sec\n";
    
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\nSample result:\n";
    std::cout << "  Call price: $" << results[0].price << "\n";
    std::cout << "  Delta: " << results[0].greeks.delta << "\n";
    std::cout << "  Gamma: " << results[0].greeks.gamma << "\n";
    std::cout << "  Theta: $" << results[0].greeks.theta << "/day\n";
    std::cout << "  Vega: $" << results[0].greeks.vega << "/1%\n";
    
    if (avg_ns < 1000) {
        std::cout << "\n[EXCELLENT] Sub-microsecond pricing achieved\n";
    } else if (avg_ns < 10000) {
        std::cout << "\n[GOOD] Average latency < 10 us\n";
    }
    
    std::cout << "\nIndustry target: < 1 us for real-time pricing\n";
    std::cout << "Status: " << (avg_ns < 1000 ? "[PRODUCTION READY]" : "[ACCEPTABLE]") << "\n";
}

void benchmark_option_chain() {
    std::cout << "\n=== OPTION CHAIN CALCULATION BENCHMARK ===\n\n";
    
    const double S = 150.0;
    const double T = 0.25;
    const double r = 0.05;
    const double sigma = 0.25;
    
    // Generate strikes from 80% to 120% of spot
    std::vector<double> strikes;
    for (double K = S * 0.8; K <= S * 1.2; K += 2.5) {
        strikes.push_back(K);
    }
    
    std::cout << "Calculating option chain for " << strikes.size() << " strikes...\n";
    std::cout << "Strike range: $" << strikes.front() << " - $" << strikes.back() << "\n\n";
    
    const auto start = std::chrono::steady_clock::now();
    
    auto chain = BlackScholesPricer::option_chain(S, strikes, T, r, sigma);
    
    const auto end = std::chrono::steady_clock::now();
    const int64_t total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    const double avg_ns_per_strike = static_cast<double>(total_ns) / strikes.size();
    
    std::cout << "📊 RESULTS:\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "Total time:                " << total_ns / 1000 << " μs\n";
    std::cout << "Time per strike (C+P):     " << std::fixed << std::setprecision(0) 
              << avg_ns_per_strike << " ns\n";
    std::cout << "Throughput:                " << (strikes.size() * 1e9) / total_ns 
              << " strikes/sec\n";
    
    std::cout << "\nSample chain (first 5 strikes):\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Strike | Call Price | Call Delta | Put Price  | Put Delta\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    
    for (size_t i = 0; i < std::min<size_t>(5, chain.size()); ++i) {
        const auto& [call, put] = chain[i];
        std::cout << "$" << std::setw(6) << strikes[i] << " | "
                  << "$" << std::setw(9) << call.price << " | "
                  << std::setw(10) << call.greeks.delta << " | "
                  << "$" << std::setw(9) << put.price << " | "
                  << std::setw(9) << put.greeks.delta << "\n";
    }
}

void benchmark_implied_volatility() {
    std::cout << "\n=== IMPLIED VOLATILITY SOLVER BENCHMARK ===\n\n";
    
    const double S = 150.0;
    const double K = 150.0;
    const double T = 0.25;
    const double r = 0.05;
    const double true_sigma = 0.25;
    
    // Calculate theoretical price
    auto theoretical = BlackScholesPricer::price(S, K, T, r, true_sigma, OptionType::CALL);
    const double market_price = theoretical.price;
    
    std::cout << "Solving for IV given market price: $" << std::fixed << std::setprecision(4) 
              << market_price << "\n";
    std::cout << "Expected IV: " << (true_sigma * 100) << "%\n";
    std::cout << "Running 10,000 IV calculations...\n\n";
    
    const int NUM_CALCS = 10000;
    std::vector<int64_t> times;
    times.reserve(NUM_CALCS);
    
    double sum_iv = 0.0;
    
    for (int i = 0; i < NUM_CALCS; ++i) {
        const auto start = std::chrono::steady_clock::now();
        
        double iv = BlackScholesPricer::implied_volatility(
            market_price, S, K, T, r, OptionType::CALL
        );
        
        const auto end = std::chrono::steady_clock::now();
        times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
        sum_iv += iv;
    }
    
    const double avg_iv = sum_iv / NUM_CALCS;
    const int64_t min_time = *std::min_element(times.begin(), times.end());
    const int64_t max_time = *std::max_element(times.begin(), times.end());
    const int64_t avg_time = std::accumulate(times.begin(), times.end(), 0LL) / NUM_CALCS;
    
    std::cout << "📊 RESULTS:\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "Solved IV:                 " << std::setprecision(2) << (avg_iv * 100) << "%\n";
    std::cout << "Error vs true σ:           " << std::scientific << std::abs(avg_iv - true_sigma) << "\n";
    std::cout << std::fixed << std::setprecision(0);
    std::cout << "Average solve time:        " << avg_time / 1000.0 << " μs\n";
    std::cout << "Minimum solve time:        " << min_time / 1000.0 << " μs\n";
    std::cout << "Maximum solve time:        " << max_time / 1000.0 << " μs\n";
    
    if (avg_time < 10000) {
        std::cout << "\n[GOOD] IV solver performance within target\n";
    }
}

int main() {
    std::cout << "============================================================\n";
    std::cout << "         ARBOR OPTIONS PRICING ENGINE - BENCHMARK           \n";
    std::cout << "============================================================\n";
    
    benchmark_single_pricing();
    benchmark_option_chain();
    benchmark_implied_volatility();
    
    std::cout << "\nAll benchmarks complete.\n\n";
    
    return 0;
}
