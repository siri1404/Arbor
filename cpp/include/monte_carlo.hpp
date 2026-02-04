#pragma once

#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <thread>
#include <future>

namespace arbor::montecarlo {

// Simulation parameters
struct SimulationParams {
    double S0;           // Initial price
    double mu;           // Expected return (drift)
    double sigma;        // Volatility
    double T;            // Time horizon (years)
    double dt;           // Time step
    size_t num_paths;    // Number of simulation paths
    size_t num_steps;    // Number of time steps
    uint64_t seed;       // Random seed
};

// Statistical results
struct Statistics {
    double mean_final_price;
    double std_final_price;
    double min_final_price;
    double max_final_price;
    double median_final_price;
    double expected_return;
    double realized_volatility;
    double sharpe_ratio;
    double max_drawdown;
    double skewness;
    double kurtosis;
};

// Value at Risk results
struct VaRResult {
    double var_95;
    double var_99;
    double cvar_95;  // Conditional VaR (Expected Shortfall)
    double cvar_99;
    std::vector<std::pair<double, double>> percentiles;  // (percentile, value)
};

// Complete simulation result
struct SimulationResult {
    std::vector<std::vector<double>> paths;  // paths x steps
    std::vector<double> final_prices;
    Statistics stats;
    VaRResult var;
    int64_t calc_time_ns;
};

// High-performance Monte Carlo engine with multi-threading
class MonteCarloEngine {
public:
    explicit MonteCarloEngine(unsigned int num_threads = 0);
    
    // Simulate Geometric Brownian Motion: dS = μS dt + σS dW
    [[nodiscard]] SimulationResult simulate_gbm(const SimulationParams& params);
    
    // Calculate VaR from returns distribution
    [[nodiscard]] static VaRResult calculate_var(
        const std::vector<double>& returns,
        double portfolio_value
    );
    
    // Calculate statistics from price paths
    [[nodiscard]] static Statistics calculate_statistics(
        const std::vector<double>& final_prices,
        double initial_price,
        double time_horizon
    );
    
private:
    unsigned int num_threads_;
    
    // Box-Muller transform for normal random generation (cache-friendly)
    [[nodiscard]] static double box_muller(std::mt19937_64& rng, 
                                          std::normal_distribution<double>& dist);
    
    // Single path simulation (vectorized inner loop)
    void simulate_path(
        std::vector<double>& path_out,
        const SimulationParams& params,
        std::mt19937_64& rng,
        std::normal_distribution<double>& dist
    ) const;
    
    // Worker thread function for parallel execution
    void simulate_paths_threaded(
        std::vector<std::vector<double>>& paths_out,
        size_t start_idx,
        size_t end_idx,
        const SimulationParams& params,
        uint64_t thread_seed
    ) const;
};

// Correlation matrix utilities
class CorrelationMatrix {
public:
    explicit CorrelationMatrix(size_t dimension);
    
    void set(size_t i, size_t j, double correlation);
    [[nodiscard]] double get(size_t i, size_t j) const;
    
    // Cholesky decomposition for correlated random generation
    [[nodiscard]] std::vector<std::vector<double>> cholesky_decomposition() const;
    
    [[nodiscard]] bool is_positive_definite() const;
    
private:
    size_t dim_;
    std::vector<double> data_;  // Stored as flat array for cache efficiency
    
    [[nodiscard]] inline size_t index(size_t i, size_t j) const noexcept {
        return i * dim_ + j;
    }
};

// Multi-asset portfolio simulation with correlations
struct PortfolioAsset {
    std::string symbol;
    double weight;
    double S0;
    double mu;
    double sigma;
};

struct PortfolioParams {
    std::vector<PortfolioAsset> assets;
    CorrelationMatrix correlation;
    double T;
    double dt;
    size_t num_paths;
    uint64_t seed;
};

struct PortfolioResult {
    std::vector<std::vector<double>> portfolio_paths;  // Portfolio value paths
    std::vector<double> final_values;
    std::vector<std::pair<std::string, std::vector<std::vector<double>>>> asset_paths;
    Statistics stats;
    VaRResult var;
    std::vector<std::pair<std::string, double>> risk_contributions;
    int64_t calc_time_ns;
};

[[nodiscard]] PortfolioResult simulate_portfolio(const PortfolioParams& params);

} // namespace arbor::montecarlo
