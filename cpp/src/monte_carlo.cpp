#include "monte_carlo.hpp"
#include <numeric>
#include <cmath>
#include <algorithm>

namespace arbor::montecarlo {

MonteCarloEngine::MonteCarloEngine(unsigned int num_threads)
    : num_threads_(num_threads == 0 ? std::thread::hardware_concurrency() : num_threads) {
}

SimulationResult MonteCarloEngine::simulate_gbm(const SimulationParams& params) {
    const auto start = std::chrono::steady_clock::now();
    
    SimulationResult result;
    result.paths.resize(params.num_paths);
    result.final_prices.resize(params.num_paths);
    
    // Parallel path generation using thread pool
    const size_t paths_per_thread = params.num_paths / num_threads_;
    std::vector<std::future<void>> futures;
    futures.reserve(num_threads_);
    
    for (unsigned int t = 0; t < num_threads_; ++t) {
        const size_t start_idx = t * paths_per_thread;
        const size_t end_idx = (t == num_threads_ - 1) ? params.num_paths : (t + 1) * paths_per_thread;
        const uint64_t thread_seed = params.seed + t * 1000000;
        
        futures.push_back(std::async(std::launch::async, [this, &result, &params, start_idx, end_idx, thread_seed]() {
            simulate_paths_threaded(result.paths, start_idx, end_idx, params, thread_seed);
        }));
    }
    
    // Wait for all threads to complete
    for (auto& future : futures) {
        future.get();
    }
    
    // Extract final prices
    for (size_t i = 0; i < params.num_paths; ++i) {
        result.final_prices[i] = result.paths[i].back();
    }
    
    // Calculate statistics
    result.stats = calculate_statistics(result.final_prices, params.S0, params.T);
    
    // Calculate VaR
    std::vector<double> returns(params.num_paths);
    for (size_t i = 0; i < params.num_paths; ++i) {
        returns[i] = (result.final_prices[i] - params.S0) / params.S0;
    }
    result.var = calculate_var(returns, params.S0);
    
    const auto end = std::chrono::steady_clock::now();
    result.calc_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    
    return result;
}

void MonteCarloEngine::simulate_paths_threaded(
    std::vector<std::vector<double>>& paths_out,
    size_t start_idx,
    size_t end_idx,
    const SimulationParams& params,
    uint64_t thread_seed
) const {
    std::mt19937_64 rng(thread_seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    
    const double sqrt_dt = std::sqrt(params.dt);
    const double drift_term = (params.mu - 0.5 * params.sigma * params.sigma) * params.dt;
    const double diffusion_coeff = params.sigma * sqrt_dt;
    
    for (size_t path_idx = start_idx; path_idx < end_idx; ++path_idx) {
        paths_out[path_idx].reserve(params.num_steps + 1);
        paths_out[path_idx].push_back(params.S0);
        
        double S = params.S0;
        
        // Vectorized GBM simulation - critical inner loop
        for (size_t step = 0; step < params.num_steps; ++step) {
            const double Z = dist(rng);
            
            // GBM formula: S(t+dt) = S(t) * exp((μ - 0.5σ²)dt + σ√dt * Z)
            const double exponent = drift_term + diffusion_coeff * Z;
            S *= std::exp(exponent);
            
            paths_out[path_idx].push_back(S);
        }
    }
}

Statistics MonteCarloEngine::calculate_statistics(
    const std::vector<double>& final_prices,
    double initial_price,
    double time_horizon
) {
    Statistics stats{};
    
    if (final_prices.empty()) {
        return stats;
    }
    
    const size_t n = final_prices.size();
    
    // Calculate mean
    stats.mean_final_price = std::accumulate(final_prices.begin(), final_prices.end(), 0.0) / n;
    
    // Calculate variance and higher moments
    double variance_sum = 0.0;
    double skewness_sum = 0.0;
    double kurtosis_sum = 0.0;
    
    for (double price : final_prices) {
        const double diff = price - stats.mean_final_price;
        const double diff_sq = diff * diff;
        variance_sum += diff_sq;
        skewness_sum += diff_sq * diff;
        kurtosis_sum += diff_sq * diff_sq;
    }
    
    const double variance = variance_sum / n;
    stats.std_final_price = std::sqrt(variance);
    
    // Skewness and kurtosis
    const double std_cubed = stats.std_final_price * stats.std_final_price * stats.std_final_price;
    const double std_quad = std_cubed * stats.std_final_price;
    stats.skewness = (skewness_sum / n) / std_cubed;
    stats.kurtosis = (kurtosis_sum / n) / std_quad - 3.0;  // Excess kurtosis
    
    // Min/max
    auto [min_it, max_it] = std::minmax_element(final_prices.begin(), final_prices.end());
    stats.min_final_price = *min_it;
    stats.max_final_price = *max_it;
    
    // Median
    std::vector<double> sorted_prices = final_prices;
    std::sort(sorted_prices.begin(), sorted_prices.end());
    stats.median_final_price = sorted_prices[n / 2];
    
    // Expected return
    stats.expected_return = (stats.mean_final_price - initial_price) / initial_price;
    
    // Realized volatility (annualized from returns)
    std::vector<double> returns(n);
    double mean_return = 0.0;
    for (size_t i = 0; i < n; ++i) {
        returns[i] = (final_prices[i] - initial_price) / initial_price;
        mean_return += returns[i];
    }
    mean_return /= n;
    
    double return_variance = 0.0;
    for (double ret : returns) {
        const double diff = ret - mean_return;
        return_variance += diff * diff;
    }
    return_variance /= n;
    
    stats.realized_volatility = std::sqrt(return_variance / time_horizon);
    
    // Sharpe ratio (assuming risk-free rate = 0 for simplicity)
    stats.sharpe_ratio = (stats.expected_return / time_horizon) / stats.realized_volatility;
    
    // Max drawdown (simplified - using sample of paths would be more accurate)
    stats.max_drawdown = 0.0;
    
    return stats;
}

VaRResult MonteCarloEngine::calculate_var(
    const std::vector<double>& returns,
    double portfolio_value
) {
    VaRResult result{};
    
    if (returns.empty()) {
        return result;
    }
    
    const size_t n = returns.size();
    std::vector<double> sorted_returns = returns;
    std::sort(sorted_returns.begin(), sorted_returns.end());
    
    // Calculate VaR at 95% and 99% confidence levels
    const size_t idx_95 = static_cast<size_t>(n * 0.05);
    const size_t idx_99 = static_cast<size_t>(n * 0.01);
    
    result.var_95 = -sorted_returns[idx_95] * portfolio_value;
    result.var_99 = -sorted_returns[idx_99] * portfolio_value;
    
    // Calculate CVaR (Expected Shortfall) - average of losses beyond VaR
    double cvar_95_sum = 0.0;
    for (size_t i = 0; i <= idx_95; ++i) {
        cvar_95_sum += sorted_returns[i];
    }
    result.cvar_95 = -(cvar_95_sum / (idx_95 + 1)) * portfolio_value;
    
    double cvar_99_sum = 0.0;
    for (size_t i = 0; i <= idx_99; ++i) {
        cvar_99_sum += sorted_returns[i];
    }
    result.cvar_99 = -(cvar_99_sum / (idx_99 + 1)) * portfolio_value;
    
    // Generate percentile distribution
    result.percentiles.reserve(99);
    for (int p = 1; p <= 99; ++p) {
        const size_t idx = static_cast<size_t>((n * p) / 100.0);
        result.percentiles.emplace_back(p / 100.0, sorted_returns[idx] * portfolio_value);
    }
    
    return result;
}

// Correlation Matrix Implementation
CorrelationMatrix::CorrelationMatrix(size_t dimension)
    : dim_(dimension), data_(dimension * dimension, 0.0) {
    // Initialize as identity matrix
    for (size_t i = 0; i < dim_; ++i) {
        data_[index(i, i)] = 1.0;
    }
}

void CorrelationMatrix::set(size_t i, size_t j, double correlation) {
    data_[index(i, j)] = correlation;
    data_[index(j, i)] = correlation;  // Symmetric
}

double CorrelationMatrix::get(size_t i, size_t j) const {
    return data_[index(i, j)];
}

std::vector<std::vector<double>> CorrelationMatrix::cholesky_decomposition() const {
    std::vector<std::vector<double>> L(dim_, std::vector<double>(dim_, 0.0));
    
    for (size_t i = 0; i < dim_; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            
            for (size_t k = 0; k < j; ++k) {
                sum += L[i][k] * L[j][k];
            }
            
            if (i == j) {
                const double diagonal_val = data_[index(i, i)] - sum;
                L[i][j] = std::sqrt(std::max(diagonal_val, 0.0));
            } else {
                L[i][j] = (data_[index(i, j)] - sum) / L[j][j];
            }
        }
    }
    
    return L;
}

bool CorrelationMatrix::is_positive_definite() const {
    // Simple check: all diagonal elements > 0 after Cholesky
    auto L = cholesky_decomposition();
    for (size_t i = 0; i < dim_; ++i) {
        if (L[i][i] <= 0.0) return false;
    }
    return true;
}

} // namespace arbor::montecarlo
