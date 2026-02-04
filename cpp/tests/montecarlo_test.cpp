#include <gtest/gtest.h>
#include "../include/monte_carlo.hpp"
#include <cmath>

using namespace arbor::montecarlo;

TEST(MonteCarloTest, BasicSimulation) {
    SimulationParams params{
        .S0 = 100.0,
        .mu = 0.10,
        .sigma = 0.25,
        .T = 1.0,
        .dt = 1.0 / 252.0,
        .num_paths = 1000,
        .num_steps = 252,
        .seed = 42
    };
    
    MonteCarloEngine engine(4);
    auto result = engine.simulate_gbm(params);
    
    EXPECT_EQ(result.paths.size(), 1000);
    EXPECT_EQ(result.final_prices.size(), 1000);
    EXPECT_GT(result.stats.mean_final_price, 0.0);
}

TEST(MonteCarloTest, ExpectedReturn) {
    SimulationParams params{
        .S0 = 100.0,
        .mu = 0.10,  // 10% expected return
        .sigma = 0.25,
        .T = 1.0,
        .dt = 1.0 / 252.0,
        .num_paths = 10000,
        .num_steps = 252,
        .seed = 123
    };
    
    MonteCarloEngine engine;
    auto result = engine.simulate_gbm(params);
    
    // Mean should be close to S0 * e^(mu*T)
    double expected = params.S0 * std::exp(params.mu * params.T);
    double tolerance = expected * 0.05;  // 5% tolerance
    
    EXPECT_NEAR(result.stats.mean_final_price, expected, tolerance);
}

TEST(MonteCarloTest, VaRCalculation) {
    // Use more data points for meaningful percentile calculations
    std::vector<double> returns = {
        -0.15, -0.10, -0.08, -0.06, -0.05, -0.04, -0.03, -0.02, -0.01, 0.00,
        0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15, 0.20
    };
    double portfolio_value = 100.0;
    
    auto var_result = MonteCarloEngine::calculate_var(returns, portfolio_value);
    
    // VaR should be positive (representing loss)
    EXPECT_GT(var_result.var_95, 0.0);
    // CVaR >= VaR (expected shortfall is at least as bad as VaR)
    EXPECT_GE(var_result.cvar_95, var_result.var_95);
    // VaR99 >= VaR95 (higher confidence means more extreme loss)
    EXPECT_GE(var_result.var_99, var_result.var_95);
}

TEST(MonteCarloTest, MultiThreading) {
    SimulationParams params{
        .S0 = 100.0,
        .mu = 0.10,
        .sigma = 0.25,
        .T = 1.0,
        .dt = 1.0 / 252.0,
        .num_paths = 10000,
        .num_steps = 252,
        .seed = 999
    };
    
    // Single-threaded
    MonteCarloEngine engine_single(1);
    auto start1 = std::chrono::steady_clock::now();
    auto result1 = engine_single.simulate_gbm(params);
    auto time1 = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start1
    ).count();
    
    // Multi-threaded
    MonteCarloEngine engine_multi(4);
    auto start2 = std::chrono::steady_clock::now();
    auto result2 = engine_multi.simulate_gbm(params);
    auto time2 = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start2
    ).count();
    
    // Multi-threaded should be faster
    EXPECT_LT(time2, time1);
    
    // Results should be similar (different seeds, so not exact)
    double diff = std::abs(result1.stats.mean_final_price - result2.stats.mean_final_price);
    EXPECT_LT(diff / result1.stats.mean_final_price, 0.1);  // Within 10%
}

TEST(MonteCarloTest, CorrelationMatrix) {
    CorrelationMatrix corr(3);
    
    corr.set(0, 1, 0.5);
    corr.set(0, 2, 0.3);
    corr.set(1, 2, 0.4);
    
    EXPECT_EQ(corr.get(0, 0), 1.0);
    EXPECT_EQ(corr.get(0, 1), 0.5);
    EXPECT_EQ(corr.get(1, 0), 0.5);  // Symmetric
    
    EXPECT_TRUE(corr.is_positive_definite());
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
