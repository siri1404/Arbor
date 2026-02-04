#include <gtest/gtest.h>
#include "../include/options_pricing.hpp"
#include <cmath>

using namespace arbor::options;

TEST(BlackScholesTest, ATMCallPrice) {
    auto result = BlackScholesPricer::price(
        100.0,  // S
        100.0,  // K
        1.0,    // T
        0.05,   // r
        0.25,   // sigma
        OptionType::CALL
    );
    
    // ATM call should have reasonable price
    EXPECT_GT(result.price, 8.0);
    EXPECT_LT(result.price, 15.0);
    
    // ATM call delta should be around 0.5-0.6
    EXPECT_GT(result.greeks.delta, 0.5);
    EXPECT_LT(result.greeks.delta, 0.7);
    
    // Gamma should be positive
    EXPECT_GT(result.greeks.gamma, 0.0);
    
    // Theta should be negative (time decay)
    EXPECT_LT(result.greeks.theta, 0.0);
}

TEST(BlackScholesTest, PutCallParity) {
    double S = 100.0, K = 100.0, T = 1.0, r = 0.05, sigma = 0.25;
    
    auto call = BlackScholesPricer::price(S, K, T, r, sigma, OptionType::CALL);
    auto put = BlackScholesPricer::price(S, K, T, r, sigma, OptionType::PUT);
    
    auto parity = check_put_call_parity(call.price, put.price, S, K, T, r);
    
    EXPECT_TRUE(parity.is_valid);
    EXPECT_LT(parity.difference, 0.01);  // Within 1 cent
}

TEST(BlackScholesTest, ImpliedVolatility) {
    double S = 100.0, K = 100.0, T = 1.0, r = 0.05, true_sigma = 0.25;
    
    // Calculate theoretical price
    auto result = BlackScholesPricer::price(S, K, T, r, true_sigma, OptionType::CALL);
    
    // Solve for IV
    double iv = BlackScholesPricer::implied_volatility(
        result.price, S, K, T, r, OptionType::CALL
    );
    
    // Should recover original volatility
    EXPECT_NEAR(iv, true_sigma, 0.0001);
}

TEST(BlackScholesTest, IntrinsicValue) {
    // Deep ITM call
    auto call = BlackScholesPricer::price(150.0, 100.0, 1.0, 0.05, 0.25, OptionType::CALL);
    EXPECT_NEAR(call.intrinsic_value, 50.0, 0.01);
    EXPECT_GT(call.time_value, 0.0);
    
    // Deep ITM put
    auto put = BlackScholesPricer::price(100.0, 150.0, 1.0, 0.05, 0.25, OptionType::PUT);
    EXPECT_NEAR(put.intrinsic_value, 50.0, 0.01);
}

TEST(BlackScholesTest, Performance) {
    const int NUM_CALCS = 10000;
    
    auto start = std::chrono::steady_clock::now();
    
    for (int i = 0; i < NUM_CALCS; ++i) {
        BlackScholesPricer::price(100.0, 100.0, 1.0, 0.05, 0.25, OptionType::CALL);
    }
    
    auto end = std::chrono::steady_clock::now();
    auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    auto avg_ns = duration_ns / NUM_CALCS;
    
    // Should be under 10 microseconds per pricing
    EXPECT_LT(avg_ns, 10000);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
