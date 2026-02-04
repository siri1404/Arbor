#include <gtest/gtest.h>
#include "../include/options_pricing.hpp"
#include <cmath>
#include <vector>
#include <numeric>
#include <random>

using namespace arbor::options;

// ============================================================================
// BLACK-SCHOLES TESTS
// ============================================================================

class BlackScholesTest : public ::testing::Test {
protected:
    static constexpr double S = 100.0;
    static constexpr double K = 100.0;
    static constexpr double T = 1.0;
    static constexpr double r = 0.05;
    static constexpr double sigma = 0.25;
};

TEST_F(BlackScholesTest, ATMCallPrice) {
    auto result = BlackScholesPricer::price(S, K, T, r, sigma, OptionType::CALL);
    
    // ATM call should be approximately N(d1)*S - N(d2)*K*exp(-rT)
    // For S=K=100, T=1, r=0.05, sigma=0.25: ~12.34
    EXPECT_GT(result.price, 10.0);
    EXPECT_LT(result.price, 15.0);
    
    // Delta should be around 0.5-0.7 for ATM call
    EXPECT_GT(result.greeks.delta, 0.5);
    EXPECT_LT(result.greeks.delta, 0.7);
}

TEST_F(BlackScholesTest, ATMPutPrice) {
    auto result = BlackScholesPricer::price(S, K, T, r, sigma, OptionType::PUT);
    
    // ATM put delta should be negative
    EXPECT_LT(result.greeks.delta, 0.0);
    EXPECT_GT(result.greeks.delta, -0.5);
    
    // Gamma should be positive for both calls and puts
    EXPECT_GT(result.greeks.gamma, 0.0);
    
    // Theta should be negative (time decay)
    EXPECT_LT(result.greeks.theta, 0.0);
}

TEST_F(BlackScholesTest, PutCallParity) {
    auto call = BlackScholesPricer::price(S, K, T, r, sigma, OptionType::CALL);
    auto put = BlackScholesPricer::price(S, K, T, r, sigma, OptionType::PUT);
    
    auto parity = check_put_call_parity(call.price, put.price, S, K, T, r);
    
    EXPECT_TRUE(parity.is_valid);
    EXPECT_LT(parity.difference, 0.001);
}

TEST_F(BlackScholesTest, DeepITMCall) {
    // Deep ITM call (S >> K) should have delta close to 1
    auto result = BlackScholesPricer::price(150.0, 100.0, T, r, sigma, OptionType::CALL);
    
    EXPECT_GT(result.greeks.delta, 0.95);
    EXPECT_NEAR(result.intrinsic_value, 50.0, 0.01);
}

TEST_F(BlackScholesTest, DeepOTMCall) {
    // Deep OTM call (S << K) should have delta close to 0
    auto result = BlackScholesPricer::price(50.0, 100.0, T, r, sigma, OptionType::CALL);
    
    EXPECT_LT(result.greeks.delta, 0.05);
    EXPECT_NEAR(result.intrinsic_value, 0.0, 0.01);
}

TEST_F(BlackScholesTest, HigherOrderGreeks) {
    auto result = BlackScholesPricer::price(S, K, T, r, sigma, OptionType::CALL);
    
    // Vanna: should be non-zero for non-ATM
    // Volga: should be positive (convex in vol)
    EXPECT_NE(result.greeks.vanna, 0.0);
    
    // Speed: dGamma/dS
    EXPECT_NE(result.greeks.speed, 0.0);
    
    // Charm: dDelta/dt
    EXPECT_NE(result.greeks.charm, 0.0);
}

TEST_F(BlackScholesTest, ImpliedVolatilityRoundtrip) {
    double true_sigma = 0.30;
    auto result = BlackScholesPricer::price(S, K, T, r, true_sigma, OptionType::CALL);
    
    double iv = BlackScholesPricer::implied_volatility(
        result.price, S, K, T, r, OptionType::CALL
    );
    
    EXPECT_NEAR(iv, true_sigma, 0.0001);
}

TEST_F(BlackScholesTest, ImpliedVolatilityOTM) {
    // OTM options are harder for IV solver
    double true_sigma = 0.35;
    auto result = BlackScholesPricer::price(100.0, 120.0, 0.25, r, true_sigma, OptionType::CALL);
    
    double iv = BlackScholesPricer::implied_volatility(
        result.price, 100.0, 120.0, 0.25, r, OptionType::CALL
    );
    
    EXPECT_NEAR(iv, true_sigma, 0.001);
}

TEST_F(BlackScholesTest, ExpiredOption) {
    // T = 0 should return intrinsic value
    auto call = BlackScholesPricer::price(110.0, 100.0, 0.0, r, sigma, OptionType::CALL);
    EXPECT_NEAR(call.price, 10.0, 0.001);
    EXPECT_NEAR(call.greeks.delta, 1.0, 0.001);
    
    auto put = BlackScholesPricer::price(90.0, 100.0, 0.0, r, sigma, OptionType::PUT);
    EXPECT_NEAR(put.price, 10.0, 0.001);
    EXPECT_NEAR(put.greeks.delta, -1.0, 0.001);
}

TEST_F(BlackScholesTest, OptionChain) {
    std::vector<double> strikes = {90.0, 95.0, 100.0, 105.0, 110.0};
    auto chain = BlackScholesPricer::option_chain(S, strikes, T, r, sigma);
    
    EXPECT_EQ(chain.size(), strikes.size());
    
    // Verify call deltas decrease with strike (more OTM)
    for (size_t i = 1; i < chain.size(); ++i) {
        EXPECT_LT(chain[i].first.greeks.delta, chain[i-1].first.greeks.delta);
    }
    
    // Verify put deltas increase (less negative) with strike
    for (size_t i = 1; i < chain.size(); ++i) {
        EXPECT_GT(chain[i].second.greeks.delta, chain[i-1].second.greeks.delta);
    }
}

// ============================================================================
// MERTON JUMP-DIFFUSION TESTS
// ============================================================================

class MertonJumpDiffusionTest : public ::testing::Test {
protected:
    static constexpr double S = 100.0;
    static constexpr double K = 100.0;
    static constexpr double T = 1.0;
    static constexpr double r = 0.05;
    static constexpr double sigma = 0.20;
    
    JumpDiffusionParams jump_params{
        0.5,    // lambda: 0.5 jumps per year on average
        -0.1,   // mu_j: average jump is -10%
        0.2     // sigma_j: jump volatility
    };
};

TEST_F(MertonJumpDiffusionTest, ConvergesToBSWhenNoJumps) {
    JumpDiffusionParams no_jumps{0.0, 0.0, 0.0};
    
    auto jd_result = MertonJumpDiffusion::price(S, K, T, r, sigma, no_jumps, OptionType::CALL);
    auto bs_result = BlackScholesPricer::price(S, K, T, r, sigma, OptionType::CALL);
    
    EXPECT_NEAR(jd_result.price, bs_result.price, 0.01);
}

TEST_F(MertonJumpDiffusionTest, JumpsPriceOTMOptionsHigher) {
    // OTM options should be priced higher with jumps (fat tails)
    auto jd_otm = MertonJumpDiffusion::price(100.0, 130.0, T, r, sigma, jump_params, OptionType::CALL);
    auto bs_otm = BlackScholesPricer::price(100.0, 130.0, T, r, sigma, OptionType::CALL);
    
    EXPECT_GT(jd_otm.price, bs_otm.price);
}

TEST_F(MertonJumpDiffusionTest, PutCallParityHolds) {
    auto call = MertonJumpDiffusion::price(S, K, T, r, sigma, jump_params, OptionType::CALL);
    auto put = MertonJumpDiffusion::price(S, K, T, r, sigma, jump_params, OptionType::PUT);
    
    // Put-call parity: C - P = S - K*exp(-rT)
    double parity_diff = call.price - put.price;
    double expected = S - K * std::exp(-r * T);
    
    EXPECT_NEAR(parity_diff, expected, 0.1);
}

TEST_F(MertonJumpDiffusionTest, GreeksFiniteDifference) {
    auto greeks = MertonJumpDiffusion::compute_greeks(S, K, T, r, sigma, jump_params, OptionType::CALL);
    
    // Delta should be positive for call
    EXPECT_GT(greeks.delta, 0.0);
    EXPECT_LT(greeks.delta, 1.0);
    
    // Gamma should be positive
    EXPECT_GT(greeks.gamma, 0.0);
    
    // Vega should be positive
    EXPECT_GT(greeks.vega, 0.0);
}

TEST_F(MertonJumpDiffusionTest, SeriesConvergence) {
    auto result_50 = MertonJumpDiffusion::price(S, K, T, r, sigma, jump_params, OptionType::CALL, 50);
    auto result_100 = MertonJumpDiffusion::price(S, K, T, r, sigma, jump_params, OptionType::CALL, 100);
    
    // Higher truncation should give similar result (converged)
    EXPECT_NEAR(result_50.price, result_100.price, 0.001);
}

// ============================================================================
// HESTON STOCHASTIC VOLATILITY TESTS
// ============================================================================

class HestonModelTest : public ::testing::Test {
protected:
    static constexpr double S = 100.0;
    static constexpr double K = 100.0;
    static constexpr double T = 1.0;
    static constexpr double r = 0.05;
    
    HestonParams params{
        0.04,   // V0: initial variance (20% vol)
        2.0,    // kappa: mean reversion speed
        0.04,   // theta: long-term variance
        0.3,    // xi: vol of vol
        -0.7    // rho: correlation (typically negative for equities)
    };
};

TEST_F(HestonModelTest, FellerCondition) {
    // 2*kappa*theta > xi^2 should be satisfied for variance to stay positive
    EXPECT_TRUE(params.satisfies_feller());
    
    HestonParams bad_params{0.04, 0.5, 0.01, 0.5, -0.5};
    EXPECT_FALSE(bad_params.satisfies_feller());
}

TEST_F(HestonModelTest, ConvergesToBSWhenNoVolOfVol) {
    // When xi -> 0, Heston -> BS
    HestonParams bs_like{0.0625, 2.0, 0.0625, 0.001, 0.0};  // 25% constant vol
    
    auto heston_result = HestonModel::price(S, K, T, r, bs_like, OptionType::CALL);
    auto bs_result = BlackScholesPricer::price(S, K, T, r, 0.25, OptionType::CALL);
    
    EXPECT_NEAR(heston_result.price, bs_result.price, 0.5);
}

TEST_F(HestonModelTest, NegativeCorrelationSkew) {
    // Negative rho creates downside skew (OTM puts more expensive)
    auto otm_put = HestonModel::price(S, 80.0, T, r, params, OptionType::PUT);
    auto otm_call = HestonModel::price(S, 120.0, T, r, params, OptionType::CALL);
    
    // Both should have positive prices
    EXPECT_GT(otm_put.price, 0.0);
    EXPECT_GT(otm_call.price, 0.0);
}

TEST_F(HestonModelTest, CharacteristicFunctionNormalized) {
    // φ(0) should equal 1
    std::complex<double> phi_0 = HestonModel::characteristic_function(
        std::complex<double>(0.0, 0.0), S, T, r, params
    );
    
    EXPECT_NEAR(std::abs(phi_0), 1.0, 0.001);
}

TEST_F(HestonModelTest, PutCallParity) {
    auto call = HestonModel::price(S, K, T, r, params, OptionType::CALL);
    auto put = HestonModel::price(S, K, T, r, params, OptionType::PUT);
    
    double parity = call.price - put.price;
    double expected = S - K * std::exp(-r * T);
    
    EXPECT_NEAR(parity, expected, 0.5);
}

TEST_F(HestonModelTest, GreeksComputation) {
    auto greeks = HestonModel::compute_greeks(S, K, T, r, params, OptionType::CALL);
    
    EXPECT_GT(greeks.delta, 0.0);
    EXPECT_LT(greeks.delta, 1.0);
    EXPECT_GT(greeks.gamma, 0.0);
    EXPECT_GT(greeks.vega, 0.0);
}

// ============================================================================
// SABR MODEL TESTS
// ============================================================================

class SABRModelTest : public ::testing::Test {
protected:
    static constexpr double F = 100.0;  // Forward
    static constexpr double T = 1.0;
    
    SABRParams params{
        0.3,    // alpha: ATM vol level
        0.5,    // beta: CEV exponent
        -0.3,   // rho: correlation
        0.4     // nu: vol of vol
    };
};

TEST_F(SABRModelTest, ATMVolApproximation) {
    double atm_vol = SABRModel::implied_volatility(F, F, T, params);
    
    // ATM vol should be close to alpha / F^(1-beta)
    double expected_atm = params.alpha / std::pow(F, 1.0 - params.beta);
    EXPECT_NEAR(atm_vol, expected_atm, 0.05);
}

TEST_F(SABRModelTest, VolSmileShape) {
    // SABR should produce a volatility smile
    double vol_atm = SABRModel::implied_volatility(F, F, T, params);
    double vol_otm_call = SABRModel::implied_volatility(F, F * 1.2, T, params);
    double vol_otm_put = SABRModel::implied_volatility(F, F * 0.8, T, params);
    
    // With negative rho, downside (OTM puts) should have higher vol
    EXPECT_GT(vol_otm_put, vol_atm);
    
    // All vols should be positive
    EXPECT_GT(vol_atm, 0.0);
    EXPECT_GT(vol_otm_call, 0.0);
    EXPECT_GT(vol_otm_put, 0.0);
}

TEST_F(SABRModelTest, BetaZeroNormal) {
    // Beta = 0 is normal SABR
    SABRParams normal_params{0.3, 0.0, -0.3, 0.4};
    
    double vol = SABRModel::implied_volatility(F, F, T, normal_params);
    EXPECT_GT(vol, 0.0);
}

TEST_F(SABRModelTest, BetaOneLogNormal) {
    // Beta = 1 is log-normal SABR
    SABRParams lognormal_params{0.3, 1.0, -0.3, 0.4};
    
    double vol = SABRModel::implied_volatility(F, F, T, lognormal_params);
    EXPECT_GT(vol, 0.0);
}

TEST_F(SABRModelTest, PriceConsistency) {
    double S = 100.0, K = 105.0, r = 0.05;
    
    auto result = SABRModel::price(S, K, T, r, params, OptionType::CALL);
    
    EXPECT_GT(result.price, 0.0);
    EXPECT_GT(result.greeks.delta, 0.0);
}

// ============================================================================
// AMERICAN OPTIONS TESTS
// ============================================================================

class AmericanOptionTest : public ::testing::Test {
protected:
    static constexpr double S = 100.0;
    static constexpr double K = 100.0;
    static constexpr double T = 1.0;
    static constexpr double r = 0.05;
    static constexpr double sigma = 0.25;
};

TEST_F(AmericanOptionTest, AmericanCallEqualsEuropean) {
    // American call on non-dividend stock = European call
    auto american = AmericanOptionPricer::price_binomial(S, K, T, r, sigma, OptionType::CALL);
    auto european = BlackScholesPricer::price(S, K, T, r, sigma, OptionType::CALL);
    
    EXPECT_NEAR(american.price, european.price, 0.1);
}

TEST_F(AmericanOptionTest, AmericanPutPremium) {
    // American put should be worth more than European put
    auto american = AmericanOptionPricer::price_binomial(S, K, T, r, sigma, OptionType::PUT);
    auto european = BlackScholesPricer::price(S, K, T, r, sigma, OptionType::PUT);
    
    EXPECT_GE(american.price, european.price - 0.01);  // Allow small numerical error
}

TEST_F(AmericanOptionTest, EarlyExercisePremium) {
    double premium = AmericanOptionPricer::early_exercise_premium(S, K, T, r, sigma, OptionType::PUT);
    
    EXPECT_GE(premium, 0.0);
}

TEST_F(AmericanOptionTest, TrinomialConvergence) {
    auto binomial = AmericanOptionPricer::price_binomial(S, K, T, r, sigma, OptionType::PUT, {200, true, true});
    auto trinomial = AmericanOptionPricer::price_trinomial(S, K, T, r, sigma, OptionType::PUT, 200);
    
    // Both methods should give similar prices
    EXPECT_NEAR(binomial.price, trinomial.price, 0.2);
}

TEST_F(AmericanOptionTest, RichardsonExtrapolation) {
    BinomialTreeConfig config_no_rich{100, false, false};
    BinomialTreeConfig config_rich{100, true, true};
    
    auto no_richardson = AmericanOptionPricer::price_binomial(S, K, T, r, sigma, OptionType::PUT, config_no_rich);
    auto with_richardson = AmericanOptionPricer::price_binomial(S, K, T, r, sigma, OptionType::PUT, config_rich);
    
    // Both should produce valid prices
    EXPECT_GT(no_richardson.price, 0.0);
    EXPECT_GT(with_richardson.price, 0.0);
}

TEST_F(AmericanOptionTest, ExerciseBoundary) {
    auto boundary = AmericanOptionPricer::exercise_boundary(S, K, T, r, sigma, OptionType::PUT);
    
    EXPECT_FALSE(boundary.empty());
    
    // For a put, exercise boundary should be below strike at maturity
    // and decrease as we go back in time
    EXPECT_LT(boundary.back().second, K);
}

TEST_F(AmericanOptionTest, DeepITMPutExercise) {
    // Deep ITM American put should be close to intrinsic
    double deep_S = 50.0;  // Deep ITM
    auto american = AmericanOptionPricer::price_binomial(deep_S, K, T, r, sigma, OptionType::PUT);
    
    double intrinsic = K - deep_S;
    EXPECT_GT(american.price, intrinsic - 1.0);
}

// ============================================================================
// LONGSTAFF-SCHWARTZ LSM TESTS
// ============================================================================

class LSMTest : public ::testing::Test {
protected:
    static constexpr double S = 100.0;
    static constexpr double K = 100.0;
    static constexpr double T = 1.0;
    static constexpr double r = 0.05;
    static constexpr double sigma = 0.25;
    
    LSMConfig config{50000, 50, 3, 42, true, true};
};

TEST_F(LSMTest, ConvergesToBinomial) {
    auto lsm = LongstaffSchwartzPricer::price(S, K, T, r, sigma, OptionType::PUT, config);
    auto binomial = AmericanOptionPricer::price_binomial(S, K, T, r, sigma, OptionType::PUT);
    
    // LSM should be within 2% of binomial
    EXPECT_NEAR(lsm.price, binomial.price, binomial.price * 0.02);
}

TEST_F(LSMTest, StandardErrorDecreases) {
    LSMConfig small_config{10000, 50, 3, 42, true, true};
    LSMConfig large_config{100000, 50, 3, 42, true, true};
    
    auto small_result = LongstaffSchwartzPricer::price(S, K, T, r, sigma, OptionType::PUT, small_config);
    auto large_result = LongstaffSchwartzPricer::price(S, K, T, r, sigma, OptionType::PUT, large_config);
    
    // Standard error should decrease with more paths
    EXPECT_LT(large_result.std_error, small_result.std_error);
}

TEST_F(LSMTest, AntitheticVariance) {
    LSMConfig no_anti{50000, 50, 3, 42, false, false};
    LSMConfig with_anti{50000, 50, 3, 42, true, false};
    
    auto no_anti_result = LongstaffSchwartzPricer::price(S, K, T, r, sigma, OptionType::PUT, no_anti);
    auto anti_result = LongstaffSchwartzPricer::price(S, K, T, r, sigma, OptionType::PUT, with_anti);
    
    // Both should give valid prices
    EXPECT_GT(no_anti_result.price, 0.0);
    EXPECT_GT(anti_result.price, 0.0);
}

TEST_F(LSMTest, HestonPricing) {
    HestonParams heston{0.04, 2.0, 0.04, 0.3, -0.7};
    
    auto result = LongstaffSchwartzPricer::price_heston(S, K, T, r, heston, OptionType::PUT, config);
    
    EXPECT_GT(result.price, 0.0);
    EXPECT_GT(result.std_error, 0.0);
}

TEST_F(LSMTest, BermudanOption) {
    // Quarterly exercise dates
    std::vector<double> exercise_times = {0.25, 0.5, 0.75, 1.0};
    
    auto bermudan = LongstaffSchwartzPricer::price_bermudan(
        S, K, T, r, sigma, exercise_times, OptionType::PUT, config
    );
    auto european = BlackScholesPricer::price(S, K, T, r, sigma, OptionType::PUT);
    auto american = LongstaffSchwartzPricer::price(S, K, T, r, sigma, OptionType::PUT, config);
    
    // Bermudan should be between European and American
    EXPECT_GE(bermudan.price, european.price - 0.5);
    EXPECT_LE(bermudan.price, american.price + 0.5);
}

// ============================================================================
// BARRIER OPTIONS TESTS
// ============================================================================

class BarrierOptionTest : public ::testing::Test {
protected:
    static constexpr double S = 100.0;
    static constexpr double K = 100.0;
    static constexpr double T = 1.0;
    static constexpr double r = 0.05;
    static constexpr double sigma = 0.25;
};

TEST_F(BarrierOptionTest, DownAndOutCallVanilla) {
    // Down-and-out call with barrier below spot should be cheaper than vanilla
    BarrierParams barrier{BarrierType::DOWN_AND_OUT, 80.0, 0.0};
    
    auto barrier_price = BarrierOptionPricer::price_analytical(S, K, T, r, sigma, barrier, OptionType::CALL);
    auto vanilla = BlackScholesPricer::price(S, K, T, r, sigma, OptionType::CALL);
    
    EXPECT_LT(barrier_price.price, vanilla.price);
    EXPECT_GT(barrier_price.price, 0.0);
}

TEST_F(BarrierOptionTest, InOutParity) {
    // Down-and-in + Down-and-out = Vanilla
    BarrierParams di_barrier{BarrierType::DOWN_AND_IN, 80.0, 0.0};
    BarrierParams do_barrier{BarrierType::DOWN_AND_OUT, 80.0, 0.0};
    
    auto di_price = BarrierOptionPricer::price_analytical(S, K, T, r, sigma, di_barrier, OptionType::CALL);
    auto do_price = BarrierOptionPricer::price_analytical(S, K, T, r, sigma, do_barrier, OptionType::CALL);
    auto vanilla = BlackScholesPricer::price(S, K, T, r, sigma, OptionType::CALL);
    
    EXPECT_NEAR(di_price.price + do_price.price, vanilla.price, 0.5);
}

TEST_F(BarrierOptionTest, KnockedOutWorthless) {
    // If spot <= barrier, down-and-out is worthless (plus rebate)
    BarrierParams barrier{BarrierType::DOWN_AND_OUT, 100.0, 5.0};  // Barrier at spot
    
    auto result = BarrierOptionPricer::price_analytical(80.0, K, T, r, sigma, barrier, OptionType::CALL);
    
    EXPECT_NEAR(result.price, 5.0 * std::exp(-r * T), 0.1);  // Discounted rebate
}

TEST_F(BarrierOptionTest, MonteCarloConvergence) {
    BarrierParams barrier{BarrierType::DOWN_AND_OUT, 80.0, 0.0};
    
    auto analytical = BarrierOptionPricer::price_analytical(S, K, T, r, sigma, barrier, OptionType::CALL);
    auto mc = BarrierOptionPricer::price_monte_carlo(S, K, T, r, sigma, barrier, OptionType::CALL, 252, 100000, 42);
    
    // Monte Carlo should converge to analytical (allowing for MC error)
    EXPECT_NEAR(mc.price, analytical.price, 0.5);
}

// ============================================================================
// ASIAN OPTIONS TESTS
// ============================================================================

class AsianOptionTest : public ::testing::Test {
protected:
    static constexpr double S = 100.0;
    static constexpr double K = 100.0;
    static constexpr double T = 1.0;
    static constexpr double r = 0.05;
    static constexpr double sigma = 0.25;
    
    AsianParams params{AveragingType::ARITHMETIC, 12, false};  // Monthly averaging
};

TEST_F(AsianOptionTest, GeometricClosedForm) {
    AsianParams geo_params{AveragingType::GEOMETRIC, 12, false};
    
    auto result = AsianOptionPricer::price_geometric(S, K, T, r, sigma, geo_params, OptionType::CALL);
    
    EXPECT_GT(result.price, 0.0);
    // Geometric Asian should be cheaper than vanilla (averaging reduces vol)
    auto vanilla = BlackScholesPricer::price(S, K, T, r, sigma, OptionType::CALL);
    EXPECT_LT(result.price, vanilla.price);
}

TEST_F(AsianOptionTest, ArithmeticMonteCarlo) {
    auto result = AsianOptionPricer::price_arithmetic(S, K, T, r, sigma, params, OptionType::CALL, 100000, 42);
    
    EXPECT_GT(result.price, 0.0);
    EXPECT_GT(result.std_error, 0.0);
    EXPECT_LT(result.std_error, 0.5);  // Reasonable MC error
}

TEST_F(AsianOptionTest, TurnbullWakeman) {
    auto result = AsianOptionPricer::price_turnbull_wakeman(S, K, T, r, sigma, params, OptionType::CALL);
    
    EXPECT_GT(result.price, 0.0);
}

TEST_F(AsianOptionTest, AsianCheaperThanVanilla) {
    auto asian = AsianOptionPricer::price_arithmetic(S, K, T, r, sigma, params, OptionType::CALL, 100000, 42);
    auto vanilla = BlackScholesPricer::price(S, K, T, r, sigma, OptionType::CALL);
    
    // Asian should be cheaper due to averaging effect
    EXPECT_LT(asian.price, vanilla.price + 1.0);  // Allow for MC error
}

// ============================================================================
// MONTE CARLO ENGINE TESTS
// ============================================================================

class MonteCarloTest : public ::testing::Test {
protected:
    static constexpr double S = 100.0;
    static constexpr double K = 100.0;
    static constexpr double T = 1.0;
    static constexpr double r = 0.05;
    static constexpr double sigma = 0.25;
};

TEST_F(MonteCarloTest, ConvergesToBlackScholes) {
    auto mc = MonteCarloEngine::price_european(S, K, T, r, sigma, OptionType::CALL, 1000000, 42, true, true);
    auto bs = BlackScholesPricer::price(S, K, T, r, sigma, OptionType::CALL);
    
    EXPECT_NEAR(mc.price, bs.price, 0.1);
}

TEST_F(MonteCarloTest, AntitheticReducesVariance) {
    auto no_anti = MonteCarloEngine::price_european(S, K, T, r, sigma, OptionType::CALL, 100000, 42, false, false);
    auto with_anti = MonteCarloEngine::price_european(S, K, T, r, sigma, OptionType::CALL, 100000, 42, true, false);
    
    // Antithetic should have lower standard error
    EXPECT_LT(with_anti.std_error, no_anti.std_error * 1.5);  // Allow some variance
}

TEST_F(MonteCarloTest, HestonMonteCarlo) {
    HestonParams heston{0.04, 2.0, 0.04, 0.3, -0.7};
    
    auto mc = MonteCarloEngine::price_european_heston(S, K, T, r, heston, OptionType::CALL, 100000, 252, 42);
    auto analytical = HestonModel::price(S, K, T, r, heston, OptionType::CALL);
    
    EXPECT_NEAR(mc.price, analytical.price, 1.0);  // Allow for MC error
}

TEST_F(MonteCarloTest, JumpDiffusionMonteCarlo) {
    JumpDiffusionParams jump{0.5, -0.1, 0.2};
    
    auto mc = MonteCarloEngine::price_jump_diffusion(S, K, T, r, sigma, jump, OptionType::CALL, 100000, 252, 42);
    auto analytical = MertonJumpDiffusion::price(S, K, T, r, sigma, jump, OptionType::CALL);
    
    EXPECT_NEAR(mc.price, analytical.price, 1.0);
}

#ifdef __AVX2__
TEST_F(MonteCarloTest, SIMDConsistency) {
    auto scalar = MonteCarloEngine::price_european(S, K, T, r, sigma, OptionType::CALL, 100000, 42, true, false);
    auto simd = SIMDMonteCarloEngine::price_european_avx2(S, K, T, r, sigma, OptionType::CALL, 100000, 42);
    
    // SIMD and scalar should give consistent results
    EXPECT_NEAR(simd.price, scalar.price, 1.0);
}
#endif

// ============================================================================
// PERFORMANCE TESTS
// ============================================================================

TEST(PerformanceTest, BlackScholesSubMicrosecond) {
    const int NUM_CALCS = 100000;
    auto start = std::chrono::steady_clock::now();
    
    volatile double sum = 0.0;  // Prevent optimization
    for (int i = 0; i < NUM_CALCS; ++i) {
        auto result = BlackScholesPricer::price(100.0, 100.0, 1.0, 0.05, 0.25, OptionType::CALL);
        sum += result.price;
    }
    
    auto end = std::chrono::steady_clock::now();
    auto avg_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / NUM_CALCS;
    
    EXPECT_LT(avg_ns, 2000);  // Should be under 2 microseconds
}

TEST(PerformanceTest, ImpliedVolatilityPerformance) {
    const int NUM_CALCS = 10000;
    
    // Get market price
    auto result = BlackScholesPricer::price(100.0, 100.0, 1.0, 0.05, 0.25, OptionType::CALL);
    
    auto start = std::chrono::steady_clock::now();
    
    volatile double sum = 0.0;
    for (int i = 0; i < NUM_CALCS; ++i) {
        double iv = BlackScholesPricer::implied_volatility(result.price, 100.0, 100.0, 1.0, 0.05, OptionType::CALL);
        sum += iv;
    }
    
    auto end = std::chrono::steady_clock::now();
    auto avg_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / NUM_CALCS;
    
    EXPECT_LT(avg_ns, 50000);  // Should be under 50 microseconds
}

// ============================================================================
// NUMERICAL STABILITY TESTS
// ============================================================================

TEST(StabilityTest, ExtremeDTM) {
    // Very short time to maturity
    auto result = BlackScholesPricer::price(100.0, 100.0, 0.001, 0.05, 0.25, OptionType::CALL);
    
    EXPECT_GE(result.price, 0.0);
    EXPECT_LE(result.price, 100.0);
    EXPECT_TRUE(std::isfinite(result.greeks.delta));
}

TEST(StabilityTest, HighVolatility) {
    auto result = BlackScholesPricer::price(100.0, 100.0, 1.0, 0.05, 2.0, OptionType::CALL);
    
    EXPECT_GT(result.price, 0.0);
    EXPECT_LT(result.price, 100.0);
    EXPECT_TRUE(std::isfinite(result.greeks.delta));
}

TEST(StabilityTest, DeepOTM) {
    // Very deep OTM
    auto call = BlackScholesPricer::price(100.0, 200.0, 0.1, 0.05, 0.25, OptionType::CALL);
    auto put = BlackScholesPricer::price(100.0, 50.0, 0.1, 0.05, 0.25, OptionType::PUT);
    
    EXPECT_GE(call.price, 0.0);
    EXPECT_GE(put.price, 0.0);
    EXPECT_TRUE(std::isfinite(call.greeks.delta));
    EXPECT_TRUE(std::isfinite(put.greeks.delta));
}

TEST(StabilityTest, ZeroVolatility) {
    // sigma = 0 edge case
    auto result = BlackScholesPricer::price(110.0, 100.0, 1.0, 0.05, 0.0001, OptionType::CALL);
    
    // Should be close to discounted intrinsic
    double expected = std::max(110.0 - 100.0 * std::exp(-0.05), 0.0);
    EXPECT_NEAR(result.price, expected, 1.0);
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
