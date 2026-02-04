#pragma once

#include <cmath>
#include <numbers>
#include <array>
#include <chrono>
#include <algorithm>

namespace arbor::options {

using namespace std::numbers;  // For pi

enum class OptionType : uint8_t { CALL = 0, PUT = 1 };

// Options Greeks
struct Greeks {
    double delta;    // ∂V/∂S
    double gamma;    // ∂²V/∂S²
    double theta;    // ∂V/∂t (per day)
    double vega;     // ∂V/∂σ (per 1%)
    double rho;      // ∂V/∂r (per 1%)
};

// Complete pricing result
struct PricingResult {
    double price;
    double intrinsic_value;
    double time_value;
    Greeks greeks;
    int64_t calc_time_ns;
};

// Black-Scholes pricer with SIMD-friendly vectorization
class BlackScholesPricer {
public:
    BlackScholesPricer() = default;
    
    // Core pricing function - optimized for sub-microsecond execution
    [[nodiscard]] static PricingResult price(
        double S,           // Spot price
        double K,           // Strike price
        double T,           // Time to expiry (years)
        double r,           // Risk-free rate
        double sigma,       // Volatility
        OptionType type
    ) noexcept;
    
    // Implied volatility solver - Newton-Raphson with analytic vega
    [[nodiscard]] static double implied_volatility(
        double market_price,
        double S,
        double K,
        double T,
        double r,
        OptionType type,
        double initial_guess = 0.3,
        double tolerance = 1e-8,
        int max_iterations = 100
    ) noexcept;
    
    // Vectorized option chain calculation for multiple strikes
    [[nodiscard]] static std::vector<std::pair<PricingResult, PricingResult>> 
    option_chain(
        double S,
        const std::vector<double>& strikes,
        double T,
        double r,
        double sigma
    ) noexcept;
    
private:
    // Fast cumulative normal distribution - Abramowitz & Stegun approximation
    [[nodiscard]] static inline double norm_cdf(double x) noexcept {
        static constexpr double a1 =  0.254829592;
        static constexpr double a2 = -0.284496736;
        static constexpr double a3 =  1.421413741;
        static constexpr double a4 = -1.453152027;
        static constexpr double a5 =  1.061405429;
        static constexpr double p  =  0.3275911;
        
        const int sign = (x < 0) ? -1 : 1;
        x = std::abs(x) * inv_sqrt2;
        
        const double t = 1.0 / (1.0 + p * x);
        const double t2 = t * t;
        const double t3 = t2 * t;
        const double t4 = t3 * t;
        const double t5 = t4 * t;
        
        const double y = 1.0 - (((((a5 * t5 + a4 * t4) + a3 * t3) + a2 * t2) + a1 * t) 
                         * std::exp(-x * x));
        
        return 0.5 * (1.0 + sign * y);
    }
    
    // Fast probability density function
    [[nodiscard]] static inline double norm_pdf(double x) noexcept {
        static constexpr double inv_sqrt_2pi = 0.3989422804014327;  // 1/sqrt(2π)
        return inv_sqrt_2pi * std::exp(-0.5 * x * x);
    }
    
    // Calculate d1 and d2 - memoized within pricing call
    static inline void calc_d1_d2(double S, double K, double T, double r, double sigma,
                                   double& d1_out, double& d2_out) noexcept {
        const double sqrt_T = std::sqrt(T);
        const double sigma_sqrt_T = sigma * sqrt_T;
        const double log_S_K = std::log(S / K);
        const double r_plus_half_sigma_sq = r + 0.5 * sigma * sigma;
        
        d1_out = (log_S_K + r_plus_half_sigma_sq * T) / sigma_sqrt_T;
        d2_out = d1_out - sigma_sqrt_T;
    }
    
    static constexpr double inv_sqrt2 = 0.7071067811865475;  // 1/√2
};

// Put-Call parity validator
struct PutCallParityCheck {
    bool is_valid;
    double difference;
    double expected;
};

[[nodiscard]] PutCallParityCheck check_put_call_parity(
    double call_price,
    double put_price,
    double S,
    double K,
    double T,
    double r
) noexcept;

} // namespace arbor::options
