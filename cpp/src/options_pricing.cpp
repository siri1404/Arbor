#include "options_pricing.hpp"
#include <algorithm>
#include <chrono>

namespace arbor::options {

PricingResult BlackScholesPricer::price(
    double S, double K, double T, double r, double sigma, OptionType type
) noexcept {
    const auto start = std::chrono::steady_clock::now();
    
    PricingResult result{};
    
    // Handle edge case: expired option
    if (T <= 0.0) {
        if (type == OptionType::CALL) {
            result.intrinsic_value = std::max(S - K, 0.0);
            result.greeks.delta = (S > K) ? 1.0 : 0.0;
        } else {
            result.intrinsic_value = std::max(K - S, 0.0);
            result.greeks.delta = (S < K) ? -1.0 : 0.0;
        }
        result.price = result.intrinsic_value;
        result.time_value = 0.0;
        
        const auto end = std::chrono::steady_clock::now();
        result.calc_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        return result;
    }
    
    // Calculate d1 and d2
    double d1, d2;
    calc_d1_d2(S, K, T, r, sigma, d1, d2);
    
    const double sqrt_T = std::sqrt(T);
    const double discount_factor = std::exp(-r * T);
    
    // Calculate price based on option type
    double price_value;
    double delta_value;
    double rho_value;
    
    if (type == OptionType::CALL) {
        const double Nd1 = norm_cdf(d1);
        const double Nd2 = norm_cdf(d2);
        
        price_value = S * Nd1 - K * discount_factor * Nd2;
        delta_value = Nd1;
        rho_value = K * T * discount_factor * Nd2 * 0.01;  // Per 1% change
    } else {
        const double N_minus_d1 = norm_cdf(-d1);
        const double N_minus_d2 = norm_cdf(-d2);
        
        price_value = K * discount_factor * N_minus_d2 - S * N_minus_d1;
        delta_value = N_minus_d1 - 1.0;
        rho_value = -K * T * discount_factor * N_minus_d2 * 0.01;
    }
    
    // Calculate gamma (same for calls and puts)
    const double pdf_d1 = norm_pdf(d1);
    const double gamma_value = pdf_d1 / (S * sigma * sqrt_T);
    
    // Calculate theta (time decay)
    const double theta_term1 = -(S * pdf_d1 * sigma) / (2.0 * sqrt_T);
    double theta_value;
    
    if (type == OptionType::CALL) {
        const double Nd2 = norm_cdf(d2);
        theta_value = (theta_term1 - r * K * discount_factor * Nd2) / 365.0;
    } else {
        const double N_minus_d2 = norm_cdf(-d2);
        theta_value = (theta_term1 + r * K * discount_factor * N_minus_d2) / 365.0;
    }
    
    // Calculate vega (same for calls and puts)
    const double vega_value = S * sqrt_T * pdf_d1 * 0.01;  // Per 1% change
    
    // Calculate intrinsic and time value
    if (type == OptionType::CALL) {
        result.intrinsic_value = std::max(S - K, 0.0);
    } else {
        result.intrinsic_value = std::max(K - S, 0.0);
    }
    
    result.price = price_value;
    result.time_value = price_value - result.intrinsic_value;
    result.greeks = Greeks{
        .delta = delta_value,
        .gamma = gamma_value,
        .theta = theta_value,
        .vega = vega_value,
        .rho = rho_value
    };
    
    const auto end = std::chrono::steady_clock::now();
    result.calc_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    
    return result;
}

double BlackScholesPricer::implied_volatility(
    double market_price, double S, double K, double T, double r, 
    OptionType type, double initial_guess, double tolerance, int max_iterations
) noexcept {
    // Brenner-Subrahmanyam initial approximation
    double sigma = std::sqrt(2.0 * pi / T) * (market_price / S);
    sigma = std::clamp(sigma, 0.001, 5.0);
    
    if (initial_guess > 0.0) {
        sigma = initial_guess;
    }
    
    // Newton-Raphson iteration
    for (int i = 0; i < max_iterations; ++i) {
        const auto result = price(S, K, T, r, sigma, type);
        const double price_error = result.price - market_price;
        
        if (std::abs(price_error) < tolerance) {
            return sigma;
        }
        
        const double vega_100 = result.greeks.vega * 100.0;  // Convert from per 1% to absolute
        
        if (std::abs(vega_100) < 1e-10) {
            break;  // Vega too small, can't converge
        }
        
        // Newton-Raphson update
        sigma -= price_error / vega_100;
        sigma = std::clamp(sigma, 0.001, 10.0);
    }
    
    return sigma;
}

std::vector<std::pair<PricingResult, PricingResult>> 
BlackScholesPricer::option_chain(
    double S, const std::vector<double>& strikes, double T, double r, double sigma
) noexcept {
    std::vector<std::pair<PricingResult, PricingResult>> chain;
    chain.reserve(strikes.size());
    
    for (double K : strikes) {
        auto call_result = price(S, K, T, r, sigma, OptionType::CALL);
        auto put_result = price(S, K, T, r, sigma, OptionType::PUT);
        chain.emplace_back(std::move(call_result), std::move(put_result));
    }
    
    return chain;
}

PutCallParityCheck check_put_call_parity(
    double call_price, double put_price, double S, double K, double T, double r
) noexcept {
    const double expected = S - K * std::exp(-r * T);
    const double actual = call_price - put_price;
    const double difference = std::abs(actual - expected);
    
    return PutCallParityCheck{
        .is_valid = difference < 0.01,  // Within 1 cent
        .difference = difference,
        .expected = expected
    };
}

} // namespace arbor::options
