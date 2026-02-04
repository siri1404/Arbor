#pragma once

#include <cmath>
#include <numbers>
#include <array>
#include <chrono>
#include <algorithm>
#include <vector>
#include <complex>
#include <functional>
#include <random>
#include <numeric>
#include <optional>
#include <span>

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace arbor::options {

using namespace std::numbers;

// ============================================================================
// CORE TYPES AND ENUMS
// ============================================================================

enum class OptionType : uint8_t { CALL = 0, PUT = 1 };
enum class ExerciseStyle : uint8_t { EUROPEAN = 0, AMERICAN = 1, BERMUDAN = 2 };
enum class BarrierType : uint8_t { 
    DOWN_AND_OUT = 0, DOWN_AND_IN = 1, 
    UP_AND_OUT = 2, UP_AND_IN = 3 
};

// Options Greeks - first and second order sensitivities
struct Greeks {
    double delta;     // dV/dS
    double gamma;     // d²V/dS²
    double theta;     // dV/dt (per day)
    double vega;      // dV/dσ (per 1%)
    double rho;       // dV/dr (per 1%)
    double vanna;     // d²V/dSdσ - delta sensitivity to vol
    double volga;     // d²V/dσ² - vega convexity (vomma)
    double charm;     // dΔ/dt - delta decay
    double speed;     // dΓ/dS - gamma sensitivity to spot
};

// Complete pricing result with diagnostics
struct PricingResult {
    double price;
    double intrinsic_value;
    double time_value;
    Greeks greeks;
    int64_t calc_time_ns;
    double std_error;           // Monte Carlo standard error
    int convergence_iterations; // For iterative methods
    bool converged;
};

// ============================================================================
// MATHEMATICAL UTILITIES - CACHE-LINE ALIGNED FOR SIMD
// ============================================================================

class alignas(64) MathUtils {
public:
    // Abramowitz & Stegun approximation - max error 7.5e-8
    [[nodiscard]] static inline double norm_cdf(double x) noexcept {
        static constexpr double a1 =  0.254829592;
        static constexpr double a2 = -0.284496736;
        static constexpr double a3 =  1.421413741;
        static constexpr double a4 = -1.453152027;
        static constexpr double a5 =  1.061405429;
        static constexpr double p  =  0.3275911;
        
        const int sign = (x < 0) ? -1 : 1;
        x = std::abs(x) * INV_SQRT2;
        
        const double t = 1.0 / (1.0 + p * x);
        const double t2 = t * t;
        const double t3 = t2 * t;
        const double t4 = t3 * t;
        const double t5 = t4 * t;
        
        const double y = 1.0 - (((((a5 * t5 + a4 * t4) + a3 * t3) + a2 * t2) + a1 * t) 
                         * std::exp(-x * x));
        
        return 0.5 * (1.0 + sign * y);
    }
    
    // Probability density function
    [[nodiscard]] static inline double norm_pdf(double x) noexcept {
        return INV_SQRT_2PI * std::exp(-0.5 * x * x);
    }
    
    // Inverse normal CDF - Beasley-Springer-Moro algorithm
    // Accurate to 1e-9 for p in (1e-15, 1-1e-15)
    [[nodiscard]] static double norm_inv(double p) noexcept {
        static constexpr double a[] = {
            -3.969683028665376e+01,  2.209460984245205e+02,
            -2.759285104469687e+02,  1.383577518672690e+02,
            -3.066479806614716e+01,  2.506628277459239e+00
        };
        static constexpr double b[] = {
            -5.447609879822406e+01,  1.615858368580409e+02,
            -1.556989798598866e+02,  6.680131188771972e+01,
            -1.328068155288572e+01
        };
        static constexpr double c[] = {
            -7.784894002430293e-03, -3.223964580411365e-01,
            -2.400758277161838e+00, -2.549732539343734e+00,
             4.374664141464968e+00,  2.938163982698783e+00
        };
        static constexpr double d[] = {
             7.784695709041462e-03,  3.224671290700398e-01,
             2.445134137142996e+00,  3.754408661907416e+00
        };
        
        static constexpr double p_low  = 0.02425;
        static constexpr double p_high = 1.0 - p_low;
        
        double q, r;
        
        if (p < p_low) {
            q = std::sqrt(-2.0 * std::log(p));
            return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) /
                   ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0);
        } else if (p <= p_high) {
            q = p - 0.5;
            r = q * q;
            return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q /
                   (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0);
        } else {
            q = std::sqrt(-2.0 * std::log(1.0 - p));
            return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) /
                    ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0);
        }
    }
    
    // Poisson PMF for jump-diffusion
    [[nodiscard]] static inline double poisson_pmf(int k, double lambda) noexcept {
        if (k < 0 || lambda <= 0) return 0.0;
        return std::exp(k * std::log(lambda) - lambda - std::lgamma(k + 1));
    }
    
    // Factorial lookup table (precomputed for n <= 20)
    [[nodiscard]] static inline double factorial(int n) noexcept {
        static constexpr double factorials[] = {
            1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880,
            3628800, 39916800, 479001600, 6227020800, 87178291200,
            1307674368000, 20922789888000, 355687428096000,
            6402373705728000, 121645100408832000, 2432902008176640000
        };
        return (n <= 20) ? factorials[n] : std::tgamma(n + 1);
    }
    
    static constexpr double INV_SQRT2 = 0.7071067811865475;
    static constexpr double INV_SQRT_2PI = 0.3989422804014327;
    static constexpr double SQRT_2PI = 2.5066282746310002;
};

// ============================================================================
// BLACK-SCHOLES PRICER - BASELINE IMPLEMENTATION
// ============================================================================

class alignas(64) BlackScholesPricer {
public:
    [[nodiscard]] static PricingResult price(
        double S, double K, double T, double r, double sigma, OptionType type
    ) noexcept;
    
    [[nodiscard]] static double implied_volatility(
        double market_price, double S, double K, double T, double r,
        OptionType type, double initial_guess = 0.3,
        double tolerance = 1e-10, int max_iterations = 100
    ) noexcept;
    
    [[nodiscard]] static std::vector<std::pair<PricingResult, PricingResult>> 
    option_chain(double S, const std::vector<double>& strikes, 
                 double T, double r, double sigma) noexcept;
    
    // Extended Greeks calculation
    [[nodiscard]] static Greeks compute_all_greeks(
        double S, double K, double T, double r, double sigma, OptionType type
    ) noexcept;

private:
    static inline void calc_d1_d2(double S, double K, double T, double r, double sigma,
                                   double& d1, double& d2) noexcept {
        const double sqrt_T = std::sqrt(T);
        const double sigma_sqrt_T = sigma * sqrt_T;
        d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / sigma_sqrt_T;
        d2 = d1 - sigma_sqrt_T;
    }
};

// ============================================================================
// MERTON JUMP-DIFFUSION MODEL
// Handles discontinuous price movements (earnings, news events)
// dS/S = (μ - λk)dt + σdW + (J-1)dN
// where N is Poisson process, J is lognormal jump size
// ============================================================================

struct JumpDiffusionParams {
    double lambda;      // Jump intensity (expected jumps per year)
    double mu_j;        // Mean jump size (log-normal)
    double sigma_j;     // Jump size volatility
    
    // Validate parameters
    [[nodiscard]] bool is_valid() const noexcept {
        return lambda >= 0 && sigma_j >= 0;
    }
};

class alignas(64) MertonJumpDiffusion {
public:
    // Analytical pricing via infinite series (truncated)
    // Uses the fact that conditional on n jumps, price follows adjusted BS
    [[nodiscard]] static PricingResult price(
        double S, double K, double T, double r, double sigma,
        const JumpDiffusionParams& jump_params,
        OptionType type,
        int max_terms = 50  // Truncation for infinite series
    ) noexcept;
    
    // Greeks via finite difference (analytical Greeks complex for JD)
    [[nodiscard]] static Greeks compute_greeks(
        double S, double K, double T, double r, double sigma,
        const JumpDiffusionParams& jump_params,
        OptionType type,
        double bump_size = 0.01
    ) noexcept;
    
    // Calibrate jump parameters from option prices
    [[nodiscard]] static JumpDiffusionParams calibrate(
        double S, double r,
        const std::vector<double>& strikes,
        const std::vector<double>& expiries,
        const std::vector<double>& market_prices,
        const std::vector<OptionType>& types,
        double sigma_initial = 0.2
    );

private:
    // Compensator: E[J-1] = exp(mu_j + 0.5*sigma_j²) - 1
    [[nodiscard]] static inline double jump_compensator(
        double mu_j, double sigma_j
    ) noexcept {
        return std::exp(mu_j + 0.5 * sigma_j * sigma_j) - 1.0;
    }
};

// ============================================================================
// HESTON STOCHASTIC VOLATILITY MODEL
// dS = μS dt + √V S dW₁
// dV = κ(θ - V)dt + ξ√V dW₂
// Corr(dW₁, dW₂) = ρ
// ============================================================================

struct HestonParams {
    double V0;      // Initial variance
    double kappa;   // Mean reversion speed
    double theta;   // Long-term variance
    double xi;      // Vol of vol (volatility of variance)
    double rho;     // Correlation between spot and vol
    
    // Feller condition: 2κθ > ξ² ensures variance stays positive
    [[nodiscard]] bool satisfies_feller() const noexcept {
        return 2.0 * kappa * theta > xi * xi;
    }
    
    [[nodiscard]] bool is_valid() const noexcept {
        return V0 > 0 && kappa > 0 && theta > 0 && xi > 0 && 
               rho >= -1 && rho <= 1;
    }
};

class alignas(64) HestonModel {
public:
    // Pricing via characteristic function and FFT
    // Uses Carr-Madan / Lewis (2001) approach
    [[nodiscard]] static PricingResult price(
        double S, double K, double T, double r,
        const HestonParams& params,
        OptionType type,
        int fft_points = 4096  // Power of 2 for FFT efficiency
    ) noexcept;
    
    // Direct numerical integration (more accurate, slower)
    [[nodiscard]] static PricingResult price_quadrature(
        double S, double K, double T, double r,
        const HestonParams& params,
        OptionType type,
        int integration_points = 1000
    ) noexcept;
    
    // Greeks via characteristic function differentiation
    [[nodiscard]] static Greeks compute_greeks(
        double S, double K, double T, double r,
        const HestonParams& params,
        OptionType type
    ) noexcept;
    
    // Calibration using Levenberg-Marquardt
    [[nodiscard]] static HestonParams calibrate(
        double S, double r,
        const std::vector<double>& strikes,
        const std::vector<double>& expiries,
        const std::vector<double>& market_prices,
        const std::vector<OptionType>& types,
        const HestonParams& initial_guess
    );
    
    // Characteristic function: E[exp(iu*log(S_T))]
    [[nodiscard]] static std::complex<double> characteristic_function(
        std::complex<double> u,
        double S, double T, double r,
        const HestonParams& params
    ) noexcept;

private:
    // Heston characteristic function components
    [[nodiscard]] static std::complex<double> D_component(
        std::complex<double> u, double T,
        const HestonParams& params
    ) noexcept;
    
    [[nodiscard]] static std::complex<double> C_component(
        std::complex<double> u, double T, double r,
        const HestonParams& params
    ) noexcept;
};

// ============================================================================
// SABR MODEL - Industry Standard for Vol Surface
// dF = σF^β dW₁  (forward dynamics)
// dσ = ασ dW₂
// Corr(dW₁, dW₂) = ρ
// ============================================================================

struct SABRParams {
    double alpha;   // Initial vol level
    double beta;    // CEV exponent (0 = normal, 1 = lognormal)
    double rho;     // Correlation
    double nu;      // Vol of vol
    
    [[nodiscard]] bool is_valid() const noexcept {
        return alpha > 0 && beta >= 0 && beta <= 1 && 
               rho >= -1 && rho <= 1 && nu >= 0;
    }
};

class alignas(64) SABRModel {
public:
    // Hagan et al. (2002) approximation for implied vol
    [[nodiscard]] static double implied_volatility(
        double F,           // Forward price
        double K,           // Strike
        double T,           // Time to expiry
        const SABRParams& params
    ) noexcept;
    
    // Price using SABR implied vol + Black formula
    [[nodiscard]] static PricingResult price(
        double S, double K, double T, double r,
        const SABRParams& params,
        OptionType type
    ) noexcept;
    
    // Calibrate SABR to market smile for single expiry
    [[nodiscard]] static SABRParams calibrate(
        double F, double T,
        const std::vector<double>& strikes,
        const std::vector<double>& market_vols,
        double beta_fixed = 0.5  // Often fixed based on asset class
    );
    
    // Vol surface interpolation
    [[nodiscard]] static double interpolate_vol(
        double K, double T,
        const std::vector<double>& expiries,
        const std::vector<SABRParams>& params_by_expiry
    ) noexcept;

private:
    // Helper for ATM vol approximation
    [[nodiscard]] static double atm_vol(
        double F, double T, const SABRParams& params
    ) noexcept;
};

// ============================================================================
// AMERICAN OPTIONS - BINOMIAL TREE WITH RICHARDSON EXTRAPOLATION
// ============================================================================

struct BinomialTreeConfig {
    int steps;                  // Number of time steps
    bool use_richardson;        // Richardson extrapolation for accuracy
    bool smooth_convergence;    // Smoothing for Greeks
};

class alignas(64) AmericanOptionPricer {
public:
    // CRR Binomial Tree with early exercise
    [[nodiscard]] static PricingResult price_binomial(
        double S, double K, double T, double r, double sigma,
        OptionType type,
        const BinomialTreeConfig& config = {200, true, true}
    ) noexcept;
    
    // Trinomial tree - better convergence
    [[nodiscard]] static PricingResult price_trinomial(
        double S, double K, double T, double r, double sigma,
        OptionType type,
        int steps = 200
    ) noexcept;
    
    // Early exercise boundary
    [[nodiscard]] static std::vector<std::pair<double, double>> 
    exercise_boundary(
        double S, double K, double T, double r, double sigma,
        OptionType type,
        int steps = 100
    ) noexcept;
    
    // Early exercise premium
    [[nodiscard]] static double early_exercise_premium(
        double S, double K, double T, double r, double sigma,
        OptionType type
    ) noexcept;

private:
    // Build CRR tree parameters
    struct TreeParams {
        double u, d, p;  // Up factor, down factor, risk-neutral prob
        double dt;
    };
    
    [[nodiscard]] static TreeParams build_crr_params(
        double r, double sigma, double dt
    ) noexcept;
};

// ============================================================================
// LONGSTAFF-SCHWARTZ LEAST SQUARES MONTE CARLO
// For American/Bermudan options with path dependency
// ============================================================================

struct LSMConfig {
    int num_paths;          // Number of Monte Carlo paths
    int time_steps;         // Exercise opportunities
    int basis_degree;       // Polynomial degree for regression
    uint64_t seed;          // RNG seed for reproducibility
    bool use_antithetic;    // Antithetic variates
    bool use_control;       // Control variate (European price)
};

class alignas(64) LongstaffSchwartzPricer {
public:
    [[nodiscard]] static PricingResult price(
        double S, double K, double T, double r, double sigma,
        OptionType type,
        const LSMConfig& config = {100000, 50, 3, 42, true, true}
    ) noexcept;
    
    // With stochastic volatility (Heston)
    [[nodiscard]] static PricingResult price_heston(
        double S, double K, double T, double r,
        const HestonParams& vol_params,
        OptionType type,
        const LSMConfig& config = {100000, 50, 3, 42, true, false}
    ) noexcept;
    
    // Bermudan with specific exercise dates
    [[nodiscard]] static PricingResult price_bermudan(
        double S, double K, double T, double r, double sigma,
        const std::vector<double>& exercise_times,  // Fraction of T
        OptionType type,
        const LSMConfig& config = {100000, 50, 3, 42, true, true}
    ) noexcept;

private:
    // Laguerre polynomial basis functions
    [[nodiscard]] static std::vector<double> laguerre_basis(
        double x, int degree
    ) noexcept;
    
    // Regression for continuation value
    [[nodiscard]] static std::vector<double> regression_coefficients(
        const std::vector<double>& X,
        const std::vector<double>& Y,
        int degree
    ) noexcept;
};

// ============================================================================
// BARRIER OPTIONS - ANALYTICAL AND MONTE CARLO
// ============================================================================

struct BarrierParams {
    BarrierType type;
    double barrier_level;
    double rebate;          // Payment if knocked out
};

class alignas(64) BarrierOptionPricer {
public:
    // Analytical formula for continuous monitoring
    [[nodiscard]] static PricingResult price_analytical(
        double S, double K, double T, double r, double sigma,
        const BarrierParams& barrier,
        OptionType option_type
    ) noexcept;
    
    // Monte Carlo for discrete monitoring
    [[nodiscard]] static PricingResult price_monte_carlo(
        double S, double K, double T, double r, double sigma,
        const BarrierParams& barrier,
        OptionType option_type,
        int monitoring_freq,    // Observations per year
        int num_paths = 100000,
        uint64_t seed = 42
    ) noexcept;

private:
    // Reflection principle components
    [[nodiscard]] static double A_component(
        double S, double K, double H, double T, double r, double sigma, double phi
    ) noexcept;
    
    [[nodiscard]] static double B_component(
        double S, double K, double H, double T, double r, double sigma, double phi
    ) noexcept;
};

// ============================================================================
// ASIAN OPTIONS - ARITHMETIC AND GEOMETRIC AVERAGE
// ============================================================================

enum class AveragingType : uint8_t { ARITHMETIC = 0, GEOMETRIC = 1 };

struct AsianParams {
    AveragingType averaging;
    int num_observations;
    bool average_strike;    // true = avg strike, false = avg price
};

class alignas(64) AsianOptionPricer {
public:
    // Geometric average - closed form
    [[nodiscard]] static PricingResult price_geometric(
        double S, double K, double T, double r, double sigma,
        const AsianParams& params,
        OptionType type
    ) noexcept;
    
    // Arithmetic average - Monte Carlo
    [[nodiscard]] static PricingResult price_arithmetic(
        double S, double K, double T, double r, double sigma,
        const AsianParams& params,
        OptionType type,
        int num_paths = 100000,
        uint64_t seed = 42
    ) noexcept;
    
    // Turnbull-Wakeman approximation for arithmetic
    [[nodiscard]] static PricingResult price_turnbull_wakeman(
        double S, double K, double T, double r, double sigma,
        const AsianParams& params,
        OptionType type
    ) noexcept;
};

// ============================================================================
// VARIANCE REDUCTION TECHNIQUES
// ============================================================================

class VarianceReduction {
public:
    // Antithetic variates - uses correlation of -1
    template<typename PathGenerator, typename Payoff>
    [[nodiscard]] static std::pair<double, double> antithetic_estimate(
        PathGenerator&& gen, Payoff&& payoff, int num_paths
    );
    
    // Control variate using European BS price as control
    [[nodiscard]] static std::pair<double, double> control_variate(
        const std::vector<double>& raw_payoffs,
        const std::vector<double>& control_values,
        double control_mean,  // E[control] = BS price
        double discount
    ) noexcept;
    
    // Importance sampling for rare events
    template<typename Sampler>
    [[nodiscard]] static std::pair<double, double> importance_sampling(
        Sampler&& sample, int num_paths, 
        double drift_shift  // Shift to center sampling
    );
};

// ============================================================================
// SIMD VECTORIZED MONTE CARLO ENGINE
// ============================================================================

#ifdef __AVX2__
class alignas(64) SIMDMonteCarloEngine {
public:
    // Generate 4 GBM paths simultaneously using AVX2
    static void generate_gbm_paths_avx2(
        double S0, double r, double sigma, double T,
        int steps, int num_path_batches,
        double* __restrict__ paths,  // Output: 4 * num_path_batches paths
        uint64_t seed = 42
    ) noexcept;
    
    // Vectorized payoff calculation
    static void compute_call_payoffs_avx2(
        const double* __restrict__ terminal_prices,
        double K, int num_prices,
        double* __restrict__ payoffs
    ) noexcept;
    
    // Full vectorized European option pricer
    [[nodiscard]] static PricingResult price_european_avx2(
        double S, double K, double T, double r, double sigma,
        OptionType type,
        int num_paths = 1000000,
        uint64_t seed = 42
    ) noexcept;
    
private:
    // Vectorized Box-Muller transform
    static __m256d box_muller_avx2(__m256d u1, __m256d u2) noexcept;
    
    // Fast xorshift128+ PRNG (vectorized)
    struct alignas(32) XorShift128Plus {
        __m256i state[2];
        
        void seed(uint64_t s) noexcept;
        __m256d next_uniform() noexcept;
    };
};
#endif

// Non-SIMD fallback
class alignas(64) MonteCarloEngine {
public:
    [[nodiscard]] static PricingResult price_european(
        double S, double K, double T, double r, double sigma,
        OptionType type,
        int num_paths = 100000,
        uint64_t seed = 42,
        bool use_antithetic = true,
        bool use_control = true
    ) noexcept;
    
    // European with Heston vol dynamics
    [[nodiscard]] static PricingResult price_european_heston(
        double S, double K, double T, double r,
        const HestonParams& params,
        OptionType type,
        int num_paths = 100000,
        int time_steps = 252,
        uint64_t seed = 42
    ) noexcept;
    
    // Jump-diffusion Monte Carlo
    [[nodiscard]] static PricingResult price_jump_diffusion(
        double S, double K, double T, double r, double sigma,
        const JumpDiffusionParams& jump_params,
        OptionType type,
        int num_paths = 100000,
        int time_steps = 252,
        uint64_t seed = 42
    ) noexcept;
};

// ============================================================================
// VOL SURFACE AND SMILE FITTING
// ============================================================================

struct VolSurfacePoint {
    double strike;
    double expiry;
    double implied_vol;
};

class VolSurface {
public:
    // Build from market data
    void build(const std::vector<VolSurfacePoint>& market_data);
    
    // Interpolate vol at any (K, T)
    [[nodiscard]] double get_vol(double K, double T) const noexcept;
    
    // Get SABR params for specific expiry
    [[nodiscard]] SABRParams get_sabr_params(double T) const;
    
    // Check for arbitrage (butterfly spread, calendar spread)
    [[nodiscard]] bool check_no_arbitrage() const noexcept;
    
private:
    std::vector<double> expiries_;
    std::vector<SABRParams> sabr_params_;  // One per expiry
    double forward_;
};

// ============================================================================
// PUT-CALL PARITY AND VALIDATION
// ============================================================================

struct PutCallParityCheck {
    bool is_valid;
    double difference;
    double expected;
};

[[nodiscard]] PutCallParityCheck check_put_call_parity(
    double call_price, double put_price,
    double S, double K, double T, double r
) noexcept;

// Forward price from put-call parity
[[nodiscard]] inline double implied_forward(
    double call_price, double put_price, double K, double T, double r
) noexcept {
    return (call_price - put_price) * std::exp(r * T) + K;
}

} // namespace arbor::options
