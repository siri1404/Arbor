#pragma once

#include <vector>
#include <string>
#include <cstdint>

namespace arbor::indicators {

// Technical indicator calculations with optimized implementations

struct OHLCBar {
    double open;
    double high;
    double low;
    double close;
    uint64_t volume;
};

// Simple Moving Average
std::vector<double> calculate_sma(const std::vector<double>& prices, size_t period);

// Exponential Moving Average
std::vector<double> calculate_ema(const std::vector<double>& prices, size_t period);

// Relative Strength Index
std::vector<double> calculate_rsi(const std::vector<double>& prices, size_t period = 14);

// MACD (Moving Average Convergence Divergence)
struct MACDResult {
    std::vector<double> macd_line;
    std::vector<double> signal_line;
    std::vector<double> histogram;
};

MACDResult calculate_macd(
    const std::vector<double>& prices,
    size_t fast_period = 12,
    size_t slow_period = 26,
    size_t signal_period = 9
);

// Bollinger Bands
struct BollingerBands {
    std::vector<double> upper;
    std::vector<double> middle;
    std::vector<double> lower;
};

BollingerBands calculate_bollinger_bands(
    const std::vector<double>& prices,
    size_t period = 20,
    double std_dev_multiplier = 2.0
);

// Average True Range (volatility)
std::vector<double> calculate_atr(const std::vector<OHLCBar>& bars, size_t period = 14);

} // namespace arbor::indicators
