#include "technical_indicators.hpp"
#include <numeric>
#include <cmath>
#include <algorithm>

namespace arbor::indicators {

std::vector<double> calculate_sma(const std::vector<double>& prices, size_t period) {
    std::vector<double> sma(prices.size(), 0.0);
    
    if (prices.size() < period) {
        return sma;
    }
    
    // Calculate first SMA
    double sum = std::accumulate(prices.begin(), prices.begin() + period, 0.0);
    sma[period - 1] = sum / period;
    
    // Rolling window
    for (size_t i = period; i < prices.size(); ++i) {
        sum = sum - prices[i - period] + prices[i];
        sma[i] = sum / period;
    }
    
    return sma;
}

std::vector<double> calculate_ema(const std::vector<double>& prices, size_t period) {
    std::vector<double> ema(prices.size(), 0.0);
    
    if (prices.size() < period) {
        return ema;
    }
    
    const double multiplier = 2.0 / (period + 1.0);
    
    // Start with SMA
    double sum = std::accumulate(prices.begin(), prices.begin() + period, 0.0);
    ema[period - 1] = sum / period;
    
    // Calculate EMA
    for (size_t i = period; i < prices.size(); ++i) {
        ema[i] = (prices[i] - ema[i - 1]) * multiplier + ema[i - 1];
    }
    
    return ema;
}

std::vector<double> calculate_rsi(const std::vector<double>& prices, size_t period) {
    std::vector<double> rsi(prices.size(), 50.0);
    
    if (prices.size() < period + 1) {
        return rsi;
    }
    
    std::vector<double> gains, losses;
    
    for (size_t i = 1; i < prices.size(); ++i) {
        double change = prices[i] - prices[i - 1];
        gains.push_back(change > 0 ? change : 0.0);
        losses.push_back(change < 0 ? -change : 0.0);
    }
    
    for (size_t i = period; i < prices.size(); ++i) {
        double avg_gain = std::accumulate(
            gains.begin() + i - period, 
            gains.begin() + i, 
            0.0
        ) / period;
        
        double avg_loss = std::accumulate(
            losses.begin() + i - period,
            losses.begin() + i,
            0.0
        ) / period;
        
        if (avg_loss == 0.0) {
            rsi[i] = 100.0;
        } else {
            double rs = avg_gain / avg_loss;
            rsi[i] = 100.0 - (100.0 / (1.0 + rs));
        }
    }
    
    return rsi;
}

MACDResult calculate_macd(
    const std::vector<double>& prices,
    size_t fast_period,
    size_t slow_period,
    size_t signal_period
) {
    MACDResult result;
    
    auto ema_fast = calculate_ema(prices, fast_period);
    auto ema_slow = calculate_ema(prices, slow_period);
    
    result.macd_line.resize(prices.size());
    for (size_t i = 0; i < prices.size(); ++i) {
        result.macd_line[i] = ema_fast[i] - ema_slow[i];
    }
    
    result.signal_line = calculate_ema(result.macd_line, signal_period);
    
    result.histogram.resize(prices.size());
    for (size_t i = 0; i < prices.size(); ++i) {
        result.histogram[i] = result.macd_line[i] - result.signal_line[i];
    }
    
    return result;
}

BollingerBands calculate_bollinger_bands(
    const std::vector<double>& prices,
    size_t period,
    double std_dev_multiplier
) {
    BollingerBands bands;
    bands.middle = calculate_sma(prices, period);
    bands.upper.resize(prices.size());
    bands.lower.resize(prices.size());
    
    for (size_t i = period - 1; i < prices.size(); ++i) {
        double sum_sq = 0.0;
        for (size_t j = i - period + 1; j <= i; ++j) {
            double diff = prices[j] - bands.middle[i];
            sum_sq += diff * diff;
        }
        
        double std_dev = std::sqrt(sum_sq / period);
        bands.upper[i] = bands.middle[i] + std_dev_multiplier * std_dev;
        bands.lower[i] = bands.middle[i] - std_dev_multiplier * std_dev;
    }
    
    return bands;
}

std::vector<double> calculate_atr(const std::vector<OHLCBar>& bars, size_t period) {
    std::vector<double> atr(bars.size(), 0.0);
    
    if (bars.size() < period) {
        return atr;
    }
    
    std::vector<double> tr(bars.size());
    
    for (size_t i = 1; i < bars.size(); ++i) {
        double hl = bars[i].high - bars[i].low;
        double hc = std::abs(bars[i].high - bars[i - 1].close);
        double lc = std::abs(bars[i].low - bars[i - 1].close);
        
        tr[i] = std::max({hl, hc, lc});
    }
    
    // Calculate ATR using EMA-like smoothing
    double sum = std::accumulate(tr.begin() + 1, tr.begin() + period + 1, 0.0);
    atr[period] = sum / period;
    
    for (size_t i = period + 1; i < bars.size(); ++i) {
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period;
    }
    
    return atr;
}

} // namespace arbor::indicators
