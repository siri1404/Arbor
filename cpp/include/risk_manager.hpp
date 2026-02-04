#pragma once

/**
 * Position Management & Risk Controls
 * 
 * Production-grade position tracking with:
 * - Real-time P&L calculation
 * - Pre-trade risk checks
 * - Position limits enforcement
 * - Fill tracking with sequence numbers
 */

#include <unordered_map>
#include <string>
#include <vector>
#include <atomic>
#include <mutex>
#include <chrono>
#include <cstdint>
#include <optional>
#include <cstring>

namespace arbor::risk {

// Fixed-point price representation (8 decimal places)
constexpr int64_t PRICE_MULTIPLIER = 100000000LL;

inline int64_t to_fixed(double price) {
    return static_cast<int64_t>(price * PRICE_MULTIPLIER);
}

inline double from_fixed(int64_t fixed) {
    return static_cast<double>(fixed) / PRICE_MULTIPLIER;
}

/**
 * Fill record for audit trail
 */
struct Fill {
    uint64_t fill_id;
    uint64_t order_id;
    uint64_t sequence_num;
    char symbol[16];
    char side;  // 'B' or 'S'
    int64_t price;  // Fixed point
    uint32_t quantity;
    int64_t timestamp_ns;
    char venue[8];
    char exec_id[32];
};

/**
 * Position for a single instrument
 */
struct Position {
    char symbol[16];
    
    // Current position
    int64_t quantity{0};  // Positive = long, negative = short
    int64_t avg_price{0};  // Volume-weighted average price (fixed point)
    
    // Realized P&L from closed positions
    int64_t realized_pnl{0};
    
    // Daily statistics
    int64_t day_buy_qty{0};
    int64_t day_sell_qty{0};
    int64_t day_buy_value{0};
    int64_t day_sell_value{0};
    
    // High water mark for drawdown
    int64_t peak_pnl{0};
    int64_t max_drawdown{0};
    
    // Fill tracking
    uint64_t last_fill_id{0};
    uint64_t last_sequence{0};
    
    /**
     * Calculate unrealized P&L given current market price
     */
    int64_t unrealized_pnl(int64_t current_price) const {
        if (quantity == 0) return 0;
        return quantity * (current_price - avg_price) / PRICE_MULTIPLIER;
    }
    
    /**
     * Total P&L (realized + unrealized)
     */
    int64_t total_pnl(int64_t current_price) const {
        return realized_pnl + unrealized_pnl(current_price);
    }
    
    /**
     * Process a fill and update position
     */
    void apply_fill(const Fill& fill) {
        const int64_t fill_value = fill.price * fill.quantity;
        
        if (fill.side == 'B') {
            // Buying
            day_buy_qty += fill.quantity;
            day_buy_value += fill_value;
            
            if (quantity >= 0) {
                // Adding to long or opening long
                int64_t new_qty = quantity + fill.quantity;
                avg_price = (avg_price * quantity + fill.price * fill.quantity) / new_qty;
                quantity = new_qty;
            } else {
                // Covering short
                int64_t cover_qty = std::min(static_cast<int64_t>(fill.quantity), -quantity);
                realized_pnl += cover_qty * (avg_price - fill.price) / PRICE_MULTIPLIER;
                
                quantity += fill.quantity;
                if (quantity > 0) {
                    // Flipped to long
                    avg_price = fill.price;
                }
            }
        } else {
            // Selling
            day_sell_qty += fill.quantity;
            day_sell_value += fill_value;
            
            if (quantity <= 0) {
                // Adding to short or opening short
                int64_t new_qty = quantity - fill.quantity;
                avg_price = (avg_price * (-quantity) + fill.price * fill.quantity) / (-new_qty);
                quantity = new_qty;
            } else {
                // Closing long
                int64_t close_qty = std::min(static_cast<int64_t>(fill.quantity), quantity);
                realized_pnl += close_qty * (fill.price - avg_price) / PRICE_MULTIPLIER;
                
                quantity -= fill.quantity;
                if (quantity < 0) {
                    // Flipped to short
                    avg_price = fill.price;
                }
            }
        }
        
        last_fill_id = fill.fill_id;
        last_sequence = fill.sequence_num;
        
        // Update drawdown tracking
        int64_t current_pnl = realized_pnl;  // Use realized for HWM
        if (current_pnl > peak_pnl) {
            peak_pnl = current_pnl;
        }
        int64_t drawdown = peak_pnl - current_pnl;
        if (drawdown > max_drawdown) {
            max_drawdown = drawdown;
        }
    }
};


/**
 * Risk Limits Configuration
 */
struct RiskLimits {
    // Position limits
    int64_t max_position_qty{10000};         // Max shares per symbol
    int64_t max_position_value{1000000};     // Max notional per symbol ($)
    int64_t max_total_exposure{10000000};    // Max total notional ($)
    
    // Order limits
    uint32_t max_order_qty{5000};            // Max shares per order
    int64_t max_order_value{500000};         // Max notional per order ($)
    uint32_t max_orders_per_second{100};     // Rate limit
    
    // P&L limits
    int64_t max_loss_per_symbol{50000};      // Stop loss per symbol ($)
    int64_t max_daily_loss{200000};          // Daily loss limit ($)
    int64_t max_drawdown{100000};            // Max drawdown ($)
    
    // Concentration limits
    double max_sector_exposure_pct{0.25};    // Max 25% in one sector
    uint32_t max_symbols{100};               // Max concurrent positions
};


/**
 * Risk Check Result
 */
struct RiskCheckResult {
    bool passed{true};
    uint32_t reject_code{0};
    char reject_reason[128]{};
    
    // Timing
    int64_t check_latency_ns{0};
    
    static RiskCheckResult pass() {
        return RiskCheckResult{true, 0, {}};
    }
    
    static RiskCheckResult reject(uint32_t code, const char* reason) {
        RiskCheckResult r{false, code, {}};
        strncpy(r.reject_reason, reason, sizeof(r.reject_reason) - 1);
        return r;
    }
};

// Reject codes
constexpr uint32_t REJECT_MAX_POSITION_QTY = 1001;
constexpr uint32_t REJECT_MAX_POSITION_VALUE = 1002;
constexpr uint32_t REJECT_MAX_ORDER_QTY = 1003;
constexpr uint32_t REJECT_MAX_ORDER_VALUE = 1004;
constexpr uint32_t REJECT_RATE_LIMIT = 1005;
constexpr uint32_t REJECT_MAX_LOSS = 1006;
constexpr uint32_t REJECT_MAX_DRAWDOWN = 1007;
constexpr uint32_t REJECT_MAX_EXPOSURE = 1008;
constexpr uint32_t REJECT_SYMBOL_HALTED = 1009;


/**
 * Pre-Trade Risk Manager
 * 
 * All checks designed for <1μs latency
 */
class RiskManager {
public:
    explicit RiskManager(const RiskLimits& limits) : limits_(limits) {}

    /**
     * Pre-trade risk check
     * Must be called before every order submission
     * 
     * @return RiskCheckResult with pass/fail and reason
     * 
     * Target latency: <500ns
     */
    RiskCheckResult check_order(
        const char* symbol,
        char side,
        uint32_t quantity,
        int64_t price,  // Fixed point
        int64_t current_market_price  // For exposure calculation
    ) {
        auto start = std::chrono::steady_clock::now();
        
        RiskCheckResult result = RiskCheckResult::pass();
        
        // 1. Order size limit
        if (quantity > limits_.max_order_qty) {
            result = RiskCheckResult::reject(REJECT_MAX_ORDER_QTY, 
                "Order quantity exceeds limit");
            goto done;
        }
        
        // 2. Order value limit
        {
            int64_t order_value = (price * quantity) / PRICE_MULTIPLIER;
            if (order_value > limits_.max_order_value) {
                result = RiskCheckResult::reject(REJECT_MAX_ORDER_VALUE,
                    "Order value exceeds limit");
                goto done;
            }
        }
        
        // 3. Rate limit check
        {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - last_order_time_).count();
            
            if (elapsed < 1000) {
                orders_this_second_++;
                if (orders_this_second_ > limits_.max_orders_per_second) {
                    result = RiskCheckResult::reject(REJECT_RATE_LIMIT,
                        "Rate limit exceeded");
                    goto done;
                }
            } else {
                orders_this_second_ = 1;
                last_order_time_ = now;
            }
        }
        
        // 4. Position limit check
        {
            std::lock_guard<std::mutex> lock(positions_mutex_);
            auto it = positions_.find(symbol);
            
            if (it != positions_.end()) {
                const auto& pos = it->second;
                
                // Check resulting position
                int64_t new_qty = pos.quantity;
                if (side == 'B') {
                    new_qty += quantity;
                } else {
                    new_qty -= quantity;
                }
                
                if (std::abs(new_qty) > limits_.max_position_qty) {
                    result = RiskCheckResult::reject(REJECT_MAX_POSITION_QTY,
                        "Position quantity would exceed limit");
                    goto done;
                }
                
                // Check position value
                int64_t new_value = std::abs(new_qty * current_market_price / PRICE_MULTIPLIER);
                if (new_value > limits_.max_position_value) {
                    result = RiskCheckResult::reject(REJECT_MAX_POSITION_VALUE,
                        "Position value would exceed limit");
                    goto done;
                }
                
                // Check loss limits
                int64_t current_pnl = pos.total_pnl(current_market_price);
                if (-current_pnl > limits_.max_loss_per_symbol) {
                    result = RiskCheckResult::reject(REJECT_MAX_LOSS,
                        "Symbol loss limit reached");
                    goto done;
                }
                
                // Check drawdown
                if (pos.max_drawdown > limits_.max_drawdown) {
                    result = RiskCheckResult::reject(REJECT_MAX_DRAWDOWN,
                        "Drawdown limit reached");
                    goto done;
                }
            }
        }
        
        // 5. Total exposure check
        {
            int64_t order_exposure = (price * quantity) / PRICE_MULTIPLIER;
            int64_t new_total = total_exposure_.load(std::memory_order_relaxed) + order_exposure;
            if (new_total > limits_.max_total_exposure) {
                result = RiskCheckResult::reject(REJECT_MAX_EXPOSURE,
                    "Total exposure would exceed limit");
                goto done;
            }
        }
        
    done:
        auto end = std::chrono::steady_clock::now();
        result.check_latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            end - start).count();
        
        return result;
    }

    /**
     * Process a fill and update positions
     */
    void process_fill(const Fill& fill) {
        std::lock_guard<std::mutex> lock(positions_mutex_);
        
        std::string sym(fill.symbol);
        auto& pos = positions_[sym];
        strncpy(pos.symbol, fill.symbol, sizeof(pos.symbol));
        
        int64_t old_exposure = std::abs(pos.quantity * pos.avg_price / PRICE_MULTIPLIER);
        pos.apply_fill(fill);
        int64_t new_exposure = std::abs(pos.quantity * pos.avg_price / PRICE_MULTIPLIER);
        
        // Update total exposure
        total_exposure_.fetch_add(new_exposure - old_exposure, std::memory_order_relaxed);
        
        // Track for daily P&L
        daily_realized_pnl_.fetch_add(
            pos.realized_pnl - last_realized_pnl_[sym],
            std::memory_order_relaxed);
        last_realized_pnl_[sym] = pos.realized_pnl;
    }

    /**
     * Get position for a symbol
     */
    std::optional<Position> get_position(const std::string& symbol) const {
        std::lock_guard<std::mutex> lock(positions_mutex_);
        auto it = positions_.find(symbol);
        if (it != positions_.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    /**
     * Get all positions
     */
    std::vector<Position> get_all_positions() const {
        std::lock_guard<std::mutex> lock(positions_mutex_);
        std::vector<Position> result;
        result.reserve(positions_.size());
        for (const auto& [_, pos] : positions_) {
            result.push_back(pos);
        }
        return result;
    }

    /**
     * Daily P&L
     */
    int64_t daily_pnl() const {
        return daily_realized_pnl_.load(std::memory_order_relaxed);
    }

    /**
     * Total exposure
     */
    int64_t total_exposure() const {
        return total_exposure_.load(std::memory_order_relaxed);
    }

    /**
     * Reset daily counters (call at start of trading day)
     */
    void reset_daily() {
        std::lock_guard<std::mutex> lock(positions_mutex_);
        for (auto& [_, pos] : positions_) {
            pos.day_buy_qty = 0;
            pos.day_sell_qty = 0;
            pos.day_buy_value = 0;
            pos.day_sell_value = 0;
        }
        daily_realized_pnl_ = 0;
        last_realized_pnl_.clear();
    }

private:
    RiskLimits limits_;
    
    mutable std::mutex positions_mutex_;
    std::unordered_map<std::string, Position> positions_;
    std::unordered_map<std::string, int64_t> last_realized_pnl_;
    
    std::atomic<int64_t> total_exposure_{0};
    std::atomic<int64_t> daily_realized_pnl_{0};
    
    // Rate limiting
    std::chrono::steady_clock::time_point last_order_time_;
    uint32_t orders_this_second_{0};
};


/**
 * Order Sequencer
 * 
 * Ensures strict ordering of orders and fills
 * Critical for reconciliation and replay
 */
class OrderSequencer {
public:
    struct SequencedOrder {
        uint64_t sequence_num;
        uint64_t order_id;
        int64_t timestamp_ns;
        char symbol[16];
        char side;
        char order_type;  // 'L'imit, 'M'arket
        int64_t price;
        uint32_t quantity;
        uint64_t client_order_id;
    };

    OrderSequencer() : next_sequence_(1), next_order_id_(1) {}

    /**
     * Assign sequence number to new order
     * Thread-safe with atomic increment
     */
    SequencedOrder sequence_order(
        const char* symbol,
        char side,
        char order_type,
        int64_t price,
        uint32_t quantity,
        uint64_t client_order_id
    ) {
        SequencedOrder order{};
        order.sequence_num = next_sequence_.fetch_add(1, std::memory_order_relaxed);
        order.order_id = next_order_id_.fetch_add(1, std::memory_order_relaxed);
        order.timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        strncpy(order.symbol, symbol, sizeof(order.symbol));
        order.side = side;
        order.order_type = order_type;
        order.price = price;
        order.quantity = quantity;
        order.client_order_id = client_order_id;
        
        return order;
    }

    /**
     * Get next expected sequence number (for gap detection)
     */
    uint64_t expected_sequence() const {
        return next_sequence_.load(std::memory_order_relaxed);
    }

    /**
     * Check for sequence gap
     */
    bool check_sequence(uint64_t received_seq) const {
        uint64_t expected = next_sequence_.load(std::memory_order_relaxed);
        return received_seq == expected;
    }

private:
    std::atomic<uint64_t> next_sequence_;
    std::atomic<uint64_t> next_order_id_;
};

} // namespace arbor::risk
