#pragma once

#include <cstdint>
#include <array>
#include <vector>
#include <unordered_map>
#include <memory>
#include <chrono>
#include <algorithm>
#include <string_view>

namespace arbor::orderbook {

// Cache-line aligned for optimal CPU cache performance
static constexpr size_t CACHE_LINE_SIZE = 64;

// High-resolution timestamp using steady_clock
using Timestamp = std::chrono::time_point<std::chrono::steady_clock>;
using Nanoseconds = std::chrono::nanoseconds;

enum class Side : uint8_t { BUY = 0, SELL = 1 };
enum class OrderType : uint8_t { LIMIT = 0, MARKET = 1 };
enum class OrderStatus : uint8_t { NEW = 0, PARTIAL = 1, FILLED = 2, CANCELLED = 3 };

// Compact order representation - 64 bytes (fits in cache line)
struct alignas(CACHE_LINE_SIZE) Order {
    uint64_t order_id;
    uint64_t price_ticks;  // Price in ticks for integer arithmetic (avoid floating point)
    uint32_t quantity;
    uint32_t filled_quantity;
    Side side;
    OrderType type;
    OrderStatus status;
    uint8_t padding[5];  // Align to cache line
    Timestamp timestamp;

    [[nodiscard]] inline bool is_buy() const noexcept { return side == Side::BUY; }
    [[nodiscard]] inline bool is_sell() const noexcept { return side == Side::SELL; }
    [[nodiscard]] inline uint32_t remaining_qty() const noexcept { 
        return quantity - filled_quantity; 
    }
};

// Trade execution record
struct Trade {
    uint64_t trade_id;
    uint64_t buy_order_id;
    uint64_t sell_order_id;
    uint64_t price_ticks;
    uint32_t quantity;
    Timestamp timestamp;
    int64_t latency_ns;  // Matching latency in nanoseconds
};

// Price level in order book - uses pointer to maintain FIFO without reallocation
struct PriceLevel {
    uint64_t price_ticks;
    std::vector<Order*> orders;  // FIFO queue of order pointers
    uint64_t total_quantity{0};
    
    inline void add_order(Order* order) {
        orders.push_back(order);
        total_quantity += order->remaining_qty();
    }
    
    inline void remove_front() {
        if (!orders.empty()) {
            total_quantity -= orders.front()->remaining_qty();
            orders.erase(orders.begin());
        }
    }
    
    [[nodiscard]] inline bool empty() const noexcept { return orders.empty(); }
};

// Latency statistics tracker
struct LatencyStats {
    int64_t last_ns{0};
    int64_t min_ns{INT64_MAX};
    int64_t max_ns{0};
    int64_t sum_ns{0};
    uint64_t count{0};
    std::vector<int64_t> samples;  // For percentile calculation
    
    inline void record(int64_t latency_ns) {
        last_ns = latency_ns;
        min_ns = std::min(min_ns, latency_ns);
        max_ns = std::max(max_ns, latency_ns);
        sum_ns += latency_ns;
        ++count;
        samples.push_back(latency_ns);
    }
    
    [[nodiscard]] double avg_ns() const noexcept {
        return count > 0 ? static_cast<double>(sum_ns) / count : 0.0;
    }
    
    [[nodiscard]] int64_t p99_ns() const {
        if (samples.empty()) return 0;
        auto sorted = samples;
        std::sort(sorted.begin(), sorted.end());
        size_t idx = static_cast<size_t>(sorted.size() * 0.99);
        return sorted[idx];
    }
};

// High-performance limit order book with sub-microsecond matching
class LimitOrderBook {
public:
    explicit LimitOrderBook(std::string_view symbol, uint32_t tick_size = 1);
    ~LimitOrderBook() = default;
    
    // Non-copyable but movable
    LimitOrderBook(const LimitOrderBook&) = delete;
    LimitOrderBook& operator=(const LimitOrderBook&) = delete;
    LimitOrderBook(LimitOrderBook&&) noexcept = default;
    LimitOrderBook& operator=(LimitOrderBook&&) noexcept = default;
    
    // Core operations - optimized for minimal latency
    [[nodiscard]] uint64_t add_order(Side side, OrderType type, uint64_t price_ticks, 
                                     uint32_t quantity, std::vector<Trade>& trades_out);
    
    bool cancel_order(uint64_t order_id);
    bool modify_order(uint64_t order_id, uint32_t new_quantity);
    
    // Market data queries - const and noexcept for zero overhead
    [[nodiscard]] uint64_t best_bid() const noexcept;
    [[nodiscard]] uint64_t best_ask() const noexcept;
    [[nodiscard]] uint64_t spread() const noexcept { return best_ask() - best_bid(); }
    [[nodiscard]] uint64_t mid_price() const noexcept { return (best_bid() + best_ask()) / 2; }
    
    [[nodiscard]] const std::vector<PriceLevel>& get_bids(size_t depth = 10) const;
    [[nodiscard]] const std::vector<PriceLevel>& get_asks(size_t depth = 10) const;
    
    [[nodiscard]] const LatencyStats& get_latency_stats() const noexcept { return latency_stats_; }
    [[nodiscard]] size_t total_orders() const noexcept { return orders_.size(); }
    [[nodiscard]] size_t total_trades() const noexcept { return trades_.size(); }
    
    void clear();
    
private:
    std::string symbol_;
    uint32_t tick_size_;
    uint64_t next_order_id_{1};
    uint64_t next_trade_id_{1};
    
    // Memory pool for orders - heap allocated to avoid stack overflow
    static constexpr size_t MAX_ORDERS = 100000;
    std::unique_ptr<std::array<Order, MAX_ORDERS>> order_pool_;
    size_t next_order_slot_{0};
    
    // Fast lookup structures
    std::unordered_map<uint64_t, Order*> orders_;  // order_id -> Order*
    std::unordered_map<uint64_t, PriceLevel> bids_;  // price -> PriceLevel
    std::unordered_map<uint64_t, PriceLevel> asks_;
    
    // Sorted price levels for fast best bid/ask
    std::vector<uint64_t> bid_prices_;  // Descending
    std::vector<uint64_t> ask_prices_;  // Ascending
    
    // Trade history
    std::vector<Trade> trades_;
    
    // Performance metrics
    LatencyStats latency_stats_;
    
    // Internal matching engine - the critical path
    void match_order(Order* order, std::vector<Trade>& trades_out);
    void match_aggressive_buy(Order* order, std::vector<Trade>& trades_out);
    void match_aggressive_sell(Order* order, std::vector<Trade>& trades_out);
    
    // Price level management
    void insert_price_level(Side side, uint64_t price);
    void remove_price_level(Side side, uint64_t price);
    
    // Helper functions
    [[nodiscard]] Order* allocate_order();
    [[nodiscard]] inline Nanoseconds elapsed_ns(const Timestamp& start) const noexcept {
        return std::chrono::duration_cast<Nanoseconds>(
            std::chrono::steady_clock::now() - start
        );
    }
};

// Snapshot for external consumption
struct OrderBookSnapshot {
    std::string symbol;
    std::vector<std::pair<uint64_t, uint64_t>> bids;  // price, quantity
    std::vector<std::pair<uint64_t, uint64_t>> asks;
    uint64_t best_bid_price;
    uint64_t best_ask_price;
    uint64_t spread;
    uint64_t mid_price;
    size_t total_orders;
    size_t total_trades;
    LatencyStats stats;
};

[[nodiscard]] OrderBookSnapshot create_snapshot(const LimitOrderBook& book, size_t depth = 15);

} // namespace arbor::orderbook
