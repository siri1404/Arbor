#include "orderbook.hpp"
#include <algorithm>
#include <stdexcept>

namespace arbor::orderbook {

LimitOrderBook::LimitOrderBook(std::string_view symbol, uint32_t tick_size)
    : symbol_(symbol), tick_size_(tick_size),
      order_pool_(std::make_unique<std::array<Order, MAX_ORDERS>>()) {
    // Pre-allocate to avoid reallocation during trading
    bid_prices_.reserve(1000);
    ask_prices_.reserve(1000);
    trades_.reserve(10000);
    orders_.reserve(10000);
}

Order* LimitOrderBook::allocate_order() {
    if (next_order_slot_ >= MAX_ORDERS) {
        throw std::runtime_error("Order pool exhausted");
    }
    return &(*order_pool_)[next_order_slot_++];
}

uint64_t LimitOrderBook::add_order(Side side, OrderType type, uint64_t price_ticks, 
                                    uint32_t quantity, std::vector<Trade>& trades_out) {
    // START CRITICAL SECTION - MINIMIZE LATENCY
    const auto start_time = std::chrono::steady_clock::now();
    
    // Allocate from pool - O(1)
    Order* order = allocate_order();
    order->order_id = next_order_id_++;
    order->price_ticks = price_ticks;
    order->quantity = quantity;
    order->filled_quantity = 0;
    order->side = side;
    order->type = type;
    order->status = OrderStatus::NEW;
    order->timestamp = start_time;
    
    orders_[order->order_id] = order;
    
    // Attempt matching - this is the HOT PATH
    match_order(order, trades_out);
    
    // If limit order with remaining quantity, add to book
    if (type == OrderType::LIMIT && order->remaining_qty() > 0) {
        auto& book = (side == Side::BUY) ? bids_ : asks_;
        
        if (book.find(price_ticks) == book.end()) {
            insert_price_level(side, price_ticks);
        }
        
        book[price_ticks].add_order(order);
    }
    
    // Record latency
    const int64_t latency_ns = elapsed_ns(start_time).count();
    latency_stats_.record(latency_ns);
    
    return order->order_id;
}

void LimitOrderBook::match_order(Order* order, std::vector<Trade>& trades_out) {
    if (order->side == Side::BUY) {
        match_aggressive_buy(order, trades_out);
    } else {
        match_aggressive_sell(order, trades_out);
    }
}

void LimitOrderBook::match_aggressive_buy(Order* order, std::vector<Trade>& trades_out) {
    // Buy order matches against asks (lowest price first)
    while (order->remaining_qty() > 0 && !ask_prices_.empty()) {
        const uint64_t best_ask_price = ask_prices_.front();
        
        // Price check for limit orders
        if (order->type == OrderType::LIMIT && best_ask_price > order->price_ticks) {
            break;
        }
        
        PriceLevel& level = asks_[best_ask_price];
        
        // Match against all orders at this price level (FIFO)
        while (!level.orders.empty() && order->remaining_qty() > 0) {
            Order* passive_order = level.orders.front();
            
            // Calculate fill quantity
            const uint32_t fill_qty = std::min(order->remaining_qty(), 
                                               passive_order->remaining_qty());
            
            // Update quantities
            order->filled_quantity += fill_qty;
            passive_order->filled_quantity += fill_qty;
            level.total_quantity -= fill_qty;
            
            // Create trade record
            Trade trade{
                .trade_id = next_trade_id_++,
                .buy_order_id = order->order_id,
                .sell_order_id = passive_order->order_id,
                .price_ticks = best_ask_price,
                .quantity = fill_qty,
                .timestamp = std::chrono::steady_clock::now(),
                .latency_ns = elapsed_ns(order->timestamp).count()
            };
            
            trades_out.push_back(trade);
            trades_.push_back(trade);
            
            // Update order statuses
            if (passive_order->remaining_qty() == 0) {
                passive_order->status = OrderStatus::FILLED;
                level.remove_front();
            } else {
                passive_order->status = OrderStatus::PARTIAL;
            }
        }
        
        // Remove empty price level
        if (level.empty()) {
            remove_price_level(Side::SELL, best_ask_price);
        }
    }
    
    // Update aggressor order status
    if (order->remaining_qty() == 0) {
        order->status = OrderStatus::FILLED;
    } else if (order->filled_quantity > 0) {
        order->status = OrderStatus::PARTIAL;
    }
}

void LimitOrderBook::match_aggressive_sell(Order* order, std::vector<Trade>& trades_out) {
    // Sell order matches against bids (highest price first)
    while (order->remaining_qty() > 0 && !bid_prices_.empty()) {
        const uint64_t best_bid_price = bid_prices_.front();
        
        if (order->type == OrderType::LIMIT && best_bid_price < order->price_ticks) {
            break;
        }
        
        PriceLevel& level = bids_[best_bid_price];
        
        while (!level.orders.empty() && order->remaining_qty() > 0) {
            Order* passive_order = level.orders.front();
            
            const uint32_t fill_qty = std::min(order->remaining_qty(), 
                                               passive_order->remaining_qty());
            
            order->filled_quantity += fill_qty;
            passive_order->filled_quantity += fill_qty;
            level.total_quantity -= fill_qty;
            
            Trade trade{
                .trade_id = next_trade_id_++,
                .buy_order_id = passive_order->order_id,
                .sell_order_id = order->order_id,
                .price_ticks = best_bid_price,
                .quantity = fill_qty,
                .timestamp = std::chrono::steady_clock::now(),
                .latency_ns = elapsed_ns(order->timestamp).count()
            };
            
            trades_out.push_back(trade);
            trades_.push_back(trade);
            
            if (passive_order->remaining_qty() == 0) {
                passive_order->status = OrderStatus::FILLED;
                level.remove_front();
            } else {
                passive_order->status = OrderStatus::PARTIAL;
            }
        }
        
        if (level.empty()) {
            remove_price_level(Side::BUY, best_bid_price);
        }
    }
    
    if (order->remaining_qty() == 0) {
        order->status = OrderStatus::FILLED;
    } else if (order->filled_quantity > 0) {
        order->status = OrderStatus::PARTIAL;
    }
}

void LimitOrderBook::insert_price_level(Side side, uint64_t price) {
    if (side == Side::BUY) {
        // Binary search insertion for descending order
        auto it = std::lower_bound(bid_prices_.begin(), bid_prices_.end(), price,
                                   std::greater<uint64_t>());
        bid_prices_.insert(it, price);
    } else {
        // Binary search insertion for ascending order
        auto it = std::lower_bound(ask_prices_.begin(), ask_prices_.end(), price);
        ask_prices_.insert(it, price);
    }
}

void LimitOrderBook::remove_price_level(Side side, uint64_t price) {
    if (side == Side::BUY) {
        bids_.erase(price);
        auto it = std::find(bid_prices_.begin(), bid_prices_.end(), price);
        if (it != bid_prices_.end()) {
            bid_prices_.erase(it);
        }
    } else {
        asks_.erase(price);
        auto it = std::find(ask_prices_.begin(), ask_prices_.end(), price);
        if (it != ask_prices_.end()) {
            ask_prices_.erase(it);
        }
    }
}

bool LimitOrderBook::cancel_order(uint64_t order_id) {
    auto it = orders_.find(order_id);
    if (it == orders_.end()) return false;
    
    Order* order = it->second;
    if (order->status == OrderStatus::FILLED) return false;
    
    // Remove from price level if still in book
    if (order->remaining_qty() > 0) {
        auto& book = (order->side == Side::BUY) ? bids_ : asks_;
        auto level_it = book.find(order->price_ticks);
        
        if (level_it != book.end()) {
            PriceLevel& level = level_it->second;
            auto& orders_vec = level.orders;
            
            orders_vec.erase(
                std::remove(orders_vec.begin(), orders_vec.end(), order),
                orders_vec.end()
            );
            
            level.total_quantity -= order->remaining_qty();
            
            if (level.empty()) {
                remove_price_level(order->side, order->price_ticks);
            }
        }
    }
    
    order->status = OrderStatus::CANCELLED;
    return true;
}

uint64_t LimitOrderBook::best_bid() const noexcept {
    return bid_prices_.empty() ? 0 : bid_prices_.front();
}

uint64_t LimitOrderBook::best_ask() const noexcept {
    return ask_prices_.empty() ? UINT64_MAX : ask_prices_.front();
}

const std::vector<PriceLevel>& LimitOrderBook::get_bids(size_t depth) const {
    static thread_local std::vector<PriceLevel> result;
    result.clear();
    
    const size_t n = std::min(depth, bid_prices_.size());
    result.reserve(n);
    
    for (size_t i = 0; i < n; ++i) {
        auto it = bids_.find(bid_prices_[i]);
        if (it != bids_.end()) {
            result.push_back(it->second);
        }
    }
    
    return result;
}

const std::vector<PriceLevel>& LimitOrderBook::get_asks(size_t depth) const {
    static thread_local std::vector<PriceLevel> result;
    result.clear();
    
    const size_t n = std::min(depth, ask_prices_.size());
    result.reserve(n);
    
    for (size_t i = 0; i < n; ++i) {
        auto it = asks_.find(ask_prices_[i]);
        if (it != asks_.end()) {
            result.push_back(it->second);
        }
    }
    
    return result;
}

void LimitOrderBook::clear() {
    orders_.clear();
    bids_.clear();
    asks_.clear();
    bid_prices_.clear();
    ask_prices_.clear();
    trades_.clear();
    next_order_slot_ = 0;
    next_order_id_ = 1;
    next_trade_id_ = 1;
    latency_stats_ = LatencyStats{};
}

OrderBookSnapshot create_snapshot(const LimitOrderBook& book, size_t depth) {
    OrderBookSnapshot snapshot;
    snapshot.symbol = "";  // Set externally
    snapshot.best_bid_price = book.best_bid();
    snapshot.best_ask_price = book.best_ask();
    snapshot.spread = book.spread();
    snapshot.mid_price = book.mid_price();
    snapshot.total_orders = book.total_orders();
    snapshot.total_trades = book.total_trades();
    snapshot.stats = book.get_latency_stats();
    
    const auto& bids = book.get_bids(depth);
    const auto& asks = book.get_asks(depth);
    
    for (const auto& level : bids) {
        snapshot.bids.emplace_back(level.price_ticks, level.total_quantity);
    }
    
    for (const auto& level : asks) {
        snapshot.asks.emplace_back(level.price_ticks, level.total_quantity);
    }
    
    return snapshot;
}

} // namespace arbor::orderbook
