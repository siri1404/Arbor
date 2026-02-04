#include "orderbook.hpp"
#include <stdexcept>

namespace arbor::orderbook {

// =============================================================================
// CONSTRUCTOR
// =============================================================================

LimitOrderBook::LimitOrderBook(const std::string& symbol, uint32_t tick_size)
    : symbol_(symbol)
    , tick_size_(tick_size)
    , order_allocator_()
    , level_allocator_()
{
    // All memory pre-allocated in constructors
    // Ready for zero-allocation trading
}

// =============================================================================
// ADD ORDER
// =============================================================================

uint64_t LimitOrderBook::add_order(
    Side side,
    OrderType type,
    uint64_t price_ticks,
    uint32_t quantity,
    TimeInForce tif,
    uint64_t client_order_id,
    std::vector<Trade>* trades_out
) {
    // -------------------------------------------------------------------------
    // CRITICAL PATH START - Every nanosecond counts
    // -------------------------------------------------------------------------
    const Timestamp entry_time = std::chrono::steady_clock::now();
    
    // Validate inputs
    if (quantity == 0) {
        return 0;  // Rejected
    }
    
    // Allocate from pool - O(1), no malloc
    Order* order = order_allocator_.allocate();
    if (!order) {
        return 0;  // Pool exhausted
    }
    
    // Initialize order
    const uint64_t order_id = next_order_id_.fetch_add(1, std::memory_order_relaxed);
    
    order->order_id = order_id;
    order->client_order_id = client_order_id ? client_order_id : order_id;
    order->price_ticks = price_ticks;
    order->quantity = quantity;
    order->filled_quantity = 0;
    order->visible_quantity = quantity;  // Full visibility by default
    order->hidden_quantity = 0;
    order->side = side;
    order->type = type;
    order->status = OrderStatus::NEW;
    order->tif = tif;
    order->entry_time = entry_time;
    order->last_update_time = entry_time;
    order->price_level = nullptr;
    order->list_node.prev = nullptr;
    order->list_node.next = nullptr;
    
    // Add to order map for O(1) lookup
    order_map_.insert(order_id, order);
    
    // -------------------------------------------------------------------------
    // MATCHING ENGINE - Hot path
    // -------------------------------------------------------------------------
    
    // Attempt matching for marketable orders
    if (type == OrderType::MARKET || type == OrderType::IOC || type == OrderType::FOK) {
        // Market orders must be fully matched or killed
        if (type == OrderType::FOK) {
            // Check if FOK can be fully filled before matching
            uint32_t available = 0;
            if (side == Side::BUY) {
                for (auto& level : asks_) {
                    if (price_ticks >= level.price_ticks || type == OrderType::MARKET) {
                        available += level.total_quantity;
                        if (available >= quantity) break;
                    } else {
                        break;
                    }
                }
            } else {
                for (auto& level : bids_) {
                    if (price_ticks <= level.price_ticks || type == OrderType::MARKET) {
                        available += level.total_quantity;
                        if (available >= quantity) break;
                    } else {
                        break;
                    }
                }
            }
            
            if (available < quantity) {
                // Cannot fill FOK, reject
                order->status = OrderStatus::REJECTED;
                order_map_.remove(order_id);
                order_allocator_.deallocate(order);
                return 0;
            }
        }
        
        match_order(order, trades_out);
        
        // Handle remaining quantity based on order type
        if (order->remaining_qty() > 0) {
            if (type == OrderType::MARKET || type == OrderType::IOC) {
                // Cancel remaining
                order->status = OrderStatus::CANCELLED;
            }
        }
    } else {
        // Limit order - match then rest
        match_order(order, trades_out);
    }
    
    // -------------------------------------------------------------------------
    // REST ON BOOK (for limit orders with remaining quantity)
    // -------------------------------------------------------------------------
    
    if (type == OrderType::LIMIT && order->remaining_qty() > 0 && 
        order->status != OrderStatus::CANCELLED) {
        
        // Get or create price level - O(log P)
        PriceLevel* level = get_or_create_level(side, price_ticks);
        
        // Add to level - O(1)
        level->add_order(order);
        
        if (order->filled_quantity > 0) {
            order->status = OrderStatus::PARTIAL;
        }
    } else if (order->remaining_qty() == 0) {
        order->status = OrderStatus::FILLED;
    }
    
    // -------------------------------------------------------------------------
    // RECORD LATENCY
    // -------------------------------------------------------------------------
    
    const int64_t latency = elapsed_ns(entry_time);
    latency_stats_.record(latency);
    
    // Callback
    if (order_callback_) {
        order_callback_(*order);
    }
    
    return order_id;
}

// =============================================================================
// CANCEL ORDER
// =============================================================================

bool LimitOrderBook::cancel_order(uint64_t order_id) {
    // O(1) lookup
    Order* order = order_map_.find(order_id);
    if (!order) {
        return false;
    }
    
    if (order->status == OrderStatus::FILLED || 
        order->status == OrderStatus::CANCELLED) {
        return false;
    }
    
    // Remove from price level - O(1) via intrusive list
    if (order->price_level) {
        PriceLevel* level = order->price_level;
        level->remove_order(order);
        
        // Remove empty price level - O(log P)
        remove_level_if_empty(level);
    }
    
    order->status = OrderStatus::CANCELLED;
    order->last_update_time = std::chrono::steady_clock::now();
    
    if (order_callback_) {
        order_callback_(*order);
    }
    
    return true;
}

// =============================================================================
// MODIFY ORDER (Cancel/Replace)
// =============================================================================

uint64_t LimitOrderBook::modify_order(
    uint64_t order_id,
    uint64_t new_price_ticks,
    uint32_t new_quantity,
    std::vector<Trade>* trades_out
) {
    Order* old_order = order_map_.find(order_id);
    if (!old_order) {
        return 0;
    }
    
    if (old_order->status == OrderStatus::FILLED ||
        old_order->status == OrderStatus::CANCELLED) {
        return 0;
    }
    
    // Quantity can only be reduced without losing priority
    bool loses_priority = (new_price_ticks != old_order->price_ticks) ||
                          (new_quantity > old_order->remaining_qty());
    
    if (!loses_priority && new_quantity <= old_order->remaining_qty()) {
        // In-place modification - keep time priority
        if (old_order->price_level) {
            old_order->price_level->total_quantity -= old_order->remaining_qty();
            old_order->quantity = old_order->filled_quantity + new_quantity;
            old_order->price_level->total_quantity += old_order->remaining_qty();
        }
        old_order->last_update_time = std::chrono::steady_clock::now();
        
        if (order_callback_) {
            order_callback_(*old_order);
        }
        return order_id;
    }
    
    // Cancel and replace - loses time priority
    Side side = old_order->side;
    OrderType type = old_order->type;
    TimeInForce tif = old_order->tif;
    uint64_t client_id = old_order->client_order_id;
    
    cancel_order(order_id);
    
    return add_order(side, type, new_price_ticks, new_quantity, tif, client_id, trades_out);
}

// =============================================================================
// MATCHING ENGINE
// =============================================================================

void LimitOrderBook::match_order(Order* order, std::vector<Trade>* trades_out) {
    if (order->side == Side::BUY) {
        match_aggressive<Side::BUY>(order, trades_out);
    } else {
        match_aggressive<Side::SELL>(order, trades_out);
    }
}

template<Side AggSide>
void LimitOrderBook::match_aggressive(Order* aggressor, std::vector<Trade>* trades_out) {
    // Select opposite side book
    auto& passive_book = []() -> auto& {
        if constexpr (AggSide == Side::BUY) {
            return asks_;
        } else {
            return bids_;
        }
    }();
    
    while (aggressor->remaining_qty() > 0) {
        PriceLevel* best_level = passive_book.best();
        if (!best_level) {
            break;  // No liquidity
        }
        
        // Price check for limit orders
        if (aggressor->type == OrderType::LIMIT) {
            if constexpr (AggSide == Side::BUY) {
                if (best_level->price_ticks > aggressor->price_ticks) {
                    break;  // Ask price too high
                }
            } else {
                if (best_level->price_ticks < aggressor->price_ticks) {
                    break;  // Bid price too low
                }
            }
        }
        
        // Match against all orders at this price level (FIFO)
        while (aggressor->remaining_qty() > 0 && !best_level->empty()) {
            Order* passive = best_level->orders.front();
            
            // Calculate fill quantity
            uint32_t fill_qty = std::min(aggressor->remaining_qty(), passive->remaining_qty());
            
            // Execute trade
            execute_trade(aggressor, passive, fill_qty, best_level->price_ticks, trades_out);
            
            // Update passive order
            if (passive->remaining_qty() == 0) {
                passive->status = OrderStatus::FILLED;
                best_level->remove_order(passive);
                
                if (order_callback_) {
                    order_callback_(*passive);
                }
            } else {
                passive->status = OrderStatus::PARTIAL;
            }
        }
        
        // Remove empty price level
        if (best_level->empty()) {
            remove_level_if_empty(best_level);
        }
    }
    
    // Update aggressor status
    if (aggressor->remaining_qty() == 0) {
        aggressor->status = OrderStatus::FILLED;
    } else if (aggressor->filled_quantity > 0) {
        aggressor->status = OrderStatus::PARTIAL;
    }
}

// Explicit instantiations
template void LimitOrderBook::match_aggressive<Side::BUY>(Order*, std::vector<Trade>*);
template void LimitOrderBook::match_aggressive<Side::SELL>(Order*, std::vector<Trade>*);

void LimitOrderBook::execute_trade(
    Order* aggressive,
    Order* passive,
    uint32_t fill_qty,
    uint64_t price,
    std::vector<Trade>* trades_out
) {
    // Update quantities
    aggressive->filled_quantity += fill_qty;
    passive->filled_quantity += fill_qty;
    aggressive->last_update_time = std::chrono::steady_clock::now();
    passive->last_update_time = aggressive->last_update_time;
    
    // Update price level aggregates
    if (passive->price_level) {
        uint32_t visible_fill = std::min(fill_qty, passive->displayable_qty());
        passive->price_level->update_quantity_after_fill(fill_qty, visible_fill);
    }
    
    // Create trade record
    Trade trade{
        .trade_id = next_trade_id_.fetch_add(1, std::memory_order_relaxed),
        .buy_order_id = aggressive->is_buy() ? aggressive->order_id : passive->order_id,
        .sell_order_id = aggressive->is_sell() ? aggressive->order_id : passive->order_id,
        .price_ticks = price,
        .quantity = fill_qty,
        .aggressor_side = aggressive->side,
        .timestamp = aggressive->last_update_time,
        .latency_ns = elapsed_ns(aggressive->entry_time)
    };
    
    ++trade_count_;
    
    if (trades_out) {
        trades_out->push_back(trade);
    }
    
    if (trade_callback_) {
        trade_callback_(trade);
    }
}

// =============================================================================
// PRICE LEVEL MANAGEMENT
// =============================================================================

PriceLevel* LimitOrderBook::get_or_create_level(Side side, uint64_t price) {
    // First, try to find existing level - O(log P)
    PriceLevel* level = nullptr;
    
    if (side == Side::BUY) {
        level = bids_.find(price);
        if (!level) {
            level = level_allocator_.allocate();
            if (level) {
                level->price_ticks = price;
                level->side = side;
                level->total_quantity = 0;
                level->visible_quantity = 0;
                level->order_count = 0;
                level->forward.fill(nullptr);
                level->level = 0;
                bids_.insert(level);
            }
        }
    } else {
        level = asks_.find(price);
        if (!level) {
            level = level_allocator_.allocate();
            if (level) {
                level->price_ticks = price;
                level->side = side;
                level->total_quantity = 0;
                level->visible_quantity = 0;
                level->order_count = 0;
                level->forward.fill(nullptr);
                level->level = 0;
                asks_.insert(level);
            }
        }
    }
    
    return level;
}

void LimitOrderBook::remove_level_if_empty(PriceLevel* level) {
    if (!level || !level->empty()) {
        return;
    }
    
    if (level->side == Side::BUY) {
        bids_.remove(level);
    } else {
        asks_.remove(level);
    }
    
    level_allocator_.deallocate(level);
}

// =============================================================================
// MARKET DATA QUERIES
// =============================================================================

uint64_t LimitOrderBook::best_bid() const noexcept {
    PriceLevel* best = bids_.best();
    return best ? best->price_ticks : 0;
}

uint64_t LimitOrderBook::best_ask() const noexcept {
    PriceLevel* best = asks_.best();
    return best ? best->price_ticks : std::numeric_limits<uint64_t>::max();
}

uint64_t LimitOrderBook::spread() const noexcept {
    uint64_t bid = best_bid();
    uint64_t ask = best_ask();
    if (bid == 0 || ask == std::numeric_limits<uint64_t>::max()) {
        return 0;
    }
    return ask - bid;
}

uint64_t LimitOrderBook::mid_price() const noexcept {
    uint64_t bid = best_bid();
    uint64_t ask = best_ask();
    if (bid == 0 || ask == std::numeric_limits<uint64_t>::max()) {
        return 0;
    }
    return (bid + ask) / 2;
}

uint64_t LimitOrderBook::bid_quantity_at(uint64_t price) const noexcept {
    PriceLevel* level = bids_.find(price);
    return level ? level->total_quantity : 0;
}

uint64_t LimitOrderBook::ask_quantity_at(uint64_t price) const noexcept {
    PriceLevel* level = asks_.find(price);
    return level ? level->total_quantity : 0;
}

void LimitOrderBook::get_bids(std::vector<LevelInfo>& out, size_t depth) const {
    out.clear();
    out.reserve(depth);
    
    size_t count = 0;
    for (const auto& level : bids_) {
        if (count >= depth) break;
        out.push_back({level.price_ticks, level.total_quantity, level.order_count});
        ++count;
    }
}

void LimitOrderBook::get_asks(std::vector<LevelInfo>& out, size_t depth) const {
    out.clear();
    out.reserve(depth);
    
    size_t count = 0;
    for (const auto& level : asks_) {
        if (count >= depth) break;
        out.push_back({level.price_ticks, level.total_quantity, level.order_count});
        ++count;
    }
}

// =============================================================================
// LIFECYCLE
// =============================================================================

void LimitOrderBook::clear() {
    bids_.clear();
    asks_.clear();
    order_map_.clear();
    order_allocator_.reset();
    level_allocator_.reset();
    latency_stats_.reset();
    trade_count_ = 0;
    next_order_id_.store(1, std::memory_order_relaxed);
    next_trade_id_.store(1, std::memory_order_relaxed);
}

// =============================================================================
// SNAPSHOT
// =============================================================================

OrderBookSnapshot create_snapshot(const LimitOrderBook& book, size_t depth) {
    OrderBookSnapshot snapshot;
    snapshot.symbol = book.symbol();
    snapshot.best_bid = book.best_bid();
    snapshot.best_ask = book.best_ask();
    snapshot.spread = book.spread();
    snapshot.mid_price = book.mid_price();
    snapshot.total_orders = book.total_orders();
    snapshot.total_trades = book.total_trades();
    snapshot.stats = book.get_latency_stats();
    snapshot.timestamp = std::chrono::steady_clock::now();
    
    book.get_bids(snapshot.bids, depth);
    book.get_asks(snapshot.asks, depth);
    
    return snapshot;
}

} // namespace arbor::orderbook
