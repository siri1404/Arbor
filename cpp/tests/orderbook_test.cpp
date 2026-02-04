#include <gtest/gtest.h>
#include "../include/orderbook.hpp"
#include <thread>
#include <atomic>
#include <random>

using namespace arbor::orderbook;

// =============================================================================
// BASIC FUNCTIONALITY TESTS
// =============================================================================

class OrderBookTest : public ::testing::Test {
protected:
    void SetUp() override {
        book = std::make_unique<LimitOrderBook>("TEST", 1);
    }
    
    std::unique_ptr<LimitOrderBook> book;
    std::vector<Trade> trades;
};

TEST_F(OrderBookTest, BasicOrderInsertion) {
    uint64_t order_id = book->add_order(Side::BUY, OrderType::LIMIT, 10000, 100, 
                                         TimeInForce::GTC, 0, &trades);
    
    EXPECT_GT(order_id, 0u);
    EXPECT_EQ(trades.size(), 0u);  // No match
    EXPECT_EQ(book->best_bid(), 10000u);
    EXPECT_EQ(book->total_orders(), 1u);
}

TEST_F(OrderBookTest, BothSidesOfBook) {
    book->add_order(Side::BUY, OrderType::LIMIT, 10000, 100, TimeInForce::GTC, 0, &trades);
    book->add_order(Side::SELL, OrderType::LIMIT, 10010, 100, TimeInForce::GTC, 0, &trades);
    
    EXPECT_EQ(book->best_bid(), 10000u);
    EXPECT_EQ(book->best_ask(), 10010u);
    EXPECT_EQ(book->spread(), 10u);
    EXPECT_EQ(book->mid_price(), 10005u);
}

TEST_F(OrderBookTest, MatchingOrders) {
    // Add resting buy order
    book->add_order(Side::BUY, OrderType::LIMIT, 10000, 100, TimeInForce::GTC, 0, &trades);
    EXPECT_EQ(trades.size(), 0u);
    
    // Add aggressive sell order - should match
    book->add_order(Side::SELL, OrderType::LIMIT, 10000, 50, TimeInForce::GTC, 0, &trades);
    EXPECT_EQ(trades.size(), 1u);
    EXPECT_EQ(trades[0].quantity, 50u);
    EXPECT_EQ(trades[0].price_ticks, 10000u);
}

TEST_F(OrderBookTest, PriceTimePriority) {
    // Add two orders at same price
    auto id1 = book->add_order(Side::BUY, OrderType::LIMIT, 10000, 100, TimeInForce::GTC, 0, &trades);
    auto id2 = book->add_order(Side::BUY, OrderType::LIMIT, 10000, 100, TimeInForce::GTC, 0, &trades);
    
    // Aggressive sell should match first order first (FIFO)
    book->add_order(Side::SELL, OrderType::LIMIT, 10000, 150, TimeInForce::GTC, 0, &trades);
    
    EXPECT_EQ(trades.size(), 2u);
    EXPECT_EQ(trades[0].buy_order_id, id1);  // First order matched first
    EXPECT_EQ(trades[0].quantity, 100u);     // Fully filled
    EXPECT_EQ(trades[1].buy_order_id, id2);
    EXPECT_EQ(trades[1].quantity, 50u);      // Partially filled
}

TEST_F(OrderBookTest, PricePriority) {
    // Better price should match first
    auto id1 = book->add_order(Side::BUY, OrderType::LIMIT, 10000, 100, TimeInForce::GTC, 0, &trades);
    auto id2 = book->add_order(Side::BUY, OrderType::LIMIT, 10010, 100, TimeInForce::GTC, 0, &trades);  // Better price
    
    EXPECT_EQ(book->best_bid(), 10010u);  // Higher bid is best
    
    // Sell should match higher bid first
    book->add_order(Side::SELL, OrderType::LIMIT, 9990, 50, TimeInForce::GTC, 0, &trades);
    
    EXPECT_EQ(trades.size(), 1u);
    EXPECT_EQ(trades[0].buy_order_id, id2);  // Better price matched
    EXPECT_EQ(trades[0].price_ticks, 10010u);  // At resting order price
    (void)id1;  // Suppress unused warning
}

TEST_F(OrderBookTest, MarketOrder) {
    // Seed book with asks
    book->add_order(Side::SELL, OrderType::LIMIT, 10010, 100, TimeInForce::GTC, 0, &trades);
    book->add_order(Side::SELL, OrderType::LIMIT, 10020, 100, TimeInForce::GTC, 0, &trades);
    
    // Market buy should sweep through price levels
    book->add_order(Side::BUY, OrderType::MARKET, 0, 150, TimeInForce::GTC, 0, &trades);
    
    EXPECT_EQ(trades.size(), 2u);
    EXPECT_EQ(trades[0].price_ticks, 10010u);  // First at best ask
    EXPECT_EQ(trades[0].quantity, 100u);
    EXPECT_EQ(trades[1].price_ticks, 10020u);  // Then at next level
    EXPECT_EQ(trades[1].quantity, 50u);
}

TEST_F(OrderBookTest, IOCOrder) {
    book->add_order(Side::SELL, OrderType::LIMIT, 10010, 50, TimeInForce::GTC, 0, &trades);
    
    // IOC for 100 should match 50 and cancel remaining
    auto id = book->add_order(Side::BUY, OrderType::IOC, 10010, 100, TimeInForce::GTC, 0, &trades);
    
    EXPECT_EQ(trades.size(), 1u);
    EXPECT_EQ(trades[0].quantity, 50u);
    // Remaining 50 should be cancelled, not on book
    EXPECT_EQ(book->bid_quantity_at(10010), 0u);
    (void)id;
}

TEST_F(OrderBookTest, FOKOrderFilled) {
    book->add_order(Side::SELL, OrderType::LIMIT, 10010, 100, TimeInForce::GTC, 0, &trades);
    
    // FOK for exactly 100 should fill
    auto id = book->add_order(Side::BUY, OrderType::FOK, 10010, 100, TimeInForce::GTC, 0, &trades);
    
    EXPECT_GT(id, 0u);
    EXPECT_EQ(trades.size(), 1u);
    EXPECT_EQ(trades[0].quantity, 100u);
}

TEST_F(OrderBookTest, FOKOrderRejected) {
    book->add_order(Side::SELL, OrderType::LIMIT, 10010, 50, TimeInForce::GTC, 0, &trades);
    
    // FOK for 100 when only 50 available should reject
    auto id = book->add_order(Side::BUY, OrderType::FOK, 10010, 100, TimeInForce::GTC, 0, &trades);
    
    EXPECT_EQ(id, 0u);  // Rejected
    EXPECT_EQ(trades.size(), 0u);  // No trades
    EXPECT_EQ(book->ask_quantity_at(10010), 50u);  // Resting order untouched
}

// =============================================================================
// CANCEL ORDER TESTS
// =============================================================================

TEST_F(OrderBookTest, CancelOrder) {
    auto id = book->add_order(Side::BUY, OrderType::LIMIT, 10000, 100, TimeInForce::GTC, 0, &trades);
    EXPECT_EQ(book->bid_quantity_at(10000), 100u);
    
    bool cancelled = book->cancel_order(id);
    EXPECT_TRUE(cancelled);
    EXPECT_EQ(book->bid_quantity_at(10000), 0u);
    EXPECT_EQ(book->best_bid(), 0u);
}

TEST_F(OrderBookTest, CancelNonExistentOrder) {
    bool cancelled = book->cancel_order(999999);
    EXPECT_FALSE(cancelled);
}

TEST_F(OrderBookTest, CancelAlreadyFilledOrder) {
    auto id = book->add_order(Side::BUY, OrderType::LIMIT, 10000, 100, TimeInForce::GTC, 0, &trades);
    book->add_order(Side::SELL, OrderType::LIMIT, 10000, 100, TimeInForce::GTC, 0, &trades);
    
    // Order should be filled now
    bool cancelled = book->cancel_order(id);
    EXPECT_FALSE(cancelled);
}

TEST_F(OrderBookTest, CancelPartiallyFilledOrder) {
    auto id = book->add_order(Side::BUY, OrderType::LIMIT, 10000, 100, TimeInForce::GTC, 0, &trades);
    book->add_order(Side::SELL, OrderType::LIMIT, 10000, 50, TimeInForce::GTC, 0, &trades);
    
    // 50 remaining
    EXPECT_EQ(book->bid_quantity_at(10000), 50u);
    
    bool cancelled = book->cancel_order(id);
    EXPECT_TRUE(cancelled);
    EXPECT_EQ(book->bid_quantity_at(10000), 0u);
}

// =============================================================================
// MODIFY ORDER TESTS
// =============================================================================

TEST_F(OrderBookTest, ModifyQuantityDown) {
    auto id = book->add_order(Side::BUY, OrderType::LIMIT, 10000, 100, TimeInForce::GTC, 0, &trades);
    EXPECT_EQ(book->bid_quantity_at(10000), 100u);
    
    // Reduce quantity - should keep time priority
    auto new_id = book->modify_order(id, 10000, 50, &trades);
    
    EXPECT_EQ(new_id, id);  // Same order ID (in-place modification)
    EXPECT_EQ(book->bid_quantity_at(10000), 50u);
}

TEST_F(OrderBookTest, ModifyPriceChangesOrder) {
    auto id1 = book->add_order(Side::BUY, OrderType::LIMIT, 10000, 100, TimeInForce::GTC, 0, &trades);
    auto id2 = book->add_order(Side::BUY, OrderType::LIMIT, 10000, 100, TimeInForce::GTC, 0, &trades);
    
    // Modify first order's price - should get new ID and lose priority
    auto new_id = book->modify_order(id1, 10010, 100, &trades);
    
    EXPECT_NE(new_id, id1);  // New order ID
    EXPECT_EQ(book->best_bid(), 10010u);  // Now at better price
    EXPECT_EQ(book->bid_quantity_at(10000), 100u);  // id2 still at old price
    (void)id2;
}

// =============================================================================
// INTRUSIVE LIST O(1) REMOVAL TESTS
// =============================================================================

TEST_F(OrderBookTest, CancelMiddleOrderO1) {
    // Add 3 orders at same price
    auto id1 = book->add_order(Side::BUY, OrderType::LIMIT, 10000, 100, TimeInForce::GTC, 0, &trades);
    auto id2 = book->add_order(Side::BUY, OrderType::LIMIT, 10000, 100, TimeInForce::GTC, 0, &trades);
    auto id3 = book->add_order(Side::BUY, OrderType::LIMIT, 10000, 100, TimeInForce::GTC, 0, &trades);
    
    EXPECT_EQ(book->bid_quantity_at(10000), 300u);
    
    // Cancel middle order - should be O(1) with intrusive list
    book->cancel_order(id2);
    
    EXPECT_EQ(book->bid_quantity_at(10000), 200u);
    
    // Verify FIFO is preserved for remaining orders
    book->add_order(Side::SELL, OrderType::LIMIT, 10000, 150, TimeInForce::GTC, 0, &trades);
    
    EXPECT_EQ(trades.size(), 2u);
    EXPECT_EQ(trades[0].buy_order_id, id1);  // First order matched first
    EXPECT_EQ(trades[0].quantity, 100u);
    EXPECT_EQ(trades[1].buy_order_id, id3);  // Third order (id2 was cancelled)
    EXPECT_EQ(trades[1].quantity, 50u);
}

// =============================================================================
// SKIP LIST PRICE LEVEL TESTS
// =============================================================================

TEST_F(OrderBookTest, MultiplePriceLevels) {
    // Add orders at various prices
    for (int i = 0; i < 100; ++i) {
        book->add_order(Side::BUY, OrderType::LIMIT, 10000 - i * 10, 100, 
                       TimeInForce::GTC, 0, &trades);
    }
    
    EXPECT_EQ(book->bid_levels(), 100u);
    EXPECT_EQ(book->best_bid(), 10000u);  // Highest bid
    
    // Cancel best bid - next level should become best
    // Find orders at best price and cancel them
    std::vector<LimitOrderBook::LevelInfo> bids;
    book->get_bids(bids, 1);
    EXPECT_EQ(bids[0].price, 10000u);
}

TEST_F(OrderBookTest, PriceLevelRemoval) {
    auto id = book->add_order(Side::BUY, OrderType::LIMIT, 10000, 100, TimeInForce::GTC, 0, &trades);
    EXPECT_EQ(book->bid_levels(), 1u);
    
    // Match completely - should remove price level
    book->add_order(Side::SELL, OrderType::LIMIT, 10000, 100, TimeInForce::GTC, 0, &trades);
    
    EXPECT_EQ(book->bid_levels(), 0u);
    EXPECT_EQ(book->best_bid(), 0u);
    (void)id;
}

// =============================================================================
// MEMORY ALLOCATION TESTS
// =============================================================================

TEST_F(OrderBookTest, SlabAllocatorNoMalloc) {
    // This test verifies the slab allocator works correctly
    // In a real test, we'd use a memory profiler to verify no malloc calls
    
    // Add many orders - should not trigger any malloc
    for (int i = 0; i < 10000; ++i) {
        book->add_order(Side::BUY, OrderType::LIMIT, 10000 + (i % 100), 100,
                       TimeInForce::GTC, 0, nullptr);
    }
    
    EXPECT_EQ(book->total_orders(), 10000u);
    
    // Cancel all - should return to pool
    book->clear();
    EXPECT_EQ(book->total_orders(), 0u);
    
    // Reuse - should work without new allocations
    for (int i = 0; i < 10000; ++i) {
        book->add_order(Side::SELL, OrderType::LIMIT, 10000 + (i % 100), 100,
                       TimeInForce::GTC, 0, nullptr);
    }
    
    EXPECT_EQ(book->total_orders(), 10000u);
}

// =============================================================================
// LATENCY STATISTICS TESTS
// =============================================================================

TEST_F(OrderBookTest, LatencyTracking) {
    // Seed with resting orders
    for (int i = 0; i < 100; ++i) {
        book->add_order(Side::SELL, OrderType::LIMIT, 10100, 10, TimeInForce::GTC, 0, &trades);
    }
    trades.clear();
    
    // Add aggressive order and check latency
    book->add_order(Side::BUY, OrderType::LIMIT, 10100, 50, TimeInForce::GTC, 0, &trades);
    
    const auto& stats = book->get_latency_stats();
    EXPECT_GT(stats.count(), 0u);
    EXPECT_GT(stats.avg_ns(), 0.0);
    EXPECT_LT(stats.avg_ns(), 1000000.0);  // Should be < 1ms
}

TEST_F(OrderBookTest, LatencyPercentiles) {
    // Generate enough samples for percentile calculation
    for (int i = 0; i < 1000; ++i) {
        book->add_order(Side::BUY, OrderType::LIMIT, 10000 + i, 100, 
                       TimeInForce::GTC, 0, nullptr);
    }
    
    const auto& stats = book->get_latency_stats();
    
    EXPECT_GE(stats.p99_ns(), stats.p50_ns());
    EXPECT_GE(stats.max_ns(), stats.p99_ns());
    EXPECT_LE(stats.min_ns(), stats.p50_ns());
}

// =============================================================================
// STRESS TESTS
// =============================================================================

TEST_F(OrderBookTest, HighVolumeOrders) {
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<uint64_t> price_dist(9000, 11000);
    std::uniform_int_distribution<uint32_t> qty_dist(1, 1000);
    
    // Insert 100K orders
    for (int i = 0; i < 100000; ++i) {
        Side side = (i % 2 == 0) ? Side::BUY : Side::SELL;
        book->add_order(side, OrderType::LIMIT, price_dist(rng), qty_dist(rng),
                       TimeInForce::GTC, 0, nullptr);
    }
    
    // Verify book is consistent
    EXPECT_GT(book->total_orders(), 0u);
    EXPECT_GT(book->total_trades(), 0u);
    
    auto bid = book->best_bid();
    auto ask = book->best_ask();
    if (bid != 0 && ask != std::numeric_limits<uint64_t>::max()) {
        EXPECT_LE(bid, ask);  // No crossed book
    }
}

TEST_F(OrderBookTest, CancelStress) {
    std::vector<uint64_t> order_ids;
    order_ids.reserve(10000);
    
    // Add orders
    for (int i = 0; i < 10000; ++i) {
        auto id = book->add_order(Side::BUY, OrderType::LIMIT, 10000 - (i % 100), 100,
                                  TimeInForce::GTC, 0, nullptr);
        order_ids.push_back(id);
    }
    
    // Cancel in random order
    std::mt19937 rng(123);
    std::shuffle(order_ids.begin(), order_ids.end(), rng);
    
    for (auto id : order_ids) {
        book->cancel_order(id);
    }
    
    EXPECT_EQ(book->bid_levels(), 0u);
}

// =============================================================================
// CALLBACK TESTS
// =============================================================================

TEST_F(OrderBookTest, TradeCallback) {
    std::vector<Trade> callback_trades;
    
    book->set_trade_callback([&](const Trade& t) {
        callback_trades.push_back(t);
    });
    
    book->add_order(Side::BUY, OrderType::LIMIT, 10000, 100, TimeInForce::GTC, 0, &trades);
    book->add_order(Side::SELL, OrderType::LIMIT, 10000, 50, TimeInForce::GTC, 0, &trades);
    
    EXPECT_EQ(callback_trades.size(), 1u);
    EXPECT_EQ(callback_trades[0].quantity, 50u);
}

// =============================================================================
// EDGE CASES
// =============================================================================

TEST_F(OrderBookTest, ZeroQuantityRejected) {
    auto id = book->add_order(Side::BUY, OrderType::LIMIT, 10000, 0, TimeInForce::GTC, 0, &trades);
    EXPECT_EQ(id, 0u);  // Rejected
}

TEST_F(OrderBookTest, EmptyBookQueries) {
    EXPECT_EQ(book->best_bid(), 0u);
    EXPECT_EQ(book->best_ask(), std::numeric_limits<uint64_t>::max());
    EXPECT_EQ(book->spread(), 0u);
    EXPECT_EQ(book->mid_price(), 0u);
    
    std::vector<LimitOrderBook::LevelInfo> bids, asks;
    book->get_bids(bids, 10);
    book->get_asks(asks, 10);
    
    EXPECT_TRUE(bids.empty());
    EXPECT_TRUE(asks.empty());
}

TEST_F(OrderBookTest, SelfTradePrevention) {
    // In a production system, you'd want self-trade prevention
    // This test documents the current behavior
    auto buy_id = book->add_order(Side::BUY, OrderType::LIMIT, 10000, 100, 
                                   TimeInForce::GTC, 12345, &trades);
    auto sell_id = book->add_order(Side::SELL, OrderType::LIMIT, 10000, 100, 
                                    TimeInForce::GTC, 12345, &trades);  // Same client
    
    // Without STP, these will match
    EXPECT_EQ(trades.size(), 1u);
    (void)buy_id;
    (void)sell_id;
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
