#include <gtest/gtest.h>
#include "../include/orderbook.hpp"

using namespace arbor::orderbook;

TEST(OrderBookTest, BasicOrderInsertion) {
    LimitOrderBook book("TEST", 1);
    std::vector<Trade> trades;
    
    uint64_t order_id = book.add_order(Side::BUY, OrderType::LIMIT, 10000, 100, trades);
    
    EXPECT_GT(order_id, 0);
    EXPECT_EQ(trades.size(), 0);  // No match
    EXPECT_EQ(book.best_bid(), 10000);
}

TEST(OrderBookTest, MatchingOrders) {
    LimitOrderBook book("TEST", 1);
    std::vector<Trade> trades;
    
    // Add resting buy order
    book.add_order(Side::BUY, OrderType::LIMIT, 10000, 100, trades);
    EXPECT_EQ(trades.size(), 0);
    
    // Add aggressive sell order - should match
    book.add_order(Side::SELL, OrderType::LIMIT, 10000, 50, trades);
    EXPECT_EQ(trades.size(), 1);
    EXPECT_EQ(trades[0].quantity, 50);
    EXPECT_EQ(trades[0].price_ticks, 10000);
}

TEST(OrderBookTest, PriceTimePriority) {
    LimitOrderBook book("TEST", 1);
    std::vector<Trade> trades;
    
    // Add two orders at same price
    auto id1 = book.add_order(Side::BUY, OrderType::LIMIT, 10000, 100, trades);
    auto id2 = book.add_order(Side::BUY, OrderType::LIMIT, 10000, 100, trades);
    
    // Aggressive sell should match first order first
    book.add_order(Side::SELL, OrderType::LIMIT, 10000, 150, trades);
    
    EXPECT_EQ(trades.size(), 2);
    EXPECT_EQ(trades[0].buy_order_id, id1);  // First order matched first
    EXPECT_EQ(trades[1].buy_order_id, id2);
}

TEST(OrderBookTest, LatencyTracking) {
    LimitOrderBook book("TEST", 1);
    std::vector<Trade> trades;
    
    // Seed with resting orders
    for (int i = 0; i < 100; ++i) {
        book.add_order(Side::SELL, OrderType::LIMIT, 10100, 10, trades);
    }
    trades.clear();
    
    // Add aggressive order and check latency
    book.add_order(Side::BUY, OrderType::LIMIT, 10100, 50, trades);
    
    const auto& stats = book.get_latency_stats();
    EXPECT_GT(stats.count, 0);
    EXPECT_GT(stats.avg_ns(), 0);
    EXPECT_LT(stats.avg_ns(), 100000);  // Should be < 100 μs
}

TEST(OrderBookTest, Spread) {
    LimitOrderBook book("TEST", 1);
    std::vector<Trade> trades;
    
    book.add_order(Side::BUY, OrderType::LIMIT, 10000, 100, trades);
    book.add_order(Side::SELL, OrderType::LIMIT, 10010, 100, trades);
    
    EXPECT_EQ(book.spread(), 10);
    EXPECT_EQ(book.mid_price(), 10005);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
