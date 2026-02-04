#include "../include/orderbook.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>

using namespace arbor::orderbook;

void benchmark_matching_latency() {
    std::cout << "\n=== ORDER BOOK MATCHING LATENCY BENCHMARK ===\n\n";
    
    LimitOrderBook book("AAPL", 1);
    std::vector<Trade> trades;
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<uint64_t> price_dist(14900, 15100);  // $149.00 - $151.00
    std::uniform_int_distribution<uint32_t> qty_dist(1, 1000);
    
    // Seed the book with resting orders
    std::cout << "Seeding order book with 1000 resting limit orders...\n";
    for (int i = 0; i < 500; ++i) {
        book.add_order(Side::BUY, OrderType::LIMIT, price_dist(rng) - 50, qty_dist(rng), trades);
        book.add_order(Side::SELL, OrderType::LIMIT, price_dist(rng) + 50, qty_dist(rng), trades);
    }
    trades.clear();
    
    std::cout << "Order book depth: " << book.total_orders() << " orders\n";
    std::cout << "Best bid: " << book.best_bid() << " | Best ask: " << book.best_ask() << "\n";
    std::cout << "Spread: " << book.spread() << " ticks\n\n";
    
    // Benchmark aggressive market orders (worst case - full matching)
    std::cout << "Running 10,000 aggressive market orders...\n";
    const int NUM_ORDERS = 10000;
    
    const auto start = std::chrono::steady_clock::now();
    
    for (int i = 0; i < NUM_ORDERS; ++i) {
        Side side = (i % 2 == 0) ? Side::BUY : Side::SELL;
        uint64_t best_ask = book.best_ask();
        uint64_t best_bid = book.best_bid();
        // Use safe price calculation - default to mid-range if no prices
        uint64_t price = (side == Side::BUY) 
            ? (best_ask > 0 ? best_ask + 100 : 15100)
            : (best_bid > 0 ? best_bid - 100 : 14900);
        book.add_order(side, OrderType::LIMIT, price, qty_dist(rng), trades);
    }
    
    const auto end = std::chrono::steady_clock::now();
    const int64_t total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    
    const auto& stats = book.get_latency_stats();
    
    std::cout << "\nRESULTS:\n";
    std::cout << "-------------------------------------------\n";
    std::cout << "Total orders processed:    " << NUM_ORDERS << "\n";
    std::cout << "Total trades executed:     " << book.total_trades() << "\n";
    std::cout << "Total time:                " << total_ns / 1000 << " us\n";
    std::cout << "Throughput:                " << (NUM_ORDERS * 1e9) / total_ns << " orders/sec\n";
    std::cout << "\nLATENCY STATISTICS:\n";
    std::cout << "-------------------------------------------\n";
    std::cout << "Average matching latency:  " << std::fixed << std::setprecision(2) 
              << stats.avg_ns() / 1000.0 << " us (" << stats.avg_ns() << " ns)\n";
    std::cout << "Minimum matching latency:  " << stats.min_ns / 1000.0 << " us (" 
              << stats.min_ns << " ns)\n";
    std::cout << "Maximum matching latency:  " << stats.max_ns / 1000.0 << " us (" 
              << stats.max_ns << " ns)\n";
    std::cout << "P99 matching latency:      " << stats.p99_ns() / 1000.0 << " us (" 
              << stats.p99_ns() << " ns)\n";
    
    if (stats.avg_ns() < 1000) {
        std::cout << "\n[EXCELLENT] Sub-microsecond average latency achieved!\n";
    } else if (stats.avg_ns() < 10000) {
        std::cout << "\n[GOOD] Average latency < 10 us\n";
    } else {
        std::cout << "\n[NOTE] Average latency > 10 us\n";
    }
    
    std::cout << "\nINDUSTRY BENCHMARK: < 1 us for low-latency trading systems\n";
    std::cout << "Status: " << (stats.avg_ns() < 1000 ? "[PRODUCTION READY]" : "[ACCEPTABLE]") << "\n";
}

void benchmark_throughput() {
    std::cout << "\n=== HIGH-FREQUENCY THROUGHPUT TEST ===\n\n";
    
    LimitOrderBook book("SPY", 1);
    std::vector<Trade> trades;
    std::mt19937_64 rng(123);
    std::uniform_int_distribution<uint64_t> price_dist(40000, 41000);
    std::uniform_int_distribution<uint32_t> qty_dist(100, 500);
    
    const int NUM_ORDERS = 50000;  // 50k orders (within order pool limit)
    std::cout << "Processing " << NUM_ORDERS << " orders at maximum speed...\n";
    
    const auto start = std::chrono::steady_clock::now();
    
    for (int i = 0; i < NUM_ORDERS; ++i) {
        Side side = (i % 2 == 0) ? Side::BUY : Side::SELL;
        OrderType type = (i % 10 == 0) ? OrderType::MARKET : OrderType::LIMIT;
        uint64_t price = price_dist(rng);
        
        book.add_order(side, type, price, qty_dist(rng), trades);
        
        if (i % 10000 == 0 && i > 0) {
            std::cout << "  Processed " << i << " orders...\n";
        }
    }
    
    const auto end = std::chrono::steady_clock::now();
    const int64_t total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    const double total_ms = total_ns / 1e6;
    const double orders_per_sec = (NUM_ORDERS * 1e9) / total_ns;
    
    std::cout << "\nTHROUGHPUT RESULTS:\n";
    std::cout << "-------------------------------------------\n";
    std::cout << "Total time:          " << std::fixed << std::setprecision(2) << total_ms << " ms\n";
    std::cout << "Orders per second:   " << std::setprecision(0) << orders_per_sec << "\n";
    std::cout << "Avg time per order:  " << std::setprecision(3) << (total_ns / NUM_ORDERS) << " ns\n";
    std::cout << "Final book depth:    " << book.total_orders() << " orders\n";
    std::cout << "Total trades:        " << book.total_trades() << " executions\n";
    
    std::cout << "\nINDUSTRY COMPARISON:\n";
    std::cout << "-------------------------------------------\n";
    std::cout << "Arbor Engine:            " << std::setprecision(0) << orders_per_sec << " orders/sec\n";
    std::cout << "NASDAQ ITCH Feed:        ~500,000 messages/sec\n";
    std::cout << "CME Globex:              ~1,000,000 messages/sec\n";
    std::cout << "Top-tier HFT systems:    ~5,000,000+ orders/sec\n";
}

int main() {
    std::cout << "============================================================\n";
    std::cout << "          ARBOR ORDER BOOK ENGINE - BENCHMARK SUITE         \n";
    std::cout << "============================================================\n";
    
    benchmark_matching_latency();
    benchmark_throughput();
    
    std::cout << "\nBenchmark complete.\n";
    std::cout << "Note: Compile with -O3 -march=native for optimal performance.\n\n";
    
    return 0;
}
