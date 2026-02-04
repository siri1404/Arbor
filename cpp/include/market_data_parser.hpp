#pragma once

#include <string>
#include <vector>
#include <chrono>

namespace arbor::parser {

// High-performance market data parser for various formats

// FIX protocol message
struct FIXMessage {
    std::string msg_type;
    std::string symbol;
    double price;
    uint64_t quantity;
    char side;  // 'B' or 'S'
    std::chrono::nanoseconds timestamp;
    int64_t parse_time_ns;
};

// CSV tick data
struct CSVTick {
    std::string symbol;
    double bid;
    double ask;
    uint64_t bid_size;
    uint64_t ask_size;
    std::chrono::nanoseconds timestamp;
};

// Parse FIX 4.2 message
// Example: "8=FIX.4.2|9=100|35=D|49=SENDER|56=TARGET|34=1|52=20240204-12:00:00|55=AAPL|54=1|38=100|40=2|44=150.50|"
FIXMessage parse_fix_message(const std::string& fix_msg);

// Parse CSV line (optimized for low latency)
// Format: "AAPL,150.25,150.26,1000,1500,2024-02-04T12:00:00.123456789Z"
CSVTick parse_csv_line(const std::string& csv_line);

// Batch parsing with SIMD optimization hints
std::vector<FIXMessage> parse_fix_batch(const std::vector<std::string>& messages);
std::vector<CSVTick> parse_csv_batch(const std::vector<std::string>& lines);

// Latency statistics for parser performance
struct ParserStats {
    size_t messages_parsed;
    int64_t total_time_ns;
    int64_t min_parse_ns;
    int64_t max_parse_ns;
    double avg_parse_ns;
    
    void record(int64_t parse_time);
    void reset();
};

} // namespace arbor::parser
