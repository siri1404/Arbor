#include "market_data_parser.hpp"
#include <sstream>
#include <algorithm>
#include <chrono>

namespace arbor::parser {

FIXMessage parse_fix_message(const std::string& fix_msg) {
    const auto start = std::chrono::steady_clock::now();
    
    FIXMessage msg{};
    
    // Simple FIX parser - in production, use a proper FIX library
    std::istringstream ss(fix_msg);
    std::string field;
    
    while (std::getline(ss, field, '|')) {
        if (field.empty()) continue;
        
        size_t eq = field.find('=');
        if (eq == std::string::npos) continue;
        
        std::string tag = field.substr(0, eq);
        std::string value = field.substr(eq + 1);
        
        if (tag == "35") {
            msg.msg_type = value;
        } else if (tag == "55") {
            msg.symbol = value;
        } else if (tag == "44") {
            msg.price = std::stod(value);
        } else if (tag == "38") {
            msg.quantity = std::stoull(value);
        } else if (tag == "54") {
            msg.side = (value == "1") ? 'B' : 'S';
        }
    }
    
    const auto end = std::chrono::steady_clock::now();
    msg.parse_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    msg.timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    );
    
    return msg;
}

CSVTick parse_csv_line(const std::string& csv_line) {
    const auto start = std::chrono::steady_clock::now();
    
    CSVTick tick{};
    std::istringstream ss(csv_line);
    std::string field;
    int field_num = 0;
    
    while (std::getline(ss, field, ',')) {
        switch (field_num) {
            case 0: tick.symbol = field; break;
            case 1: tick.bid = std::stod(field); break;
            case 2: tick.ask = std::stod(field); break;
            case 3: tick.bid_size = std::stoull(field); break;
            case 4: tick.ask_size = std::stoull(field); break;
        }
        ++field_num;
    }
    
    tick.timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    );
    
    return tick;
}

std::vector<FIXMessage> parse_fix_batch(const std::vector<std::string>& messages) {
    std::vector<FIXMessage> results;
    results.reserve(messages.size());
    
    for (const auto& msg : messages) {
        results.push_back(parse_fix_message(msg));
    }
    
    return results;
}

std::vector<CSVTick> parse_csv_batch(const std::vector<std::string>& lines) {
    std::vector<CSVTick> results;
    results.reserve(lines.size());
    
    for (const auto& line : lines) {
        results.push_back(parse_csv_line(line));
    }
    
    return results;
}

void ParserStats::record(int64_t parse_time) {
    ++messages_parsed;
    total_time_ns += parse_time;
    
    if (messages_parsed == 1) {
        min_parse_ns = parse_time;
        max_parse_ns = parse_time;
    } else {
        min_parse_ns = std::min(min_parse_ns, parse_time);
        max_parse_ns = std::max(max_parse_ns, parse_time);
    }
    
    avg_parse_ns = static_cast<double>(total_time_ns) / messages_parsed;
}

void ParserStats::reset() {
    messages_parsed = 0;
    total_time_ns = 0;
    min_parse_ns = 0;
    max_parse_ns = 0;
    avg_parse_ns = 0.0;
}

} // namespace arbor::parser
