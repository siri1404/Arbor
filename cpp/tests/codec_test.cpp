/**
 * Unit tests for ITCH, OUCH, and SBE codecs
 */

#include "../include/itch_codec.hpp"
#include "../include/ouch_codec.hpp"
#include "../include/sbe_codec.hpp"
#include <cassert>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <vector>
#include <chrono>

using namespace arbor;

// =============================================================================
// TEST HELPERS
// =============================================================================

#define TEST(name) void test_##name()
#define RUN_TEST(name) do { \
    std::cout << "Running " #name "... "; \
    test_##name(); \
    std::cout << "PASSED\n"; \
} while(0)

#define ASSERT_EQ(a, b) do { \
    if ((a) != (b)) { \
        std::cerr << "\nAssertion failed: " << #a << " == " << #b \
                  << "\n  " << (a) << " != " << (b) << std::endl; \
        assert(false); \
    } \
} while(0)

#define ASSERT_NEAR(a, b, eps) do { \
    if (std::abs((a) - (b)) > (eps)) { \
        std::cerr << "\nAssertion failed: |" << #a << " - " << #b << "| < " << eps \
                  << "\n  |" << (a) << " - " << (b) << "| = " << std::abs((a)-(b)) << std::endl; \
        assert(false); \
    } \
} while(0)

// =============================================================================
// ITCH CODEC TESTS
// =============================================================================

TEST(itch_add_order) {
    // Build a valid ITCH Add Order message (type 'A')
    // Format: type(1) + locate(2) + tracking(2) + timestamp(6) + ref(8) + side(1) + shares(4) + stock(8) + price(4) = 36 bytes
    uint8_t buffer[36];
    std::memset(buffer, 0, sizeof(buffer));
    
    buffer[0] = 'A';  // Message type
    buffer[1] = 0; buffer[2] = 1;  // Stock locate = 1
    buffer[3] = 0; buffer[4] = 2;  // Tracking number = 2
    // Timestamp (6 bytes big-endian): 123456789012ns
    uint64_t ts = 123456789012ULL;
    buffer[5] = (ts >> 40) & 0xFF;
    buffer[6] = (ts >> 32) & 0xFF;
    buffer[7] = (ts >> 24) & 0xFF;
    buffer[8] = (ts >> 16) & 0xFF;
    buffer[9] = (ts >> 8) & 0xFF;
    buffer[10] = ts & 0xFF;
    // Order reference (8 bytes big-endian): 12345678
    uint64_t ref = 12345678;
    buffer[11] = (ref >> 56) & 0xFF;
    buffer[12] = (ref >> 48) & 0xFF;
    buffer[13] = (ref >> 40) & 0xFF;
    buffer[14] = (ref >> 32) & 0xFF;
    buffer[15] = (ref >> 24) & 0xFF;
    buffer[16] = (ref >> 16) & 0xFF;
    buffer[17] = (ref >> 8) & 0xFF;
    buffer[18] = ref & 0xFF;
    buffer[19] = 'B';  // Side = Buy
    // Shares (4 bytes big-endian): 1000
    buffer[20] = 0; buffer[21] = 0; buffer[22] = 0x03; buffer[23] = 0xE8;  // 1000
    // Stock (8 bytes): "AAPL    "
    std::memcpy(buffer + 24, "AAPL    ", 8);
    // Price (4 bytes big-endian): 15000 ($150.00 with 4 decimal places -> 1500000)
    uint32_t price = 1500000;
    buffer[32] = (price >> 24) & 0xFF;
    buffer[33] = (price >> 16) & 0xFF;
    buffer[34] = (price >> 8) & 0xFF;
    buffer[35] = price & 0xFF;
    
    itch::ITCHParser parser;
    auto msg_type = parser.parse(buffer, sizeof(buffer));
    
    ASSERT_EQ(msg_type.has_value(), true);
    ASSERT_EQ(*msg_type, itch::MessageType::ADD_ORDER);
    
    const auto& order = parser.add_order();
    ASSERT_EQ(order.stock_locate, 1);
    ASSERT_EQ(order.tracking_number, 2);
    ASSERT_EQ(order.timestamp_ns, 123456789012ULL);
    ASSERT_EQ(order.order_reference, 12345678ULL);
    ASSERT_EQ(order.side, 'B');
    ASSERT_EQ(order.shares, 1000U);
    ASSERT_EQ(std::string(order.stock), "AAPL");
    ASSERT_EQ(order.price, 1500000);
    ASSERT_EQ(order.has_mpid, false);
}

TEST(itch_order_executed) {
    // Build Order Executed message (type 'E')
    // type(1) + locate(2) + tracking(2) + timestamp(6) + ref(8) + shares(4) + match(8) = 31 bytes
    uint8_t buffer[31];
    std::memset(buffer, 0, sizeof(buffer));
    
    buffer[0] = 'E';
    // Stock locate = 5
    buffer[1] = 0; buffer[2] = 5;
    // Tracking = 10
    buffer[3] = 0; buffer[4] = 10;
    // Timestamp: 500000000000ns
    uint64_t ts = 500000000000ULL;
    buffer[5] = (ts >> 40) & 0xFF;
    buffer[6] = (ts >> 32) & 0xFF;
    buffer[7] = (ts >> 24) & 0xFF;
    buffer[8] = (ts >> 16) & 0xFF;
    buffer[9] = (ts >> 8) & 0xFF;
    buffer[10] = ts & 0xFF;
    // Order ref: 99999
    uint64_t ref = 99999;
    for (int i = 0; i < 8; ++i) {
        buffer[11 + i] = (ref >> (56 - 8*i)) & 0xFF;
    }
    // Executed shares: 500
    buffer[19] = 0; buffer[20] = 0; buffer[21] = 0x01; buffer[22] = 0xF4;
    // Match number: 7654321
    uint64_t match = 7654321;
    for (int i = 0; i < 8; ++i) {
        buffer[23 + i] = (match >> (56 - 8*i)) & 0xFF;
    }
    
    itch::ITCHParser parser;
    auto msg_type = parser.parse(buffer, sizeof(buffer));
    
    ASSERT_EQ(msg_type.has_value(), true);
    ASSERT_EQ(*msg_type, itch::MessageType::ORDER_EXECUTED);
    
    const auto& exec = parser.order_executed();
    ASSERT_EQ(exec.stock_locate, 5);
    ASSERT_EQ(exec.order_reference, 99999ULL);
    ASSERT_EQ(exec.executed_shares, 500U);
    ASSERT_EQ(exec.match_number, 7654321ULL);
}

TEST(itch_stream_processor) {
    // Build multiple messages with length prefix
    std::vector<uint8_t> stream;
    
    // First message: Add Order (36 bytes)
    uint8_t add_order[36];
    std::memset(add_order, 0, sizeof(add_order));
    add_order[0] = 'A';
    add_order[19] = 'B';
    std::memcpy(add_order + 24, "MSFT    ", 8);
    
    // Length prefix (2 bytes big-endian)
    stream.push_back(0);
    stream.push_back(36);
    stream.insert(stream.end(), add_order, add_order + 36);
    
    // Second message: Order Delete (19 bytes)
    uint8_t order_delete[19];
    std::memset(order_delete, 0, sizeof(order_delete));
    order_delete[0] = 'D';
    
    stream.push_back(0);
    stream.push_back(19);
    stream.insert(stream.end(), order_delete, order_delete + 19);
    
    // Process stream
    itch::ITCHStreamProcessor processor;
    
    int add_count = 0;
    int delete_count = 0;
    
    processor.set_add_order_callback([&](const itch::AddOrder& order) {
        ++add_count;
        ASSERT_EQ(std::string(order.stock), "MSFT");
    });
    
    processor.set_order_delete_callback([&](const itch::OrderDelete&) {
        ++delete_count;
    });
    
    size_t consumed = processor.process(stream.data(), stream.size());
    
    ASSERT_EQ(consumed, stream.size());
    ASSERT_EQ(add_count, 1);
    ASSERT_EQ(delete_count, 1);
    ASSERT_EQ(processor.messages_processed(), 2ULL);
}

// =============================================================================
// OUCH CODEC TESTS
// =============================================================================

TEST(ouch_encode_enter_order) {
    uint8_t buffer[128];
    ouch::OUCHEncoder encoder;
    
    size_t len = encoder.encode_enter_order(
        buffer,
        "ORDER12345678",  // 14-char order token
        'B',              // Buy
        1000,             // Shares
        "AAPL",           // Symbol (8 chars)
        1500000,          // Price ($150.00 * 10000)
        0,                // Time in force (day)
        "FIRM",           // Firm (4 chars)
        'A',              // Display (attributable)
        'O',              // Capacity (principal)
        'N'               // Intermarket sweep (no)
    );
    
    ASSERT_EQ(buffer[0], 'O');  // Message type
    
    // Verify order token (bytes 1-14)
    char token[15];
    std::memcpy(token, buffer + 1, 14);
    token[14] = '\0';
    ASSERT_EQ(std::string(token).substr(0, 13), "ORDER12345678");
    
    // Verify side
    ASSERT_EQ(buffer[15], 'B');
    
    // Verify message was encoded
    ASSERT_EQ(len > 0, true);
}

TEST(ouch_encode_cancel_order) {
    uint8_t buffer[128];
    ouch::OUCHEncoder encoder;
    
    size_t len = encoder.encode_cancel_order(buffer, "ORDER12345678", 0);
    
    ASSERT_EQ(buffer[0], 'X');  // Message type
    ASSERT_EQ(len, 19ULL);      // type(1) + token(14) + shares(4)
}

TEST(ouch_decode_executed) {
    // Build an executed message
    uint8_t buffer[40];
    std::memset(buffer, 0, sizeof(buffer));
    
    buffer[0] = 'E';  // Executed
    
    // Timestamp (8 bytes little-endian for OUCH): 1234567890
    uint64_t ts = 1234567890;
    std::memcpy(buffer + 1, &ts, 8);
    
    // Order token (14 bytes)
    std::memcpy(buffer + 9, "ORDER12345678 ", 14);
    
    // Executed shares (4 bytes little-endian): 500
    uint32_t shares = 500;
    std::memcpy(buffer + 23, &shares, 4);
    
    // Execution price (4 bytes little-endian): 1505000
    uint32_t price = 1505000;
    std::memcpy(buffer + 27, &price, 4);
    
    // Liquidity flag
    buffer[31] = 'R';  // Removed
    
    // Match number (8 bytes)
    uint64_t match = 9876543210ULL;
    std::memcpy(buffer + 32, &match, 8);
    
    ouch::OUCHDecoder decoder;
    auto msg_type = decoder.decode(buffer, sizeof(buffer));
    
    ASSERT_EQ(msg_type, ouch::OUCHDecoder::MessageType::EXECUTED);
    
    const auto& exec = decoder.executed();
    ASSERT_EQ(exec.timestamp, 1234567890ULL);
    ASSERT_EQ(exec.executed_shares, 500U);
    ASSERT_EQ(exec.execution_price, 1505000);
    ASSERT_EQ(exec.liquidity_flag, 'R');
    ASSERT_EQ(exec.match_number, 9876543210ULL);
}

// =============================================================================
// SBE CODEC TESTS
// =============================================================================

TEST(sbe_encode_decode_primitives) {
    uint8_t buffer[256];
    sbe::SBEEncoder encoder(buffer, sizeof(buffer));
    
    // Write header and primitives
    ASSERT_EQ(encoder.wrap_header(100, 1, 0), true);
    ASSERT_EQ(encoder.put_uint8(42), true);
    ASSERT_EQ(encoder.put_int16(-1000), true);
    ASSERT_EQ(encoder.put_uint32(123456789), true);
    ASSERT_EQ(encoder.put_int64(-9876543210LL), true);
    ASSERT_EQ(encoder.put_double(3.14159), true);
    ASSERT_EQ(encoder.put_string("AAPL", 8), true);
    encoder.complete_root_block();
    
    size_t encoded_len = encoder.encoded_length();
    ASSERT_EQ(encoded_len > sizeof(sbe::MessageHeader), true);
    
    // Decode
    sbe::SBEDecoder decoder(buffer, encoded_len);
    
    auto header = decoder.read_header();
    ASSERT_EQ(header.has_value(), true);
    ASSERT_EQ(header->template_id, 100);
    ASSERT_EQ(header->schema_id, 1);
    ASSERT_EQ(header->version, 0);
    
    auto u8 = decoder.get_uint8();
    ASSERT_EQ(u8.has_value(), true);
    ASSERT_EQ(*u8, 42);
    
    auto i16 = decoder.get_int16();
    ASSERT_EQ(i16.has_value(), true);
    ASSERT_EQ(*i16, -1000);
    
    auto u32 = decoder.get_uint32();
    ASSERT_EQ(u32.has_value(), true);
    ASSERT_EQ(*u32, 123456789U);
    
    auto i64 = decoder.get_int64();
    ASSERT_EQ(i64.has_value(), true);
    ASSERT_EQ(*i64, -9876543210LL);
    
    auto dbl = decoder.get_double();
    ASSERT_EQ(dbl.has_value(), true);
    ASSERT_NEAR(*dbl, 3.14159, 1e-10);
    
    auto str = decoder.get_string(8);
    ASSERT_EQ(str.has_value(), true);
    ASSERT_EQ(*str, "AAPL");
}

TEST(sbe_cme_market_data) {
    uint8_t buffer[256];
    
    // Encode CME-style market data incremental
    size_t len = sbe::CME::encode_md_incremental(
        buffer, sizeof(buffer),
        1234567890000ULL,  // Transact time (ns)
        12345,             // Security ID
        sbe::CME::MDEntryType::BID,
        sbe::CME::MDUpdateAction::NEW,
        1500000,           // Price mantissa ($150.00)
        1000,              // Quantity
        100                // Rpt seq
    );
    
    ASSERT_EQ(len > 0, true);
    
    // Decode
    uint64_t transact_time;
    uint32_t rpt_seq;
    std::vector<sbe::CME::MDIncrementalEntry> entries;
    
    bool ok = sbe::CME::decode_md_incremental(
        buffer, len, transact_time, rpt_seq, entries);
    
    ASSERT_EQ(ok, true);
    ASSERT_EQ(transact_time, 1234567890000ULL);
    ASSERT_EQ(rpt_seq, 100U);
    ASSERT_EQ(entries.size(), 1ULL);
    ASSERT_EQ(entries[0].security_id, 12345U);
    ASSERT_EQ(entries[0].entry_type, sbe::CME::MDEntryType::BID);
    ASSERT_EQ(entries[0].update_action, sbe::CME::MDUpdateAction::NEW);
    ASSERT_EQ(entries[0].price_mantissa, 1500000);
    ASSERT_EQ(entries[0].quantity, 1000);
}

// =============================================================================
// PERFORMANCE TESTS
// =============================================================================

TEST(itch_parse_performance) {
    // Build a sample message
    uint8_t buffer[36];
    std::memset(buffer, 0, sizeof(buffer));
    buffer[0] = 'A';
    buffer[19] = 'B';
    std::memcpy(buffer + 24, "AAPL    ", 8);
    
    itch::ITCHParser parser;
    
    constexpr int ITERATIONS = 1000000;
    
    auto start = std::chrono::steady_clock::now();
    
    for (int i = 0; i < ITERATIONS; ++i) {
        auto type = parser.parse(buffer, sizeof(buffer));
        (void)type;  // Prevent optimization
    }
    
    auto end = std::chrono::steady_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    
    double ns_per_msg = static_cast<double>(ns) / ITERATIONS;
    
    std::cout << "\n  ITCH parse: " << std::fixed << std::setprecision(1) 
              << ns_per_msg << " ns/msg (" 
              << (1e9 / ns_per_msg / 1e6) << "M msg/sec)";
    
    ASSERT_EQ(ns_per_msg < 100, true);  // Should be under 100ns
}

TEST(ouch_encode_performance) {
    uint8_t buffer[128];
    ouch::OUCHEncoder encoder;
    
    constexpr int ITERATIONS = 1000000;
    
    auto start = std::chrono::steady_clock::now();
    
    for (int i = 0; i < ITERATIONS; ++i) {
        size_t len = encoder.encode_enter_order(
            buffer, "ORDER12345678", 'B', 1000, "AAPL", 1500000);
        (void)len;
    }
    
    auto end = std::chrono::steady_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    
    double ns_per_msg = static_cast<double>(ns) / ITERATIONS;
    
    std::cout << "\n  OUCH encode: " << std::fixed << std::setprecision(1)
              << ns_per_msg << " ns/msg";
    
    ASSERT_EQ(ns_per_msg < 50, true);  // Should be under 50ns
}

TEST(sbe_encode_decode_performance) {
    uint8_t buffer[256];
    
    constexpr int ITERATIONS = 1000000;
    
    // Encode performance
    auto start = std::chrono::steady_clock::now();
    
    for (int i = 0; i < ITERATIONS; ++i) {
        sbe::SBEEncoder encoder(buffer, sizeof(buffer));
        encoder.wrap_header(100, 1, 0);
        encoder.put_uint64(12345678);
        encoder.put_string("AAPL", 8);
        encoder.put_int32(1500000);
        encoder.put_int32(1000);
        encoder.complete_root_block();
    }
    
    auto end = std::chrono::steady_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    
    double ns_per_encode = static_cast<double>(ns) / ITERATIONS;
    
    std::cout << "\n  SBE encode: " << std::fixed << std::setprecision(1)
              << ns_per_encode << " ns/msg";
    
    // Decode performance
    sbe::SBEEncoder encoder(buffer, sizeof(buffer));
    encoder.wrap_header(100, 1, 0);
    encoder.put_uint64(12345678);
    encoder.put_string("AAPL", 8);
    encoder.put_int32(1500000);
    encoder.put_int32(1000);
    encoder.complete_root_block();
    size_t len = encoder.encoded_length();
    
    start = std::chrono::steady_clock::now();
    
    for (int i = 0; i < ITERATIONS; ++i) {
        sbe::SBEDecoder decoder(buffer, len);
        auto h = decoder.read_header();
        auto v1 = decoder.get_uint64();
        auto v2 = decoder.get_string(8);
        auto v3 = decoder.get_int32();
        auto v4 = decoder.get_int32();
        (void)h; (void)v1; (void)v2; (void)v3; (void)v4;
    }
    
    end = std::chrono::steady_clock::now();
    ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    
    double ns_per_decode = static_cast<double>(ns) / ITERATIONS;
    
    std::cout << "\n  SBE decode: " << std::fixed << std::setprecision(1)
              << ns_per_decode << " ns/msg";
    
    ASSERT_EQ(ns_per_encode < 30, true);
    ASSERT_EQ(ns_per_decode < 30, true);
}

// =============================================================================
// MAIN
// =============================================================================

int main() {
    std::cout << "=== CODEC TESTS ===\n\n";
    
    std::cout << "ITCH Tests:\n";
    RUN_TEST(itch_add_order);
    RUN_TEST(itch_order_executed);
    RUN_TEST(itch_stream_processor);
    
    std::cout << "\nOUCH Tests:\n";
    RUN_TEST(ouch_encode_enter_order);
    RUN_TEST(ouch_encode_cancel_order);
    RUN_TEST(ouch_decode_executed);
    
    std::cout << "\nSBE Tests:\n";
    RUN_TEST(sbe_encode_decode_primitives);
    RUN_TEST(sbe_cme_market_data);
    
    std::cout << "\nPerformance Tests:";
    RUN_TEST(itch_parse_performance);
    RUN_TEST(ouch_encode_performance);
    RUN_TEST(sbe_encode_decode_performance);
    
    std::cout << "\n\n=== ALL TESTS PASSED ===\n";
    return 0;
}
