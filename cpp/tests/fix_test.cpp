/**
 * FIX Protocol Parser Tests
 * 
 * Test categories:
 * 1. Basic parsing correctness
 * 2. Edge cases and error handling
 * 3. Performance regression tests
 * 4. Checksum validation
 */

#include "../include/fix_parser.hpp"
#include <iostream>
#include <cassert>
#include <chrono>
#include <vector>
#include <cstring>

using namespace arbor::fix;

// =============================================================================
// TEST UTILITIES
// =============================================================================

#define ASSERT_TRUE(cond) \
    do { if (!(cond)) { \
        std::cerr << "FAILED: " << #cond << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        return false; \
    }} while(0)

#define ASSERT_EQ(a, b) \
    do { if ((a) != (b)) { \
        std::cerr << "FAILED: " << #a << " != " << #b << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        std::cerr << "  Got: " << (a) << " Expected: " << (b) << "\n"; \
        return false; \
    }} while(0)

#define ASSERT_NEAR(a, b, eps) \
    do { if (std::abs((a) - (b)) > (eps)) { \
        std::cerr << "FAILED: " << #a << " != " << #b << " (tolerance " << (eps) << ") at " << __FILE__ << ":" << __LINE__ << "\n"; \
        return false; \
    }} while(0)

int tests_passed = 0;
int tests_failed = 0;

template<typename F>
void run_test(const char* name, F&& func) {
    std::cout << "  Testing " << name << "... ";
    if (func()) {
        std::cout << "PASSED\n";
        ++tests_passed;
    } else {
        std::cout << "FAILED\n";
        ++tests_failed;
    }
}

// =============================================================================
// TEST: BASIC PARSING
// =============================================================================

bool test_parse_new_order_single() {
    // Realistic NewOrderSingle message
    const char* msg = 
        "8=FIX.4.4\x01"
        "9=148\x01"
        "35=D\x01"
        "49=SENDER\x01"
        "56=TARGET\x01"
        "34=1\x01"
        "52=20240101-12:00:00.000\x01"
        "11=ORDER123\x01"
        "55=AAPL\x01"
        "54=1\x01"
        "38=100\x01"
        "40=2\x01"
        "44=150.50\x01"
        "59=0\x01"
        "10=123\x01";
    
    FIXParser parser;
    auto parsed = parser.parse(msg, std::strlen(msg));
    
    ASSERT_TRUE(parsed.valid());
    ASSERT_EQ(parsed.msg_type(), MsgType::NewOrderSingle);
    
    // Check string fields
    ASSERT_EQ(parsed.get(Tag::ClOrdID).view(), "ORDER123");
    ASSERT_EQ(parsed.get(Tag::Symbol).view(), "AAPL");
    ASSERT_EQ(parsed.get(Tag::SenderCompID).view(), "SENDER");
    ASSERT_EQ(parsed.get(Tag::TargetCompID).view(), "TARGET");
    
    // Check numeric fields
    auto qty = parsed.get_int<Tag::OrderQty>();
    ASSERT_TRUE(qty.has_value());
    ASSERT_EQ(*qty, 100);
    
    // Check character fields
    ASSERT_EQ(parsed.get_char<Tag::Side>(), '1');  // Buy
    ASSERT_EQ(parsed.get_char<Tag::OrdType>(), '2');  // Limit
    
    // Check price
    auto price = parsed.get_double<Tag::Price>();
    ASSERT_TRUE(price.has_value());
    ASSERT_NEAR(*price, 150.50, 0.01);
    
    return true;
}

bool test_parse_execution_report() {
    const char* msg =
        "8=FIX.4.4\x01"
        "9=200\x01"
        "35=8\x01"
        "49=EXCHANGE\x01"
        "56=TRADER\x01"
        "34=42\x01"
        "52=20240101-12:00:01.123\x01"
        "37=EXEC456\x01"
        "11=ORDER123\x01"
        "17=EXECID789\x01"
        "150=F\x01"
        "39=2\x01"
        "55=AAPL\x01"
        "54=1\x01"
        "38=100\x01"
        "32=100\x01"
        "31=150.45\x01"
        "14=100\x01"
        "6=150.45\x01"
        "151=0\x01"
        "10=255\x01";
    
    FIXParser parser;
    auto parsed = parser.parse(msg, std::strlen(msg));
    
    ASSERT_TRUE(parsed.valid());
    ASSERT_EQ(parsed.msg_type(), MsgType::ExecutionReport);
    
    // Execution details
    ASSERT_EQ(parsed.get(Tag::OrderID).view(), "EXEC456");
    ASSERT_EQ(parsed.get(Tag::ExecID).view(), "EXECID789");
    ASSERT_EQ(parsed.get_char<Tag::ExecType>(), 'F');  // Fill
    ASSERT_EQ(parsed.get_char<Tag::OrdStatus>(), '2');  // Filled
    
    // Fill details
    auto last_qty = parsed.get_int<Tag::LastQty>();
    ASSERT_TRUE(last_qty.has_value());
    ASSERT_EQ(*last_qty, 100);
    
    auto last_px = parsed.get_double<Tag::LastPx>();
    ASSERT_TRUE(last_px.has_value());
    ASSERT_NEAR(*last_px, 150.45, 0.01);
    
    // Cumulative
    auto cum_qty = parsed.get_int<Tag::CumQty>();
    ASSERT_TRUE(cum_qty.has_value());
    ASSERT_EQ(*cum_qty, 100);
    
    auto leaves_qty = parsed.get_int<Tag::LeavesQty>();
    ASSERT_TRUE(leaves_qty.has_value());
    ASSERT_EQ(*leaves_qty, 0);
    
    return true;
}

// =============================================================================
// TEST: EDGE CASES
// =============================================================================

bool test_parse_empty_buffer() {
    FIXParser parser;
    auto parsed = parser.parse(nullptr, 0);
    ASSERT_TRUE(!parsed.valid());
    
    parsed = parser.parse("", 0);
    ASSERT_TRUE(!parsed.valid());
    
    return true;
}

bool test_parse_truncated_message() {
    // Message without checksum
    const char* msg = 
        "8=FIX.4.4\x01"
        "9=50\x01"
        "35=D\x01"
        "49=SENDER\x01"
        "56=TARGET\x01";
    
    FIXParser parser;
    auto parsed = parser.parse(msg, std::strlen(msg));
    
    // Should parse but not be marked valid (no checksum)
    ASSERT_TRUE(!parsed.valid());
    
    // But fields should still be accessible
    ASSERT_EQ(parsed.msg_type(), MsgType::NewOrderSingle);
    ASSERT_EQ(parsed.get(Tag::SenderCompID).view(), "SENDER");
    
    return true;
}

bool test_parse_large_tag_numbers() {
    // Custom tags (>1000) should work via fallback
    const char* msg =
        "8=FIX.4.4\x01"
        "9=100\x01"
        "35=D\x01"
        "49=SENDER\x01"
        "5001=CUSTOMVALUE\x01"
        "10=000\x01";
    
    FIXParser parser;
    auto parsed = parser.parse(msg, std::strlen(msg));
    
    ASSERT_TRUE(parsed.valid());
    ASSERT_EQ(parsed.get(5001).view(), "CUSTOMVALUE");
    
    return true;
}

bool test_missing_field() {
    const char* msg =
        "8=FIX.4.4\x01"
        "9=50\x01"
        "35=D\x01"
        "10=000\x01";
    
    FIXParser parser;
    auto parsed = parser.parse(msg, std::strlen(msg));
    
    // Missing field should return empty view
    auto missing = parsed.get(Tag::ClOrdID);
    ASSERT_TRUE(!missing);
    ASSERT_TRUE(missing.empty());
    
    // get_int on missing field should return nullopt
    auto missing_int = parsed.get_int<Tag::OrderQty>();
    ASSERT_TRUE(!missing_int.has_value());
    
    return true;
}

// =============================================================================
// TEST: CHECKSUM VALIDATION
// =============================================================================

bool test_checksum_valid() {
    // Valid checksum (calculated manually)
    const char* msg =
        "8=FIX.4.4\x01"
        "9=5\x01"
        "35=0\x01"
        "10=045\x01";  // Correct checksum
    
    ASSERT_TRUE(FIXParser::validate_checksum(msg, std::strlen(msg)));
    
    return true;
}

bool test_checksum_invalid() {
    const char* msg =
        "8=FIX.4.4\x01"
        "9=5\x01"
        "35=0\x01"
        "10=999\x01";  // Wrong checksum
    
    ASSERT_TRUE(!FIXParser::validate_checksum(msg, std::strlen(msg)));
    
    return true;
}

// =============================================================================
// TEST: MESSAGE BUILDER
// =============================================================================

bool test_builder_new_order() {
    FIXBuilder builder("SENDER", "TARGET");
    
    builder.start(MsgType::NewOrderSingle)
           .add(Tag::ClOrdID, "ORDER001")
           .add(Tag::Symbol, "AAPL")
           .add(Tag::Side, '1')
           .add(Tag::OrderQty, int64_t(100))
           .add(Tag::OrdType, '2')
           .add(Tag::Price, 150.50);
    
    auto msg = builder.finish();
    
    // Parse the generated message
    FIXParser parser;
    auto parsed = parser.parse(msg.data(), msg.size());
    
    ASSERT_TRUE(parsed.valid());
    ASSERT_EQ(parsed.msg_type(), MsgType::NewOrderSingle);
    ASSERT_EQ(parsed.get(Tag::ClOrdID).view(), "ORDER001");
    ASSERT_EQ(parsed.get(Tag::Symbol).view(), "AAPL");
    
    auto qty = parsed.get_int<Tag::OrderQty>();
    ASSERT_TRUE(qty.has_value());
    ASSERT_EQ(*qty, 100);
    
    return true;
}

// =============================================================================
// TEST: PERFORMANCE REGRESSION
// =============================================================================

bool test_parse_throughput() {
    // Realistic order message
    const char* msg = 
        "8=FIX.4.4\x01"
        "9=148\x01"
        "35=D\x01"
        "49=SENDER\x01"
        "56=TARGET\x01"
        "34=1\x01"
        "52=20240101-12:00:00.000\x01"
        "11=ORDER123\x01"
        "55=AAPL\x01"
        "54=1\x01"
        "38=100\x01"
        "40=2\x01"
        "44=150.50\x01"
        "59=0\x01"
        "10=123\x01";
    
    size_t len = std::strlen(msg);
    FIXParser parser;
    
    // Warmup
    for (int i = 0; i < 1000; ++i) {
        auto parsed = parser.parse(msg, len);
        (void)parsed;
    }
    
    // Benchmark
    constexpr int ITERATIONS = 100000;
    auto start = std::chrono::steady_clock::now();
    
    for (int i = 0; i < ITERATIONS; ++i) {
        auto parsed = parser.parse(msg, len);
        (void)parsed.get(Tag::ClOrdID);  // Access a field to prevent optimization
    }
    
    auto end = std::chrono::steady_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    
    double ns_per_parse = static_cast<double>(ns) / ITERATIONS;
    double msgs_per_sec = 1e9 / ns_per_parse;
    
    std::cout << "\n    [Throughput: " << static_cast<int>(msgs_per_sec / 1e6) << "M msg/sec, "
              << static_cast<int>(ns_per_parse) << " ns/msg]\n    ";
    
    // Performance threshold: at least 2M messages/sec
    ASSERT_TRUE(msgs_per_sec > 2'000'000);
    
    return true;
}

// =============================================================================
// MAIN
// =============================================================================

int main() {
    std::cout << "============================================================\n";
    std::cout << "             FIX PROTOCOL PARSER - TEST SUITE               \n";
    std::cout << "============================================================\n\n";
    
    std::cout << "Basic Parsing Tests:\n";
    run_test("NewOrderSingle parsing", test_parse_new_order_single);
    run_test("ExecutionReport parsing", test_parse_execution_report);
    
    std::cout << "\nEdge Case Tests:\n";
    run_test("Empty buffer", test_parse_empty_buffer);
    run_test("Truncated message", test_parse_truncated_message);
    run_test("Large tag numbers", test_parse_large_tag_numbers);
    run_test("Missing field", test_missing_field);
    
    std::cout << "\nChecksum Tests:\n";
    run_test("Valid checksum", test_checksum_valid);
    run_test("Invalid checksum", test_checksum_invalid);
    
    std::cout << "\nBuilder Tests:\n";
    run_test("Build NewOrderSingle", test_builder_new_order);
    
    std::cout << "\nPerformance Tests:\n";
    run_test("Parse throughput", test_parse_throughput);
    
    std::cout << "\n============================================================\n";
    std::cout << "RESULTS: " << tests_passed << " passed, " << tests_failed << " failed\n";
    std::cout << "============================================================\n";
    
    return tests_failed > 0 ? 1 : 0;
}
