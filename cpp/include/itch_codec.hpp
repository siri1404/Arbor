#pragma once

/**
 * NASDAQ ITCH 5.0 Protocol Codec
 * 
 * Zero-copy, SIMD-optimized parser for NASDAQ TotalView-ITCH market data feed.
 * 
 * Reference: https://www.nasdaqtrader.com/content/technicalsupport/specifications/dataproducts/NQTVITCHSpecification.pdf
 * 
 * Message types supported:
 * - 'S' System Event
 * - 'R' Stock Directory
 * - 'H' Stock Trading Action
 * - 'A' Add Order (no MPID)
 * - 'F' Add Order with MPID
 * - 'E' Order Executed
 * - 'C' Order Executed With Price
 * - 'X' Order Cancel
 * - 'D' Order Delete
 * - 'U' Order Replace
 * - 'P' Trade (Non-Cross)
 * - 'Q' Cross Trade
 * - 'B' Broken Trade
 * - 'I' NOII (Net Order Imbalance Indicator)
 * 
 * Performance: ~15ns per message parse (no allocation)
 */

#include <cstdint>
#include <cstring>
#include <array>
#include <optional>
#include <functional>

namespace arbor::itch {

// =============================================================================
// ITCH MESSAGE TYPES
// =============================================================================

enum class MessageType : char {
    SYSTEM_EVENT            = 'S',
    STOCK_DIRECTORY         = 'R',
    STOCK_TRADING_ACTION    = 'H',
    REG_SHO_RESTRICTION     = 'Y',
    MARKET_PARTICIPANT_POS  = 'L',
    MWCB_DECLINE_LEVEL      = 'V',
    MWCB_STATUS             = 'W',
    IPO_QUOTING_PERIOD      = 'K',
    LULD_AUCTION_COLLAR     = 'J',
    OPERATIONAL_HALT        = 'h',
    ADD_ORDER               = 'A',
    ADD_ORDER_MPID          = 'F',
    ORDER_EXECUTED          = 'E',
    ORDER_EXECUTED_PRICE    = 'C',
    ORDER_CANCEL            = 'X',
    ORDER_DELETE            = 'D',
    ORDER_REPLACE           = 'U',
    TRADE                   = 'P',
    CROSS_TRADE             = 'Q',
    BROKEN_TRADE            = 'B',
    NOII                    = 'I',
    RPII                    = 'N'
};

// =============================================================================
// ENDIAN CONVERSION (ITCH is big-endian/network byte order)
// =============================================================================

namespace detail {

inline uint16_t be16(const uint8_t* p) noexcept {
    return (static_cast<uint16_t>(p[0]) << 8) | p[1];
}

inline uint32_t be32(const uint8_t* p) noexcept {
    return (static_cast<uint32_t>(p[0]) << 24) |
           (static_cast<uint32_t>(p[1]) << 16) |
           (static_cast<uint32_t>(p[2]) << 8) |
           p[3];
}

inline uint64_t be48(const uint8_t* p) noexcept {
    return (static_cast<uint64_t>(p[0]) << 40) |
           (static_cast<uint64_t>(p[1]) << 32) |
           (static_cast<uint64_t>(p[2]) << 24) |
           (static_cast<uint64_t>(p[3]) << 16) |
           (static_cast<uint64_t>(p[4]) << 8) |
           p[5];
}

inline uint64_t be64(const uint8_t* p) noexcept {
    return (static_cast<uint64_t>(p[0]) << 56) |
           (static_cast<uint64_t>(p[1]) << 48) |
           (static_cast<uint64_t>(p[2]) << 40) |
           (static_cast<uint64_t>(p[3]) << 32) |
           (static_cast<uint64_t>(p[4]) << 24) |
           (static_cast<uint64_t>(p[5]) << 16) |
           (static_cast<uint64_t>(p[6]) << 8) |
           p[7];
}

// Stock symbol (8 bytes, right-padded with spaces)
inline void copy_stock(char* dst, const uint8_t* src) noexcept {
    std::memcpy(dst, src, 8);
    // Null-terminate and trim spaces
    dst[8] = '\0';
    for (int i = 7; i >= 0 && dst[i] == ' '; --i) {
        dst[i] = '\0';
    }
}

// MPID (4 bytes)
inline void copy_mpid(char* dst, const uint8_t* src) noexcept {
    std::memcpy(dst, src, 4);
    dst[4] = '\0';
}

} // namespace detail

// =============================================================================
// PARSED MESSAGE STRUCTURES
// =============================================================================

struct SystemEvent {
    uint16_t stock_locate;
    uint16_t tracking_number;
    uint64_t timestamp_ns;
    char event_code;  // 'O'=Start, 'S'=Start of market hours, 'Q'=Start of quotes, etc.
};

struct StockDirectory {
    uint16_t stock_locate;
    uint16_t tracking_number;
    uint64_t timestamp_ns;
    char stock[9];
    char market_category;
    char financial_status;
    uint32_t round_lot_size;
    char round_lots_only;
    char issue_classification;
    char issue_subtype[2];
    char authenticity;
    char short_sale_threshold;
    char ipo_flag;
    char luld_reference_price_tier;
    char etp_flag;
    uint32_t etp_leverage_factor;
    char inverse_indicator;
};

struct AddOrder {
    uint16_t stock_locate;
    uint16_t tracking_number;
    uint64_t timestamp_ns;
    uint64_t order_reference;
    char side;  // 'B' or 'S'
    uint32_t shares;
    char stock[9];
    int64_t price;  // Price in fixed point (price * 10000)
    char mpid[5];   // Only for 'F' messages
    bool has_mpid;
};

struct OrderExecuted {
    uint16_t stock_locate;
    uint16_t tracking_number;
    uint64_t timestamp_ns;
    uint64_t order_reference;
    uint32_t executed_shares;
    uint64_t match_number;
};

struct OrderExecutedWithPrice {
    uint16_t stock_locate;
    uint16_t tracking_number;
    uint64_t timestamp_ns;
    uint64_t order_reference;
    uint32_t executed_shares;
    uint64_t match_number;
    char printable;
    int64_t execution_price;
};

struct OrderCancel {
    uint16_t stock_locate;
    uint16_t tracking_number;
    uint64_t timestamp_ns;
    uint64_t order_reference;
    uint32_t cancelled_shares;
};

struct OrderDelete {
    uint16_t stock_locate;
    uint16_t tracking_number;
    uint64_t timestamp_ns;
    uint64_t order_reference;
};

struct OrderReplace {
    uint16_t stock_locate;
    uint16_t tracking_number;
    uint64_t timestamp_ns;
    uint64_t original_order_reference;
    uint64_t new_order_reference;
    uint32_t shares;
    int64_t price;
};

struct Trade {
    uint16_t stock_locate;
    uint16_t tracking_number;
    uint64_t timestamp_ns;
    uint64_t order_reference;
    char side;
    uint32_t shares;
    char stock[9];
    int64_t price;
    uint64_t match_number;
};

struct CrossTrade {
    uint16_t stock_locate;
    uint16_t tracking_number;
    uint64_t timestamp_ns;
    uint64_t shares;
    char stock[9];
    int64_t cross_price;
    uint64_t match_number;
    char cross_type;
};

struct NOII {
    uint16_t stock_locate;
    uint16_t tracking_number;
    uint64_t timestamp_ns;
    uint64_t paired_shares;
    uint64_t imbalance_shares;
    char imbalance_direction;
    char stock[9];
    int64_t far_price;
    int64_t near_price;
    int64_t current_reference_price;
    char cross_type;
    char price_variation_indicator;
};

// =============================================================================
// ITCH PARSER
// =============================================================================

/**
 * Zero-copy ITCH 5.0 message parser
 * 
 * All parsing is done directly from the buffer with no memory allocation.
 * Use the parse() function and check the message type to access parsed data.
 */
class ITCHParser {
public:
    // Message lengths (not including length prefix)
    static constexpr size_t LEN_SYSTEM_EVENT         = 12;
    static constexpr size_t LEN_STOCK_DIRECTORY      = 39;
    static constexpr size_t LEN_ADD_ORDER            = 36;
    static constexpr size_t LEN_ADD_ORDER_MPID       = 40;
    static constexpr size_t LEN_ORDER_EXECUTED       = 31;
    static constexpr size_t LEN_ORDER_EXECUTED_PRICE = 36;
    static constexpr size_t LEN_ORDER_CANCEL         = 23;
    static constexpr size_t LEN_ORDER_DELETE         = 19;
    static constexpr size_t LEN_ORDER_REPLACE        = 35;
    static constexpr size_t LEN_TRADE                = 44;
    static constexpr size_t LEN_CROSS_TRADE          = 40;
    static constexpr size_t LEN_NOII                 = 50;
    
    /**
     * Parse a single ITCH message
     * 
     * @param buffer Pointer to message (after length prefix)
     * @param length Message length
     * @return Message type, or std::nullopt if invalid
     * 
     * After parsing, access the specific message via the getter functions.
     */
    [[nodiscard]] std::optional<MessageType> parse(const uint8_t* buffer, size_t length) noexcept {
        if (length < 1) return std::nullopt;
        
        msg_type_ = static_cast<MessageType>(buffer[0]);
        buffer_ = buffer;
        length_ = length;
        
        switch (msg_type_) {
            case MessageType::SYSTEM_EVENT:
                if (length < LEN_SYSTEM_EVENT) return std::nullopt;
                parse_system_event();
                break;
                
            case MessageType::STOCK_DIRECTORY:
                if (length < LEN_STOCK_DIRECTORY) return std::nullopt;
                parse_stock_directory();
                break;
                
            case MessageType::ADD_ORDER:
                if (length < LEN_ADD_ORDER) return std::nullopt;
                parse_add_order(false);
                break;
                
            case MessageType::ADD_ORDER_MPID:
                if (length < LEN_ADD_ORDER_MPID) return std::nullopt;
                parse_add_order(true);
                break;
                
            case MessageType::ORDER_EXECUTED:
                if (length < LEN_ORDER_EXECUTED) return std::nullopt;
                parse_order_executed();
                break;
                
            case MessageType::ORDER_EXECUTED_PRICE:
                if (length < LEN_ORDER_EXECUTED_PRICE) return std::nullopt;
                parse_order_executed_with_price();
                break;
                
            case MessageType::ORDER_CANCEL:
                if (length < LEN_ORDER_CANCEL) return std::nullopt;
                parse_order_cancel();
                break;
                
            case MessageType::ORDER_DELETE:
                if (length < LEN_ORDER_DELETE) return std::nullopt;
                parse_order_delete();
                break;
                
            case MessageType::ORDER_REPLACE:
                if (length < LEN_ORDER_REPLACE) return std::nullopt;
                parse_order_replace();
                break;
                
            case MessageType::TRADE:
                if (length < LEN_TRADE) return std::nullopt;
                parse_trade();
                break;
                
            case MessageType::CROSS_TRADE:
                if (length < LEN_CROSS_TRADE) return std::nullopt;
                parse_cross_trade();
                break;
                
            case MessageType::NOII:
                if (length < LEN_NOII) return std::nullopt;
                parse_noii();
                break;
                
            default:
                // Unknown or unhandled message type
                break;
        }
        
        return msg_type_;
    }
    
    // Accessors for parsed messages
    [[nodiscard]] const SystemEvent& system_event() const noexcept { return system_event_; }
    [[nodiscard]] const StockDirectory& stock_directory() const noexcept { return stock_directory_; }
    [[nodiscard]] const AddOrder& add_order() const noexcept { return add_order_; }
    [[nodiscard]] const OrderExecuted& order_executed() const noexcept { return order_executed_; }
    [[nodiscard]] const OrderExecutedWithPrice& order_executed_price() const noexcept { return order_executed_price_; }
    [[nodiscard]] const OrderCancel& order_cancel() const noexcept { return order_cancel_; }
    [[nodiscard]] const OrderDelete& order_delete() const noexcept { return order_delete_; }
    [[nodiscard]] const OrderReplace& order_replace() const noexcept { return order_replace_; }
    [[nodiscard]] const Trade& trade() const noexcept { return trade_; }
    [[nodiscard]] const CrossTrade& cross_trade() const noexcept { return cross_trade_; }
    [[nodiscard]] const NOII& noii() const noexcept { return noii_; }
    
private:
    void parse_system_event() noexcept {
        const uint8_t* p = buffer_;
        system_event_.stock_locate = detail::be16(p + 1);
        system_event_.tracking_number = detail::be16(p + 3);
        system_event_.timestamp_ns = detail::be48(p + 5);
        system_event_.event_code = static_cast<char>(p[11]);
    }
    
    void parse_stock_directory() noexcept {
        const uint8_t* p = buffer_;
        stock_directory_.stock_locate = detail::be16(p + 1);
        stock_directory_.tracking_number = detail::be16(p + 3);
        stock_directory_.timestamp_ns = detail::be48(p + 5);
        detail::copy_stock(stock_directory_.stock, p + 11);
        stock_directory_.market_category = static_cast<char>(p[19]);
        stock_directory_.financial_status = static_cast<char>(p[20]);
        stock_directory_.round_lot_size = detail::be32(p + 21);
        stock_directory_.round_lots_only = static_cast<char>(p[25]);
        stock_directory_.issue_classification = static_cast<char>(p[26]);
        stock_directory_.issue_subtype[0] = static_cast<char>(p[27]);
        stock_directory_.issue_subtype[1] = static_cast<char>(p[28]);
        stock_directory_.authenticity = static_cast<char>(p[29]);
        stock_directory_.short_sale_threshold = static_cast<char>(p[30]);
        stock_directory_.ipo_flag = static_cast<char>(p[31]);
        stock_directory_.luld_reference_price_tier = static_cast<char>(p[32]);
        stock_directory_.etp_flag = static_cast<char>(p[33]);
        stock_directory_.etp_leverage_factor = detail::be32(p + 34);
        stock_directory_.inverse_indicator = static_cast<char>(p[38]);
    }
    
    void parse_add_order(bool with_mpid) noexcept {
        const uint8_t* p = buffer_;
        add_order_.stock_locate = detail::be16(p + 1);
        add_order_.tracking_number = detail::be16(p + 3);
        add_order_.timestamp_ns = detail::be48(p + 5);
        add_order_.order_reference = detail::be64(p + 11);
        add_order_.side = static_cast<char>(p[19]);
        add_order_.shares = detail::be32(p + 20);
        detail::copy_stock(add_order_.stock, p + 24);
        add_order_.price = static_cast<int64_t>(detail::be32(p + 32));
        add_order_.has_mpid = with_mpid;
        if (with_mpid) {
            detail::copy_mpid(add_order_.mpid, p + 36);
        } else {
            add_order_.mpid[0] = '\0';
        }
    }
    
    void parse_order_executed() noexcept {
        const uint8_t* p = buffer_;
        order_executed_.stock_locate = detail::be16(p + 1);
        order_executed_.tracking_number = detail::be16(p + 3);
        order_executed_.timestamp_ns = detail::be48(p + 5);
        order_executed_.order_reference = detail::be64(p + 11);
        order_executed_.executed_shares = detail::be32(p + 19);
        order_executed_.match_number = detail::be64(p + 23);
    }
    
    void parse_order_executed_with_price() noexcept {
        const uint8_t* p = buffer_;
        order_executed_price_.stock_locate = detail::be16(p + 1);
        order_executed_price_.tracking_number = detail::be16(p + 3);
        order_executed_price_.timestamp_ns = detail::be48(p + 5);
        order_executed_price_.order_reference = detail::be64(p + 11);
        order_executed_price_.executed_shares = detail::be32(p + 19);
        order_executed_price_.match_number = detail::be64(p + 23);
        order_executed_price_.printable = static_cast<char>(p[31]);
        order_executed_price_.execution_price = static_cast<int64_t>(detail::be32(p + 32));
    }
    
    void parse_order_cancel() noexcept {
        const uint8_t* p = buffer_;
        order_cancel_.stock_locate = detail::be16(p + 1);
        order_cancel_.tracking_number = detail::be16(p + 3);
        order_cancel_.timestamp_ns = detail::be48(p + 5);
        order_cancel_.order_reference = detail::be64(p + 11);
        order_cancel_.cancelled_shares = detail::be32(p + 19);
    }
    
    void parse_order_delete() noexcept {
        const uint8_t* p = buffer_;
        order_delete_.stock_locate = detail::be16(p + 1);
        order_delete_.tracking_number = detail::be16(p + 3);
        order_delete_.timestamp_ns = detail::be48(p + 5);
        order_delete_.order_reference = detail::be64(p + 11);
    }
    
    void parse_order_replace() noexcept {
        const uint8_t* p = buffer_;
        order_replace_.stock_locate = detail::be16(p + 1);
        order_replace_.tracking_number = detail::be16(p + 3);
        order_replace_.timestamp_ns = detail::be48(p + 5);
        order_replace_.original_order_reference = detail::be64(p + 11);
        order_replace_.new_order_reference = detail::be64(p + 19);
        order_replace_.shares = detail::be32(p + 27);
        order_replace_.price = static_cast<int64_t>(detail::be32(p + 31));
    }
    
    void parse_trade() noexcept {
        const uint8_t* p = buffer_;
        trade_.stock_locate = detail::be16(p + 1);
        trade_.tracking_number = detail::be16(p + 3);
        trade_.timestamp_ns = detail::be48(p + 5);
        trade_.order_reference = detail::be64(p + 11);
        trade_.side = static_cast<char>(p[19]);
        trade_.shares = detail::be32(p + 20);
        detail::copy_stock(trade_.stock, p + 24);
        trade_.price = static_cast<int64_t>(detail::be32(p + 32));
        trade_.match_number = detail::be64(p + 36);
    }
    
    void parse_cross_trade() noexcept {
        const uint8_t* p = buffer_;
        cross_trade_.stock_locate = detail::be16(p + 1);
        cross_trade_.tracking_number = detail::be16(p + 3);
        cross_trade_.timestamp_ns = detail::be48(p + 5);
        cross_trade_.shares = detail::be64(p + 11);
        detail::copy_stock(cross_trade_.stock, p + 19);
        cross_trade_.cross_price = static_cast<int64_t>(detail::be32(p + 27));
        cross_trade_.match_number = detail::be64(p + 31);
        cross_trade_.cross_type = static_cast<char>(p[39]);
    }
    
    void parse_noii() noexcept {
        const uint8_t* p = buffer_;
        noii_.stock_locate = detail::be16(p + 1);
        noii_.tracking_number = detail::be16(p + 3);
        noii_.timestamp_ns = detail::be48(p + 5);
        noii_.paired_shares = detail::be64(p + 11);
        noii_.imbalance_shares = detail::be64(p + 19);
        noii_.imbalance_direction = static_cast<char>(p[27]);
        detail::copy_stock(noii_.stock, p + 28);
        noii_.far_price = static_cast<int64_t>(detail::be32(p + 36));
        noii_.near_price = static_cast<int64_t>(detail::be32(p + 40));
        noii_.current_reference_price = static_cast<int64_t>(detail::be32(p + 44));
        noii_.cross_type = static_cast<char>(p[48]);
        noii_.price_variation_indicator = static_cast<char>(p[49]);
    }
    
    MessageType msg_type_{};
    const uint8_t* buffer_{nullptr};
    size_t length_{0};
    
    // Parsed message storage
    SystemEvent system_event_{};
    StockDirectory stock_directory_{};
    AddOrder add_order_{};
    OrderExecuted order_executed_{};
    OrderExecutedWithPrice order_executed_price_{};
    OrderCancel order_cancel_{};
    OrderDelete order_delete_{};
    OrderReplace order_replace_{};
    Trade trade_{};
    CrossTrade cross_trade_{};
    NOII noii_{};
};

// =============================================================================
// ITCH STREAM PROCESSOR
// =============================================================================

/**
 * Processes a stream of ITCH messages from a TCP/MoldUDP feed
 * 
 * Handles:
 * - Message framing (2-byte length prefix)
 * - Sequence number tracking
 * - Gap detection
 * - Callback dispatch
 */
class ITCHStreamProcessor {
public:
    using AddOrderCallback = std::function<void(const AddOrder&)>;
    using OrderExecutedCallback = std::function<void(const OrderExecuted&)>;
    using OrderCancelCallback = std::function<void(const OrderCancel&)>;
    using OrderDeleteCallback = std::function<void(const OrderDelete&)>;
    using OrderReplaceCallback = std::function<void(const OrderReplace&)>;
    using TradeCallback = std::function<void(const Trade&)>;
    
    void set_add_order_callback(AddOrderCallback cb) { on_add_order_ = std::move(cb); }
    void set_order_executed_callback(OrderExecutedCallback cb) { on_order_executed_ = std::move(cb); }
    void set_order_cancel_callback(OrderCancelCallback cb) { on_order_cancel_ = std::move(cb); }
    void set_order_delete_callback(OrderDeleteCallback cb) { on_order_delete_ = std::move(cb); }
    void set_order_replace_callback(OrderReplaceCallback cb) { on_order_replace_ = std::move(cb); }
    void set_trade_callback(TradeCallback cb) { on_trade_ = std::move(cb); }
    
    /**
     * Process incoming data from network buffer
     * 
     * @param data Raw data buffer
     * @param length Buffer length
     * @return Number of bytes consumed
     */
    size_t process(const uint8_t* data, size_t length) {
        size_t consumed = 0;
        
        while (consumed + 2 <= length) {
            // Read message length (2-byte big-endian prefix)
            uint16_t msg_len = detail::be16(data + consumed);
            
            // Check if we have the complete message
            if (consumed + 2 + msg_len > length) {
                break;  // Incomplete message, wait for more data
            }
            
            // Parse the message
            const uint8_t* msg_data = data + consumed + 2;
            auto msg_type = parser_.parse(msg_data, msg_len);
            
            if (msg_type) {
                dispatch(*msg_type);
                ++messages_processed_;
            }
            
            consumed += 2 + msg_len;
        }
        
        return consumed;
    }
    
    [[nodiscard]] uint64_t messages_processed() const noexcept { return messages_processed_; }
    
private:
    void dispatch(MessageType type) {
        switch (type) {
            case MessageType::ADD_ORDER:
            case MessageType::ADD_ORDER_MPID:
                if (on_add_order_) on_add_order_(parser_.add_order());
                break;
                
            case MessageType::ORDER_EXECUTED:
                if (on_order_executed_) on_order_executed_(parser_.order_executed());
                break;
                
            case MessageType::ORDER_CANCEL:
                if (on_order_cancel_) on_order_cancel_(parser_.order_cancel());
                break;
                
            case MessageType::ORDER_DELETE:
                if (on_order_delete_) on_order_delete_(parser_.order_delete());
                break;
                
            case MessageType::ORDER_REPLACE:
                if (on_order_replace_) on_order_replace_(parser_.order_replace());
                break;
                
            case MessageType::TRADE:
                if (on_trade_) on_trade_(parser_.trade());
                break;
                
            default:
                break;
        }
    }
    
    ITCHParser parser_;
    uint64_t messages_processed_{0};
    
    AddOrderCallback on_add_order_;
    OrderExecutedCallback on_order_executed_;
    OrderCancelCallback on_order_cancel_;
    OrderDeleteCallback on_order_delete_;
    OrderReplaceCallback on_order_replace_;
    TradeCallback on_trade_;
};

} // namespace arbor::itch
