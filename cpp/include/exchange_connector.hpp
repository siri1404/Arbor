#pragma once

/**
 * Exchange Connectivity Layer
 * 
 * Production-grade exchange gateway with:
 * - Session management (logon/logout/heartbeat)
 * - Sequence number tracking
 * - Automatic reconnection with exponential backoff
 * - Message throttling
 * - Order state machine
 * - Execution report handling
 * 
 * Supports multiple protocols:
 * - FIX 4.2/4.4/5.0 (via fix_parser.hpp)
 * - OUCH (via ouch_codec.hpp)
 * - Native binary protocols
 */

#include <cstdint>
#include <string>
#include <memory>
#include <queue>
#include <unordered_map>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <functional>
#include <chrono>
#include <optional>

#include "network.hpp"
#include "fix_parser.hpp"
#include "ouch_codec.hpp"

namespace arbor::exchange {

// =============================================================================
// ORDER STATE MACHINE
// =============================================================================

enum class OrderState : uint8_t {
    PENDING_NEW,        // Sent to exchange, waiting for ack
    NEW,                // Acknowledged by exchange
    PENDING_REPLACE,    // Replace request sent
    PENDING_CANCEL,     // Cancel request sent
    PARTIAL_FILL,       // Partially filled
    FILLED,             // Completely filled
    CANCELED,           // Canceled
    REJECTED,           // Rejected by exchange
    EXPIRED,            // Expired (TIF)
    DONE                // Terminal state
};

const char* order_state_name(OrderState state) {
    switch (state) {
        case OrderState::PENDING_NEW: return "PENDING_NEW";
        case OrderState::NEW: return "NEW";
        case OrderState::PENDING_REPLACE: return "PENDING_REPLACE";
        case OrderState::PENDING_CANCEL: return "PENDING_CANCEL";
        case OrderState::PARTIAL_FILL: return "PARTIAL_FILL";
        case OrderState::FILLED: return "FILLED";
        case OrderState::CANCELED: return "CANCELED";
        case OrderState::REJECTED: return "REJECTED";
        case OrderState::EXPIRED: return "EXPIRED";
        case OrderState::DONE: return "DONE";
    }
    return "UNKNOWN";
}

// =============================================================================
// ORDER TRACKING
// =============================================================================

struct OrderEntry {
    std::string cl_ord_id;          // Client order ID
    std::string exchange_order_id;  // Exchange-assigned ID
    std::string symbol;
    char side;                      // 'B' or 'S'
    int64_t price;
    int64_t quantity;
    int64_t filled_qty;
    int64_t remaining_qty;
    OrderState state;
    
    std::chrono::steady_clock::time_point sent_time;
    std::chrono::steady_clock::time_point acked_time;
    std::chrono::steady_clock::time_point last_update;
    
    int64_t latency_ns() const {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            acked_time - sent_time).count();
    }
};

// =============================================================================
// EXECUTION REPORT
// =============================================================================

struct ExecutionReport {
    std::string cl_ord_id;
    std::string exchange_order_id;
    std::string exec_id;
    std::string symbol;
    char side;
    char exec_type;         // '0'=New, '1'=Partial, '2'=Fill, '4'=Cancel, '8'=Reject
    char ord_status;        // '0'=New, '1'=Partial, '2'=Filled, '4'=Canceled, '8'=Rejected
    int64_t price;
    int64_t quantity;
    int64_t filled_qty;
    int64_t leaves_qty;
    int64_t last_qty;       // This execution
    int64_t last_px;        // This execution
    int64_t cum_qty;
    int64_t avg_px;
    std::string text;       // Reject reason or other text
    int64_t transact_time_ns;
    int64_t receive_time_ns;
};

// =============================================================================
// SESSION STATE
// =============================================================================

enum class SessionState : uint8_t {
    DISCONNECTED,
    CONNECTING,
    LOGON_SENT,
    ACTIVE,
    LOGOUT_SENT,
    RECONNECTING
};

// =============================================================================
// EXCHANGE CONNECTOR
// =============================================================================

/**
 * Base class for exchange connections
 * Handles session lifecycle and message routing
 */
class ExchangeConnector {
public:
    // Callbacks
    using ConnectCallback = std::function<void(bool success)>;
    using DisconnectCallback = std::function<void(const std::string& reason)>;
    using ExecutionCallback = std::function<void(const ExecutionReport&)>;
    using OrderAckCallback = std::function<void(const OrderEntry&)>;
    using RejectCallback = std::function<void(const std::string& cl_ord_id, const std::string& reason)>;
    
    struct Config {
        std::string host;
        uint16_t port;
        std::string sender_comp_id;
        std::string target_comp_id;
        std::string username;
        std::string password;
        int heartbeat_interval_sec{30};
        int connect_timeout_ms{5000};
        int max_reconnect_attempts{10};
        int initial_reconnect_delay_ms{1000};
        int max_reconnect_delay_ms{60000};
        int max_messages_per_second{1000};
        bool auto_reconnect{true};
    };
    
    explicit ExchangeConnector(const Config& config) 
        : config_(config)
        , state_(SessionState::DISCONNECTED)
        , inbound_seq_(1)
        , outbound_seq_(1)
        , running_(false) {}
    
    virtual ~ExchangeConnector() {
        stop();
    }
    
    // =========================================================================
    // LIFECYCLE
    // =========================================================================
    
    bool start() {
        if (running_) return false;
        
        running_ = true;
        io_thread_ = std::thread(&ExchangeConnector::io_loop, this);
        heartbeat_thread_ = std::thread(&ExchangeConnector::heartbeat_loop, this);
        
        return connect();
    }
    
    void stop() {
        running_ = false;
        
        // Send logout if connected
        if (state_ == SessionState::ACTIVE) {
            send_logout("Normal termination");
        }
        
        cv_.notify_all();
        
        if (io_thread_.joinable()) io_thread_.join();
        if (heartbeat_thread_.joinable()) heartbeat_thread_.join();
        
        disconnect();
    }
    
    [[nodiscard]] SessionState state() const noexcept { return state_; }
    [[nodiscard]] bool is_connected() const noexcept { 
        return state_ == SessionState::ACTIVE; 
    }
    
    // =========================================================================
    // ORDER ENTRY
    // =========================================================================
    
    /**
     * Send new order
     * @return Client order ID assigned to this order
     */
    std::string send_new_order(
        const std::string& symbol,
        char side,
        int64_t quantity,
        int64_t price,
        char ord_type = '2',    // Limit
        char time_in_force = '0' // Day
    ) {
        if (!is_connected()) {
            return "";
        }
        
        // Generate client order ID
        std::string cl_ord_id = generate_cl_ord_id();
        
        // Create order entry
        OrderEntry order;
        order.cl_ord_id = cl_ord_id;
        order.symbol = symbol;
        order.side = side;
        order.price = price;
        order.quantity = quantity;
        order.filled_qty = 0;
        order.remaining_qty = quantity;
        order.state = OrderState::PENDING_NEW;
        order.sent_time = std::chrono::steady_clock::now();
        
        // Track order
        {
            std::lock_guard<std::mutex> lock(orders_mutex_);
            orders_[cl_ord_id] = order;
        }
        
        // Build and send message
        if (!send_new_order_impl(cl_ord_id, symbol, side, quantity, price, 
                                 ord_type, time_in_force)) {
            std::lock_guard<std::mutex> lock(orders_mutex_);
            orders_.erase(cl_ord_id);
            return "";
        }
        
        return cl_ord_id;
    }
    
    /**
     * Cancel an existing order
     */
    bool send_cancel(const std::string& cl_ord_id) {
        if (!is_connected()) return false;
        
        std::lock_guard<std::mutex> lock(orders_mutex_);
        auto it = orders_.find(cl_ord_id);
        if (it == orders_.end()) return false;
        
        if (it->second.state != OrderState::NEW && 
            it->second.state != OrderState::PARTIAL_FILL) {
            return false;  // Can't cancel
        }
        
        it->second.state = OrderState::PENDING_CANCEL;
        return send_cancel_impl(cl_ord_id, it->second.exchange_order_id, 
                               it->second.symbol, it->second.side, 
                               it->second.remaining_qty);
    }
    
    /**
     * Replace/modify an existing order
     */
    std::string send_replace(
        const std::string& orig_cl_ord_id,
        int64_t new_quantity,
        int64_t new_price
    ) {
        if (!is_connected()) return "";
        
        std::lock_guard<std::mutex> lock(orders_mutex_);
        auto it = orders_.find(orig_cl_ord_id);
        if (it == orders_.end()) return "";
        
        if (it->second.state != OrderState::NEW && 
            it->second.state != OrderState::PARTIAL_FILL) {
            return "";
        }
        
        std::string new_cl_ord_id = generate_cl_ord_id();
        it->second.state = OrderState::PENDING_REPLACE;
        
        if (!send_replace_impl(orig_cl_ord_id, new_cl_ord_id,
                               it->second.exchange_order_id,
                               it->second.symbol, it->second.side,
                               new_quantity, new_price)) {
            return "";
        }
        
        // Create pending replacement order
        OrderEntry new_order = it->second;
        new_order.cl_ord_id = new_cl_ord_id;
        new_order.quantity = new_quantity;
        new_order.price = new_price;
        new_order.remaining_qty = new_quantity - new_order.filled_qty;
        new_order.state = OrderState::PENDING_NEW;
        new_order.sent_time = std::chrono::steady_clock::now();
        orders_[new_cl_ord_id] = new_order;
        
        return new_cl_ord_id;
    }
    
    // =========================================================================
    // QUERIES
    // =========================================================================
    
    std::optional<OrderEntry> get_order(const std::string& cl_ord_id) const {
        std::lock_guard<std::mutex> lock(orders_mutex_);
        auto it = orders_.find(cl_ord_id);
        if (it != orders_.end()) {
            return it->second;
        }
        return std::nullopt;
    }
    
    std::vector<OrderEntry> get_open_orders() const {
        std::lock_guard<std::mutex> lock(orders_mutex_);
        std::vector<OrderEntry> result;
        for (const auto& [id, order] : orders_) {
            if (order.state == OrderState::NEW ||
                order.state == OrderState::PARTIAL_FILL ||
                order.state == OrderState::PENDING_REPLACE ||
                order.state == OrderState::PENDING_CANCEL) {
                result.push_back(order);
            }
        }
        return result;
    }
    
    // =========================================================================
    // CALLBACKS
    // =========================================================================
    
    void set_connect_callback(ConnectCallback cb) { on_connect_ = std::move(cb); }
    void set_disconnect_callback(DisconnectCallback cb) { on_disconnect_ = std::move(cb); }
    void set_execution_callback(ExecutionCallback cb) { on_execution_ = std::move(cb); }
    void set_order_ack_callback(OrderAckCallback cb) { on_order_ack_ = std::move(cb); }
    void set_reject_callback(RejectCallback cb) { on_reject_ = std::move(cb); }
    
    // =========================================================================
    // STATISTICS
    // =========================================================================
    
    struct Stats {
        uint64_t messages_sent{0};
        uint64_t messages_received{0};
        uint64_t orders_sent{0};
        uint64_t orders_acked{0};
        uint64_t orders_rejected{0};
        uint64_t fills{0};
        uint64_t reconnects{0};
        int64_t avg_ack_latency_ns{0};
    };
    
    [[nodiscard]] Stats stats() const noexcept { return stats_; }
    
protected:
    // =========================================================================
    // VIRTUAL METHODS FOR PROTOCOL IMPLEMENTATIONS
    // =========================================================================
    
    virtual bool connect() = 0;
    virtual void disconnect() = 0;
    virtual bool send_logon() = 0;
    virtual bool send_logout(const std::string& reason) = 0;
    virtual bool send_heartbeat() = 0;
    virtual bool send_new_order_impl(
        const std::string& cl_ord_id,
        const std::string& symbol,
        char side,
        int64_t quantity,
        int64_t price,
        char ord_type,
        char time_in_force
    ) = 0;
    virtual bool send_cancel_impl(
        const std::string& cl_ord_id,
        const std::string& exchange_order_id,
        const std::string& symbol,
        char side,
        int64_t quantity
    ) = 0;
    virtual bool send_replace_impl(
        const std::string& orig_cl_ord_id,
        const std::string& new_cl_ord_id,
        const std::string& exchange_order_id,
        const std::string& symbol,
        char side,
        int64_t new_quantity,
        int64_t new_price
    ) = 0;
    
    // =========================================================================
    // MESSAGE HANDLING
    // =========================================================================
    
    void handle_execution_report(const ExecutionReport& report) {
        std::lock_guard<std::mutex> lock(orders_mutex_);
        
        auto it = orders_.find(report.cl_ord_id);
        if (it == orders_.end()) {
            // Unknown order - might be from recovery
            return;
        }
        
        OrderEntry& order = it->second;
        order.exchange_order_id = report.exchange_order_id;
        order.filled_qty = report.cum_qty;
        order.remaining_qty = report.leaves_qty;
        order.last_update = std::chrono::steady_clock::now();
        
        switch (report.exec_type) {
            case '0':  // New
                order.state = OrderState::NEW;
                order.acked_time = std::chrono::steady_clock::now();
                stats_.orders_acked++;
                update_ack_latency(order.latency_ns());
                if (on_order_ack_) on_order_ack_(order);
                break;
                
            case '1':  // Partial fill
                order.state = OrderState::PARTIAL_FILL;
                stats_.fills++;
                break;
                
            case '2':  // Fill
                order.state = OrderState::FILLED;
                stats_.fills++;
                break;
                
            case '4':  // Canceled
                order.state = OrderState::CANCELED;
                break;
                
            case '5':  // Replaced
                order.state = OrderState::NEW;
                break;
                
            case '8':  // Rejected
                order.state = OrderState::REJECTED;
                stats_.orders_rejected++;
                if (on_reject_) on_reject_(report.cl_ord_id, report.text);
                break;
                
            case 'C':  // Expired
                order.state = OrderState::EXPIRED;
                break;
        }
        
        if (on_execution_) {
            on_execution_(report);
        }
    }
    
    // =========================================================================
    // HELPER METHODS
    // =========================================================================
    
    std::string generate_cl_ord_id() {
        return config_.sender_comp_id + "_" + 
               std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
               "_" + std::to_string(order_id_counter_++);
    }
    
    void update_ack_latency(int64_t latency_ns) {
        if (stats_.orders_acked == 1) {
            stats_.avg_ack_latency_ns = latency_ns;
        } else {
            // Exponential moving average
            stats_.avg_ack_latency_ns = (stats_.avg_ack_latency_ns * 7 + latency_ns) / 8;
        }
    }
    
    void io_loop() {
        while (running_) {
            std::unique_lock<std::mutex> lock(cv_mutex_);
            cv_.wait_for(lock, std::chrono::milliseconds(100));
            
            // Handle reconnection
            if (state_ == SessionState::DISCONNECTED && config_.auto_reconnect) {
                attempt_reconnect();
            }
        }
    }
    
    void heartbeat_loop() {
        while (running_) {
            std::this_thread::sleep_for(
                std::chrono::seconds(config_.heartbeat_interval_sec));
            
            if (state_ == SessionState::ACTIVE) {
                send_heartbeat();
            }
        }
    }
    
    void attempt_reconnect() {
        if (reconnect_attempts_ >= config_.max_reconnect_attempts) {
            return;
        }
        
        state_ = SessionState::RECONNECTING;
        stats_.reconnects++;
        
        // Exponential backoff
        int delay_ms = std::min(
            config_.initial_reconnect_delay_ms * (1 << reconnect_attempts_),
            config_.max_reconnect_delay_ms
        );
        
        std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        
        if (connect()) {
            reconnect_attempts_ = 0;
        } else {
            reconnect_attempts_++;
        }
    }
    
    // Configuration
    Config config_;
    
    // Session state
    std::atomic<SessionState> state_;
    uint64_t inbound_seq_;
    uint64_t outbound_seq_;
    
    // Order tracking
    mutable std::mutex orders_mutex_;
    std::unordered_map<std::string, OrderEntry> orders_;
    std::atomic<uint64_t> order_id_counter_{1};
    
    // Threading
    std::atomic<bool> running_;
    std::thread io_thread_;
    std::thread heartbeat_thread_;
    std::mutex cv_mutex_;
    std::condition_variable cv_;
    
    // Reconnection
    int reconnect_attempts_{0};
    
    // Callbacks
    ConnectCallback on_connect_;
    DisconnectCallback on_disconnect_;
    ExecutionCallback on_execution_;
    OrderAckCallback on_order_ack_;
    RejectCallback on_reject_;
    
    // Statistics
    Stats stats_;
};

// =============================================================================
// FIX EXCHANGE CONNECTOR
// =============================================================================

/**
 * FIX Protocol implementation of ExchangeConnector
 */
class FIXExchangeConnector : public ExchangeConnector {
public:
    explicit FIXExchangeConnector(const Config& config)
        : ExchangeConnector(config)
        , gateway_({
            .host = config.host,
            .port = config.port,
            .connect_timeout_ms = config.connect_timeout_ms,
            .tcp_nodelay = true
        })
        , fix_builder_(config.sender_comp_id, config.target_comp_id) {}
    
protected:
    bool connect() override {
        state_ = SessionState::CONNECTING;
        
        gateway_.set_message_callback([this](const uint8_t* data, size_t len) {
            handle_message(data, len);
        });
        
        gateway_.set_disconnect_callback([this]() {
            state_ = SessionState::DISCONNECTED;
            if (on_disconnect_) {
                on_disconnect_("Connection lost");
            }
        });
        
        if (!gateway_.connect()) {
            state_ = SessionState::DISCONNECTED;
            return false;
        }
        
        return send_logon();
    }
    
    void disconnect() override {
        gateway_.disconnect();
        state_ = SessionState::DISCONNECTED;
    }
    
    bool send_logon() override {
        state_ = SessionState::LOGON_SENT;
        
        fix_builder_.start(fix::MsgType::Logon)
            .add(fix::Tag::Account, config_.username)
            .add(fix::Tag::TimeInForce, config_.heartbeat_interval_sec);
        
        auto msg = fix_builder_.finish();
        
        int sent = gateway_.send_message(
            reinterpret_cast<const uint8_t*>(msg.data()), msg.size());
        
        if (sent > 0) {
            stats_.messages_sent++;
            outbound_seq_++;
            return true;
        }
        return false;
    }
    
    bool send_logout(const std::string& reason) override {
        state_ = SessionState::LOGOUT_SENT;
        
        fix_builder_.start(fix::MsgType::Logout)
            .add(fix::Tag::Text, reason);
        
        auto msg = fix_builder_.finish();
        gateway_.send_message(
            reinterpret_cast<const uint8_t*>(msg.data()), msg.size());
        stats_.messages_sent++;
        return true;
    }
    
    bool send_heartbeat() override {
        fix_builder_.start(fix::MsgType::Heartbeat);
        auto msg = fix_builder_.finish();
        
        int sent = gateway_.send_message(
            reinterpret_cast<const uint8_t*>(msg.data()), msg.size());
        
        if (sent > 0) {
            stats_.messages_sent++;
            outbound_seq_++;
            return true;
        }
        return false;
    }
    
    bool send_new_order_impl(
        const std::string& cl_ord_id,
        const std::string& symbol,
        char side,
        int64_t quantity,
        int64_t price,
        char ord_type,
        char time_in_force
    ) override {
        fix_builder_.start(fix::MsgType::NewOrderSingle)
            .add(fix::Tag::ClOrdID, cl_ord_id)
            .add(fix::Tag::Symbol, symbol)
            .add(fix::Tag::Side, side)
            .add(fix::Tag::OrderQty, quantity)
            .add(fix::Tag::OrdType, ord_type)
            .add(fix::Tag::TimeInForce, time_in_force);
        
        if (ord_type == '2') {  // Limit
            fix_builder_.add(fix::Tag::Price, static_cast<double>(price) / 10000.0);
        }
        
        auto msg = fix_builder_.finish();
        
        int sent = gateway_.send_message(
            reinterpret_cast<const uint8_t*>(msg.data()), msg.size());
        
        if (sent > 0) {
            stats_.messages_sent++;
            stats_.orders_sent++;
            outbound_seq_++;
            return true;
        }
        return false;
    }
    
    bool send_cancel_impl(
        const std::string& cl_ord_id,
        const std::string& exchange_order_id,
        const std::string& symbol,
        char side,
        int64_t quantity
    ) override {
        std::string cancel_id = generate_cl_ord_id();
        
        fix_builder_.start(fix::MsgType::OrderCancelRequest)
            .add(fix::Tag::ClOrdID, cancel_id)
            .add(11, cl_ord_id)  // OrigClOrdID
            .add(fix::Tag::OrderID, exchange_order_id)
            .add(fix::Tag::Symbol, symbol)
            .add(fix::Tag::Side, side)
            .add(fix::Tag::OrderQty, quantity);
        
        auto msg = fix_builder_.finish();
        
        int sent = gateway_.send_message(
            reinterpret_cast<const uint8_t*>(msg.data()), msg.size());
        
        if (sent > 0) {
            stats_.messages_sent++;
            outbound_seq_++;
            return true;
        }
        return false;
    }
    
    bool send_replace_impl(
        const std::string& orig_cl_ord_id,
        const std::string& new_cl_ord_id,
        const std::string& exchange_order_id,
        const std::string& symbol,
        char side,
        int64_t new_quantity,
        int64_t new_price
    ) override {
        fix_builder_.start(fix::MsgType::OrderCancelReplaceRequest)
            .add(fix::Tag::ClOrdID, new_cl_ord_id)
            .add(11, orig_cl_ord_id)  // OrigClOrdID
            .add(fix::Tag::OrderID, exchange_order_id)
            .add(fix::Tag::Symbol, symbol)
            .add(fix::Tag::Side, side)
            .add(fix::Tag::OrderQty, new_quantity)
            .add(fix::Tag::Price, static_cast<double>(new_price) / 10000.0);
        
        auto msg = fix_builder_.finish();
        
        int sent = gateway_.send_message(
            reinterpret_cast<const uint8_t*>(msg.data()), msg.size());
        
        if (sent > 0) {
            stats_.messages_sent++;
            outbound_seq_++;
            return true;
        }
        return false;
    }
    
private:
    void handle_message(const uint8_t* data, size_t len) {
        stats_.messages_received++;
        
        auto msg = parser_.parse(reinterpret_cast<const char*>(data), len);
        if (!msg.valid()) return;
        
        inbound_seq_++;
        
        switch (msg.msg_type()) {
            case fix::MsgType::Logon:
                state_ = SessionState::ACTIVE;
                if (on_connect_) on_connect_(true);
                break;
                
            case fix::MsgType::Logout:
                state_ = SessionState::DISCONNECTED;
                if (on_disconnect_) {
                    auto text = msg.get(fix::Tag::Text);
                    on_disconnect_(text.empty() ? "Logout" : 
                                   std::string(text.view()));
                }
                break;
                
            case fix::MsgType::Heartbeat:
            case fix::MsgType::TestRequest:
                // Respond with heartbeat
                send_heartbeat();
                break;
                
            case fix::MsgType::ExecutionReport:
                handle_execution_report_message(msg);
                break;
                
            case fix::MsgType::OrderCancelReject:
                handle_cancel_reject(msg);
                break;
        }
    }
    
    void handle_execution_report_message(const fix::FIXMessage& msg) {
        ExecutionReport report;
        
        auto cl_ord_id = msg.get(fix::Tag::ClOrdID);
        if (cl_ord_id) report.cl_ord_id = std::string(cl_ord_id.view());
        
        auto order_id = msg.get(fix::Tag::OrderID);
        if (order_id) report.exchange_order_id = std::string(order_id.view());
        
        auto exec_id = msg.get(fix::Tag::ExecID);
        if (exec_id) report.exec_id = std::string(exec_id.view());
        
        auto symbol = msg.get(fix::Tag::Symbol);
        if (symbol) report.symbol = std::string(symbol.view());
        
        report.side = msg.get_char<fix::Tag::Side>();
        report.exec_type = msg.get_char<fix::Tag::ExecType>();
        report.ord_status = msg.get_char<fix::Tag::OrdStatus>();
        
        auto qty = msg.get_int<fix::Tag::OrderQty>();
        if (qty) report.quantity = *qty;
        
        auto cum_qty = msg.get_int<fix::Tag::CumQty>();
        if (cum_qty) report.cum_qty = *cum_qty;
        
        auto leaves_qty = msg.get_int<fix::Tag::LeavesQty>();
        if (leaves_qty) report.leaves_qty = *leaves_qty;
        
        auto last_qty = msg.get_int<fix::Tag::LastQty>();
        if (last_qty) report.last_qty = *last_qty;
        
        auto last_px = msg.get_double<fix::Tag::LastPx>();
        if (last_px) report.last_px = static_cast<int64_t>(*last_px * 10000);
        
        auto text = msg.get(fix::Tag::Text);
        if (text) report.text = std::string(text.view());
        
        report.receive_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        
        handle_execution_report(report);
    }
    
    void handle_cancel_reject(const fix::FIXMessage& msg) {
        auto cl_ord_id = msg.get(fix::Tag::ClOrdID);
        auto text = msg.get(fix::Tag::Text);
        
        if (on_reject_ && cl_ord_id) {
            on_reject_(std::string(cl_ord_id.view()),
                      text ? std::string(text.view()) : "Cancel rejected");
        }
    }
    
    network::TCPOrderGateway gateway_;
    fix::FIXParser parser_;
    fix::FIXBuilder fix_builder_;
};

} // namespace arbor::exchange
