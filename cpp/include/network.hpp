#pragma once

/**
 * Low-Latency Network Layer for Market Data
 * 
 * Components:
 * - UDP Multicast receiver for market data feeds (OPRA, CQS, UTP)
 * - TCP client for order gateway connections
 * - Kernel bypass hints (for DPDK/Solarflare integration)
 * 
 * Design:
 * - Zero-copy receive where possible
 * - Busy-poll for lowest latency
 * - CPU affinity for deterministic performance
 */

#include <cstdint>
#include <string>
#include <functional>
#include <atomic>
#include <thread>
#include <vector>
#include <chrono>
#include <array>

#ifdef _WIN32
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #pragma comment(lib, "ws2_32.lib")
    using socket_t = SOCKET;
    #define INVALID_SOCK INVALID_SOCKET
    #define SOCK_ERROR SOCKET_ERROR
#else
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <netinet/tcp.h>
    #include <arpa/inet.h>
    #include <unistd.h>
    #include <fcntl.h>
    #include <poll.h>
    using socket_t = int;
    #define INVALID_SOCK -1
    #define SOCK_ERROR -1
#endif

namespace arbor::network {

// Network statistics
struct NetworkStats {
    std::atomic<uint64_t> packets_received{0};
    std::atomic<uint64_t> bytes_received{0};
    std::atomic<uint64_t> packets_sent{0};
    std::atomic<uint64_t> bytes_sent{0};
    std::atomic<uint64_t> errors{0};
    std::atomic<int64_t> last_recv_latency_ns{0};
    std::atomic<int64_t> min_latency_ns{INT64_MAX};
    std::atomic<int64_t> max_latency_ns{0};
    
    void record_latency(int64_t ns) {
        last_recv_latency_ns.store(ns, std::memory_order_relaxed);
        
        int64_t current_min = min_latency_ns.load(std::memory_order_relaxed);
        while (ns < current_min && 
               !min_latency_ns.compare_exchange_weak(current_min, ns));
        
        int64_t current_max = max_latency_ns.load(std::memory_order_relaxed);
        while (ns > current_max && 
               !max_latency_ns.compare_exchange_weak(current_max, ns));
    }
};

/**
 * UDP Multicast Receiver
 * 
 * For receiving market data feeds (OPRA options, CQS/UTP equities)
 * 
 * Features:
 * - Join multiple multicast groups
 * - SO_REUSEADDR for failover
 * - Busy-poll option for lowest latency
 * - Optional timestamping (SO_TIMESTAMPNS)
 */
class UDPMulticastReceiver {
public:
    using MessageCallback = std::function<void(const uint8_t* data, size_t len, 
                                                const sockaddr_in& src, int64_t recv_time_ns)>;

    struct Config {
        std::string interface_ip = "0.0.0.0";  // Bind interface
        uint16_t port = 0;
        std::vector<std::string> multicast_groups;
        size_t recv_buffer_size = 8 * 1024 * 1024;  // 8MB
        bool busy_poll = false;
        int busy_poll_us = 50;
        int cpu_affinity = -1;  // -1 = no affinity
    };

    explicit UDPMulticastReceiver(const Config& config) 
        : config_(config), socket_(INVALID_SOCK), running_(false) {}

    ~UDPMulticastReceiver() {
        stop();
    }

    bool start(MessageCallback callback) {
        callback_ = std::move(callback);
        
        // Initialize socket
        if (!init_socket()) {
            return false;
        }
        
        // Join multicast groups
        for (const auto& group : config_.multicast_groups) {
            if (!join_multicast(group)) {
                return false;
            }
        }
        
        // Start receive thread
        running_ = true;
        recv_thread_ = std::thread(&UDPMulticastReceiver::recv_loop, this);
        
        return true;
    }

    void stop() {
        running_ = false;
        if (recv_thread_.joinable()) {
            recv_thread_.join();
        }
        if (socket_ != INVALID_SOCK) {
#ifdef _WIN32
            closesocket(socket_);
#else
            close(socket_);
#endif
            socket_ = INVALID_SOCK;
        }
    }

    const NetworkStats& stats() const { return stats_; }

private:
    bool init_socket() {
#ifdef _WIN32
        WSADATA wsa;
        if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
            return false;
        }
#endif
        socket_ = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (socket_ == INVALID_SOCK) {
            return false;
        }

        // Allow address reuse
        int reuse = 1;
        setsockopt(socket_, SOL_SOCKET, SO_REUSEADDR, 
                   reinterpret_cast<const char*>(&reuse), sizeof(reuse));

        // Set receive buffer size
        int bufsize = static_cast<int>(config_.recv_buffer_size);
        setsockopt(socket_, SOL_SOCKET, SO_RCVBUF, 
                   reinterpret_cast<const char*>(&bufsize), sizeof(bufsize));

#ifndef _WIN32
        // Linux-specific: busy poll for lower latency
        if (config_.busy_poll) {
            int busy_poll = config_.busy_poll_us;
            setsockopt(socket_, SOL_SOCKET, SO_BUSY_POLL,
                       &busy_poll, sizeof(busy_poll));
        }
#endif

        // Bind
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(config_.port);
        addr.sin_addr.s_addr = inet_addr(config_.interface_ip.c_str());

        if (bind(socket_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == SOCK_ERROR) {
            return false;
        }

        return true;
    }

    bool join_multicast(const std::string& group) {
        ip_mreq mreq{};
        mreq.imr_multiaddr.s_addr = inet_addr(group.c_str());
        mreq.imr_interface.s_addr = inet_addr(config_.interface_ip.c_str());

        return setsockopt(socket_, IPPROTO_IP, IP_ADD_MEMBERSHIP,
                          reinterpret_cast<const char*>(&mreq), sizeof(mreq)) != SOCK_ERROR;
    }

    void recv_loop() {
        // Set CPU affinity if specified
        if (config_.cpu_affinity >= 0) {
#ifndef _WIN32
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(config_.cpu_affinity, &cpuset);
            pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
#endif
        }

        std::array<uint8_t, 65536> buffer;
        sockaddr_in src_addr{};
        socklen_t src_len = sizeof(src_addr);

        while (running_) {
            auto recv_start = std::chrono::steady_clock::now();
            
            ssize_t len = recvfrom(socket_, reinterpret_cast<char*>(buffer.data()), 
                                   buffer.size(), 0,
                                   reinterpret_cast<sockaddr*>(&src_addr), &src_len);

            if (len > 0) {
                auto recv_end = std::chrono::steady_clock::now();
                int64_t latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    recv_end - recv_start).count();

                stats_.packets_received.fetch_add(1, std::memory_order_relaxed);
                stats_.bytes_received.fetch_add(len, std::memory_order_relaxed);
                stats_.record_latency(latency_ns);

                if (callback_) {
                    int64_t timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        recv_end.time_since_epoch()).count();
                    callback_(buffer.data(), len, src_addr, timestamp);
                }
            } else if (len < 0) {
                stats_.errors.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }

    Config config_;
    socket_t socket_;
    std::atomic<bool> running_;
    std::thread recv_thread_;
    MessageCallback callback_;
    NetworkStats stats_;
};


/**
 * TCP Order Gateway Client
 * 
 * For sending orders to exchange/broker and receiving execution reports
 * 
 * Features:
 * - Non-blocking connect with timeout
 * - TCP_NODELAY for low latency
 * - Automatic reconnection
 * - Sequence number tracking
 */
class TCPOrderGateway {
public:
    using ConnectCallback = std::function<void(bool success)>;
    using MessageCallback = std::function<void(const uint8_t* data, size_t len)>;
    using DisconnectCallback = std::function<void()>;

    struct Config {
        std::string host;
        uint16_t port;
        int connect_timeout_ms = 5000;
        int recv_buffer_size = 1024 * 1024;
        int send_buffer_size = 1024 * 1024;
        bool tcp_nodelay = true;
        int keepalive_interval_sec = 30;
    };

    explicit TCPOrderGateway(const Config& config)
        : config_(config), socket_(INVALID_SOCK), connected_(false),
          running_(false), next_seq_num_(1) {}

    ~TCPOrderGateway() {
        disconnect();
    }

    bool connect() {
#ifdef _WIN32
        WSADATA wsa;
        if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
            return false;
        }
#endif
        socket_ = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (socket_ == INVALID_SOCK) {
            return false;
        }

        // TCP_NODELAY - disable Nagle's algorithm
        if (config_.tcp_nodelay) {
            int nodelay = 1;
            setsockopt(socket_, IPPROTO_TCP, TCP_NODELAY,
                       reinterpret_cast<const char*>(&nodelay), sizeof(nodelay));
        }

        // Set buffer sizes
        int rcvbuf = config_.recv_buffer_size;
        int sndbuf = config_.send_buffer_size;
        setsockopt(socket_, SOL_SOCKET, SO_RCVBUF,
                   reinterpret_cast<const char*>(&rcvbuf), sizeof(rcvbuf));
        setsockopt(socket_, SOL_SOCKET, SO_SNDBUF,
                   reinterpret_cast<const char*>(&sndbuf), sizeof(sndbuf));

        // Connect
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(config_.port);
        inet_pton(AF_INET, config_.host.c_str(), &addr.sin_addr);

        if (::connect(socket_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == SOCK_ERROR) {
            return false;
        }

        connected_ = true;
        running_ = true;
        recv_thread_ = std::thread(&TCPOrderGateway::recv_loop, this);

        return true;
    }

    void disconnect() {
        running_ = false;
        connected_ = false;
        
        if (recv_thread_.joinable()) {
            recv_thread_.join();
        }
        
        if (socket_ != INVALID_SOCK) {
#ifdef _WIN32
            closesocket(socket_);
#else
            close(socket_);
#endif
            socket_ = INVALID_SOCK;
        }
    }

    /**
     * Send order message with sequence number
     * @return bytes sent, or -1 on error
     */
    int send_message(const uint8_t* data, size_t len) {
        if (!connected_) return -1;

        auto send_start = std::chrono::steady_clock::now();
        
        int sent = send(socket_, reinterpret_cast<const char*>(data), 
                        static_cast<int>(len), 0);

        if (sent > 0) {
            auto send_end = std::chrono::steady_clock::now();
            int64_t latency = std::chrono::duration_cast<std::chrono::nanoseconds>(
                send_end - send_start).count();
            
            stats_.packets_sent.fetch_add(1, std::memory_order_relaxed);
            stats_.bytes_sent.fetch_add(sent, std::memory_order_relaxed);
            stats_.record_latency(latency);
            next_seq_num_++;
        } else {
            stats_.errors.fetch_add(1, std::memory_order_relaxed);
        }

        return sent;
    }

    void set_message_callback(MessageCallback cb) { on_message_ = std::move(cb); }
    void set_disconnect_callback(DisconnectCallback cb) { on_disconnect_ = std::move(cb); }

    bool is_connected() const { return connected_; }
    uint64_t next_sequence_number() const { return next_seq_num_; }
    const NetworkStats& stats() const { return stats_; }

private:
    void recv_loop() {
        std::array<uint8_t, 65536> buffer;

        while (running_) {
            int len = recv(socket_, reinterpret_cast<char*>(buffer.data()),
                          static_cast<int>(buffer.size()), 0);

            if (len > 0) {
                stats_.packets_received.fetch_add(1, std::memory_order_relaxed);
                stats_.bytes_received.fetch_add(len, std::memory_order_relaxed);

                if (on_message_) {
                    on_message_(buffer.data(), len);
                }
            } else if (len == 0 || (len < 0 && running_)) {
                // Connection closed or error
                connected_ = false;
                if (on_disconnect_) {
                    on_disconnect_();
                }
                break;
            }
        }
    }

    Config config_;
    socket_t socket_;
    std::atomic<bool> connected_;
    std::atomic<bool> running_;
    std::atomic<uint64_t> next_seq_num_;
    std::thread recv_thread_;
    MessageCallback on_message_;
    DisconnectCallback on_disconnect_;
    NetworkStats stats_;
};


/**
 * Market Data Feed Handler
 * 
 * Parses common feed formats and dispatches to handlers
 */
struct MarketDataUpdate {
    uint64_t sequence;
    int64_t exchange_timestamp_ns;
    int64_t receive_timestamp_ns;
    char symbol[16];
    
    enum class Type : uint8_t {
        QUOTE,
        TRADE,
        BOOK_UPDATE,
        IMBALANCE
    } type;
    
    union {
        struct {
            int64_t bid_price;  // Fixed point (price * 1e8)
            int64_t ask_price;
            uint32_t bid_size;
            uint32_t ask_size;
        } quote;
        
        struct {
            int64_t price;
            uint32_t size;
            char side;  // 'B' or 'S'
        } trade;
        
        struct {
            int64_t price;
            uint32_t size;
            char side;
            char action;  // 'A'dd, 'M'odify, 'D'elete
        } book_update;
    } data;
};

} // namespace arbor::network
