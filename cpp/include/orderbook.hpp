#pragma once

/**
 * Production-Grade HFT Order Book Engine
 * 
 * Key Design Decisions:
 * 1. Intrusive doubly-linked lists for O(1) order insertion/removal
 * 2. Custom slab allocator - ZERO malloc in hot path
 * 3. Price levels stored in sorted intrusive skip list for O(log n) price lookup
 * 4. Cache-line aligned structures to prevent false sharing
 * 5. Memory ordering carefully chosen for each atomic operation
 * 6. No std::unordered_map (unpredictable rehash latency)
 * 7. No std::vector::erase (O(n) removal)
 * 
 * Complexity Guarantees:
 * - Add order:     O(log P) where P = number of price levels
 * - Cancel order:  O(1) via direct pointer
 * - Match order:   O(1) per fill
 * - Best bid/ask:  O(1)
 * 
 * Memory Model:
 * - All orders pre-allocated in contiguous slab
 * - Free list for O(1) allocation/deallocation
 * - No memory allocation during trading hours
 */

#include <cstdint>
#include <cstring>
#include <array>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <string>
#include <functional>
#include <new>
#include <bit>
#include <limits>
#include <vector>

namespace arbor::orderbook {

// =============================================================================
// CONSTANTS AND CONFIGURATION
// =============================================================================

static constexpr size_t CACHE_LINE_SIZE = 64;
static constexpr size_t MAX_ORDERS = 1'000'000;        // 1M orders
static constexpr size_t MAX_PRICE_LEVELS = 100'000;    // 100K price levels
static constexpr size_t SKIPLIST_MAX_LEVEL = 16;       // Log2(MAX_PRICE_LEVELS)
static constexpr double SKIPLIST_P = 0.5;              // Probability for level promotion

using Timestamp = std::chrono::time_point<std::chrono::steady_clock>;
using Nanoseconds = std::chrono::nanoseconds;

// =============================================================================
// ENUMS
// =============================================================================

enum class Side : uint8_t { BUY = 0, SELL = 1 };
enum class OrderType : uint8_t { LIMIT = 0, MARKET = 1, IOC = 2, FOK = 3 };
enum class OrderStatus : uint8_t { 
    NEW = 0, 
    PARTIAL = 1, 
    FILLED = 2, 
    CANCELLED = 3,
    REJECTED = 4 
};
enum class TimeInForce : uint8_t {
    GTC = 0,  // Good Till Cancel
    IOC = 1,  // Immediate Or Cancel
    FOK = 2,  // Fill Or Kill
    DAY = 3   // Day order
};

// =============================================================================
// INTRUSIVE LIST NODE
// Forward/backward pointers embedded in the object itself
// =============================================================================

template<typename T>
struct IntrusiveListNode {
    T* prev{nullptr};
    T* next{nullptr};
    
    void unlink() noexcept {
        if (prev) prev->list_node.next = next;
        if (next) next->list_node.prev = prev;
        prev = nullptr;
        next = nullptr;
    }
    
    [[nodiscard]] bool is_linked() const noexcept {
        return prev != nullptr || next != nullptr;
    }
};

// =============================================================================
// INTRUSIVE DOUBLY-LINKED LIST
// O(1) insert/remove, cache-friendly iteration
// =============================================================================

template<typename T>
class IntrusiveList {
public:
    IntrusiveList() noexcept : head_(nullptr), tail_(nullptr), size_(0) {}
    
    // Disable copy
    IntrusiveList(const IntrusiveList&) = delete;
    IntrusiveList& operator=(const IntrusiveList&) = delete;
    
    // Enable move
    IntrusiveList(IntrusiveList&& other) noexcept 
        : head_(other.head_), tail_(other.tail_), size_(other.size_) {
        other.head_ = other.tail_ = nullptr;
        other.size_ = 0;
    }
    
    IntrusiveList& operator=(IntrusiveList&& other) noexcept {
        if (this != &other) {
            head_ = other.head_;
            tail_ = other.tail_;
            size_ = other.size_;
            other.head_ = other.tail_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    // O(1) push back
    void push_back(T* node) noexcept {
        node->list_node.prev = tail_;
        node->list_node.next = nullptr;
        
        if (tail_) {
            tail_->list_node.next = node;
        } else {
            head_ = node;
        }
        tail_ = node;
        ++size_;
    }
    
    // O(1) push front
    void push_front(T* node) noexcept {
        node->list_node.prev = nullptr;
        node->list_node.next = head_;
        
        if (head_) {
            head_->list_node.prev = node;
        } else {
            tail_ = node;
        }
        head_ = node;
        ++size_;
    }
    
    // O(1) pop front
    T* pop_front() noexcept {
        if (!head_) return nullptr;
        
        T* node = head_;
        head_ = head_->list_node.next;
        
        if (head_) {
            head_->list_node.prev = nullptr;
        } else {
            tail_ = nullptr;
        }
        
        node->list_node.prev = nullptr;
        node->list_node.next = nullptr;
        --size_;
        return node;
    }
    
    // O(1) remove arbitrary node (if node has valid prev/next pointers)
    void remove(T* node) noexcept {
        if (!node) return;
        
        if (node == head_) {
            head_ = node->list_node.next;
        }
        if (node == tail_) {
            tail_ = node->list_node.prev;
        }
        
        if (node->list_node.prev) {
            node->list_node.prev->list_node.next = node->list_node.next;
        }
        if (node->list_node.next) {
            node->list_node.next->list_node.prev = node->list_node.prev;
        }
        
        node->list_node.prev = nullptr;
        node->list_node.next = nullptr;
        --size_;
    }
    
    [[nodiscard]] T* front() const noexcept { return head_; }
    [[nodiscard]] T* back() const noexcept { return tail_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
    [[nodiscard]] size_t size() const noexcept { return size_; }
    
    // Iterator support for range-based for loops
    class Iterator {
    public:
        explicit Iterator(T* node) : node_(node) {}
        T& operator*() const { return *node_; }
        T* operator->() const { return node_; }
        Iterator& operator++() { node_ = node_->list_node.next; return *this; }
        bool operator!=(const Iterator& other) const { return node_ != other.node_; }
    private:
        T* node_;
    };
    
    [[nodiscard]] Iterator begin() const { return Iterator(head_); }
    [[nodiscard]] Iterator end() const { return Iterator(nullptr); }
    
private:
    T* head_;
    T* tail_;
    size_t size_;
};

// =============================================================================
// ORDER STRUCTURE
// Cache-line aligned, intrusive list node embedded
// =============================================================================

struct alignas(CACHE_LINE_SIZE) Order {
    // Intrusive list node for O(1) removal from price level
    IntrusiveListNode<Order> list_node;
    
    // Order identifiers
    uint64_t order_id;
    uint64_t client_order_id;      // Client-assigned ID
    
    // Price and quantity (using integer ticks to avoid FP)
    uint64_t price_ticks;
    uint32_t quantity;
    uint32_t filled_quantity;
    uint32_t visible_quantity;     // For iceberg orders
    uint32_t hidden_quantity;      // Hidden portion of iceberg
    
    // Order attributes
    Side side;
    OrderType type;
    OrderStatus status;
    TimeInForce tif;
    
    // Timestamps for latency measurement
    Timestamp entry_time;
    Timestamp last_update_time;
    
    // Pointer back to price level for O(1) access
    struct PriceLevel* price_level;
    
    // Methods
    [[nodiscard]] inline bool is_buy() const noexcept { return side == Side::BUY; }
    [[nodiscard]] inline bool is_sell() const noexcept { return side == Side::SELL; }
    [[nodiscard]] inline uint32_t remaining_qty() const noexcept { 
        return quantity - filled_quantity; 
    }
    [[nodiscard]] inline uint32_t displayable_qty() const noexcept {
        return std::min(visible_quantity, remaining_qty());
    }
    [[nodiscard]] inline bool is_fully_filled() const noexcept {
        return filled_quantity >= quantity;
    }
};

static_assert(sizeof(Order) <= 2 * CACHE_LINE_SIZE, "Order should fit in 2 cache lines");

// =============================================================================
// PRICE LEVEL STRUCTURE
// Contains intrusive list of orders at this price
// =============================================================================

struct alignas(CACHE_LINE_SIZE) PriceLevel {
    // Intrusive skip list node for price level ordering
    uint64_t price_ticks;
    Side side;
    
    // Orders at this price level (FIFO queue)
    IntrusiveList<Order> orders;
    
    // Aggregated quantities for market data
    uint64_t total_quantity{0};
    uint64_t visible_quantity{0};
    uint32_t order_count{0};
    
    // Skip list forward pointers (for sorted price level traversal)
    std::array<PriceLevel*, SKIPLIST_MAX_LEVEL> forward{};
    int level{0};  // Height of this node in skip list
    
    void add_order(Order* order) noexcept {
        orders.push_back(order);
        order->price_level = this;
        total_quantity += order->remaining_qty();
        visible_quantity += order->displayable_qty();
        ++order_count;
    }
    
    void remove_order(Order* order) noexcept {
        total_quantity -= order->remaining_qty();
        visible_quantity -= order->displayable_qty();
        --order_count;
        orders.remove(order);
        order->price_level = nullptr;
    }
    
    void update_quantity_after_fill(uint32_t fill_qty, uint32_t visible_fill) noexcept {
        total_quantity -= fill_qty;
        visible_quantity -= visible_fill;
    }
    
    [[nodiscard]] bool empty() const noexcept { return orders.empty(); }
};

// =============================================================================
// SLAB ALLOCATOR
// Pre-allocated memory pool with O(1) alloc/free, zero malloc in hot path
// =============================================================================

template<typename T, size_t Capacity>
class SlabAllocator {
public:
    SlabAllocator() {
        // Allocate contiguous memory block
        storage_ = static_cast<T*>(std::aligned_alloc(alignof(T), sizeof(T) * Capacity));
        if (!storage_) {
            throw std::bad_alloc();
        }
        
        // Initialize free list
        free_head_ = 0;
        for (size_t i = 0; i < Capacity - 1; ++i) {
            next_free_[i] = i + 1;
        }
        next_free_[Capacity - 1] = INVALID_INDEX;
        allocated_count_ = 0;
    }
    
    ~SlabAllocator() {
        if (storage_) {
            std::free(storage_);
        }
    }
    
    // Disable copy
    SlabAllocator(const SlabAllocator&) = delete;
    SlabAllocator& operator=(const SlabAllocator&) = delete;
    
    // O(1) allocation - no system call, no lock
    [[nodiscard]] T* allocate() noexcept {
        if (free_head_ == INVALID_INDEX) {
            return nullptr;  // Pool exhausted
        }
        
        size_t index = free_head_;
        free_head_ = next_free_[index];
        ++allocated_count_;
        
        // Construct in place
        T* ptr = &storage_[index];
        new (ptr) T{};
        return ptr;
    }
    
    // O(1) deallocation - return to free list
    void deallocate(T* ptr) noexcept {
        if (!ptr) return;
        
        // Call destructor
        ptr->~T();
        
        // Return to free list
        size_t index = ptr - storage_;
        next_free_[index] = free_head_;
        free_head_ = index;
        --allocated_count_;
    }
    
    [[nodiscard]] size_t capacity() const noexcept { return Capacity; }
    [[nodiscard]] size_t allocated() const noexcept { return allocated_count_; }
    [[nodiscard]] size_t available() const noexcept { return Capacity - allocated_count_; }
    [[nodiscard]] bool full() const noexcept { return free_head_ == INVALID_INDEX; }
    
    // Get index from pointer (for order ID generation)
    [[nodiscard]] size_t index_of(const T* ptr) const noexcept {
        return ptr - storage_;
    }
    
    // Get pointer from index
    [[nodiscard]] T* at_index(size_t index) noexcept {
        return (index < Capacity) ? &storage_[index] : nullptr;
    }
    
    void reset() noexcept {
        free_head_ = 0;
        for (size_t i = 0; i < Capacity - 1; ++i) {
            next_free_[i] = i + 1;
        }
        next_free_[Capacity - 1] = INVALID_INDEX;
        allocated_count_ = 0;
    }
    
private:
    static constexpr size_t INVALID_INDEX = std::numeric_limits<size_t>::max();
    
    T* storage_{nullptr};
    std::array<size_t, Capacity> next_free_{};
    size_t free_head_{0};
    size_t allocated_count_{0};
};

// =============================================================================
// SKIP LIST FOR PRICE LEVELS
// O(log n) insert/search, O(1) min/max access, better cache behavior than tree
// =============================================================================

template<Side S>
class PriceLevelSkipList {
public:
    PriceLevelSkipList() : level_(0), size_(0), rng_(std::random_device{}()) {
        // Head is a sentinel node
        head_.level = SKIPLIST_MAX_LEVEL - 1;
        head_.forward.fill(nullptr);
    }
    
    // O(log n) insert
    void insert(PriceLevel* node) noexcept {
        std::array<PriceLevel*, SKIPLIST_MAX_LEVEL> update{};
        PriceLevel* current = &head_;
        
        // Find insertion point at each level
        for (int i = level_; i >= 0; --i) {
            while (current->forward[i] && compare(current->forward[i], node)) {
                current = current->forward[i];
            }
            update[i] = current;
        }
        
        // Randomize height for new node
        int new_level = random_level();
        if (new_level > level_) {
            for (int i = level_ + 1; i <= new_level; ++i) {
                update[i] = &head_;
            }
            level_ = new_level;
        }
        
        // Insert at each level
        node->level = new_level;
        for (int i = 0; i <= new_level; ++i) {
            node->forward[i] = update[i]->forward[i];
            update[i]->forward[i] = node;
        }
        
        ++size_;
    }
    
    // O(log n) remove
    void remove(PriceLevel* node) noexcept {
        std::array<PriceLevel*, SKIPLIST_MAX_LEVEL> update{};
        PriceLevel* current = &head_;
        
        for (int i = level_; i >= 0; --i) {
            while (current->forward[i] && current->forward[i] != node && 
                   compare(current->forward[i], node)) {
                current = current->forward[i];
            }
            update[i] = current;
        }
        
        // Remove from each level
        for (int i = 0; i <= node->level; ++i) {
            if (update[i]->forward[i] == node) {
                update[i]->forward[i] = node->forward[i];
            }
        }
        
        // Update list level if necessary
        while (level_ > 0 && head_.forward[level_] == nullptr) {
            --level_;
        }
        
        node->forward.fill(nullptr);
        --size_;
    }
    
    // O(log n) find
    [[nodiscard]] PriceLevel* find(uint64_t price) const noexcept {
        PriceLevel* current = const_cast<PriceLevel*>(&head_);
        
        for (int i = level_; i >= 0; --i) {
            while (current->forward[i]) {
                if (current->forward[i]->price_ticks == price) {
                    return current->forward[i];
                }
                if (!compare_price(current->forward[i]->price_ticks, price)) {
                    break;
                }
                current = current->forward[i];
            }
        }
        
        return nullptr;
    }
    
    // O(1) best price access
    [[nodiscard]] PriceLevel* best() const noexcept {
        return head_.forward[0];
    }
    
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
    [[nodiscard]] size_t size() const noexcept { return size_; }
    
    // Iterator for level 0 traversal
    class Iterator {
    public:
        explicit Iterator(PriceLevel* node) : node_(node) {}
        PriceLevel& operator*() const { return *node_; }
        PriceLevel* operator->() const { return node_; }
        Iterator& operator++() { node_ = node_->forward[0]; return *this; }
        bool operator!=(const Iterator& other) const { return node_ != other.node_; }
    private:
        PriceLevel* node_;
    };
    
    [[nodiscard]] Iterator begin() const { return Iterator(head_.forward[0]); }
    [[nodiscard]] Iterator end() const { return Iterator(nullptr); }
    
    void clear() noexcept {
        head_.forward.fill(nullptr);
        level_ = 0;
        size_ = 0;
    }
    
private:
    // Compare function: for BUY side, higher prices come first (descending)
    //                   for SELL side, lower prices come first (ascending)
    [[nodiscard]] static bool compare(const PriceLevel* a, const PriceLevel* b) noexcept {
        if constexpr (S == Side::BUY) {
            return a->price_ticks > b->price_ticks;  // Descending for bids
        } else {
            return a->price_ticks < b->price_ticks;  // Ascending for asks
        }
    }
    
    [[nodiscard]] static bool compare_price(uint64_t a, uint64_t b) noexcept {
        if constexpr (S == Side::BUY) {
            return a > b;
        } else {
            return a < b;
        }
    }
    
    [[nodiscard]] int random_level() noexcept {
        int lvl = 0;
        while (lvl < SKIPLIST_MAX_LEVEL - 1 && (rng_() & 1)) {
            ++lvl;
        }
        return lvl;
    }
    
    PriceLevel head_;
    int level_;
    size_t size_;
    mutable std::minstd_rand rng_;  // Fast LCG RNG
};

// =============================================================================
// TRADE EXECUTION RECORD
// =============================================================================

struct Trade {
    uint64_t trade_id;
    uint64_t buy_order_id;
    uint64_t sell_order_id;
    uint64_t price_ticks;
    uint32_t quantity;
    Side aggressor_side;
    Timestamp timestamp;
    int64_t latency_ns;  // Order-to-trade latency
};

// =============================================================================
// LATENCY STATISTICS with Online Algorithms
// No allocation during recording, accurate percentiles
// =============================================================================

class LatencyStats {
public:
    void record(int64_t latency_ns) noexcept {
        ++count_;
        sum_ns_ += latency_ns;
        last_ns_ = latency_ns;
        
        if (latency_ns < min_ns_) min_ns_ = latency_ns;
        if (latency_ns > max_ns_) max_ns_ = latency_ns;
        
        // Reservoir sampling for percentiles (fixed memory)
        if (sample_count_ < SAMPLE_SIZE) {
            samples_[sample_count_++] = latency_ns;
        } else {
            // Probabilistic replacement
            size_t j = rng_() % count_;
            if (j < SAMPLE_SIZE) {
                samples_[j] = latency_ns;
            }
        }
        
        // Online variance calculation (Welford's algorithm)
        double delta = latency_ns - mean_;
        mean_ += delta / count_;
        double delta2 = latency_ns - mean_;
        m2_ += delta * delta2;
    }
    
    [[nodiscard]] int64_t last_ns() const noexcept { return last_ns_; }
    [[nodiscard]] int64_t min_ns() const noexcept { return min_ns_; }
    [[nodiscard]] int64_t max_ns() const noexcept { return max_ns_; }
    [[nodiscard]] uint64_t count() const noexcept { return count_; }
    
    [[nodiscard]] double avg_ns() const noexcept { 
        return count_ > 0 ? static_cast<double>(sum_ns_) / count_ : 0.0; 
    }
    
    [[nodiscard]] double stddev_ns() const noexcept {
        return count_ > 1 ? std::sqrt(m2_ / (count_ - 1)) : 0.0;
    }
    
    [[nodiscard]] int64_t percentile(double p) const {
        if (sample_count_ == 0) return 0;
        
        // Sort samples for percentile calculation
        std::array<int64_t, SAMPLE_SIZE> sorted;
        std::copy(samples_.begin(), samples_.begin() + sample_count_, sorted.begin());
        std::sort(sorted.begin(), sorted.begin() + sample_count_);
        
        size_t idx = static_cast<size_t>(sample_count_ * p);
        if (idx >= sample_count_) idx = sample_count_ - 1;
        return sorted[idx];
    }
    
    [[nodiscard]] int64_t p50_ns() const { return percentile(0.50); }
    [[nodiscard]] int64_t p90_ns() const { return percentile(0.90); }
    [[nodiscard]] int64_t p99_ns() const { return percentile(0.99); }
    [[nodiscard]] int64_t p999_ns() const { return percentile(0.999); }
    
    void reset() noexcept {
        count_ = 0;
        sum_ns_ = 0;
        last_ns_ = 0;
        min_ns_ = std::numeric_limits<int64_t>::max();
        max_ns_ = 0;
        mean_ = 0.0;
        m2_ = 0.0;
        sample_count_ = 0;
    }
    
private:
    static constexpr size_t SAMPLE_SIZE = 10000;
    
    uint64_t count_{0};
    int64_t sum_ns_{0};
    int64_t last_ns_{0};
    int64_t min_ns_{std::numeric_limits<int64_t>::max()};
    int64_t max_ns_{0};
    
    // Welford's online variance
    double mean_{0.0};
    double m2_{0.0};
    
    // Reservoir sample for percentiles
    std::array<int64_t, SAMPLE_SIZE> samples_{};
    size_t sample_count_{0};
    mutable std::minstd_rand rng_{42};
};

// =============================================================================
// ORDER ID TO POINTER MAPPING
// Open-addressed hash table with linear probing - cache friendly
// =============================================================================

class OrderMap {
public:
    OrderMap() {
        entries_.fill({0, nullptr});
    }
    
    void insert(uint64_t order_id, Order* order) noexcept {
        size_t idx = hash(order_id);
        size_t start = idx;
        
        do {
            if (entries_[idx].order_id == 0 || entries_[idx].order_id == DELETED) {
                entries_[idx] = {order_id, order};
                ++size_;
                return;
            }
            idx = (idx + 1) & MASK;
        } while (idx != start);
    }
    
    [[nodiscard]] Order* find(uint64_t order_id) const noexcept {
        size_t idx = hash(order_id);
        size_t start = idx;
        
        do {
            if (entries_[idx].order_id == order_id) {
                return entries_[idx].order;
            }
            if (entries_[idx].order_id == 0) {
                return nullptr;
            }
            idx = (idx + 1) & MASK;
        } while (idx != start);
        
        return nullptr;
    }
    
    void remove(uint64_t order_id) noexcept {
        size_t idx = hash(order_id);
        size_t start = idx;
        
        do {
            if (entries_[idx].order_id == order_id) {
                entries_[idx].order_id = DELETED;
                entries_[idx].order = nullptr;
                --size_;
                return;
            }
            if (entries_[idx].order_id == 0) {
                return;
            }
            idx = (idx + 1) & MASK;
        } while (idx != start);
    }
    
    [[nodiscard]] size_t size() const noexcept { return size_; }
    
    void clear() noexcept {
        entries_.fill({0, nullptr});
        size_ = 0;
    }
    
private:
    static constexpr size_t CAPACITY = 1 << 20;  // 1M entries, must be power of 2
    static constexpr size_t MASK = CAPACITY - 1;
    static constexpr uint64_t DELETED = std::numeric_limits<uint64_t>::max();
    
    struct Entry {
        uint64_t order_id;
        Order* order;
    };
    
    [[nodiscard]] static size_t hash(uint64_t key) noexcept {
        // Fast hash function (FNV-1a variant)
        key ^= key >> 33;
        key *= 0xff51afd7ed558ccd;
        key ^= key >> 33;
        key *= 0xc4ceb9fe1a85ec53;
        key ^= key >> 33;
        return key & MASK;
    }
    
    std::array<Entry, CAPACITY> entries_;
    size_t size_{0};
};

// =============================================================================
// LIMIT ORDER BOOK
// Main order book class with all the production-grade features
// =============================================================================

class LimitOrderBook {
public:
    // Callback types for execution reporting
    using TradeCallback = std::function<void(const Trade&)>;
    using OrderUpdateCallback = std::function<void(const Order&)>;
    
    explicit LimitOrderBook(const std::string& symbol, uint32_t tick_size = 1);
    ~LimitOrderBook() = default;
    
    // Non-copyable, movable
    LimitOrderBook(const LimitOrderBook&) = delete;
    LimitOrderBook& operator=(const LimitOrderBook&) = delete;
    LimitOrderBook(LimitOrderBook&&) noexcept = default;
    LimitOrderBook& operator=(LimitOrderBook&&) noexcept = default;
    
    // =========================================================================
    // CORE ORDER MANAGEMENT
    // =========================================================================
    
    /**
     * Add a new order to the book
     * @return order_id (0 if rejected)
     * 
     * Complexity: O(log P) for price level lookup + O(M) for matching
     *             where P = number of price levels, M = number of fills
     */
    [[nodiscard]] uint64_t add_order(
        Side side,
        OrderType type,
        uint64_t price_ticks,
        uint32_t quantity,
        TimeInForce tif = TimeInForce::GTC,
        uint64_t client_order_id = 0,
        std::vector<Trade>* trades_out = nullptr
    );
    
    /**
     * Cancel an existing order
     * @return true if cancelled, false if not found or already filled
     * 
     * Complexity: O(1) - direct pointer access via order map
     */
    bool cancel_order(uint64_t order_id);
    
    /**
     * Modify an existing order (cancel/replace)
     * @return new order_id (0 if failed)
     * 
     * Complexity: O(log P) - cancel is O(1), new order is O(log P)
     */
    [[nodiscard]] uint64_t modify_order(
        uint64_t order_id,
        uint64_t new_price_ticks,
        uint32_t new_quantity,
        std::vector<Trade>* trades_out = nullptr
    );
    
    // =========================================================================
    // MARKET DATA
    // =========================================================================
    
    [[nodiscard]] uint64_t best_bid() const noexcept;
    [[nodiscard]] uint64_t best_ask() const noexcept;
    [[nodiscard]] uint64_t spread() const noexcept;
    [[nodiscard]] uint64_t mid_price() const noexcept;
    
    [[nodiscard]] uint64_t bid_quantity_at(uint64_t price) const noexcept;
    [[nodiscard]] uint64_t ask_quantity_at(uint64_t price) const noexcept;
    
    [[nodiscard]] size_t bid_levels() const noexcept { return bids_.size(); }
    [[nodiscard]] size_t ask_levels() const noexcept { return asks_.size(); }
    
    // Get top N price levels
    struct LevelInfo {
        uint64_t price;
        uint64_t quantity;
        uint32_t order_count;
    };
    
    void get_bids(std::vector<LevelInfo>& out, size_t depth = 10) const;
    void get_asks(std::vector<LevelInfo>& out, size_t depth = 10) const;
    
    // =========================================================================
    // STATISTICS
    // =========================================================================
    
    [[nodiscard]] const LatencyStats& get_latency_stats() const noexcept { 
        return latency_stats_; 
    }
    [[nodiscard]] size_t total_orders() const noexcept { return order_map_.size(); }
    [[nodiscard]] size_t total_trades() const noexcept { return trade_count_; }
    [[nodiscard]] const std::string& symbol() const noexcept { return symbol_; }
    
    // =========================================================================
    // CALLBACKS
    // =========================================================================
    
    void set_trade_callback(TradeCallback cb) { trade_callback_ = std::move(cb); }
    void set_order_callback(OrderUpdateCallback cb) { order_callback_ = std::move(cb); }
    
    // =========================================================================
    // LIFECYCLE
    // =========================================================================
    
    void clear();
    
private:
    std::string symbol_;
    uint32_t tick_size_;
    
    // Order ID generation
    std::atomic<uint64_t> next_order_id_{1};
    std::atomic<uint64_t> next_trade_id_{1};
    
    // Memory pools - ZERO malloc in hot path
    SlabAllocator<Order, MAX_ORDERS> order_allocator_;
    SlabAllocator<PriceLevel, MAX_PRICE_LEVELS> level_allocator_;
    
    // Price level skip lists (sorted by price)
    PriceLevelSkipList<Side::BUY> bids_;   // Descending (best = highest)
    PriceLevelSkipList<Side::SELL> asks_;  // Ascending (best = lowest)
    
    // Order lookup
    OrderMap order_map_;
    
    // Statistics
    LatencyStats latency_stats_;
    uint64_t trade_count_{0};
    
    // Callbacks
    TradeCallback trade_callback_;
    OrderUpdateCallback order_callback_;
    
    // =========================================================================
    // INTERNAL MATCHING ENGINE
    // =========================================================================
    
    void match_order(Order* order, std::vector<Trade>* trades_out);
    
    template<Side AggSide>
    void match_aggressive(Order* order, std::vector<Trade>* trades_out);
    
    void execute_trade(
        Order* aggressive,
        Order* passive,
        uint32_t fill_qty,
        uint64_t price,
        std::vector<Trade>* trades_out
    );
    
    // =========================================================================
    // PRICE LEVEL MANAGEMENT
    // =========================================================================
    
    PriceLevel* get_or_create_level(Side side, uint64_t price);
    void remove_level_if_empty(PriceLevel* level);
    
    // =========================================================================
    // HELPERS
    // =========================================================================
    
    [[nodiscard]] inline int64_t elapsed_ns(const Timestamp& start) const noexcept {
        return std::chrono::duration_cast<Nanoseconds>(
            std::chrono::steady_clock::now() - start
        ).count();
    }
};

// =============================================================================
// SNAPSHOT FOR EXTERNAL CONSUMPTION
// =============================================================================

struct OrderBookSnapshot {
    std::string symbol;
    uint64_t best_bid;
    uint64_t best_ask;
    uint64_t spread;
    uint64_t mid_price;
    std::vector<LimitOrderBook::LevelInfo> bids;
    std::vector<LimitOrderBook::LevelInfo> asks;
    size_t total_orders;
    size_t total_trades;
    LatencyStats stats;
    Timestamp timestamp;
};

[[nodiscard]] OrderBookSnapshot create_snapshot(const LimitOrderBook& book, size_t depth = 10);

} // namespace arbor::orderbook
