#pragma once

/**
 * Lock-Free Queue Implementations for HFT Systems
 * 
 * SPSC (Single Producer Single Consumer) - Zero contention, wait-free
 * MPSC (Multiple Producer Single Consumer) - Lock-free with CAS
 * 
 * Key optimizations:
 * - Cache line padding to prevent false sharing
 * - Memory ordering with acquire/release semantics
 * - Bounded circular buffer to avoid allocations
 */

#include <atomic>
#include <array>
#include <optional>
#include <cstdint>
#include <new>

namespace arbor::lockfree {

// Cache line size for x86-64
constexpr size_t CACHE_LINE_SIZE = 64;

// Padding to prevent false sharing between cache lines
template<typename T>
struct alignas(CACHE_LINE_SIZE) CacheLinePadded {
    T value;
    char padding[CACHE_LINE_SIZE - sizeof(T) % CACHE_LINE_SIZE];
};

/**
 * Wait-Free SPSC Queue (Lamport Queue variant)
 * 
 * Guarantees:
 * - Wait-free for both producer and consumer
 * - No locks, no CAS (only relaxed/acquire/release atomics)
 * - Bounded memory, zero allocations in hot path
 * - Cache-optimized with separated head/tail cache lines
 * 
 * Latency: ~5-15ns per operation
 */
template<typename T, size_t Capacity>
class SPSCQueue {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");
    static_assert(Capacity >= 2, "Capacity must be at least 2");

public:
    SPSCQueue() : head_{0}, tail_{0} {
        for (size_t i = 0; i < Capacity; ++i) {
            buffer_[i].sequence.store(i, std::memory_order_relaxed);
        }
    }

    // Non-copyable, non-movable (atomics)
    SPSCQueue(const SPSCQueue&) = delete;
    SPSCQueue& operator=(const SPSCQueue&) = delete;

    /**
     * Push element to queue (producer only)
     * @return true if successful, false if queue is full
     * 
     * Wait-free: Always completes in bounded time
     */
    template<typename U>
    bool push(U&& item) noexcept {
        const size_t head = head_.load(std::memory_order_relaxed);
        const size_t next_head = (head + 1) & mask_;
        
        // Check if queue is full
        if (next_head == tail_.load(std::memory_order_acquire)) {
            return false;  // Queue full
        }
        
        // Store item
        buffer_[head].data = std::forward<U>(item);
        
        // Publish to consumer
        head_.store(next_head, std::memory_order_release);
        return true;
    }

    /**
     * Pop element from queue (consumer only)
     * @return element if available, nullopt if queue is empty
     * 
     * Wait-free: Always completes in bounded time
     */
    std::optional<T> pop() noexcept {
        const size_t tail = tail_.load(std::memory_order_relaxed);
        
        // Check if queue is empty
        if (tail == head_.load(std::memory_order_acquire)) {
            return std::nullopt;  // Queue empty
        }
        
        // Read item
        T item = std::move(buffer_[tail].data);
        
        // Advance tail
        tail_.store((tail + 1) & mask_, std::memory_order_release);
        return item;
    }

    /**
     * Try to pop without removing (peek)
     */
    const T* front() const noexcept {
        const size_t tail = tail_.load(std::memory_order_relaxed);
        if (tail == head_.load(std::memory_order_acquire)) {
            return nullptr;
        }
        return &buffer_[tail].data;
    }

    bool empty() const noexcept {
        return tail_.load(std::memory_order_relaxed) == 
               head_.load(std::memory_order_relaxed);
    }

    size_t size() const noexcept {
        const size_t head = head_.load(std::memory_order_relaxed);
        const size_t tail = tail_.load(std::memory_order_relaxed);
        return (head - tail) & mask_;
    }

    static constexpr size_t capacity() noexcept { return Capacity; }

private:
    struct Slot {
        T data;
        std::atomic<size_t> sequence;
    };

    static constexpr size_t mask_ = Capacity - 1;

    // Separate cache lines for producer and consumer
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> head_;
    char pad1_[CACHE_LINE_SIZE - sizeof(std::atomic<size_t>)];
    
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> tail_;
    char pad2_[CACHE_LINE_SIZE - sizeof(std::atomic<size_t>)];
    
    alignas(CACHE_LINE_SIZE) std::array<Slot, Capacity> buffer_;
};


/**
 * Lock-Free MPSC Queue (Multiple Producer, Single Consumer)
 * 
 * Uses CAS for thread-safe multi-producer access
 * Single consumer remains wait-free
 * 
 * Guarantees:
 * - Lock-free (not wait-free) for producers
 * - Wait-free for consumer
 * - No ABA problem (uses sequence counters)
 * 
 * Latency: ~15-50ns per operation (depends on contention)
 */
template<typename T, size_t Capacity>
class MPSCQueue {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");

public:
    MPSCQueue() : head_{0}, tail_{0} {
        for (size_t i = 0; i < Capacity; ++i) {
            buffer_[i].sequence.store(i, std::memory_order_relaxed);
        }
    }

    MPSCQueue(const MPSCQueue&) = delete;
    MPSCQueue& operator=(const MPSCQueue&) = delete;

    /**
     * Push element (multiple producers can call concurrently)
     * @return true if successful, false if queue is full
     * 
     * Lock-free: May retry on contention, but progress is guaranteed system-wide
     */
    template<typename U>
    bool push(U&& item) noexcept {
        size_t head = head_.load(std::memory_order_relaxed);
        
        while (true) {
            auto& slot = buffer_[head & mask_];
            const size_t seq = slot.sequence.load(std::memory_order_acquire);
            const intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(head);
            
            if (diff == 0) {
                // Slot is available, try to claim it
                if (head_.compare_exchange_weak(head, head + 1,
                    std::memory_order_relaxed, std::memory_order_relaxed)) {
                    // Successfully claimed slot
                    slot.data = std::forward<U>(item);
                    slot.sequence.store(head + 1, std::memory_order_release);
                    return true;
                }
                // CAS failed, another producer won, retry with updated head
            } else if (diff < 0) {
                // Queue is full
                return false;
            } else {
                // Slot not yet consumed, reload head
                head = head_.load(std::memory_order_relaxed);
            }
        }
    }

    /**
     * Pop element (single consumer only!)
     * @return element if available, nullopt if empty
     * 
     * Wait-free: Always completes in bounded time
     */
    std::optional<T> pop() noexcept {
        auto& slot = buffer_[tail_ & mask_];
        const size_t seq = slot.sequence.load(std::memory_order_acquire);
        const intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(tail_ + 1);
        
        if (diff < 0) {
            return std::nullopt;  // Empty
        }
        
        T item = std::move(slot.data);
        slot.sequence.store(tail_ + Capacity, std::memory_order_release);
        ++tail_;
        return item;
    }

    bool empty() const noexcept {
        const auto& slot = buffer_[tail_ & mask_];
        const size_t seq = slot.sequence.load(std::memory_order_acquire);
        return static_cast<intptr_t>(seq) - static_cast<intptr_t>(tail_ + 1) < 0;
    }

    size_t size_approx() const noexcept {
        return head_.load(std::memory_order_relaxed) - tail_;
    }

private:
    struct Slot {
        T data;
        std::atomic<size_t> sequence;
    };

    static constexpr size_t mask_ = Capacity - 1;

    alignas(CACHE_LINE_SIZE) std::atomic<size_t> head_;
    char pad1_[CACHE_LINE_SIZE - sizeof(std::atomic<size_t>)];
    
    alignas(CACHE_LINE_SIZE) size_t tail_;  // Only accessed by consumer
    char pad2_[CACHE_LINE_SIZE - sizeof(size_t)];
    
    alignas(CACHE_LINE_SIZE) std::array<Slot, Capacity> buffer_;
};


/**
 * Lock-Free MPMC Queue (Multiple Producer, Multiple Consumer)
 * 
 * Full concurrent access from any thread
 * Uses sequence counters to avoid ABA problem
 * 
 * Latency: ~30-100ns (highest contention)
 */
template<typename T, size_t Capacity>
class MPMCQueue {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");

public:
    MPMCQueue() : head_{0}, tail_{0} {
        for (size_t i = 0; i < Capacity; ++i) {
            buffer_[i].sequence.store(i, std::memory_order_relaxed);
        }
    }

    MPMCQueue(const MPMCQueue&) = delete;
    MPMCQueue& operator=(const MPMCQueue&) = delete;

    template<typename U>
    bool push(U&& item) noexcept {
        size_t head = head_.load(std::memory_order_relaxed);
        
        while (true) {
            auto& slot = buffer_[head & mask_];
            const size_t seq = slot.sequence.load(std::memory_order_acquire);
            const intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(head);
            
            if (diff == 0) {
                if (head_.compare_exchange_weak(head, head + 1,
                    std::memory_order_relaxed, std::memory_order_relaxed)) {
                    slot.data = std::forward<U>(item);
                    slot.sequence.store(head + 1, std::memory_order_release);
                    return true;
                }
            } else if (diff < 0) {
                return false;
            } else {
                head = head_.load(std::memory_order_relaxed);
            }
        }
    }

    std::optional<T> pop() noexcept {
        size_t tail = tail_.load(std::memory_order_relaxed);
        
        while (true) {
            auto& slot = buffer_[tail & mask_];
            const size_t seq = slot.sequence.load(std::memory_order_acquire);
            const intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(tail + 1);
            
            if (diff == 0) {
                if (tail_.compare_exchange_weak(tail, tail + 1,
                    std::memory_order_relaxed, std::memory_order_relaxed)) {
                    T item = std::move(slot.data);
                    slot.sequence.store(tail + Capacity, std::memory_order_release);
                    return item;
                }
            } else if (diff < 0) {
                return std::nullopt;
            } else {
                tail = tail_.load(std::memory_order_relaxed);
            }
        }
    }

    bool empty() const noexcept {
        size_t tail = tail_.load(std::memory_order_relaxed);
        const auto& slot = buffer_[tail & mask_];
        const size_t seq = slot.sequence.load(std::memory_order_acquire);
        return static_cast<intptr_t>(seq) - static_cast<intptr_t>(tail + 1) < 0;
    }

private:
    struct Slot {
        T data;
        std::atomic<size_t> sequence;
    };

    static constexpr size_t mask_ = Capacity - 1;

    alignas(CACHE_LINE_SIZE) std::atomic<size_t> head_;
    char pad1_[CACHE_LINE_SIZE - sizeof(std::atomic<size_t>)];
    
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> tail_;
    char pad2_[CACHE_LINE_SIZE - sizeof(std::atomic<size_t>)];
    
    alignas(CACHE_LINE_SIZE) std::array<Slot, Capacity> buffer_;
};

} // namespace arbor::lockfree
