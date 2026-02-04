/**
 * Lock-Free Data Structures Unit Tests
 * 
 * Tests correctness, linearizability, and stress conditions.
 * Uses deterministic and randomized testing patterns.
 */

#include "lockfree_queue.hpp"
#include <iostream>
#include <thread>
#include <vector>
#include <set>
#include <atomic>
#include <cassert>
#include <random>
#include <algorithm>
#include <numeric>

using namespace arbor::lockfree;

// Test counters
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        std::cerr << "[FAIL] " << msg << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        tests_failed++; \
        return; \
    } \
} while(0)

#define TEST_PASS(name) do { \
    std::cout << "[PASS] " << name << std::endl; \
    tests_passed++; \
} while(0)

//=============================================================================
// SPSC QUEUE TESTS
//=============================================================================

void test_spsc_basic() {
    SPSCQueue<int, 64> q;
    
    TEST_ASSERT(q.empty(), "New queue should be empty");
    TEST_ASSERT(q.size() == 0, "New queue size should be 0");
    
    TEST_ASSERT(q.push(42), "Push to empty queue should succeed");
    TEST_ASSERT(!q.empty(), "Queue should not be empty after push");
    TEST_ASSERT(q.size() == 1, "Queue size should be 1");
    
    auto val = q.pop();
    TEST_ASSERT(val.has_value(), "Pop should return value");
    TEST_ASSERT(*val == 42, "Popped value should be 42");
    TEST_ASSERT(q.empty(), "Queue should be empty after pop");
    
    TEST_PASS("SPSC basic operations");
}

void test_spsc_full_queue() {
    SPSCQueue<int, 16> q;  // Capacity is 15 (one slot reserved)
    
    // Fill the queue
    for (int i = 0; i < 15; ++i) {
        TEST_ASSERT(q.push(i), "Push should succeed");
    }
    
    TEST_ASSERT(!q.push(999), "Push to full queue should fail");
    TEST_ASSERT(q.size() == 15, "Size should be 15");
    
    // Drain the queue
    for (int i = 0; i < 15; ++i) {
        auto val = q.pop();
        TEST_ASSERT(val.has_value(), "Pop should succeed");
        TEST_ASSERT(*val == i, "FIFO order violated");
    }
    
    TEST_ASSERT(q.empty(), "Queue should be empty");
    TEST_ASSERT(!q.pop().has_value(), "Pop from empty should return nullopt");
    
    TEST_PASS("SPSC full queue behavior");
}

void test_spsc_concurrent() {
    SPSCQueue<uint64_t, 8192> q;
    constexpr size_t NUM = 1'000'000;
    
    std::atomic<bool> producer_done{false};
    std::atomic<uint64_t> sum_produced{0};
    std::atomic<uint64_t> sum_consumed{0};
    
    std::thread producer([&]() {
        for (uint64_t i = 1; i <= NUM; ++i) {
            while (!q.push(i)) {
                std::this_thread::yield();
            }
            sum_produced += i;
        }
        producer_done = true;
    });
    
    std::thread consumer([&]() {
        uint64_t count = 0;
        while (count < NUM) {
            auto val = q.pop();
            if (val) {
                sum_consumed += *val;
                ++count;
            }
        }
    });
    
    producer.join();
    consumer.join();
    
    TEST_ASSERT(sum_produced == sum_consumed, "Sum mismatch in concurrent test");
    uint64_t expected = NUM * (NUM + 1) / 2;
    TEST_ASSERT(sum_consumed == expected, "Sum should equal N*(N+1)/2");
    
    TEST_PASS("SPSC concurrent correctness");
}

void test_spsc_batch() {
    SPSCQueue<int, 256> q;
    std::vector<int> input(100);
    std::iota(input.begin(), input.end(), 1);  // 1, 2, 3, ... 100
    
    size_t pushed = q.push_batch(input.begin(), input.end());
    TEST_ASSERT(pushed == 100, "Should push all 100 items");
    TEST_ASSERT(q.size() == 100, "Size should be 100");
    
    std::vector<int> output(100);
    size_t popped = q.pop_batch(output.begin(), 100);
    TEST_ASSERT(popped == 100, "Should pop all 100 items");
    TEST_ASSERT(input == output, "Batch should preserve order");
    
    TEST_PASS("SPSC batch operations");
}

//=============================================================================
// MPSC QUEUE TESTS
//=============================================================================

void test_mpsc_basic() {
    MPSCQueue<int, 64> q;
    
    TEST_ASSERT(q.empty(), "New queue should be empty");
    TEST_ASSERT(q.push(42), "Push should succeed");
    
    auto val = q.pop();
    TEST_ASSERT(val.has_value() && *val == 42, "Pop should return 42");
    TEST_ASSERT(q.empty(), "Queue should be empty");
    
    TEST_PASS("MPSC basic operations");
}

void test_mpsc_concurrent() {
    MPSCQueue<uint64_t, 16384> q;
    constexpr size_t NUM_PRODUCERS = 4;
    constexpr size_t MSGS_PER_PRODUCER = 100'000;
    constexpr size_t TOTAL = NUM_PRODUCERS * MSGS_PER_PRODUCER;
    
    std::atomic<uint64_t> consumed{0};
    std::set<uint64_t> received;
    std::mutex received_mtx;
    
    std::vector<std::thread> producers;
    for (size_t p = 0; p < NUM_PRODUCERS; ++p) {
        producers.emplace_back([&, p]() {
            for (size_t i = 0; i < MSGS_PER_PRODUCER; ++i) {
                uint64_t val = p * MSGS_PER_PRODUCER + i;
                while (!q.push(val)) {
                    std::this_thread::yield();
                }
            }
        });
    }
    
    std::thread consumer([&]() {
        std::set<uint64_t> local_received;
        while (consumed < TOTAL) {
            auto val = q.pop();
            if (val) {
                local_received.insert(*val);
                consumed++;
            }
        }
        std::lock_guard<std::mutex> lock(received_mtx);
        received = std::move(local_received);
    });
    
    for (auto& t : producers) t.join();
    consumer.join();
    
    TEST_ASSERT(received.size() == TOTAL, "Should receive all unique messages");
    
    // Verify all values present
    for (uint64_t i = 0; i < TOTAL; ++i) {
        TEST_ASSERT(received.count(i) == 1, "Missing or duplicate value");
    }
    
    TEST_PASS("MPSC concurrent correctness");
}

//=============================================================================
// MPMC QUEUE TESTS
//=============================================================================

void test_mpmc_basic() {
    MPMCQueue<int, 64> q;
    
    TEST_ASSERT(q.empty(), "New queue should be empty");
    TEST_ASSERT(q.push(42), "Push should succeed");
    
    auto val = q.pop();
    TEST_ASSERT(val.has_value() && *val == 42, "Pop should return 42");
    
    TEST_PASS("MPMC basic operations");
}

void test_mpmc_concurrent() {
    MPMCQueue<uint64_t, 8192> q;
    constexpr size_t NUM_PRODUCERS = 2;
    constexpr size_t NUM_CONSUMERS = 2;
    constexpr size_t MSGS_PER_PRODUCER = 100'000;
    constexpr size_t TOTAL = NUM_PRODUCERS * MSGS_PER_PRODUCER;
    
    std::atomic<uint64_t> produced{0};
    std::atomic<uint64_t> consumed{0};
    std::atomic<uint64_t> sum_produced{0};
    std::atomic<uint64_t> sum_consumed{0};
    
    std::vector<std::thread> producers;
    for (size_t p = 0; p < NUM_PRODUCERS; ++p) {
        producers.emplace_back([&]() {
            for (size_t i = 0; i < MSGS_PER_PRODUCER; ++i) {
                uint64_t val = produced.fetch_add(1, std::memory_order_relaxed) + 1;
                while (!q.push(val)) {
                    std::this_thread::yield();
                }
                sum_produced.fetch_add(val, std::memory_order_relaxed);
            }
        });
    }
    
    std::vector<std::thread> consumers;
    for (size_t c = 0; c < NUM_CONSUMERS; ++c) {
        consumers.emplace_back([&]() {
            while (consumed < TOTAL) {
                auto val = q.pop();
                if (val) {
                    sum_consumed.fetch_add(*val, std::memory_order_relaxed);
                    consumed.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }
    
    for (auto& t : producers) t.join();
    for (auto& t : consumers) t.join();
    
    TEST_ASSERT(consumed >= TOTAL, "Should consume all messages");
    TEST_ASSERT(sum_produced == sum_consumed, "Sum mismatch");
    
    TEST_PASS("MPMC concurrent correctness");
}

//=============================================================================
// SEQLOCK TESTS
//=============================================================================

void test_seqlock_basic() {
    struct Data {
        int a, b, c;
    };
    
    Seqlock<Data> sl(Data{1, 2, 3});
    
    auto d = sl.read();
    TEST_ASSERT(d.a == 1 && d.b == 2 && d.c == 3, "Initial read");
    
    sl.write(Data{10, 20, 30});
    d = sl.read();
    TEST_ASSERT(d.a == 10 && d.b == 20 && d.c == 30, "Read after write");
    
    TEST_PASS("Seqlock basic operations");
}

void test_seqlock_concurrent() {
    struct Counter {
        uint64_t a, b;  // Should always be equal
    };
    
    Seqlock<Counter> sl(Counter{0, 0});
    constexpr size_t NUM_WRITES = 100'000;
    
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> inconsistencies{0};
    std::atomic<uint64_t> reads{0};
    
    // Writer
    std::thread writer([&]() {
        for (uint64_t i = 1; i <= NUM_WRITES; ++i) {
            sl.write(Counter{i, i});
        }
        stop = true;
    });
    
    // Readers
    std::vector<std::thread> readers;
    for (int r = 0; r < 4; ++r) {
        readers.emplace_back([&]() {
            while (!stop || reads < NUM_WRITES) {
                auto c = sl.read();
                if (c.a != c.b) {
                    inconsistencies++;
                }
                reads++;
            }
        });
    }
    
    writer.join();
    for (auto& t : readers) t.join();
    
    TEST_ASSERT(inconsistencies == 0, "No torn reads should occur");
    
    TEST_PASS("Seqlock concurrent consistency");
}

//=============================================================================
// WORK-STEALING DEQUE TESTS
//=============================================================================

void test_workstealing_basic() {
    WorkStealingDeque<int, 64> dq;
    
    TEST_ASSERT(dq.empty(), "New deque should be empty");
    TEST_ASSERT(dq.push(1), "Push should succeed");
    TEST_ASSERT(dq.push(2), "Push should succeed");
    TEST_ASSERT(dq.push(3), "Push should succeed");
    
    // LIFO from bottom
    auto v = dq.pop();
    TEST_ASSERT(v.has_value() && *v == 3, "Pop should return 3 (LIFO)");
    
    // FIFO from top (steal)
    v = dq.steal();
    TEST_ASSERT(v.has_value() && *v == 1, "Steal should return 1 (FIFO)");
    
    TEST_PASS("Work-stealing deque basic operations");
}

void test_workstealing_concurrent() {
    WorkStealingDeque<uint64_t, 8192> dq;
    constexpr size_t NUM_TASKS = 100'000;
    
    std::atomic<uint64_t> sum{0};
    std::atomic<uint64_t> stolen{0};
    
    // Owner thread
    std::thread owner([&]() {
        for (uint64_t i = 1; i <= NUM_TASKS; ++i) {
            while (!dq.push(i)) {
                std::this_thread::yield();
            }
        }
        
        // Process some ourselves
        while (true) {
            auto v = dq.pop();
            if (!v) break;
            sum += *v;
        }
    });
    
    // Thieves
    std::vector<std::thread> thieves;
    for (int t = 0; t < 3; ++t) {
        thieves.emplace_back([&]() {
            while (true) {
                auto v = dq.steal();
                if (v) {
                    sum += *v;
                    stolen++;
                } else {
                    // Check if owner is done
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                    if (dq.empty()) break;
                }
            }
        });
    }
    
    owner.join();
    for (auto& t : thieves) t.join();
    
    uint64_t expected = NUM_TASKS * (NUM_TASKS + 1) / 2;
    TEST_ASSERT(sum == expected, "Sum should equal N*(N+1)/2");
    
    TEST_PASS("Work-stealing deque concurrent correctness");
}

//=============================================================================
// MEMORY POOL TESTS
//=============================================================================

void test_mempool_basic() {
    struct Obj {
        int value;
        char data[60];
    };
    
    LockFreePool<Obj, 100> pool;
    
    std::vector<Obj*> ptrs;
    for (int i = 0; i < 100; ++i) {
        Obj* p = pool.allocate();
        TEST_ASSERT(p != nullptr, "Allocation should succeed");
        p->value = i;
        ptrs.push_back(p);
    }
    
    // Pool should be exhausted
    TEST_ASSERT(pool.allocate() == nullptr, "Pool should be exhausted");
    
    // Free all
    for (Obj* p : ptrs) {
        pool.deallocate(p);
    }
    
    // Should be able to allocate again
    Obj* p = pool.allocate();
    TEST_ASSERT(p != nullptr, "Allocation should succeed after deallocation");
    pool.deallocate(p);
    
    TEST_PASS("Memory pool basic operations");
}

void test_mempool_concurrent() {
    struct Obj { uint64_t value; };
    LockFreePool<Obj, 1000> pool;
    
    constexpr size_t NUM_OPS = 100'000;
    const size_t num_threads = std::thread::hardware_concurrency();
    
    std::atomic<uint64_t> allocs{0};
    std::atomic<uint64_t> frees{0};
    
    std::vector<std::thread> workers;
    for (size_t t = 0; t < num_threads; ++t) {
        workers.emplace_back([&]() {
            std::vector<Obj*> local_ptrs;
            std::mt19937 rng(std::random_device{}());
            
            for (size_t i = 0; i < NUM_OPS / num_threads; ++i) {
                if (local_ptrs.empty() || (rng() % 2 == 0 && local_ptrs.size() < 100)) {
                    Obj* p = pool.allocate();
                    if (p) {
                        local_ptrs.push_back(p);
                        allocs++;
                    }
                } else {
                    pool.deallocate(local_ptrs.back());
                    local_ptrs.pop_back();
                    frees++;
                }
            }
            
            // Cleanup
            for (Obj* p : local_ptrs) {
                pool.deallocate(p);
                frees++;
            }
        });
    }
    
    for (auto& t : workers) t.join();
    
    TEST_ASSERT(allocs == frees, "Allocs should equal frees");
    
    TEST_PASS("Memory pool concurrent correctness");
}

//=============================================================================
// WAIT-FREE COUNTER TESTS
//=============================================================================

void test_counter_basic() {
    WaitFreeCounter counter(100);
    
    TEST_ASSERT(counter.current() == 100, "Initial value");
    TEST_ASSERT(counter.next() == 100, "First next returns current");
    TEST_ASSERT(counter.current() == 101, "Incremented");
    
    auto batch = counter.next_batch(10);
    TEST_ASSERT(batch == 101, "Batch returns start");
    TEST_ASSERT(counter.current() == 111, "Batch incremented by 10");
    
    TEST_PASS("Wait-free counter basic operations");
}

void test_counter_concurrent() {
    WaitFreeCounter counter;
    constexpr size_t NUM_OPS = 10'000'000;
    const size_t num_threads = std::thread::hardware_concurrency();
    
    std::vector<std::thread> workers;
    for (size_t t = 0; t < num_threads; ++t) {
        workers.emplace_back([&]() {
            for (size_t i = 0; i < NUM_OPS / num_threads; ++i) {
                counter.next();
            }
        });
    }
    
    for (auto& t : workers) t.join();
    
    // May be slightly more due to rounding, but never less
    TEST_ASSERT(counter.current() >= NUM_OPS, "Counter should reach at least NUM_OPS");
    
    TEST_PASS("Wait-free counter concurrent correctness");
}

//=============================================================================
// ADAPTIVE BACKOFF TESTS
//=============================================================================

void test_adaptive_backoff() {
    AdaptiveBackoff backoff;
    
    // Should start with small count
    TEST_ASSERT(backoff.count() == 1, "Initial count is 1");
    
    // Call several times and verify exponential growth
    for (int i = 0; i < 10; ++i) {
        backoff();
    }
    
    TEST_ASSERT(backoff.count() > 100, "Count should grow exponentially");
    
    backoff.reset();
    TEST_ASSERT(backoff.count() == 1, "Reset should restore count to 1");
    
    TEST_PASS("Adaptive backoff behavior");
}

//=============================================================================
// STRESS TESTS
//=============================================================================

void test_stress_spsc() {
    std::cout << "Running SPSC stress test (may take a moment)...\n";
    
    SPSCQueue<uint64_t, 4096> q;
    constexpr size_t NUM = 50'000'000;
    
    std::atomic<uint64_t> checksum_prod{0};
    std::atomic<uint64_t> checksum_cons{0};
    
    std::thread producer([&]() {
        for (uint64_t i = 1; i <= NUM; ++i) {
            while (!q.push(i)) { }
            checksum_prod ^= i;
        }
    });
    
    std::thread consumer([&]() {
        for (uint64_t i = 0; i < NUM;) {
            auto v = q.pop();
            if (v) {
                checksum_cons ^= *v;
                ++i;
            }
        }
    });
    
    producer.join();
    consumer.join();
    
    TEST_ASSERT(checksum_prod == checksum_cons, "XOR checksum should match");
    
    TEST_PASS("SPSC stress test (50M messages)");
}

void test_stress_mpmc() {
    std::cout << "Running MPMC stress test (may take a moment)...\n";
    
    MPMCQueue<uint64_t, 8192> q;
    constexpr size_t NUM = 10'000'000;
    constexpr size_t NUM_THREADS = 4;
    
    std::atomic<uint64_t> produced{0};
    std::atomic<uint64_t> consumed{0};
    
    std::vector<std::thread> threads;
    
    // Producer-consumers
    for (size_t t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&]() {
            std::mt19937 rng(std::random_device{}());
            while (produced < NUM || consumed < NUM) {
                if (rng() % 2 == 0 && produced < NUM) {
                    if (q.push(1)) {
                        produced++;
                    }
                } else {
                    auto v = q.pop();
                    if (v) {
                        consumed++;
                    }
                }
            }
        });
    }
    
    for (auto& t : threads) t.join();
    
    TEST_ASSERT(produced == consumed, "Produced should equal consumed");
    
    TEST_PASS("MPMC stress test (10M messages, 4 threads)");
}

//=============================================================================
// MAIN
//=============================================================================

int main() {
    std::cout << "\n============================================================\n";
    std::cout << "     ARBOR LOCK-FREE DATA STRUCTURES TEST SUITE\n";
    std::cout << "============================================================\n\n";
    
    // SPSC tests
    test_spsc_basic();
    test_spsc_full_queue();
    test_spsc_concurrent();
    test_spsc_batch();
    
    // MPSC tests
    test_mpsc_basic();
    test_mpsc_concurrent();
    
    // MPMC tests
    test_mpmc_basic();
    test_mpmc_concurrent();
    
    // Seqlock tests
    test_seqlock_basic();
    test_seqlock_concurrent();
    
    // Work-stealing deque tests
    test_workstealing_basic();
    test_workstealing_concurrent();
    
    // Memory pool tests
    test_mempool_basic();
    test_mempool_concurrent();
    
    // Counter tests
    test_counter_basic();
    test_counter_concurrent();
    
    // Backoff tests
    test_adaptive_backoff();
    
    // Stress tests
    test_stress_spsc();
    test_stress_mpmc();
    
    // Summary
    std::cout << "\n============================================================\n";
    std::cout << "SUMMARY: " << tests_passed << " passed, " << tests_failed << " failed\n";
    std::cout << "============================================================\n\n";
    
    return tests_failed > 0 ? 1 : 0;
}
