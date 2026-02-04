#pragma once

/**
 * Linux Performance Counters Interface
 * 
 * Direct integration with perf_event_open() for hardware counter access.
 * Measures: cycles, instructions, cache misses, branch mispredictions.
 * 
 * This is what quant firms actually care about - not just wall-clock time.
 */

#include <cstdint>
#include <array>
#include <string>
#include <vector>
#include <cstring>
#include <iostream>
#include <iomanip>

#if defined(__linux__)
#include <linux/perf_event.h>
#include <linux/hw_breakpoint.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <unistd.h>
#define ARBOR_HAS_PERF_COUNTERS 1
#else
#define ARBOR_HAS_PERF_COUNTERS 0
#endif

namespace arbor::perf {

// =============================================================================
// HARDWARE COUNTER TYPES
// =============================================================================

enum class CounterType : uint32_t {
    CYCLES,                    // CPU cycles
    INSTRUCTIONS,              // Instructions retired
    CACHE_REFERENCES,          // L3 cache accesses
    CACHE_MISSES,              // L3 cache misses
    BRANCH_INSTRUCTIONS,       // Branch instructions
    BRANCH_MISSES,             // Branch mispredictions
    L1D_READ_ACCESS,           // L1 data cache reads
    L1D_READ_MISS,             // L1 data cache read misses
    L1D_WRITE_ACCESS,          // L1 data cache writes
    L1I_READ_MISS,             // L1 instruction cache misses
    LLC_READ_MISS,             // Last-level cache read misses
    LLC_WRITE_MISS,            // Last-level cache write misses
    DTLB_READ_MISS,            // Data TLB misses
    ITLB_READ_MISS,            // Instruction TLB misses
    NUM_COUNTERS
};

inline const char* counter_name(CounterType type) {
    switch (type) {
        case CounterType::CYCLES: return "cycles";
        case CounterType::INSTRUCTIONS: return "instructions";
        case CounterType::CACHE_REFERENCES: return "cache-references";
        case CounterType::CACHE_MISSES: return "cache-misses";
        case CounterType::BRANCH_INSTRUCTIONS: return "branches";
        case CounterType::BRANCH_MISSES: return "branch-misses";
        case CounterType::L1D_READ_ACCESS: return "L1-dcache-loads";
        case CounterType::L1D_READ_MISS: return "L1-dcache-load-misses";
        case CounterType::L1D_WRITE_ACCESS: return "L1-dcache-stores";
        case CounterType::L1I_READ_MISS: return "L1-icache-load-misses";
        case CounterType::LLC_READ_MISS: return "LLC-load-misses";
        case CounterType::LLC_WRITE_MISS: return "LLC-store-misses";
        case CounterType::DTLB_READ_MISS: return "dTLB-load-misses";
        case CounterType::ITLB_READ_MISS: return "iTLB-load-misses";
        default: return "unknown";
    }
}

// =============================================================================
// COUNTER RESULTS
// =============================================================================

struct CounterResult {
    uint64_t cycles = 0;
    uint64_t instructions = 0;
    uint64_t cache_references = 0;
    uint64_t cache_misses = 0;
    uint64_t branch_instructions = 0;
    uint64_t branch_misses = 0;
    uint64_t l1d_read_misses = 0;
    uint64_t l1d_write_misses = 0;
    uint64_t llc_misses = 0;
    int64_t time_ns = 0;
    
    // Derived metrics
    [[nodiscard]] double ipc() const {
        return cycles > 0 ? static_cast<double>(instructions) / cycles : 0.0;
    }
    
    [[nodiscard]] double cache_miss_rate() const {
        return cache_references > 0 ? 
            static_cast<double>(cache_misses) / cache_references * 100.0 : 0.0;
    }
    
    [[nodiscard]] double branch_miss_rate() const {
        return branch_instructions > 0 ? 
            static_cast<double>(branch_misses) / branch_instructions * 100.0 : 0.0;
    }
    
    [[nodiscard]] double cycles_per_op(uint64_t num_ops) const {
        return num_ops > 0 ? static_cast<double>(cycles) / num_ops : 0.0;
    }
    
    [[nodiscard]] double ns_per_op(uint64_t num_ops) const {
        return num_ops > 0 ? static_cast<double>(time_ns) / num_ops : 0.0;
    }
    
    void print(const std::string& benchmark_name, uint64_t num_ops = 0) const {
        std::cout << "\n" << benchmark_name << " - Hardware Counter Results:\n";
        std::cout << "-------------------------------------------\n";
        std::cout << std::fixed << std::setprecision(2);
        
        std::cout << "  Time:              " << (time_ns / 1e6) << " ms\n";
        std::cout << "  Cycles:            " << cycles << "\n";
        std::cout << "  Instructions:      " << instructions << "\n";
        std::cout << "  IPC:               " << ipc() << "\n";
        std::cout << "\n";
        std::cout << "  Cache refs:        " << cache_references << "\n";
        std::cout << "  Cache misses:      " << cache_misses << "\n";
        std::cout << "  Cache miss rate:   " << cache_miss_rate() << "%\n";
        std::cout << "\n";
        std::cout << "  Branch instrs:     " << branch_instructions << "\n";
        std::cout << "  Branch misses:     " << branch_misses << "\n";
        std::cout << "  Branch miss rate:  " << branch_miss_rate() << "%\n";
        
        if (num_ops > 0) {
            std::cout << "\n";
            std::cout << "  Per-operation:\n";
            std::cout << "    Cycles/op:       " << cycles_per_op(num_ops) << "\n";
            std::cout << "    ns/op:           " << ns_per_op(num_ops) << "\n";
            std::cout << "    Throughput:      " << std::setprecision(0) 
                      << (num_ops * 1e9 / time_ns) << " ops/sec\n";
        }
    }
};

// =============================================================================
// PERF COUNTER GROUP (Linux perf_event interface)
// =============================================================================

#if ARBOR_HAS_PERF_COUNTERS

class PerfCounterGroup {
public:
    PerfCounterGroup() {
        // Open counter group - leader first
        leader_fd_ = open_counter(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, -1);
        
        if (leader_fd_ < 0) {
            available_ = false;
            return;
        }
        
        // Open grouped counters
        fds_[0] = leader_fd_;
        fds_[1] = open_counter(PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, leader_fd_);
        fds_[2] = open_counter(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_REFERENCES, leader_fd_);
        fds_[3] = open_counter(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES, leader_fd_);
        fds_[4] = open_counter(PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_INSTRUCTIONS, leader_fd_);
        fds_[5] = open_counter(PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES, leader_fd_);
        
        available_ = true;
    }
    
    ~PerfCounterGroup() {
        for (int fd : fds_) {
            if (fd >= 0) close(fd);
        }
    }
    
    // Non-copyable
    PerfCounterGroup(const PerfCounterGroup&) = delete;
    PerfCounterGroup& operator=(const PerfCounterGroup&) = delete;
    
    [[nodiscard]] bool available() const { return available_; }
    
    void start() {
        if (!available_) return;
        
        // Reset and enable all counters atomically
        ioctl(leader_fd_, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
        ioctl(leader_fd_, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
        
        start_time_ = read_tsc();
    }
    
    [[nodiscard]] CounterResult stop() {
        CounterResult result;
        
        if (!available_) {
            // Fall back to wall-clock time only
            result.time_ns = (read_tsc() - start_time_) * tsc_ns_ratio_;
            return result;
        }
        
        // Disable all counters atomically
        ioctl(leader_fd_, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
        
        uint64_t end_time = read_tsc();
        result.time_ns = static_cast<int64_t>((end_time - start_time_) * tsc_ns_ratio_);
        
        // Read counter values
        result.cycles = read_counter(fds_[0]);
        result.instructions = read_counter(fds_[1]);
        result.cache_references = read_counter(fds_[2]);
        result.cache_misses = read_counter(fds_[3]);
        result.branch_instructions = read_counter(fds_[4]);
        result.branch_misses = read_counter(fds_[5]);
        
        return result;
    }
    
private:
    static constexpr int NUM_COUNTERS = 6;
    std::array<int, NUM_COUNTERS> fds_{-1, -1, -1, -1, -1, -1};
    int leader_fd_ = -1;
    bool available_ = false;
    uint64_t start_time_ = 0;
    
    // TSC to nanoseconds ratio (calibrated at startup)
    static inline double tsc_ns_ratio_ = calibrate_tsc();
    
    static int open_counter(uint32_t type, uint64_t config, int group_fd) {
        struct perf_event_attr pe{};
        pe.type = type;
        pe.size = sizeof(pe);
        pe.config = config;
        pe.disabled = 1;
        pe.exclude_kernel = 1;
        pe.exclude_hv = 1;
        
        if (group_fd >= 0) {
            pe.read_format = PERF_FORMAT_GROUP;
        }
        
        int fd = syscall(__NR_perf_event_open, &pe, 0, -1, group_fd, 0);
        return fd;
    }
    
    static uint64_t read_counter(int fd) {
        uint64_t value = 0;
        if (fd >= 0) {
            read(fd, &value, sizeof(value));
        }
        return value;
    }
    
    static inline uint64_t read_tsc() {
        uint32_t lo, hi;
        asm volatile("rdtsc" : "=a"(lo), "=d"(hi));
        return (static_cast<uint64_t>(hi) << 32) | lo;
    }
    
    static double calibrate_tsc() {
        // Calibrate TSC frequency using clock_gettime
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        uint64_t tsc_start = read_tsc();
        
        // Busy wait for ~10ms
        volatile uint64_t dummy = 0;
        for (int i = 0; i < 10000000; ++i) {
            dummy += i;
        }
        
        uint64_t tsc_end = read_tsc();
        clock_gettime(CLOCK_MONOTONIC, &end);
        
        int64_t ns = (end.tv_sec - start.tv_sec) * 1000000000LL + 
                     (end.tv_nsec - start.tv_nsec);
        uint64_t tsc_diff = tsc_end - tsc_start;
        
        return static_cast<double>(ns) / tsc_diff;
    }
};

#else  // Non-Linux fallback

class PerfCounterGroup {
public:
    PerfCounterGroup() = default;
    [[nodiscard]] bool available() const { return false; }
    
    void start() {
        start_time_ = std::chrono::steady_clock::now();
    }
    
    [[nodiscard]] CounterResult stop() {
        auto end_time = std::chrono::steady_clock::now();
        CounterResult result;
        result.time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            end_time - start_time_).count();
        return result;
    }
    
private:
    std::chrono::steady_clock::time_point start_time_;
};

#endif

// =============================================================================
// RDTSC-BASED HIGH-RESOLUTION TIMER
// =============================================================================

/**
 * RDTSC timer for nanosecond-precision latency measurement
 * Used when perf counters are not available or for simple timing
 */
class RdtscTimer {
public:
    RdtscTimer() {
        calibrate();
    }
    
    __attribute__((always_inline))
    void start() noexcept {
        // CPUID serializes the instruction stream
        asm volatile("cpuid" ::: "eax", "ebx", "ecx", "edx");
        start_ = rdtsc();
    }
    
    __attribute__((always_inline))
    [[nodiscard]] int64_t stop_ns() noexcept {
        uint64_t end = rdtscp();
        asm volatile("cpuid" ::: "eax", "ebx", "ecx", "edx");
        return static_cast<int64_t>((end - start_) * ns_per_cycle_);
    }
    
    [[nodiscard]] double ns_per_cycle() const { return ns_per_cycle_; }
    [[nodiscard]] double ghz() const { return 1.0 / ns_per_cycle_; }
    
private:
    uint64_t start_ = 0;
    double ns_per_cycle_ = 1.0;
    
    static inline uint64_t rdtsc() noexcept {
        uint32_t lo, hi;
        asm volatile("rdtsc" : "=a"(lo), "=d"(hi));
        return (static_cast<uint64_t>(hi) << 32) | lo;
    }
    
    static inline uint64_t rdtscp() noexcept {
        uint32_t lo, hi;
        asm volatile("rdtscp" : "=a"(lo), "=d"(hi) :: "ecx");
        return (static_cast<uint64_t>(hi) << 32) | lo;
    }
    
    void calibrate() {
        constexpr int CALIBRATION_ROUNDS = 5;
        double sum = 0.0;
        
        for (int i = 0; i < CALIBRATION_ROUNDS; ++i) {
            auto t1 = std::chrono::steady_clock::now();
            uint64_t c1 = rdtsc();
            
            // Busy wait
            volatile uint64_t dummy = 0;
            for (int j = 0; j < 1000000; ++j) dummy += j;
            
            uint64_t c2 = rdtsc();
            auto t2 = std::chrono::steady_clock::now();
            
            int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
            uint64_t cycles = c2 - c1;
            
            sum += static_cast<double>(ns) / cycles;
        }
        
        ns_per_cycle_ = sum / CALIBRATION_ROUNDS;
    }
};

// =============================================================================
// LATENCY HISTOGRAM WITH P99, P99.9, P99.99 TRACKING
// =============================================================================

/**
 * Lock-free latency histogram for high-frequency measurement
 * Pre-computes percentiles for O(1) access
 */
template<size_t MAX_NS = 1000000, size_t BUCKET_NS = 10>
class LatencyHistogram {
public:
    static constexpr size_t NUM_BUCKETS = MAX_NS / BUCKET_NS;
    
    LatencyHistogram() {
        buckets_.fill(0);
    }
    
    __attribute__((always_inline))
    void record(int64_t ns) noexcept {
        if (ns < 0) ns = 0;
        size_t bucket = static_cast<size_t>(ns) / BUCKET_NS;
        if (bucket >= NUM_BUCKETS) bucket = NUM_BUCKETS - 1;
        
        ++buckets_[bucket];
        ++count_;
        sum_ += ns;
        
        if (ns < min_) min_ = ns;
        if (ns > max_) max_ = ns;
    }
    
    [[nodiscard]] int64_t percentile(double p) const {
        if (count_ == 0) return 0;
        
        uint64_t target = static_cast<uint64_t>(count_ * p);
        uint64_t cumulative = 0;
        
        for (size_t i = 0; i < NUM_BUCKETS; ++i) {
            cumulative += buckets_[i];
            if (cumulative >= target) {
                return static_cast<int64_t>(i * BUCKET_NS + BUCKET_NS / 2);
            }
        }
        
        return max_;
    }
    
    [[nodiscard]] int64_t p50() const { return percentile(0.50); }
    [[nodiscard]] int64_t p90() const { return percentile(0.90); }
    [[nodiscard]] int64_t p99() const { return percentile(0.99); }
    [[nodiscard]] int64_t p999() const { return percentile(0.999); }
    [[nodiscard]] int64_t p9999() const { return percentile(0.9999); }
    
    [[nodiscard]] double mean() const {
        return count_ > 0 ? static_cast<double>(sum_) / count_ : 0.0;
    }
    
    [[nodiscard]] int64_t min() const { return min_; }
    [[nodiscard]] int64_t max() const { return max_; }
    [[nodiscard]] uint64_t count() const { return count_; }
    
    void print(const std::string& name) const {
        std::cout << name << " Latency Distribution:\n";
        std::cout << "-------------------------------------------\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  Count:    " << count_ << "\n";
        std::cout << "  Mean:     " << mean() << " ns\n";
        std::cout << "  Min:      " << min_ << " ns\n";
        std::cout << "  Max:      " << max_ << " ns\n";
        std::cout << "  P50:      " << p50() << " ns\n";
        std::cout << "  P90:      " << p90() << " ns\n";
        std::cout << "  P99:      " << p99() << " ns\n";
        std::cout << "  P99.9:    " << p999() << " ns\n";
        std::cout << "  P99.99:   " << p9999() << " ns\n";
    }
    
    void reset() {
        buckets_.fill(0);
        count_ = 0;
        sum_ = 0;
        min_ = INT64_MAX;
        max_ = 0;
    }
    
private:
    std::array<uint64_t, NUM_BUCKETS> buckets_;
    uint64_t count_ = 0;
    int64_t sum_ = 0;
    int64_t min_ = INT64_MAX;
    int64_t max_ = 0;
};

} // namespace arbor::perf
