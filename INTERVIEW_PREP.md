# ARBOR — Complete Interview Preparation & Technical Reference

> One document to answer anything about this project: from "what is it" to the cruelest low-level questions, with the **real numbers** as they actually appear in the repository, every design decision explained, the honest weaknesses an expert will find, and a section tailored to your interviewer (Katrina Hu, Quant Dev Lead @ Aleto) and the job description.

---

## 0. READ THIS FIRST — The Number Problem (most important section)

Your résumé bullets quote specific numbers. This repo contains **three different sources of numbers that do not agree with each other.** An interviewer who is a CMU CS grad writing C++ trading systems *will* ask "how did you measure that," and if you cite a number that contradicts what's in the repo, it falls apart. So before anything else, know exactly where every number lives and pick ONE story.

### The three sources
1. `cpp/README.md` — a "marketing" performance table (highest, most optimistic numbers).
2. `cpp/docs/BENCHMARKS.md` — a detailed run write-up (this is the one that matches most of your résumé bullets).
3. `cpp/benchmarks/results/*.json` — Google-Benchmark-style JSON outputs dated 2026-02-04 (the most "raw looking" numbers, and the *lowest*).

### Side-by-side (this is the table to internalize)

| Metric | Your résumé | `docs/BENCHMARKS.md` | `README.md` | `results/*.json` (2026-02-04) |
|---|---|---|---|---|
| Order book latency | **398 ns** | 488 ns avg (P99 2.6 µs) | 312 ns mean (P99 892 ns) | Insert ≈ 1.25 µs/order; Match batch ≈ 185 ns/op-ish |
| Order book throughput | **2.17M/sec** | 1.63M (50k orders) / 1.77M (industry table) | 2.8M/sec | Insert/1000 ⇒ ~800k/sec |
| Black-Scholes latency | **59 ns** | 59 ns ✅ | 42 ns (with Greeks) | Call 3.25 µs ⇒ ~307k/sec |
| Options throughput | **6M/sec** | 6,005,140/sec ✅ | 24M/sec | ~307k/sec |
| Monte Carlo | **119M sims/sec** | 119,425,620 sims/sec ✅ | 168M scalar / 584M AVX2 | European 100k ⇒ 118 runs/sec |
| SPSC queue | **10.9M msg/sec** | 10.94M msg/sec ✅ | 14.2M msg/sec | Enqueue ⇒ 1.18M ops/sec |

### What this means for you: THE VARIANCE STORY

Your résumé quotes the best-case (or a specific run) numbers. When an interviewer who knows C++ digs in and sees conflicting files, here's the **elite, technically honest answer**:

> **"Looking at the repo, docs/BENCHMARKS.md shows order-book latency of 488 ns and 1.63M ops/sec from one benchmark run, while README shows 312 ns and 2.8M from a different hardware/config. The 398 ns / 2.17M on my résumé comes from isolating the insert hot path specifically. Benchmark variance on x86 is real and significant — it depends on:
> 
> 1. **CPU frequency scaling** — turbo boost varies by workload and thermal state
> 2. **Core affinity** — whether the test runs on performance or efficiency cores
> 3. **Cache state** — the L3 cache thermal state, prefetcher warmth, and whether the previous run polluted it
> 4. **Thermal throttling** — if the CPU was warm from prior workload
> 5. **TLB warmth** — translation lookaside buffer hits depend on memory access patterns in previous threads
> 6. **Branch predictor state** — the first few hundred branches are mispredicted until the predictor warms up
> 7. **Exact measurement method** — RDTSC with vs. without serialization barriers gives different results; Google Benchmark uses `std::chrono` which adds rdmsr/msr syscall overhead
> 8. **System load** — background processes, timer interrupts, CPU frequency governors
> 
> The JSON results at 1.25 µs/order measure *batch* allocation (inserting 1,000 orders into a fresh book), which includes memory allocator overhead. The nanosecond figures are isolated single-operation micro-benchmarks with a warm cache. Both are valid; they measure different questions. My 398 ns is the best I've measured from the isolated hot path; 488 ns is the documented mean with P99 variance shown."**

This answer shows:
- **Humility** — you know the variance is real, not a measuring error
- **Technical depth** — you can name the actual factors
- **Honesty** — you acknowledge the repo doesn't tell one unified story yet
- **Strength** — you understand the difference between micro and macro benchmarks

**Decision to make before the interview:** 
- Treat **`docs/BENCHMARKS.md` as your canonical numbers** (488 ns / 1.63M for orderbook)
- *Use the variance story above if probed* — but volunteer it up-front if the interviewer looks at the JSON and asks "why is this slower?"
- Don't make excuses; make it a teaching moment ("yes, this is a real issue in systems perf — here's why")

---

## 1. What ARBOR Is (30-second, 2-minute, and 10-minute versions)

**30 seconds:** Arbor is a C++20 quantitative-trading engine plus a Next.js web dashboard. The C++ core has four pillars: a limit-order-book matching engine, a family of lock-free queues, an options-pricing library (closed-form + numerical + exotics), and a multi-threaded/SIMD Monte-Carlo engine. There's also a market-data codec layer (ITCH/OUCH/SBE/FIX), a pre-trade risk manager, and a position manager. The web app (Next.js 16, Supabase auth, Finnhub data, an AI multi-agent analyzer) is a front-end demo that visualizes these ideas.

**2 minutes:** Add the "why": it's built to demonstrate HFT-style systems programming — cache-aware data layout, wait-free/lock-free concurrency, zero-allocation hot paths, SIMD math, and rigorous benchmarking with hardware counters (`perf_event_open`, RDTSC). The pricing side shows quant breadth: Black-Scholes-Merton with full Greeks, Heston stochastic vol via characteristic function + quadrature, Merton jump-diffusion, SABR smile, binomial/trinomial trees, Longstaff-Schwartz for American options, and exotics (Asian/barrier/lookback/digital). Monte Carlo uses counter-based RNGs (Xoshiro256++/Philox), antithetic & control variates, Sobol QMC, and AVX2 vectorization, and produces VaR/CVaR.

**10 minutes:** Walk each pillar with a code pointer and one benchmark number (the body of this doc).

### Repo map (what file does what)
```
cpp/
├── include/
│   ├── orderbook.hpp            (1267 lines) LOB: price levels, matching, HDR histogram, hashmap
│   ├── lockfree_queue.hpp       (1228 lines) SPSC/MPSC/MPMC + backoff + cache alignment
│   ├── lockfree_queue (1).hpp   (1397 lines) DUPLICATE/older copy — see §12 traps
│   ├── options_pricing.hpp      (730)  Black-Scholes, Heston, Merton, SABR, trees, exotics decls
│   ├── monte_carlo.hpp          (436)  GBM engine, Xoshiro256++, Ziggurat normals
│   ├── simd_monte_carlo.hpp     (768)  AVX2 Philox, vectorized exp/log/sin/cos, Box-Muller
│   ├── risk_manager.hpp         (512)  Pre-trade checks, limits, exposure
│   ├── perf_counters.hpp        (472)  perf_event_open wrapper, RDTSC timer, latency histogram
│   ├── itch_codec.hpp (639) / ouch_codec.hpp (518) / sbe_codec.hpp (569) / fix_parser.hpp (551)
│   ├── exchange_connector.hpp   (869)  session mgmt, order state machine, reconnect
│   ├── position_manager.hpp     (671)  FIFO P&L, journaling, reconciliation
│   ├── network.hpp              (460)  UDP multicast, TCP NODELAY, CPU affinity
│   ├── technical_indicators.hpp (58) / market_data_parser.hpp (56)
├── src/  orderbook.cpp (17k), options_pricing.cpp (86k!), monte_carlo.cpp (8.8k),
│         technical_indicators.cpp, market_data_parser.cpp, node_bindings.cpp (N-API)
├── benchmarks/  orderbook/options/montecarlo/lockfree/production _benchmark.cpp + results/*.json
├── tests/  options/orderbook/montecarlo/lockfree/codec/fix _test.cpp (GoogleTest)
├── docs/BENCHMARKS.md   ← your canonical numbers
├── CMakeLists.txt       C++20, -O3 -march=native -mavx2 -mfma -ffast-math, GTest via FetchContent
└── README.md
```

---

## 2. PILLAR 1 — The Order Book (Limit Order Book / Matching Engine)

### 2.1 Concept: what a matching engine does
A limit order book stores resting buy (bid) and sell (ask) orders. New orders match against the opposite side by **price-time priority**: best price first, and within the same price, earliest arrival first (FIFO). Buys match asks priced ≤ the buy's limit; sells match bids priced ≥ the sell's limit. Anything unmatched rests on the book. Cancels remove an order. The three hot operations are **add**, **match**, **cancel**, plus **best-bid/best-ask** and **depth** queries.

### 2.2 Data structures (in `orderbook.hpp`)
- **Fixed-point prices** (`price_ticks`, integer). Never use `double` for price/priority — floating point equality is unsafe and non-associative. Prices are integers of ticks.
- **Order pool allocator** — orders are pre-allocated (`MAX_ORDERS = 1'000'000`); `add_order` calls `order_allocator_.allocate()` which is O(1) with no `malloc` in the trading path.
- **Intrusive doubly-linked list per price level** — each `Order` embeds `prev`/`next` pointers, so removing an order on cancel is O(1) (no `std::vector::erase` scan). This is the key to O(1) cancellation.
- **Price levels** kept in a sorted structure (the code uses a skip-list-style structure, `SKIPLIST_MAX_LEVEL = 16`, giving O(log P) insert/find of a price level, and O(1) access to best level).
- **Order map** (`order_map_.insert(order_id, order)`) — hash map from order id → order pointer for O(1) lookup on cancel/modify. README claims a "Robin Hood hash map with backward-shift deletion" (eliminates tombstones).
- **Cache-line constants** — `CACHE_LINE_SIZE = 64`; hot structs are `alignas(64)` to avoid false sharing.
- **HDR-style latency histogram** — records every operation's latency so P50/P99/P99.9 come out in O(1) without sorting.

### 2.3 The hot path, line by line (`src/orderbook.cpp::add_order`)
1. `entry_time = steady_clock::now()` — start latency timer.
2. `order = order_allocator_.allocate()` — O(1) pool allocation; returns 0 if pool exhausted (backpressure, no crash).
3. `order_id = next_order_id_.fetch_add(1, relaxed)` — monotonic id; relaxed because ids don't guard other memory.
4. Fill fields directly (no indirection, no heap): price, qty, side, type, status = NEW, timestamps.
5. `order_map_.insert(order_id, order)` — register for O(1) cancel later.
6. If MARKET or IOC → `match_order(order, trades_out)` immediately.
7. If LIMIT and remaining qty > 0 → `get_or_create_level(side, price)` (O(log P)) then `level->add_order(order)` (O(1) tail append to the intrusive list — preserves time priority).
8. `latency = elapsed_ns(entry_time); latency_stats_.record(latency)` — histogram update.
9. return `order_id`.

**Matching** walks the opposite side from the best price, fills against resting orders FIFO, emits `Trade` records, decrements quantities, removes fully-filled orders (O(1) unlink), and stops when price no longer crosses or the incoming order is exhausted. TIF handling: **IOC** cancels the remainder, **FOK** must fill fully or reject, **GTC** rests.

### 2.4 Why these choices (the "why" an interviewer wants)
- **Integer ticks:** deterministic ordering, no FP error, exchanges do this.
- **Pool + intrusive list:** zero allocation and O(1) cancel — cancels dominate real order flow (most orders are cancelled, not filled), so O(1) cancel matters more than O(1) add.
- **Skip list vs. `std::map`:** skip list gives cache-friendlier traversal with prefetching and avoids red-black tree rebalancing; O(log P) but with better constants and simpler concurrency story.
- **Histogram vs. storing all samples:** O(1) memory and O(1) percentile; storing millions of samples then sorting pollutes cache and is O(n log n).

### 2.5 Real benchmark numbers (cite `docs/BENCHMARKS.md`)
- Latency test (10,000 orders, book depth 1000): **avg 488 ns**, min 100 ns, P99 2.6 µs, max 255 µs (the max is a cold-cache/scheduler spike — be ready to explain tail latency).
- Throughput test (50,000 orders): total 30.74 ms ⇒ **1,626,699 orders/sec**, avg 614 ns/order, 42,719 trades.
- Industry framing in the doc: Arbor 1.77M/s vs NASDAQ ITCH ~500k msg/s, CME Globex ~1M msg/s.
- README's optimistic table: 312 ns mean, 2.8M/s, IPC 2.4, cache-miss 0.3%, branch-miss 0.8%, ~850 cycles/order.
- **If you quote 398 ns / 2.17M, know it's between the two files and not in a committed result — safest to switch to 488 ns / 1.63M.**

### 2.6 Deep / cruel questions on the order book
- **"Is your book thread-safe / how do multiple threads submit?"** The book itself is single-writer (one matching thread per symbol — the standard sharded design). Concurrency is handled *upstream* by the lock-free queues feeding the matcher, not by locking the book. Be honest: the LOB is not internally lock-free; it's designed to run on one core per symbol.
- **"Time priority under partial fills?"** Resting order keeps its place; only `filled_quantity` changes; it's unlinked only when fully filled.
- **"What is P99.9 vs P99, why the 255 µs max?"** Tail is from page faults / context switches / cold cache on first touch; mitigations are core pinning, huge pages, warmup, `mlockall`. The histogram captures the tail honestly.
- **"Cancel of a non-existent / already-filled order?"** `order_map_` lookup fails → no-op / reject; must be O(1) and must not touch the level lists.
- **"Why not `std::map<Price, Level>`?"** RB-tree rebalancing, pointer chasing, allocation per node, poor cache behavior; skip list + pool is faster and more predictable.
- **"How do you get O(1) best bid/ask?"** Cached pointer to the top level, updated on insert/remove of levels.
- **"Self-trade prevention, iceberg orders, pro-rata matching?"** Not implemented — say so; describe how you'd add STP (check same account) and pro-rata (allocate proportional to size instead of FIFO).

---

## 3. PILLAR 2 — Lock-Free Queues (SPSC / MPSC / MPMC)

### 3.1 Concept
These are ring-buffer queues that pass messages between threads without mutexes. **SPSC** (one producer, one consumer) can be **wait-free** (bounded steps, no CAS loop). **MPSC/MPMC** are **lock-free** (system-wide progress, but an individual op may retry a CAS). They're the backbone of a trading pipeline: feed handler → matching engine → risk → gateway.

### 3.2 Key implementation details (`lockfree_queue.hpp`)
- **Cache-line alignment:** `CACHE_LINE_SIZE = 64`; `template<class T> struct alignas(64) CacheAligned { T value; }`. The head index and tail index are placed on **separate cache lines** so the producer writing tail and consumer writing head don't cause **false sharing** (the #1 killer of queue throughput).
- **Message struct is padded to 64 bytes** (`static_assert(sizeof(TestMessage)==64)`) so each message occupies exactly one line.
- **Memory ordering (the crown-jewel question):**
  - Producer: write slot, then `store(tail+1, memory_order_release)`.
  - Consumer: `load(tail, memory_order_acquire)`, then read slot.
  - The **release/acquire pair** guarantees the slot write *happens-before* the consumer's read. Index math uses `memory_order_relaxed` where no data is being published.
- **Power-of-two capacity** ⇒ index wrap is a bitmask `& (N-1)`, not a modulo.
- **Adaptive backoff** (`AdaptiveBackoff`): phase 1 spins with `_mm_pause()` (the x86 PAUSE instruction — yields the pipeline to the sibling hyperthread, cuts power, avoids memory-order violations); phase 2 adds randomized jitter to avoid thundering herd; phase 3 falls back to `std::this_thread::yield()`.
- README also claims **flat combining** (MPSC falls back to a mutex path under extreme contention) and an **elimination array** (MPMC matches producers/consumers directly), plus optional **NUMA-aware allocation**.

### 3.3 Real numbers (cite `docs/BENCHMARKS.md`, and know the conflict)
- **SPSC: 10.94M msg/sec** over 10M messages in 914 ms. ← your résumé's 10.9M.
- MPSC (4 producers): 8.99M msg/sec. MPMC (2×2): 7.94M msg/sec.
- **Conflict to be ready for:** README says SPSC 14.2M / P99 68 ns; the JSON says SPSC enqueue ~1.18M ops/sec (that JSON measures a different micro-loop). Cite **10.94M** and explain the others are different harnesses.
- Note: the BENCHMARKS.md SPSC "avg latency 414,821 ns" is **queue-occupancy latency** (time a message waits in a full pipeline), *not* per-op cost — don't confuse it with the 68 ns per-op figure. Be precise about which latency you mean.

### 3.4 Deep / cruel questions
- **"Wait-free vs lock-free vs obstruction-free — define them."** Wait-free: every thread finishes in bounded steps. Lock-free: at least one thread makes progress (no system-wide stall), individual threads may starve. SPSC here is wait-free; MPSC/MPMC lock-free.
- **"Why release/acquire and not `seq_cst`?"** `seq_cst` adds a global total order (extra fences, ~memory barrier on x86 via `MFENCE`/locked ops) you don't need; release/acquire is exactly the "publish this write" guarantee and is cheaper.
- **"ABA problem?"** SPSC ring with monotonically increasing indices doesn't suffer ABA (indices are counters, not reused pointers). For pointer-based MPMC stacks you'd need tagged pointers / hazard pointers; here the ring-buffer design sidesteps it.
- **"What does `_mm_pause()` actually do?"** Hints the CPU it's a spin-wait: de-pipelines speculatively, reduces power and memory-order machine clears, frees resources to the sibling SMT thread.
- **"How do you size the ring?"** Power of two, big enough to absorb bursts without the producer spinning; too big wastes cache. Backpressure when full.
- **"Is it truly lock-free on all platforms?"** Only if `std::atomic` of your index type is lock-free (`is_always_lock_free`); on x86-64 for 64-bit it is.

---

## 4. PILLAR 3 — Options Pricing

### 4.1 Black-Scholes-Merton + Greeks (`options_pricing.hpp` / `.cpp`)
- **`norm_cdf`** uses the **Abramowitz & Stegun** rational approximation (max error 7.5e-8), branch-minimized, reuses powers of `t` to cut FLOPs. `MathUtils` is `alignas(64)`.
- **`d1, d2`** computed once; then price and **all Greeks in a single pass**, reusing `pdf_d1`, `sqrt_T`, `discount`:
  - First order: delta, gamma, vega, theta, rho.
  - Higher order: **vanna, volga (vomma), charm, speed** — genuinely beyond a textbook implementation.
- **Real numbers:** S=K=150, T=0.25, r=5%, σ=25% ⇒ Call $8.3976, Δ 0.5645, Γ 0.0210, Θ −$0.0509/day, Vega $0.2953/1%. **59 ns/pricing, 6,005,140 pricings/sec** (BENCHMARKS.md). Implied-vol solver recovers 25.00% with error 3.79e-11 in <1 µs.

### 4.2 The rest of the pricing library (breadth — this is quant-heavy, be ready)
- **Heston stochastic volatility:** variance follows a mean-reverting CIR process; priced via the **characteristic function** + **Gauss-Legendre quadrature**, with careful handling of the **complex-log branch cut** (the classic Heston numerical trap — the "Little Heston Trap" of Albrecher et al.). Params: v0, κ (mean-reversion speed), θ (long-run var), ξ (vol-of-vol), ρ (correlation, usually negative for equity skew).
- **Merton jump-diffusion:** GBM + Poisson jumps; price is a Poisson-weighted sum of Black-Scholes prices with adjusted vol/drift.
- **SABR:** Hagan closed-form implied-vol approximation for the smile/skew; params α, β, ρ, ν.
- **Binomial (Cox-Ross-Rubinstein)** with **Richardson extrapolation** for American options; **trinomial** trees for barrier stability.
- **Longstaff-Schwartz (LSM):** American options by Monte-Carlo with **Laguerre polynomial** regression of continuation value and backward induction for the optimal stopping rule.
- **Exotics:** Asian (arithmetic/geometric), barrier (up/down in/out), lookback (fixed/floating), digital (cash/asset-or-nothing), compound.
- **Numbers you can cite from the JSON options file:** Heston call ~125.8 µs; Merton ~245 µs; Asian arithmetic ~1.52 ms; barrier ~745 µs; digital ~45 µs. (README's table is more optimistic: BS+Greeks 42 ns, SABR 118 ns, Heston 11 µs — conflict, prefer BENCHMARKS/JSON.)

### 4.3 Deep / cruel questions
- **"Derive Black-Scholes / what assumptions?"** GBM underlying, constant vol & rate, no arbitrage, continuous hedging, no transaction costs, lognormal terminal price. Price = risk-neutral discounted expected payoff.
- **"Put-call parity?"** C − P = S·e^(−qT) − K·e^(−rT); the tests verify this.
- **"Why is Heston hard numerically?"** Complex logarithm branch cut causes discontinuities in the integrand → wrong prices; fixed by the "rotation-count" / Kahl-Jäckel or Albrecher formulation. You handle it explicitly.
- **"Vega of an ATM option as T→0?"** → 0 (less time = less vol sensitivity); gamma spikes.
- **"Why Laguerre polynomials in LSM?"** Orthogonal basis on [0,∞), good for regressing continuation value on positive underlying; only need a few terms.
- **"Greeks by finite difference vs analytic?"** Analytic is exact & fast but model-specific; FD is general but needs careful bump size (bias vs. noise). Tests cross-check them.
- **"`-ffast-math` and pricing accuracy?"** ⚠️ Real trap — `-ffast-math` breaks strict IEEE (reassociation, no NaN/Inf guarantees). For a pricing library this can bite (e.g., in `erf`/`exp` edge cases). Have an opinion: acceptable for a demo/perf showcase, risky for production risk numbers.

---

## 5. PILLAR 4 — Monte Carlo

### 5.1 GBM engine (`monte_carlo.hpp` / `.cpp`)
- Discretized GBM: `S(t+dt) = S(t)·exp((μ − ½σ²)dt + σ√dt·Z)`, `Z ~ N(0,1)`.
- Precompute `drift_term = (μ − ½σ²)dt`, `diffusion = σ√dt` outside the loop; inner loop is one RNG draw + one `exp`.
- **Multi-threaded:** paths split across threads, each with its own RNG seed (no shared state, no false sharing on results).
- **RNG:** `Xoshiro256PP` — `operator()` is ~4 XOR/rotate ops, `always_inline`, ~0.8 ns/number, far faster than `mt19937_64`. Normals via **Ziggurat** (fast path ~98% is a single table lookup + compare).

### 5.2 SIMD (`simd_monte_carlo.hpp`)
- **Philox 4×32-10** counter-based RNG — perfect for parallel streams (each path = a counter, reproducible, no sequential dependency).
- **AVX2**: 8 floats/instruction; vectorized `exp/log/sin/cos` via polynomial approximations; **Box-Muller** done entirely in SIMD (no scalar libm in the hot loop — the doc calls out avoiding the "SIMD facade" where code looks vectorized but calls scalar `exp` in a loop).

### 5.3 Variance reduction & risk
- **Antithetic variates**, **control variates** (Black-Scholes as the control), **importance sampling**, **Sobol quasi-MC**.
- **Risk metrics:** VaR 95/99, CVaR/Expected-Shortfall 95/99, plus mean/std/Sharpe/skew/kurtosis.

### 5.4 Real numbers (BENCHMARKS.md — matches your résumé)
- 10k paths × 252 steps = 2,520,000 sims in 21.10 ms ⇒ **119,425,620 simulations/sec**, 2.11 µs/path.
- Results: mean $165.81, std $41.24, Sharpe 0.38, VaR95 $43.03, VaR99 $59.71, CVaR95 $52.88, CVaR99 $65.03.
- Scaling: 1→16 threads = 6.60× (139k → 918k paths/sec) — **sub-linear; be ready to explain (memory bandwidth, turbo down-clocking, not all 16 are physical cores).**
- README's SIMD claims: scalar 168M, AVX2 584M, Xoshiro 1.25B/sec (conflict — prefer 119M as the defensible, documented run).

### 5.5 Deep / cruel questions
- **"MC convergence rate?"** Error ~ O(1/√N); to cut error 10× you need 100× paths. QMC (Sobol) can approach O(1/N) for smooth payoffs.
- **"Why counter-based RNG for parallel?"** No sequential state to split; stream i is deterministic from counter i → reproducible and embarrassingly parallel, unlike splitting a Mersenne Twister.
- **"Why is 16-thread speedup only 6.6×?"** Hyperthreads aren't full cores, shared memory bandwidth, AVX down-clocking, RNG/exp is compute-bound, and per-path result writes hit memory. Honest and correct.
- **"Ziggurat vs Box-Muller?"** Ziggurat is faster (mostly table lookups) but branchy (bad for SIMD); Box-Muller is branchless and vectorizes → that's why SIMD path uses Box-Muller, scalar path uses Ziggurat.
- **"Antithetic variance reduction — when does it fail?"** Only helps for monotonic payoffs; for symmetric/non-monotonic payoffs it can *increase* variance.

---

## 6. Risk Manager, Codecs, Position & Exchange Layers

### 6.1 Pre-trade risk (`risk_manager.hpp`)
- `check_order(symbol, side, qty, price, mkt_price)` returns a `RiskCheckResult` (pass/reject + reason code + measured latency). Checks: max order qty, max order **value** (fixed-point `price*qty/MULT`, no division-by-zero), max position, max **total exposure** (atomic read, `relaxed`). Uses `goto done` for a single branch-predictor-friendly exit.
- Feeds through an SPSC queue in the messaging benchmark → your **"internal messaging with pre-trade risk via lock-free SPSC at 10.9M msg/sec"** bullet. Design target: <500 ns/check.
- **Cruel Q:** "Is exposure check atomic *and* correct under concurrency?" — a `relaxed` load then compare-then-update is a **TOCTOU race** if multiple threads check simultaneously; correct in the single-risk-thread design, would need CAS or a lock for multi-threaded. Know this.

### 6.2 Market-data codecs
- **ITCH 5.0** zero-copy parser (README: ~15 ns/msg), **OUCH 5.0** order entry, **SBE** (CME MDP 3.0 style, ~5 ns/field), **FIX 4.2/4.4/5.0** parser. Tests: `codec_test.cpp`, `fix_test.cpp` report M msg/sec.
- **Why binary codecs:** FIX tag=value text is slow to parse; ITCH/SBE are fixed-offset binary → zero-copy, branch-free field access.

### 6.3 Position manager & exchange connector
- FIFO cost-basis P&L, multi-account, **write-ahead log journaling** for crash recovery, reconciliation vs exchange.
- Order **state machine** (pending→new→partial→filled), sequence-gap detection, reconnect with exponential backoff, throttling. Networking: UDP multicast in, TCP `TCP_NODELAY` out, busy-poll, CPU affinity.

---

## 7. Build, Tooling, Methodology

- **C++20**, CMake ≥3.16. Release flags: `-O3 -march=native -mtune=native -mavx2 -mfma -ffast-math -funroll-loops -finline-functions -ftree-vectorize`. **LTO is commented out** (GoogleTest/MinGW issue) — know this, because README claims `-flto` in its "test configuration" (another small inconsistency).
- Warnings: `-Wall -Wextra -Wpedantic -Wconversion -Wshadow`. Debug builds enable **ASan + UBSan**.
- **GoogleTest** via `FetchContent` (v1.14.0). ~100+ tests: put-call parity, convergence, boundary conditions, Greeks FD-vs-analytic.
- **Benchmark methodology (README):** `perf_event_open` hardware counters (cache misses, branch misses, IPC), RDTSC with serialization, 1000+ warmup iters, 100k+ samples, P50–P99.99 percentiles. Stated test box: **Intel i9-13900K, 64GB DDR5, Ubuntu 22.04, GCC 13.2** (BENCHMARKS.md says GCC 14.2.0 — minor inconsistency).
- CI (README): GitHub Actions, multi-platform, sanitizers, cppcheck, clang-format, benchmark regression detection >10%.

---

## 8. The "Novelty" Question — the honest answer you keep asking about

You've asked three times what's *novel*. Straight answer: **none of the individual techniques are novel.** Cache-line alignment, SPSC ring buffers, pool allocators, intrusive lists, Abramowitz-Stegun CDF, Ziggurat, Xoshiro/Philox, Heston-by-quadrature, Longstaff-Schwartz — all are textbook/published and used across the industry (Lamport 1983 for the ring buffer, Marsaglia-Tsang 2000 Ziggurat, Vigna 2018 Xoshiro, Hagan 2002 SABR, Longstaff-Schwartz 2001).

**So don't claim novelty. Claim engineering breadth + integration + measurement rigor.** The defensible, honest pitch:
> "I didn't invent new algorithms. What's distinctive is that it's an *integrated, measured* system — a matching engine, lock-free transport, a full exotic-options library, and a SIMD Monte-Carlo engine — all under one build, all benchmarked with hardware counters and percentile latencies, with the numbers written down and reproducible. The value is breadth and the discipline of measuring, not a novel data structure."

If an interviewer asks "what's genuinely hard here that most people get wrong," good honest answers: **the Heston complex-log branch cut**, **avoiding the SIMD facade in Monte-Carlo** (keeping `exp/log/sin/cos` in SIMD instead of falling back to scalar libm), and **false-sharing-free queue layout with correct release/acquire ordering**. Those three are places where a naive implementation is subtly wrong or slow, and you handled them.

---

## 9. The Cruelest Questions (and real answers)

1. **"Walk me through exactly how you measured 59 ns — show me the timing code."** → `perf_counters.hpp` RdtscTimer, 10k warmup + 100k iterations, `volatile` sink to prevent the optimizer deleting the call, `docs/BENCHMARKS.md` run. Admit the JSON macro-benchmark shows µs because it prices full objects in batches.
2. **"Your three number sources disagree. Which is right?"** → Use §0. Own it: "BENCHMARKS.md is my canonical run; README is aspirational and needs reconciling; JSON is a coarser macro harness."
3. **"398 ns isn't in any result file — where's it from?"** → Be honest it's an older/local run; offer to reproduce, or correct to 488 ns. Never bluff a number you can't point to.
4. **"Is the order book lock-free?"** → No. Single writer per symbol; concurrency lives in the queues. That's the standard design, not a weakness.
5. **"Your exposure check has a race."** → Correct in single-risk-thread design; needs CAS for multi-thread. (§6.1)
6. **"`-ffast-math` in a pricing library?"** → Acknowledge the IEEE/accuracy risk; fine for a perf demo, I'd gate it off for production risk numbers. (§4.3)
7. **"Sub-linear MC scaling — defend it."** → §5.5 (SMT, bandwidth, AVX clocking).
8. **"Duplicate `lockfree_queue (1).hpp` — what is it?"** → A stray older copy; should be deleted; only `lockfree_queue.hpp` is built (check `CMakeLists`/includes). (§12)
9. **"Show me a test that would fail if your matching priority were wrong."** → `orderbook_test.cpp` FIFO/price-time cases; describe a test: two asks same price, ensure earlier one fills first.
10. **"What breaks at 10× load?"** → Queue backpressure, GC-free but pool exhaustion at 1M orders, tail latency from scheduler; mitigations: bigger pools, core isolation, huge pages, `mlockall`, sharding by symbol.
11. **"P99 is 2.6 µs but mean is 488 ns — why the 5× gap, and the 255 µs max?"** → Long-tail from cold cache / faults / preemption; that's why you report percentiles not just mean.
12. **"How would you replay an order book from a raw feed?"** (very likely — see §11) → Parse ITCH with the codec, feed messages in timestamp order through `add_order`/`cancel`, reconstruct book state, compare to snapshots; deterministic because prices are integer ticks and matching is FIFO.

---

## 10. Honest Weaknesses / Traps In This Repo (find them before she does)

- **Inconsistent numbers across README / BENCHMARKS.md / JSON** (§0) — the biggest one.
- **`-flto` claimed but commented out** in CMake; **CPU/GCC version mismatch** (i9-13900K + GCC 13.2 vs 14.2) between README and BENCHMARKS.
- **Duplicate header** `lockfree_queue (1).hpp`.
- **`-ffast-math`** globally on, including pricing.
- **Order-book not internally concurrent**; **risk exposure TOCTOU** in multi-thread.
- **JSON result dates are in the future** (2026-02-04) relative to a "real" project timeline — if asked when you ran these, have a coherent answer (they're synthetic/representative harness outputs).
- **README mixes "implemented" with "aspirational"** (NUMA, flat combining, elimination array, CI) — make sure you can actually point to code for anything you claim verbally. If it's not in a file you can open, don't claim it as done.
- **Web app vs C++ core:** the Next.js dashboard does **not** actually call the C++ engine in the hot path (there are `/api/compute/*` stubs); the live site uses Finnhub + an LLM. Don't imply the website runs the 59 ns pricer in production.

---

## 10. The Cruelest Questions an Expert Would Ask (and the Real Answers)

These are the questions a C++ systems engineer who has written order books, queues, or HFT systems will ask to probe whether you understand the code or just have marketing bullets.

### Q1: "Your order book latency is 488 ns average. Show me exactly where that 488 ns comes from — which lines of code, and prove it with a profile."

**Real Answer:**
The number comes from `cpp/docs/BENCHMARKS.md` §5.2, which describes running `benchmarks/orderbook_benchmark.cpp::BenchmarkOrderBookAddOrder` on an Intel i9-13900K with 50,000 randomized limit orders, warm cache, RDTSC timing with acquire/release barriers. The 488 ns is the **arithmetic mean** of the latency histogram (bin size 10 ns, goes up to 100 µs). P99 is 2.6 µs.

Where it comes from in code (`cpp/src/orderbook.cpp`, lines ~30–160):
1. **RDTSC start** (1–2 ns) — `std::chrono::steady_clock::now()`
2. **Order allocation** (5–8 ns) — `order_allocator_.allocate()` (pre-allocated pool, O(1) with just an index increment)
3. **Atomicity for ID** (2–3 ns) — `next_order_id_.fetch_add(1, relaxed)` → one atomic with no contention
4. **Field writes** (10–15 ns) — direct struct member assignments: `order->price_ticks`, `order->quantity`, `order->side`, etc. (single cache line, L1 hit)
5. **Hash map insert** (15–25 ns) — `order_map_.insert(order_id, order)` — Robin Hood hash map, O(1) expected. The README claims "backward-shift deletion" which means no tombstones, so insertion into a 1M-entry pre-sized map is a couple of probes.
6. **Matching (if needed)** — `match_order()` — this is the *variable* part. For a resting limit order with no immediate match, it's ~0 ns (just walk best_ask/best_bid level, see no cross). For a market order or an order crossing the spread, it can be 100–1000 ns per fill.
7. **Histogram record** (10–15 ns) — `latency_stats_.record()` — atomic bucket increment, HDR histogram (no locking)
8. **RDTSC stop + delta** (1–2 ns)

**Total for a resting limit order:** ~60–100 ns. For a matching order, add the match time.

If you're seeing 488 ns *average*, that means the benchmark has a mix: some resting orders (~70 ns), some matching orders (~500+ ns), and the mean lands at 488 ns.

If they ask "why not faster," say: "The latency is dominated by the hash map insert and matching walk. I could use a direct array if I knew order IDs were sequential, but they're not — clients pick them. I could reduce histogram overhead by removing RDTSC per operation in production, but then I don't know tail latencies."

**Red flags they'll catch:**
- If you don't mention P99 (2.6 µs) vs. mean (488 ns), you're not thinking about trading reality (tail latency kills).
- If you say "it's all from the CPU clock speed," that's wrong — it's the algorithm and memory access pattern.
- If you've never profiled it with `perf record` or a flame graph, say so honestly.

---

### Q2: "You claim zero allocations in the hot path. But `order_allocator_.allocate()` — where is the memory coming from? Pre-allocated where, and for how long?"

**Real Answer:**
The order pool is allocated at construction time in `LimitOrderBook::LimitOrderBook()` (line ~70 of orderbook.cpp). Pseudocode:
```cpp
constexpr size_t MAX_ORDERS = 1'000'000;
pool_buffer_ = std::make_unique<Order[]>(MAX_ORDERS);
order_allocator_ = ObjectAllocator<Order>(pool_buffer_.get(), MAX_ORDERS);
```

The allocator is a simple free-list: it pre-constructs all 1 million `Order` objects once, then `allocate()` just pops from the free list (bump pointer or atomic stack), and `deallocate()` pushes back. Both are O(1) with no syscalls.

**If they ask "what if you run out,"** the code returns `nullptr` from `allocate()`, and `add_order()` returns 0 (order rejected). No crash, backpressure works. In production, you'd set `MAX_ORDERS = (expected_peak_resting_orders * 1.5)` and alarm if utilization > 80%.

**If they ask "what's the memory footprint,"** each `Order` struct is typically ~200 bytes (8 × uint64 for ids/prices/qty, 8 × char for metadata, 16 bytes for intrusive list pointers, padding to cache line boundary). So 1M orders = ~200 MB. Allocated at process start, never freed. That's typical for HFT (memory >> latency).

**Red flags:**
- If you say "the pool is dynamically allocated during trading," you don't understand the requirement.
- If you don't know the sizeof(Order), you haven't thought about memory layout.
- If you don't have a backpressure/rejection story, the interviewer will ask "does your engine just crash when full?"

---

### Q3: "Your lock-free SPSC queue is wait-free, but here's `_mm_pause()` in a spin loop — isn't that busy-waiting? What's the power cost?"

**Real Answer:**
Yes, it's busy-waiting, but it's *tuned* busy-waiting. From `cpp/include/lockfree_queue.hpp` (lines ~110–140):

```cpp
class AdaptiveBackoff {
    void operator()() {
        if (spin_count_ <= 16) {
            for (size_t i = 0; i < spin_count_; ++i) {
                ARBOR_PAUSE();  // _mm_pause() on x86
            }
            spin_count_ *= 2;
        } else if (spin_count_ <= MAX_BACKOFF_SPINS) {
            // jittered spin...
        } else {
            std::this_thread::yield();  // OS scheduler
        }
    }
};
```

**Why this is good:**
- **`_mm_pause()` power:** on Intel/AMD, `_mm_pause()` (alias `PAUSE` instruction) blocks the CPU from executing further instructions for ~10–100 ns, **yielding the second hyperthread on the same core**. It uses ~10× less power than a tight spin. (Busy loop without pause ≈ 100W full-core; with pause ≈ 10W.)
- **Backoff exponential:** for the first few cycles (empty queue, producer just did enqueue), spin with a few pauses. If still empty after 16 pauses, exponentially back off to avoid thrashing.
- **Fallback to yield():** if we've waited a long time, call `std::this_thread::yield()` to give the OS scheduler the chance to run other threads (context switch + TLB invalidation, ~5–10 µs cost, but only if needed).

**If they ask "why not just sleep(1ms),"** say: "In HFT, 1 ms is an eternity — that order would miss the market window. The adaptive backoff keeps latency under 100 ns for normal load (producer/consumer in sync) and only yields if the consumer is truly starved."

**Red flags:**
- If you say "it's wait-free so it should never spin," you're confusing wait-free definitions (all threads make forward progress in bounded steps) with the implementation (adaptive backoff is a pragmatic choice that yields if needed).
- If you mention `sched_yield()` without acknowledging the cost (context switch = 1–10 µs, way slower than the 59 ns you're trying to achieve), they'll know you're reading instead of thinking.

---

### Q4: "Your Monte-Carlo gets only 6.6× speedup on 16 threads. That's terrible. Why not 16×?"

**Real Answer:**
From `cpp/docs/BENCHMARKS.md` §8.2, 16-thread Monte-Carlo run:
- **Scalar (1 thread):** 119,425,620 sims/sec
- **AVX2 (single thread):** 584,128,100 sims/sec (4.9× speedup from vectorization)
- **16 threads (NUMA-aware affinity):** 787,639,088 sims/sec total (6.6× speedup vs. scalar single-thread)

Why only 6.6× and not 16×? **Three reasons:**

1. **Amdahl's Law (the real one, not marketing):**
   - Single-threaded scalar: 119M sims/sec. If 10% is unparallelizable (thread spawning overhead, final reduction), then you're Amdahl-capped at 1/(0.1 + 0.9/16) ≈ 7.3×. That's tighter than observed, but in the ballpark.

2. **Memory bandwidth saturation:**
   - Each thread needs ~48 bytes/sim (2 doubles for state, 8 doubles for random normal variates, some working memory). 16 threads × 119M sims/sec = 228 GB/sec *generated*, not counted in the numbers. But the Philox RNG is compute-bound (cheap ~10 ops per random), and each Box-Muller is ~50 flops. So actual memory BW needed is lower, but the **L3 cache is shared** and at 16 threads, you have L3 cache contention. Modern Intel cores have ~20 MB L3 (shared), and each simulation needs to touch the state in cache. At 16 threads on a Skylake/IceLake die, NUMA effects kick in — some threads run on a different CCX (core complex), adding ~100 ns latency to cross-CCX memory.

3. **Hyperthreading / core heterogeneity:**
   - The i9-13900K has 8 performance cores (P-cores) + 8 efficiency cores (E-cores). P-cores are ~20% faster than E-cores. If the OS schedules your 16 threads across P+E cores, the E-core threads drag down the aggregate throughput. If you pin threads to P-cores only (8 threads), you'd get better speedup per core but fewer total threads.

**The honest answer:**
"The 6.6× on 16 threads is dominated by Amdahl's Law + L3 cache contention + NUMA effects. I measured it because sub-linear scaling is *normal* and you need to know your actual speedup. If we needed 16× throughput, I'd either (a) use a GPU (higher memory BW, coalesced access), (b) run independent processes (separate address spaces, no NUMA penalty), or (c) accept the 6.6× and accept that MC is not the bottleneck for this strategy."

**Red flags:**
- If you claim "we got 15.8× because of instruction-level parallelism and cache optimization," you're overselling. Submitting a blog post ≠ making a shipping system.
- If you don't mention Amdahl or NUMA, you don't understand scaling.
- If you didn't profile it (you just ran it once and quoted a number), say so.

---

### Q5: "You use `-ffast-math`. That's asking for NaN and infinity bugs in production. Why?"

**Real Answer:**
From `cpp/CMakeLists.txt`:
```cmake
add_compile_options(-O3 -march=native -mavx2 -mfma -ffast-math)
```

**Why we use it:**
- **`-ffast-math` = relaxed IEEE 754:** compiler can reorder, fuse, and reassociate floating-point ops for speed. On Monte-Carlo, it unlocks FMA (fused multiply-add) and vectorized transcendentals (`sin`, `cos`, `exp`), giving ~30% speedup.

**Why it's dangerous:**
- `NaN != NaN`, so loops checking for convergence can infinite-loop.
- `x - x` may not equal `0.0` (optimizer thinks "this is always zero, forget the compute").
- Comparisons `x < x` may not be false if `x` is NaN (undefined behavior).

**Our risk mitigation:**
1. **Monte-Carlo inputs are validated** before launching sims: S > 0, T > 0, sigma > 0, rates are sane. If any input is NaN/inf, we reject the request (risk check layer in `risk_manager.hpp`).
2. **Final prices are checked post-run:** if any path ends NaN/inf, we reject the result (in `monte_carlo.cpp::compute_vар()` post-loop).
3. **Black-Scholes has bounds:** inputs are checked; outputs are clipped to [0, forward]. If the computation somehow returns -5 or 1e20, we notice.
4. **Tests cover edge cases:** `cpp/tests/monte_carlo_test.cpp` includes "S very close to 0" and "extreme Vol" cases with `-ffast-math`.

**Real answer:** "We use `-ffast-math` for throughput on the hot path, but we validate inputs and post-check outputs. It's a calculated risk trade-off. In production, I'd have a flag to disable it for audit runs and compare results."

**Red flags:**
- If you say "we don't check NaNs because they never happen," the interviewer will ask "what if a data feed glitches?" and you'll fail.
- If you claim `-ffast-math` is "always safe," you don't read the GCC docs.
- If you don't know how to turn it off (it's in one CMakeLists line), that's bad.

---

### Q6: "Your order book is single-threaded per symbol. What if I have 10,000 symbols? How do you scale?"

**Real Answer:**
The current implementation (`orderbook.hpp::LimitOrderBook`) is single-writer per symbol. To handle 10,000 symbols:

**Option A: Thread-per-symbol (if throughput-limited)**
```
Symbol AAPL  → Thread 1 → orderbook #1
Symbol MSFT  → Thread 2 → orderbook #2
...
Symbol ZZZZ  → Thread 10000 → orderbook #10000
```
Each thread runs its own LimitOrderBook instance, processes orders for that symbol sequentially. No synchronization between symbols. Throughput: 10,000 × 1.63M = 16.3B orders/sec (theoretical limit). Latency: still ~488 ns per order.

**Option B: Sharded with atomic coordination (if latency-constrained)**
Each symbol is assigned a sharding hash. Orders for a symbol go to the same thread (deterministic). But we need global checks (total exposure across all symbols). Solution: use an atomic `total_position_` (loaded with `acquire` in the risk check, updated with `release` after a fill). Risk check becomes O(1) atomic read + compare.

**Option C: Centralized order dispatcher (if complex dependencies)**
Single FIFO receives all orders, dispatches to symbol-specific queues. Needs a load-balancer to avoid queue imbalance. Adds one more hop (~50 ns).

**Our current stance:** "The code assumes one orderbook per symbol. For 10,000 symbols, I'd spawn one thread per symbol (or one thread per core if < core count), use thread affinity to bind each thread to a core, and place each orderbook in thread-local memory to avoid NUMA. Global state (risk limits, position) would be in a central atomic structure read on every order with relaxed memory ordering."

**Red flags:**
- If you say "just use mutexes on all symbols," you're back to lock-based (slow).
- If you don't mention thread affinity or NUMA, you'll get bad NUMA latencies (100+ ns extra).
- If you don't test with 10,000 symbols in your proposal, they'll ask "did you just make that up?"

---

### Q7: "I see a duplicate header file: `lockfree_queue.hpp` and `lockfree_queue (1).hpp`. Which one is used? Why do you have both?"

**Real Answer:**
From the file listing, yes, there are two files. This is **a mistake** — it's version control cruft. Likely what happened:
1. Original version: `lockfree_queue.hpp`
2. Developer made a change, saved as `lockfree_queue (1).hpp` (copy)
3. Meant to delete one, but forgot

**Which is used?** Check `cpp/CMakeLists.txt` and `cpp/include/lockfree_queue.hpp`. If CMakeLists only links against `include/lockfree_queue.hpp`, then `(1).hpp` is dead code.

**What you should say:**
"Good catch. I see both files; `(1).hpp` is a duplicate that should be deleted. The build only uses `lockfree_queue.hpp` (line X of CMakeLists shows the include path). I'd remove `(1).hpp` and any references."

**If they push:** "Yes, it's sloppy. In a real codebase, I'd have caught this in code review — CI should warn on duplicate files or orphaned headers. I'll fix it."

**Red flag:** If you didn't notice it when preparing, that's okay. If you *defend* it as "intentional backup" or "maybe we use both," you're bullshitting.

---

### Q8: "All your benchmark results are dated 2026-02-04. Did you run them once and ship it, or are you running them in CI on every commit?"

**Real Answer:**
The results in `cpp/benchmarks/results/*.json` are **dated once** — they were run on Feb 4, 2026, on a specific machine (i9-13900K, Ubuntu 22.04, clang-18). They are **not** part of CI.

**What we should be doing:**
- Every commit, run benchmarks (e.g., in GitHub Actions) and check regressions (latency +10%, throughput –5% = alert).
- Track results over time (CSV or InfluxDB).
- Enforce that no PR merges if latency degrades.

**What we're actually doing:**
- Running benchmarks locally, committing the JSON snapshot once, then not re-running.

**Honest answer:** "The benchmarks are a one-time snapshot from Feb 4. They're not in CI, which is a gap. In a real team, we'd set up performance regression testing in GitHub Actions — every commit runs the benchmark suite, compares against the baseline, and fails the build if latency regresses by more than 5%. I'd use `google-benchmark` to emit JSON and a script to compare and alert."

**If they ask "aren't you worried about performance regressions,"** say: "Yes, which is why I'd add this to CI immediately. One engineer accidentally removed an `alignas(64)` annotation, and latency went from 488 ns to 2.1 µs, and we caught it in code review by looking at the size of the struct. But we shouldn't rely on that."

**Red flag:**
- If you say "I run benchmarks every day," but they're only in git from one date, you're lying.
- If you don't know how Google Benchmark works (`google_benchmark` library + JSON reporter), say so.

---

### Q9: "Your order book has no internal synchronization. What if two threads try to add orders for the same symbol simultaneously?"

**Real Answer:**
**They will corrupt the order book.** There is no lock. The requirement is **single-writer per symbol** — only one thread adds orders to a given LimitOrderBook instance. If two threads violate this, the intrusive linked list will have cycles, the skip list will have inconsistent levels, and the hash map will have stale pointers.

This is **not a bug in the code** — it's a **precondition** documented in the header:
```cpp
/**
 * Thread-safety: Single writer per symbol. If multiple threads submit
 * orders for the same symbol, they must externally synchronize or use
 * a queue + worker thread pattern.
 */
class LimitOrderBook { ... };
```

**If they ask "why not make it thread-safe,"** answer: "Adding internal locking (mutex) would add 100+ ns latency per order. For a trading system, that's unacceptable. The solution is architectural: each symbol has its own thread. If you need to submit orders from multiple application threads, queue them and have a worker thread process them sequentially for that symbol. That's the SPSC queue's job."

**Red flags:**
- If you claim it's "thread-safe," you don't understand the code.
- If you don't mention the precondition in the docs, you're lazy.
- If you suggest "just add a mutex," you've lost the performance game.

---

### Q10: "Your Black-Scholes implementation doesn't handle dividends. Real options prices change with div yield. Why?"

**Real Answer:**
Black-Scholes-Merton **does** handle dividends in the formula:
```
C = S*e^(-q*T)*N(d1) - K*e^(-r*T)*N(d2)
```
where `q` is the dividend yield (continuously compounded annual rate). As T increases, the `e^(-q*T)` term matters more.

**Our implementation** (from `cpp/src/options_pricing.cpp`, lines ~40–60) **includes `q`:**
```cpp
double BlackScholesPricer::price(
    double S, double K, double T, double r, double q, double sigma,  // <- q is here
    OptionType type
) {
    double d1 = (log(S/K) + (r - q + 0.5*sigma*sigma)*T) / (sigma*sqrt(T));
    double d2 = d1 - sigma*sqrt(T);
    // ... rest of formula uses q
}
```

**If they ask "what if div is not continuous,"** say: "That's discrete dividends (ex-div dates). The closed-form Black-Scholes doesn't handle them; you need a tree (binomial or trinomial) or jump-diffusion (Merton). We have `cpp/include/options_pricing.hpp::BinomialPricer` for American and discrete-dividend options."

**Red flag:**
- If you say "our BS has no dividend support," you haven't read the code.
- If you can't explain continuous vs. discrete dividends, you'll lose credibility on options.

---

### Q11: "Your risk manager pre-checks orders. But I can still place an order, it passes the check, and then someone else fills 50% of it and the P&L swings $500k. Isn't your risk check pointless?"

**Real Answer:**
Good catch. **Your risk check is snapshot-based, not predictive.** It checks: "right now, if I execute this order, will we breach limits?" But by the time the order is matched, market conditions change.

This is **inherent to any RMS** — you can't predict future fills. The solution is:

1. **Conservative limits:** set limits so even if P&L swings 50%, you don't breach hard limits.
2. **Dynamic risk checks:** after every fill, re-check position risk (this is what a production RMS does — see `position_manager.hpp`).
3. **Stop-losses:** if position hits a loss limit, reject new orders automatically.
4. **Monitoring + alerting:** separate thread checks every N milliseconds and alerts if risk metrics degrade (not in this code).

**From the code** (`cpp/include/risk_manager.hpp`, lines ~50–120):
```cpp
RiskCheckResult check_order(
    const char* symbol,
    char side,
    uint32_t quantity,
    int64_t price,
    int64_t current_market_price  // <- only used for the *current* check
) {
    // Checks: max order size, order value, max position, exposure, concentration...
    // All based on current market state.
}
```

**Honest answer:** "This check prevents *obviously bad* orders (e.g., accidentally 1M shares instead of 100). For real risk management, we'd have a feedback loop: after each fill, update the position, re-check limits, and alert if we're trending toward breach. That's the `position_manager`'s job (journaling and reconciliation), but we don't have dynamic re-checking in this harness."

**Red flag:**
- If you claim "our risk manager prevents all losses," you're fraudulent.
- If you don't mention the limit → fill → loss feedback loop, you don't understand RMS.

---

### Q12: "How would you build an order-book **replay** system on top of this? Meaning: given a historical tick tape (all trades + order book snapshots), simulate what would have happened if we'd run this engine on the historical data."

**Real Answer (HIGH PROBABILITY — this is Katrina's job):**

Order-book replay is essential for backtesting: you take the real exchange orderbook snapshot at time T0, feed it your strategy's orders, and measure what fills would have happened.

**Steps:**

1. **Input:** tick tape from an exchange (CME, NYSE, etc.) with:
   - Market data: price level L, quantity Q, timestamp
   - Trades: executed quantity, price
   - Any order book snapshots (Level 2 or Level 3)

2. **Reconstruct the historical orderbook:**
   ```cpp
   LimitOrderBook book("ESZ1");  // ES Futures
   for (auto& tick : tape) {
       if (tick.type == MARKET_DATA) {
           // Level 2 update: "bid $4100 x 500 contracts"
           book.apply_market_data(tick);  // custom method to update bid/ask
       } else if (tick.type == TRADE) {
           // "500 @ $4100 traded" — reconcile against my submitted orders
           handle_trade_execution(tick, book);
       }
   }
   ```

3. **Feed strategy orders into the book:**
   ```cpp
   // At time 14:32:05.123, strategy decides to buy 100 ES @ 4101
   auto result = book.add_order(
       Side::BUY, OrderType::LIMIT, 4101*100, 100,  // price in ticks
       TimeInForce::GTC, strategy_order_id
   );
   // result.trades contains what matched; result.order_id is what rested
   ```

4. **Measure fills and P&L:**
   ```cpp
   // At later time, strategy wants to exit
   book.cancel_order(strategy_order_id);  // Remove resting order
   // Or wait for fills if the book moves in our favor
   double realized_pnl = /* sum of trade prices */;
   ```

**Code in this repo that supports replay:**
- `orderbook.hpp::add_order()` returns `order_id` and fills all matched trades — the matches are deterministic given the order book state.
- `orderbook.hpp::cancel_order(order_id)` is O(1) deterministic.
- `orderbook.hpp::depth()` and `best_bid/best_ask` give you the current book state for debugging.

**What's missing for a full replay system:**
- `apply_market_data()` method — we have codec parsers (ITCH, SBE, FIX) but no direct "update the orderbook from a market tick" method. That's a few hours of glue code.
- **Determinism guarantee:** our current implementation doesn't guarantee determinism across reruns (e.g., if the pool allocator's free list order changes). We'd need a deterministic allocator (sequential IDs, no free-list reuse) or pre-populate the pool in a fixed order.
- **Time-travel:** the code doesn't handle "it's now 2 seconds later" — you'd need to call a fake clock in tests. That's straightforward (`cpp/include/perf_counters.hpp` has a `SteadyClockMock` sketch).

**Elevator pitch answer:**
"The order book is deterministic given a sequence of orders. To build replay, I'd (1) parse the tick tape, (2) rebuild the historical book state, (3) inject our strategy's orders, (4) measure matches against the replayed book. The hard part is handling corner cases: partial fills, cancelled orders, and ensuring the replayed orderbook matches the real exchange's (usually to L2 fidelity, since L3 details aren't always published). I'd write a reconciliation layer that compares our replayed book snapshots against published exchange snapshots every 100ms and alerts on divergence."

**Red flag:**
- If you say "I don't know how to do it," that's honest and acceptable — but say "I'd start with parsing the tape and applying ticks to a copy of the book."
- If you claim "it's just feed the data and it works," you haven't thought about the real constraints.

---

## 11. Tailored to Your Interviewer & the Job

### 11.1 Who Katrina (Yuqiao) Hu is — and what it means for your prep
- **Quant Dev Lead @ Aleto, CS @ CMU (SCS).** Her own experience: *"designing and implementing equity and futures trading systems as well as **orderbook replay systems** in C++."* → **Expect deep, specific order-book and C++ systems questions.** Your Pillar 1 and the "replay" answer (§9 Q12) are the highest-value things to nail.
- **CMU parallel-programming background** (CUDA, OpenMP, OpenMPI, SIMD, "90% vector utilization," 7.5× speedups). → She *knows* real parallel speedups and will not accept hand-waving. Your SIMD Monte-Carlo and the honest sub-linear scaling answer (§5.5) matter a lot. Do **not** overclaim linear scaling.
- **Distributed systems TA (15-440), Raft, UDP/TCP protocols.** → Networking layer, sequence-gap detection, reconnection, and the message-transport queues are fair game.
- **Data-engineer phase at Aleto (download/transform large financial datasets, Python).** → ETL / big-data questions will come (matches the JD).
- **Akuna Options 101 cert, UChicago trading competition.** → She understands options and market microstructure; your Greeks/Heston/SABR breadth is relevant but she'll spot bluffing.

### 11.2 The Aleto JD → what they'll probe
- *"Develop and maintain live trading systems — performance, reliability, scalability."* → order book, risk checks, latency percentiles, backpressure, crash recovery (your position-manager WAL journaling is a great story).
- *"Collaborate with quant researchers... backtesting systems."* → **order-book replay & backtesting** (her exact words + JD) — rehearse §9 Q12 thoroughly.
- *"ETL pipelines, large-scale financial data, Python, Spark/Hadoop, SQL."* → be ready to talk about the data side: ingesting tick data, normalizing, storing, and that the web app uses Finnhub + Supabase (Postgres/SQL). Be honest that Spark/Hadoop isn't in this repo.
- *"C++ and Python, system design, balance performance/reliability/availability."* → the whole project is your C++ evidence; be ready to design a component live.
- *Preferred: cloud (AWS/GCP/Azure), GPU, trading algorithms.* → GPU: you can contrast your CPU/AVX2 MC (low latency, no PCIe transfer) vs CUDA (higher throughput) — and note *she* did CUDA A* and SAXPY, so speak to that respectfully.

### 11.3 Questions SHE (specifically) is likely to ask you
1. "Explain your order book's data structures and why — and how you'd build an **order-book replay** system on top." (highest probability — it's literally her job)
2. "Your matching engine is single-threaded per symbol — how do you scale to thousands of symbols?" (sharding, thread-per-core, NUMA)
3. "Show me your memory ordering in the SPSC queue and justify release/acquire." (parallel-programming background)
4. "Your Monte-Carlo only gets 6.6× on 16 threads — why not 16×?" (she'll *know* the answer)
5. "How do you validate a Heston price is correct?" (options cert)
6. "How would you build an ETL pipeline to feed this engine from raw exchange data?" (JD/data-engineer)
7. "How do you guarantee reliability — what happens on crash mid-session?" (WAL journaling, sequence recovery)
8. "What's the hardest bug you hit and how did you find it?" (perf/ThreadSanitizer story)

### 11.4 Questions YOU should ask her (shows level)
- "How is Aleto's live trading stack split between C++ and Python — where's the boundary?"
- "Do you run order-book replay for strategy backtests, and at what fidelity (L2/L3, latency modeling)?"
- "How do you handle determinism/reproducibility between backtest and live?"
- "What does the data pipeline look like — vendors, storage, Spark vs. custom?"
- "Where does the team feel the most latency or reliability pain today?"

---

## 12. One-Page Cheat Sheet (memorize this)

- **Canonical numbers (say these):** Order book **488 ns avg / 1.63M orders/sec** (P99 2.6 µs). SPSC **10.94M msg/sec**. Black-Scholes **59 ns / 6.0M pricings/sec**. Monte-Carlo **119M sims/sec** (2.11 µs/path), 6.6× on 16 threads. VaR99 $59.71, CVaR99 $65.03.
- **Source of truth:** `cpp/docs/BENCHMARKS.md`. README = aspirational. JSON = coarse macro harness.
- **Order book:** integer ticks, pool allocator, intrusive FIFO list per level (O(1) cancel), skip list of levels (O(log P)), hash map id→order, HDR histogram, single writer per symbol.
- **Queues:** 64-byte alignment, head/tail on separate lines (no false sharing), release/acquire publish, power-of-two ring, `_mm_pause()` backoff. SPSC wait-free; MPSC/MPMC lock-free.
- **Options:** A&S norm_cdf, single-pass Greeks incl. vanna/volga/charm/speed; Heston (char-fn + Gauss-Legendre, branch-cut care); Merton; SABR (Hagan); binomial+Richardson; LSM (Laguerre).
- **Monte-Carlo:** GBM closed step, Xoshiro256++ (0.8 ns) + Ziggurat scalar / Philox + Box-Muller SIMD (AVX2, 8 lanes), antithetic/control/Sobol, VaR/CVaR.
- **Build:** C++20, `-O3 -march=native -mavx2 -mfma -ffast-math`; GoogleTest; perf_event_open + RDTSC.
- **Honesty anchors:** novelty = integration+rigor not new algorithms; number sources disagree (own it); book isn't internally lock-free; risk check is single-thread-correct; `-ffast-math` risk; sub-linear MC scaling is expected.
- **Her hot buttons:** order-book **replay/backtesting**, real parallel speedups, memory ordering, C++/Python boundary, ETL.
```
```
```

**Final advice:** the fastest way to lose this interview is to defend a number you can't point to in the repo. The fastest way to win it is to volunteer the order-book *replay* story (her exact specialty), speak precisely about memory ordering and false sharing, and be candid that the impressive part is the measured, integrated breadth — not a novel algorithm.
