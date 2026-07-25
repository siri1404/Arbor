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

### What this means for you
- Your bullets (**59 ns / 6M, 119M sims/sec, 10.9M msg/sec**) come straight from **`docs/BENCHMARKS.md`**. The order-book bullet (398 ns / 2.17M) is *close to but not exactly* any file — BENCHMARKS says 488 ns/1.63M, README says 312 ns/2.8M. **398 ns and 2.17M is a number that is not currently reproduced by any committed result file.**
- **Decision to make before the interview:** treat `docs/BENCHMARKS.md` as your single source of truth, and either (a) update your résumé order-book numbers to 488 ns / 1.63M, or (b) re-run the benchmark and capture a fresh result you can defend. Do **not** quote README's 2.8M and BENCHMARKS' 488 ns in the same breath.
- The JSON files measure *different things* (batch inserts of 1,000 orders including allocation, full option objects, 100k-path MC runs), which is *why* they look slower. That's a legitimate explanation — "the JSON is a coarse macro-benchmark; the nanosecond figures are the isolated hot-path micro-benchmark with RDTSC" — but you must be able to say that calmly and point at the code.

### The honest framing if pushed hard
> "The headline nanosecond numbers are isolated hot-path micro-benchmarks — single operation, warm cache, RDTSC timing, `-O3 -march=native`. The JSON macro-benchmarks include allocation, book setup and larger batches, so they're deliberately lower and more end-to-end. On an Intel i9-13900K the isolated Black-Scholes pricing is ~59 ns; a full option-chain or a cold run is microseconds. I should tighten the repo so all three sources tell the same story."

Saying that shows maturity. Pretending 2.8M and 488 ns are both "the" number does not.

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
