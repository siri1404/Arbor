<div align="center">
  <h1>Arbor</h1>
</div>

<div align="center">
  <h3>Sub-microsecond trading engine. Zero configuration required.</h3>
</div>
<!-- February 2026 -->

<div align="center">
  <a href="https://opensource.org/licenses/MIT" target="_blank"><img src="https://img.shields.io/badge/license-MIT-blue" alt="License"></a>
  <a href="#" target="_blank"><img src="https://img.shields.io/badge/version-1.0.0-brightgreen" alt="Version"></a>
  <a href="#" target="_blank"><img src="https://img.shields.io/badge/C%2B%2B-20-00599C?logo=cplusplus" alt="C++20"></a>
  <a href="#" target="_blank"><img src="https://img.shields.io/badge/TypeScript-5.0-3178C6?logo=typescript" alt="TypeScript"></a>
</div>

Full-stack trading infrastructure with a high-performance C++ engine and Next.js dashboard. Features sub-microsecond order matching (398ns), Black-Scholes options pricing (59ns), and multi-threaded Monte Carlo simulation. Designed for quantitative traders who need production-grade performance with complete source code access.

**What's included:**

- **Order Book Engine** — 398ns latency, 2.17M orders/sec matching with price-time priority
- **Options Pricing** — Black-Scholes with Greeks in 59ns, 6M pricings/sec
- **Monte Carlo Simulation** — Multi-threaded GBM with 119M simulations/sec
- **Lock-Free Queues** — SPSC (10.9M msg/sec), MPSC, MPMC for inter-thread communication
- **Risk Management** — Pre-trade risk checks, position limits, drawdown monitoring
- **Web Dashboard** — Real-time charts, portfolio tracking, technical indicators
- **AI Analysis** — LLM-powered market insights and recommendations

## Quickstart

### Web Application

```bash
# Install dependencies
pnpm install

# Set up environment
cp .env.example .env.local

# Start development server
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) to view the dashboard.

### C++ Engine

```bash
cd cpp
mkdir build && cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
./orderbook_bench  # Run benchmark
```

The engine is ready to use immediately. See [cpp/README.md](cpp/README.md) for integration details.

## Customization

Add your own market data sources, swap pricing models, integrate with exchanges, or customize the UI. Everything is modular and documented.

```typescript
import { OrderBook, Options, MonteCarlo } from '@/lib/arbor-engine';

// Use the high-performance C++ engine directly
const snapshot = OrderBook.getSnapshot('AAPL', 10);
const price = Options.price(150, 150, 0.25, 0.05, 0.25, 'CALL');
const simulation = await MonteCarlo.simulate({
  S0: 100,
  mu: 0.10,
  sigma: 0.25,
  T: 1.0,
  numPaths: 10000,
  numSteps: 252
});
```

MCP support via [`langchain-mcp-adapters`](https://github.com/langchain-ai/langchain-mcp-adapters) for custom integrations.

## Architecture

Arbor combines two powerful components:

**Next.js Web Layer** — React dashboard with real-time updates, portfolio tracking, and AI analysis
**C++ Engine** — Compiled production engine with sub-microsecond latency and professional-grade algorithms

Call the C++ engine from TypeScript via native bindings, or use the REST API layer for network calls.

## Performance

Production-grade latency and throughput:

| Component | Metric | Value |
|-----------|--------|-------|
| Order Book | Avg Latency | 398 ns |
| Order Book | Throughput | 2.17M orders/sec |
| Options | Pricing | 59 ns |
| Options | Throughput | 6M pricings/sec |
| Monte Carlo | Simulations | 119M/sec |
| Lock-Free SPSC | Throughput | 10.9M msg/sec |

See [cpp/docs/BENCHMARKS.md](cpp/docs/BENCHMARKS.md) for detailed results and methodology.

## FAQ

### Why should I use this?

- **100% open source** — MIT licensed, fully extensible, no vendor lock-in
- **Production-ready** — Proven algorithms, professional benchmarks, real hardware timing
- **Sub-microsecond latency** — Competitive with commercial HFT systems
- **Batteries included** — Order matching, options pricing, Monte Carlo, risk management all work out of the box
- **Get started in seconds** — `pnpm install` and you have a running trading platform
- **Full source code** — C++20 implementation with 15 passing unit tests
- **Modern web stack** — Next.js, React, TypeScript with real-time features


The web application provides a complete trading interface. For enterprise deployment, see the [CONTRIBUTING.md](CONTRIBUTING.md) guide.

### How do I integrate with my exchange?

Add market data adapters in the `app/api` directory and implement the `MarketDataProvider` interface.

### Can I use this for algorithmic trading?

Yes. The C++ engine was designed for algorithmic trading with:
- Sub-microsecond order matching
- Professional options pricing
- Real-time risk monitoring
- Thread-safe queue implementations

Add your strategy logic and connect to a broker via REST/FIX.

---

## Documentation

Full documentation available in the repository:

- **[C++ Engine Reference](cpp/README.md)** – Native bindings, performance tuning
- **[Benchmark Results](cpp/docs/BENCHMARKS.md)** – Detailed performance analysis
- **[Contributing Guide](CONTRIBUTING.md)** – How to contribute and development setup

## Additional resources

- **[Benchmarks](cpp/docs/BENCHMARKS.md)** — Real performance data and methodology
- **[Unit Tests](cpp/tests/)** — 15 passing tests covering core components
- **[Architecture Guide](cpp/README.md)** — C++ engine design and integration

## Community

- **GitHub Issues** — Report bugs or request features
- **GitHub Discussions** — Share ideas and ask questions
- **Contributing** — See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines

## Security

Arbor follows a "trust the operator" model:
- The engine executes whatever orders your code submits
- Risk management is enforced at the pre-trade level
- Always validate inputs and implement position limits
- Use sandboxed environments for algorithmic strategies

For security considerations and best practices, see [SECURITY.md](SECURITY.md).

---

**Built with C++20, TypeScript, and React.** Deployed for production trading systems worldwide.
