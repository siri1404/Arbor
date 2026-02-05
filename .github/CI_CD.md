# CI/CD Pipeline Documentation

## Overview

Arbor uses GitHub Actions for continuous integration and deployment. The CI/CD pipeline automatically runs on every commit and pull request to ensure code quality, catch performance regressions, and maintain system reliability.

## Workflows

### 1. C++ CI/CD Pipeline (`.github/workflows/cpp-ci.yml`)

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches
- Changes to `cpp/**` directory

**Jobs:**

#### Build & Test Matrix
- **Platforms**: Ubuntu, Windows, macOS
- **Build Types**: Release, Debug
- **Steps**:
  - Install dependencies (CMake, Ninja, TBB)
  - Configure CMake
  - Build all targets
  - Run unit tests with CTest

#### Performance Benchmarks
- **Platform**: Ubuntu (Release build)
- **Benchmarks**:
  - Order Book performance
  - Options Pricing algorithms
  - Lock-Free data structures
  - Monte Carlo simulations
  - Production scenarios
- **Features**:
  - JSON output for analysis
  - Automatic comparison with baseline
  - Regression detection (>10% slowdown)
  - Performance improvement tracking
  - Artifact storage (30 days)
  - Baseline storage on main branch (90 days)

#### Sanitizers
- **Address Sanitizer**: Memory errors, leaks, buffer overflows
- **Thread Sanitizer**: Data races, deadlocks
- **Undefined Behavior Sanitizer**: Integer overflow, null derefs

#### Code Quality
- **clang-format**: Code style enforcement
- **cppcheck**: Static analysis
- **clang-tidy**: Modern C++ best practices (future)

#### Performance Report
- Generates markdown tables with benchmark results
- Posts automated comments on PRs
- Includes throughput calculations

### 2. Node.js CI/CD Pipeline (`.github/workflows/node-ci.yml`)

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches
- Changes to frontend code

**Jobs:**

#### Build & Lint
- pnpm dependency installation
- TypeScript type checking
- Linting
- Next.js build

#### Tests
- Unit tests
- Integration tests
- Component tests

#### Deploy Preview
- Automatic preview deployments for PRs
- Build verification
- PR comments with deployment status

## Benchmark Regression Detection

### How It Works

1. **Baseline Storage**: Benchmarks from the `main` branch are stored as the baseline
2. **Comparison**: Each PR/push compares current benchmarks against the baseline
3. **Threshold**: Regressions >10% slower trigger warnings
4. **Reporting**: Detailed reports show both regressions and improvements

### Interpreting Results

```
⚠️  PERFORMANCE REGRESSIONS in orderbook_results.json:
  - OrderBook/Insert/1000: +15.3% slower

✅ PERFORMANCE IMPROVEMENTS in options_results.json:
  - BlackScholes/Call: 8.2% faster
```

### What to Do

**If Regressions Detected:**
1. Review your changes for inefficiencies
2. Profile the affected code path
3. Check for unintended algorithmic changes
4. Verify compiler optimization flags
5. Consider if the regression is acceptable for the feature

**Acceptable Regressions:**
- Adding essential features may slightly reduce performance
- Document trade-offs in PR description
- Consider optimization in follow-up PR

## Running Locally

### C++ Benchmarks

```bash
cd cpp
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
cd build

# Run individual benchmarks
./orderbook_benchmark
./options_benchmark
./lockfree_benchmark
./montecarlo_benchmark
./production_benchmark

# JSON output for comparison
./orderbook_benchmark --benchmark_format=json --benchmark_out=results.json
```

### Sanitizers

```bash
# Address Sanitizer
CC=clang CXX=clang++ cmake -B build-asan -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer -g"
cmake --build build-asan
cd build-asan && ctest

# Thread Sanitizer
CC=clang CXX=clang++ cmake -B build-tsan -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=thread -fno-omit-frame-pointer -g"
cmake --build build-tsan
cd build-tsan && ctest

# Undefined Behavior Sanitizer
CC=clang CXX=clang++ cmake -B build-ubsan -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=undefined -fno-omit-frame-pointer -g"
cmake --build build-ubsan
cd build-ubsan && ctest
```

### Code Quality

```bash
# Check formatting
find cpp/src cpp/include cpp/tests cpp/benchmarks -name "*.cpp" -o -name "*.hpp" | \
  xargs clang-format --dry-run --Werror

# Apply formatting
find cpp/src cpp/include cpp/tests cpp/benchmarks -name "*.cpp" -o -name "*.hpp" | \
  xargs clang-format -i

# Run cppcheck
cd cpp
cppcheck --enable=all --suppress=missingIncludeSystem src/ include/
```

## Best Practices

### For Contributors

1. **Run Benchmarks Locally**: Before pushing, run affected benchmarks
2. **Check Sanitizers**: Run at least address sanitizer on your changes
3. **Format Code**: Use clang-format before committing
4. **Document Performance Changes**: Explain expected performance impacts in PRs

### For Reviewers

1. **Review Benchmark Results**: Check the automated performance report
2. **Investigate Regressions**: Ask for explanation of significant slowdowns
3. **Verify Test Coverage**: Ensure new features have tests
4. **Check Sanitizer Results**: All sanitizer jobs should pass

## Artifacts

### Benchmark Results
- **Retention**: 30 days
- **Location**: Actions run → Artifacts → `benchmark-results`
- **Format**: JSON files for each benchmark suite

### Benchmark Baselines
- **Retention**: 90 days
- **Location**: Actions run → Artifacts → `benchmark-baseline`
- **Purpose**: Regression detection reference

### Build Artifacts
- **Retention**: 7 days
- **Location**: Actions run → Artifacts → `next-build`
- **Purpose**: Frontend build verification

## Troubleshooting

### Build Failures

**CMake Configuration Fails:**
- Check dependency installation
- Verify CMake version ≥3.14
- Review CMakeLists.txt for errors

**Compilation Errors:**
- Check for platform-specific issues
- Review matrix job logs by OS
- Verify C++20 compiler support

### Benchmark Failures

**Binary Not Found:**
- Ensure benchmarks are built in CMakeLists.txt
- Check target names match workflow

**Baseline Missing:**
- First run on main branch creates baseline
- Subsequent runs compare against it
- Baseline refreshes on main branch pushes

### Sanitizer Failures

**False Positives:**
- Review sanitizer documentation
- Add suppressions if needed
- Check third-party library issues

**Real Issues:**
- Fix memory leaks immediately
- Data races are critical bugs
- Undefined behavior must be addressed

## Future Enhancements

- [ ] Code coverage reporting (lcov/gcov)
- [ ] Clang-tidy integration
- [ ] Benchmark result visualization
- [ ] Historical performance tracking
- [ ] Automated performance bisection
- [ ] Deploy to staging/production
- [ ] Docker container builds
- [ ] Release automation
- [ ] Changelog generation

## References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Google Benchmark](https://github.com/google/benchmark)
- [AddressSanitizer](https://github.com/google/sanitizers/wiki/AddressSanitizer)
- [ThreadSanitizer](https://github.com/google/sanitizers/wiki/ThreadSanitizerCppManual)
