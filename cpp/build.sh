#!/bin/bash
# Build script for ARBOR C++ Quantitative Engine

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║   Building ARBOR C++ Engine - Production Configuration   ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Create build directory
mkdir -p build
cd build

# Configure with aggressive optimizations
echo "🔧 Configuring CMake with Release optimizations..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native -flto -ffast-math" \
    -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON

# Build
echo ""
echo "🔨 Building C++ components..."
cmake --build . --config Release -j$(nproc)

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Build successful!"
    echo ""
    echo "📊 Run benchmarks with:"
    echo "   ./orderbook_bench"
    echo "   ./options_bench"
    echo "   ./montecarlo_bench"
    echo ""
    echo "🎯 Expected performance:"
    echo "   • Order book matching: < 10 μs"
    echo "   • Options pricing: < 1 μs"
    echo "   • Monte Carlo: 10,000+ paths/sec"
    echo ""
else
    echo "❌ Build failed. Check compiler errors above."
    exit 1
fi
