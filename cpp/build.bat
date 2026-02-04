@echo off
REM Build script for Windows (MSVC)

echo ╔═══════════════════════════════════════════════════════════╗
echo ║   Building ARBOR C++ Engine - Windows MSVC               ║
echo ╚═══════════════════════════════════════════════════════════╝
echo.

REM Create build directory
if not exist build mkdir build
cd build

REM Configure for Visual Studio 2022
echo 🔧 Configuring CMake for Visual Studio...
cmake .. -G "Visual Studio 17 2022" -A x64

if errorlevel 1 (
    echo ❌ CMake configuration failed
    exit /b 1
)

REM Build in Release mode
echo.
echo 🔨 Building C++ components in Release mode...
cmake --build . --config Release --parallel

if errorlevel 1 (
    echo ❌ Build failed
    exit /b 1
)

echo.
echo ✅ Build successful!
echo.
echo 📊 Run benchmarks with:
echo    Release\orderbook_bench.exe
echo    Release\options_bench.exe
echo    Release\montecarlo_bench.exe
echo.
echo 🎯 Expected performance on modern CPU:
echo    • Order book matching: ^< 10 μs
echo    • Options pricing: ^< 1 μs  
echo    • Monte Carlo: 10,000+ paths/sec
echo.

cd ..
