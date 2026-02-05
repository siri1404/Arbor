#!/usr/bin/env python3
"""
Benchmark result analyzer and comparison tool for Arbor quantitative engine.

Usage:
    python3 analyze_benchmarks.py                    # Analyze current results
    python3 analyze_benchmarks.py --compare old.json # Compare with baseline
    python3 analyze_benchmarks.py --trend             # Show performance trends
    python3 analyze_benchmarks.py --html             # Generate HTML report
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import argparse


class BenchmarkAnalyzer:
    """Analyze and compare benchmark results."""
    
    def __init__(self, results_file: str):
        self.results_file = Path(results_file)
        self.data = self._load_json()
    
    def _load_json(self) -> dict:
        """Load benchmark JSON file."""
        with open(self.results_file) as f:
            return json.load(f)
    
    def get_context(self) -> dict:
        """Get benchmark context (date, platform, build type)."""
        return self.data.get('context', {})
    
    def get_benchmarks(self) -> List[dict]:
        """Get list of benchmarks."""
        return self.data.get('benchmarks', [])
    
    def print_summary(self) -> None:
        """Print benchmark summary table."""
        ctx = self.get_context()
        benchmarks = self.get_benchmarks()
        
        print(f"\n📊 Benchmark Results: {Path(self.results_file).stem}")
        print(f"Date: {ctx.get('date', 'N/A')}")
        print(f"Platform: {ctx.get('host_name', 'N/A')}")
        print(f"Build: {ctx.get('build_type', 'N/A')}")
        print(f"Tests: {len(benchmarks)}\n")
        
        print(f"{'Benchmark':<50} {'Time':<15} {'Throughput':<15}")
        print("-" * 80)
        
        for bench in benchmarks:
            name = bench['name'].split('/')[-1][:47]
            time_val = bench.get('real_time', bench.get('cpu_time', 0))
            time_unit = bench.get('time_unit', 'us')
            throughput = bench.get('items_per_second', 0)
            
            time_str = f"{time_val:.2f} {time_unit}"
            throughput_str = f"{throughput:.0f} ops/s" if throughput > 0 else "N/A"
            
            print(f"{name:<50} {time_str:<15} {throughput_str:<15}")
    
    def compare_with_baseline(self, baseline_file: str, threshold: float = 0.10) -> Tuple[List, List]:
        """Compare with baseline results.
        
        Args:
            baseline_file: Path to baseline benchmark JSON
            threshold: Regression threshold (0.10 = 10%)
        
        Returns:
            Tuple of (regressions, improvements) lists
        """
        baseline = BenchmarkAnalyzer(baseline_file)
        
        current_map = {b['name']: b for b in self.get_benchmarks()}
        baseline_map = {b['name']: b for b in baseline.get_benchmarks()}
        
        regressions = []
        improvements = []
        
        for name, curr_bench in current_map.items():
            if name not in baseline_map:
                continue
            
            base_bench = baseline_map[name]
            curr_time = curr_bench.get('real_time', curr_bench.get('cpu_time', 0))
            base_time = base_bench.get('real_time', base_bench.get('cpu_time', 0))
            
            if base_time <= 0:
                continue
            
            change = (curr_time - base_time) / base_time
            
            if change > threshold:
                regressions.append((name, change * 100, base_time, curr_time))
            elif change < -threshold:
                improvements.append((name, abs(change) * 100, base_time, curr_time))
        
        return regressions, improvements
    
    def print_comparison(self, baseline_file: str, threshold: float = 0.10) -> int:
        """Print comparison with baseline."""
        print(f"\n📈 Comparing with baseline: {Path(baseline_file).name}\n")
        
        regressions, improvements = self.compare_with_baseline(baseline_file, threshold)
        
        if regressions:
            print(f"⚠️  PERFORMANCE REGRESSIONS (>{threshold*100:.0f}%):")
            print(f"{'Benchmark':<50} {'Change':<10} {'Old':<12} {'New':<12}")
            print("-" * 84)
            for name, pct, old, new in sorted(regressions, key=lambda x: -x[1]):
                short_name = name.split('/')[-1][:45]
                old_time = f"{old:.2f} µs"
                new_time = f"{new:.2f} µs"
                print(f"{short_name:<50} +{pct:>6.1f}%  {old_time:<12} {new_time:<12}")
            print()
        
        if improvements:
            print(f"✅ PERFORMANCE IMPROVEMENTS (>{threshold*100:.0f}%):")
            print(f"{'Benchmark':<50} {'Change':<10} {'Old':<12} {'New':<12}")
            print("-" * 84)
            for name, pct, old, new in sorted(improvements, key=lambda x: -x[1]):
                short_name = name.split('/')[-1][:45]
                old_time = f"{old:.2f} µs"
                new_time = f"{new:.2f} µs"
                print(f"{short_name:<50} -{pct:>6.1f}%  {old_time:<12} {new_time:<12}")
            print()
        
        if not regressions and not improvements:
            print(f"✅ No significant changes detected (threshold: {threshold*100:.0f}%)\n")
        
        return 1 if regressions else 0
    
    def generate_markdown_report(self, output_file: str = "benchmark_report.md") -> None:
        """Generate markdown report."""
        ctx = self.get_context()
        benchmarks = self.get_benchmarks()
        
        md = f"""# Benchmark Report

**Date:** {ctx.get('date', 'N/A')}  
**Platform:** {ctx.get('host_name', 'N/A')}  
**Build:** {ctx.get('build_type', 'N/A')}  
**Tests:** {len(benchmarks)}

## Results

| Benchmark | Time | Unit | Throughput |
|-----------|------|------|-----------|
"""
        
        for bench in benchmarks:
            name = bench['name'].split('/')[-1]
            time_val = bench.get('real_time', bench.get('cpu_time', 0))
            time_unit = bench.get('time_unit', 'us')
            throughput = bench.get('items_per_second', 0)
            throughput_str = f"{throughput:.0f} ops/s" if throughput > 0 else "N/A"
            md += f"| {name} | {time_val:.2f} | {time_unit} | {throughput_str} |\n"
        
        with open(output_file, 'w') as f:
            f.write(md)
        
        print(f"✅ Markdown report written to {output_file}")
    
    def generate_html_report(self, output_file: str = "benchmark_report.html") -> None:
        """Generate HTML report with charts."""
        ctx = self.get_context()
        benchmarks = self.get_benchmarks()
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Arbor Benchmark Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .metadata {{ background: #f5f5f5; padding: 10px; border-radius: 5px; margin: 20px 0; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
    </style>
</head>
<body>
    <h1>Arbor Benchmark Report</h1>
    <div class="metadata">
        <p><strong>Date:</strong> {ctx.get('date', 'N/A')}</p>
        <p><strong>Platform:</strong> {ctx.get('host_name', 'N/A')}</p>
        <p><strong>Build:</strong> {ctx.get('build_type', 'N/A')}</p>
    </div>
    
    <h2>Results</h2>
    <table>
        <tr>
            <th>Benchmark</th>
            <th>Time (µs)</th>
            <th>Iterations</th>
            <th>Throughput (ops/s)</th>
        </tr>
"""
        
        for bench in benchmarks:
            name = bench['name'].split('/')[-1]
            time_val = bench.get('real_time', bench.get('cpu_time', 0))
            iterations = bench.get('iterations', 0)
            throughput = bench.get('items_per_second', 0)
            throughput_str = f"{throughput:.0f}" if throughput > 0 else "N/A"
            
            html += f"""        <tr>
            <td>{name}</td>
            <td>{time_val:.2f}</td>
            <td>{iterations}</td>
            <td>{throughput_str}</td>
        </tr>
"""
        
        html += """    </table>
</body>
</html>
"""
        
        with open(output_file, 'w') as f:
            f.write(html)
        
        print(f"✅ HTML report written to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Arbor benchmark results")
    parser.add_argument("results_file", nargs='?', default="results.json",
                        help="Benchmark results JSON file")
    parser.add_argument("--compare", "-c", help="Compare with baseline JSON file")
    parser.add_argument("--threshold", "-t", type=float, default=0.10,
                        help="Regression threshold (default: 0.10 = 10%%)")
    parser.add_argument("--markdown", "-md", action="store_true",
                        help="Generate markdown report")
    parser.add_argument("--html", action="store_true",
                        help="Generate HTML report")
    
    args = parser.parse_args()
    
    if not Path(args.results_file).exists():
        print(f"❌ File not found: {args.results_file}")
        sys.exit(1)
    
    analyzer = BenchmarkAnalyzer(args.results_file)
    
    # Print summary
    analyzer.print_summary()
    
    # Compare if baseline provided
    if args.compare:
        exit_code = analyzer.print_comparison(args.compare, args.threshold)
        if exit_code:
            sys.exit(1)
    
    # Generate reports
    if args.markdown:
        analyzer.generate_markdown_report()
    
    if args.html:
        analyzer.generate_html_report()


if __name__ == "__main__":
    main()
