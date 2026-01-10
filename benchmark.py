import time
import tracemalloc
import statistics
from typing import Dict, Callable, Any


def measure_algorithm(
    algorithm_func: Callable,
    stocks_metrics: Dict,
    algorithm_name: str,
    num_runs: int = 5,
    **algo_kwargs
) -> Dict[str, Any]:
    """
    Measure time, memory, and solution quality for an algorithm.
    """
    execution_times = []
    memory_usages = []
    results_list = []
    
    for run in range(num_runs):
        # Start memory tracking
        tracemalloc.start()
        
        # Start time measurement
        start_time = time.perf_counter()
        
        # Run algorithm
        allocations, results = algorithm_func(
            stocks_metrics, 
            display_results=False,
            **algo_kwargs
        )
        
        # End time measurement
        end_time = time.perf_counter()
        
        # Get peak memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Store measurements (time in ms, memory in BYTES now)
        elapsed_ms = (end_time - start_time) * 1000
        execution_times.append(elapsed_ms)
        memory_usages.append(peak)
        results_list.append(results)

        # Per-run progress log
        print(
            f"    Run {run + 1}/{num_runs}: "
            f"time={elapsed_ms:.2f} ms, "
            f"peak_memory={peak} bytes"
        )
    
    # Calculate statistics
    avg_result = results_list[0]  # Results should be consistent
    
    return {
        # Performance metrics
        "algorithm": algorithm_name,
        "execution_time_ms": statistics.mean(execution_times),
        "execution_time_std": statistics.stdev(execution_times) if num_runs > 1 else 0,
        "memory_usage_bytes": statistics.mean(memory_usages),
        "memory_peak_bytes": max(memory_usages),
        
        # Solution quality
        "portfolio_return": avg_result["portfolio_return"],
        "portfolio_sharpe": avg_result["portfolio_sharpe"],
        "portfolio_std": avg_result["portfolio_std"],
        "num_stocks": avg_result["num_stocks"],
        
        # Metadata
        "input_size": len(stocks_metrics),
        "num_runs": num_runs,
    }


def benchmark_all_algorithms(stocks_metrics: Dict, target_num_stocks: int = 50, num_runs: int = 5):
    """
    Benchmark all three algorithms and return comparison results.
    """
    from algorithms.greedy import greedy_portfolio_allocation
    from algorithms.dp_knapsack import dp_knapsack_portfolio_allocation
    from algorithms.equal_weight import equal_weight_allocation
    
    algorithms = [
        ("Greedy", greedy_portfolio_allocation, {"target_num_stocks": target_num_stocks}),
        ("DP Knapsack", dp_knapsack_portfolio_allocation, {"target_num_stocks": target_num_stocks}),
        ("Equal Weight", equal_weight_allocation, {}),
    ]
    
    results = []
    
    print("=" * 100)
    print("ALGORITHM BENCHMARKING")
    print("=" * 100)
    
    for name, func, kwargs in algorithms:
        print(f"\nBenchmarking {name}... {target_num_stocks}")
        metrics = measure_algorithm(func, stocks_metrics, name, num_runs, **kwargs)
        results.append(metrics)
    
    return results


def display_benchmark_results(results: list):
    """Display benchmark comparison table."""
    print("\n" + "=" * 120)
    print("BENCHMARK COMPARISON")
    print("=" * 120)
    
    # Performance comparison
    print(f"\n{'Algorithm':<15} {'Time (ms)':<20} {'Memory (bytes)':<20} {'Return':<12} {'Sharpe':<12} {'Stocks':<10}")
    print("-" * 120)
    
    for r in results:
        print(
            f"{r['algorithm']:<15} "
            f"{r['execution_time_ms']:>10.2f} ± {r['execution_time_std']:<7.2f} "
            f"{r['memory_usage_bytes']:>15.0f}     "
            f"{r['portfolio_return']:>10.2%}  "
            f"{r['portfolio_sharpe']:>10.4f}  "
            f"{r['num_stocks']:>8}"
        )
    
    print("=" * 120)
    
    # Find winners
    fastest = min(results, key=lambda x: x['execution_time_ms'])
    best_return = max(results, key=lambda x: x['portfolio_return'])
    best_sharpe = max(results, key=lambda x: x['portfolio_sharpe'])
    
    print("\nWinners:")
    print(f"  Fastest: {fastest['algorithm']} ({fastest['execution_time_ms']:.2f} ms)")
    print(f"  Best Return: {best_return['algorithm']} ({best_return['portfolio_return']:.2%})")
    print(f"  Best Sharpe: {best_sharpe['algorithm']} ({best_sharpe['portfolio_sharpe']:.4f})")
