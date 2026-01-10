import pandas as pd
from monte_carlo_method import monte_carlo_method
from benchmark import benchmark_all_algorithms, display_benchmark_results
from constants import CSV_FILE_100, CSV_FILE_250, CSV_FILE_500


if __name__ == "__main__":
    print("=" * 100)
    print("PORTFOLIO ALLOCATION ALGORITHM ANALYSIS")
    print("=" * 100)
    
    # Run Monte Carlo simulation
    print("\nStep 1: Running Monte Carlo simulation (10000 iterations)...")
    stocks_metrics_50 = monte_carlo_method(num_simulations=10000)
    stocks_metrics_100 = monte_carlo_method(10000, 252, CSV_FILE_100)
    stocks_metrics_250 = monte_carlo_method(10000, 252, CSV_FILE_250)
    stocks_metrics_500 = monte_carlo_method(10000, 252, CSV_FILE_500)

    print(f"Simulation complete. Analyzing {len(stocks_metrics_50)} stocks.")
    
    # Benchmark all algorithms
    print("\nStep 2: Benchmarking algorithms (5 runs each for statistical stability)...")
    num_runs = 5
    results_50 = benchmark_all_algorithms(
        stocks_metrics_50, 
        target_num_stocks=50, 
        num_runs=num_runs
    )

    results_100 = benchmark_all_algorithms(
        stocks_metrics_100, 
        target_num_stocks=100, 
        num_runs=num_runs
    )

    results_250 = benchmark_all_algorithms(
        stocks_metrics_250, 
        target_num_stocks=250, 
        num_runs=num_runs
    )
    
    results_500 = benchmark_all_algorithms(
        stocks_metrics_500, 
        target_num_stocks=500,
        num_runs=num_runs
    )

    # Display results
    print("\nStep 3: Analysis Results")
    display_benchmark_results(results_50)
    display_benchmark_results(results_100)
    display_benchmark_results(results_250)
    display_benchmark_results(results_500)

    # Save results to CSV
    print("\nStep 4: Saving results...")
    df_50 = pd.DataFrame(results_50)
    df_50.to_csv("data/benchmark_results_50.csv", index=False)
    print("Results saved to 'data/benchmark_results.csv'")
    
    # Additional analysis
    print("\n" + "=" * 100)
    print("DETAILED ANALYSIS")
    print("=" * 100)
    
    combined_results = results_50 + results_100 + results_250 + results_500

    # Time efficiency ranking
    sorted_by_time = sorted(combined_results, key=lambda x: x['execution_time_ms'])
    print("\nTime Efficiency Ranking:")
    for i, r in enumerate(sorted_by_time, 1):
        print(f"  {i}. {r['algorithm']:<15} {r['execution_time_ms']:>8.2f} ms")
    
    # Solution quality ranking (by Sharpe ratio)
    sorted_by_sharpe = sorted(combined_results, key=lambda x: x['portfolio_sharpe'], reverse=True)
    print("\nSolution Quality Ranking (by Sharpe Ratio):")
    for i, r in enumerate(sorted_by_sharpe, 1):
        print(f"  {i}. {r['algorithm']:<15} Sharpe: {r['portfolio_sharpe']:>8.4f}, Return: {r['portfolio_return']:>7.2%}")
    
    # Trade-off analysis
    print("\nTrade-off Analysis:")
    fastest = sorted_by_time[0]
    best_quality = sorted_by_sharpe[0]
    
    if fastest['algorithm'] == best_quality['algorithm']:
        print(f"  {fastest['algorithm']} is both fastest AND highest quality!")
    else:
        time_diff = best_quality['execution_time_ms'] - fastest['execution_time_ms']
        sharpe_diff = best_quality['portfolio_sharpe'] - fastest['portfolio_sharpe']
        print(f"  {best_quality['algorithm']} achieves {sharpe_diff:+.4f} higher Sharpe")
        print(f"  but takes {time_diff:+.2f} ms longer than {fastest['algorithm']}")
    
    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)