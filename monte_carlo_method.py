import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from constants import CSV_FILE_50, NUM_SIMULATIONS, TRADING_DAYS_PER_YEAR


def load_and_prepare_data(filepath):
    """Load CSV and prepare data for analysis"""
    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df = df.sort_index()
    return df


def calculate_returns(prices):
    """Calculate daily returns"""
    return prices.pct_change().dropna()  # percentage change then, drops first NaN value


def monte_carlo_simulation(returns, num_simulations=10000, days=252):
    """
    Perform Monte Carlo simulation

    Parameters:
    - returns: DataFrame of daily returns
    - num_simulations: Number of simulation paths
    - days: Number of trading days to simulate (252 = 1 year)
    """
    mean_returns = returns.mean()

    # Store results for each stock
    results = {}

    for stock in returns.columns:
        stock_mean = mean_returns[stock]
        stock_std = returns[stock].std()

        # Run simulations
        simulated_returns = np.random.normal(
            stock_mean, stock_std, size=(num_simulations, days)
        )  # Draw numbers from the normal distribution 

        # Calculate cumulative returns for each simulation
        cumulative_returns = (1 + simulated_returns).prod(axis=1) - 1  # simulate what if scenarios

        results[stock] = {
            "simulated_annual_returns": cumulative_returns,
            "mean_annual_return": cumulative_returns.mean(),
            "median_annual_return": np.median(cumulative_returns),
            "std_annual_return": cumulative_returns.std(),
            "percentile_5": np.percentile(cumulative_returns, 5),
            "percentile_95": np.percentile(cumulative_returns, 95),
        }

    return results


def display_results(results):
    """Display simulation results in a formatted table"""
    print("\n" + "=" * 100)
    print("MONTE CARLO SIMULATION RESULTS - ANNUAL EXPECTED RETURNS")
    print("=" * 100)
    print(
        f"{'Stock':<8} {'Mean Return':<15} {'Median Return':<15} {'Std Dev':<15} {'5th %ile':<15} {'95th %ile':<15}"
    )
    print("-" * 100)

    # Sort by mean return
    sorted_results = sorted(
        results.items(), key=lambda x: x[1]["mean_annual_return"], reverse=True
    )

    for stock, metrics in sorted_results:
        print(
            f"{stock:<8} {metrics['mean_annual_return']:>13.2%} {metrics['median_annual_return']:>14.2%} "
            f"{metrics['std_annual_return']:>14.2%} {metrics['percentile_5']:>14.2%} {metrics['percentile_95']:>14.2%}"
        )

    print("=" * 100)


def plot_top_stocks(results, top_n=10):
    """Plot distribution of returns for top performing stocks"""
    sorted_results = sorted(
        results.items(), key=lambda x: x[1]["mean_annual_return"], reverse=True
    )
    top_stocks = sorted_results[:top_n]

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for idx, (stock, metrics) in enumerate(top_stocks):
        axes[idx].hist(
            metrics["simulated_annual_returns"], bins=50, alpha=0.7, edgecolor="black"
        )
        axes[idx].axvline(
            metrics["mean_annual_return"],
            color="red",
            linestyle="--",
            label=f"Mean: {metrics['mean_annual_return']:.2%}",
        )
        axes[idx].axvline(
            metrics["median_annual_return"],
            color="green",
            linestyle="--",
            label=f"Median: {metrics['median_annual_return']:.2%}",
        )
        axes[idx].set_title(f"{stock}", fontweight="bold")
        axes[idx].set_xlabel("Annual Return")
        axes[idx].set_ylabel("Frequency")
        axes[idx].legend(fontsize=8)
        axes[idx].grid(alpha=0.3)

    plt.show()


def monte_carlo_method(
    num_simulations=NUM_SIMULATIONS, trading_days_per_year=TRADING_DAYS_PER_YEAR, csv_data=CSV_FILE_50
):
    print("Loading stock data...")
    prices = load_and_prepare_data(csv_data)

    print(f"Data loaded: {len(prices)} days of data")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"Number of stocks: {len(prices.columns)}")

    print("\nCalculating daily returns...")
    returns = calculate_returns(prices)

    print(f"Running Monte Carlo simulation with {num_simulations:,} iterations...")
    results = monte_carlo_simulation(returns, num_simulations, trading_days_per_year)

    display_results(results)

    print("\nGenerating visualization...")
    # plot_top_stocks(results)

    # Export results to CSV
    return {
        stock: {
            "mean_annual_return": metrics["mean_annual_return"],
            "median_annual_return": metrics["median_annual_return"],
            "std_annual_return": metrics["std_annual_return"],
            "percentile_5": metrics["percentile_5"],
            "percentile_95": metrics["percentile_95"],
        }
        for stock, metrics in results.items()
    }
