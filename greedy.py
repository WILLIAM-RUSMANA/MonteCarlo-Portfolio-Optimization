import numpy as np
from helper import calculate_sharpe_ratio

MAX_ALLOCATION_PER_STOCK = 0.20
MIN_ALLOCATION_PER_STOCK = 0.005


def greedy_portfolio_allocation(
    stocks_metrics,
    target_num_stocks=10,
    display_results=True,
):
    """
    Greedy portfolio allocation based on Sharpe ratio.

    Parameters:
    - stocks_metrics: dictionary from monte_carlo_method()
    - target_num_stocks: number of stocks to include
    - display_results: whether to print summary

    Returns:
    - allocations: {ticker: weight}
    - results: dictionary with portfolio metrics
    """

    # Calculate Sharpe ratios
    sharpe_ratios = {
        stock: {
            "sharpe_ratio": calculate_sharpe_ratio(
                metrics["mean_annual_return"], metrics["std_annual_return"]
            ),
            "mean_return": metrics["mean_annual_return"],
            "std_return": metrics["std_annual_return"],
        }
        for stock, metrics in stocks_metrics.items()
    }

    # Select top K stocks by Sharpe ratio
    sorted_stocks = sorted(
        sharpe_ratios.items(), key=lambda x: x[1]["sharpe_ratio"], reverse=True
    )
    selected = sorted_stocks[:target_num_stocks]

    if not selected:
        return {}, {
            "portfolio_return": 0.0,
            "portfolio_std": 0.0,
            "portfolio_sharpe": 0.0,
            "num_stocks": 0,
        }

    # Allocate weights proportional to Sharpe ratio
    total_sharpe = sum(
        s[1]["sharpe_ratio"] for s in selected if s[1]["sharpe_ratio"] > 0
    )

    if total_sharpe > 0:
        allocations = {
            stock: metrics["sharpe_ratio"] / total_sharpe
            for stock, metrics in selected
            if metrics["sharpe_ratio"] > 0
        }

        # Apply constraints if using few stocks
        if target_num_stocks <= 15:
            allocations = {
                s: min(max(w, MIN_ALLOCATION_PER_STOCK), MAX_ALLOCATION_PER_STOCK)
                for s, w in allocations.items()
            }

        # Normalize
        total = sum(allocations.values())
        if total > 0:
            allocations = {s: w / total for s, w in allocations.items()}
    else:
        allocations = {}

    # Compute portfolio metrics
    portfolio_return = sum(
        allocations[s] * sharpe_ratios[s]["mean_return"] for s in allocations
    )
    portfolio_variance = sum(
        allocations[s] ** 2 * sharpe_ratios[s]["std_return"] ** 2 for s in allocations
    )
    portfolio_std = np.sqrt(portfolio_variance)
    portfolio_sharpe = calculate_sharpe_ratio(portfolio_return, portfolio_std)

    results = {
        "allocations": allocations,
        "portfolio_return": portfolio_return,
        "portfolio_std": portfolio_std,
        "portfolio_sharpe": portfolio_sharpe,
        "num_stocks": len(allocations),
    }

    if display_results:
        print("\n" + "=" * 100)
        print("GREEDY SHARPE RATIO PORTFOLIO ALLOCATION")
        print("=" * 100)
        print("\nSummary:")
        print(f"  Expected Annual Return: {portfolio_return:.2%}")
        print(f"  Portfolio Std Deviation: {portfolio_std:.2%}")
        print(f"  Portfolio Sharpe Ratio: {portfolio_sharpe:.4f}")
        print(f"  Number of Stocks: {len(allocations)}")
        print("\n" + "-" * 100)
        print(f"{'Stock':<8} {'Weight':<10} {'Mean Return':<15} {'Std Dev':<12}")
        print("-" * 100)
        for stock in sorted(
            allocations.keys(), key=lambda s: allocations[s], reverse=True
        ):
            print(
                f"{stock:<8} {allocations[stock]:>8.2%} "
                f"{sharpe_ratios[stock]['mean_return']:>14.2%} "
                f"{sharpe_ratios[stock]['std_return']:>11.2%}"
            )
        print("=" * 100)

    return allocations, results


if __name__ == "__main__":
    print("This file is for import")
