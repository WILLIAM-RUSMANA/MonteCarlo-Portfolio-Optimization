import numpy as np
from helper import calculate_sharpe_ratio


def equal_weight_allocation(
    stocks_metrics,
    selected_tickers=None,
    display_results=True,
):
    """
    Equal-weight portfolio over a chosen set of stocks.

    Parameters:
    - stocks_metrics: dictionary from monte_carlo_method()
    - selected_tickers: list of tickers to include (optional).
        If None, use all the tickers in stocks_metrics.
    - display_results: whether to print summary to console.

    Returns:
    - allocations: {ticker: weight}, weights sum to 1.0
    - results: dictionary with portfolio
    """
    if selected_tickers is None:
        selected_tickers = list(stocks_metrics.keys())

    # Filter to tickers present in metrics
    selected_tickers = [t for t in selected_tickers if t in stocks_metrics]

    n = len(selected_tickers)
    if n == 0:
        return {}, {
            "portfolio_return": 0.0,
            "portfolio_std": 0.0,
            "portfolio_sharpe": 0.0,
            "num_stocks": 0,
        }

    # Equal weight 1/N
    weight = 1.0 / n
    allocations = {t: weight for t in selected_tickers}

    # Compute portfolio metrics
    portfolio_return = sum(
        weight * stocks_metrics[t]["mean_annual_return"] for t in selected_tickers
    )

    portfolio_variance = sum(
        (weight**2) * (stocks_metrics[t]["std_annual_return"] ** 2)
        for t in selected_tickers
    )
    portfolio_std = np.sqrt(portfolio_variance)
    portfolio_sharpe = calculate_sharpe_ratio(portfolio_return, portfolio_std)

    results = {
        "allocations": allocations,
        "portfolio_return": portfolio_return,
        "portfolio_std": portfolio_std,
        "portfolio_sharpe": portfolio_sharpe,
        "num_stocks": n,
    }

    if display_results:
        print("\n" + "=" * 100)
        print("EQUAL-WEIGHT PORTFOLIO ALLOCATION")
        print("=" * 100)
        print("\nSummary:")
        print(f"  Expected Annual Return: {portfolio_return:.2%}")
        print(f"  Portfolio Std Deviation: {portfolio_std:.2%}")
        print(f"  Portfolio Sharpe Ratio: {portfolio_sharpe:.4f}")
        print(f"  Number of Stocks: {n}")
        print("\n" + "-" * 100)
        print(f"{'Stock':<8} {'Weight':<10} {'Mean Return':<15} {'Std Dev':<12}")
        print("-" * 100)
        for t in selected_tickers:
            m = stocks_metrics[t]
            print(
                f"{t:<8} {weight:>8.2%} "
                f"{m['mean_annual_return']:>14.2%} "
                f"{m['std_annual_return']:>11.2%}"
            )
        print("=" * 100)

    return allocations, results


if __name__ == "__main__":
    print("This file is for import")
