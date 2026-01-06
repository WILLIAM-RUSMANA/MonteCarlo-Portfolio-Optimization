import numpy as np
from helper import calculate_sharpe_ratio

MAX_ALLOCATION_PER_STOCK = 0.20
MIN_ALLOCATION_PER_STOCK = 0.005
DISCRETIZATION_STEPS = 100


def dp_knapsack_portfolio_allocation(
    stocks_metrics,
    target_num_stocks=10,
    display_results=True,
):
    """
    DP knapsack portfolio optimization maximizing Sharpe ratio.

    Parameters:
    - stocks_metrics: dictionary from monte_carlo_method()
    - target_num_stocks: number of stocks to include
    - display_results: whether to print summary

    Returns:
    - allocations: {ticker: weight}
    - results: dictionary with portfolio metrics
    """

    # Calculate Sharpe ratios and select top K stocks
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

    sorted_stocks = sorted(
        sharpe_ratios.items(), key=lambda x: x[1]["sharpe_ratio"], reverse=True
    )
    selected = [s for s, _ in sorted_stocks[:target_num_stocks]]

    if not selected:
        return {}, {
            "portfolio_return": 0.0,
            "portfolio_std": 0.0,
            "portfolio_sharpe": 0.0,
            "num_stocks": 0,
        }

    # DP: dp[units] = (best_sharpe, weights)
    units = DISCRETIZATION_STEPS
    dp = {0: (float("-inf"), {})}
    min_u = max(1, int(MIN_ALLOCATION_PER_STOCK * units))
    max_u = int(MAX_ALLOCATION_PER_STOCK * units)

    for stock in selected:
        new_dp = dp.copy()
        for alloc_u in range(min_u, max_u + 1):
            w = alloc_u / units
            for prev_u in list(dp.keys()):
                if prev_u + alloc_u <= units:
                    _, prev_w = dp[prev_u]
                    cand = {**prev_w, stock: w}
                    total = sum(cand.values())
                    norm = {s: v / total for s, v in cand.items()} if total > 0 else cand

                    ret = sum(norm.get(s, 0) * sharpe_ratios[s]["mean_return"] for s in selected)
                    var = sum(norm.get(s, 0) ** 2 * sharpe_ratios[s]["std_return"] ** 2 for s in selected)
                    sharpe = calculate_sharpe_ratio(ret, np.sqrt(var))

                    if prev_u + alloc_u not in new_dp or sharpe > new_dp[prev_u + alloc_u][0]:
                        new_dp[prev_u + alloc_u] = (sharpe, cand)
        dp = new_dp

    # Extract best allocation
    best = max((dp[u] for u in range(max(0, units - 10), units + 1) if u in dp), default=(0, {}), key=lambda x: x[0])
    allocations = best[1] if best[1] else {s: 1.0 / len(selected) for s in selected}

    # Normalize
    total = sum(allocations.values())
    if total > 0:
        allocations = {s: w / total for s, w in allocations.items()}

    # Compute metrics
    portfolio_return = sum(allocations[s] * sharpe_ratios[s]["mean_return"] for s in allocations)
    portfolio_variance = sum(allocations[s] ** 2 * sharpe_ratios[s]["std_return"] ** 2 for s in allocations)
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
        print("DP KNAPSACK PORTFOLIO ALLOCATION")
        print("=" * 100)
        print("\nSummary:")
        print(f"  Expected Annual Return: {portfolio_return:.2%}")
        print(f"  Portfolio Std Deviation: {portfolio_std:.2%}")
        print(f"  Portfolio Sharpe Ratio: {portfolio_sharpe:.4f}")
        print(f"  Number of Stocks: {len(allocations)}")
        print("=" * 100)

    return allocations, results


if __name__ == "__main__":
    print("This file is for import")
