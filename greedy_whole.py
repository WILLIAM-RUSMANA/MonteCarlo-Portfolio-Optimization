import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from helper import calculate_sharpe_ratio

# Configuration


def get_current_stock_prices(tickers):
    """
    Fetch current stock prices from Yahoo Finance

    Parameters:
    - tickers: List of stock ticker symbols

    Returns:
    - Dictionary of {ticker: current_price}
    """
    print(f"\nFetching current prices for {len(tickers)} stocks from Yahoo Finance...")
    prices = {}

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            # Get the most recent close price
            hist = stock.history(period="1d")
            if not hist.empty:
                prices[ticker] = hist["Close"].iloc[-1]
                print(f"  {ticker}: ${prices[ticker]:.2f}")
            else:
                print(f"  {ticker}: No data available")
                prices[ticker] = None
        except Exception as e:
            print(f"  {ticker}: Error fetching price - {e}")
            prices[ticker] = None

    return prices


def allocate_whole_shares(target_allocations, stock_prices, budget):
    """
    Allocate portfolio using whole shares only (no fractional shares)
    Includes cash sweep to minimize leftover cash

    Parameters:
    - target_allocations: Dictionary of {ticker: target_weight}
    - stock_prices: Dictionary of {ticker: current_price}
    - budget: Total budget in USD

    Returns:
    - Dictionary with share allocations and actual portfolio composition
    """
    # Calculate target dollar amounts
    target_dollars = {
        ticker: weight * budget for ticker, weight in target_allocations.items()
    }

    # Calculate whole shares for each stock
    shares = {}
    actual_spent = {}

    for ticker, target_amount in target_dollars.items():
        price = stock_prices.get(ticker)
        if price and price > 0:
            # Buy whole shares
            num_shares = int(target_amount / price)
            shares[ticker] = num_shares
            actual_spent[ticker] = num_shares * price
        else:
            shares[ticker] = 0
            actual_spent[ticker] = 0

    # Calculate initial cash remaining
    total_spent = sum(actual_spent.values())
    cash_remaining = budget - total_spent

    # CASH SWEEP: Use remaining cash to buy more shares while respecting target allocations
    print(f"\nCash sweep starting with ${cash_remaining:,.2f}...")
    sweep_count = 0

    while cash_remaining > 0:
        # Find the stock that, when buying one more share, gets closest to its target allocation
        best_ticker = None
        best_error = float('inf')
        
        for ticker, price in stock_prices.items():
            if ticker in target_allocations and price and price <= cash_remaining:
                # Calculate what the allocation would be if we buy one more share
                test_shares = shares.get(ticker, 0) + 1
                test_spent = test_shares * price
                test_total = sum(actual_spent.values()) + price
                test_allocation = test_spent / test_total
                target_allocation = target_allocations[ticker]
                
                # Calculate error from target
                error = abs(test_allocation - target_allocation)
                
                # Choose stock with smallest error (closest to target)
                if error < best_error:
                    best_error = error
                    best_ticker = ticker
        
        if best_ticker is None:
            # Can't afford any more shares or no improvement possible
            break
        
        # Buy one more share of the stock that keeps allocation closest to target
        price = stock_prices[best_ticker]
        shares[best_ticker] = shares.get(best_ticker, 0) + 1
        actual_spent[best_ticker] = shares[best_ticker] * price
        cash_remaining -= price
        sweep_count += 1

    print(f"Cash sweep complete: bought {sweep_count} additional shares")
    print(f"Final cash remaining: ${cash_remaining:,.2f}")

    # Recalculate total spent and actual allocations
    total_spent = sum(actual_spent.values())
    actual_allocations = {
        ticker: spent / total_spent if total_spent > 0 else 0
        for ticker, spent in actual_spent.items()
    }

    # Remove stocks with 0 shares
    shares = {ticker: num for ticker, num in shares.items() if num > 0}
    actual_allocations = {
        ticker: alloc for ticker, alloc in actual_allocations.items() if alloc > 0
    }

    return {
        "shares": shares,
        "actual_allocations": actual_allocations,
        "actual_spent": actual_spent,
        "total_spent": total_spent,
        "cash_remaining": cash_remaining,
        "budget": budget,
        "stock_prices": stock_prices,
        "sweep_count": sweep_count,
    }


def display_share_allocation(allocation_result):
    """Display the whole share allocation results"""
    print("\n" + "=" * 100)
    print("WHOLE SHARE ALLOCATION")
    print("=" * 100)

    print(f"\nBudget: ${allocation_result['budget']:,.2f}")
    print(f"Total Spent: ${allocation_result['total_spent']:,.2f}")
    print(f"Cash Remaining: ${allocation_result['cash_remaining']:,.2f}")
    print(
        f"Cash Utilization: {(allocation_result['total_spent'] / allocation_result['budget'] * 100):.2f}%"
    )
    print(f"Additional shares from sweep: {allocation_result.get('sweep_count', 0)}")

    print("\n" + "-" * 100)
    print(f"{'Ticker':<8} {'Shares':<10} {'Price':<12} {'Amount':<15} {'Actual %':<12}")
    print("-" * 100)

    shares = allocation_result["shares"]
    prices = allocation_result["stock_prices"]
    actual_allocs = allocation_result["actual_allocations"]

    # Sort by actual allocation
    sorted_shares = sorted(
        shares.items(), key=lambda x: actual_allocs.get(x[0], 0), reverse=True
    )

    for ticker, num_shares in sorted_shares:
        price = prices[ticker]
        amount = num_shares * price
        actual_pct = actual_allocs.get(ticker, 0)

        print(
            f"{ticker:<8} {num_shares:<10} ${price:>10.2f} ${amount:>13,.2f} {actual_pct:>10.2%}"
        )

    print("=" * 100)


def display_allocation_results(results):
    """Display portfolio allocation results"""
    print("\n" + "=" * 100)
    print("GREEDY PORTFOLIO ALLOCATION BASED ON SHARPE RATIO")
    print("=" * 100)

    print("\nPortfolio Summary:")
    print(f"  Expected Annual Return: {results['portfolio_return']:.2%}")
    print(f"  Portfolio Std Deviation: {results['portfolio_std']:.2%}")
    print(f"  Portfolio Sharpe Ratio: {results['portfolio_sharpe']:.4f}")
    print(f"  Number of Stocks: {results['num_stocks']}")

    print("\n" + "-" * 100)
    print(
        f"{'Stock':<8} {'Weight':<10} {'Sharpe':<12} {'Mean Return':<15} {'Std Dev':<12} {'5th %ile':<12} {'95th %ile':<12}"
    )
    print("-" * 100)

    # Sort by allocation weight
    sorted_allocations = sorted(
        results["allocations"].items(), key=lambda x: x[1], reverse=True
    )

    for stock, weight in sorted_allocations:
        metrics = results["sharpe_ratios"][stock]
        print(
            f"{stock:<8} {weight:>8.2%} {metrics['sharpe_ratio']:>11.4f} "
            f"{metrics['mean_return']:>14.2%} {metrics['std_return']:>11.2%} "
            f"{metrics['percentile_5']:>11.2%} {metrics['percentile_95']:>11.2%}"
        )

    print("=" * 100)


def plot_allocation(results):
    """Visualize portfolio allocation"""
    allocations = results["allocations"]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Pie chart of allocations
    stocks = list(allocations.keys())
    weights = list(allocations.values())
    colors = plt.cm.Set3(range(len(stocks)))

    wedges, texts, autotexts = ax1.pie(
        weights, labels=stocks, autopct="%1.1f%%", colors=colors, startangle=90
    )
    ax1.set_title("Portfolio Allocation", fontsize=14, fontweight="bold")

    # Make percentage text more readable
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")
        autotext.set_fontsize(9)

    # Bar chart of Sharpe ratios
    ax2.set_xticks(range(len(stocks)))
    ax2.set_xticklabels(stocks, rotation=45, ha="right")
    ax2.set_ylabel("Sharpe Ratio", fontsize=12)
    ax2.set_title("Sharpe Ratio by Stock", fontsize=14, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    plt.tight_layout()
    plt.savefig("portfolio_allocation_whole.png", dpi=300, bbox_inches="tight")
    print("\nAllocation plot saved as 'portfolio_allocation_whole.png'")
    plt.show()


def compare_with_equal_weight(results, stocks_metrics):
    """Compare greedy allocation with equal-weight portfolio"""
    selected_stocks = list(results["allocations"].keys())

    # Check if there are any selected stocks
    if not selected_stocks:
        print("\nNo stocks selected - cannot compare with equal-weight")
        return

    equal_weight = 1.0 / len(selected_stocks)

    equal_portfolio_return = sum(
        equal_weight * stocks_metrics[stock]["mean_annual_return"]
        for stock in selected_stocks
    )

    equal_portfolio_variance = sum(
        (equal_weight**2) * (stocks_metrics[stock]["std_annual_return"] ** 2)
        for stock in selected_stocks
    )

    equal_portfolio_std = np.sqrt(equal_portfolio_variance)
    equal_portfolio_sharpe = calculate_sharpe_ratio(
        equal_portfolio_return, equal_portfolio_std
    )

    print("\n" + "=" * 100)
    print("COMPARISON: GREEDY vs EQUAL-WEIGHT ALLOCATION")
    print("=" * 100)
    print(
        f"{'Metric':<30} {'Greedy Allocation':<25} {'Equal-Weight':<25} {'Difference':<15}"
    )
    print("-" * 100)
    print(
        f"{'Expected Annual Return':<30} {results['portfolio_return']:>23.2%} "
        f"{equal_portfolio_return:>23.2%} {(results['portfolio_return'] - equal_portfolio_return):>13.2%}"
    )
    print(
        f"{'Portfolio Std Deviation':<30} {results['portfolio_std']:>23.2%} "
        f"{equal_portfolio_std:>23.2%} {(results['portfolio_std'] - equal_portfolio_std):>13.2%}"
    )
    print(
        f"{'Portfolio Sharpe Ratio':<30} {results['portfolio_sharpe']:>23.4f} "
        f"{equal_portfolio_sharpe:>23.4f} {(results['portfolio_sharpe'] - equal_portfolio_sharpe):>13.4f}"
    )
    print("=" * 100)


def greedy_portfolio_allocation(
    stocks_metrics,
    amount,
    target_num_stocks=10,
    display_results=True,
    plot_results=True,
    compare_equal_weight=True,
):
    """
    Greedy algorithm to allocate portfolio based on Sharpe ratio with whole shares

    Parameters:
    - stocks_metrics: Dictionary with stock metrics from Monte Carlo simulation
                     Expected keys: 'mean_annual_return', 'std_annual_return',
                     'percentile_5', 'percentile_95'
    - amount: Budget in USD for portfolio allocation
    - target_num_stocks: Number of stocks to include in portfolio (default: 10)
    - display_results: Print allocation table to console (default: True)
    - plot_results: Generate visualization (default: True)
    - compare_equal_weight: Show comparison with equal-weight portfolio (default: True)

    Returns:
    - Dictionary with:
        * 'shares': {ticker: number_of_shares}
        * 'cash_remaining': Unspent cash in USD
      Example: {
          'shares': {'AAPL': 25, 'GOOGL': 15, 'MSFT': 30},
          'cash_remaining': 1250.75
      }
    """
    # Calculate Sharpe ratio for each stock
    sharpe_ratios = {}

    for stock, metrics in stocks_metrics.items():
        mean_return = metrics["mean_annual_return"]
        std_return = metrics["std_annual_return"]
        sharpe = calculate_sharpe_ratio(mean_return, std_return)

        sharpe_ratios[stock] = {
            "sharpe_ratio": sharpe,
            "mean_return": mean_return,
            "std_return": std_return,
            "percentile_5": metrics["percentile_5"],
            "percentile_95": metrics["percentile_95"],
        }

    # Sort stocks by Sharpe ratio (descending)
    sorted_stocks = sorted(
        sharpe_ratios.items(), key=lambda x: x[1]["sharpe_ratio"], reverse=True
    )

    # Select top N stocks
    selected_stocks = sorted_stocks[:target_num_stocks]

    # Calculate total Sharpe ratio for normalization
    total_sharpe = sum(
        stock[1]["sharpe_ratio"]
        for stock in selected_stocks
        if stock[1]["sharpe_ratio"] > 0
    )

    # Allocate weights proportional to Sharpe ratio
    target_allocations = {}

    if total_sharpe > 0:
        for stock, metrics in selected_stocks:
            if metrics["sharpe_ratio"] > 0:
                raw_weight = metrics["sharpe_ratio"] / total_sharpe
                target_allocations[stock] = raw_weight

        # Normalize to ensure weights sum to 1.0
        total_weight = sum(target_allocations.values())
        target_allocations = {
            stock: weight / total_weight for stock, weight in target_allocations.items()
        }

    # Check if we have any allocations
    if not target_allocations:
        print("\nNo valid allocations - all stocks have negative or zero Sharpe ratios")
        return {"shares": {}, "cash_remaining": amount}

    # Get current stock prices
    tickers = list(target_allocations.keys())
    stock_prices = get_current_stock_prices(tickers)

    # Allocate using whole shares
    allocation_result = allocate_whole_shares(target_allocations, stock_prices, amount)

    # Display results
    if display_results:
        display_share_allocation(allocation_result)

    # Calculate portfolio metrics for display
    portfolio_return = sum(
        target_allocations[stock] * sharpe_ratios[stock]["mean_return"]
        for stock in target_allocations
    )

    portfolio_variance = sum(
        (target_allocations[stock] ** 2) * (sharpe_ratios[stock]["std_return"] ** 2)
        for stock in target_allocations
    )
    portfolio_std = np.sqrt(portfolio_variance)

    portfolio_sharpe = calculate_sharpe_ratio(portfolio_return, portfolio_std)

    results = {
        "allocations": target_allocations,
        "sharpe_ratios": sharpe_ratios,
        "portfolio_return": portfolio_return,
        "portfolio_std": portfolio_std,
        "portfolio_sharpe": portfolio_sharpe,
        "num_stocks": len(target_allocations),
    }

    # Optional outputs for original allocation (before whole share conversion)
    if display_results:
        display_allocation_results(results)

    if plot_results:
        plot_allocation(results)

    if compare_equal_weight:
        compare_with_equal_weight(results, stocks_metrics)

    # Return dictionary of shares and cash remaining
    return {
        "shares": allocation_result["shares"],
        "cash_remaining": allocation_result["cash_remaining"],
        "stock_prices": allocation_result["stock_prices"],
    }