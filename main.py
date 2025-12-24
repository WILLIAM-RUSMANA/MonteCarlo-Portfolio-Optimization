from monte_carlo_method import monte_carlo_method
from greedy_whole import greedy_portfolio_allocation as greedy_whole
from greedy import greedy_portfolio_allocation as greedy


if __name__ == "__main__":
    amount = 100_000
    excepted_return_data = monte_carlo_method()

    greedy_whole_results = greedy_whole(excepted_return_data, amount, 50, plot_results=False)
    greedy_results = greedy(excepted_return_data, 50, plot_results=False)

    # Extract shares and cash from greedy_whole results
    shares = greedy_whole_results['shares']
    cash_remaining = greedy_whole_results['cash_remaining']
    
    print(f"{'Ticker':<10} {'Allocation (%)':<15} {'Shares':<10}")
    print("-" * 40)
    
    for ticker, allocation in greedy_results.items():
        num_shares = shares.get(ticker, 0)
        print(f"{ticker:<10} {allocation:<15.2%} {num_shares:<10}")
    
    print(f"\nCash Remaining: ${cash_remaining:,.2f}")
