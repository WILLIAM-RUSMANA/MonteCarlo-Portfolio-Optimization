import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from helper import calculate_sharpe_ratio

# Configuration
RISK_FREE_RATE = 0.04  # 4% annual risk-free rate (adjust as needed)
MAX_ALLOCATION_PER_STOCK = 0.20  # Maximum 20% in any single stock
MIN_ALLOCATION_PER_STOCK = 0.005  # Minimum 0.5% if stock is selected


def display_allocation_results(results):
    """Display portfolio allocation results"""
    print("\n" + "="*100)
    print("GREEDY PORTFOLIO ALLOCATION BASED ON SHARPE RATIO")
    print("="*100)
    
    print(f"\nPortfolio Summary:")
    print(f"  Expected Annual Return: {results['portfolio_return']:.2%}")
    print(f"  Portfolio Std Deviation: {results['portfolio_std']:.2%}")
    print(f"  Portfolio Sharpe Ratio: {results['portfolio_sharpe']:.4f}")
    print(f"  Number of Stocks: {results['num_stocks']}")
    print(f"  Risk-Free Rate Used: {RISK_FREE_RATE:.2%}")
    
    print("\n" + "-"*100)
    print(f"{'Stock':<8} {'Weight':<10} {'Sharpe':<12} {'Mean Return':<15} {'Std Dev':<12} {'5th %ile':<12} {'95th %ile':<12}")
    print("-"*100)
    
    # Sort by allocation weight
    sorted_allocations = sorted(results['allocations'].items(), 
                               key=lambda x: x[1], 
                               reverse=True)
    
    for stock, weight in sorted_allocations:
        metrics = results['sharpe_ratios'][stock]
        print(f"{stock:<8} {weight:>8.2%} {metrics['sharpe_ratio']:>11.4f} "
              f"{metrics['mean_return']:>14.2%} {metrics['std_return']:>11.2%} "
              f"{metrics['percentile_5']:>11.2%} {metrics['percentile_95']:>11.2%}")
    
    print("="*100)

def plot_allocation(results):
    """Visualize portfolio allocation"""
    allocations = results['allocations']
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Pie chart of allocations
    stocks = list(allocations.keys())
    weights = list(allocations.values())
    colors = plt.cm.Set3(range(len(stocks)))
    
    wedges, texts, autotexts = ax1.pie(weights, labels=stocks, autopct='%1.1f%%',
                                        colors=colors, startangle=90)
    ax1.set_title('Portfolio Allocation', fontsize=14, fontweight='bold')
    
    # Make percentage text more readable
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)
    
    # Bar chart of Sharpe ratios
    sharpe_values = [results['sharpe_ratios'][stock]['sharpe_ratio'] 
                     for stock in stocks]
    bars = ax2.bar(range(len(stocks)), sharpe_values, color=colors)
    ax2.set_xticks(range(len(stocks)))
    ax2.set_xticklabels(stocks, rotation=45, ha='right')
    ax2.set_ylabel('Sharpe Ratio', fontsize=12)
    ax2.set_title('Sharpe Ratio by Stock', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('portfolio_allocation.png', dpi=300, bbox_inches='tight')
    print("\nAllocation plot saved as 'portfolio_allocation.png'")
    plt.show()

def compare_with_equal_weight(results, stocks_metrics):
    """Compare greedy allocation with equal-weight portfolio"""
    selected_stocks = list(results['allocations'].keys())
    
    # Add this check:
    if not selected_stocks:
        print("\nNo stocks selected - cannot compare with equal-weight")
        return
    
    equal_weight = 1.0 / len(selected_stocks)

    equal_portfolio_return = sum(equal_weight * stocks_metrics[stock]['mean_annual_return'] 
                                 for stock in selected_stocks)
    
    equal_portfolio_variance = sum((equal_weight ** 2) * 
                                  (stocks_metrics[stock]['std_annual_return'] ** 2) 
                                  for stock in selected_stocks)
    equal_portfolio_std = np.sqrt(equal_portfolio_variance)
    equal_portfolio_sharpe = calculate_sharpe_ratio(equal_portfolio_return, equal_portfolio_std)
    
    print("\n" + "="*100)
    print("COMPARISON: GREEDY vs EQUAL-WEIGHT ALLOCATION")
    print("="*100)
    print(f"{'Metric':<30} {'Greedy Allocation':<25} {'Equal-Weight':<25} {'Difference':<15}")
    print("-"*100)
    print(f"{'Expected Annual Return':<30} {results['portfolio_return']:>23.2%} "
          f"{equal_portfolio_return:>23.2%} {(results['portfolio_return']-equal_portfolio_return):>13.2%}")
    print(f"{'Portfolio Std Deviation':<30} {results['portfolio_std']:>23.2%} "
          f"{equal_portfolio_std:>23.2%} {(results['portfolio_std']-equal_portfolio_std):>13.2%}")
    print(f"{'Portfolio Sharpe Ratio':<30} {results['portfolio_sharpe']:>23.4f} "
          f"{equal_portfolio_sharpe:>23.4f} {(results['portfolio_sharpe']-equal_portfolio_sharpe):>13.4f}")
    print("="*100)

def greedy_portfolio_allocation(stocks_metrics, target_num_stocks=10, 
                               display_results=True, plot_results=True, 
                               compare_equal_weight=True):
    """
    Greedy algorithm to allocate portfolio based on Sharpe ratio
    
    Parameters:
    - stocks_metrics: Dictionary with stock metrics from Monte Carlo simulation
                     Expected keys: 'mean_annual_return', 'std_annual_return', 
                     'percentile_5', 'percentile_95'
    - target_num_stocks: Number of stocks to include in portfolio (default: 10)
    - display_results: Print allocation table to console (default: True)
    - plot_results: Generate visualization (default: True)
    - compare_equal_weight: Show comparison with equal-weight portfolio (default: True)
    
    Returns:
    - Dictionary of {ticker: allocation_percentage}
      Example: {'AAPL': 0.15, 'GOOGL': 0.12, 'MSFT': 0.18, ...}
      where values are decimals (0.15 = 15%)
    """
    # Calculate Sharpe ratio for each stock
    sharpe_ratios = {}
    
    for stock, metrics in stocks_metrics.items():
        mean_return = metrics['mean_annual_return']
        std_return = metrics['std_annual_return']
        sharpe = calculate_sharpe_ratio(mean_return, std_return)
        
        sharpe_ratios[stock] = {
            'sharpe_ratio': sharpe,
            'mean_return': mean_return,
            'std_return': std_return,
            'percentile_5': metrics['percentile_5'],
            'percentile_95': metrics['percentile_95']
        }
    
    # Sort stocks by Sharpe ratio (descending)
    sorted_stocks = sorted(sharpe_ratios.items(), 
                          key=lambda x: x[1]['sharpe_ratio'], 
                          reverse=True)
    
    # Select top N stocks
    selected_stocks = sorted_stocks[:target_num_stocks]
    
    # Calculate total Sharpe ratio for normalization
    total_sharpe = sum(stock[1]['sharpe_ratio'] for stock in selected_stocks 
                       if stock[1]['sharpe_ratio'] > 0)
    
    # Allocate weights proportional to Sharpe ratio
    allocations = {}
    
    if total_sharpe > 0:
        for stock, metrics in selected_stocks:
            if metrics['sharpe_ratio'] > 0:
                raw_weight = metrics['sharpe_ratio'] / total_sharpe
                
                # Only apply constraints if using fewer stocks
                if target_num_stocks <= 15:
                    weight = min(max(raw_weight, MIN_ALLOCATION_PER_STOCK), 
                            MAX_ALLOCATION_PER_STOCK)
                    allocations[stock] = weight
                else:
                    allocations[stock] = raw_weight
        
        # Normalize to ensure weights sum to 1.0
        total_weight = sum(allocations.values())
        allocations = {stock: weight/total_weight 
                      for stock, weight in allocations.items()}
    
    # Calculate portfolio metrics
    portfolio_return = sum(allocations[stock] * sharpe_ratios[stock]['mean_return'] 
                          for stock in allocations)
    
    # Simplified portfolio std (assuming independence - not perfect but reasonable)
    portfolio_variance = sum((allocations[stock] ** 2) * 
                           (sharpe_ratios[stock]['std_return'] ** 2) 
                           for stock in allocations)
    portfolio_std = np.sqrt(portfolio_variance)
    
    portfolio_sharpe = calculate_sharpe_ratio(portfolio_return, portfolio_std)
    
    results = {
        'allocations': allocations,
        'sharpe_ratios': sharpe_ratios,
        'portfolio_return': portfolio_return,
        'portfolio_std': portfolio_std,
        'portfolio_sharpe': portfolio_sharpe,
        'num_stocks': len(allocations)
    }
    
    # Optional outputs
    if display_results:
        display_allocation_results(results)
    
    if plot_results:
        plot_allocation(results)
    
    if compare_equal_weight:
        compare_with_equal_weight(results, stocks_metrics)
    
    # Return just the allocations dictionary: {ticker: allocation_percentage}
    return allocations
