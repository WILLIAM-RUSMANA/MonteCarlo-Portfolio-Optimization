import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


# Set seed for reproducibility
np.random.seed(42)

# Define assets
assets = ['Stock A', 'Stock B', 'Stock C', 'Bond A', 'Crypto X']

# Expected annual returns (in %)
mean_returns = np.array([0.10, 0.08, 0.12, 0.05, 0.18])

# Standard deviation (volatility)
std_devs = np.array([0.15, 0.10, 0.20, 0.05, 0.30])

# Correlation matrix (simplified assumption)
correlation_matrix = np.array([
    [1.0, 0.6, 0.7, 0.3, 0.4],
    [0.6, 1.0, 0.5, 0.4, 0.3],
    [0.7, 0.5, 1.0, 0.2, 0.5],
    [0.3, 0.4, 0.2, 1.0, 0.2],
    [0.4, 0.3, 0.5, 0.2, 1.0]
])

# Convert to covariance matrix
cov_matrix = np.outer(std_devs, std_devs) * correlation_matrix

# Assume each asset costs a different amount (e.g., per unit)
costs = np.array([400, 350, 500, 300, 600])

# Available budget
budget = 1500

num_portfolios = 10000
num_assets = len(assets)

results = np.zeros((3, num_portfolios))

for i in range(num_portfolios):
    # Random weights (normalized to sum to 1)
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    
    # Expected return
    portfolio_return = np.dot(weights, mean_returns)
    
    # Portfolio volatility (sqrt of variance)
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # Store results
    results[0, i] = portfolio_return
    results[1, i] = portfolio_vol
    results[2, i] = portfolio_return / portfolio_vol  # Sharpe ratio (risk-adjusted)
    
# Convert results to DataFrame
results_df = pd.DataFrame({
    'Return': results[0],
    'Volatility': results[1],
    'Sharpe_Ratio': results[2]
})

# Plot efficient frontier
plt.figure(figsize=(10,6))
plt.scatter(results_df['Volatility'], results_df['Return'], c=results_df['Sharpe_Ratio'], cmap='viridis', s=5)
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility (Risk)')
plt.ylabel('Expected Return')
plt.title('Monte Carlo Simulation - Portfolio Efficient Frontier')
plt.show()


# Compute return-to-cost ratio
ratio = mean_returns / costs

# Create DataFrame for sorting
df = pd.DataFrame({'Asset': assets, 'Return': mean_returns, 'Cost': costs, 'Ratio': ratio})
df = df.sort_values(by='Ratio', ascending=False).reset_index(drop=True)

# Greedy selection
budget_remaining = budget
chosen_assets = []

for _, row in df.iterrows():
    if row['Cost'] <= budget_remaining:
        chosen_assets.append(row['Asset'])
        budget_remaining -= row['Cost']

greedy_alloc = df[df['Asset'].isin(chosen_assets)]

print("Greedy Portfolio Allocation:")
print(greedy_alloc[['Asset', 'Return', 'Cost']])
print("\nTotal Expected Return:", greedy_alloc['Return'].sum())
print("Total Cost:", greedy_alloc['Cost'].sum())


def knapsack_dp(costs, returns, budget):
    n = len(costs)
    dp = [[0 for _ in range(budget + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        for b in range(1, budget + 1):
            if costs[i-1] <= b:
                dp[i][b] = max(dp[i-1][b], returns[i-1] + dp[i-1][b - costs[i-1]])
            else:
                dp[i][b] = dp[i-1][b]

    # Backtrack to find chosen assets
    chosen = []
    b = budget
    for i in range(n, 0, -1):
        if dp[i][b] != dp[i-1][b]:
            chosen.append(i-1)
            b -= costs[i-1]

    chosen.reverse()
    return dp[n][budget], chosen

# Run DP algorithm
max_return, chosen_indices = knapsack_dp(costs, mean_returns * 1000, budget)  # scaled up
chosen_assets = [assets[i] for i in chosen_indices]

print("\nDynamic Programming Optimal Portfolio:")
for i in chosen_indices:
    print(f"{assets[i]} - Cost: {costs[i]}, Return: {mean_returns[i]:.2f}")
print(f"\nTotal Expected Return: {max_return / 1000:.2f}")
print(f"Total Cost: {sum(costs[i] for i in chosen_indices)}")

summary = pd.DataFrame({
    'Method': ['Monte Carlo (Best Sharpe)', 'Greedy Heuristic', 'Dynamic Programming'],
    'Expected Return (%)': [
        results_df.loc[results_df['Sharpe_Ratio'].idxmax(), 'Return'] * 100,
        greedy_alloc['Return'].sum() * 100,
        max_return / 10  # rescaled
    ],
    'Total Cost ($)': [
        budget,  # Monte Carlo uses all budget proportionally
        greedy_alloc['Cost'].sum(),
        sum(costs[i] for i in chosen_indices)
    ]
})

print(summary)
