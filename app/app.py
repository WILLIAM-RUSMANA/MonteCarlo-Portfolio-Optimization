import streamlit as st
import pandas as pd
import plotly.express as px
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from monte_carlo_method import monte_carlo_method
from greedy import greedy_portfolio_allocation as greedy_cont
from greedy_whole import greedy_portfolio_allocation as greedy_whole
from equal_weight import equal_weight_allocation
from dp_knapsack import dp_knapsack_portfolio_allocation
from constants import CSV_BACKTEST


# Page config
st.set_page_config(page_title="Portfolio Allocator", layout="wide")

# Title
st.title("Portfolio Allocator")

# Amount input (shared across all pages)
amount = st.number_input(
    "Amount of USD to allocate",
    min_value=100.0,
    value=100000.0,
    step=1000.0,
)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Algorithm",
    ["Greedy", "DP Knapsack", "Equal Weight"],
)


# Helper functions
def run_monte_carlo():
    return monte_carlo_method(num_simulations=3000)


@st.cache_data
def load_prices():
    df = pd.read_csv(CSV_BACKTEST, parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    return df


def projected_return_from_alloc(alloc, results):
    return sum(
        w * results[t]["mean_annual_return"] for t, w in alloc.items() if t in results
    )


# ========== PAGE 1: GREEDY ==========
if page == "Greedy":
    st.header("Greedy Algorithm")

    # Add a button to start simulation
    if st.button("Run Allocation", type="primary"):
        with st.spinner("Running Monte Carlo simulation and allocation algorithms..."):
            # Run Monte Carlo and algorithms
            results = run_monte_carlo()

            cont_alloc = greedy_cont(
                results,
                target_num_stocks=50,
                plot_results=False,
                compare_equal_weight=False,
                display_results=False,
            )

            whole_alloc = greedy_whole(
                results,
                amount,
                target_num_stocks=50,
                plot_results=False,
                compare_equal_weight=False,
                display_results=False,
            )

            shares = whole_alloc["shares"]
            cash_remaining = whole_alloc["cash_remaining"]
            stock_prices = whole_alloc["stock_prices"]

            # === CONTINUOUS WEIGHTS ===
            st.subheader("1. Continuous-weight allocation")

            cont_df = pd.DataFrame(
                {
                    "Stock": list(cont_alloc.keys()),
                    "Weight": list(cont_alloc.values()),
                }
            )

            fig_cont = px.pie(
                cont_df,
                names="Stock",
                values="Weight",
                title="Portfolio Allocation (Continuous Weights)",
            )
            st.plotly_chart(fig_cont, width="stretch")

            proj_return_cont = projected_return_from_alloc(cont_alloc, results)
            st.metric("Projected annual return (continuous)", f"{proj_return_cont:.2%}")

            # === WHOLE SHARES ===
            st.subheader("2. Whole-share allocation")

            # Calculate dollar amounts and weights
            amounts = {t: shares[t] * stock_prices[t] for t in shares}
            total_amount = sum(amounts.values()) or 1.0

            whole_df = pd.DataFrame(
                {
                    "Stock": list(amounts.keys()),
                    "Weight": [v / total_amount for v in amounts.values()],
                }
            )

            fig_whole = px.pie(
                whole_df,
                names="Stock",
                values="Weight",
                title="Portfolio Allocation (Whole Shares)",
            )
            st.plotly_chart(fig_whole, width="stretch")

            # Calculate projected return for whole share portfolio
            weights_whole = {t: amt / total_amount for t, amt in amounts.items()}
            proj_return_whole = projected_return_from_alloc(weights_whole, results)

            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Projected annual return (whole shares)", f"{proj_return_whole:.2%}"
                )
            with col2:
                st.metric("Cash remaining", f"${cash_remaining:,.2f}")

            # === SIDEBAR ===
            st.sidebar.subheader(" Shares to Buy")
            st.sidebar.caption("Sorted by number of shares (descending)")

            share_items = sorted(shares.items(), key=lambda x: x[1], reverse=True)
            for ticker, n_shares in share_items:
                st.sidebar.write(f"**{ticker}**: {n_shares} shares")

            st.sidebar.divider()
            st.sidebar.write(f" **Cash remaining**: ${cash_remaining:,.2f}")

            # === HISTORICAL PERFORMANCE ===
            st.subheader("3. Historical Stock Performance")

            prices = load_prices()
            st.line_chart(prices)
    else:
        st.info("Set amount here")

# ========== PAGE 2: ALGORITHM 2  ==========
elif page == "DP Knapsack":
    st.header("DP Knapsack Algorithm")

    if st.button("Run Allocation", type="primary", key="dp_btn"):
        with st.spinner("Running Monte Carlo and DP Knapsack allocation..."):
            results = run_monte_carlo()

            allocations, dp_results = dp_knapsack_portfolio_allocation(
                results,
                target_num_stocks=50,
                display_results=False,
            )

            # Pie chart
            df = pd.DataFrame({
                "Stock": list(allocations.keys()),
                "Weight": list(allocations.values()),
            })

            fig = px.pie(
                df,
                names="Stock",
                values="Weight",
                title="DP Knapsack Allocation",
            )
            st.plotly_chart(fig, width="stretch")

            # Metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Projected annual return",
                    f"{dp_results['portfolio_return']:.2%}",
                )
            with col2:
                st.metric(
                    "Portfolio Sharpe Ratio",
                    f"{dp_results['portfolio_sharpe']:.4f}",
                )

            # Sidebar
            st.sidebar.subheader("Allocation Weights")
            st.sidebar.caption("Sorted by weight (descending)")
            for ticker, w in sorted(allocations.items(), key=lambda x: x[1], reverse=True):
                st.sidebar.write(f"**{ticker}**: {w:.2%}")

            # Historical
            st.subheader("Historical Stock Performance")
            prices = load_prices()
            st.line_chart(prices)
    else:
        st.info("Set amount here")


# ========== PAGE 3: ALGORITHM 3  ==========
elif page == "Equal Weight":
    st.header("Equal-Weight Algorithm")

    if st.button("Run Allocation", type="primary", key="eq_btn"):
        with st.spinner(
            "Running Monte Carlo simulation and equal-weight allocation..."
        ):
            # Use same Monte Carlo results
            results = run_monte_carlo()

            allocations_eq, eq_results = equal_weight_allocation(
                results,
                display_results=False,
            )

            # Pie chart of equal weights
            eq_df = pd.DataFrame(
                {
                    "Stock": list(allocations_eq.keys()),
                    "Weight": list(allocations_eq.values()),
                }
            )

            fig_eq = px.pie(
                eq_df,
                names="Stock",
                values="Weight",
                title="Equal-Weight Portfolio Allocation",
            )
            st.plotly_chart(fig_eq, width="stretch")

            # Metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Projected annual return (equal weight)",
                    f"{eq_results['portfolio_return']:.2%}",
                )
            with col2:
                st.metric(
                    "Portfolio Sharpe Ratio (equal weight)",
                    f"{eq_results['portfolio_sharpe']:.4f}",
                )

            # Sidebar (no shares, just weights)
            st.sidebar.subheader("Equal Weights")
            st.sidebar.caption("Each selected stock has the same weight")
            for ticker, w in sorted(
                allocations_eq.items(), key=lambda x: x[1], reverse=True
            ):
                st.sidebar.write(f"**{ticker}**: {w:.2%}")

            # Historical performance
            st.subheader("Historical Stock Performance")
            prices = load_prices()
            st.line_chart(prices)
    else:
        st.info("set amount here")
