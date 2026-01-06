import streamlit as st
import pandas as pd
import plotly.express as px
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from monte_carlo_method import monte_carlo_method
from algorithms.greedy import greedy_portfolio_allocation
from algorithms.equal_weight import equal_weight_allocation
from algorithms.dp_knapsack import dp_knapsack_portfolio_allocation
from constants import CSV_BACKTEST


st.set_page_config(page_title="Portfolio Allocator", layout="wide")
st.title("Portfolio Allocator")

amount = st.number_input(
    "Amount of USD to allocate",
    min_value=100.0,
    value=100000.0,
    step=1000.0,
)

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Algorithm",
    ["Greedy", "DP Knapsack", "Equal Weight"],
)


def run_monte_carlo():
    return monte_carlo_method(num_simulations=3000)


@st.cache_data
def load_prices():
    df = pd.read_csv(CSV_BACKTEST, parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    return df


# ========== PAGE 1: GREEDY ==========
if page == "Greedy":
    st.header("Greedy Sharpe Algorithm")

    if st.button("Run Allocation", type="primary", key="greedy_btn"):
        with st.spinner("Running Monte Carlo and Greedy allocation..."):
            results = run_monte_carlo()

            allocations, greedy_results = greedy_portfolio_allocation(
                results,
                target_num_stocks=50,
                display_results=False,
            )

            # Pie chart
            df = pd.DataFrame(
                {
                    "Stock": list(allocations.keys()),
                    "Weight": list(allocations.values()),
                }
            )

            fig = px.pie(
                df,
                names="Stock",
                values="Weight",
                title="Greedy Sharpe Allocation",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Projected annual return",
                    f"{greedy_results['portfolio_return']:.2%}",
                )
            with col2:
                st.metric(
                    "Portfolio Sharpe Ratio",
                    f"{greedy_results['portfolio_sharpe']:.4f}",
                )

            # Sidebar
            st.sidebar.subheader("Allocation Weights")
            st.sidebar.caption("Sorted by weight (descending)")
            for ticker, w in sorted(
                allocations.items(), key=lambda x: x[1], reverse=True
            ):
                st.sidebar.write(f"**{ticker}**: {w:.2%}")

            # Historical
            st.subheader("Historical Stock Performance")
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
            df = pd.DataFrame(
                {
                    "Stock": list(allocations.keys()),
                    "Weight": list(allocations.values()),
                }
            )

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
            for ticker, w in sorted(
                allocations.items(), key=lambda x: x[1], reverse=True
            ):
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
