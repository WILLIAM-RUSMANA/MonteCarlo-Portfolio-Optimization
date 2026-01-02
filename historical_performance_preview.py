import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # -----------------------------
    # Load data
    # -----------------------------
    df = pd.read_csv("data/stocks_close_2013_2025.csv")

    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    # -----------------------------
    # Percentage returns (normalized)
    # -----------------------------
    df_pct = (df / df.iloc[0] - 1) * 100

    # -----------------------------
    # Create dynamic color map
    # -----------------------------
    tickers = df_pct.columns.tolist()
    cmap = plt.get_cmap("tab20", len(tickers))
    colors = {ticker: cmap(i) for i, ticker in enumerate(tickers)}

    # -----------------------------
    # Figure 1: All stocks (with NVDA)
    # -----------------------------
    plt.figure(figsize=(14, 8))

    sorted_columns = df_pct.iloc[-1].sort_values(ascending=False).index

    for column in sorted_columns:
        plt.plot(
            df_pct.index,
            df_pct[column],
            label=column,
            color=colors[column],
            linewidth=1.3
        )

    plt.title("Historical Stock Returns – All Stocks (2015–2025)",
            fontsize=16, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Percentage Return (%)")
    plt.legend(ncol=3, fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Figure 2: Excluding NVDA
    # -----------------------------
    if "NVDA" in df_pct.columns:
        df_pct_no_nvda = df_pct.drop(columns=["NVDA"])
    else:
        df_pct_no_nvda = df_pct.copy()

    plt.figure(figsize=(14, 8))

    sorted_columns_no_nvda = (
        df_pct_no_nvda.iloc[-1].sort_values(ascending=False).index
    )

    for column in sorted_columns_no_nvda:
        plt.plot(
            df_pct_no_nvda.index,
            df_pct_no_nvda[column],
            label=column,
            color=colors[column],
            linewidth=1.3
        )

    plt.title("Historical Stock Returns – Excluding NVDA (2015–2025)",
            fontsize=16, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Percentage Return (%)")
    plt.legend(ncol=3, fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
