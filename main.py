import yfinance as yf

tickers = [
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "NVDA",  # Nvidia
    "JPM",   # JPMorgan Chase
    "V",     # Visa
    "AMZN",  # Amazon
    "META",  # Meta Platforms
    "JNJ",   # Johnson & Johnson
    "PFE",   # Pfizer
    "XOM",   # ExxonMobil
    "CVX",   # Chevron
    "CAT",   # Caterpillar
    "KO",    # Coca-Cola
    "PG",    # Procter & Gamble
    "GOOGL"  # Alphabet (Google)
]

complete_data = yf.download(tickers, start='2015-01-01', end='2025-11-02')

# Check what columns exist
print(complete_data.columns.levels[0])

# Use "Close" instead of "Adj Close"
if 'Adj Close' in complete_data.columns.levels[0]:
    target_price_data = complete_data['Adj Close']
else:
    target_price_data = complete_data['Close']

target_price_data.to_csv("stocks_close_2015_2025.csv")

print("✅ Saved closing prices:", target_price_data.shape)

