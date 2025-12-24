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
    "GOOGL", # Alphabet (Google)
    "AVGO",  # Broadcom
    "BRK-B", # Berkshire Hathaway Class B
    "WMT",   # Walmart
    "LLY",   # Eli Lilly
    "ORCL",  # Oracle
    "MA",    # Mastercard
    "HD",    # Home Depot
    "COST",  # Costco
    "MRK",   # Merck & Co.
    "PEP",   # PepsiCo
    "BAC",   # Bank of America
    "CSCO",  # Cisco Systems
    "CRM",   # Salesforce
    "SAP",   # SAP SE
    "DIS",   # Walt Disney
    "TMO",   # Thermo Fisher Scientific
    "MCD",   # McDonald's
    "ABT",   # Abbott Laboratories
    "WFC",   # Wells Fargo
    "CMCSA", # Comcast
    "INTC",  # Intel
    "VZ",    # Verizon
    "QCOM",  # Qualcomm
    "TXN",   # Texas Instruments
    "NKE",   # Nike
    "UPS",   # United Parcel Service
    "PM",    # Philip Morris International
    "UNH",   # UnitedHealth Group
    "MS",    # Morgan Stanley
    "BA",    # Boeing
    "HON",   # Honeywell
    "IBM",   # IBM
    "GE",    # General Electric
    "GS",    # Goldman Sachs
    "SBUX"   # Starbucks
]

RF_rate = 0

if __name__ == "__main__":
    start = '2013-01-01'
    end = '2025-01-01'
    start_year = start[0:4]
    end_year = end[0:4]

    complete_data = yf.download(tickers, start=start, end=end)  # Meta IPO 2012

    # Check what columns exist
    print(complete_data.columns.levels[0])

    # Use "Close" instead of "Adj Close"
    if 'Adj Close' in complete_data.columns.levels[0]:
        target_price_data = complete_data['Adj Close']
    else:
        target_price_data = complete_data['Close']

    target_price_data.to_csv(f"stocks_close_{start_year}_{end_year}.csv")

    print("✅ Saved closing prices:", target_price_data.shape)
