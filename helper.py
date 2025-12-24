from constants import RFR


def calculate_sharpe_ratio(mean_return, std_return, risk_free_rate=RFR):
    """Calculate Sharpe ratio for a stock"""
    return (mean_return - risk_free_rate) / std_return if std_return > 0 else 0