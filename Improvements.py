import pandas as pd

# Load your data
df = pd.read_csv('BTC_USDT_orderbook_data_20241021_213114.csv')

# Assuming you have timestamps and can identify transactions by HFTs
# You might need to adjust the logic based on available data identifiers for HFT
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# Example to calculate rolling standard deviation of price changes as a proxy for market impact
df['price_change'] = df['price'].diff()
df['rolling_std_dev'] = df['price_change'].rolling(window='5min').std()

# Calculate volume changes and correlate with price changes
df['volume_change'] = df['volume'].diff()
df['corr_price_volume'] = df['price_change'].rolling(window=10).corr(df['volume_change'])

# Save or print your results for further analysis
df.to_csv('enhanced_market_data.csv')
print(df.head())