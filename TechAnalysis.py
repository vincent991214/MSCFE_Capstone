import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import VECM
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("BTC_USDT_orderbook_data_20240930_235535.csv", index_col='timestamp', parse_dates=True)

# Calculate returns and dollar volume
df['returns'] = df['mid_price'].pct_change()
df['dollar_volume'] = df['mid_price'] * (df['bid_volume'] + df['ask_volume'])

# Amihud Illiquidity Ratio
def calculate_amihud(returns, dollar_volume, window=5):
    return np.abs(returns) / dollar_volume.rolling(window=window).mean()

df['amihud'] = calculate_amihud(df['returns'], df['dollar_volume'])

# Kyle's Lambda
def calculate_kyles_lambda(price_change, volume, window=5):
    return np.abs(price_change).rolling(window=window).sum() / volume.rolling(window=window).sum()

df['price_change'] = df['mid_price'].diff()
df['kyles_lambda'] = calculate_kyles_lambda(df['price_change'], df['bid_volume'] + df['ask_volume'])

# VECM
df['exchange2_price'] = df['mid_price'] + np.random.normal(0, 0.1, len(df))  # Simulated second exchange

# Prepare data for VECM
price_data = df[['mid_price', 'exchange2_price']].dropna()

# Fit VECM model
model = VECM(price_data, k_ar_diff=2, deterministic="co")
results = model.fit()

# Print VECM summary
print(results.summary())

# Plot results
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

ax1.plot(df.index, df['amihud'])
ax1.set_title('Amihud Illiquidity Ratio')
ax1.set_xlabel('Time')
ax1.set_ylabel('Illiquidity')

ax2.plot(df.index, df['kyles_lambda'])
ax2.set_title("Kyle's Lambda")
ax2.set_xlabel('Time')
ax2.set_ylabel('Price Impact')

ax3.plot(price_data.index, price_data['mid_price'], label='Exchange 1')
ax3.plot(price_data.index, price_data['exchange2_price'], label='Exchange 2')
ax3.set_title('Price Discovery')
ax3.set_xlabel('Time')
ax3.set_ylabel('Price')
ax3.legend()

plt.tight_layout()
plt.show()