import ccxt
import pandas as pd
import time
from datetime import datetime
import csv

# Initialize the Binance exchange
exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future'  # Use this for futures trading
    }
})

symbol = 'BTC/USDT'
timeframe = '1m'  # 1-minute intervals


def fetch_order_book(symbol, limit=20):
    try:
        order_book = exchange.fetch_order_book(symbol, limit)
        timestamp = exchange.milliseconds()
        return order_book, timestamp
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None


def collect_data(duration_seconds=3600):  # Collect data for 1 hour by default
    start_time = time.time()
    filename = f"BTC_USDT_orderbook_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    # Open the file in write mode and write the header
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['timestamp', 'best_bid', 'best_ask', 'bid_volume', 'ask_volume', 'mid_price', 'spread']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        while time.time() - start_time < duration_seconds:
            order_book, timestamp = fetch_order_book(symbol)

            if order_book:
                bids = order_book['bids']
                asks = order_book['asks']

                record = {
                    'timestamp': timestamp,
                    'best_bid': bids[0][0],
                    'best_ask': asks[0][0],
                    'bid_volume': sum(bid[1] for bid in bids),
                    'ask_volume': sum(ask[1] for ask in asks),
                    'mid_price': (bids[0][0] + asks[0][0]) / 2,
                    'spread': asks[0][0] - bids[0][0],
                }
                print(record)
                # Write the record to the CSV file
                writer.writerow(record)

            time.sleep(1)  # Wait for 1 second before the next request


def main():
    print("Starting data collection...")
    collect_data()
    print("Data collection completed.")


# Run the main function
if __name__ == "__main__":
    main()
