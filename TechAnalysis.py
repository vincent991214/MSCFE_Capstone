import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import VECM
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf  # Changed import location
from statsmodels.regression.linear_model import OLS
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')


# Load and preprocess data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df


# Calculate basic metrics
def calculate_metrics(df):
    # Returns and volume
    df['returns'] = df['mid_price'].pct_change()
    df['total_volume'] = df['bid_volume'] + df['ask_volume']
    df['dollar_volume'] = df['mid_price'] * df['total_volume']

    # Volatility
    df['volatility'] = df['returns'].rolling(window=10).std()

    # Amihud Illiquidity
    df['amihud'] = np.abs(df['returns']) / df['dollar_volume']

    # Kyle's Lambda
    df['price_change'] = df['mid_price'].diff()
    df['kyles_lambda'] = df['price_change'].abs() / df['total_volume']

    return df


# EDA functions
def plot_time_series_decomposition(df):
    decomposition = seasonal_decompose(df['mid_price'], period=60)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 20))

    decomposition.observed.plot(ax=ax1)
    ax1.set_title('Observed')
    decomposition.trend.plot(ax=ax2)
    ax2.set_title('Trend')
    decomposition.seasonal.plot(ax=ax3)
    ax3.set_title('Seasonal')
    decomposition.resid.plot(ax=ax4)
    ax4.set_title('Residual')

    plt.tight_layout()
    return fig


def plot_volume_analysis(df):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    df['total_volume'].plot(ax=ax1)
    ax1.set_title('Trading Volume Over Time')

    sns.boxplot(data=df['total_volume'].resample('5T').mean(), ax=ax2)
    ax2.set_title('Volume Distribution (5-minute intervals)')

    plt.tight_layout()
    return fig


def plot_liquidity_metrics_relationship(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Remove infinite values and outliers for better visualization
    df_clean = df.copy()
    df_clean['amihud'] = df_clean['amihud'].replace([np.inf, -np.inf], np.nan)
    df_clean['kyles_lambda'] = df_clean['kyles_lambda'].replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.dropna(subset=['amihud', 'kyles_lambda'])

    # Calculate percentiles for outlier removal
    amihud_99th = np.percentile(df_clean['amihud'], 99)
    lambda_99th = np.percentile(df_clean['kyles_lambda'], 99)

    df_clean = df_clean[
        (df_clean['amihud'] < amihud_99th) &
        (df_clean['kyles_lambda'] < lambda_99th)
        ]

    sns.regplot(data=df_clean, x='amihud', y='kyles_lambda', ax=ax1)
    ax1.set_title("Amihud vs Kyle's Lambda")

    correlation = df_clean['amihud'].corr(df_clean['kyles_lambda'])
    sns.heatmap([[1, correlation], [correlation, 1]],
                annot=True, cmap='coolwarm',
                xticklabels=['Amihud', "Kyle's λ"],
                yticklabels=['Amihud', "Kyle's λ"],
                ax=ax2)
    ax2.set_title('Correlation Matrix')

    plt.tight_layout()
    return fig


# Enhanced VECM analysis
def perform_vecm_analysis(df):
    # Simulate second exchange price with realistic correlation
    df['exchange2_price'] = df['mid_price'] + np.random.normal(0, df['mid_price'].std() * 0.01, len(df))

    # Prepare data for VECM
    price_data = df[['mid_price', 'exchange2_price']].dropna()

    # Fit VECM
    model = VECM(price_data, k_ar_diff=2, deterministic="co")
    results = model.fit()

    # Make predictions
    predictions = results.predict(steps=10)

    return results, predictions


def plot_vecm_results(df, results, predictions):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

    # Plot actual vs predicted
    ax1.plot(df.index[-100:], df['mid_price'][-100:], label='Actual')
    ax1.plot(df.index[-10:], predictions[:, 0], 'r--', label='Predicted')
    ax1.set_title('VECM Price Prediction')
    ax1.legend()

    # Plot impulse response
    irf = results.irf(10)
    irf_data = irf.irfs
    periods = range(len(irf_data))

    # Plot IRF for each variable
    ax2.plot(periods, irf_data[:, 0, 0], label='Response of Exchange 1 to Exchange 1')
    ax2.plot(periods, irf_data[:, 0, 1], label='Response of Exchange 1 to Exchange 2')
    ax2.plot(periods, irf_data[:, 1, 0], label='Response of Exchange 2 to Exchange 1')
    ax2.plot(periods, irf_data[:, 1, 1], label='Response of Exchange 2 to Exchange 2')

    ax2.set_title('Impulse Response Function')
    ax2.set_xlabel('Periods')
    ax2.set_ylabel('Response')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    return fig



def calculate_price_discovery_metrics(df):
    """Calculate additional price discovery metrics"""
    # Calculate price efficiency ratio
    returns = df['mid_price'].pct_change()
    variance_ratio = returns.var() / returns.rolling(window=5).mean().var()

    # Calculate autocorrelation of returns
    acf_values = acf(returns.dropna(), nlags=10)

    return {
        'variance_ratio': variance_ratio,
        'autocorrelation': acf_values
    }


def plot_additional_analytics(df):
    """Generate additional analytical plots"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Plot return distribution
    sns.histplot(df['returns'].dropna(), kde=True, ax=ax1)
    ax1.set_title('Return Distribution')

    # Plot volatility clustering
    df['volatility'].plot(ax=ax2)
    ax2.set_title('Volatility Clustering')

    # Plot bid-ask spread over time
    df['spread'].plot(ax=ax3)
    ax3.set_title('Bid-Ask Spread Over Time')

    # Plot volume-price relationship
    sns.scatterplot(data=df, x='total_volume', y='returns', ax=ax4, alpha=0.5)
    ax4.set_title('Volume-Return Relationship')

    plt.tight_layout()
    return fig


def main():
    try:
        # Load data
        print("Loading data...")
        df = load_and_preprocess_data("BTC_USDT_orderbook_data_20241021_213114.csv")
        df = calculate_metrics(df)

        # Generate all plots
        print("Generating time series decomposition...")
        time_series_fig = plot_time_series_decomposition(df)
        time_series_fig.savefig('time_series_decomposition.png')

        print("Generating volume analysis...")
        volume_fig = plot_volume_analysis(df)
        volume_fig.savefig('volume_analysis.png')

        print("Generating liquidity metrics...")
        liquidity_fig = plot_liquidity_metrics_relationship(df)
        liquidity_fig.savefig('liquidity_metrics.png')

        # VECM analysis
        print("Performing VECM analysis...")
        results, predictions = perform_vecm_analysis(df)
        vecm_fig = plot_vecm_results(df, results, predictions)
        vecm_fig.savefig('vecm_analysis.png')

        # Print summary statistics
        print("\nSummary Statistics:")
        print(df.describe())

        print("\nVECM Results:")
        print(results.summary())

        print("\nAnalysis completed successfully!")

        # Add new analytics
        print("Calculating price discovery metrics...")
        price_discovery_metrics = calculate_price_discovery_metrics(df)

        print("Generating additional analytics...")
        additional_analytics_fig = plot_additional_analytics(df)
        additional_analytics_fig.savefig('additional_analytics.png')

        # Print additional metrics
        print("\nPrice Discovery Metrics:")
        print(f"Variance Ratio: {price_discovery_metrics['variance_ratio']}")
        print("Autocorrelation Values:")
        print(price_discovery_metrics['autocorrelation'])

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()