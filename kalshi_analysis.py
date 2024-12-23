import kalshi_python
from KalshiClientsBaseV2ApiKey import ExchangeClient
import requests
import pandas as pd
import json
import time
import uuid
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.ticker import PercentFormatter
from tqdm import tqdm
from pathlib import Path
import os

# Constants
API_BASE = "https://api.elections.kalshi.com/trade-api/v2"
TRADE_API_ENDPOINT = f"{API_BASE}/markets/trades"
EVENTS_API_ENDPOINT = f"{API_BASE}/events"
BASE_PATH = Path("data/kalshi")  # Base directory for all output files
PLOTS_DIR = BASE_PATH / "plots"  # Directory for plot images
DATA_DIR = BASE_PATH / "data"    # Directory for CSV files

class KalshiAnalyzer:
    """
    A class to analyze Kalshi market data and generate visualizations.
    
    Attributes:
        key_id (str): API key identifier
        private_key_path (str): Path to private key file
        api_base (str): Base URL for Kalshi API
        client (ExchangeClient): Initialized Kalshi client
    """
    
    def __init__(self, key_id: str, private_key_path: str, api_base: str = API_BASE) -> None:
        self.key_id = key_id
        self.private_key_path = private_key_path
        self.api_base = api_base
        self.client = self._initialize_client()
        
        # Create all required directories
        BASE_PATH.mkdir(parents=True, exist_ok=True)
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    def _load_private_key_from_file(self):
        with open(self.private_key_path, "rb") as key_file:
            private_key = serialization.load_pem_private_key(
                key_file.read(),
                password=None,
                backend=default_backend()
            )
        return private_key

    def _initialize_client(self):
        private_key = self._load_private_key_from_file()
        return ExchangeClient(exchange_api_base=self.api_base, key_id=self.key_id, private_key=private_key)

    def fetch_kalshi_data(self, limit: int = 200) -> pd.DataFrame:
        """
        Fetch all Kalshi events data using pagination
        
        Args:
            limit: Number of events to fetch per request (max 200)
            
        Returns:
            pd.DataFrame: DataFrame containing events data
            
        Raises:
            requests.exceptions.RequestException: If API request fails
        """
        try:
            print("\n📊 Fetching Kalshi events data...")
            all_events = []
            cursor = None
            
            with tqdm(desc="Fetching events", unit="event") as pbar:
                while True:
                    params = {
                        'status': 'open',
                        'with_nested_markets': 'true',
                        'limit': limit
                    }
                    if cursor:
                        params['cursor'] = cursor
                    
                    r = requests.get(
                        EVENTS_API_ENDPOINT,
                        params=params,
                        headers={'accept': 'application/json'}
                    )
                    r.raise_for_status()
                    
                    response_data = r.json()
                    events_list = response_data.get('events', [])
                    
                    # If no events returned, we've reached the end
                    if not events_list:
                        print("✅ All events retrieved")
                        break
                    
                    all_events.extend(events_list)
                    
                    # Get next cursor
                    cursor = response_data.get('cursor')
                    
                    # If no cursor, we've reached the end
                    if not cursor:
                        break
                        
                    # Optional: Add delay to avoid rate limiting
                    time.sleep(0.1)
                    
            print(f"📈 Total unique events fetched: {len(all_events)}")
            
            # Convert to DataFrame
            df = pd.json_normalize(all_events)
            return df
        
        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching data: {e}")
            return pd.DataFrame()  # Return empty DataFrame on error

    def get_markets_by_categories(self, df, categories=['Financials', 'Economics', 'Politics']):
        """
        Filter markets by specified categories
        
        Args:
            df (pd.DataFrame): DataFrame containing market data
            categories (list): List of categories to filter by
        
        Returns:
            pd.DataFrame: Filtered DataFrame containing only markets in specified categories
        """
        if 'category' not in df.columns:
            print("Warning: 'category' column not found in DataFrame")
            return df
        
        mask = df['category'].isin(categories)
        filtered_df = df[mask].copy()
        print(f"Found {len(filtered_df)} markets in categories: {categories}")
        return filtered_df

    def get_active_markets(self, df):
        """
        Filter for currently active markets
        
        Args:
            df (pd.DataFrame): DataFrame containing market data
        
        Returns:
            pd.DataFrame: Filtered DataFrame containing only active markets
        """
        if 'status' not in df.columns:
            print("Warning: 'status' column not found in DataFrame")
            return df
        
        active_df = df[df['status'] == 'active'].copy()
        print(f"Found {len(active_df)} active markets")
        return active_df

    def filter_and_analyze_markets(self, df, categories=None, min_volume=100000):
        """
        Filter markets by categories and minimum volume, then analyze key metrics
        
        Args:
            df (pd.DataFrame): DataFrame containing market data
            categories (list): List of categories to filter by
            min_volume (int): Minimum trading volume threshold
        
        Returns:
            pd.DataFrame: Filtered and analyzed DataFrame
        """
        # Start with active markets
        filtered_df = self.get_active_markets(df)
        
        # Filter by categories if specified
        if categories:
            filtered_df = self.get_markets_by_categories(filtered_df, categories)
        
        # Filter by minimum volume
        if 'volume' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['volume'] >= min_volume].copy()
            print(f"Found {len(filtered_df)} markets with volume >= {min_volume}")
        
        # Calculate additional metrics
        if 'yes_price' in filtered_df.columns:
            filtered_df['probability'] = filtered_df['yes_price'] / 100
        
        if 'volume' in filtered_df.columns and 'open_interest' in filtered_df.columns:
            filtered_df['turnover_ratio'] = filtered_df['volume'] / filtered_df['open_interest']
        
        # Sort by volume descending
        if 'volume' in filtered_df.columns:
            filtered_df = filtered_df.sort_values('volume', ascending=False)
        
        return filtered_df

    def fetch_market_trades(self, ticker: str, limit: int = 1000, 
                          days_back: int = 100, save_csv: bool = True) -> pd.DataFrame:
        max_ts = int(time.time())
        min_ts = max_ts - (days_back * 24 * 60 * 60)
        
        params = {
            'ticker': ticker,
            'limit': limit,
            'min_ts': min_ts,
            'max_ts': max_ts
        }
        
        try:
            all_trades = []
            cursor = None
            
            while True:
                if cursor:
                    params['cursor'] = cursor
                    
                response = requests.get(TRADE_API_ENDPOINT, params=params)
                response.raise_for_status()
                
                data = response.json()
                trades = data.get('trades', [])
                
                if not trades:
                    break
                    
                all_trades.extend(trades)
                cursor = data.get('cursor')
                
                if not cursor:
                    break
                    
                time.sleep(0.1)
            
            # Convert to DataFrame
            trades_df = pd.DataFrame(all_trades) if all_trades else pd.DataFrame()
            
            # Save to CSV if requested
            if save_csv and not trades_df.empty:
                end_date = pd.Timestamp.now().strftime('%Y%m%d')
                start_date = (pd.Timestamp.now() - pd.Timedelta(days=days_back)).strftime('%Y%m%d')
                filename = f'kalshi_trades_{ticker}_{start_date}_to_{end_date}.csv'
                filepath = DATA_DIR / filename  # Updated to use Path
                trades_df.to_csv(filepath, index=False)
                print(f"\nSaved trades to: {filepath}")
            
            return trades_df
        
        except Exception as e:
            print(f"Error fetching trades: {str(e)}")
            return pd.DataFrame()

    def plot_kalshi_price_history(self, df, title=None, resample='1h', display=True, save=True):
        """
        Plot price history for Kalshi market data
        
        Args:
            df (pd.DataFrame): DataFrame with columns: created_time, yes_price, count
            title (str): Plot title (optional)
            resample (str): Resampling frequency for smoothing ('1h' for hourly, '1D' for daily, etc.)
            display (bool): Whether to display the plot
            save (bool): Whether to save the plot
        """
        print("\n📊 Generating price history plot...")
        
        # Create figure with white background
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Ensure created_time is datetime
        df['created_time'] = pd.to_datetime(df['created_time'])
        
        # Resample data to smooth the line
        df_resampled = df.set_index('created_time').resample(resample).agg({
            'yes_price': 'last',
            'count': 'sum'
        }).fillna(method='ffill')
        
        # Plot the price line
        ax.plot(df_resampled.index, df_resampled['yes_price'] / 100,  # Convert to probability
                color='#FF4B4B',  # Kalshi red
                linewidth=2,
                solid_capstyle='round',
                path_effects=[pe.SimpleLineShadow(shadow_color='gray', alpha=0.2),
                             pe.Normal()])
        
        # Add volume bars
        volume_height = 0.2
        max_count = df_resampled['count'].max()
        ax.fill_between(df_resampled.index, 
                       0, 
                       df_resampled['count'] / max_count * volume_height,
                       alpha=0.2,
                       color='gray')
        
        # Styling
        ax.grid(True, linestyle='--', alpha=0.2, color='gray')
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        # Format axes
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        plt.xticks(rotation=0)
        
        # Clean up spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_alpha(0.3)
        ax.spines['bottom'].set_alpha(0.3)
        
        # Add title if provided
        if title:
            plt.title(title, pad=20, fontsize=14)
        else:
            plt.title(f"Market: {df['ticker'].iloc[0]}", pad=20, fontsize=14)
        
        # Add source
        plt.text(0.01, 0.02, 'Source: Kalshi.com', 
                 transform=ax.transAxes, 
                 color='gray', 
                 alpha=0.6,
                 fontsize=8)
        
        plt.tight_layout()
        
        # Save the plot if requested
        if save:
            filename = f'kalshi_price_history_{df["ticker"].iloc[0]}.png'
            filepath = PLOTS_DIR / filename  # Updated to use Path
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"🖼️  Plot saved as: {filepath}")
        
        # Display the plot if requested
        if display:
            print("📈 Displaying plot...")
            plt.show()
        else:
            plt.close()
        
        return fig

def main() -> None:
    # Load configuration
    config = {
        'KEY_ID': "b88d11b1-f76f-46fc-b2e7-53cba60a15f8",
        'KEY_PATH': '/Users/matiasgomezpaz/Library/CloudStorage/GoogleDrive-gomezpaz.mat@gmail.com/My Drive/mgomezpaz/M2024/BYU/Polyshi/keys/api_key.key',
        'CATEGORIES': ['Politics', 'Science and Technology', 'Finance'],
        'MIN_VOLUME': 100000,
        'EXAMPLE_TICKER': "POPVOTE-24-R"
    }
    
    # Initialize analyzer
    analyzer = KalshiAnalyzer(config['KEY_ID'], config['KEY_PATH'])
    
    # Fetch initial data
    kalshi_df = analyzer.fetch_kalshi_data(limit=200)
    
    # Filter and analyze markets
    filtered_df = analyzer.filter_and_analyze_markets(kalshi_df, categories=config['CATEGORIES'], min_volume=config['MIN_VOLUME'])
    
    # Example: Fetch and plot trades for a specific market
    ticker = config['EXAMPLE_TICKER']
    trades_df = analyzer.fetch_market_trades(ticker, limit=1000, days_back=100, save_csv=True)
    
    if not trades_df.empty:
        analyzer.plot_kalshi_price_history(
            trades_df,
            title="Will Republicans win the popular vote in 2024?",
            resample='1h',
            display=False,
            save=True
        )

if __name__ == "__main__":
    main() 