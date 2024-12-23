import requests
import pandas as pd
import json
import time
import os
from typing import Dict, Any
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
from matplotlib.ticker import PercentFormatter
from py_clob_client.client import ClobClient
from dotenv import load_dotenv
from ollama import chat
from ollama import ChatResponse
import ast
from pathlib import Path

class PolymarketConfig:
    """Configuration settings for PolymarketAnalyzer"""
    GAMMA_API = "https://gamma-api.polymarket.com/events"
    ORDERS_ENDPOINT = "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/polymarket-orderbook-resync/prod/gn"
    CHAIN_ID = 137  # Polygon Mainnet
    CLOB_HOST = "https://clob.polymarket.com"
    DEFAULT_BATCH_SIZE = 500
    MIN_VOLUME_THRESHOLD = 1_000_000
    LLM_MODEL = 'llama3.2'
    BASE_PATH = Path("data/polymarket")  # Base directory for all output files
    PLOTS_DIR = BASE_PATH / "plots"      # Directory for plot images
    DATA_DIR = BASE_PATH / "data"        # Directory for CSV files
    
    @classmethod
    def setup_directories(cls) -> None:
        """Create necessary directories if they don't exist"""
        cls.BASE_PATH.mkdir(parents=True, exist_ok=True)
        cls.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_data_path(cls, filename: str) -> Path:
        """Get full path for a data file"""
        return cls.DATA_DIR / filename
    
    @classmethod
    def get_plot_path(cls, filename: str) -> Path:
        """Get full path for a plot file"""
        return cls.PLOTS_DIR / filename

class PolymarketAnalyzer:
    def __init__(self):
        """Initialize the PolymarketAnalyzer with API endpoints and client setup."""
        self.gamma_api = PolymarketConfig.GAMMA_API
        self.orders_endpoint = PolymarketConfig.ORDERS_ENDPOINT
        load_dotenv('keys/keys.env')
        PolymarketConfig.setup_directories()  # Ensure directories exist
        self.setup_clob_client()

    def setup_clob_client(self) -> None:
        """Initialize the CLOB client with API credentials."""
        key = os.getenv("PK")
        if not key:
            raise ValueError("Private key not found in environment variables")
        
        self.client = ClobClient(
            PolymarketConfig.CLOB_HOST, 
            key=key, 
            chain_id=PolymarketConfig.CHAIN_ID
        )

    def fetch_polymarket_data(self, batch_size=PolymarketConfig.DEFAULT_BATCH_SIZE):
        """
        Fetch all active Polymarket events using pagination.
        
        Args:
            batch_size (int): Number of markets to fetch per request
            
        Returns:
            pd.DataFrame: DataFrame containing market data, sorted by volume
            
        Note:
            - Uses pagination to fetch all markets
            - Includes automatic retry and delay between requests
            - Removes duplicate markets based on ID
        """
        try:
            print("🔄 Starting data fetch...")
            all_markets = []
            offset = 0
            total_fetched = 0
            
            while True:
                url = f"{self.gamma_api}?closed=false&limit={batch_size}&offset={offset}"
                print(f"Fetching events with offset {offset}...")
                
                r = requests.get(url)
                markets_list = r.json()
                
                if not markets_list:
                    print("No more events available")
                    break
                
                all_markets.extend(markets_list)
                total_fetched += len(markets_list)
                print(f"Fetched {len(markets_list)} events. Total so far: {total_fetched}")
                
                if len(markets_list) < batch_size:
                    break
                    
                offset += batch_size
                time.sleep(0.5)
            
            df = pd.DataFrame(all_markets)
            
            if 'volume' in df.columns:
                df = df.sort_values('volume', ascending=False)
            
            df = df.drop_duplicates(subset='id')
            print(f"\nTotal unique events fetched: {len(df)}")
            
            return df
            
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            return pd.DataFrame()

    def filter_markets_by_volume(self, df: pd.DataFrame, min_volume: float = PolymarketConfig.MIN_VOLUME_THRESHOLD) -> pd.DataFrame:
        """
        Filter markets based on a minimum volume threshold.
        
        Args:
            df (pd.DataFrame): Input DataFrame containing market data
            min_volume (float): Minimum volume threshold (default: 1,000,000)
            
        Returns:
            pd.DataFrame: Filtered DataFrame containing only markets above the volume threshold
        """
        print(f"📈 Filtering markets with volume > {min_volume:,}")
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        df = df.sort_values('volume', ascending=False)
        return df[df['volume'] > min_volume]

    def get_relevant_labels(self, unique_labels, num_passes=3):
        """
        Use LLM to identify relevant market labels related to politics or finance.
        
        Args:
            unique_labels (list): List of all unique labels from markets
            num_passes (int): Number of LLM passes to aggregate results (default: 3)
            
        Returns:
            list: Filtered list of relevant labels
            
        Note:
            - Uses multiple passes to improve accuracy
            - Excludes sports and entertainment labels
            - Handles both structured and unstructured LLM responses
        """
        print(f"🤖 Processing {len(unique_labels)} labels with {num_passes} LLM passes...")
        prompt = """Analyze these labels and identify those related to politics or finance, and make sure to AVOID any labels related to sports or entertainment.
        Return ONLY a Python list containing the relevant labels, formatted exactly like this:
        ['label1', 'label2', 'label3']
        No other text or explanation, just the list.
        Labels: """ + str(unique_labels)

        all_relevant_labels = set()
        
        for _ in range(num_passes):
            response = chat(
                model=PolymarketConfig.LLM_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
            )
            
            try:
                labels = ast.literal_eval(response.message.content.strip())
                all_relevant_labels.update(labels)
            except (SyntaxError, ValueError) as e:
                print(f"Error parsing response: {e}")
                content = response.message.content.strip()
                content = content.replace('[', '').replace(']', '')
                labels = [label.strip().strip("'\"") for label in content.split(',')]
                all_relevant_labels.update(labels)
            
            time.sleep(1)
        
        return list(all_relevant_labels)

    def fetch_order_history(self, condition_id, start_date="2023-10-08", batch_size=1000):
        """
        Fetch complete order history for a specific market.
        
        Args:
            condition_id (str): Market identifier
            start_date (str): Start date in YYYY-MM-DD format
            batch_size (int): Number of orders to fetch per request
            
        Returns:
            pd.DataFrame: DataFrame containing order history with timestamps, prices, and sizes
            
        Note:
            - Fetches all orders since start_date
            - Converts timestamps to datetime
            - Sorts by timestamp ascending
        """
        print(f"📜 Fetching order history for market {condition_id}")
        all_orders = []
        offset = 0
        start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
        
        while True:
            query = """
            query OrderHistory($conditionId: String!, $limit: Int!, $offset: Int!, $startTime: Int!) {
                enrichedOrderFilleds(
                    first: $limit
                    skip: $offset
                    where: {
                        market: $conditionId
                        timestamp_gte: $startTime
                    }
                    orderBy: timestamp
                    orderDirection: asc
                ) {
                    timestamp
                    price
                    size
                    side
                }
            }
            """
            
            result = self._make_request(query, {
                'conditionId': condition_id,
                'limit': batch_size,
                'offset': offset,
                'startTime': start_timestamp
            })
            
            if not result or 'enrichedOrderFilleds' not in result or not result['enrichedOrderFilleds']:
                break
            
            orders = result['enrichedOrderFilleds']
            all_orders.extend(orders)
            print(f"Fetched {len(orders)} orders. Total: {len(all_orders)}")
            
            if len(orders) < batch_size:
                break
                
            offset += batch_size
        
        if not all_orders:
            print("No orders found")
            return None
            
        df = pd.DataFrame(all_orders)
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
        df['price'] = pd.to_numeric(df['price'])
        df['size'] = pd.to_numeric(df['size'])
        df = df.sort_values('timestamp')
        
        print(f"\nData range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        return df

    def _make_request(self, query, variables):
        """
        Make GraphQL request to Goldsky API with error handling.
        
        Args:
            query (str): GraphQL query string
            variables (dict): Query variables
            
        Returns:
            dict: API response data or None if request fails
            
        Note:
            - Internal method used by other functions
            - Includes error handling and logging
        """
        try:
            response = requests.post(
                self.orders_endpoint,
                json={'query': query, 'variables': variables}
            )
            response.raise_for_status()
            return response.json().get('data', {})
        except Exception as e:
            print(f"Error making request: {str(e)}")
            return None

    def plot_price_history(self, df, title="Market Price History", resample='1H'):
        """
        Create a styled price history plot with Polymarket branding.
        
        Args:
            df (pd.DataFrame): DataFrame containing price history
            title (str): Plot title
            resample (str): Pandas resample rule for data aggregation (default: '1H')
            
        Returns:
            matplotlib.figure.Figure: The generated plot figure
            
        Note:
            - Includes Polymarket styling
            - Resamples data for smoother visualization
            - Adds shadow effects and grid
        """
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(15, 8))
        
        df_resampled = df.set_index('timestamp').resample(resample).agg({
            'price': 'last',
            'size': 'sum'
        }).fillna(method='ffill')
        
        ax.plot(df_resampled.index, df_resampled['price'], 
                color='#0066FF',
                linewidth=2,
                solid_capstyle='round',
                path_effects=[pe.SimpleLineShadow(shadow_color='gray', alpha=0.2),
                             pe.Normal()])
        
        ax.grid(True, linestyle='--', alpha=0.2, color='gray')
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        plt.xticks(rotation=0)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_alpha(0.3)
        ax.spines['bottom'].set_alpha(0.3)
        
        plt.text(0.01, 0.02, 'Source: Polymarket.com', 
                 transform=ax.transAxes, 
                 color='gray', 
                 alpha=0.6,
                 fontsize=8)
        
        plt.tight_layout()
        return fig

def process_market_data(analyzer: PolymarketAnalyzer) -> pd.DataFrame:
    """Process and filter market data"""
    markets_df = analyzer.fetch_polymarket_data()
    filtered_df = analyzer.filter_markets_by_volume(markets_df)
    
    # Save raw market data
    markets_df.to_csv(
        PolymarketConfig.get_data_path("raw_markets.csv"), 
        index=False
    )
    print(f"✨ Found {len(filtered_df)} markets above volume threshold")
    return filtered_df

def process_market_labels(filtered_df: pd.DataFrame, analyzer: PolymarketAnalyzer) -> pd.DataFrame:
    """Extract and process market labels"""
    unique_labels = list({
        tag["label"] 
        for tags in filtered_df['tags'] 
        for tag in tags
    })
    print(f"📝 Found {len(unique_labels)} unique labels")
    
    # Save unique labels
    pd.Series(unique_labels).to_csv(
        PolymarketConfig.get_data_path("unique_labels.csv"), 
        index=False
    )
    
    relevant_labels = analyzer.get_relevant_labels(unique_labels)
    print(f"🎯 Identified {len(relevant_labels)} relevant labels")
    
    relevant_markets = filtered_df[
        filtered_df['tags'].apply(
            lambda tags: any(tag['label'] in relevant_labels for tag in tags)
        )
    ].reset_index(drop=True)
    
    print(f"💎 Found {len(relevant_markets)} markets of interest")
    return relevant_markets

def process_order_history(
    analyzer: PolymarketAnalyzer, 
    clob_token_id: str, 
    start_date: str = "2024-01-01"
) -> None:
    """Process and save order history data and plots"""
    trades_df = analyzer.fetch_order_history(clob_token_id, start_date)
    if trades_df is None:
        return
    
    token_suffix = clob_token_id[-8:]
    
    # Save trade history data
    trades_df.to_csv(
        PolymarketConfig.get_data_path(f'price_history_{token_suffix}.csv'),
        index=False
    )
    print(f"Data saved to price_history_{token_suffix}.csv")
    
    # Create and save plot
    fig = analyzer.plot_price_history(trades_df, resample='1H')
    fig.savefig(
        PolymarketConfig.get_plot_path(f'price_history_{token_suffix}.png')
    )
    print(f"Plot saved to price_history_{token_suffix}.png")

def main() -> None:
    """Main execution function"""
    print("\n🚀 === Starting Polymarket Analysis === 🚀")
    analyzer = PolymarketAnalyzer()
    
    filtered_df = process_market_data(analyzer)
    markets_of_interest = process_market_labels(filtered_df, analyzer)
    
    # Save filtered markets
    markets_of_interest.to_csv(
        PolymarketConfig.get_data_path('filtered_markets.csv'),
        index=False
    )
    
    clob_token_id = "55223339147513557002753346210723654663683660449692044699329423663012565950662"
    process_order_history(analyzer, clob_token_id)
    print("✅ Analysis complete!")

if __name__ == "__main__":
    main() 