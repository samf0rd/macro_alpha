"""
Macro-Alpha Forecast Engine - Data Harvester (ETL Pipeline)
============================================================

This script fetches live market and macroeconomic data, merges them on
business days, and saves clean datasets for model training.

Author: Samuel Garcia
Date: February 2026

Required packages:
    pip install yfinance pandas-datareader pyarrow

Usage:
    python data_pipeline.py
"""

import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr

# Suppress pandas_datareader warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO, # Set the logging level to INFO (INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s', # Set the format of the log message
    handlers=[
        logging.FileHandler('data_pipeline.log'), # Log to a file
        logging.StreamHandler() # Log to the console
    ]
)
logger = logging.getLogger(__name__) # Get the logger for the current module


class DataHarvester:
    """
    Fetches and merges market and macroeconomic data for ML pipeline.
    
    Attributes:
        start_date (str): Start date for data fetch (YYYY-MM-DD)
        end_date (str): End date for data fetch (YYYY-MM-DD)
        data_dir (Path): Directory to save processed data
    """
    
    def __init__(self, start_date='2010-01-01', end_date=None, data_dir='data'):
        """
        Initialize the DataHarvester.
        
        Args:
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format (default: today)
            data_dir (str): Directory path for saving data
        """
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        logger.info(f"DataHarvester initialized: {self.start_date} to {self.end_date}")
    
    def fetch_market_data(self):
        """
        Fetch S&P 500 and VIX data from Yahoo Finance.
        
        Returns:
            pd.DataFrame: Market data with columns [close_sp500, vix]
        """
        logger.info("Fetching market data from Yahoo Finance...")  # Log the start of the market data fetch
        
        try:
            # Fetch S&P 500
            # Download the S&P 500 index data from Yahoo Finance
            sp500 = yf.download('^GSPC', start=self.start_date, end=self.end_date, 
                               progress=False)
            
            # Handle multi-level columns from yfinance
            if isinstance(sp500.columns, pd.MultiIndex):
                sp500.columns = sp500.columns.get_level_values(0)
            # Rename the 'Close' column to 'close_sp500'
            sp500 = sp500[['Close']].rename(columns={'Close': 'close_sp500'})
            
            # Fetch VIX
            vix = yf.download('^VIX', start=self.start_date, end=self.end_date,
                             progress=False)
            
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.get_level_values(0)
            
            vix = vix[['Close']].rename(columns={'Close': 'vix'})
            
            # Merge on date index
            market_data = sp500.join(vix, how='inner')
            
            # Ensure timezone-naive datetime index
            if market_data.index.tz is not None:
                market_data.index = market_data.index.tz_localize(None)
            
            logger.info(f"Market data fetched: {len(market_data)} rows, "
                       f"{market_data.index.min()} to {market_data.index.max()}")  # Log the number of rows and the date range
            
            return market_data
        
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")   # Log the error if the market data fetch fails
            raise
    
    def fetch_macro_data(self):
        """
        Fetch macroeconomic indicators from FRED.
        
        Returns:
            pd.DataFrame: Macro data with columns for yields, rates, and inflation
        """
        logger.info("Fetching macroeconomic data from FRED...")
        
        # FRED series codes
        fred_series = {
            'DGS10': 'yield_10y',        # 10-Year Treasury
            'DGS2': 'yield_2y',          # 2-Year Treasury
            'DFF': 'fed_funds_rate',     # Federal Funds Effective Rate (replaced DFEDTARR)
            'T10Y2Y': 'yield_spread',    # 10Y-2Y Spread (recession indicator)
            'CPIAUCSL': 'cpi'            # Consumer Price Index
        }
        
        macro_data = pd.DataFrame()
        
        try:
            for fred_code, column_name in fred_series.items():
                try:
                    series = pdr.DataReader(fred_code, 'fred', 
                                           self.start_date, self.end_date)
                    series.columns = [column_name]
                    
                    if macro_data.empty:
                        macro_data = series
                    else:
                        macro_data = macro_data.join(series, how='outer')
                    
                    logger.info(f"  [OK] Fetched {column_name} ({fred_code})")
                
                except Exception as e:
                    logger.warning(f"  [FAILED] Failed to fetch {column_name} ({fred_code}): {e}")
            
            # Calculate month-over-month CPI change (inflation rate)
            if 'cpi' in macro_data.columns:
                macro_data['inflation_mom'] = macro_data['cpi'].pct_change() * 100
            
            logger.info(f"Macro data fetched: {len(macro_data)} rows")
            
            return macro_data
        
        except Exception as e:
            logger.error(f"Error fetching macro data: {e}")
            raise
    
    def merge_and_clean(self, market_data, macro_data):
        """
        Merge market and macro data, handling business days and missing values.
        
        Critical rules:
        1. Use forward-fill (ffill) for macro data - NO look-ahead bias
        2. Drop rows with missing market data (can't predict without price)
        3. Keep only business days where market was open
        
        Args:
            market_data (pd.DataFrame): Market data from fetch_market_data()
            macro_data (pd.DataFrame): Macro data from fetch_macro_data()
        
        Returns:
            pd.DataFrame: Clean, merged dataset
        """
        logger.info("Merging and cleaning data...")
        
        # Merge on date index (outer join to keep all market days)
        df = market_data.join(macro_data, how='left')
        
        # Forward-fill macro data (macro indicators are published with lag)
        # This prevents look-ahead bias: we use the LAST KNOWN value
        macro_cols = [col for col in df.columns if col not in ['close_sp500', 'vix']]
        df[macro_cols] = df[macro_cols].ffill()
        
        # Backward-fill ONLY for the first few rows (to handle initial NaNs)
        df[macro_cols] = df[macro_cols].bfill(limit=5)
        
        # Drop any remaining rows with missing data
        initial_rows = len(df)
        df = df.dropna()
        dropped_rows = initial_rows - len(df)
        
        if dropped_rows > 0:
            logger.warning(f"Dropped {dropped_rows} rows due to missing data")
        
        # Ensure data is sorted by date
        df = df.sort_index()
        
        # Data quality check
        logger.info("Data quality check:")
        logger.info(f"  - Date range: {df.index.min()} to {df.index.max()}")
        logger.info(f"  - Total rows: {len(df)}")
        logger.info(f"  - Columns: {list(df.columns)}")
        logger.info(f"  - Missing values: {df.isnull().sum().sum()}")
        
        return df
    
    def save_to_disk(self, df, filename='market_macro_data.parquet'):
        """
        Save the clean dataset to disk in Parquet format.
        
        Args:
            df (pd.DataFrame): Clean dataset
            filename (str): Output filename
        """
        filepath = self.data_dir / filename
        
        try:
            df.to_parquet(filepath, engine='pyarrow', compression='snappy')
            logger.info(f"[SUCCESS] Data saved to: {filepath}")
            logger.info(f"  File size: {filepath.stat().st_size / 1024:.2f} KB")
        
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            # Fallback to CSV if parquet fails
            csv_filepath = self.data_dir / filename.replace('.parquet', '.csv')
            df.to_csv(csv_filepath)
            logger.info(f"[FALLBACK] Data saved to CSV at {csv_filepath}")
    
    def run_pipeline(self):
        """
        Execute the full ETL pipeline.
        
        Returns:
            pd.DataFrame: Clean, merged dataset
        """
        logger.info("="*60)
        logger.info("Starting ETL Pipeline")
        logger.info("="*60)
        
        # Step 1: Fetch market data
        market_data = self.fetch_market_data()
        
        # Step 2: Fetch macro data
        macro_data = self.fetch_macro_data()
        
        # Step 3: Merge and clean
        clean_data = self.merge_and_clean(market_data, macro_data)
        
        # Step 4: Save to disk
        self.save_to_disk(clean_data)
        
        logger.info("="*60)
        logger.info("ETL Pipeline Complete!")
        logger.info("="*60)
        
        # Display sample
        print("\n[SAMPLE] First 5 rows of cleaned data:")
        print(clean_data.head())
        print("\n[SAMPLE] Last 5 rows of cleaned data:")
        print(clean_data.tail())
        
        return clean_data


def main():
    """Main execution function."""
    # Initialize and run the pipeline
    harvester = DataHarvester(
        start_date='2010-01-01',
        end_date=None,  # Will use today's date
        data_dir='data'
    )
    
    df = harvester.run_pipeline()
    
    print(f"\n[SUCCESS] Dataset ready with {len(df)} rows and {len(df.columns)} columns")
    print(f"[FILE] Saved to: data/market_macro_data.parquet")


if __name__ == "__main__":
    main()
