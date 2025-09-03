import os
import json
import warnings
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from pymongo import MongoClient, DESCENDING
from prophet import Prophet
import functions_framework

# Suppress warnings for cleaner logs
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
MONGO_URI = os.environ.get("MONGO_DB_URI")
DB_NAME = os.environ.get("DB_NAME", "userdata-dev")
HISTORICAL_COLLECTION_NAME = "crypto_historical_data"
FORECAST_COLLECTION_NAME = "prophet_forecasts"

ASSET_CONFIGS = {
    'BTC-USD': {
        'params': {'changepoint_prior_scale': 1, 'seasonality_mode': 'multiplicative'},
        'dynamic_cap_multiplier': 1,
        'start_date': '2020-01-01'
    },
    'ETH-USD': {
        'params': {'changepoint_prior_scale': 0.2, 'seasonality_mode': 'multiplicative'},
        'dynamic_cap_multiplier': 1,
        'start_date': '2020-01-01'
    },
    'BNB-USD': {
        'params': {'changepoint_prior_scale': 0.2, 'seasonality_mode': 'multiplicative'},
        'dynamic_cap_multiplier': 1,
        'start_date': '2021-08-01'  
    },
    'SOL-USD': {
        'params': {'changepoint_prior_scale': 0.5, 'seasonality_mode': 'additive'},
        'dynamic_cap_multiplier': 1.5,
        'start_date': '2021-06-01'
    },
    'XRP-USD': {
        'params': {'changepoint_prior_scale': 0.2, 'seasonality_mode': 'additive'},
        'dynamic_cap_multiplier': 7,
        'start_date': '2021-01-01'
    },
    'ADA-USD': {
        'params': {'changepoint_prior_scale': 0.2, 'seasonality_mode': 'additive'},
        'dynamic_cap_multiplier': 7,
        'start_date': '2020-09-01'
    },
}


COINGECKO_MAP = {
    'BTC-USD': 'bitcoin', 'ETH-USD': 'ethereum', 'BNB-USD': 'binancecoin',
    'XRP-USD': 'ripple', 'ADA-USD': 'cardano', 'SOL-USD': 'solana'
}

@functions_framework.http
def main(request):
    """ Main Google Cloud Function entry point. """
    print("--- Starting Cryptocurrency Forecast Pipeline with Unified Model ---")
    if not MONGO_URI: return "Configuration Error: MONGO_DB_URI not set.", 500

    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        historical_collection = db[HISTORICAL_COLLECTION_NAME]
        forecast_collection = db[FORECAST_COLLECTION_NAME]
        client.admin.command('ping')
        print("MongoDB connection successful.")
    except Exception as e:
        return f"Database Connection Error: {e}", 500

    for ticker, config in ASSET_CONFIGS.items():
        print(f"\n--- Processing Ticker: {ticker} ---")
        try:
            update_data_from_coingecko(historical_collection, ticker)
            
            manual_start_date = config.get('start_date')
            full_df = get_all_data_from_mongo(historical_collection, ticker, manual_start_date)
            
            if full_df.empty or len(full_df) < 365:
                print(f"Insufficient data for {ticker}. Skipping forecast.")
                continue

            forecast_df = generate_prophet_forecast(full_df, ticker, config)
            
            store_forecast_in_mongo(forecast_collection, forecast_df, ticker, full_df)

        except Exception as e:
            print(f"ERROR processing {ticker}: {e}")
            continue

    print("\n--- Cryptocurrency Forecast Pipeline Finished Successfully ---")
    return "Pipeline executed successfully", 200


def update_data_from_coingecko(collection, ticker):
    """ Fetches new price data from CoinGecko. (No changes here) """
    print(f"Checking for new data for {ticker}...")
    last_entry = collection.find_one({'coin_ticker': ticker}, sort=[('date', DESCENDING)])
    start_date = (last_entry['date'] + timedelta(days=1)) if last_entry else datetime(2020, 1, 1)
    end_date = datetime.now()

    if start_date.date() >= end_date.date():
        print("Data is already up to date.")
        return

    print(f"Fetching data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}.")
    cg_id = COINGECKO_MAP.get(ticker)
    if not cg_id: return

    from_ts, to_ts = int(start_date.timestamp()), int(end_date.timestamp())
    url = f"https://api.coingecko.com/api/v3/coins/{cg_id}/market_chart/range?vs_currency=usd&from={from_ts}&to={to_ts}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"  Could not fetch data from CoinGecko API: {e}")
        return

    if not data or 'prices' not in data or not data['prices']: return
    
    new_records = []
    for item in data['prices']:
        record_date = datetime.fromtimestamp(item[0] / 1000)
        day_key = datetime(record_date.year, record_date.month, record_date.day)
        price = item[1]
        new_records.append({
            'date': day_key, 'coin_ticker': ticker, 'coin_name': ticker.split('-')[0],
            'close': price, 'low': price, 'high': price, 'open': price, 'volume': 0
        })
    
    unique_records = list({rec['date']: rec for rec in new_records}.values())
    if unique_records:
        print(f"  Adding {len(unique_records)} new records to the database.")
        collection.insert_many(unique_records)


def get_all_data_from_mongo(collection, ticker, start_date=None):
    """ Retrieves and cleans historical data. """
    print("Loading all historical data from MongoDB...")
    cursor = collection.find({'coin_ticker': ticker})
    df = pd.DataFrame(list(cursor))
    if df.empty: return pd.DataFrame()
    
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['close', 'low', 'high'], inplace=True)

    if start_date:
        try:
            df = df.loc[start_date:]
            print(f"Cleaned data. Using records from manual start date {start_date} onwards.")
        except KeyError:
            print(f"Warning: Manual start date {start_date} not found in data. Using all available data.")
    else:
        first_significant_date = df[df['high'] > 1.0].index.min()
        if pd.notnull(first_significant_date):
            df = df.loc[first_significant_date:]
            print(f"Cleaned data. Using records from {first_significant_date.strftime('%Y-%m-%d')} onwards.")

    print(f"Loaded {len(df)} total records from database for training.")
    return df


def generate_prophet_forecast(df, ticker, config):
    """
    Generates a 3-year forecast using the UNIFIED 'Best of Both Worlds' model.
    """
    print("Generating forecast with the unified model...")
    prophet_df = df.reset_index().rename(columns={'date': 'ds', 'close': 'y', 'low': 'low', 'high': 'high'})

    try:
        floor_df = df.loc['2022-01-01':'2023-01-01']
        floor_price = floor_df['low'].min()
        if floor_df.empty or pd.isna(floor_price):
             raise ValueError("No data in 2022 range.")
    except Exception:
        print("  Warning: Could not use 2022 floor. Using historical minimum as fallback.")
        floor_price = prophet_df['low'].min()
    if pd.isna(floor_price) or floor_price <= 0: floor_price = 1

    historical_max_price = prophet_df['high'].max()
    cap_price = historical_max_price * config['dynamic_cap_multiplier']

    best_params = config['params']
    if cap_price <= floor_price:
        cap_price = floor_price * 2

    print(f"  - Using Parameters: {best_params}")
    print(f"  - Using DYNAMIC CAP: {cap_price:,.2f} (Based on ATH of {historical_max_price:,.2f})")
    print(f"  - Using 2022 BEAR MARKET FLOOR: {floor_price:.4f}")

    model = Prophet(growth='logistic', seasonality_prior_scale=10, **best_params)

    model.add_seasonality(name='monthly', period=30.5, fourier_order=5, prior_scale=5.0)
    model.add_seasonality(name='halving_cycle', period=1461, fourier_order=15, prior_scale=10.0)

    prophet_df['cap'] = cap_price
    prophet_df['floor'] = floor_price
    
    model.fit(prophet_df[['ds', 'y', 'cap', 'floor']])
    future = model.make_future_dataframe(periods=3 * 365)
    future['cap'] = cap_price
    future['floor'] = floor_price
    forecast = model.predict(future)
    
    forecast['yhat'] = forecast['yhat'].clip(lower=0)
    print("Forecast generation complete.")
    return forecast


def store_forecast_in_mongo(collection, forecast_df, ticker, historical_df):
    """ Stores the new forecast in MongoDB, containing ONLY future dates. """
    print("Storing new forecast in MongoDB...")

    last_historical_date = historical_df.index.max()
    print(f"Filtering forecast to only include dates after the last historical entry: {last_historical_date.strftime('%Y-%m-%d')}")
    
    future_only_forecast = forecast_df[forecast_df['ds'] > last_historical_date].copy()
    
    collection.delete_many({'coin_ticker': ticker})
    
    if future_only_forecast.empty:
        print("No future forecast points to store.")
        return
        
    simplified_forecast = future_only_forecast[['ds', 'yhat']]
    simplified_forecast['coin_ticker'] = ticker
    forecast_records = simplified_forecast.to_dict('records')
    
    if forecast_records:
        collection.insert_many(forecast_records)
        print(f"Successfully stored {len(forecast_records)} future forecast points.")