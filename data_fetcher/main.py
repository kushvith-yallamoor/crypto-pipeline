import os
import warnings
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from ta.wrapper import add_all_ta_features
import numpy as np
from pymongo import MongoClient, UpdateOne
from google.cloud import secretmanager

warnings.filterwarnings('ignore')

# --- Configuration ---
PROJECT_ID = os.environ.get("GCP_PROJECT")
SECRET_ID = "MONGO_URI"
DB_NAME = "crypto_db"
COLLECTION_NAME = "historical_data"
COINS = {
    'BTC': 'BTC-USD', 'ETH': 'ETH-USD', 'BNB': 'BNB-USD',
    'XRP': 'XRP-USD', 'ADA': 'ADA-USD', 'SOL': 'SOL-USD'
}

# --- MongoDB and Secret Manager Client Initialization ---
# This part runs only once when the function instance starts (cold start)
mongo_client = None

def get_mongo_client():
    global mongo_client
    if mongo_client is None:
        print("Initializing MongoDB client...")
        secret_client = secretmanager.SecretManagerServiceClient()
        secret_name = f"projects/{PROJECT_ID}/secrets/{SECRET_ID}/versions/latest"
        response = secret_client.access_secret_version(name=secret_name)
        mongo_uri = response.payload.data.decode("UTF-8")
        mongo_client = MongoClient(mongo_uri)
    return mongo_client

def apply_feature_engineering(df):
    df = add_all_ta_features(
        df, open="open", high="high", low="low", close="close", volume="volume", fillna=True
    )
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['week_of_year'] = df.index.isocalendar().week.astype(int)
    df['month'] = df.index.month
    df['year'] = df.index.year
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

def fetch_and_store_data(request):
    """
    HTTP-triggered Cloud Function to fetch and store crypto data.
    """
    print("Cloud Function `fetch_and_store_data` triggered.")
    client = get_mongo_client()
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    collection.create_index([("coin", 1), ("date", -1)], unique=True)

    today = datetime.utcnow().date()
    all_updates = []

    for coin_name, ticker in COINS.items():
        print(f"\nProcessing {coin_name}...")
        
        # Find the last date we have for this coin in the DB
        latest_entry = collection.find_one({"coin": coin_name}, sort=[("date", -1)])
        start_date = (latest_entry['date'] + timedelta(days=1)) if latest_entry else datetime(2020, 1, 1)
        
        # Ensure start_date is timezone-naive for comparison
        start_date = start_date.replace(tzinfo=None)
        end_date = today

        if start_date.date() >= end_date:
            print(f"Data for {coin_name} is already up to date.")
            continue
            
        print(f"Fetching data for {coin_name} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if data.empty:
                print(f"No new data found for {coin_name}.")
                continue

            data.columns = [str(col).lower().strip() for col in data.columns]
            
            # Apply feature engineering
            featured_data = apply_feature_engineering(data)
            
            # Prepare for bulk upsert
            featured_data.reset_index(inplace=True)
            featured_data.rename(columns={'Date': 'date'}, inplace=True)
            featured_data['coin'] = coin_name
            
            # Convert Timestamps to datetime objects for MongoDB
            featured_data['date'] = featured_data['date'].dt.to_pydatetime()

            for record in featured_data.to_dict('records'):
                all_updates.append(
                    UpdateOne(
                        {"coin": record['coin'], "date": record['date']},
                        {"$set": record},
                        upsert=True
                    )
                )
        except Exception as e:
            print(f"Could not process data for {ticker}: {e}")

    if not all_updates:
        print("No data to update.")
        return "Pipeline finished. No new data was available.", 200

    print(f"\nPerforming bulk write of {len(all_updates)} documents to MongoDB...")
    try:
        collection.bulk_write(all_updates)
        print("Bulk write successful.")
        return f"Successfully fetched and stored {len(all_updates)} new records.", 200
    except Exception as e:
        print(f"Error during bulk write: {e}")
        return "An error occurred during the database operation.", 500