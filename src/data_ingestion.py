import os
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta

def generate_mock_data(n_rows=5000):
    print("WARNING: 'data/ecommerce_fraud.csv' not found. Generating mock e-commerce fraud data with the same schema for testing...")
    np.random.seed(42)
    start_date = datetime(2023, 1, 1)
    
    mock_data = {
        'user_id': np.random.randint(10000, 99999, n_rows),
        'signup_time': [(start_date + timedelta(days=int(d))).strftime('%Y-%m-%d %H:%M:%S') for d in np.random.randint(0, 300, n_rows)],
        'purchase_time': [(start_date + timedelta(days=int(d), hours=int(h))).strftime('%Y-%m-%d %H:%M:%S') for d, h in zip(np.random.randint(10, 320, n_rows), np.random.randint(0, 24, n_rows))],
        'purchase_value': np.random.randint(10, 500, n_rows),
        'device_id': [f"DEV{np.random.randint(1000, 9999)}" for _ in range(n_rows)],
        'source': np.random.choice(['SEO', 'Ads', 'Direct'], n_rows),
        'browser': np.random.choice(['Chrome', 'Safari', 'FireFox', 'IE'], n_rows),
        'sex': np.random.choice(['M', 'F'], n_rows),
        'age': np.random.randint(18, 65, n_rows),
        'ip_address': [f"192.168.{np.random.randint(0, 255)}.{np.random.randint(0, 255)}" for _ in range(n_rows)],
        'class': np.random.choice([0, 1], p=[0.9, 0.1], size=n_rows) # 10% fraud target
    }
    
    return pd.DataFrame(mock_data)

def ingest_data():
    csv_path = "data/ecommerce_fraud.csv"
    db_path = "data/fraud_cases.db"
    
    # Ensure data dir exists
    os.makedirs('data', exist_ok=True)
    
    if os.path.exists(csv_path):
        print(f"Loading real dataset from {csv_path}...")
        df = pd.read_csv(csv_path)
    else:
        df = generate_mock_data(10000)
        df.to_csv("data/mock_ecommerce_fraud.csv", index=False)
        print("Mock data saved for testing. Please place the real Kaggle 'Fraud_Data.csv' inside 'data/ecommerce_fraud.csv' when ready.")
        
    print(f"Dataset Shape: {df.shape}")
    print(f"Fraud cases (%): {df['class'].mean() * 100:.2f}%")
    
    # Process dataset types for SQL database
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    
    print("Writing to SQLite Database...")
    conn = sqlite3.connect(db_path)
    
    # Table for transaction history
    df.to_sql('transactions', conn, if_exists='replace', index=False)
    
    # Table for past explicit fraud case resolutions (for RAG embedding later)
    fraud_cases = df[df['class'] == 1].copy()
    fraud_cases['resolution_notes'] = "Confirmed fraud based on suspicious IPs or device velocity."
    fraud_cases.to_sql('historical_fraud', conn, if_exists='replace', index=False)
    
    conn.close()
    print("Data ingestion complete. Database created at:", db_path)

if __name__ == "__main__":
    ingest_data()
