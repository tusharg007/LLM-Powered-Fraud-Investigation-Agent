import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json
import os

def train_model():
    print("Loading data from SQLite...")
    conn = sqlite3.connect("data/fraud_cases.db")
    df = pd.read_sql("SELECT * FROM transactions", conn)
    conn.close()

    print("Data loaded. Shape:", df.shape)

    # Feature Engineering
    # Convert datetime to numeric features
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    
    df['purchase_hour'] = df['purchase_time'].dt.hour
    df['purchase_dayofweek'] = df['purchase_time'].dt.dayofweek
    df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600 # in hours

    # Encode categorical variables
    cat_columns = ['device_id', 'source', 'browser', 'sex']
    label_encoders = {}
    for col in cat_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # IP address is tricky for basic ML without geolocation, we'll hash it or just use length/prefix as dummy
    df['ip_prefix'] = df['ip_address'].apply(lambda x: int(x.split('.')[0]) if isinstance(x, str) else 0)
    
    # Drop columns not suitable for the model directly
    X = df.drop(columns=['class', 'user_id', 'signup_time', 'purchase_time', 'ip_address'])
    y = df['class']

    # Train Test Split
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost
    print("Training XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=5, 
        random_state=42, 
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    print(f"Model Training Complete. Test Accuracy: {accuracy:.4f}")

    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = "models/xgb_fraud_model.json"
    model.save_model(model_path)
    print(f"Model saved to {model_path}")

    # Calculate and save global feature importance
    importance = model.feature_importances_
    feat_imp = {feat: float(imp) for feat, imp in zip(X.columns, importance)}
    with open("models/feature_importance.json", "w") as f:
        json.dump(feat_imp, f, indent=4)
    print("Feature importance saved.")
    
if __name__ == "__main__":
    train_model()
