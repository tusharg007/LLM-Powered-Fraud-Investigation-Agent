import sqlite3
import json
import os
from langchain.tools import tool

@tool
def query_user_history(user_id: str) -> str:
    """Useful to query the SQLite database for the transaction history of a specific user.
    Input should be the user_id (as a string or int).
    Returns a list of recent transactions for that user."""
    
    try:
        conn = sqlite3.connect("data/fraud_cases.db")
        cursor = conn.cursor()
        
        # We query the last 10 transactions for this user_id
        cursor.execute('''
            SELECT purchase_time, purchase_value, device_id, ip_address, browser 
            FROM transactions 
            WHERE user_id = ? 
            ORDER BY purchase_time DESC 
            LIMIT 10
        ''', (user_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return f"No transaction history found for user {user_id}."
            
        history = [
            f"Time: {r[0]}, Amount: ${r[1]}, Device: {r[2]}, IP: {r[3]}, Browser: {r[4]}"
            for r in rows
        ]
        
        return "\n".join(history)
    except Exception as e:
        return f"Error executing SQL query: {str(e)}"


@tool
def ml_feature_explainer(transaction_json: str) -> str:
    """Analyzes a specific transaction and explains which features are globally important 
    for fraud detection based on the XGBoost model.
    Pass a JSON string of the transaction.
    """
    try:
        # We read the global feature importance
        path = "models/feature_importance.json"
        if not os.path.exists(path):
            return "ML feature importance data not found. The model might not be trained yet."
            
        with open(path, "r") as f:
            importance = json.load(f)
            
        # Parse transaction to see what the values are
        txn = json.loads(transaction_json)
        
        # Sort importance
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        top_features = [f[0] for f in sorted_imp[:3]]
        
        explanation = f"The ML model (XGBoost) flags fraud mostly based on these top 3 features globally: {', '.join(top_features)}.\n"
        
        specifics = []
        for feat in top_features:
            if feat in txn:
                specifics.append(f"In this transaction, {feat} is '{txn[feat]}'.")
                
        explanation += " ".join(specifics)
        return explanation
    except Exception as e:
        return f"Error parsing transaction for ML explanation: {str(e)}"
