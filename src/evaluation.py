import sqlite3
import pandas as pd
import json
import time
import os
from src.agent import run_agent

def evaluate_agent(num_cases=20):
    print(f"Starting evaluation of LLM Agent on {num_cases} cases...")
    conn = sqlite3.connect("data/fraud_cases.db")
    
    # We select 50% fraud and 50% normal to get a balanced evaluation
    df_fraud = pd.read_sql(f"SELECT * FROM transactions WHERE class = 1 LIMIT {num_cases//2}", conn)
    df_normal = pd.read_sql(f"SELECT * FROM transactions WHERE class = 0 LIMIT {num_cases//2}", conn)
    df = pd.concat([df_fraud, df_normal]).sample(frac=1, random_state=42).reset_index(drop=True)
    conn.close()

    if len(df) == 0:
        print("No data available for evaluation.")
        return

    results = []
    correct = 0
    false_positives = 0
    true_negatives = 0
    total_time = 0

    for idx, row in df.iterrows():
        txn = dict(row.drop('class'))
        ground_truth = int(row['class'])
        
        # Example RAG lookup (normally handled by a retriever node or passed in)
        rag_context = ""
        
        start_time = time.time()
        
        try:
            decision_json = run_agent(txn, rag_context)
            agent_is_fraud = 1 if "fraud" in str(decision_json.get("decision", "")).lower() and "not" not in str(decision_json.get("decision", "")).lower() else 0
        except Exception as e:
            decision_json = {"decision": "Error", "reasoning": str(e)}
            agent_is_fraud = 0
            
        elapsed = time.time() - start_time
        total_time += elapsed
        
        # Metrics logic
        is_correct = (agent_is_fraud == ground_truth)
        if is_correct:
            correct += 1
            if ground_truth == 0:
                true_negatives += 1
        else:
            if agent_is_fraud == 1 and ground_truth == 0:
                false_positives += 1

        results.append({
            "user_id": row['user_id'],
            "ground_truth": ground_truth,
            "agent_decision": agent_is_fraud,
            "reasoning": decision_json.get("reasoning", ""),
            "time_taken_sec": elapsed
        })
        print(f"Case {idx+1}/{num_cases} | Ground Truth: {ground_truth} | Agent: {agent_is_fraud} | Correct: {is_correct}")

    accuracy = correct / num_cases
    actual_negatives = len(df_normal)
    fpr = false_positives / actual_negatives if actual_negatives > 0 else 0

    # Business Metrics simulating human equivalent
    human_time_per_case_mins = 15
    llm_avg_time_secs = total_time / num_cases if num_cases > 0 else 0
    time_saved_hrs = (num_cases * human_time_per_case_mins) / 60 - (total_time / 3600)

    metrics = {
        "accuracy": accuracy,
        "false_positive_rate": fpr,
        "human_time_saved_hours": round(time_saved_hrs, 2),
        "llm_avg_response_time_sec": round(llm_avg_time_secs, 2),
        "total_cases_evaluated": num_cases
    }

    print("\n--- EVALUATION RESULTS ---")
    print(json.dumps(metrics, indent=4))

    os.makedirs('models', exist_ok=True)
    with open("models/eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    pd.DataFrame(results).to_csv("models/eval_results.csv", index=False)
    print("Metrics saved to models/")

if __name__ == "__main__":
    # Ensure database exists before running
    if not os.path.exists("data/fraud_cases.db"):
        print("Database not found. Please run Phase 1: data_ingestion.py first.")
    else:
        evaluate_agent()
