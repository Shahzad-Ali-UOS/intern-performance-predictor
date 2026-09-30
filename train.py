import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb


def train_and_export():
    data_path = os.path.join("data", "intern_data.csv")
    
    if not os.path.exists(data_path):
        from data_generator import generate_intern_telemetry
        print("Data file not found. Generating fresh dataset...")
        generate_intern_telemetry()
        
    df = pd.read_csv(data_path)
    
    features = [
        'task_completion_rate',
        'avg_turnaround_days',
        'attendance_rate',
        'mentor_feedback_rating',
        'peer_collaboration_score'
    ]
    target = 'performance_score'
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    
    # 1. Random Forest Regressor
    rf_model = RandomForestRegressor(
        n_estimators=150, 
        max_depth=7, 
        min_samples_split=4, 
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    
    # 2. XGBoost Regressor
    xgb_model = xgb.XGBRegressor(
        n_estimators=150, 
        learning_rate=0.05, 
        max_depth=4, 
        subsample=0.85, 
        colsample_bytree=0.85, 
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    
    def evaluate(name, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        return {'Model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2}
        
    res_rf = evaluate("Random Forest", y_test, rf_preds)
    res_xgb = evaluate("XGBoost", y_test, xgb_preds)
    
    benchmark = pd.DataFrame([res_rf, res_xgb])
    print("\n--- Model Benchmark Summary ---")
    print(benchmark.to_string(index=False))
    
    # Select winning model
    winning_model = xgb_model if res_xgb['R2'] >= res_rf['R2'] else rf_model
    winning_name = "XGBoost" if res_xgb['R2'] >= res_rf['R2'] else "Random Forest"
    
    os.makedirs("models", exist_ok=True)
    artifacts = {
        'model': winning_model,
        'model_name': winning_name,
        'features': features,
        'metrics': res_xgb if winning_name == "XGBoost" else res_rf
    }
    
    export_path = os.path.join("models", "best_model.pkl")
    joblib.dump(artifacts, export_path)
    print(f"\n[OK] Production pipeline exported -> {export_path}")


if __name__ == "__main__":
    train_and_export()