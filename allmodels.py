import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import joblib

# 1. Simulate Data (or load real data instead)
def simulate_data(n_engines=100, max_cycles=300, n_sensors=5):
    data = []
    for eng in range(1, n_engines + 1):
        RUL = np.random.randint(150, max_cycles)
        for cycle in range(1, RUL + 1):
            sensors = np.random.normal(loc=50, scale=10, size=n_sensors)
            row = [eng, cycle] + sensors.tolist() + [RUL - cycle]
            data.append(row)
    columns = ['engine_id', 'cycle'] + [f'sensor{i+1}' for i in range(n_sensors)] + ['RUL']
    return pd.DataFrame(data, columns=columns)

df = simulate_data()

# 2. Feature Selection
FEATURES = [col for col in df.columns if col.startswith('sensor')]
X = df[FEATURES]
y = df['RUL']

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Evaluation Function (fixed version)
def evaluate_model(name, model):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5  # Avoids use of `squared=False`
    print(f"{name} -> MAE: {mae:.2f} | RMSE: {rmse:.2f}")
    return model

# 5. Train Random Forest
print("Training Random Forest...")
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
evaluate_model("Random Forest", rf)
joblib.dump(rf, "random_forest_model.pkl")

# 6. Train XGBoost
print("Training XGBoost...")
xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
xgb_model.fit(X_train, y_train)
evaluate_model("XGBoost", xgb_model)
joblib.dump(xgb_model, "xgboost_model.pkl")

# 7. Train LightGBM
print("Training LightGBM...")
lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
lgb_model.fit(X_train, y_train)
evaluate_model("LightGBM", lgb_model)
joblib.dump(lgb_model, "lightgbm_model.pkl")

print("\n✅ All models trained and saved successfully.")
