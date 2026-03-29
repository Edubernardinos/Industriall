import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

train = pd.read_parquet('data/processed/train.parquet')
NON_FEATURE_COLS = ['TARGET', 'target_futuro', 'processo_ativo_futuro']
FEATURE_COLS = [c for c in train.columns if c not in NON_FEATURE_COLS]
X_train = train[FEATURE_COLS]
y_train_delta = train['target_futuro'] - train['TARGET']
y_train_real = train['target_futuro']

tscv = TimeSeriesSplit(n_splits=3)
models = {
    'Ridge': Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=1.0))]),
    'RandomForest': RandomForestRegressor(n_estimators=50, max_depth=8, n_jobs=-1, random_state=42),
    'LightGBM': lgb.LGBMRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1, verbose=-1),
    'XGBoost': xgb.XGBRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
}

print("=== Resultados do CV ===")
for name, model in models.items():
    rmse_folds = []
    for tr_idx, val_idx in tscv.split(X_train):
        model.fit(X_train.iloc[tr_idx], y_train_delta.iloc[tr_idx])
        delta_preds = model.predict(X_train.iloc[val_idx])
        real_preds = train['TARGET'].iloc[val_idx].values + delta_preds
        fold_rmse = np.sqrt(np.mean((y_train_real.iloc[val_idx].values - real_preds)**2))
        rmse_folds.append(fold_rmse)
    print(f"{name}: RMSE = {np.mean(rmse_folds):.4f}")
