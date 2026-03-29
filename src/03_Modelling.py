import pandas as pd
import numpy as np
import time
import joblib
import warnings
import sys
from pathlib import Path

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
pd.set_option('display.float_format', '{:.4f}'.format)

print("=== Iniciando Pipeline de Modelagem (Arquitetura DELTA) ===")

# --- 1. Setup e Carga de Dados ---
try:
    PROCESSED_PATH = Path('../data/processed/')
    train = pd.read_parquet(PROCESSED_PATH / 'train.parquet')
    test  = pd.read_parquet(PROCESSED_PATH / 'test.parquet')
    print(f'Treino: {train.shape}   |   Teste: {test.shape}')
except Exception as e:
    print(f"Erro ao carregar arquivos .parquet: {e}")
    sys.exit(1)

# --- 2. Preparação X/y (Alvo = Delta) ---
NON_FEATURE_COLS = ['TARGET', 'target_futuro', 'processo_ativo_futuro']
FEATURE_COLS = [c for c in train.columns if c not in NON_FEATURE_COLS]

X_train = train[FEATURE_COLS]
y_train_delta = train['target_futuro'] - train['TARGET']
y_train_real  = train['target_futuro']

X_test  = test[FEATURE_COLS]
y_test_delta  = test['target_futuro'] - test['TARGET']
y_test_real   = test['target_futuro']

print(f'Média do Target Delta (treino): {y_train_delta.mean():.4f}')

# --- 3. Definição de Métricas (Igual ao Notebook) ---
def mape(y_true, y_pred, eps=1e-6):
    mask = np.abs(y_true) > eps
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def eval_metrics(y_true, y_pred, label=''):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    mae   = mean_absolute_error(y_true, y_pred)
    mape_val = mape(y_true, y_pred)
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    
    if label:
        print(f'{label:<25}  RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}')
    return dict(RMSE=rmse, MAE=mae, MAPE=mape_val, R2=r2)

# --- 4. Baseline (Inércia Atual) ---
baseline_pred_real = test['TARGET'].values
baseline_metrics = eval_metrics(y_test_real, baseline_pred_real, label='Baseline (Inércia Atual)')

# --- 5. Cross-Validation ---
N_SPLITS = 5
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

MODELS = {
    'Ridge': Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=1.0))]),
    'RandomForest': RandomForestRegressor(n_estimators=50, max_depth=8, n_jobs=-1, random_state=42),
    'LightGBM': lgb.LGBMRegressor(n_estimators=100, n_jobs=-1, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=6, n_jobs=-1, random_state=42),
    'CatBoost': CatBoostRegressor(iterations=50, depth=8, random_state=42, verbose=False, thread_count=-1)
}

print("\n=== Iniciando Cross-Validation Temporal ===")
for name, model in MODELS.items():
    fold_metrics = []
    t0 = time.time()
    for tr_idx, val_idx in tscv.split(X_train):
        # Treino no Delta
        model.fit(X_train.iloc[tr_idx], y_train_delta.iloc[tr_idx])
        delta_preds = model.predict(X_train.iloc[val_idx])
        # Validação no Real (Inércia + Delta)
        real_preds = train['TARGET'].iloc[val_idx].values + delta_preds
        fold_metrics.append(eval_metrics(y_train_real.iloc[val_idx], real_preds))
    
    mean_rmse = np.mean([m['RMSE'] for m in fold_metrics])
    mean_r2   = np.mean([m['R2']   for m in fold_metrics])
    print(f'{name:<20} CV-RMSE={mean_rmse:.4f}  CV-R2={mean_r2:.4f}   ({time.time()-t0:.1f}s)')

# --- 6. Treinamento Final e Leaderboard de Teste ---
print("\n=== Métricas Finais no Conjunto de Teste ===")
test_results = {'Baseline (Inércia)': baseline_metrics}

for name, model in MODELS.items():
    model.fit(X_train, y_train_delta)
    delta_preds = model.predict(X_test)
    real_preds = test['TARGET'].values + delta_preds
    test_results[name] = eval_metrics(y_test_real, real_preds, label=name)

# --- 7. Treinamento Específico do Modelo Simples e Salvamento ---
print("\nTreinando e salvando o modelo RandomForest Simples (Final)...")
simple_rf_model = RandomForestRegressor(n_estimators=50, max_depth=8, n_jobs=-1, random_state=42)
simple_rf_model.fit(X_train, y_train_delta)

# Salva o modelo
Path('../models').mkdir(parents=True, exist_ok=True)
joblib.dump(simple_rf_model, '../models/simple_rf_model.pkl')
print(f'Modelo simples salvo em: ../models/simple_rf_model.pkl')

# --- 8. Gráfico de Performance (Últimos 7 dias) ---
DAYS = 7
n_pts = DAYS * 24 * 60
y_plot = y_test_real.iloc[-n_pts:]
idx_plot = y_plot.index

plt.figure(figsize=(16, 5))
plt.plot(idx_plot, y_plot.values, color='black', lw=1.3, label='Real', zorder=10)
plt.plot(idx_plot, test['TARGET'].iloc[-n_pts:].values, color='gray', lw=1, ls='--', alpha=0.9, label='Baseline (Inércia)')

# Predição do modelo final para o gráfico
final_delta_preds = simple_rf_model.predict(X_test)
final_real_preds = test['TARGET'].values + final_delta_preds
plt.plot(idx_plot, final_real_preds[-n_pts:], color='steelblue', lw=1.5, label='RandomForest (Delta)', alpha=0.9)

plt.title(f'RandomForest Delta vs Baseline vs Real — últimos {DAYS} dias')
plt.legend()
plt.tight_layout()
plt.savefig('../reports/figures/performance_final.png')
print("Gráfico de performance salvo em: ../reports/figures/performance_final.png")

print("\n=== Pipeline encerrada com sucesso ===")