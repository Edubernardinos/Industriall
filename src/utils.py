import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

def mape(y_true, y_pred, eps=1e-6):
    mask = np.abs(y_true) > eps
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def eval_metrics(y_true, y_pred, label=""):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape_val = mape(y_true, y_pred)
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    
    if label:
        print(f"{label:<25} RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")
        
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape_val, "R2": r2}

def encode_cyclic_time(df):
    hour = df.index.hour
    minute_of_day = df.index.hour * 60 + df.index.minute
    dow = df.index.dayofweek

    df['hora_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hora_cos'] = np.cos(2 * np.pi * hour / 24)
    
    df['minuto_sin'] = np.sin(2 * np.pi * minute_of_day / 1440)
    df['minuto_cos'] = np.cos(2 * np.pi * minute_of_day / 1440)
    
    df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    df['dow_cos'] = np.cos(2 * np.pi * dow / 7)
    
    return df
