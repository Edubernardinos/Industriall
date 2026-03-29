import pandas as pd
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Configurações de estilo (conforme o notebook)
pd.set_option('display.max_columns', None)
plt.rcParams['figure.figsize'] = (14, 4)
sns.set_theme(style='darkgrid')

print("=== Iniciando Pipeline de Pre-processamento ===")

# Caminhos de arquivos
RAW_DATA_PATH = Path('../data/raw/dados_planta.csv')
PROCESSED_PATH = Path('../data/processed/')
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = Path('../reports/figures')
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = 'TARGET'
TS_COL = 'TS'
FEATURE_COLS = ['var1','var2','var3','var4','var5','var6',
                'var8','var9','var10','var11','var12','var13','var14','var15']

# --- Bloco 1: Carregamento da Base ---
try:
    print("Carregando base de dados bruta...")
    df = pd.read_csv(RAW_DATA_PATH, parse_dates=[TS_COL])
    
    # Normalizar timezone para UTC e remover tzinfo (conforme notebook)
    df[TS_COL] = pd.to_datetime(df[TS_COL], utc=True).dt.tz_localize(None)
    df = df.sort_values(TS_COL).reset_index(drop=True)
    df = df.set_index(TS_COL)
    print(f"Shape: {df.shape}")
except Exception as e:
    print(f"Erro no carregamento: {e}")
    sys.exit(1)

# --- Bloco 2: Limpeza, Outliers e Targets ---
try:
    print("Tratando outliers de var14 (Clipping)...")
    if 'var14' in df.columns:
        p01 = df['var14'].quantile(0.01)
        p99 = df['var14'].quantile(0.99)
        df['var14'] = df['var14'].clip(lower=p01, upper=p99)
        
        # Descartando var14 após clipping (conforme decisão no notebook)
        df = df.drop(columns=['var14'])
        FEATURE_COLS = [c for c in FEATURE_COLS if c != 'var14']
        print("var14 processada e removida.")

    # Flag de processo ativo
    ACTIVE_THRESHOLD = 1.0
    df['processo_ativo'] = (df[TARGET_COL] > ACTIVE_THRESHOLD).astype(int)
    
    # Horizonte de 1 hora
    HORIZON = 60
    df['target_futuro'] = df[TARGET_COL].shift(-HORIZON)
    df['processo_ativo_futuro'] = df['processo_ativo'].shift(-HORIZON)

    # Tratamento de nulos (Estratégia Híbrida do Notebook)
    print("Aplicando preenchimento inercial (ffill limit=5)...")
    df = df.ffill(limit=5)
    
    # Dropna inicial para limpar buracos maiores que 5 min
    df = df.dropna(subset=[TARGET_COL]) # Garante que temos o target atual
    
except Exception as e:
    print(f"Erro na limpeza/targets: {e}")
    sys.exit(1)

# --- Bloco 3: Feature Engineering ---
try:
    print("Gerando features cíclicas de tempo...")
    # Hora do dia
    hour = df.index.hour
    df['hora_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hora_cos'] = np.cos(2 * np.pi * hour / 24)

    # Minuto do dia
    minute_of_day = df.index.hour * 60 + df.index.minute
    df['minuto_sin'] = np.sin(2 * np.pi * minute_of_day / 1440)
    df['minuto_cos'] = np.cos(2 * np.pi * minute_of_day / 1440)

    # Dia da semana
    dow = df.index.dayofweek
    df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    df['dow_cos'] = np.cos(2 * np.pi * dow / 7)

    print("Criando Lags e Medianas Móveis (MLOps)...")
    LAG_FEATURES = ['var4', 'var6', 'var2', 'var10', 'var13', 'var15', 'var1', TARGET_COL]
    LAG_WINDOWS = [10, 30, 60, 90]
    ROLLING_WINDOWS = [5, 15, 30]

    for feat in LAG_FEATURES:
        for lag in LAG_WINDOWS:
            df[f'{feat}_lag{lag}'] = df[feat].shift(lag)
        for window in ROLLING_WINDOWS:
            # Importante: min_periods=1 evita NaNs desnecessários
            df[f'{feat}_roll{window}'] = df[feat].rolling(window=window, min_periods=1).median()

    # --- LIMPEZA FINAL (3 Passos do Notebook) ---
    print("Executando limpeza final (Aparando bordas temporais)...")
    df = df.dropna(subset=['target_futuro'])      # Bordas do futuro
    df = df.dropna(subset=[f'{TARGET_COL}_lag60']) # Bordas do passado
    df = df.dropna()                              # Residuais no meio

except Exception as e:
    print(f"Erro na engenharia de features: {e}")
    sys.exit(1)

# --- Bloco 4: Split e Exportação ---
try:
    print("Dividindo bases (Split temporal - 2 semanas para teste)...")
    # Cutoff dinâmico conforme o notebook
    TEST_CUTOFF = df.index.max() - pd.Timedelta(weeks=2)
    
    train = df[df.index <= TEST_CUTOFF].copy()
    test = df[df.index > TEST_CUTOFF].copy()
    
    print(f"Shape Treino: {train.shape} | Shape Teste: {test.shape}")

    # Exportação Parquet
    train.to_parquet(PROCESSED_PATH / 'train.parquet')
    test.to_parquet(PROCESSED_PATH / 'test.parquet')
    print(f"Dados salvos em {PROCESSED_PATH}")

    # Gerar gráfico de conferência
    plt.figure(figsize=(14, 6))
    plt.plot(df.index[:2000], df[TARGET_COL][:2000], label='TARGET', alpha=0.7)
    plt.fill_between(df.index[:2000], 0, df[TARGET_COL].max(), 
                     where=df['processo_ativo'][:2000] == 1, 
                     color='green', alpha=0.1, label='Processo Ativo')
    plt.title('TARGET vs Flag Processo Ativo (Primeiros 2000 min)')
    plt.legend()
    plt.savefig(REPORTS_DIR / 'processo_ativo_vs_target.png')
    plt.close()

except Exception as e:
    print(f"Erro na exportação: {e}")

# --- Bloco 5: Relatório Final ---
report_path = Path('../statistics/02_preprocessing_report.txt')
report_path.parent.mkdir(parents=True, exist_ok=True)
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=== RELATORIO DE PRE-PROCESSAMENTO ===\n\n")
    f.write(f"Variavel var14 clipada (p01-p99) e removida.\n")
    f.write(f"Estrategia de Nulos: ffill(limit=5) + dropna() final.\n")
    f.write(f"Features criadas: Ciclicas (hora/min/dow), Lags ({LAG_WINDOWS}) e Rolling Medians ({ROLLING_WINDOWS}).\n")
    f.write(f"Treino: {train.shape[0]} amostras | Teste: {test.shape[0]} amostras.\n")

print("\n=== Pre-processamento Finalizado com Sucesso ===")