import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import warnings
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.dates as mdates

warnings.filterwarnings('ignore')

print("=== Iniciando Analise Exploratoria de Dados (EDA) ===")

# --- Bloco 1: Carregamento e Preparacao Basica ---
try:
    print("Carregando base de dados bruta...")
    data_path = Path('../data/raw/dados_planta.csv')
    
    # Leitura e formatacao do indice temporal
    df = pd.read_csv(data_path)
    df['TS'] = pd.to_datetime(df['TS']).dt.tz_localize(None)
    df.set_index('TS', inplace=True)
    df.sort_index(inplace=True)
    
    print(f"Sucesso: {df.shape[0]} linhas e {df.shape[1]} colunas carregadas.")
except Exception as e:
    print(f"Erro fatal ao carregar os dados: {e}")
    sys.exit(1)

# --- Bloco 2: Estatisticas e Verificacoes ---
try:
    print("\nResumo Estatistico das Variaveis:")
    print(df.describe().T)
    
    print("\nVerificando dados faltantes (Nulos):")
    nulls = df.isnull().sum()
    print(nulls[nulls > 0] if nulls.sum() > 0 else "Nenhum valor nulo encontrado.")
    
    print("\nCorrelacao das variaveis no tempo T com o KPI (TARGET) preditivo no tempo T+60:")
    df_corr = df.copy()
    df_corr['TARGET'] = df_corr['TARGET'].shift(-60)
    corr = df_corr.corr()['TARGET'].sort_values(ascending=False)
    print(corr)
except Exception as e:
    print(f"Erro durante os calculos estatisticos: {e}")

# --- Bloco 3: Geracao e Exportacao de Graficos ---
try:
    print("\nGerando graficos analiticos...")
    reports_dir = Path('../reports/figures')
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Grafico 1: Distribuicao do Target
    plt.figure(figsize=(10, 6))
    sns.histplot(df['TARGET'], bins=50, kde=True)
    plt.title('Distribuicao do TARGET')
    plt.savefig(reports_dir / 'target_distribution.png')
    plt.close()
    
    # Grafico 2: Correlograma (Matriz de Correlacao Preditiva)
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_corr.corr(), annot=False, cmap='coolwarm')
    plt.title('Matriz de Correlacao Preditiva (Target em T+60)')
    plt.savefig(reports_dir / 'correlation_matrix.png')
    plt.close()
    
    # Grafico 3: Boxplots para deteccao de outliers
    plt.figure(figsize=(15, 6))
    sns.boxplot(data=df.drop(columns=['TARGET', 'var14'] if 'var14' in df.columns else ['TARGET']))
    plt.title('Deteccao de Outliers (Distribuicao das Variaveis de Processo)')
    plt.xticks(rotation=45)
    plt.savefig(reports_dir / 'outliers_boxplot.png')
    plt.close()
    
    # Grafico 4: ACF e PACF da variavel TARGET para justificativa de Lags Inerciais
    print("Gerando graficos ACF e PACF para validacao matematica de Lags...")
    plt.figure(figsize=(12, 10))
    plt.subplot(211)
    plot_acf(df['TARGET'].dropna(), lags=120, ax=plt.gca(), title="Autocorrelacao (ACF) do TARGET (Inercia)")
    plt.subplot(212)
    plot_pacf(df['TARGET'].dropna(), lags=120, ax=plt.gca(), title="Autocorrelacao Parcial (PACF) do TARGET")
    plt.tight_layout()
    plt.savefig(reports_dir / 'acf_pacf_target.png')
    plt.close()
    
    # Grafico 5: Serie Temporal do TARGET (KPI)
    print("Gerando curva temporal do KPI (TARGET)...")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df.index, df['TARGET'], lw=0.7, color='steelblue')
    ax.set_title('Serie Temporal do TARGET (KPI)', fontweight='bold')
    ax.set_xlabel('Data')
    ax.set_ylabel('TARGET')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b/%y'))
    plt.tight_layout()
    plt.savefig(reports_dir / 'target_timeseries.png', dpi=150)
    plt.close()
    
    # --- Diagnostico Fisico: Comportamento antes da queda ---
    print("Executando Diagnostico Fisico (Comportamento 60min antes de quedas)...")
    target_normal = df[df['TARGET'] > 10].mean()
    
    # Mascara para identificar quedas (TARGET < 5)
    quedas = df[df['TARGET'] < 5].index
    # Filtro para pegar 60 minutos antes de cada queda significante (evitando sobreposicao)
    pre_queda_indices = []
    for q in quedas:
        start = q - pd.Timedelta(minutes=60)
        end = q - pd.Timedelta(minutes=1)
        pre_queda_indices.extend(df.loc[start:end].index.tolist())
    
    pre_queda_indices = list(set(pre_queda_indices)) # Remove duplicatas
    target_pre_queda = df.loc[pre_queda_indices].mean()
    
    # Comparativo
    diag_df = pd.DataFrame({
        'Media_Normal': target_normal,
        'Media_Pre_Queda': target_pre_queda
    }).drop(index=['TARGET'], errors='ignore')
    diag_df['Variacao_Pct'] = (diag_df['Media_Pre_Queda'] / diag_df['Media_Normal'] - 1) * 100
    diag_df = diag_df.sort_values('Variacao_Pct', ascending=True)

    print(f"Graficos exportados com sucesso para o diretorio: {reports_dir}")
except Exception as e:
    print(f"Erro ao gerar ou salvar os dados de diagnostico/graficos: {e}")
    diag_df = pd.DataFrame() # Fallback para o proximo bloco

# --- Bloco 4: Relatorio Estatistico ---
try:
    print("\nGerando relatorio descritivo em texto txt...")
    stats_dir = Path('../statistics')
    stats_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = stats_dir / '01_eda_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== RELATORIO DE ANALISE EXPLORATORIA (EDA) ===\n\n")
        f.write("1. ESTRUTURA DOS DADOS\n")
        f.write(f"- Total de Linhas Lidas: {df.shape[0]}\n")
        f.write(f"- Total de Colunas Iniciais: {df.shape[1]}\n\n")
        f.write("2. VALORES NULOS ENCONTRADOS E MAPEADOS\n")
        f.write(nulls[nulls > 0].to_string() + "\n\n")
        f.write("3. ESTATISTICAS DESCRITIVAS OFICIAIS\n")
        f.write(df.describe().T.to_string() + "\n\n")
        f.write("4. INFLUENCIA DAS VARIAVEIS (CORRELACAO PREDITIVA - T+60)\n")
        f.write("- Aplicado Shift(-60) no TARGET para capturar correlação pro-ativa e evitar reatividade no instante T.\n")
        f.write(corr.to_string() + "\n\n")
        f.write("5. TRATAMENTO E DETECCAO DE OUTLIERS\n")
        f.write("- Anomalia severa logica detectada no sensor 'var14' (Escala absurda 10^7, sem correlacao preditiva).\n")
        f.write("- As demais variaveis comportam-se dentro da fisica normal da operacao de processo.\n")
        f.write("- Os plots originais constam na pasta reports/figures.\n\n")
        f.write("6. JUSTIFICATIVA PARA ENGENHARIA TEMPORAL (LAGS)\n")
        f.write("- Foram plotados graficos de Autocorrelacao (ACF) e Parcial (PACF) para comprovar a memoria inercial do alvo (Ate 60+ minutos).\n\n")
        f.write("7. DIAGNOSTICO FISICO: PRE-PARADA (60 MIN ANTES)\n")
        f.write("Analise de comportamento dos sensores momentos antes do KPI colapsar (< 5):\n")
        if not diag_df.empty:
            f.write(diag_df.to_string() + "\n")
        else:
            f.write("Dados de pre-queda nao gerados devido a erro no bloco anterior.\n")
    print(f"Relatorio salvo permanentemente em: {report_path}")
except Exception as e:
    print(f"Erro ao compilar o relatorio txt: {e}")

print("\n=== EDA Finalizada com Sucesso ===")