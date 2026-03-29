import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import numpy as np

# 1. Configuração inicial da página web
st.set_page_config(
    page_title="Industriall - Digital Twin v1.0", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- INJEÇÃO DE CSS PREMIUM ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=Outfit:wght@300;500;700&display=swap');

    :root {
        --primary-orange: #FF5733;
        --deep-navy: #0B192E;
        --accent-blue: #1B3A57;
        --glass-bg: rgba(255, 255, 255, 0.05);
        --glass-border: rgba(255, 255, 255, 0.1);
    }

    /* Reset e Tipografia Global */
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
        color: #E0E0E0;
    }

    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    /* Estilo da Sidebar Customizada */
    [data-testid="stSidebar"] {
        background-color: #050B14;
        border-right: 1px solid var(--glass-border);
    }
    
    [data-testid="stSidebar"] .stMarkdown h2 {
        color: var(--primary-orange);
        font-size: 1.5rem;
        padding-bottom: 20px;
    }

    /* Estilo dos Sliders Industriais */
    .stSlider > div > div > div > div {
        color: var(--primary-orange);
    }
    .stSlider > div > div > div {
        background-color: var(--accent-blue);
    }

    /* Cartões de Métricas Glassmorphism */
    .metric-card {
        background: var(--glass-bg);
        border: 1px solid var(--glass-border);
        border-radius: 12px;
        padding: 24px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: var(--primary-orange);
    }

    .metric-label {
        font-size: 0.85rem;
        color: #99AABB;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 2.2rem;
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        color: #FFFFFF;
    }

    .metric-delta {
        font-size: 0.95rem;
        font-weight: 700;
        margin-top: 5px;
    }

    /* Banner de Status Dinâmico */
    .status-banner {
        padding: 20px;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 30px;
        font-weight: 800;
        font-size: 1.2rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        border-width: 2px;
        border-style: solid;
    }

    .status-stable { 
        background-color: rgba(46, 204, 113, 0.1); 
        color: #2ECC71; 
        border-color: #2ECC71;
        box-shadow: 0 0 15px rgba(46, 204, 113, 0.2);
    }
    .status-warning { 
        background-color: rgba(241, 196, 15, 0.1); 
        color: #F1C40F; 
        border-color: #F1C40F;
        box-shadow: 0 0 15px rgba(241, 196, 15, 0.2);
    }
    .status-critical { 
        background-color: rgba(231, 76, 60, 0.1); 
        color: #E74C3C; 
        border-color: #E74C3C;
        box-shadow: 0 0 15px rgba(231, 76, 60, 0.2);
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.6; }
        100% { opacity: 1; }
    }
    
    /* Outras melhorias */
    .stExpander {
        border: none !important;
        background: var(--glass-bg);
        border-radius: 8px;
    }
    
    </style>
""", unsafe_allow_html=True)

# 2. Lógica de Negócio (Baseline e Assets)
THRESHOLDS_NORMAIS = {
    'var10': 64.70, 'var2': 221.88, 'var8': 372.27, 'var15': 161.05,
    'var5': 583.01, 'var9': 346.02, 'var1': 82.03, 'var3': 5.85
}

@st.cache_resource
def load_assets():
    try:
        # Define o diretório raiz do projeto relativo a este arquivo
        root_dir = Path(__file__).parent.parent
        
        model_path = root_dir / 'models' / 'simple_rf_model.pkl'
        data_path = root_dir / 'data' / 'processed' / 'test.parquet'
        
        model = joblib.load(model_path)
        test_df = pd.read_parquet(data_path)
        base_row = test_df.iloc[-1:].copy()
        return model, base_row
    except Exception as e:
        st.error(f"Erro ao carregar ativos: {e}")
        st.stop()
        st.stop()

model, base_row = load_assets()

# 3. Preparação das Colunas de Features
NON_FEATURE_COLS = ['TARGET', 'target_futuro', 'processo_ativo_futuro', 'FLAG_PARADA']
FEATURE_COLS = [c for c in base_row.columns if c not in NON_FEATURE_COLS]
VARIAVEIS_BRUTAS = [c for c in FEATURE_COLS if '_' not in c]

# 4. Interface do Usuário (Side Bar - Estilo Industriall)
st.sidebar.markdown(f"## industriall")
st.sidebar.markdown("---")
st.sidebar.write("Controle de Ativos Georreferenciados")

user_inputs = {}
for var_name in VARIAVEIS_BRUTAS:
    val_atual = float(base_row[var_name].iloc[0])
    limite = max(float(val_atual * 2.5), 10.0)
    user_inputs[var_name] = st.sidebar.slider(
        var_name.upper(), 0.0, limite, val_atual, step=limite/100.0
    )

# 5. Processamento (Inferência ML + Heurística Fisíca)
input_row = base_row.copy()
for var_name, val in user_inputs.items():
    input_row[var_name] = val
    prefixo = f"{var_name}_"
    for col in FEATURE_COLS:
        if col.startswith(prefixo):
            if any(x in col for x in ['_std', '_diff', '_var']):
                input_row[col] = 0.0
            else:
                input_row[col] = val

target_atual = float(base_row['TARGET'].iloc[0])
delta_predito = model.predict(input_row[FEATURE_COLS])[0]
target_futuro = target_atual + delta_predito

# Heurística
pontuacao_risco = 0
alertas = []
for var, thresh in THRESHOLDS_NORMAIS.items():
    if var in user_inputs:
        variacao = ((user_inputs[var] - thresh) / thresh) * 100
        if variacao <= -40.0:
            pontuacao_risco += 1
            alertas.append(f"{var} queda de {abs(variacao):.1f}%")

# 6. Decisão de Status
if (target_futuro <= target_atual * 0.85) or (pontuacao_risco >= 3):
    status_msg, status_class = "ALERTA CRÍTICO: Risco Iminente", "status-critical"
elif (target_futuro <= target_atual * 0.95) or (pontuacao_risco > 0):
    status_msg, status_class = "ATENÇÃO: Anomalia Detectada", "status-warning"
else:
    status_msg, status_class = "OPERAÇÃO ESTÁVEL E ATIVA", "status-stable"

# 7. Layout Principal (Visual Premium)
st.title("Digital Twin: Monitoramento de KPI")
st.markdown(f'<div class="status-banner {status_class}">{status_msg}</div>', unsafe_allow_html=True)

if alertas:
    st.info("**Diagnóstico de Sensores:** " + " | ".join(alertas))

st.markdown("---")

# Métricas Estilizadas
col1, col2, col3 = st.columns(3)

def custom_metric_card(label, value, delta=None, delta_color="normal"):
    color_code = "#2ECC71" if delta_color == "normal" else "#E74C3C"
    delta_str = f'<div class="metric-delta" style="color: {color_code}">{delta}</div>' if delta else ""
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {delta_str}
        </div>
    """, unsafe_allow_html=True)

with col1:
    custom_metric_card("KPI Atendimento (t)", f"{target_atual:.2f}")
with col2:
    color = "normal" if delta_predito >= 0 else "inverse"
    custom_metric_card("Variação Estimada (ML)", f"{delta_predito:+.2f}", f"Delta T+60", color)
with col3:
    custom_metric_card("Predição KPI (t+60)", f"{target_futuro:.2f}", f"{(target_futuro - target_atual):+.2f}")

st.markdown("---")

# Rodapé Técnico
with st.expander("Especificações Técnicas do Gemini Model"):
    st.markdown(f"""
    - **Backbone Model:** Random Forest Regressor (Arquitetura Delta).
    - **Feature Engineering:** {len(FEATURE_COLS)} indicadores processados em tempo real (Lags, Janelas Móveis e Sazonalidade).
    - **Rede de Segurança:** Monitoramento Heurístico de ativos físicos com baseline de operação normal.
    - **Input Data:** Streaming simulado via painel de controle georreferenciado.
    """)