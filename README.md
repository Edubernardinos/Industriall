# Industrial KPI Predictor & Monitoramento Híbrido

![Status](https://img.shields.io/badge/Status-Produção-green)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Framework](https://img.shields.io/badge/Framework-Streamlit-FF4B4B)

Sistema de Inteligência Operacional desenvolvido para predição de KPIs industriais com 60 minutos de antecedência. Atua como um Gêmeo Digital (Digital Twin), permitindo que a operação antecipe paradas e instabilidades através de uma abordagem híbrida: Inteligência Artificial aliada a Heurísticas de Engenharia de Processo.

---

## Contexto: IndustriAll 

A **[IndustriAll](https://industriall.ai/)** é uma IndTech brasileira (Vitoria/ES) que acelera o salto tecnológico de grandes indústrias (como Vale, Suzano e ArcelorMittal) através da **previsibilidade do futuro**.

Este projeto busca integrar-se à filosofia do motor **Ally (IA Industrial)** — provavelmente a tecnologia utilizada para predição de variáveis e otimização de recursos em tempo real. O foco aqui é transformar dados brutos de sensores em visão estratégica, prevenindo falhas e paradas críticas.

---

## Funcionalidades Principais

- **Predição T+60 (Arquitetura Delta):** O modelo estima a variação ($\Delta$) do KPI, focando no aprendizado de sinais que precedem mudanças, em vez de apenas replicar o estado atual.
- **Monitoramento Heurístico (Safety Net):** Camada de segurança baseada na física do processo. Detecta quedas severas (>40%) em sensores críticos para disparar alertas de prioridade máxima.
- **Simulador Operacional:** Interface interativa em Streamlit para simulação de cenários "What-if" e validação de sensores.

---

## Estrutura do Projeto

O projeto é modularizado para garantir manutenibilidade e escalabilidade:

```bash
├── application/      # Interface Streamlit (Dashboard do Operador)
├── src/             # Pipeline de produção (Single Source of Truth)
│   ├── 01_Eda.py           # Análise exploratória e diagnóstico físico
│   ├── 02_Preprocessing.py # Limpeza e Feature Engineering
│   └── 03_Modelling.py     # Treinamento e Validação Temporal
├── data/            # Dados (Raw e Processed em .parquet)
├── models/          # Modelos serializados (.pkl)
├── reports/         # Gráficos de performance e figura de diagnóstico
└── statistics/      # Relatórios técnicos em formato texto
```

---

## Detalhes Técnicos & Decisões de Engenharia

### Feature Engineering & Filtros
- **Robustez a Ruído:** Substituição de médias móveis por Medianas Móveis (janelas de 5 a 30m) para mitigar picos elétricos sem distorcer a tendência.
- **Lags de Memória Profunda:** Utilização de atrasos entre 10 e 90 minutos para capturar a inércia real do processo, removendo lags curtos para evitar o efeito de "stutter" do hardware.
- **Features Cíclicas:** Codificação de tempo (hora/minuto/dia) via funções seno/cosseno para capturar sazonalidade de turnos.

### Saneamento de Dados
- **Sensor var14:** Removido do aprendizado por apresentar escala incoerente ($10^7$) e correlação nula, indicando falha de hardware.
- **Tratamento de Nulos:** Estratégia de Forward Fill com limite de 5 minutos para preservar a continuidade temporal sem introduzir viés de interpolação em paradas longas.

### Performance do Modelo (Random Forest)
O modelo final apresentou excelente capacidade de generalização em janelas temporais de teste não vistas no treino:

| Métrica | Valor Obtido |
| :--- | :--- |
| **RMSE** | 0.41 |
| **R² (Coef. Determinação)** | 0.76 |
| **MAE** | 0.24 |

---

## Como Executar

O projeto utiliza um ambiente virtual (`.venv`) para gerenciar dependências. **Não é necessário (e nem recomendado) transferir a pasta `.venv`**, pois ela é pesada e específica de cada máquina.

### 1. Preparação do Ambiente

Crie e ative o ambiente virtual, e então instale as dependências:

**Windows (PowerShell/CMD):**
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Pipeline de Dados (Ordem Obrigatória)
Execute os scripts na pasta src/ para processar os dados e treinar o modelo:
```bash
python src/01_Eda.py
python src/02_Preprocessing.py
python src/03_Modelling.py
```

### 3. Dashboard do Simulador
Para iniciar o monitoramento em tempo real:
```bash
streamlit run application/app.py
```

---

## Validação e Qualidade

- **Scripts de Produção:** O pipeline completo foi validado no ambiente virtual (`.venv`).
- **EDA Automatizada:** O script `01_Eda.py` gera automaticamente relatórios estatísticos em `/statistics` e visualizações em `/reports/figures`, garantindo a integridade dos dados antes da modelagem.
- **Logs de Execução:** Relatórios em texto (.txt) são gerados em cada etapa para auditoria de métricas e diagnóstico de sensores.


---

> [!NOTE]
> Este projeto foi desenvolvido com foco em estabilidade industrial e transparência algorítmica. O sistema de dupla validação (ML + Física) garante que a IA seja uma ferramenta de suporte, nunca substituindo o monitoramento de variáveis críticas de segurança.