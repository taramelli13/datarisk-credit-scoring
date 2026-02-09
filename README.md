# Credit Scoring Model - Datarisk

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Modelo de Score de Crédito - Probabilidade de Inadimplência

### Problema
Construir um modelo preditivo que estime a probabilidade de inadimplência (0 a 1) para cada transação em `base_pagamentos_teste.csv`. Inadimplência = pagamento com 5+ dias de atraso em relação ao vencimento.

### Solução

Modelo: LightGBM com hiperparâmetros otimizados via Optuna (expanding window CV temporal)

Features (~58 total):
- Transacionais (7): valor, taxa, dias até vencimento, componentes temporais
- Comportamentais (34): histórico de pagamentos por janela temporal (3M, 6M, 12M, ALL) - taxa de default, média/max/std de atraso, tendência, etc.
- Cadastrais (7): porte, segmento, região, email, tempo de cadastro
- Info mensal (4): renda, funcionários, flags de missing
- Contexto da safra (4): qtd/soma/média/max de transações no mês
- COVID (1): indicador de período pandêmico

Anti-leakage: Features comportamentais usam apenas dados de SAFRA_REF estritamente anterior.

### Estrutura do projeto

```
datarisk-case-ds-junior-master/
├── data/                          # Dados brutos (não incluir no .zip)
├── notebooks/
│   ├── 01_eda_analise_exploratoria.ipynb
│   ├── 02_feature_engineering_modelagem.ipynb
│   └── 03_pipeline_producao_scoring.ipynb
├── src/
│   ├── config.py                  # Caminhos, constantes, parâmetros
│   ├── data_loader.py             # Carga e limpeza dos dados
│   ├── feature_engineering.py     # Criação de features
│   └── model_utils.py             # Treinamento, avaliação, visualização
├── outputs/
│   ├── submissao_case.csv         # Predições finais (12.275 linhas)
│   ├── modelo_final.joblib        # Modelo serializado
│   └── figures/                   # Gráficos salvos
├── requirements.txt
└── README.md
```

### Como executar

```bash
pip install -r requirements.txt
cd notebooks
jupyter nbconvert --to notebook --execute 01_eda_analise_exploratoria.ipynb
jupyter nbconvert --to notebook --execute 02_feature_engineering_modelagem.ipynb
jupyter nbconvert --to notebook --execute 03_pipeline_producao_scoring.ipynb
```

Os notebooks devem ser executados sequencialmente (02 gera a config usada por 03).

### Resultados

- submissao_case.csv: 12.275 linhas com colunas ID_CLIENTE, SAFRA_REF, PROBABILIDADE_INADIMPLENCIA
- Probabilidades entre 0 e 1, sem nulos
- Taxa média predita de inadimplência: ~9.8%

### Decisões técnicas

| Decisão | Justificativa |
|---------|---------------|
| LightGBM | Melhor performance em benchmarks; eficiente com categóricas |
| `is_unbalance=True` | Trata desbalanceamento (~7% default) sem distorcer calibração |
| Sem SMOTE | Preserva calibração das probabilidades |
| Split temporal | Respeita a natureza temporal do problema; evita leakage |
| Expanding window CV | Simula produção: treina no passado, valida no futuro |

### Bases de dados

- `base_cadastral.csv`: 1.315 clientes (dados estáticos)
- `base_info.csv`: 24.401 registros mensais (renda, funcionários)
- `base_pagamentos_desenvolvimento.csv`: 77.414 transações (com DATA_PAGAMENTO)
- `base_pagamentos_teste.csv`: 12.275 transações (sem DATA_PAGAMENTO)
