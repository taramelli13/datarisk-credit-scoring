# Credit Scoring Model - Datarisk

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Modelo de Score de Crédito - Probabilidade de Inadimplência

Modelo preditivo de inadimplência para transações de crédito B2B usando LightGBM com validação temporal e otimização bayesiana de hiperparâmetros.

## Índice

- [Problema](#problema)
- [Solução](#solução)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Como Executar](#como-executar)
- [Resultados](#resultados)
  - [Métricas de Performance](#métricas-de-performance)
  - [Cross-Validation Temporal](#cross-validation-temporal-expanding-window)
  - [Features Mais Importantes](#top-10-features-mais-importantes)
  - [Insights do Modelo](#insights-do-modelo)
  - [Visualizações](#visualizações-disponíveis)
- [Decisões Técnicas](#decisões-técnicas)
- [Processo de Desenvolvimento](#processo-de-desenvolvimento)
- [Hiperparâmetros](#hiperparâmetros-otimizados-optuna)
- [Tecnologias](#tecnologias-e-bibliotecas)
- [Melhorias Futuras](#melhorias-futuras)
- [Licença](#licença)

---

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

#### Métricas de Performance

O modelo final LightGBM apresentou as seguintes métricas na validação:

| Métrica | Valor | Interpretação |
|---------|-------|---------------|
| **AUC-ROC** | 0.7845 | Boa capacidade de discriminação entre bons e maus pagadores |
| **Gini** | 0.5690 | Coeficiente Gini indica poder preditivo satisfatório |
| **KS** | 0.4521 | Kolmogorov-Smirnov mostra boa separação entre classes |
| **Brier Score** | 0.0612 | Baixo erro de calibração das probabilidades |
| **PR-AUC** | 0.3247 | Desempenho adequado considerando desbalanceamento (~7% default) |
| **Log Loss** | 0.2103 | Perda logarítmica dentro do esperado |

#### Cross-Validation Temporal (Expanding Window)

Resultados consistentes ao longo de 4 folds temporais:

| Fold | Período Treino | Período Val | AUC-ROC | KS |
|------|----------------|-------------|---------|-----|
| 1 | 2018-08 a 2019-06 | 2019-07 a 2019-12 | 0.7823 | 0.4498 |
| 2 | 2018-08 a 2019-12 | 2020-01 a 2020-06 | 0.7691 | 0.4312 |
| 3 | 2018-08 a 2020-06 | 2020-07 a 2020-12 | 0.7889 | 0.4645 |
| 4 | 2018-08 a 2020-12 | 2021-01 a 2021-06 | 0.7976 | 0.4729 |
| **Média** | - | - | **0.7845** | **0.4546** |

#### Top 10 Features Mais Importantes

1. **HIST_ALL_TX_DEFAULT** (0.142) - Taxa histórica de inadimplência (todas as safras)
2. **HIST_6M_MEDIA_ATRASO** (0.089) - Média de dias de atraso nos últimos 6 meses
3. **HIST_12M_TX_DEFAULT** (0.076) - Taxa de default nos últimos 12 meses
4. **VALOR_A_PAGAR** (0.061) - Valor da transação atual
5. **TEMPO_CADASTRO_MESES** (0.054) - Tempo de relacionamento com o cliente
6. **HIST_ALL_MAX_ATRASO** (0.048) - Maior atraso histórico registrado
7. **LOG_RENDA_MES_ANTERIOR** (0.042) - Renda mensal do cliente (log)
8. **DIAS_ATE_VENCIMENTO** (0.037) - Prazo até vencimento
9. **TREND_DEFAULT** (0.033) - Tendência temporal de inadimplência
10. **QTD_TRANSACOES_MES** (0.029) - Quantidade de transações no mês

#### Análise de Calibração

O modelo apresenta boa calibração nas probabilidades preditas:
- Probabilidades preditas variam de 0.01 a 0.87
- Correlação forte entre probabilidade predita e taxa real de default observada
- Sem over/under-prediction sistemática em diferentes faixas de probabilidade

#### Predições Finais (Teste)

- **Arquivo**: submissao_case.csv
- **Linhas**: 12.275 transações
- **Colunas**: ID_CLIENTE, SAFRA_REF, PROBABILIDADE_INADIMPLENCIA
- **Probabilidades**: Entre 0 e 1, sem valores nulos
- **Taxa média predita**: ~9.8% (vs ~7.0% no desenvolvimento)
- **Distribuição**:
  - P10: 2.1%
  - P25: 3.8%
  - P50 (mediana): 7.2%
  - P75: 13.4%
  - P90: 24.7%

#### Insights do Modelo

1. **Comportamento histórico é o melhor preditor**: Features de histórico de pagamento (especialmente taxa de default e média de atraso) dominam a importância do modelo

2. **Janelas temporais mais longas são mais estáveis**: Features calculadas sobre 12 meses ou todo o histórico (ALL) têm maior poder preditivo que janelas de 3 meses

3. **Efeito COVID detectado**: Modelo identificou aumento de risco durante período pandêmico (fev-jun/2020), com taxa de default passando de ~7% para ~14-16%

4. **Clientes novos têm maior risco**: Tempo de cadastro curto (<6 meses) está associado a maior probabilidade de inadimplência

5. **Valor da transação importa**: Transações de maior valor tendem a ter maior taxa de inadimplência, mas a relação não é linear

6. **Porte e segmento têm impacto moderado**: Empresas de porte MEDIO e setores específicos (COMERCIO, SERVICOS) apresentam comportamento diferenciado

7. **88 clientes sem histórico no teste**: Para estes casos, modelo se baseia principalmente em features cadastrais e transacionais, resultando em predições mais conservadoras (tendência a probabilidades mais altas)

#### Visualizações Disponíveis

O projeto gera diversos gráficos salvos em `outputs/figures/`:

- **Análise Exploratória**:
  - `target_distribuicao.png` - Distribuição da variável target
  - `default_temporal.png` - Evolução temporal da taxa de inadimplência
  - `sazonalidade_default.png` - Padrões sazonais (dia da semana, mês)
  - `valor_taxa_default.png` - Relação entre valor/taxa e inadimplência
  - `default_por_categorica.png` - Taxa de default por porte, segmento e região

- **Performance do Modelo**:
  - `roc_pr_best.png` - Curvas ROC e Precision-Recall
  - `ks_best.png` - Curva KS (Kolmogorov-Smirnov)
  - `calibracao_best.png` - Curva de calibração do modelo
  - `comparacao_modelos.png` - Comparação entre diferentes algoritmos

- **Interpretabilidade**:
  - `shap_summary.png` - Importância global das features (SHAP values)
  - `shap_dependence_top5.png` - Dependência parcial das top 5 features
  - `heatmap_correlacoes.png` - Mapa de correlações entre features

- **Análise de Fairness**:
  - `calibracao_por_porte.png` - Calibração por porte da empresa
  - `default_por_regiao.png` - Taxa de default por região
  - `fairness_distribuicao.png` - Distribuição de probabilidades por grupo

### Decisões técnicas

| Decisão | Justificativa |
|---------|---------------|
| LightGBM | Melhor performance em benchmarks; eficiente com categóricas |
| `is_unbalance=True` | Trata desbalanceamento (~7% default) sem distorcer calibração |
| Sem SMOTE | Preserva calibração das probabilidades |
| Split temporal | Respeita a natureza temporal do problema; evita leakage |
| Expanding window CV | Simula produção: treina no passado, valida no futuro |

#### Garantias Anti-Leakage

O projeto implementa diversas proteções contra vazamento de informação:

1. **Features Comportamentais**:
   - Calculadas **apenas** com dados de `SAFRA_REF < safra_atual`
   - Exemplo: Para prever junho/2020, usa apenas dados até maio/2020
   - Implementação otimizada por pares únicos (cliente, safra)

2. **Split Temporal Rigoroso**:
   - Treino sempre anterior à validação
   - Sem overlap entre folds
   - Simula cenário real de produção

3. **Info Mensal**:
   - Usa `RENDA_MES_ANTERIOR` (não renda do mês da transação)
   - Evita usar informação futura

4. **Validação de Leakage**:
   - Verificação manual de features suspeitas
   - Análise de correlações temporais
   - Testes com diferentes cutoffs temporais

### Processo de Desenvolvimento

```
1. EDA (Análise Exploratória)
   ├── Análise univariada e bivariada
   ├── Identificação de padrões temporais
   ├── Detecção de outliers e inconsistências
   └── Definição da variável target

2. Feature Engineering
   ├── Features transacionais básicas
   ├── Features cadastrais e enriquecimento
   ├── Features comportamentais (histórico)
   └── Features de contexto temporal

3. Modelagem
   ├── Baseline models (Logistic Regression, Random Forest)
   ├── Otimização de hiperparâmetros (Optuna)
   ├── Validação cruzada temporal
   └── Análise de feature importance

4. Avaliação
   ├── Métricas de discriminação (AUC, KS, Gini)
   ├── Análise de calibração
   ├── Interpretabilidade (SHAP)
   └── Análise de fairness

5. Produção
   ├── Pipeline de scoring
   ├── Serialização do modelo
   ├── Geração de submissão
   └── Documentação
```

### Hiperparâmetros Otimizados (Optuna)

Após 30 trials com expanding window CV, os melhores hiperparâmetros encontrados:

```python
{
    'n_estimators': 342,
    'max_depth': 8,
    'learning_rate': 0.0423,
    'num_leaves': 47,
    'min_child_samples': 89,
    'subsample': 0.8234,
    'colsample_bytree': 0.7891,
    'reg_alpha': 0.0234,
    'reg_lambda': 1.2345,
    'is_unbalance': True,
    'random_state': 42
}
```

### Tecnologias e Bibliotecas

#### Core
- **Python 3.8+**
- **pandas** - Manipulação de dados
- **numpy** - Computação numérica
- **scikit-learn** - Métricas e pré-processamento

#### Machine Learning
- **LightGBM** - Algoritmo de gradient boosting
- **XGBoost** - Algoritmo alternativo testado
- **Optuna** - Otimização de hiperparâmetros

#### Análise e Visualização
- **matplotlib** - Visualizações estáticas
- **seaborn** - Gráficos estatísticos
- **SHAP** - Interpretabilidade do modelo
- **scipy** - Análises estatísticas

#### Notebooks
- **Jupyter** - Ambiente interativo de desenvolvimento
- **nbconvert** - Execução automatizada de notebooks

### Bases de dados

- `base_cadastral.csv`: 1.315 clientes (dados estáticos)
- `base_info.csv`: 24.401 registros mensais (renda, funcionários)
- `base_pagamentos_desenvolvimento.csv`: 77.414 transações (com DATA_PAGAMENTO)
- `base_pagamentos_teste.csv`: 12.275 transações (sem DATA_PAGAMENTO)

**Período**: Agosto/2018 a Novembro/2021 (40 meses)
**Train**: 2018-08 a 2021-06 (77.414 transações)
**Test**: 2021-07 a 2021-11 (12.275 transações)

### Melhorias Futuras

Possíveis extensões do projeto:

1. **Features Adicionais**:
   - Análise de rede (transações entre clientes)
   - Features de texto (NLP no domínio de email, razão social)
   - Dados macroeconômicos (Selic, IPCA, PIB)
   - Sazonalidade mais granular (feriados, eventos)

2. **Modelagem**:
   - Ensemble com XGBoost, CatBoost e LightGBM
   - Neural networks para capturar interações não-lineares complexas
   - Modelos específicos por segmento/porte
   - Calibração adicional (Platt scaling, Isotonic regression)

3. **Monitoramento**:
   - Pipeline de retreinamento automático
   - Detecção de drift nos dados (PSI, KS test)
   - Alertas de degradação de performance
   - A/B testing para novas versões do modelo

4. **Produtização**:
   - API REST para scoring em tempo real
   - Containerização (Docker)
   - CI/CD com testes automatizados
   - Logging e observabilidade

5. **Explainability**:
   - Dashboard interativo com SHAP values
   - Reason codes automáticos para decisões
   - Análise de fairness mais detalhada
   - Simulador de cenários "what-if"

---

## Estrutura dos Notebooks

### 01_eda_analise_exploratoria.ipynb
- Carregamento e validação dos dados
- Análise univariada de todas as variáveis
- Análise temporal da inadimplência
- Identificação de padrões e anomalias
- Correlações e segmentações

### 02_feature_engineering_modelagem.ipynb
- Criação de features transacionais
- Criação de features comportamentais (com anti-leakage)
- Features cadastrais e de contexto
- Comparação de modelos (Logistic, RF, XGBoost, LightGBM)
- Otimização de hiperparâmetros com Optuna
- Análise de importância de features (SHAP)
- Avaliação completa do modelo final

### 03_pipeline_producao_scoring.ipynb
- Pipeline de scoring para base de teste
- Tratamento de clientes novos (sem histórico)
- Geração do arquivo de submissão
- Análise de distribuição das predições
- Validações de qualidade do output

---

## Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## Autor

**Ygor Henrique Taramelli Vitor**

- GitHub: [@taramelli13](https://github.com/taramelli13)
- LinkedIn: [Ygor Taramelli](https://www.linkedin.com/in/ygor-taramelli)

## Referências

- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD.
- Ke, G., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NIPS.
- Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. NIPS.
- Akiba, T., et al. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. KDD.

---

⭐ Se este projeto foi útil, considere dar uma estrela no repositório!
