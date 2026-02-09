"""Configurações globais do projeto."""
from pathlib import Path

# Diretórios
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
NOTEBOOKS_DIR = PROJECT_DIR / "notebooks"

# Garantir que diretórios existem
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Arquivos de dados
CADASTRAL_FILE = DATA_DIR / "base_cadastral.csv"
INFO_FILE = DATA_DIR / "base_info.csv"
PAGAMENTOS_DEV_FILE = DATA_DIR / "base_pagamentos_desenvolvimento.csv"
PAGAMENTOS_TESTE_FILE = DATA_DIR / "base_pagamentos_teste.csv"

# Parâmetros gerais
DELIMITER = ";"
RANDOM_SEED = 42
DEFAULT_THRESHOLD_DAYS = 5

# Janelas para features comportamentais (em meses)
HIST_WINDOWS = [3, 6, 12]

# Valores conhecidos de TAXA
TAXAS_CONHECIDAS = [4.99, 5.99, 6.99, 8.99, 11.99]

# Mapeamento DDD -> Região brasileira
DDD_REGIAO = {
    # São Paulo
    11: "Sudeste", 12: "Sudeste", 13: "Sudeste", 14: "Sudeste", 15: "Sudeste",
    16: "Sudeste", 17: "Sudeste", 18: "Sudeste", 19: "Sudeste",
    # Rio de Janeiro
    21: "Sudeste", 22: "Sudeste", 24: "Sudeste",
    # Espírito Santo
    27: "Sudeste", 28: "Sudeste",
    # Minas Gerais
    31: "Sudeste", 32: "Sudeste", 33: "Sudeste", 34: "Sudeste", 35: "Sudeste",
    37: "Sudeste", 38: "Sudeste",
    # Paraná
    41: "Sul", 42: "Sul", 43: "Sul", 44: "Sul", 45: "Sul", 46: "Sul",
    # Santa Catarina
    47: "Sul", 48: "Sul", 49: "Sul",
    # Rio Grande do Sul
    51: "Sul", 53: "Sul", 54: "Sul", 55: "Sul",
    # Distrito Federal
    61: "Centro-Oeste",
    # Goiás
    62: "Centro-Oeste", 64: "Centro-Oeste",
    # Mato Grosso do Sul
    67: "Centro-Oeste",
    # Mato Grosso
    65: "Centro-Oeste", 66: "Centro-Oeste",
    # Bahia
    71: "Nordeste", 73: "Nordeste", 74: "Nordeste", 75: "Nordeste", 77: "Nordeste",
    # Sergipe
    79: "Nordeste",
    # Pernambuco
    81: "Nordeste", 87: "Nordeste",
    # Alagoas
    82: "Nordeste",
    # Paraíba
    83: "Nordeste",
    # Rio Grande do Norte
    84: "Nordeste",
    # Ceará
    85: "Nordeste", 88: "Nordeste",
    # Piauí
    86: "Nordeste", 89: "Nordeste",
    # Maranhão
    98: "Nordeste", 99: "Nordeste",
    # Pará
    91: "Norte", 93: "Norte", 94: "Norte",
    # Amazonas
    92: "Norte", 97: "Norte",
    # Amapá
    96: "Norte",
    # Roraima
    95: "Norte",
    # Tocantins
    63: "Norte",
    # Acre
    68: "Norte",
    # Rondônia
    69: "Norte",
}

# Features categóricas e numéricas para o pipeline
CATEGORICAL_FEATURES = [
    "PORTE", "SEGMENTO_INDUSTRIAL", "DOMINIO_EMAIL", "DDD_REGIAO",
    "CEP_2_DIG", "MES_VENCIMENTO", "DIA_SEMANA_VENCIMENTO",
]

NUMERIC_FEATURES_BASE = [
    "VALOR_A_PAGAR", "TAXA", "LOG_VALOR_A_PAGAR",
    "DIAS_ATE_VENCIMENTO", "MES_REF", "ANO_REF",
    "TEMPO_CADASTRO_MESES",
    "LOG_RENDA_MES_ANTERIOR", "NO_FUNCIONARIOS",
    "RENDA_MISSING", "FUNC_MISSING",
    "QTD_TRANSACOES_MES", "SOMA_VALOR_MES", "MEDIA_VALOR_MES", "MAX_VALOR_MES",
]

# Período COVID para feature indicadora
COVID_START = "2020-02"
COVID_END = "2020-06"
