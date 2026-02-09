"""Funções de carga e limpeza básica dos dados."""
import pandas as pd
import numpy as np
import re
from src.config import (
    CADASTRAL_FILE, INFO_FILE, PAGAMENTOS_DEV_FILE,
    PAGAMENTOS_TESTE_FILE, DELIMITER
)


def load_cadastral(filepath=None):
    """Carrega base cadastral com limpeza de DDD. Descarta clientes PF (FLAG_PF='X')."""
    filepath = filepath or CADASTRAL_FILE
    df = pd.read_csv(filepath, sep=DELIMITER)

    # Descartar Pessoa Física - foco exclusivo em Pessoa Jurídica
    n_before = len(df)
    pf_mask = df["FLAG_PF"] == "X"
    df = df[~pf_mask].copy()
    n_removed = n_before - len(df)
    if n_removed > 0:
        print(f"  Cadastral: {n_removed} clientes PF descartados (foco em PJ)")

    # Remover coluna FLAG_PF (agora constante)
    df.drop(columns=["FLAG_PF"], errors="ignore", inplace=True)

    # Converter DATA_CADASTRO para datetime
    df["DATA_CADASTRO"] = pd.to_datetime(df["DATA_CADASTRO"], format="%Y-%m-%d", errors="coerce")

    # Limpar DDD: remover parênteses e converter para numérico
    if "DDD" in df.columns:
        df["DDD"] = df["DDD"].astype(str).apply(
            lambda x: re.sub(r"[^\d]", "", x) if pd.notna(x) and x != "nan" else np.nan
        )
        df["DDD"] = pd.to_numeric(df["DDD"], errors="coerce")

    return df


def load_info(filepath=None):
    """Carrega base de informações mensais."""
    filepath = filepath or INFO_FILE
    df = pd.read_csv(filepath, sep=DELIMITER)

    # Converter SAFRA_REF (formato YYYY-MM) para datetime
    df["SAFRA_REF"] = pd.to_datetime(df["SAFRA_REF"], format="%Y-%m", errors="coerce")

    return df


def _parse_dates(df, date_cols_ymd, safra_col="SAFRA_REF"):
    """Parse date columns: YYYY-MM-DD for regular dates, YYYY-MM for SAFRA_REF."""
    for col in date_cols_ymd:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="%Y-%m-%d", errors="coerce")
    if safra_col in df.columns:
        df[safra_col] = pd.to_datetime(df[safra_col], format="%Y-%m", errors="coerce")
    return df


def load_pagamentos_dev(filepath=None):
    """Carrega base de pagamentos de desenvolvimento."""
    filepath = filepath or PAGAMENTOS_DEV_FILE
    df = pd.read_csv(filepath, sep=DELIMITER)

    df = _parse_dates(df, ["DATA_EMISSAO_DOCUMENTO", "DATA_PAGAMENTO", "DATA_VENCIMENTO"])

    return df


def load_pagamentos_teste(filepath=None):
    """Carrega base de pagamentos de teste (sem DATA_PAGAMENTO)."""
    filepath = filepath or PAGAMENTOS_TESTE_FILE
    df = pd.read_csv(filepath, sep=DELIMITER)

    df = _parse_dates(df, ["DATA_EMISSAO_DOCUMENTO", "DATA_VENCIMENTO"])

    return df


def load_all_data():
    """Carrega todas as 4 bases de dados, descartando clientes PF.

    Returns:
        tuple: (cadastral, info, pagamentos_dev, pagamentos_teste)
    """
    cadastral = load_cadastral()
    info = load_info()
    pag_dev = load_pagamentos_dev()
    pag_teste = load_pagamentos_teste()

    # Filtrar transações e info apenas de clientes PJ (presentes no cadastral após remoção de PF)
    clientes_pj = set(cadastral["ID_CLIENTE"].unique())

    n_dev_before = len(pag_dev)
    pag_dev = pag_dev[pag_dev["ID_CLIENTE"].isin(clientes_pj)].copy()
    if n_dev_before - len(pag_dev) > 0:
        print(f"  Dev: {n_dev_before - len(pag_dev)} transações PF descartadas")

    n_teste_before = len(pag_teste)
    pag_teste = pag_teste[pag_teste["ID_CLIENTE"].isin(clientes_pj)].copy()
    if n_teste_before - len(pag_teste) > 0:
        print(f"  Teste: {n_teste_before - len(pag_teste)} transações PF descartadas")

    n_info_before = len(info)
    info = info[info["ID_CLIENTE"].isin(clientes_pj)].copy()
    if n_info_before - len(info) > 0:
        print(f"  Info: {n_info_before - len(info)} registros PF descartados")

    print(f"Cadastral:       {cadastral.shape[0]:>6} registros, {cadastral.shape[1]} colunas")
    print(f"Info:            {info.shape[0]:>6} registros, {info.shape[1]} colunas")
    print(f"Pagamentos Dev:  {pag_dev.shape[0]:>6} registros, {pag_dev.shape[1]} colunas")
    print(f"Pagamentos Teste:{pag_teste.shape[0]:>6} registros, {pag_teste.shape[1]} colunas")

    return cadastral, info, pag_dev, pag_teste
