"""Lógica de criação de features para o modelo de score de crédito."""
import pandas as pd
import numpy as np
from scipy import stats
from src.config import (
    HIST_WINDOWS, DDD_REGIAO, DEFAULT_THRESHOLD_DAYS,
    COVID_START, COVID_END
)


def create_target(df):
    """Calcula variável target de inadimplência.

    Inadimplência = pagamento com 5+ dias de atraso em relação ao vencimento.
    """
    df = df.copy()
    df["DIAS_ATRASO"] = (df["DATA_PAGAMENTO"] - df["DATA_VENCIMENTO"]).dt.days
    df["TARGET"] = (df["DIAS_ATRASO"] >= DEFAULT_THRESHOLD_DAYS).astype(int)
    return df


def build_transaction_features(df):
    """Features derivadas da transação atual."""
    df = df.copy()

    # Dias entre emissão e vencimento
    df["DIAS_ATE_VENCIMENTO"] = (df["DATA_VENCIMENTO"] - df["DATA_EMISSAO_DOCUMENTO"]).dt.days
    # Cap em range razoável
    df["DIAS_ATE_VENCIMENTO"] = df["DIAS_ATE_VENCIMENTO"].clip(1, 120)

    # Componentes temporais do vencimento
    df["DIA_SEMANA_VENCIMENTO"] = df["DATA_VENCIMENTO"].dt.dayofweek.astype(str)
    df["MES_VENCIMENTO"] = df["DATA_VENCIMENTO"].dt.month.astype(str)

    # Componentes da safra
    df["MES_REF"] = df["SAFRA_REF"].dt.month
    df["ANO_REF"] = df["SAFRA_REF"].dt.year

    # Log do valor
    df["LOG_VALOR_A_PAGAR"] = np.log1p(df["VALOR_A_PAGAR"])

    # Indicador COVID
    safra_str = df["SAFRA_REF"].dt.to_period("M").astype(str)
    df["FLAG_COVID"] = ((safra_str >= COVID_START) & (safra_str <= COVID_END)).astype(int)

    return df


def _compute_behavioral_for_group(group_history, safra_ref, windows):
    """Calcula features comportamentais para um cliente usando apenas dados anteriores a safra_ref."""
    features = {}
    hist = group_history[group_history["SAFRA_REF"] < safra_ref].copy()

    if len(hist) == 0:
        return features

    # Features para janela "ALL" (todo histórico)
    features.update(_calc_window_features(hist, "ALL"))

    # Features por janela temporal
    for w in windows:
        cutoff = safra_ref - pd.DateOffset(months=w)
        hist_w = hist[hist["SAFRA_REF"] >= cutoff]
        features.update(_calc_window_features(hist_w, f"{w}M"))

    # Tendência de inadimplência (slope linear sobre médias mensais)
    monthly_default = hist.groupby("SAFRA_REF")["TARGET"].mean().sort_index()
    if len(monthly_default) >= 3:
        x = np.arange(len(monthly_default))
        slope, _, _, _, _ = stats.linregress(x, monthly_default.values)
        features["TREND_DEFAULT"] = slope
    else:
        features["TREND_DEFAULT"] = np.nan

    # Meses desde último default
    defaults = hist[hist["TARGET"] == 1]
    if len(defaults) > 0:
        last_default_safra = defaults["SAFRA_REF"].max()
        features["MESES_DESDE_ULTIMO_DEFAULT"] = (
            (safra_ref.year - last_default_safra.year) * 12
            + safra_ref.month - last_default_safra.month
        )
    else:
        features["MESES_DESDE_ULTIMO_DEFAULT"] = np.nan

    return features


def _calc_window_features(hist, suffix):
    """Calcula features para uma janela temporal específica."""
    features = {}
    n = len(hist)
    if n == 0:
        return features

    prefix = f"HIST_{suffix}"

    # Taxa de inadimplência
    features[f"{prefix}_TX_DEFAULT"] = hist["TARGET"].mean()
    # Dias de atraso
    features[f"{prefix}_MEDIA_ATRASO"] = hist["DIAS_ATRASO"].mean()
    features[f"{prefix}_MAX_ATRASO"] = hist["DIAS_ATRASO"].max()
    features[f"{prefix}_STD_ATRASO"] = hist["DIAS_ATRASO"].std() if n > 1 else 0
    # Contagem de transações
    features[f"{prefix}_QTD_TRANS"] = n
    # Valores
    features[f"{prefix}_MEDIA_VALOR"] = hist["VALOR_A_PAGAR"].mean()
    features[f"{prefix}_SOMA_VALOR"] = hist["VALOR_A_PAGAR"].sum()
    # Ratio adiantados
    features[f"{prefix}_RATIO_ADIANTADO"] = (hist["DIAS_ATRASO"] < 0).mean()

    return features


def build_behavioral_features(transactions_df, history_df):
    """Constrói features comportamentais para cada transação.

    IMPORTANTE: Usa apenas dados de períodos anteriores (sem leakage).

    Otimizado: calcula por par único (cliente, safra) e depois faz merge.

    Args:
        transactions_df: DataFrame com transações que queremos featurizar
        history_df: DataFrame com histórico de pagamentos (deve ter TARGET e DIAS_ATRASO)

    Returns:
        DataFrame com features comportamentais indexado igual a transactions_df
    """
    # Pares únicos (cliente, safra) para evitar recomputação
    unique_pairs = transactions_df[["ID_CLIENTE", "SAFRA_REF"]].drop_duplicates()

    # Agrupar histórico por cliente
    history_grouped = dict(list(history_df.groupby("ID_CLIENTE")))

    pair_results = {}
    total = len(unique_pairs)
    for i, (_, row) in enumerate(unique_pairs.iterrows()):
        cliente = row["ID_CLIENTE"]
        safra = row["SAFRA_REF"]

        if cliente in history_grouped:
            feats = _compute_behavioral_for_group(
                history_grouped[cliente], safra, HIST_WINDOWS
            )
        else:
            feats = {}

        pair_results[(cliente, safra)] = feats

        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{total} pares processados...")

    # Mapear de volta para cada transação
    results = []
    for idx, row in transactions_df.iterrows():
        key = (row["ID_CLIENTE"], row["SAFRA_REF"])
        feats = pair_results.get(key, {}).copy()
        feats["_idx"] = idx
        results.append(feats)

    feat_df = pd.DataFrame(results).set_index("_idx")
    feat_df.index.name = None
    return feat_df


def build_safra_context_features(df):
    """Features de contexto do mês/safra para cada cliente."""
    df = df.copy()

    # Agregar por cliente-safra
    safra_stats = df.groupby(["ID_CLIENTE", "SAFRA_REF"]).agg(
        QTD_TRANSACOES_MES=("VALOR_A_PAGAR", "count"),
        SOMA_VALOR_MES=("VALOR_A_PAGAR", "sum"),
        MEDIA_VALOR_MES=("VALOR_A_PAGAR", "mean"),
        MAX_VALOR_MES=("VALOR_A_PAGAR", "max"),
    ).reset_index()

    df = df.merge(safra_stats, on=["ID_CLIENTE", "SAFRA_REF"], how="left")
    return df


def build_cadastral_features(df, cadastral):
    """Merge com dados cadastrais e criação de features derivadas."""
    df = df.copy()

    # Merge
    df = df.merge(cadastral, on="ID_CLIENTE", how="left")

    # Tempo de cadastro em meses
    df["TEMPO_CADASTRO_MESES"] = (
        (df["SAFRA_REF"].dt.year - df["DATA_CADASTRO"].dt.year) * 12
        + df["SAFRA_REF"].dt.month - df["DATA_CADASTRO"].dt.month
    )

    # DDD -> Região
    df["DDD_REGIAO"] = df["DDD"].map(DDD_REGIAO).fillna("DESCONHECIDO")

    # Converter CEP_2_DIG para string
    df["CEP_2_DIG"] = df["CEP_2_DIG"].astype(str)

    # Converter PORTE e SEGMENTO para string (para categóricas)
    for col in ["PORTE", "SEGMENTO_INDUSTRIAL", "DOMINIO_EMAIL"]:
        if col in df.columns:
            df[col] = df[col].fillna("MISSING").astype(str)

    # Remover coluna de data intermediária
    df.drop(columns=["DATA_CADASTRO", "DDD"], errors="ignore", inplace=True)

    return df


def build_info_features(df, info):
    """Merge com dados de info mensal (renda, funcionários)."""
    df = df.copy()

    # Merge por cliente e safra
    df = df.merge(info, on=["ID_CLIENTE", "SAFRA_REF"], how="left")

    # Flags de missing
    df["RENDA_MISSING"] = df["RENDA_MES_ANTERIOR"].isna().astype(int)
    df["FUNC_MISSING"] = df["NO_FUNCIONARIOS"].isna().astype(int)

    # Log-transform renda
    df["LOG_RENDA_MES_ANTERIOR"] = np.log1p(df["RENDA_MES_ANTERIOR"].fillna(0))

    return df


def build_full_feature_matrix(transactions_df, history_df, cadastral, info, verbose=True):
    """Orquestrador: constrói a matriz completa de features.

    Args:
        transactions_df: Transações a featurizar
        history_df: Histórico de pagamentos (com TARGET e DIAS_ATRASO)
        cadastral: Base cadastral (já limpa)
        info: Base info mensal
        verbose: Se True, imprime progresso

    Returns:
        DataFrame com todas as features
    """
    if verbose:
        print("1/5 Features transacionais...")
    df = build_transaction_features(transactions_df)

    if verbose:
        print("2/5 Features de contexto da safra...")
    df = build_safra_context_features(df)

    if verbose:
        print("3/5 Features cadastrais...")
    df = build_cadastral_features(df, cadastral)

    if verbose:
        print("4/5 Features de info mensal...")
    df = build_info_features(df, info)

    if verbose:
        print("5/5 Features comportamentais (pode demorar)...")
    behavioral = build_behavioral_features(df, history_df)
    df = df.join(behavioral)

    if verbose:
        print(f"Matriz final: {df.shape[0]} linhas x {df.shape[1]} colunas")

    return df
