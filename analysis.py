"""
analysis.py
Funções de análise descritiva, testes inferenciais e geração de gráficos.
- Descriptive statistics
- Inferential tests (t-test independent, paired not used here, ANOVA)
- Plotly interactive plots for UI
- Matplotlib static plots (PNG bytes) for inclusion no PDF
"""

import io
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import ttest_ind, f_oneway, pearsonr, norm

sns.set(style="whitegrid", palette="pastel")

# ------------------------
# Estatísticas descritivas
# ------------------------
def descriptive_stats(series: pd.Series) -> dict:
    s = series.dropna()
    desc = {
        "count": int(s.count()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "mode": list(s.mode().values) if not s.mode().empty else [],
        "variance": float(s.var(ddof=1)),
        "std": float(s.std(ddof=1)),
        "min": float(s.min()),
        "max": float(s.max()),
        "range": float(s.max() - s.min()),
        "q1": float(s.quantile(0.25)),
        "q3": float(s.quantile(0.75)),
        "iqr": float(s.quantile(0.75) - s.quantile(0.25)),
    }
    return desc

# ------------------------
# Testes inferenciais
# ------------------------
def inferential_tests(df: pd.DataFrame) -> dict:
    results = {}

    # t-test independent: Masculino vs Feminino (Satisfação Geral)
    g_m = df[df["Gênero"] == "Masculino"]["Satisfação Geral"].dropna()
    g_f = df[df["Gênero"] == "Feminino"]["Satisfação Geral"].dropna()
    if len(g_m) > 1 and len(g_f) > 1:
        t_stat, p_val = ttest_ind(g_m, g_f, equal_var=False)
        results["t_gender_satisfacao"] = {"t_stat": float(t_stat), "p_value": float(p_val)}
    else:
        results["t_gender_satisfacao"] = {"t_stat": None, "p_value": None}

    # ANOVA: Escolaridade vs Satisfação Geral
    groups = [group["Satisfação Geral"].dropna().values for _, group in df.groupby("Escolaridade")]
    if sum(len(g) > 0 for g in groups) > 1:
        f_stat, p_anova = f_oneway(*groups)
        results["anova_escolaridade"] = {"f_stat": float(f_stat), "p_value": float(p_anova)}
    else:
        results["anova_escolaridade"] = {"f_stat": None, "p_value": None}

    # Pearson correlations among numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = df[numeric_cols].corr(method="pearson")
    results["pearson_corr"] = corr_matrix

    return results

# ------------------------
# Plotly interactive plots
# ------------------------
def plotly_hist(df: pd.DataFrame, col: str, nbins: int = 30):
    fig = px.histogram(df, x=col, nbins=nbins, marginal="box", opacity=0.75)
    fig.update_layout(title=f"Distribuição: {col}", xaxis_title=col, yaxis_title="Contagem")
    return fig

def plotly_corr_heatmap(df: pd.DataFrame, cols: list):
    corr = df[cols].corr(method="pearson").round(2)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", origin="lower")
    fig.update_layout(title="Matriz de Correlação (Pearson)")
    return fig

def plotly_box(df: pd.DataFrame, x: str, y: str):
    fig = px.box(df, x=x, y=y, points="all")
    fig.update_layout(title=f"Boxplot: {y} por {x}")
    return fig

def plotly_normal_overlay(series: pd.Series):
    # create histogram + normal curve using Plotly
    x = np.linspace(series.min(), series.max(), 200)
    mu, sigma = series.mean(), series.std(ddof=1)
    pdf_vals = norm.pdf(x, mu, sigma)
    hist = go.Histogram(x=series, histnorm='probability density', name='Dados', opacity=0.6)
    line = go.Scatter(x=x, y=pdf_vals, mode='lines', name='Normal teórica', line=dict(color='red', width=2))
    fig = go.Figure(data=[hist, line])
    fig.update_layout(title=f"Histograma com Curva Normal - {series.name}", xaxis_title=series.name, yaxis_title="Densidade")
    return fig

# ------------------------
# Matplotlib figures -> PNG bytes (para PDF)
# ------------------------
def matplotlib_hist_bytes(series: pd.Series, title: str = None) -> bytes:
    title = title or f"Histograma - {series.name}"
    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(series, kde=True, stat="density", ax=ax, color="skyblue")
    mu, sigma = series.mean(), series.std(ddof=1)
    x = np.linspace(series.min(), series.max(), 200)
    ax.plot(x, norm.pdf(x, mu, sigma), 'k--', lw=2, label="Normal teórica")
    ax.set_title(title)
    ax.set_xlabel(series.name)
    ax.set_ylabel("Densidade")
    ax.legend()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()

def matplotlib_box_bytes(df: pd.DataFrame, x: str, y: str, title: str=None) -> bytes:
    title = title or f"Boxplot - {y} por {x}"
    fig, ax = plt.subplots(figsize=(6,4))
    sns.boxplot(x=x, y=y, data=df, ax=ax)
    sns.pointplot(x=x, y=y, data=df, estimator=np.mean, color="red", markers="D", ci=None, ax=ax)
    ax.set_title(title)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()

def matplotlib_corr_bytes(df: pd.DataFrame, cols: list, title: str=None) -> bytes:
    title = title or "Matriz de Correlação"
    fig, ax = plt.subplots(figsize=(6,5))
    corr = df[cols].corr(method="pearson")
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title(title)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()

