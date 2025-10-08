"""
app.py
Streamlit application that uses the modules:
- data_generator.py
- analysis.py
- report.py

Execute:
    streamlit run app.py
"""
import streamlit as st
import pandas as pd
from io import BytesIO

from data_generator import generate_synthetic_afirmacao
import analysis as an
from report import create_pdf_report

st.set_page_config(page_title="AfirmAção - Análise Exploratória (Modular)", layout="wide")
st.title("AfirmAção — App de Análise Exploratória (Modular)")

# Sidebar controls
st.sidebar.header("Configurações")
n = st.sidebar.slider("Número de respondentes", 50, 1000, 300, step=50)
seed = st.sidebar.number_input("Seed (aleatoriedade)", value=42, step=1)

if st.sidebar.button("Gerar base sintética"):
    df = generate_synthetic_afirmacao(n=n, seed=int(seed))
    st.session_state["df"] = df
elif "df" not in st.session_state:
    st.session_state["df"] = generate_synthetic_afirmacao(n=n, seed=int(seed))

df = st.session_state["df"]

st.subheader("Dados (prévia)")
st.dataframe(df.head())

# Descriptive section
st.header("1. Estatística Descritiva")
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
variavel = st.selectbox("Escolha variável numérica", numeric_cols)
series = df[variavel]
stats = an.descriptive_stats(series)
st.write(stats)

# Interactive histogram (Plotly)
st.subheader("Distribuição interativa")
fig_hist = an.plotly_hist(df, variavel)
st.plotly_chart(fig_hist, use_container_width=True)

# Normal overlay
st.subheader("Histograma + Curva Normal (Plotly)")
fig_norm = an.plotly_normal_overlay(series)
st.plotly_chart(fig_norm, use_container_width=True)

# Inferential tests
st.header("2. Testes Inferenciais")
infer = an.inferential_tests(df)

st.subheader("T-test: Masculino vs Feminino (Satisfação Geral)")
t_res = infer.get("t_gender_satisfacao", {})
st.write(t_res)

st.subheader("ANOVA: Escolaridade vs Satisfação Geral")
anova_res = infer.get("anova_escolaridade", {})
st.write(anova_res)

st.subheader("Matriz de Correlação (Pearson) - interativa")
num_cols = ["Idade", "Tempo de Uso (meses)", "Satisfação Geral", "Facilidade de Uso", "Inclusão Percebida", "Confiança na Plataforma", "Engajamento (1-5)"]
fig_corr = an.plotly_corr_heatmap(df, num_cols)
st.plotly_chart(fig_corr, use_container_width=True)

# Conclusion
st.header("3. Conclusão")
st.markdown("""
- Use os gráficos interativos e as saídas estatísticas para compor o relatório APS (máx. 3 páginas).
- Limitações: base sintética, viés geracional de geração aleatória, amostra hipotética.
""")

# Export buttons (CSV + PDF)
st.header("Exportar")
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Baixar CSV", data=csv, file_name="base_afirmacao.csv", mime="text/csv")

# Prepare images (PNG bytes) using matplotlib utilities for embedding in PDF
img_hist = an.matplotlib_hist_bytes(series, title=f"Histograma - {variavel}")
img_box = an.matplotlib_box_bytes(df, x="Escolaridade", y="Satisfação Geral", title="Boxplot - Satisfação por Escolaridade")
img_corr = an.matplotlib_corr_bytes(df, num_cols, title="Matriz de Correlação (Pearson)")

image_bytes = {
    f"Histograma_{variavel}": img_hist,
    "Boxplot_Satisfacao_Escolaridade": img_box,
    "Correlacoes_Pearson": img_corr
}

# Create simple summarized stats for PDF
stats_for_pdf = {
    "variavel": variavel,
    "n_obs": len(df),
    "mean": round(float(stats["mean"]), 3),
    "median": round(float(stats["median"]), 3),
    "std": round(float(stats["std"]), 3)
}
infer_for_pdf = {
    "t_gender_satisfacao": infer.get("t_gender_satisfacao"),
    "anova_escolaridade": infer.get("anova_escolaridade")
}

if st.button("Gerar e baixar PDF (relatório)"):
    pdf_bytes = create_pdf_report(df, variavel, stats_for_pdf, infer_for_pdf, image_bytes, n)
    st.download_button("Download PDF", data=pdf_bytes, file_name="relatorio_afirmacao.pdf", mime="application/pdf")

st.caption("Arquitetura modular: edite os módulos em data_generator.py, analysis.py e report.py conforme necessário.")

