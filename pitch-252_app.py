# ================================================================
# REPARA ANALYTICS v4.0 — STREAMLIT + GEMINI 2.5 FLASH
# Com autenticação, análise IA, dashboards e CSV
# ================================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import google.generativeai as genai
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import os

# =======================================================
# CONFIGURAÇÃO DO GEMINI VIA SECRETS
# =======================================================
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# =======================================================
# CONFIGURAÇÃO DO LOGIN via streamlit-authenticator
# =======================================================

config = {
    "credentials": {
        "usernames": {
            "admin": {
                "name": "Administrador",
                "password": st.secrets["ADMIN_PASSWORD"], 
            }
        }
    },
    "cookie": {
        "name": "repara_login",
        "key": st.secrets["SIGN_KEY"],
        "expiry_days": 3
    },
    "preauthorized": {
        "emails": ["admin@repara.com"]
    }
}

authenticator = stauth.Authenticate(
    config["credentials"], 
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
)

name, auth_status, username = authenticator.login("Login", "main")

# =======================
# LOGIN BLOCK
# =======================
if auth_status == False:
    st.error("❌ Usuário ou senha incorretos.")

if auth_status == None:
    st.warning("🟡 Digite suas credenciais para acessar.")
    st.stop()

# Usuário autenticado
authenticator.logout("Sair", "sidebar")

# =======================
# INTERFACE PRINCIPAL
# =======================

st.set_page_config(page_title="Repara Analytics", layout="wide")
st.title("📊 Repara Analytics v4.0 — IA + CSV + Dashboards")

st.write(f"Bem-vindo, **{name}** 👋")


# =======================================================
# FUNÇÃO IA PARA ANÁLISE PROFUNDA
# =======================================================
def analisar_texto_gemini(lista_textos, titulo="Análise"):
    if len(lista_textos) == 0:
        return "Nenhum texto disponível."

    texto_unificado = "\n".join([str(t) for t in lista_textos if pd.notna(t)])

    prompt = f"""
Você é um analista sênior de dados qualitativos, especialista em RH e comportamento.

Analise profundamente o conjunto de respostas e produza:

1. Resumo Executivo  
2. Top 10 temas com percentuais  
3. Emoções predominantes  
4. Clusters semânticos  
5. Principais dores  
6. Causas prováveis  
7. Recomendações para o MVP Repara  
8. Métricas sugeridas  
9. Tabela final com: Tema | Exemplo | Impacto | Ação

TEXTO:
----------------------------
{texto_unificado}
----------------------------

Responda em MARKDOWN estruturado.
"""

    model = genai.GenerativeModel("gemini-2.5-flash")
    resposta = model.generate_content(prompt)
    return resposta.text


# =======================================================
# FUNÇÕES AUXILIARES
# =======================================================
def read_csv_any(file):
    try:
        return pd.read_csv(file, engine="python", sep=None)
    except:
        return pd.read_csv(file, engine="python", sep=",")


def infer_columns(df: pd.DataFrame):
    mapping = {}
    cols = df.columns.str.lower()

    def find(patterns):
        for p in patterns:
            for c in df.columns:
                if p in c.lower():
                    return c
        return None

    mapping["age"] = find(["idade", "age"])
    mapping["gender"] = find(["genero", "sexo", "gender"])
    mapping["pain"] = find(["pain", "problema", "desafio", "dor", "feedback"])
    mapping["hr"] = find(["hr", "gestao", "human", "challenge", "desafio"])

    return mapping


# =======================================================
# UPLOAD DE ARQUIVOS
# =======================================================
st.sidebar.header("📥 Upload de CSVs")
cand_file = st.sidebar.file_uploader("Candidatos CSV", type=["csv"])
emp_file = st.sidebar.file_uploader("Empresas CSV", type=["csv"])

tab1, tab2, tab3 = st.tabs(["👤 Candidatos", "🏢 Empresas", "🔀 Cruzada"])


# =======================================================
# TAB 1 — CANDIDATOS
# =======================================================
with tab1:
    st.header("👤 Análise de Candidatos")

    if cand_file:
        df = read_csv_any(cand_file)
        mapping = infer_columns(df)

        st.subheader("📄 Prévia")
        st.dataframe(df.head())

        col1, col2, col3 = st.columns(3)
        col1.metric("Total", len(df))
        col2.metric("Colunas", len(df.columns))
        col3.metric("Campos texto", sum(df.dtypes == "object"))

        if mapping["age"]:
            st.subheader("📊 Idade")
            fig, ax = plt.subplots(figsize=(6,4))
            df[mapping["age"]].fillna("N/A").value_counts().plot(kind='bar', ax=ax)
            st.pyplot(fig)

        if mapping["gender"]:
            st.subheader("📊 Gênero")
            fig, ax = plt.subplots(figsize=(5,5))
            df[mapping["gender"]].fillna("N/A").value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
            ax.set_ylabel("")
            st.pyplot(fig)

        if mapping["pain"]:
            st.subheader("☁️ Wordcloud")
            text = " ".join(df[mapping["pain"]].dropna().astype(str))
            wc = WordCloud(width=900, height=400).generate(text)
            fig, ax = plt.subplots(figsize=(10,4))
            ax.imshow(wc)
            ax.axis("off")
            st.pyplot(fig)

            st.subheader("🤖 IA — Análise Profunda")
            if st.button("Executar IA (Candidatos)"):
                textos = df[mapping["pain"]].dropna().astype(str).tolist()
                with st.spinner("Analisando com IA..."):
                    out = analisar_texto_gemini(textos)
                st.markdown(out)


# =======================================================
# TAB 2 — EMPRESAS
# =======================================================
with tab2:
    st.header("🏢 Análise das Empresas")

    if emp_file:
        df = read_csv_any(emp_file)
        mapping = infer_columns(df)

        st.subheader("📄 Prévia")
        st.dataframe(df.head())

        if mapping["hr"]:
            st.subheader("📊 Desafios de RH")
            fig, ax = plt.subplots(figsize=(6,4))
            df[mapping["hr"]].dropna().astype(str).value_counts().head(6).plot(kind='barh', ax=ax)
            st.pyplot(fig)

            st.subheader("🤖 IA — Análise Profunda")
            if st.button("Executar IA (Empresas)"):
                textos = df[mapping["hr"]].dropna().astype(str).tolist()
                with st.spinner("IA analisando..."):
                    out = analisar_texto_gemini(textos)
                st.markdown(out)


# =======================================================
# TAB 3 — ANÁLISE CRUZADA
# =======================================================
with tab3:
    st.header("🔀 Análise Cruzada Candidatos x Empresas")

    if cand_file and emp_file:
        df_c = read_csv_any(cand_file)
        df_e = read_csv_any(emp_file)

        map_c = infer_columns(df_c)
        map_e = infer_columns(df_e)

        textos = []

        if map_c["pain"]:
            textos += df_c[map_c["pain"]].dropna().astype(str).tolist()

        if map_e["hr"]:
            textos += df_e[map_e["hr"]].dropna().astype(str).tolist()

        if st.button("Executar IA Cruzada"):
            with st.spinner("IA encontrando padrões, temas e conexões..."):
                out = analisar_texto_gemini(textos, "Cruzada")
            st.markdown(out)
    else:
        st.info("Carregue os dois CSVs para ativar esta seção.")
