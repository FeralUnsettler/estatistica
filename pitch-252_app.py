import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA
# -------------------------------------------------------------
st.set_page_config(
    page_title="Revela Talentos – Analytics",
    layout="wide",
    page_icon="🧠"
)

# -------------------------------------------------------------
# SIDEBAR – FILTROS E INPUTS
# -------------------------------------------------------------
st.sidebar.header("⚙️ Configurações do Dashboard")

uploaded_candidates = st.sidebar.file_uploader(
    "📂 Carregar base de candidatos (CSV)",
    type=["csv"]
)

uploaded_companies = st.sidebar.file_uploader(
    "🏢 Carregar base de vagas/empresas (CSV)",
    type=["csv"]
)

senioridade = st.sidebar.multiselect(
    "🧩 Senioridade",
    ["Estágio", "Júnior", "Pleno", "Sênior", "Líder"],
    default=[]
)

competencias_filtrar = st.sidebar.text_input(
    "🎯 Competências (separadas por vírgula)",
    placeholder="python, comunicação, gestão..."
)

empresa_filtrar = st.sidebar.text_input(
    "🏢 Empresa específica",
    placeholder="Nome da empresa"
)

st.sidebar.markdown("---")
st.sidebar.markdown("Plataforma: https://revela-talentos-para-todos.lovable.app")

# -------------------------------------------------------------
# CARREGAR DATASETS
# -------------------------------------------------------------
if uploaded_candidates:
    df_cand = pd.read_csv(uploaded_candidates)
else:
    # EXEMPLO (estrutura padrão)
    df_cand = pd.DataFrame({
        "nome": ["Ana", "Bruno", "Carlos", "Daniela", "Eva"] * 10,
        "idade": np.random.randint(18, 55, 50),
        "senioridade": np.random.choice(["Júnior", "Pleno", "Sênior"], 50),
        "competencias": np.random.choice([
            "python;sql;flask",
            "comunicação;vendas;crm",
            "javascript;react;css",
            "gestão;liderança;agilidade"
        ], 50),
        "pretensao": np.random.randint(2000, 12000, 50)
    })

if uploaded_companies:
    df_emp = pd.read_csv(uploaded_companies)
else:
    df_emp = pd.DataFrame({
        "empresa": ["TechX", "ComercialSul", "FinData", "HealthPOA"] * 10,
        "vaga": ["Dev Python", "Inside Sales", "Front-end", "Scrum Master"] * 10,
        "competencias_desejadas": [
            "python;sql;flask",
            "comunicação;vendas;crm",
            "javascript;react;css",
            "gestão;liderança;agilidade"
        ] * 10,
        "salario": np.random.randint(2500, 15000, 40)
    })

# -------------------------------------------------------------
# APLICAR FILTROS
# -------------------------------------------------------------
df_filtrado = df_cand.copy()

if senioridade:
    df_filtrado = df_filtrado[df_filtrado["senioridade"].isin(senioridade)]

if competencias_filtrar:
    comps = [c.strip().lower() for c in competencias_filtrar.split(",")]
    df_filtrado = df_filtrado[
        df_filtrado["competencias"].str.lower().apply(
            lambda x: all(c in x for c in comps)
        )
    ]

if empresa_filtrar:
    df_emp = df_emp[df_emp["empresa"].str.contains(empresa_filtrar, case=False)]


# -------------------------------------------------------------
# TABS DO DASHBOARD
# -------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📊 Visão Geral",
    "👤 Análise de Candidatos",
    "🤝 Matching Candidato ↔ Vaga"
])

# -------------------------------------------------------------
# TAB 1 – VISÃO GERAL
# -------------------------------------------------------------
with tab1:
    st.header("📊 Visão Geral da Plataforma")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total de Candidatos", len(df_cand))
        st.metric("Total de Empresas/Vagas", len(df_emp))

    with col2:
        st.markdown("### Distribuição de Senioridade")
        fig_sen = px.histogram(df_cand, x="senioridade", color="senioridade")
        st.plotly_chart(fig_sen, use_container_width=True)

    st.markdown("### Competências mais frequentes")
    df_skills = (
        df_cand["competencias"]
        .str.split(";")
        .explode()
        .value_counts()
        .reset_index()
    )
    df_skills.columns = ["Skill", "Frequência"]

    fig_skills = px.bar(df_skills.head(15), x="Skill", y="Frequência")
    st.plotly_chart(fig_skills, use_container_width=True)

# -------------------------------------------------------------
# TAB 2 – ANÁLISE DETALHADA DE CANDIDATOS
# -------------------------------------------------------------
with tab2:
    st.header("👤 Análise de Candidatos")

    st.dataframe(df_filtrado, use_container_width=True)

    st.markdown("### Faixa salarial dos candidatos")
    fig_sal = px.box(df_filtrado, y="pretensao", points="all")
    st.plotly_chart(fig_sal, use_container_width=True)

# -------------------------------------------------------------
# TAB 3 – MATCHING AUTOMÁTICO
# -------------------------------------------------------------
with tab3:
    st.header("🤝 Matching Automático de Competências")

    # TF-IDF para calcular similaridade entre textos (competências)
    vectorizer = TfidfVectorizer()

    cand_text = df_filtrado["competencias"].fillna("")
    vaga_text = df_emp["competencias_desejadas"].fillna("")

    try:
        tfidf_matrix = vectorizer.fit_transform(pd.concat([cand_text, vaga_text]))
        candidatos_vec = tfidf_matrix[:len(cand_text)]
        vagas_vec = tfidf_matrix[len(cand_text):]

        sim_matrix = cosine_similarity(candidatos_vec, vagas_vec)

        match_results = []

        for i, cand in df_filtrado.iterrows():
            best_idx = np.argmax(sim_matrix[i])
            best_vaga = df_emp.iloc[best_idx]
            score = sim_matrix[i][best_idx]

            match_results.append({
                "Candidato": cand["nome"],
                "Senioridade": cand["senioridade"],
                "Competências": cand["competencias"],
                "Melhor Vaga": best_vaga["vaga"],
                "Empresa": best_vaga["empresa"],
                "Score de Match": round(float(score), 3)
            })

        df_match = pd.DataFrame(match_results)

        st.subheader("📌 Resultado dos Melhores Matches")
        st.dataframe(df_match.sort_values("Score de Match", ascending=False),
                     use_container_width=True)

    except Exception as e:
        st.error(f"Erro ao gerar matching: {e}")
