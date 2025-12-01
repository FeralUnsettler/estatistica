import io
import os
import tempfile
from typing import Optional

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# Geocoding & mapping
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import pydeck as pdk

# Semantics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
try:
    from sentence_transformers import SentenceTransformer
    HAS_ST_MODEL = True
except Exception:
    HAS_ST_MODEL = False

# PDF / PPTX exports
from fpdf import FPDF
from pptx import Presentation
from pptx.util import Inches, Pt

# ------------------------------------------------------------------
# URLs dos Google Sheets (do usuário)
CANDIDATES_SHEET_CSV = "https://docs.google.com/spreadsheets/d/19HjusGjrEw7vAulmaI1bEpthzdeil1Zl3hjtGbFNWOM/export?format=csv&gid=1409532599"
COMPANIES_SHEET_CSV = "https://docs.google.com/spreadsheets/d/1VuEH3rQkdtgCS4MmdwVi_9wsoPA8i4BgYnCOEzbDlBA/export?format=csv&gid=1605068997"
# ------------------------------------------------------------------

st.set_page_config(page_title="Revela Talentos - Analytics", layout="wide", page_icon="🧭")

st.title("🔎 Revela Talentos — Analytics (Candidatos & Empresas)")
st.markdown("Dashboard com geolocalização, matching semântico, PDF/PPTX export e mapas interativos.")

# Sidebar controls
st.sidebar.header("Configurações e fontes de dados")

use_remote = st.sidebar.checkbox("Tentar carregar os Google Sheets públicos automaticamente", value=True)
uploaded_cand = st.sidebar.file_uploader("Upload CSV - Candidatos (se necessário)", type=["csv"])
uploaded_comp = st.sidebar.file_uploader("Upload CSV - Empresas/Vagas (se necessário)", type=["csv"])

# Semantic search options
st.sidebar.markdown("---")
use_sentence_transformers = st.sidebar.checkbox("Usar Sentence-Transformers (melhor semântica)", value=HAS_ST_MODEL and True)
st.sidebar.caption("Se não disponível, o app usará TF-IDF como fallback.")

# Map options
st.sidebar.markdown("---")
map_provider = st.sidebar.selectbox("Visualização de mapa", ["pydeck (WebGL)", "Plotly (scatter_mapbox)"])
st.sidebar.caption("pydeck é responsivo e rápido para muitos pontos.")

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def try_read_csv_from_url(url: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        return None

def load_data():
    # 1) candidatos
    df_cand = None
    if use_remote:
        df_cand = try_read_csv_from_url(CANDIDATES_SHEET_CSV)
        if df_cand is None:
            st.sidebar.warning("Não foi possível ler a planilha de candidatos automaticamente (pode exigir permissão). Faça upload manual se desejar.")
    if uploaded_cand is not None:
        df_cand = pd.read_csv(uploaded_cand)
    if df_cand is None:
        # fallback: exemplo mínimo
        df_cand = pd.DataFrame({
            "nome": ["Ana", "Bruno", "Carlos"],
            "email": ["a@x.com", "b@x.com", "c@x.com"],
            "cidade": ["Porto Alegre", "Porto Alegre", "Canoas"],
            "endereco": ["Av. Ipiranga, 1000", "Rua XV, 200", "Av. Sapucaia, 50"],
            "competencias": ["python;sql", "vendas;crm", "javascript;react"],
            "senioridade": ["Pleno", "Júnior", "Pleno"],
            "pretensao": [4500, 2500, 6000],
        })
        st.sidebar.info("Usando base de candidatos de exemplo (nenhum dado real carregado).")
    # 2) empresas
    df_comp = None
    if use_remote:
        df_comp = try_read_csv_from_url(COMPANIES_SHEET_CSV)
        if df_comp is None:
            st.sidebar.warning("Não foi possível ler a planilha de empresas automaticamente (pode exigir permissão). Faça upload manual se desejar.")
    if uploaded_comp is not None:
        df_comp = pd.read_csv(uploaded_comp)
    if df_comp is None:
        df_comp = pd.DataFrame({
            "empresa": ["TechX", "ComercialSul"],
            "vaga": ["Dev Python", "Inside Sales"],
            "endereco": ["Rua da Indústria, 200", "Av. Sete, 400"],
            "competencias_desejadas": ["python;sql", "vendas;crm"],
            "cidade": ["Porto Alegre", "Caxias do Sul"],
            "salario_oferecido": [7000, 3200]
        })
        st.sidebar.info("Usando base de empresas de exemplo (nenhum dado real carregado).")

    return df_cand, df_comp

# Geocoding with caching
geolocator = Nominatim(user_agent="revela_talentos_app")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, max_retries=2, error_wait_seconds=2.0)

@st.cache_data(show_spinner=False)
def geocode_dataframe(df: pd.DataFrame, address_col: str, city_col: Optional[str] = None, lat_name="lat", lon_name="lon"):
    # If df already has lat/lon columns, keep them
    if lat_name in df.columns and lon_name in df.columns:
        return df
    lats, lons = [], []
    for _, row in df.iterrows():
        addr = str(row.get(address_col, "")) if address_col in df.columns else ""
        if city_col and city_col in df.columns:
            addr = f"{addr}, {row.get(city_col, '')}"
        if not addr or addr.strip() == "":
            lats.append(np.nan); lons.append(np.nan); continue
        try:
            loc = geocode(addr)
            if loc:
                lats.append(loc.latitude); lons.append(loc.longitude)
            else:
                lats.append(np.nan); lons.append(np.nan)
        except Exception:
            lats.append(np.nan); lons.append(np.nan)
    df = df.copy()
    df[lat_name] = lats
    df[lon_name] = lons
    return df

# Semantic encoding
@st.cache_data(show_spinner=False)
def encode_texts(texts, use_st=True):
    texts = ["" if pd.isna(t) else str(t) for t in texts]
    if use_st and HAS_ST_MODEL:
        model = SentenceTransformer("all-MiniLM-L6-v2")  # rápido e leve
        emb = model.encode(texts, show_progress_bar=False)
        return np.array(emb)
    else:
        # TF-IDF -> dense vector (approx)
        tf = TfidfVectorizer(max_features=1000)
        mat = tf.fit_transform(texts).toarray()
        return mat

# Export helpers
def create_pdf_summary(df_cand, df_comp, top_matches=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, "Relat\u00f3rio - Revela Talentos", ln=1)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 6, f"Candidatos: {len(df_cand)}", ln=1)
    pdf.cell(0, 6, f"Empresas/Vagas: {len(df_comp)}", ln=1)
    pdf.ln(4)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 6, "Top matches (exemplo):", ln=1)
    pdf.set_font("Arial", "", 10)
    if top_matches is not None:
        for i, r in top_matches.head(10).iterrows():
            pdf.multi_cell(0, 6, f"{r.get('Candidato','')} -> {r.get('Melhor Vaga','')} (score: {r.get('Score de Match', '')})")
    return pdf.output(dest="S").encode("latin-1", errors="replace")

def create_pptx_summary(df_cand, df_comp, top_matches=None):
    prs = Presentation()
    title_slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(title_slide_layout)
    slide.shapes.title.text = "Relat\u00f3rio - Revela Talentos"
    slide.placeholders[1].text = f"Candidatos: {len(df_cand)}\nEmpresas: {len(df_comp)}"

    # Slide matches
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    title = slide.shapes.title
    title.text = "Top matches"
    left = Inches(0.5); top = Inches(1.2)
    tx = slide.shapes.add_textbox(left, top, Inches(9), Inches(5))
    tf = tx.text_frame
    if top_matches is not None:
        for i, r in top_matches.head(10).iterrows():
            p = tf.add_paragraph()
            p.text = f"{r.get('Candidato','')} -> {r.get('Melhor Vaga','')} (score: {r.get('Score de Match','')})"
            p.font.size = Pt(12)
    # return bytes
    bio = io.BytesIO()
    prs.save(bio)
    bio.seek(0)
    return bio.read()

# ------------------------------------------------------------------
# Load data
df_cand, df_comp = load_data()

# Basic cleaning: normalize skill columns names if exist
for col in df_cand.columns:
    if col.lower() in ["competencias", "skills", "competencies"]:
        df_cand.rename(columns={col: "competencias"}, inplace=True)
for col in df_comp.columns:
    if col.lower() in ["competencias_desejadas", "skills_required", "competenciasdesejadas"]:
        df_comp.rename(columns={col: col,}, inplace=True)

# Sidebar quick filters
st.sidebar.markdown("---")
st.sidebar.header("Filtros Rápidos")
sen_select = st.sidebar.multiselect("Senioridade", options=sorted(df_cand.get("senioridade", pd.Series([])).dropna().unique()), default=[])
min_salary = st.sidebar.number_input("Pretensão salarial mínima (R$)", min_value=0, value=0, step=500)
skill_filter = st.sidebar.text_input("Filtrar por competência (ex: python)", "")

# Apply filters
df_cand_filtered = df_cand.copy()
if sen_select:
    df_cand_filtered = df_cand_filtered[df_cand_filtered["senioridade"].isin(sen_select)]
if min_salary > 0 and "pretensao" in df_cand_filtered.columns:
    df_cand_filtered = df_cand_filtered[df_cand_filtered["pretensao"] >= min_salary]
if skill_filter:
    df_cand_filtered = df_cand_filtered[df_cand_filtered["competencias"].str.contains(skill_filter, case=False, na=False)]

# ------------------------------------------------------------------
# Tabs
tab_overview, tab_candidates, tab_companies, tab_matching, tab_export = st.tabs([
    "📈 Visão Geral", "👥 Candidatos", "🏢 Empresas/Vagas", "🤝 Matching Semântico", "⬇️ Exportar"
])

with tab_overview:
    st.header("Visão Geral")
    c1, c2, c3 = st.columns(3)
    c1.metric("Candidatos (total)", len(df_cand))
    c2.metric("Empresas/Vagas (total)", len(df_comp))
    if "pretensao" in df_cand.columns:
        c3.metric("Pretensão média", f"R$ {df_cand['pretensao'].dropna().mean():,.0f}")

    st.markdown("### Competências mais frequentes (candidatos)")
    if "competencias" in df_cand.columns:
        s = df_cand["competencias"].dropna().str.split("[;,]").explode().str.strip().str.lower()
        freq = s.value_counts().reset_index().rename(columns={"index":"skill", "competencias":"count"})
        fig = px.bar(freq.head(20), x="skill", y=0, labels={"0":"count"})
        st.plotly_chart(fig, use_container_width=True)

with tab_candidates:
    st.header("Análise de Candidatos")
    st.dataframe(df_cand_filtered, use_container_width=True)

    st.markdown("#### Mapa de Candidatos (geolocalização)")
    # Geocode candidates - try 'endereco' and 'cidade' columns if present
    address_col = "endereco" if "endereco" in df_cand_filtered.columns else None
    city_col = "cidade" if "cidade" in df_cand_filtered.columns else None

    df_cand_geo = geocode_dataframe(df_cand_filtered, address_col or "nome", city_col)
    # Drop rows without lat/lon
    df_map = df_cand_geo.dropna(subset=["lat","lon"])
    if not df_map.empty:
        if map_provider == "pydeck (WebGL)":
            initial_view = pdk.ViewState(latitude=df_map["lat"].mean(), longitude=df_map["lon"].mean(), zoom=10)
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=df_map,
                get_position='[lon, lat]',
                get_radius=200,
                get_fill_color=[66,135,245,160],
                pickable=True
            )
            r = pdk.Deck(layers=[layer], initial_view_state=initial_view, tooltip={"text":"{nome}\n{competencias}"})
            st.pydeck_chart(r)
        else:
            fig = px.scatter_mapbox(df_map, lat="lat", lon="lon", hover_name="nome", hover_data=["competencias"], zoom=10, height=500)
            fig.update_layout(mapbox_style="open-street-map")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Nenhum candidato geocodificado (sem endereço/lat-lon válido).")

with tab_companies:
    st.header("Empresas / Vagas")
    st.dataframe(df_comp, use_container_width=True)

    st.markdown("#### Mapa de Empresas")
    address_col_c = "endereco" if "endereco" in df_comp.columns else None
    city_col_c = "cidade" if "cidade" in df_comp.columns else None
    df_comp_geo = geocode_dataframe(df_comp, address_col_c or "empresa", city_col_c)
    df_comp_map = df_comp_geo.dropna(subset=["lat","lon"])
    if not df_comp_map.empty:
        if map_provider == "pydeck (WebGL)":
            initial_view = pdk.ViewState(latitude=df_comp_map["lat"].mean(), longitude=df_comp_map["lon"].mean(), zoom=10)
            layer = pdk.Layer("ScatterplotLayer", data=df_comp_map, get_position='[lon, lat]', get_radius=300, get_fill_color=[245,85,66,160], pickable=True)
            st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=initial_view, tooltip={"text":"{empresa}\n{vaga}"}))
        else:
            fig = px.scatter_mapbox(df_comp_map, lat="lat", lon="lon", hover_name="empresa", hover_data=["vaga"], zoom=10, height=500)
            fig.update_layout(mapbox_style="open-street-map")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Nenhuma empresa geocodificada (sem endereço/lat-lon válido).")

with tab_matching:
    st.header("Matching Semântico (Competências)")
    st.markdown("Escolha a fonte de codificação semântica e clique em 'Gerar Matching'.")

    use_st = st.checkbox("Forçar uso de Sentence-Transformers (se disponível)", value=use_sentence_transformers)
    text_cand = df_cand_filtered["competencias"].fillna("").astype(str)
    text_vagas = df_comp["competencias_desejadas"].fillna(df_comp.get("competencias", "")).astype(str)

    if st.button("Gerar Matching"):
        # Encode texts
        enc_cand = encode_texts(text_cand.tolist(), use_st and HAS_ST_MODEL)
        enc_vagas = encode_texts(text_vagas.tolist(), use_st and HAS_ST_MODEL)

        # Similarity
        sim = cosine_similarity(enc_cand, enc_vagas)
        matches = []
        for i, idx in enumerate(range(sim.shape[0])):
            best = sim[i].argmax()
            matches.append({
                "Candidato": df_cand_filtered.iloc[i].get("nome", f"cand_{i}"),
                "Melhor Vaga": df_comp.iloc[best].get("vaga", ""),
                "Empresa": df_comp.iloc[best].get("empresa", ""),
                "Score de Match": float(sim[i, best])
            })
        df_matches = pd.DataFrame(matches).sort_values("Score de Match", ascending=False)
        st.dataframe(df_matches, use_container_width=True)
    else:
        st.info("Clique em 'Gerar Matching' para executar a busca semântica.")

with tab_export:
    st.header("Exportar Relatórios")
    st.markdown("Gere PDF e PPTX sumarizando a análise e os top matches.")

    # if matches exist in session, use them; else allow generating quick matches
    if st.button("Gerar relatório (PDF e PPTX) com matches rápidos"):
        # quick matching (TF-IDF)
        tc = df_cand_filtered["competencias"].fillna("").astype(str).tolist()
        tv = df_comp["competencias_desejadas"].fillna("").astype(str).tolist()
        enc_c = encode_texts(tc, use_st=False)
        enc_v = encode_texts(tv, use_st=False)
        sim = cosine_similarity(enc_c, enc_v)
        matches = []
        for i in range(sim.shape[0]):
            best = sim[i].argmax()
            matches.append({
                "Candidato": df_cand_filtered.iloc[i].get("nome",""),
                "Melhor Vaga": df_comp.iloc[best].get("vaga",""),
                "Empresa": df_comp.iloc[best].get("empresa",""),
                "Score de Match": float(sim[i, best])
            })
        df_matches = pd.DataFrame(matches).sort_values("Score de Match", ascending=False)

        pdf_bytes = create_pdf_summary(df_cand_filtered, df_comp, top_matches=df_matches)
        pptx_bytes = create_pptx_summary(df_cand_filtered, df_comp, top_matches=df_matches)

        st.download_button("📄 Baixar PDF", data=pdf_bytes, file_name="relatorio_revela.pdf", mime="application/pdf")
        st.download_button("📽️ Baixar PPTX", data=pptx_bytes, file_name="relatorio_revela.pptx", mime="application/vnd.openxmlformats-officedocument.presentationml.presentation")

# ------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.caption("Desenvolvido para Luciano / Revela Talentos — ajuste conforme suas colunas reais.")
