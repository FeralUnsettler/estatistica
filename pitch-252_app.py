# ===============================================================
#  REPARA ANALYTICS — V13.4.4 SM (Super Minimal)
#  100% compatível com Streamlit Cloud
# ===============================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import io
from passlib.context import CryptContext
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
import google.generativeai as genai
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)


# =====================================================
# PASSWORD HASHING
# =====================================================
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

def verify_password(plain, hashed):
    try:
        return pwd_context.verify(plain, hashed)
    except:
        return False


# =====================================================
# GOOGLE GEMINI CONFIG
# =====================================================
if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    model = genai.GenerativeModel("gemini-1.5-flash")
else:
    model = None


# =====================================================
# SENTIMENT LEXICON
# =====================================================
POSITIVE = {
    "bom","ótimo","excelente","feliz","positivo","organizado",
    "eficiente","tranquilo","justo","acolhedor"
}

NEGATIVE = {
    "ruim","péssimo","negativo","difícil","injusto","horrível",
    "ansioso","demorado","desafio","problema"
}


# =====================================================
# WORD PROCESSING
# =====================================================
STOP_PT = set(stopwords.words("portuguese"))
CUSTOM_STOP = {
    "sim","não","nao","ok","bom","boa","coisa","coisas","dia",
    "gente","pessoa","pessoas","empresa","empresas","acho",
    "ser","estar","ter","fazer","vou","vai","já","ja","pois"
}
STOP_ALL = STOP_PT.union(CUSTOM_STOP)

pattern = re.compile(r"[A-Za-zÀ-ÿ]+")

def clean_token(t):
    return re.sub(r"[^A-Za-zÀ-ÿ]", "", t.lower()).strip()

def classify(word):
    if word.endswith(("ar","er","ir")):
        return "VERB"
    if word.endswith(("dade","ção","são","mento","agem")):
        return "NOUN"
    if word.endswith(("vel","iva","ivo","oso","osa")):
        return "ADJ"
    return "OTHER"

def weight_sentiment(w):
    if w in POSITIVE: return 4
    if w in NEGATIVE: return 4
    return 0

def extract_words(text):
    if not isinstance(text, str):
        return []
    raw = pattern.findall(text)
    words = []

    for t in raw:
        t2 = clean_token(t)
        if len(t2) <= 2: continue
        if t2 in STOP_ALL: continue
        words.append(t2)

    weighted = {}
    for w in words:
        base_weight = {
            "VERB":3,
            "ADJ":2,
            "NOUN":2,
            "OTHER":1
        }.get(classify(w),1)

        total = base_weight + weight_sentiment(w)
        weighted[w] = weighted.get(w,0) + total

    final = []
    for w,score in weighted.items():
        final.extend([w]*max(1,int(score)))

    return final


def generate_wordcloud(words, theme="Dark"):
    THEMES = {
        "Dark": ("black","viridis"),
        "Deep Purple": ("#0d001f","plasma"),
        "Neon Blue": ("#00111e","cool"),
        "Gold": ("black","cividis"),
        "Carbon": ("#1a1a1a","magma")
    }

    bg, cmap = THEMES.get(theme, ("black","viridis"))

    wc = WordCloud(
        width=900, height=450,
        background_color=bg,
        colormap=cmap,
        collocations=False,
        max_words=150
    ).generate(" ".join(words))

    fig, ax = plt.subplots(figsize=(12,6))
    ax.imshow(wc)
    ax.axis("off")
    return fig


# =====================================================
# LOAD CSV ROBUSTO
# =====================================================
def load_csv(file):
    try:
        return pd.read_csv(file, sep=None, engine="python")
    except:
        try:
            return pd.read_csv(file, sep=";")
        except:
            return pd.read_csv(file, sep=",")


# =====================================================
# AI ANALYSIS
# =====================================================
def ai_analyze(df, col):
    text = "\n".join(df[col].dropna().astype(str))

    prompt = f"""
Analise profundamente as respostas a seguir.
Identifique temas, sentimentos e oportunidades.

Texto:
{text}
"""

    return model.generate_content(prompt).text


# =====================================================
# PDF
# =====================================================
def make_pdf(text):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph(text.replace("\n","<br/>"), styles["BodyText"])]
    doc.build(story)
    buf.seek(0)
    return buf


# =====================================================
# LOGIN MODAL
# =====================================================
def open_login_modal():
    with st.modal("Login"):
        email = st.text_input("Email")
        password = st.text_input("Senha", type="password")

        if st.button("Entrar"):
            for u, data in st.secrets["users"].items():
                if email == data["email"]:
                    if verify_password(password, data["password"]):
                        st.session_state.logged = True
                        st.session_state.user = email
                        st.session_state._rerun = True
                        return
                    else:
                        st.error("Senha incorreta")
                        return
            st.error("Usuário não encontrado")


def open_recovery_modal():
    with st.modal("Recuperar Senha"):
        st.info("Contate o administrador para redefinir sua senha.")


# =====================================================
# ADMIN PANEL
# =====================================================
def admin_panel():
    st.header("Painel Admin")

    new_email = st.text_input("Novo email")
    new_name  = st.text_input("Nome")
    new_pass  = st.text_input("Senha")

    if st.button("Gerar Hash"):
        h = pwd_context.hash(new_pass)
        st.code(f"""
[users.{new_email.split('@')[0]}]
name = "{new_name}"
email = "{new_email}"
password = "{h}"
""")


# =====================================================
# MAIN APP
# =====================================================
def main_app():

    st.title("📊 REPARA Analytics — 13.4.4 SM")

    menu = st.sidebar.radio(
        "Menu",
        ["📥 Candidatos", "🏢 Empresas", "🤖 IA Chat", "🛠 Admin"]
    )

    uploaded = st.sidebar.file_uploader("Envie seu CSV", type="csv")

    if menu == "🛠 Admin":
        if st.session_state.user != "admin@repara.com":
            st.error("Acesso restrito")
            return
        admin_panel()
        return

    if not uploaded:
        st.info("Envie um CSV para começar.")
        return

    df = load_csv(uploaded)
    st.subheader("Preview dos Dados")
    st.dataframe(df.head())

    text_cols = [
        c for c in df.columns
        if df[c].astype(str).str.contains(r"[A-Za-zÀ-ÿ]").mean() > 0.15
    ]

    if not text_cols:
        st.warning("Nenhuma coluna textual encontrada.")
        return

    col = st.selectbox("Escolha a coluna", text_cols)

    st.subheader("Tema do Wordcloud")
    theme = st.selectbox("Tema", ["Dark","Deep Purple","Neon Blue","Gold","Carbon","Carbon Gray"])

    text_all = " ".join(df[col].dropna().astype(str))
    words = extract_words(text_all)
    fig = generate_wordcloud(words, theme)
    st.pyplot(fig)

    if st.button("Análise com IA"):
        result = ai_analyze(df, col)
        st.write(result)

        pdf = make_pdf(result)
        st.download_button("Baixar PDF", pdf, "analise.pdf")


# =====================================================
# APP FLOW
# =====================================================
if "logged" not in st.session_state:
    st.session_state.logged = False
    st.session_state.show_recovery = False

if not st.session_state.logged:
    st.button("Entrar", on_click=open_login_modal)
    st.button("Esqueci a senha", on_click=lambda: st.session_state.__setitem__("show_recovery", True))

    if st.session_state.show_recovery:
        open_recovery_modal()

else:
    main_app()


# =====================================================
# RERUN HANDLER
# =====================================================
if st.session_state.get("_rerun"):
    st.session_state._rerun = False
    st.rerun()
