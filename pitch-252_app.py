# ===============================================================
#  REPARA ANALYTICS — V13.4.2
#  Streamlit + Gemini + Auth + Admin + Wordcloud Inteligente
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

# -------------------- AUTH --------------------
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

def verify_password(plain, hashed):
    try:
        return pwd_context.verify(plain, hashed)
    except Exception:
        return False


# -------------------- CONFIG AI --------------------
if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    model = genai.GenerativeModel("gemini-1.5-flash")
else:
    model = None


# =====================================================
#  SENTIMENT LEXICON PT-BR
# =====================================================
SENTIMENT_POS = {
    "bom","ótimo","excelente","maravilhoso","feliz","satisfeito",
    "positivo","positiva","tranquilo","gostei","amo","adoro",
    "eficiente","competente","justo","melhor","acolhedor"
}

SENTIMENT_NEG = {
    "ruim","péssimo","triste","demorado","negativo","negativa",
    "difícil","injusto","perigoso","preocupado","ansioso",
    "fraco","insuportável","horrível","problema","desafio"
}


# =====================================================
#  TOKEN CLEANING + POS CLASSIFIER (light)
# =====================================================
STOP_PT = set(stopwords.words("portuguese"))
CUSTOM_STOP = {
    "sim","não","nao","ok","bom","boa","coisa","coisas","dia",
    "gente","pessoa","pessoas","empresa","empresas","acho",
    "acredito","ser","estar","ter","fazer","feito","vou","vai",
    "já","ja","pois","sobre","ainda","muito","pouco","tudo"
}
ALL_STOP = STOP_PT.union(CUSTOM_STOP)

_word_pattern = re.compile(r"[A-Za-zÀ-ÿ]+")

def clean_token(t):
    return re.sub(r"[^A-Za-zÀ-ÿ]", "", t.lower()).strip()

def is_number(word):
    return bool(re.fullmatch(r"\d+|\d+\.\d+", word))

def classify_token(word):
    if word.endswith(("ar","er","ir")):
        return "VERB"
    if word.endswith(("dade","ção","são","mento","idade","tude")):
        return "NOUN"
    if word.endswith(("vel","iva","ivo","oso","osa","ante","ente")):
        return "ADJ"
    return "OTHER"

def simple_lemmatize(word):
    lemmas = {
        "trabalhando":"trabalhar","estudando":"estudar",
        "melhorando":"melhorar","precisando":"precisar",
        "buscando":"buscar","aprendendo":"aprender",
        "vivendo":"viver","tentando":"tentar"
    }
    if word in lemmas:
        return lemmas[word]

    for suf in ("ando","endo","indo"):
        if word.endswith(suf):
            return word[:-4]
    return word


# =====================================================
#  WORDCLOUD INTELIGENTE
# =====================================================
def sentiment_weight(t):
    if t in SENTIMENT_POS:
        return 4
    if t in SENTIMENT_NEG:
        return 4
    return 0

def extract_relevant_words_from_text(text):
    if not isinstance(text, str):
        return []

    raw = _word_pattern.findall(text)
    cleaned = []

    for tk in raw:
        tk2 = clean_token(tk)
        if not tk2 or tk2 in ALL_STOP or is_number(tk2):
            continue
        if len(tk2) <= 2:
            continue
        cleaned.append(tk2)

    weighted_list = []
    for tk in cleaned:
        cls = classify_token(tk)
        lemma = simple_lemmatize(tk)
        
        pos_weight = {
            "VERB": 3,
            "ADJ": 2,
            "NOUN": 2,
            "OTHER": 0.5
        }.get(cls,1)

        e_weight = sentiment_weight(lemma)
        total = pos_weight + e_weight
        weighted_list.append((lemma, total))

    freq = {}
    for lemma, w in weighted_list:
        freq[lemma] = freq.get(lemma, 0) + w

    final_words = []
    for lemma, score in freq.items():
        final_words.extend([lemma] * max(1, int(score)))

    return final_words


def generate_wordcloud(words, theme="dark"):
    text = " ".join(words)

    themes = {
        "Dark Elegante": ("black", "viridis"),
        "Deep Purple": ("#0d001f", "plasma"),
        "Neon Blue": ("#00111e", "cool"),
        "Gold": ("black", "cividis"),
        "Carbon Gray": ("#1a1a1a", "magma")
    }

    bg, cmap = themes.get(theme, ("black", "viridis"))

    wc = WordCloud(
        width=1000,
        height=500,
        background_color=bg,
        colormap=cmap,
        collocations=False,
        max_words=150
    ).generate(text)

    fig, ax = plt.subplots(figsize=(12,6))
    ax.imshow(wc)
    ax.axis("off")
    return fig


# =====================================================
#  CSV LOADER
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
#  AI FUNCTIONS
# =====================================================
def ai_analyze(df, column_name):
    text = "\n".join(df[column_name].dropna().astype(str).tolist()[:200])

    prompt = f"""
    Analise profundamente as respostas abaixo (português).
    Identifique:
    - temas principais
    - sentimentos
    - oportunidades
    - ações recomendadas
    - resumo final

    Texto analisado:
    {text}
    """

    response = model.generate_content(prompt)
    return response.text


# =====================================================
#  PDF Export
# =====================================================
def create_pdf(text):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph(text.replace("\n","<br/>"), styles["BodyText"])]
    doc.build(story)
    buf.seek(0)
    return buf


# =====================================================
#  LOGIN
# =====================================================
def login_dialog():
    with st.dialog("Login"):
        email = st.text_input("Email")
        password = st.text_input("Senha", type="password")

        if st.button("Entrar"):
            for user, data in st.secrets["users"].items():
                if data["email"] == email:
                    if verify_password(password, data["password"]):
                        st.session_state.logged = True
                        st.session_state.user = data["email"]
                        st.rerun()
                    else:
                        st.error("Senha incorreta.")
                        return
            st.error("Usuário não encontrado.")


# =====================================================
#  ADMIN PANEL
# =====================================================
def admin_panel():
    st.header("Painel Admin")

    new_email = st.text_input("Email do novo usuário")
    new_name = st.text_input("Nome")
    new_pass = st.text_input("Senha")
    
    if st.button("Gerar Hash"):
        hashv = pwd_context.hash(new_pass)
        st.code(f"""
[users.{new_email.split('@')[0]}]
name = "{new_name}"
email = "{new_email}"
password = "{hashv}"
""")

    st.info("Cole o bloco acima no secrets.toml")


# =====================================================
#  MAIN APP
# =====================================================
def main_app():

    st.title("📊 REPARA Analytics v13.4.2")

    menu = st.sidebar.radio(
        "Menu",
        ["📥 Candidatos", "🏢 Empresas", "🔀 Cruzada", "🤖 Chat IA", "🛠 Admin"]
    )

    if menu == "🛠 Admin":
        if st.session_state.user != "admin@repara.com":
            st.error("Acesso restrito.")
            return
        admin_panel()
        return

    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    if not uploaded:
        st.info("Envie um arquivo CSV para começar.")
        return

    df = load_csv(uploaded)
    st.subheader("Preview")
    st.dataframe(df.head())

    # detectar colunas textuais
    cols_text = [
        c for c in df.columns
        if df[c].dtype == object or df[c].astype(str).str.contains(r"[A-Za-zÀ-ÿ]").mean() > 0.2
    ]

    if menu in ["📥 Candidatos", "🏢 Empresas"]:

        st.subheader("Selecione uma coluna textual:")
        col_sel = st.selectbox("Coluna", cols_text)

        st.subheader("Tema do Wordcloud")
        theme = st.selectbox("Tema", ["Dark Elegante","Deep Purple","Neon Blue","Gold","Carbon Gray"])

        st.subheader("Wordcloud Inteligente")
        all_text = " ".join(df[col_sel].dropna().astype(str).tolist())
        
        words = extract_relevant_words_from_text(all_text)
        fig = generate_wordcloud(words, theme=theme)
        st.pyplot(fig)

        if st.button("Análise com IA"):
            result = ai_analyze(df, col_sel)
            st.subheader("💡 Insight IA")
            st.write(result)

            pdf_data = create_pdf(result)
            st.download_button("Baixar PDF", pdf_data, "analise.pdf")

    elif menu == "🔀 Cruzada":
        st.subheader("Escolha as colunas de comparação")

        colA = st.selectbox("Coluna Candidatos", cols_text)
        colB = st.selectbox("Coluna Empresas", cols_text)

        if st.button("Análise Cruzada IA"):
            tA = "\n".join(df[colA].dropna().astype(str).tolist()[:100])
            tB = "\n".join(df[colB].dropna().astype(str).tolist()[:100])

            prompt = f"""
Compare temas, sentimentos e oportunidades entre:

Candidatos:
{tA}

Empresas:
{tB}
"""
            result = model.generate_content(prompt).text
            st.write(result)

    elif menu == "🤖 Chat IA":
        st.subheader("Chat sobre os dados")
        user_msg = st.text_area("Pergunte algo:")

        if st.button("Enviar"):
            preview = df.head().to_csv(index=False)
            prompt = f"""
Você é um analista de dados.
Use este preview para responder:

{preview}

Pergunta: {user_msg}
"""
            st.write(model.generate_content(prompt).text)


# =====================================================
#  BOOTSTRAP
# =====================================================
if "logged" not in st.session_state:
    login_dialog()
else:
    main_app()
