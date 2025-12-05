import streamlit as st
import pandas as pd
import numpy as np
import base64
import io
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from passlib.context import CryptContext
import google.generativeai as genai
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import chardet

# ------------------------
# CONFIGURAÇÃO INICIAL
# ------------------------
st.set_page_config(
    page_title="MindVision Analytics - REPARA",
    layout="wide",
)

pwd_context = CryptContext(schemes=["pbkdf2_sha256"])


# ------------------------
# Função: Carregar Logo via Base64 do secrets
# ------------------------
def load_logo_from_base64():
    try:
        logo_b64 = st.secrets.get("logo_base64", None)
        if not logo_b64:
            return None
        bytes_data = base64.b64decode(logo_b64)
        return io.BytesIO(bytes_data)
    except:
        return None


# ------------------------
# Função: Login Check
# ------------------------
def verify_user(email, password):
    users = st.secrets.get("users", {})
    for _, data in users.items():
        if data.get("email") == email:
            hashed = data.get("password")
            if pwd_context.verify(password, hashed):
                return data.get("name")
    return None


# ------------------------
# Mini Lexicon emocional
# ------------------------
emotion_words = {
    "excelente": 2, "ótimo": 2, "bom": 1, "positivo": 1,
    "feliz": 2, "alegre": 2, "motivador": 2, "engajado": 2,
    "ruim": -1, "péssimo": -2, "triste": -2, "negativo": -1,
    "ansioso": -1, "estressado": -1
}


# ------------------------
# Limpeza e Wordcloud
# ------------------------
def preprocess_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    stopwords = set([
        "de","da","do","das","dos","a","o","e","é","que","com","em","um",
        "uma","por","para","se","na","no","nas","nos","..."  # (curtos irrelevantes)
    ])

    tokens = []
    for w in text.split():
        w = w.strip(".,!?;:()[]{}\"'")
        if not w:
            continue

        # Se for uma emoção → aumenta peso
        if w in emotion_words:
            tokens.extend([w] * abs(emotion_words[w]))
            continue

        # Aceitar verbos, adjetivos e substantivos (heurística leve)
        if len(w) > 3 and w not in stopwords:
            tokens.append(w)

    return " ".join(tokens)


def generate_wordcloud(text, title="Wordcloud"):
    if not str(text).strip():
        st.info("Nenhum conteúdo para gerar nuvem de palavras.")
        return

    wc = WordCloud(
        width=1200,
        height=500,
        background_color="black",
        colormap="inferno"
    ).generate(text)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)


# ------------------------
# PDF
# ------------------------
def generate_pdf(content):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    story = [Paragraph(content, styles["Normal"])]
    doc.build(story)
    buffer.seek(0)
    return buffer


# ------------------------
# IA Gemini
# ------------------------
def ask_gemini(prompt):
    api_key = st.secrets.get("GOOGLE_API_KEY", "")
    if not api_key:
        return "Google API Key não configurada."
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    resp = model.generate_content(prompt)
    return resp.text


# ------------------------
# CSV Loader Robusto
# ------------------------
def load_csv_auto(file):
    if file is None:
        return None

    raw = file.read()
    enc = chardet.detect(raw)["encoding"]

    for sep in [",", ";", "|", "\t"]:
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc, sep=sep)
            if len(df.columns) > 1:
                return df
        except:
            pass

    raise ValueError("Não foi possível detectar o delimitador do CSV.")


# ------------------------
# Landing Page + Login
# ------------------------
def landing_page():
    st.markdown(
        """
        <style>
        body {
            background: linear-gradient(180deg, #000000 0%, #0f3b1c 50%, #ff6a00 100%) !important;
        }
        .center-box {
            padding: 60px;
            background: rgba(0,0,0,0.70);
            border-radius: 18px;
            width: 450px;
            margin-left: auto;
            margin-right: auto;
            margin-top: 80px;
            text-align: center;
        }
        .login-title {
            color: white;
            font-size: 32px;
            font-weight: 600;
        }
        .login-sub {
            color: #ff8f42;
            font-size: 18px;
            margin-bottom: 30px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='center-box'>", unsafe_allow_html=True)

    # LOGO via Base64
    logo = load_logo_from_base64()
    if logo:
        st.image(logo, width=220)

    st.markdown("<div class='login-title'>MindVision® Analytics</div>", unsafe_allow_html=True)
    st.markdown("<div class='login-sub'>Acesso Restrito</div>", unsafe_allow_html=True)

    email = st.text_input("Email", key="login_email")
    password = st.text_input("Senha", type="password", key="login_pass")

    if st.button("Entrar", key="btn_login"):
        user = verify_user(email, password)
        if user:
            st.session_state["logged"] = True
            st.session_state["username"] = user
            st.session_state["email"] = email
            st.rerun()
        else:
            st.error("Credenciais incorretas.")

    st.markdown("</div>", unsafe_allow_html=True)


# ------------------------
# Painel Admin
# ------------------------
def admin_panel():
    st.title("Painel Administrativo")

    st.subheader("Gerar Hash PBKDF2")
    senha = st.text_input("Senha para gerar hash", key="admin_newpass")
    if st.button("Gerar Hash", key="admin_genhash"):
        if senha:
            st.code(pwd_context.hash(senha))
        else:
            st.warning("Digite uma senha.")

    st.subheader("Modelo TOML para novo usuário:")
    email = st.text_input("Novo email", key="new_email")
    nome = st.text_input("Nome", key="new_nome")
    hash_senha = st.text_input("Hash PBKDF2", key="new_hash")

    if st.button("Gerar TOML", key="btn_toml"):
        if email and nome and hash_senha:
            st.code(
                f"""
[users.{email.split('@')[0]}]
name = "{nome}"
email = "{email}"
password = "{hash_senha}"
                """
            )


# ------------------------
# APP PRINCIPAL
# ------------------------
def main_app():
    st.sidebar.title(f"Bem-vindo(a), {st.session_state.get('username', '')}")
    page = st.sidebar.radio("Menu", ["Dashboard", "Candidatos", "Empresas", "Admin"], key="menu_main")

    # Dashboard
    if page == "Dashboard":
        st.title("Dashboard MindVision Analytics")
        st.write("Visualização geral das análises.")

    # Candidatos
    if page == "Candidatos":
        st.title("CSV de Candidatos")
        file = st.file_uploader("Envie o CSV de Candidatos", type=["csv"], key="cand_upload")

        if file:
            try:
                df = load_csv_auto(file)
                st.dataframe(df, use_container_width=True)

                text_cols = [c for c in df.columns if df[c].dtype == "object"]
                col_sel = st.selectbox("Selecione coluna de texto", text_cols, key="cand_col")

                if col_sel:
                    cleaned = " ".join(df[col_sel].map(preprocess_text))
                    generate_wordcloud(cleaned)

                    st.subheader("Análise com IA")
                    if st.button("Gerar Insights", key="insight_cand"):
                        prompt = f"Analise a coluna '{col_sel}' dos candidatos:\n\n{cleaned[:5000]}"
                        st.write(ask_gemini(prompt))

            except Exception as e:
                st.error(f"Erro ao ler CSV: {e}")

    # Empresas
    if page == "Empresas":
        st.title("CSV de Empresas")
        file = st.file_uploader("Envie o CSV de Empresas", type=["csv"], key="emp_upload")

        if file:
            try:
                df = load_csv_auto(file)
                st.dataframe(df, use_container_width=True)

                text_cols = [c for c in df.columns if df[c].dtype == "object"]
                col_sel = st.selectbox("Selecione coluna textual", text_cols, key="emp_col")

                if col_sel:
                    cleaned = " ".join(df[col_sel].map(preprocess_text))
                    generate_wordcloud(cleaned)

                    st.subheader("Análise IA")
                    if st.button("Gerar Insights Empresas", key="insight_emp"):
                        prompt = f"Analise a coluna '{col_sel}' das empresas:\n\n{cleaned[:5000]}"
                        st.write(ask_gemini(prompt))

            except Exception as e:
                st.error(f"Erro ao ler CSV: {e}")

    # Admin
    if page == "Admin":
        admin_panel()

    # Logout
    if st.sidebar.button("Sair", key="logout"):
        st.session_state.clear()
        st.rerun()


# ------------------------
# ROTEAMENTO DE TELAS
# ------------------------
if "logged" not in st.session_state or not st.session_state["logged"]:
    landing_page()
else:
    main_app()
