# ================================================================
# REPARA ANALYTICS — v11.0
# Compatível com Streamlit Cloud
# Login com st.dialog + Painel Admin + pbkdf2_sha256
# CSV seguro (sem erros), Gemini Flash integrado
# ================================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import google.generativeai as genai
import time
import secrets
from passlib.context import CryptContext

# ================================================================
# CONFIG GLOBAL
# ================================================================
st.set_page_config(page_title="Repara Analytics", layout="wide")

# Segurança sem bcrypt
PWD_CTX = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

RESET_TOKEN_TTL = 15 * 60  # 15 min

# Google Gemini
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# ================================================================
# LOAD USERS
# ================================================================
def load_users():
    raw = st.secrets.get("users", {})

    users = {}
    for username, info in raw.items():
        users[username] = {
            "name": info.get("name"),
            "email": info.get("email"),
            "password": info.get("password"),
        }
    return users


USERS = load_users()

# ================================================================
# TOKEN DE REDEFINIÇÃO
# ================================================================
def init_tokens():
    if "reset_tokens" not in st.session_state:
        st.session_state.reset_tokens = {}


def generate_token(username):
    token = secrets.token_urlsafe(16)
    st.session_state.reset_tokens[token] = {
        "username": username,
        "expire": time.time() + RESET_TOKEN_TTL,
    }
    return token


def validate_token(token):
    entry = st.session_state.reset_tokens.get(token)
    if not entry:
        return False, "Token inválido."
    if time.time() > entry["expire"]:
        del st.session_state.reset_tokens[token]
        return False, "Token expirado."
    return True, entry["username"]


# ================================================================
# PASSWORD / AUTH
# ================================================================
def verify_password(password, hashed):
    try:
        return PWD_CTX.verify(password, hashed)
    except:
        return False


def authenticate(username, password):
    if username not in USERS:
        return False, "Usuário não encontrado."

    if verify_password(password, USERS[username]["password"]):
        return True, USERS[username]

    return False, "Senha incorreta."


# ================================================================
# CSS BASE
# ================================================================
def inject_css():
    st.markdown("""
    <style>
        .login-box {
            background: #ffffff;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.12);
        }
    </style>
    """, unsafe_allow_html=True)


# ================================================================
# LOGIN DIALOG
# ================================================================
@st.dialog("Login")
def login_dialog():
    inject_css()

    st.markdown("<div class='login-box'>", unsafe_allow_html=True)

    username = st.text_input("Usuário")
    pwd = st.text_input("Senha", type="password")

    if st.button("Entrar"):
        ok, info = authenticate(username, pwd)
        if ok:
            st.session_state.logged = True
            st.session_state.userinfo = info
            st.session_state.page = "main"
            st.success("Bem-vindo!")
            st.rerun()
        else:
            st.error(info)

    if st.button("Esqueci a senha"):
        st.session_state.show_recovery = True

    st.markdown("</div>", unsafe_allow_html=True)


# ================================================================
# RECUPERAÇÃO (DIÁLOGO SEPARADO)
# ================================================================
@st.dialog("Recuperação de senha")
def recovery_dialog():
    username = st.text_input("Usuário para recuperação")

    if st.button("Gerar token"):
        if username not in USERS:
            st.error("Usuário não encontrado.")
        else:
            token = generate_token(username)
            st.success("Token gerado!")
            st.info(f"Use este token:\n\n`{token}`")


# ================================================================
# REDEFINIÇÃO DE SENHA (UI)
# ================================================================
def reset_password_ui():
    st.subheader("🔐 Redefinir senha")

    token = st.text_input("Token")
    new_pwd = st.text_input("Nova senha", type="password")

    if st.button("Atualizar senha"):
        ok, username = validate_token(token)
        if not ok:
            st.error(username)
            return

        hashed = PWD_CTX.hash(new_pwd)

        st.success("Senha atualizada!")
        st.code(
            f"""
[users.{username}]
name = "{USERS[username]['name']}"
email = "{USERS[username]['email']}"
password = "{hashed}"
            """,
            language="toml"
        )


# ================================================================
# GEMINI — ANÁLISE IA
# ================================================================
def gemini_analyse(text_list):
    if not text_list:
        return "Nenhum texto disponível."

    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = f"""
Analise profundamente os textos abaixo e gere:

- Resumo executivo
- Clusters temáticos
- Emoções predominantes
- Tabela: Tema | Exemplo | Impacto | Ação recomendada
- Recomendações finais

---
{chr(10).join(text_list)}
"""

    out = model.generate_content(prompt)
    return out.text


# ================================================================
# CSV READER 100% SEGURO
# ================================================================
def read_csv_any(file):
    if file is None:
        return None

    if hasattr(file, "size") and file.size == 0:
        st.warning("⚠ O arquivo enviado está vazio.")
        return None

    try:
        df = pd.read_csv(file, sep=None, engine="python")
        if df.empty:
            st.warning("⚠ O CSV não contém dados.")
            return None
        return df

    except pd.errors.EmptyDataError:
        st.error("⚠ O arquivo está vazio ou corrompido.")
        return None

    except Exception as e:
        st.error(f"Erro ao ler o CSV: {e}")
        return None


# ================================================================
# COLUNA DE TEXTO AUTOMÁTICA
# ================================================================
def infer_cols(df):
    def find(keys):
        for col in df.columns:
            if any(k in col.lower() for k in keys):
                return col
        return None

    return {
        "pain": find(["dor", "feedback", "coment", "desafio"]),
        "hr": find(["gestao", "rh", "motivo", "pain"]),
    }


# ================================================================
# PAINEL ADMIN
# ================================================================
def admin_panel():

    st.title("🛡️ Painel Administrativo")

    st.subheader("Usuários atuais")
    for user, info in USERS.items():
        st.markdown(f"- **{user}** — {info['email']}")

    st.markdown("---")
    st.subheader("Criar novo usuário")

    username = st.text_input("Username")
    name = st.text_input("Nome completo")
    email = st.text_input("Email")
    pwd = st.text_input("Senha", type="password")

    if st.button("Gerar bloco TOML"):
        hashed = PWD_CTX.hash(pwd)
        st.code(
            f"""
[users.{username}]
name = "{name}"
email = "{email}"
password = "{hashed}"
            """,
            language="toml"
        )

    st.markdown("---")
    reset_password_ui()


# ================================================================
# APP PRINCIPAL
# ================================================================
def main_app():

    st.title("📊 Repara Analytics")

    st.sidebar.success(f"Usuário: {st.session_state.userinfo['name']}")

    if st.sidebar.button("Painel Admin"):
        st.session_state.page = "admin"
        st.rerun()

    if st.sidebar.button("Sair"):
        st.session_state.logged = False
        st.rerun()

    if st.session_state.page == "admin":
        admin_panel()
        return

    st.header("🔍 Upload de CSVs para análise")

    cand = st.file_uploader("CSV de candidatos")
    emp = st.file_uploader("CSV de empresas")

    # -----------------------------------------------------------------
    # CANDIDATOS
    # -----------------------------------------------------------------
    if cand:
        df = read_csv_any(cand)
        if df is not None:
            st.subheader("Candidatos")
            st.dataframe(df)

            cols = infer_cols(df)

            if cols["pain"]:
                if st.button("IA — Análise dos candidatos"):
                    st.markdown(gemini_analyse(df[cols["pain"]].dropna().tolist()))

    # -----------------------------------------------------------------
    # EMPRESAS
    # -----------------------------------------------------------------
    if emp:
        df = read_csv_any(emp)
        if df is not None:
            st.subheader("Empresas")
            st.dataframe(df)

            cols = infer_cols(df)

            if cols["hr"]:
                if st.button("IA — Análise das empresas"):
                    st.markdown(gemini_analyse(df[cols["hr"]].dropna().tolist()))

    # -----------------------------------------------------------------
    # ANÁLISE CRUZADA
    # -----------------------------------------------------------------
    if cand and emp:
        df1 = read_csv_any(cand)
        df2 = read_csv_any(emp)

        if df1 is not None and df2 is not None:

            cols1 = infer_cols(df1)
            cols2 = infer_cols(df2)

            texts = []

            if cols1["pain"]:
                texts += df1[cols1["pain"]].dropna().tolist()
            if cols2["hr"]:
                texts += df2[cols2["hr"]].dropna().tolist()

            if st.button("IA — Análise cruzada"):
                st.markdown(gemini_analyse(texts))


# ================================================================
# EXECUÇÃO
# ================================================================
inject_css()
init_tokens()

if "logged" not in st.session_state:
    st.session_state.logged = False

if "page" not in st.session_state:
    st.session_state.page = "main"

if "show_recovery" not in st.session_state:
    st.session_state.show_recovery = False

# LOGIN / RECOVERY
if not st.session_state.logged:
    if st.button("Entrar"):
        login_dialog()

    if st.session_state.show_recovery:
        recovery_dialog()

else:
    main_app()
