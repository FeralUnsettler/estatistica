# ================================================================
# REPARA ANALYTICS — v10.0
# Compatível com Streamlit Cloud / Sem nested dialogs / Sem bcrypt
# Login com st.dialog + Painel Admin + Gemini Flash
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
# CONFIG INICIAL
# ================================================================
st.set_page_config(page_title="Repara Analytics", layout="wide")

# ---- CryptContext sem bcrypt ----
PWD_CTX = CryptContext(
    schemes=["pbkdf2_sha256"],
    default="pbkdf2_sha256",
    deprecated="auto",
)

RESET_TOKEN_TTL = 15 * 60

genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])


def load_users():
    users_raw = st.secrets.get("users", {})
    users = {}
    for username, info in users_raw.items():
        users[username] = {
            "name": info.get("name"),
            "email": info.get("email"),
            "password": info.get("password")
        }
    return users


USERS = load_users()


# ================================================================
# TOKENS
# ================================================================
def init_tokens():
    if "reset_tokens" not in st.session_state:
        st.session_state.reset_tokens = {}


def generate_token(username):
    token = secrets.token_urlsafe(16)
    st.session_state.reset_tokens[token] = {
        "username": username,
        "expire": time.time() + RESET_TOKEN_TTL
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
# SENHAS / LOGIN
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
# CSS
# ================================================================
def inject_css():
    st.markdown("""
    <style>
        .login-box {
            background: white;
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
        st.session_state.show_recovery = True  # ← NÃO ABRE NESTED DIALOG

    st.markdown("</div>", unsafe_allow_html=True)


# ================================================================
# RECUPERAÇÃO — ABERTO FORA DO LOGIN
# ================================================================
@st.dialog("Recuperação de senha")
def recovery_dialog():
    username = st.text_input("Usuário")
    if st.button("Gerar token"):
        if username not in USERS:
            st.error("Usuário não existe.")
        else:
            token = generate_token(username)
            st.success("Token gerado!")
            st.info(f"Use este token: `{token}`")


# ================================================================
# RESET DE SENHA
# ================================================================
def reset_password_ui():
    st.subheader("🔐 Redefinir senha")

    token = st.text_input("Token de redefinição")
    new_pwd = st.text_input("Nova senha", type="password")

    if st.button("Atualizar senha"):
        ok, username = validate_token(token)
        if not ok:
            st.error(username)
            return

        hashed = PWD_CTX.hash(new_pwd)

        st.success("Senha atualizada!")
        st.code(
            f'''
[users.{username}]
name = "{USERS[username]['name']}"
email = "{USERS[username]['email']}"
password = "{hashed}"
            ''',
            language="toml"
        )


# ================================================================
# GEMINI
# ================================================================
def gemini_analyse(text_list):
    if not text_list:
        return "Nenhum texto disponível."

    content = "\n".join(text_list)

    model = genai.GenerativeModel("gemini-2.5-flash")
    out = model.generate_content(f"""
Analise profundamente o conteúdo abaixo e gere:
- Resumo executivo
- Clusters temáticos
- Emoções
- Tabela Tema | Exemplo | Impacto | Ação
- Recomendações
---
{content}
""")

    return out.text


# ================================================================
# CSV UTILS
# ================================================================
def read_csv_any(file):
    try:
        return pd.read_csv(file, sep=None, engine="python")
    except:
        return pd.read_csv(file)


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
# ADMIN
# ================================================================
def admin_panel():

    st.title("🛡️ Painel Admin")

    st.subheader("Usuários atuais")
    for u, info in USERS.items():
        st.markdown(f"- **{u}** — {info['email']}")

    st.markdown("---")
    st.subheader("Criar novo usuário")

    user = st.text_input("Username")
    nome = st.text_input("Nome completo")
    email = st.text_input("Email")
    pwd = st.text_input("Senha", type="password")

    if st.button("Gerar bloco TOML"):
        hash_pwd = PWD_CTX.hash(pwd)
        st.code(
            f'''
[users.{user}]
name = "{nome}"
email = "{email}"
password = "{hash_pwd}"
            ''',
            language="toml"
        )

    st.markdown("---")
    reset_password_ui()


# ================================================================
# MAIN APP
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

    st.header("🔍 Análise de CSVs + Gemini Flash")

    cand = st.file_uploader("CSV de candidatos")
    emp = st.file_uploader("CSV de empresas")

    if cand:
        df = read_csv_any(cand)
        st.subheader("Candidatos")
        st.dataframe(df)

        cm = infer_cols(df)
        if cm["pain"]:
            if st.button("Análise IA — Candidatos"):
                st.markdown(gemini_analyse(df[cm["pain"]].dropna().tolist()))

    if emp:
        df = read_csv_any(emp)
        st.subheader("Empresas")
        st.dataframe(df)

        em = infer_cols(df)
        if em["hr"]:
            if st.button("Análise IA — Empresas"):
                st.markdown(gemini_analyse(df[em["hr"]].dropna().tolist()))

    if cand and emp:
        df1 = read_csv_any(cand)
        df2 = read_csv_any(emp)
        cm = infer_cols(df1)
        em = infer_cols(df2)

        txt = []
        if cm["pain"]: txt += df1[cm["pain"]].dropna().tolist()
        if em["hr"]: txt += df2[em["hr"]].dropna().tolist()

        if st.button("Análise IA — Cruzada"):
            st.markdown(gemini_analyse(txt))


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

if not st.session_state.logged:
    if st.button("Entrar"):
        login_dialog()

    if st.session_state.show_recovery:
        recovery_dialog()

else:
    main_app()
