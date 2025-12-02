# ================================================================
# REPARA ANALYTICS — v9.0
# Compatível com Streamlit Cloud
# Login com st.dialog + Painel Admin + Gemini 2.5 Flash
# ================================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import google.generativeai as genai
from passlib.context import CryptContext
import secrets
import time

# ================================================================
# CONFIG INICIAL
# ================================================================
st.set_page_config(page_title="Repara Analytics", layout="wide")

PWD_CTX = CryptContext(schemes=["bcrypt"], deprecated="auto")
RESET_TOKEN_TTL = 15 * 60  # 15 minutos

genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])


# ================================================================
# USUÁRIOS (carregados de secrets.toml)
# ================================================================
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
# FUNÇÕES DE SEGURANÇA
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
# TOKENS DE RECUPERAÇÃO
# ================================================================
def init_reset_tokens():
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
# CSS
# ================================================================
def inject_css():
    st.markdown("""
    <style>
        .login-box {
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        }
        .login-title {
            font-size: 26px;
            font-weight: 700;
            text-align: center;
            color: #0b63ce;
        }
    </style>
    """, unsafe_allow_html=True)


# ================================================================
# DIALOG — LOGIN
# ================================================================
@st.dialog("Login")
def login_dialog():

    inject_css()
    st.markdown("<div class='login-box'>", unsafe_allow_html=True)
    st.markdown("<div class='login-title'>REPARA Login</div>", unsafe_allow_html=True)

    user = st.text_input("Usuário")
    pwd = st.text_input("Senha", type="password")

    if st.button("Entrar"):
        ok, info = authenticate(user, pwd)
        if ok:
            st.session_state.logged = True
            st.session_state.userinfo = info
            st.session_state.page = "main"
            st.success("Login bem-sucedido!")
            st.rerun()
        else:
            st.error(info)

    if st.button("Esqueci a senha"):
        recovery_dialog()

    st.markdown("</div>", unsafe_allow_html=True)


# ================================================================
# DIALOG — RECUPERAÇÃO DE SENHA
# ================================================================
@st.dialog("Recuperação de Senha")
def recovery_dialog():

    username = st.text_input("Usuário para recuperação")

    if st.button("Gerar token"):
        if username not in USERS:
            st.error("Usuário não encontrado.")
        else:
            token = generate_token(username)
            st.success("Token gerado! (válido por 15 minutos)")
            st.info(f"Token: `{token}`")


# ================================================================
# TELA DE RESET DE SENHA
# ================================================================
def reset_password_section():
    st.subheader("🔐 Redefinir senha")

    token = st.text_input("Token")
    new_pass = st.text_input("Nova senha", type="password")

    if st.button("Atualizar senha"):
        ok, username = validate_token(token)
        if not ok:
            st.error(username)  # Aqui username é a mensagem de erro
        else:
            hashed = PWD_CTX.hash(new_pass)
            st.success("Senha atualizada! Copie o bloco TOML abaixo:")

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
# GEMINI — ANALISADOR
# ================================================================
def analisar_texto_gemini(lista):
    if len(lista) == 0:
        return "Sem dados de texto disponíveis."

    texto = "\n".join([str(t) for t in lista])

    prompt = f"""
Analise profundamente estas respostas e apresente:
- resumo executivo
- clusters temáticos
- emoções predominantes
- tabela Tema | Exemplo | Impacto | Ação
- recomendações

Texto:
{texto}
"""

    model = genai.GenerativeModel("gemini-2.5-flash")
    out = model.generate_content(prompt)
    return out.text


# ================================================================
# CSV UTILITIES
# ================================================================
def read_csv_any(f):
    try:
        return pd.read_csv(f, sep=None, engine="python")
    except:
        return pd.read_csv(f)


def infer_cols(df):
    def find(keys):
        for key in keys:
            for col in df.columns:
                if key in col.lower():
                    return col
        return None

    return {
        "age": find(["idade", "age"]),
        "gender": find(["genero", "sexo"]),
        "pain": find(["dor", "pain", "desafio", "feedback", "motivo"]),
        "hr": find(["rh", "hr", "gestao", "problema", "challenge"])
    }


# ================================================================
# PAINEL ADMIN
# ================================================================
def admin_panel():

    if st.session_state.userinfo["email"] != "admin@repara.com":
        st.error("Acesso restrito ao administrador.")
        return

    st.title("🛡️ Painel Administrativo")

    st.subheader("Usuários atuais")

    for username, info in USERS.items():
        st.markdown(f"- **{username}** — {info['name']} — {info['email']}")

    st.markdown("---")
    st.subheader("➕ Criar novo usuário")

    new_user = st.text_input("Username")
    new_name = st.text_input("Nome completo")
    new_email = st.text_input("Email")
    new_pass = st.text_input("Senha", type="password")

    if st.button("Gerar bloco TOML"):
        if not new_user or not new_pass:
            st.error("Preencha username e senha.")
        else:
            hashed = PWD_CTX.hash(new_pass)
            st.success("Copie o bloco abaixo para o secrets.toml:")
            st.code(
                f'''
[users.{new_user}]
name = "{new_name}"
email = "{new_email}"
password = "{hashed}"
                ''',
                language="toml"
            )

    st.markdown("---")
    st.subheader("🔐 Gerar Hash Isolado")

    raw = st.text_input("Senha para hash", type="password")
    if st.button("Gerar hash"):
        st.code(PWD_CTX.hash(raw))

    st.markdown("---")
    st.subheader("🧨 Remover usuário")

    rm_user = st.text_input("Username para remover")
    if st.button("Gerar instrução de remoção"):
        if rm_user not in USERS:
            st.error("Usuário não existe.")
        else:
            st.warning("Remova manualmente este bloco no secrets.toml:")
            st.code(f"[users.{rm_user}]\n# remover este bloco", language="toml")


# ================================================================
# APP PRINCIPAL
# ================================================================
def main_app():

    st.title("📊 Repara Analytics")

    st.sidebar.success(f"Logado como: {st.session_state.userinfo['name']}")

    if st.sidebar.button("Painel Admin"):
        st.session_state.page = "admin"
        st.rerun()

    if st.sidebar.button("Sair"):
        st.session_state.logged = False
        st.session_state.page = "main"
        st.rerun()

    # Abrir painel admin
    if st.session_state.page == "admin":
        admin_panel()
        st.markdown("---")
        reset_password_section()
        return

    # Página principal
    st.header("🔍 Análise de Dados + IA")

    st.sidebar.header("📥 Upload de CSVs")
    cand = st.sidebar.file_uploader("Candidatos")
    emp = st.sidebar.file_uploader("Empresas")

    tabs = st.tabs(["👤 Candidatos", "🏢 Empresas", "🔀 Cruzada"])

    # ---------------------------------------------------------
    # CANDIDATOS
    # ---------------------------------------------------------
    with tabs[0]:
        if cand:
            df = read_csv_any(cand)
            st.dataframe(df.head())

            m = infer_cols(df)

            if m["gender"]:
                fig, ax = plt.subplots()
                df[m["gender"]].fillna("N/A").value_counts().plot(
                    kind="pie", autopct="%1.1f%%", ax=ax
                )
                ax.set_ylabel("")
                st.pyplot(fig)

            if m["pain"]:
                text = " ".join(df[m["pain"]].dropna())
                wc = WordCloud(width=900, height=400).generate(text)
                fig, ax = plt.subplots()
                ax.imshow(wc)
                ax.axis("off")
                st.pyplot(fig)

                if st.button("IA — Análise dos Candidatos"):
                    st.markdown(analisar_texto_gemini(df[m["pain"]].dropna()))

    # ---------------------------------------------------------
    # EMPRESAS
    # ---------------------------------------------------------
    with tabs[1]:
        if emp:
            df = read_csv_any(emp)
            st.dataframe(df.head())

            m = infer_cols(df)

            if m["hr"]:
                fig, ax = plt.subplots()
                df[m["hr"]].dropna().value_counts().plot(kind="barh", ax=ax)
                st.pyplot(fig)

                if st.button("IA — Análise das Empresas"):
                    st.markdown(analisar_texto_gemini(df[m["hr"]].dropna()))

    # ---------------------------------------------------------
    # CRUZADA
    # ---------------------------------------------------------
    with tabs[2]:
        if cand and emp:
            df1 = read_csv_any(cand)
            df2 = read_csv_any(emp)

            m1 = infer_cols(df1)
            m2 = infer_cols(df2)

            textos = []

            if m1["pain"]:
                textos += df1[m1["pain"]].dropna().tolist()
            if m2["hr"]:
                textos += df2[m2["hr"]].dropna().tolist()

            if st.button("IA — Análise Cruzada"):
                st.markdown(analisar_texto_gemini(textos))
        else:
            st.info("Envie os dois CSVs para ativar a análise cruzada.")

    st.markdown("---")
    reset_password_section()


# ================================================================
# EXECUÇÃO
# ================================================================
inject_css()
init_reset_tokens()

if "logged" not in st.session_state:
    st.session_state.logged = False

if "page" not in st.session_state:
    st.session_state.page = "main"

if not st.session_state.logged:
    if st.button("Entrar"):
        login_dialog()
else:
    main_app()
