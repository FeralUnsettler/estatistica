# ================================================================
# REPARA ANALYTICS — v8.0
# Streamlit Cloud version (100% secrets.toml)
# Login modal + admin panel + Gemini 2.5 Flash + CSV analytics
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
# CONFIGURAÇÕES INICIAIS
# ================================================================
st.set_page_config(page_title="Repara Analytics", layout="wide")

PWD_CTX = CryptContext(schemes=["bcrypt"], deprecated="auto")
RESET_TOKEN_TTL = 15 * 60   # token válido por 15 minutos

genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# ================================================================
# CARREGAR USERS DO SECRETS.TOML
# ================================================================
def load_users():
    users_raw = st.secrets.get("users", {})
    users = {}
    for username, info in users_raw.items():
        users[username] = {
            "name": info.get("name"),
            "email": info.get("email"),
            "password": info.get("password"),
        }
    return users

USERS = load_users()

# ================================================================
# SEGURANÇA
# ================================================================
def verify_password(password, hashed):
    try:
        return PWD_CTX.verify(password, hashed)
    except:
        return False

def authenticate(username, password):
    if username not in USERS:
        return False, "Usuário não encontrado"
    if verify_password(password, USERS[username]["password"]):
        return True, USERS[username]
    return False, "Senha incorreta"

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
        return False, "Token inválido"
    if time.time() > entry["expire"]:
        del st.session_state.reset_tokens[token]
        return False, "Token expirou"
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
        color: #0b63ce;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# ================================================================
# MODAL DE LOGIN
# ================================================================
def login_modal():
    with st.modal("Login"):
        inject_css()
        st.markdown("<div class='login-box'>", unsafe_allow_html=True)

        st.markdown("<div class='login-title'>REPARA Login</div>", unsafe_allow_html=True)

        user = st.text_input("Usuário")
        pwd = st.text_input("Senha", type="password")

        if st.button("Entrar"):
            ok, data = authenticate(user, pwd)
            if ok:
                st.session_state.logged = True
                st.session_state.userinfo = data
                st.session_state.page = "main"
                st.success("Autenticado!")
                st.experimental_rerun()
            else:
                st.error(data)

        if st.button("Esqueci a senha"):
            recovery_modal()

        st.markdown("</div>", unsafe_allow_html=True)

# ================================================================
# MODAL DE RECUPERAÇÃO
# ================================================================
def recovery_modal():
    with st.modal("Recuperação de Senha"):
        username = st.text_input("Usuário para recuperação")

        if st.button("Gerar token"):
            if username not in USERS:
                st.error("Usuário não encontrado")
            else:
                token = generate_token(username)
                st.success("Token Gerado! Válido por 15 minutos.")
                st.info(f"Token: `{token}`")

# ================================================================
# RESET DE SENHA
# ================================================================
def reset_password_section():
    st.subheader("🔐 Redefinir senha")

    token = st.text_input("Token")
    new_pass = st.text_input("Nova senha", type="password")

    if st.button("Atualizar senha"):
        ok, res = validate_token(token)
        if not ok:
            st.error(res)
        else:
            username = res
            hashed = PWD_CTX.hash(new_pass)
            st.success("Senha atualizada. Copie o bloco abaixo para o secrets.toml:")

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
# IA — ANALISADOR
# ================================================================
def analisar_texto_gemini(lista):
    if len(lista) == 0:
        return "Sem dados de texto."

    texto = "\n".join([str(t) for t in lista])

    prompt = f"""
Analise este conjunto de respostas e apresente:

- Resumo executivo  
- Clusters temáticos  
- Emoções predominantes  
- Tabela Tema | Exemplo | Impacto | Ação  
- Recomendações  

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
        for k in keys:
            for c in df.columns:
                if k in c.lower():
                    return c
        return None

    return {
        "age": find(["idade","age"]),
        "gender": find(["genero","sexo","gender"]),
        "pain": find(["dor","pain","desafio","feedback","motivo"]),
        "hr": find(["rh","hr","gestao","challenge","problem"])
    }

# ================================================================
# PAINEL ADMIN
# ================================================================
def admin_panel():
    st.title("🛡️ Painel Administrativo")

    if st.session_state.userinfo["email"] != "admin@repara.com":
        st.error("Acesso restrito ao administrador.")
        return

    st.subheader("👥 Usuários Atuais")

    for username, data in USERS.items():
        st.markdown(f"- **{username}** — {data['name']} — {data['email']}")

    st.markdown("---")
    st.subheader("➕ Criar Novo Usuário")

    new_user = st.text_input("Username")
    new_name = st.text_input("Nome completo")
    new_email = st.text_input("Email")
    new_pass = st.text_input("Senha", type="password")

    if st.button("Gerar bloco TOML"):
        if not new_user or not new_pass:
            st.error("Preencha os campos obrigatórios.")
        else:
            hashed = PWD_CTX.hash(new_pass)
            st.success("Copie este bloco para o secrets.toml:")
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

    raw = st.text_input("Senha para geração de hash", type="password")
    if st.button("Gerar hash isolado"):
        st.code(PWD_CTX.hash(raw))

    st.markdown("---")
    st.subheader("🧨 Remover Usuário")

    rm_user = st.text_input("Username para remover")

    if st.button("Gerar instrução de remoção"):
        if rm_user not in USERS:
            st.error("Usuário não encontrado.")
        else:
            st.warning("Remova manualmente este bloco do secrets.toml:")
            st.code(
                f"[users.{rm_user}]\n# REMOVER ESTE BLOCO",
                language="toml"
            )

# ================================================================
# APP PRINCIPAL
# ================================================================
def main_app():
    st.title("📊 Repara Analytics")

    st.sidebar.success(f"Logado como: {st.session_state.userinfo['name']}")

    if st.sidebar.button("Painel Admin"):
        st.session_state.page = "admin"
        st.experimental_rerun()

    if st.sidebar.button("Sair"):
        st.session_state.logged = False
        st.session_state.page = "main"
        st.experimental_rerun()

    # Painel Admin
    if st.session_state.get("page") == "admin":
        admin_panel()
        st.markdown("---")
        reset_password_section()
        return

    st.header("🔍 Análise de Dados + IA")

    st.sidebar.header("📥 Upload de CSVs")
    cand = st.sidebar.file_uploader("Candidatos")
    emp = st.sidebar.file_uploader("Empresas")

    tabs = st.tabs(["👤 Candidatos", "🏢 Empresas", "🔀 Cruzada"])

    # ---- TAB CANDIDATOS ----
    with tabs[0]:
        if cand:
            df = read_csv_any(cand)
            st.dataframe(df.head())
            m = infer_cols(df)

            if m["gender"]:
                fig, ax = plt.subplots()
                df[m["gender"]].fillna("N/A").value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
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

    # ---- TAB EMPRESAS ----
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

    # ---- TAB CRUZADA ----
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
            st.info("Envie os dois CSVs para ativar esta seção.")

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
    st.button("Entrar", on_click=login_modal)
else:
    main_app()
