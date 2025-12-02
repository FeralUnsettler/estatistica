# ================================================================
# REPARA ANALYTICS — v6.0
# Autenticação via SECRETS.TOML (sem YAML)
# Compatível com Streamlit Cloud
# ================================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import google.generativeai as genai
from passlib.context import CryptContext
import secrets
import time
from datetime import datetime

# ================================================================
# CONFIGURAÇÕES
# ================================================================
PWD_CTX = CryptContext(schemes=["bcrypt"], deprecated="auto")
RESET_TOKEN_TTL = 15 * 60   # 15 minutos

genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# ================================================================
# CARREGAR USERS DO SECRETS.TOML
# ================================================================
def load_users_from_secrets():
    users_raw = st.secrets.get("users", {})
    users = {}

    for username, info in users_raw.items():
        users[username] = {
            "name": info.get("name"),
            "email": info.get("email"),
            "password": info.get("password"),
        }
    return users

USERS = load_users_from_secrets()

# ================================================================
# FUNÇÕES DE SEGURANÇA
# ================================================================
def verify_password(plain, hashed):
    try:
        return PWD_CTX.verify(plain, hashed)
    except:
        return False

def authenticate(username, password):
    if username not in USERS:
        return False, "Usuário não encontrado"
    if verify_password(password, USERS[username]["password"]):
        return True, USERS[username]
    return False, "Senha incorreta"

# ================================================================
# RESET TOKEN EM SESSION STATE
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
# CSS PARA MODAL
# ================================================================
def inject_css():
    st.markdown("""
        <style>
        .login-box {
            background:white;
            padding:25px;
            border-radius:12px;
            box-shadow:0 8px 20px rgba(0,0,0,0.15);
        }
        .login-title {
            font-size:26px;
            font-weight:700;
            color:#0b63ce;
            text-align:center;
        }
        </style>
    """, unsafe_allow_html=True)

# ================================================================
# LOGIN MODAL
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
                st.session_state.user = data
                st.success("Autenticado!")
                st.experimental_rerun()
            else:
                st.error(data)

        if st.button("Esqueci a senha"):
            recovery_modal()

        st.markdown("</div>", unsafe_allow_html=True)

# ================================================================
# RECUPERAÇÃO DE SENHA MODAL
# ================================================================
def recovery_modal():
    with st.modal("Recuperar senha"):
        user = st.text_input("Usuário para recuperação")
        if st.button("Gerar token"):
            if user not in USERS:
                st.error("Usuário não encontrado")
            else:
                token = generate_token(user)
                st.success("Token gerado!")
                st.info(f"Token: `{token}` (Válido por 15 min)")

# ================================================================
# RESET DE SENHA (EXPANDER)
# ================================================================
def reset_password_section():
    st.subheader("🔐 Redefinir senha")
    token = st.text_input("Token")
    newp = st.text_input("Nova senha", type="password")

    if st.button("Redefinir"):
        ok, res = validate_token(token)
        if not ok:
            st.error(res)
        else:
            username = res
            USERS[username]["password"] = PWD_CTX.hash(newp)
            st.success("Senha atualizada!")

# ================================================================
# IA — ANALISAR TEXTO (GEMINI 2.5 FLASH)
# ================================================================
def analisar_texto_gemini(lista):
    if len(lista) == 0:
        return "Nenhum dado de texto encontrado."

    texto = "\n".join([str(t) for t in lista])

    prompt = f"""
Analise este conjunto de respostas e descreva:

- Resumo executivo  
- Clusters temáticos  
- Emoções predominantes  
- Top temas com %  
- Dores e causas  
- Ações recomendadas  
- Tabela final Tema | Exemplo | Impacto | Ação

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
# APP PRINCIPAL
# ================================================================
def main_app():
    st.title("📊 REPARA Analytics — Dashboard + IA")

    st.sidebar.success(f"Logado: {st.session_state.user['name']}")
    if st.sidebar.button("Sair"):
        st.session_state.logged = False
        st.experimental_rerun()

    # Upload CSVs
    st.sidebar.header("📥 Upload CSVs")
    f_cand = st.sidebar.file_uploader("Candidatos")
    f_emp = st.sidebar.file_uploader("Empresas")

    tabs = st.tabs(["👤 Candidatos", "🏢 Empresas", "🔀 Cruzada"])

    # Candidatos
    with tabs[0]:
        if f_cand:
            df = read_csv_any(f_cand)
            st.dataframe(df.head())
            m = infer_cols(df)

            if m["gender"]:
                fig, ax = plt.subplots()
                df[m["gender"]].fillna("N/A").value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
                ax.set_ylabel("")
                st.pyplot(fig)

            if m["pain"]:
                st.subheader("☁️ Wordcloud")
                text = " ".join(df[m["pain"]].dropna())
                wc = WordCloud(width=900, height=400).generate(text)
                fig, ax = plt.subplots()
                ax.imshow(wc)
                ax.axis("off")
                st.pyplot(fig)

                if st.button("Análise IA (Candidatos)"):
                    st.markdown(analisar_texto_gemini(df[m["pain"]].dropna().tolist()))

    # Empresas
    with tabs[1]:
        if f_emp:
            df = read_csv_any(f_emp)
            st.dataframe(df.head())
            m = infer_cols(df)

            if m["hr"]:
                fig, ax = plt.subplots()
                df[m["hr"]].dropna().value_counts().plot(kind="barh", ax=ax)
                st.pyplot(fig)

                if st.button("Análise IA (Empresas)"):
                    st.markdown(analisar_texto_gemini(df[m["hr"]].dropna().tolist()))

    # Cruzada
    with tabs[2]:
        if f_cand and f_emp:
            dfc = read_csv_any(f_cand)
            dfe = read_csv_any(f_emp)
            mc = infer_cols(dfc)
            me = infer_cols(dfe)
            textos = []
            if mc["pain"]:
                textos += dfc[mc["pain"]].dropna().tolist()
            if me["hr"]:
                textos += dfe[me["hr"]].dropna().tolist()

            if st.button("Análise IA Cruzada"):
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

if not st.session_state.logged:
    st.button("Entrar", on_click=login_modal)
else:
    main_app()
