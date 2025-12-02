# ================================================================
# REPARA ANALYTICS — v13.2
# Full App: Auth (Dialogs), Admin Panel, Gemini Chat, PDF, CSV Robust
# Fix: NO experimental_rerun anywhere
# ================================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import google.generativeai as genai
from passlib.context import CryptContext
import time
import secrets
import io
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# ----------------------------
# CONFIGURAÇÕES
# ----------------------------
st.set_page_config(page_title="Repara Analytics", layout="wide")

PWD_CTX = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
RESET_TOKEN_TTL = 15 * 60

if "GOOGLE_API_KEY" in st.secrets:
    try:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    except:
        pass

# ----------------------------
# CARREGAR USUÁRIOS DO SECRETS
# ----------------------------
def load_users():
    raw = st.secrets.get("users", {}) or {}
    users = {}
    for u, info in raw.items():
        users[u] = {
            "name": info.get("name"),
            "email": info.get("email"),
            "password": info.get("password"),
        }
    return users

USERS = load_users()

# ----------------------------
# TOKENS
# ----------------------------
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
        return False, "Token inválido"
    if time.time() > entry["expire"]:
        del st.session_state.reset_tokens[token]
        return False, "Token expirado"
    return True, entry["username"]

# ----------------------------
# AUTENTICAÇÃO
# ----------------------------
def verify_password(plain, hashed):
    try:
        return PWD_CTX.verify(plain, hashed)
    except Exception:
        return False

def authenticate(username, password):
    if username not in USERS:
        return False, "Usuário não encontrado."
    if verify_password(password, USERS[username]["password"]):
        return True, USERS[username]
    return False, "Senha incorreta."

# ----------------------------
# CSS
# ----------------------------
def inject_css():
    st.markdown("""
    <style>
    .login-box {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.12);
    }
    .login-title {
        color: #0b63ce;
        text-align: center;
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# ----------------------------
# LOGIN DIALOG
# ----------------------------
@st.dialog("Login")
def login_dialog():
    inject_css()
    st.markdown("<div class='login-box'>", unsafe_allow_html=True)

    st.markdown("<div class='login-title'>REPARA — Login</div>", unsafe_allow_html=True)

    user = st.text_input("Usuário")
    pwd = st.text_input("Senha", type="password")

    if st.button("Entrar"):
        ok, info = authenticate(user, pwd)
        if ok:
            st.session_state.logged = True
            st.session_state.userinfo = info
            st.session_state.page = "main"
            st.session_state._rerun = True
            st.success("Login realizado!")
        else:
            st.error(info)

    if st.button("Esqueci a senha"):
        st.session_state.show_recovery = True
        st.session_state._rerun = True

    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# RECUPERAÇÃO DE SENHA
# ----------------------------
@st.dialog("Recuperação de Senha")
def recovery_dialog():
    inject_css()
    st.markdown("<div class='login-box'>", unsafe_allow_html=True)

    st.markdown("<div class='login-title'>Recuperação</div>", unsafe_allow_html=True)

    user = st.text_input("Usuário")
    if st.button("Gerar Token"):
        if user not in USERS:
            st.error("Usuário não encontrado")
        else:
            token = generate_token(user)
            st.success("Token gerado!")
            st.info(f"Use este token: `{token}`")
            st.session_state._rerun = True

    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# LER CSV (ROBUSTO)
# ----------------------------
def read_csv_any(file):
    if file is None:
        return None

    try:
        if file.size == 0:
            st.warning("Arquivo vazio.")
            return None
    except:
        pass

    delims = [",", ";", "\t", "|"]
    for d in delims:
        try:
            file.seek(0)
            df = pd.read_csv(file, sep=d)
            if not df.empty:
                return df
        except:
            pass

    try:
        file.seek(0)
        return pd.read_csv(file, sep=None, engine="python")
    except:
        st.error("Erro ao ler CSV. Conferir delimitadores.")
        return None

# ----------------------------
# INFERIR COLUNAS TEXTUAIS
# ----------------------------
def infer_cols(df):
    lower = {c.lower(): c for c in df.columns}
    def find(keys):
        for k in keys:
            for low, real in lower.items():
                if k in low:
                    return real
        return None
    return {
        "pain": find(["feedback","dor","coment","descricao","descrição","texto","observacao","obs"]),
        "hr":   find(["gestao","rh","motivo","challenge","recrut","problema","vaga"])
    }

# ----------------------------
# GEMINI
# ----------------------------
def gemini_analyse(text_list, title):
    if not text_list:
        return "Nenhum texto detectado."

    if "GOOGLE_API_KEY" not in st.secrets:
        return "Gemini não configurado no secrets."

    joined = "\n".join(str(x) for x in text_list)

    prompt = f"""
Você é um analista sênior.
Produza um relatório estruturado sobre:

Título: {title}

Inclua:
1. Resumo executivo curto
2. Principais temas
3. Sentimentos
4. Recomendações práticas
5. Tabela (Tema | Exemplo | Impacto | Ação)
    
TEXTOS:
{joined}
"""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        resp = model.generate_content(prompt)
        return resp.text
    except Exception as e:
        return f"Erro com Gemini: {e}"

# ----------------------------
# CHAT COM GEMINI
# ----------------------------
def chat_with_gemini(df1, df2):
    st.header("💬 Chat com Gemini")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    context = ""
    if df1 is not None:
        context += "CANDIDATOS PREVIEW:\n" + df1.head(10).to_csv(index=False) + "\n"
    if df2 is not None:
        context += "EMPRESAS PREVIEW:\n" + df2.head(10).to_csv(index=False) + "\n"

    # Chat history
    for msg in st.session_state.chat_history:
        speaker = "Você" if msg["role"] == "user" else "IA"
        st.markdown(f"**{speaker}:** {msg['text']}")

    st.markdown("---")

    user_q = st.text_input("Pergunte algo sobre os dados:")
    if st.button("Enviar"):
        st.session_state.chat_history.append({"role":"user","text":user_q})

        if "GOOGLE_API_KEY" not in st.secrets:
            st.session_state.chat_history.append({"role":"assistant","text":"Gemini não configurado."})
            st.rerun()

        prompt = f"""
Responda usando APENAS o contexto abaixo:

{context}

Pergunta:
{user_q}
"""
        try:
            model = genai.GenerativeModel("gemini-2.5-flash")
            resp = model.generate_content(prompt)
            ans = resp.text
        except Exception as e:
            ans = f"Erro: {e}"

        st.session_state.chat_history.append({"role":"assistant","text":ans})
        st.rerun()

# ----------------------------
# PDF
# ----------------------------
def generate_pdf(title, text):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()

    story = [Paragraph(title, styles["Title"]), Spacer(1,12)]
    for line in text.split("\n"):
        if line.strip():
            story.append(Paragraph(line, styles["Normal"]))
        else:
            story.append(Spacer(1,6))

    doc.build(story)
    buffer.seek(0)
    return buffer

# ----------------------------
# PAINEL ADMIN
# ----------------------------
def admin_panel_ui():
    st.title("🛡 Painel Admin")

    st.info("Gerar novos usuários para secrets.toml")

    st.subheader("Usuários atuais")
    for u, info in USERS.items():
        st.markdown(f"- **{u}** — {info['email']}")

    st.markdown("---")

    st.subheader("Criar novo usuário")
    un = st.text_input("Username")
    nm = st.text_input("Nome")
    em = st.text_input("Email")
    pw = st.text_input("Senha", type="password")

    if st.button("Gerar Bloco TOML"):
        if not un or not pw:
            st.error("Username e senha obrigatórios.")
        else:
            h = PWD_CTX.hash(pw)
            st.success("Cole no secrets:")
            st.code(
                f'[users.{un}]\nname="{nm}"\nemail="{em}"\npassword="{h}"',
                language="toml"
            )

# ----------------------------
# MAIN APP
# ----------------------------
def main_app():
    st.title("📊 REPARA Analytics — v13.2")

    st.sidebar.success(f"Usuário: {st.session_state.userinfo['name']}")

    if st.sidebar.button("Painel Admin"):
        st.session_state.page = "admin"
        st.session_state._rerun = True

    if st.sidebar.button("Sair"):
        st.session_state.logged = False
        st.session_state._rerun = True

    if st.session_state.page == "admin":
        if st.session_state.userinfo["email"] != "admin@repara.com":
            st.error("Acesso negado.")
            return
        admin_panel_ui()
        return

    st.sidebar.header("CSV Uploads")
    cfile = st.sidebar.file_uploader("Candidatos CSV", type=["csv"])
    efile = st.sidebar.file_uploader("Empresas CSV", type=["csv"])

    df1 = read_csv_any(cfile) if cfile else None
    df2 = read_csv_any(efile) if efile else None

    tabs = st.tabs(["👤 Candidatos", "🏢 Empresas", "🔀 Cruzado", "💬 Chat"])

    # ------------------- CANDIDATOS --------------------
    with tabs[0]:
        st.header("👤 Candidatos")
        if df1 is not None:
            st.dataframe(df1)
            cols = infer_cols(df1)
            col = cols["pain"]

            if col:
                text = " ".join(df1[col].dropna().astype(str))
                if text:
                    wc = WordCloud(width=800, height=350).generate(text)
                    fig, ax = plt.subplots(figsize=(10,4))
                    ax.imshow(wc); ax.axis("off")
                    st.pyplot(fig)

                if st.button("Analisar (IA)"):
                    result = gemini_analyse(df1[col].dropna().tolist(), "Candidatos")
                    st.markdown(result)
                    pdf = generate_pdf("Análise Candidatos", result)
                    st.download_button("Baixar PDF", pdf, "candidatos.pdf")
            else:
                st.info("Nenhuma coluna textual detectada.")

    # ------------------- EMPRESAS --------------------
    with tabs[1]:
        st.header("🏢 Empresas")
        if df2 is not None:
            st.dataframe(df2)
            cols = infer_cols(df2)
            col = cols["hr"]

            if col:
                top = df2[col].astype(str).value_counts().head(10)
                fig, ax = plt.subplots()
                top.plot(kind="barh", ax=ax)
                st.pyplot(fig)

                if st.button("Analisar (IA)", key="an_emp"):
                    result = gemini_analyse(df2[col].dropna().tolist(), "Empresas")
                    st.markdown(result)
                    pdf = generate_pdf("Análise Empresas", result)
                    st.download_button("Baixar PDF", pdf, "empresas.pdf")
            else:
                st.info("Nenhuma coluna textual detectada.")

    # ------------------- CRUZADA --------------------
    with tabs[2]:
        st.header("🔀 Cruzada")
        if df1 is None or df2 is None:
            st.info("Carregue os dois CSVs.")
        else:
            col1 = infer_cols(df1)["pain"]
            col2 = infer_cols(df2)["hr"]

            merged = []
            if col1:
                merged += df1[col1].dropna().astype(str).tolist()
            if col2:
                merged += df2[col2].dropna().astype(str).tolist()

            if merged:
                if st.button("Análise Cruzada (IA)", key="an_cross"):
                    result = gemini_analyse(merged, "Análise Cruzada")
                    st.markdown(result)
                    pdf = generate_pdf("Análise Cruzada", result)
                    st.download_button("Baixar PDF", pdf, "cruzada.pdf")
            else:
                st.info("Nenhum texto relevante encontrado.")

    # ------------------- CHAT --------------------
    with tabs[3]:
        chat_with_gemini(df1, df2)

    # ------------------- RESET PASSWORD --------------------
    st.markdown("---")
    st.header("🔐 Redefinir senha")

    tok = st.text_input("Token")
    npw = st.text_input("Nova senha", type="password")

    if st.button("Redefinir"):
        ok, resp = validate_token(tok)
        if not ok:
            st.error(resp)
        else:
            user = resp
            hashed = PWD_CTX.hash(npw)
            st.success("Atualize no secrets.toml:")
            st.code(
                f'[users.{user}]\nname="{USERS[user]["name"]}"\nemail="{USERS[user]["email"]}"\npassword="{hashed}"',
                language="toml"
            )

# ----------------------------
# EXECUÇÃO GLOBAL
# ----------------------------
inject_css()
init_tokens()

if "logged" not in st.session_state:
    st.session_state.logged = False
if "page" not in st.session_state:
    st.session_state.page = "main"
if "show_recovery" not in st.session_state:
    st.session_state.show_recovery = False
if "_rerun" not in st.session_state:
    st.session_state._rerun = False

# Execução segura do rerun fora de dialogs
if st.session_state._rerun:
    st.session_state._rerun = False
    st.rerun()

# Fluxo de login
if not st.session_state.logged:
    if st.button("Entrar"):
        login_dialog()
    if st.session_state.show_recovery:
        recovery_dialog()
else:
    main_app()
