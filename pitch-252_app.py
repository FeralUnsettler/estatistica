# ================================================================
# REPARA ANALYTICS — v13.0
# Completo: Auth (secrets.toml) + Gemini Chat + PDF + KPIs + CSV robusto
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
# CONFIGURAÇÃO INICIAL
# ----------------------------
st.set_page_config(page_title="Repara Analytics", layout="wide")
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# use pbkdf2_sha256 for compatibility on Streamlit Cloud
PWD_CTX = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
RESET_TOKEN_TTL = 15 * 60  # 15 minutos

# ----------------------------
# UTIL: carregar users do secrets.toml
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
# Tokens de recovery (session-state)
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
# Auth helpers
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
    .login-box { background: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 6px 18px rgba(0,0,0,0.12); }
    .login-title { font-size: 22px; font-weight:700; color:#0b63ce; text-align:center; }
    </style>
    """, unsafe_allow_html=True)

# ----------------------------
# DIALOGS (não aninhar)
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
            st.success("Login bem-sucedido.")
            st.experimental_rerun()
        else:
            st.error(info)

    if st.button("Esqueci a senha"):
        # não abre nested dialog: sinaliza para abrir fora
        st.session_state.show_recovery = True

    st.markdown("</div>", unsafe_allow_html=True)

@st.dialog("Recuperação de senha")
def recovery_dialog():
    st.markdown("<div class='login-box'>", unsafe_allow_html=True)
    st.markdown("<div class='login-title'>Recuperação de senha</div>", unsafe_allow_html=True)
    username = st.text_input("Usuário para gerar token", key="recovery_user")
    if st.button("Gerar token"):
        if username not in USERS:
            st.error("Usuário não encontrado.")
        else:
            token = generate_token(username)
            st.success("Token gerado (válido por 15 minutos).")
            st.info(f"Token: `{token}` — em produção, envie por e-mail.")
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Leitura CSV robusta (detecta delimitador)
# ----------------------------
def read_csv_any(file):
    if file is None:
        return None

    # file may be a UploadedFile or file-like
    try:
        size = getattr(file, "size", None)
        if size == 0:
            st.warning("⚠ Arquivo vazio enviado.")
            return None
    except Exception:
        pass

    delimiters = [",", ";", "\t", "|"]
    # try each delimiter: but we must rewind buffer between attempts if file supports seek
    for delim in delimiters:
        try:
            if hasattr(file, "seek"):
                try:
                    file.seek(0)
                except Exception:
                    pass
            df = pd.read_csv(file, sep=delim)
            if not df.empty:
                return df
        except Exception:
            continue

    # final attempt with python engine autodetect
    try:
        if hasattr(file, "seek"):
            try:
                file.seek(0)
            except Exception:
                pass
        df = pd.read_csv(file, sep=None, engine="python")
        if not df.empty:
            return df
    except pd.errors.EmptyDataError:
        st.error("⚠ CSV vazio ou corrompido.")
        return None
    except Exception:
        st.error("⚠ Não foi possível determinar o delimitador do CSV. Abra o arquivo e verifique o formato (use , ; \\t ou |).")
        return None

# ----------------------------
# Inferência de colunas textuais (robusta)
# ----------------------------
def infer_cols(df):
    lower = {col.lower(): col for col in df.columns}
    def find(keys):
        for k in keys:
            for low, real in lower.items():
                if k in low:
                    return real
        return None
    return {
        "pain": find(["feedback", "dor", "coment", "comentário", "descricao", "descrição", "texto", "observacao", "obs"]),
        "hr": find(["gestao", "rh", "motivo", "problema", "challenge", "hr", "recrut"])
    }

# ----------------------------
# Gemini analysis helper
# ----------------------------
def gemini_analyse(text_list, title="Análise"):
    if not text_list:
        return "Nenhum texto para análise."
    joined = "\n".join(str(t) for t in text_list)
    prompt = f"""
You are a senior data analyst. Provide a concise structured response for the text below:

Title: {title}

Tasks:
1) Executive summary (short)
2) Top themes (list)
3) Emotions / sentiment summary
4) Actionable recommendations
5) A table (Theme | Example | Impact | Suggested Action)

Text:
{joined}
"""
    model = genai.GenerativeModel("gemini-2.5-flash")
    resp = model.generate_content(prompt)
    return resp.text

# ----------------------------
# Chat with Gemini (session chat)
# ----------------------------
def chat_with_gemini_context(df_cand, df_emp):
    st.header("🧠 Chat com Gemini — pergunte sobre os dados")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Build concise context (markdown) — limit to 10 rows each
    context = ""
    if df_cand is not None:
        context += "### Candidatos (preview)\n"
        context += df_cand.head(10).to_markdown() + "\n\n"
    if df_emp is not None:
        context += "### Empresas (preview)\n"
        context += df_emp.head(10).to_markdown() + "\n\n"

    # show history
    for i, msg in enumerate(st.session_state.chat_history):
        role = "Você" if msg["role"] == "user" else "IA"
        st.markdown(f"**{role}:** {msg['text']}")
        if i < len(st.session_state.chat_history)-1:
            st.markdown("---")

    user_q = st.text_input("Digite sua pergunta sobre os dados", key="chat_input")
    if st.button("Enviar pergunta"):
        if not user_q.strip():
            return
        st.session_state.chat_history.append({"role":"user","text":user_q})
        # prepare prompt with context + question
        prompt = f"""Você é um analista de dados. Baseie-se apenas no contexto fornecido abaixo (trechos dos CSVs) e responda objetivamente.

Contexto:
{context}

Pergunta:
{user_q}
"""
        model = genai.GenerativeModel("gemini-2.5-flash")
        resp = model.generate_content(prompt)
        ans = resp.text
        st.session_state.chat_history.append({"role":"assistant","text":ans})
        st.experimental_rerun()

# ----------------------------
# PDF generation (ReportLab -> BytesIO)
# ----------------------------
def generate_pdf_bytes(title, markdown_text):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 12))
    # split markdown_text reasonably
    for line in markdown_text.split("\n"):
        if line.strip() == "":
            story.append(Spacer(1,6))
        else:
            # escape if too long
            story.append(Paragraph(line.replace("&","and"), styles["Normal"]))
    doc.build(story)
    buffer.seek(0)
    return buffer

# ----------------------------
# PDF download helper UI
# ----------------------------
def pdf_download_button(report_text, title="Relatório REPARA"):
    st.subheader("📄 Relatório PDF")
    if st.button("Gerar PDF do relatório"):
        pdf_bytes = generate_pdf_bytes(title, report_text)
        st.download_button(
            label="📥 Baixar relatório (PDF)",
            data=pdf_bytes,
            file_name="relatorio_repara.pdf",
            mime="application/pdf"
        )

# ----------------------------
# KPIs dashboard
# ----------------------------
def dashboard_kpis(df_cand, df_emp):
    st.header("📊 KPIs Rápidos")
    c1,c2,c3 = st.columns(3)
    if df_cand is not None:
        c1.metric("Candidatos", len(df_cand))
        c2.metric("Colunas Candidatos", len(df_cand.columns))
    else:
        c1.metric("Candidatos", "—")
        c2.metric("Colunas Candidatos", "—")
    if df_emp is not None:
        c3.metric("Empresas", len(df_emp))
    else:
        c3.metric("Empresas", "—")

# ----------------------------
# Admin panel (generate TOML blocks)
# ----------------------------
def admin_panel_ui():
    st.title("🛡️ Painel Admin")
    st.info("Alterações no secrets.toml devem ser feitas manualmente no Streamlit Cloud. Aqui você gera blocos TOML prontos.")
    st.subheader("Usuários registrados")
    for u, info in USERS.items():
        st.markdown(f"- **{u}** — {info['email']}")
    st.markdown("---")
    st.subheader("Criar novo usuário (gera bloco TOML)")
    nu = st.text_input("Username (ex: luciano)", key="admin_new_user")
    nm = st.text_input("Nome completo", key="admin_new_name")
    em = st.text_input("Email", key="admin_new_email")
    pw = st.text_input("Senha (gera hash)", type="password", key="admin_new_pw")
    if st.button("Gerar bloco TOML"):
        if not nu or not pw:
            st.error("Preencha username e senha.")
        else:
            h = PWD_CTX.hash(pw)
            st.success("Copie o bloco e cole no secrets.toml")
            st.code(f'[users.{nu}]\nname = "{nm}"\nemail = "{em}"\npassword = "{h}"', language="toml")
    st.markdown("---")
    st.subheader("Gerar hash isolado")
    ph = st.text_input("Senha para hash", type="password", key="admin_hash_pw")
    if st.button("Gerar hash isolado"):
        if not ph:
            st.error("Digite uma senha.")
        else:
            st.code(PWD_CTX.hash(ph))

# ----------------------------
# APP MAIN
# ----------------------------
def main_app():
    st.title("📊 REPARA Analytics — v13.0")

    st.sidebar.success(f"Usuário: {st.session_state.userinfo['name']}")
    if st.sidebar.button("Painel Admin"):
        st.session_state.page = "admin"
        st.experimental_rerun()
    if st.sidebar.button("Sair"):
        st.session_state.logged = False
        st.experimental_rerun()

    # page routing
    if st.session_state.page == "admin":
        # restrict admin by email
        if st.session_state.userinfo.get("email") != "admin@repara.com":
            st.error("Acesso restrito ao administrador.")
            return
        admin_panel_ui()
        return

    # Uploads
    st.sidebar.header("📥 Upload CSVs")
    cand_file = st.sidebar.file_uploader("Candidatos (CSV)", type=["csv"])
    emp_file = st.sidebar.file_uploader("Empresas (CSV)", type=["csv"])

    df_cand = read_csv_any(cand_file) if cand_file else None
    df_emp = read_csv_any(emp_file) if emp_file else None

    dashboard_kpis(df_cand, df_emp)

    tabs = st.tabs(["👤 Candidatos", "🏢 Empresas", "🔀 Análise Cruzada", "💬 Chat IA"])

    # --- Candidatos tab
    with tabs[0]:
        st.header("👤 Candidatos")
        if df_cand is not None:
            st.dataframe(df_cand)
            cols = infer_cols(df_cand)
            if cols.get("pain"):
                st.subheader("☁️ Wordcloud (campo detectado: %s)" % cols["pain"])
                text = " ".join(df_cand[cols["pain"]].dropna().astype(str))
                if text:
                    wc = WordCloud(width=900, height=400).generate(text)
                    fig, ax = plt.subplots(figsize=(10,4))
                    ax.imshow(wc); ax.axis("off")
                    st.pyplot(fig)
                if st.button("IA — analisar candidatos"):
                    result = gemini_analyse(df_cand[cols["pain"]].dropna().tolist(), title="Candidatos")
                    st.markdown(result)
                    pdf_download_button(result, title="Análise Candidatos")
            else:
                st.info("Nenhuma coluna textual detectada automaticamente. Renomeie/indique coluna com feedbacks.")
        else:
            st.info("Envie o CSV de candidatos pela sidebar.")

    # --- Empresas tab
    with tabs[1]:
        st.header("🏢 Empresas")
        if df_emp is not None:
            st.dataframe(df_emp)
            cols = infer_cols(df_emp)
            if cols.get("hr"):
                st.subheader("Top desafios (campo detectado: %s)" % cols["hr"])
                try:
                    top = df_emp[cols["hr"]].dropna().astype(str).value_counts().head(10)
                    fig, ax = plt.subplots()
                    top.plot(kind="barh", ax=ax)
                    st.pyplot(fig)
                except Exception:
                    st.info("Não foi possível plotar top — verifique dados.")
                if st.button("IA — analisar empresas"):
                    result = gemini_analyse(df_emp[cols["hr"]].dropna().tolist(), title="Empresas")
                    st.markdown(result)
                    pdf_download_button(result, title="Análise Empresas")
            else:
                st.info("Nenhuma coluna textual detectada automaticamente nas empresas.")
        else:
            st.info("Envie o CSV de empresas pela sidebar.")

    # --- Cruzada
    with tabs[2]:
        st.header("🔀 Análise Cruzada")
        if df_cand is None or df_emp is None:
            st.info("Envie ambos os CSVs (candidatos e empresas) para análise cruzada.")
        else:
            cols1 = infer_cols(df_cand)
            cols2 = infer_cols(df_emp)
            texts = []
            if cols1.get("pain"):
                texts += df_cand[cols1["pain"]].dropna().tolist()
            if cols2.get("hr"):
                texts += df_emp[cols2["hr"]].dropna().tolist()
            if not texts:
                st.info("Não foram detectadas colunas textuais para cruzamento.")
            else:
                if st.button("IA — Análise cruzada"):
                    result = gemini_analyse(texts, title="Cruzada")
                    st.markdown(result)
                    pdf_download_button(result, title="Análise Cruzada")

    # --- Chat IA
    with tabs[3]:
        st.header("💬 Chat com Gemini")
        st.info("Pergunte sobre o conteúdo dos CSVs (contexto limitado aos previews).")
        chat_with_gemini_context(df_cand, df_emp)

    # footer: reset password area
    st.markdown("---")
    st.header("🔐 Redefinir senha (se já possui token)")
    token_val = st.text_input("Token de recuperação", key="reset_token")
    new_password = st.text_input("Nova senha", type="password", key="reset_new_pw")
    if st.button("Redefinir senha"):
        if not token_val:
            st.error("Informe o token.")
        else:
            ok, resp = validate_token(token_val)
            if not ok:
                st.error(resp)
            else:
                username = resp
                hashed = PWD_CTX.hash(new_password)
                st.success("Senha atualizada — copie o bloco abaixo para o secrets.toml:")
                st.code(f'[users.{username}]\nname = "{USERS[username]["name"]}"\nemail = "{USERS[username]["email"]}"\npassword = "{hashed}"', language="toml")

# ----------------------------
# Execução
# ----------------------------
inject_css()
init_tokens = init_tokens  # keep name consistent
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
