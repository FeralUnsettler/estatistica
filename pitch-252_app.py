# ================================================================
# REPARA ANALYTICS — v13.1
# Fix: no experimental_rerun inside dialogs. Full app (auth, gemini,
# chat, PDF, KPIs, admin, robust CSV reading).
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
# Config / Setup
# ----------------------------
st.set_page_config(page_title="Repara Analytics", layout="wide")

# Configure Gemini only if secret exists — avoid hard crash
if "GOOGLE_API_KEY" in st.secrets:
    try:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    except Exception:
        # proceed; calls to gemini will then raise at runtime with clearer message
        pass

# Use pbkdf2_sha256 for Streamlit Cloud compatibility
PWD_CTX = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
RESET_TOKEN_TTL = 15 * 60  # 15 minutes

# ----------------------------
# Helpers: load users from secrets.toml
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
# Token management (session)
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
# UI styling
# ----------------------------
def inject_css():
    st.markdown(
        """
    <style>
    .login-box { background: #ffffff; padding: 18px; border-radius: 10px; box-shadow: 0 6px 18px rgba(0,0,0,0.12); }
    .login-title { font-size: 20px; font-weight:700; color:#0b63ce; text-align:center; margin-bottom:8px; }
    </style>
    """,
        unsafe_allow_html=True,
    )

# ----------------------------
# Dialogs (no rerun inside)
# ----------------------------
@st.dialog("Login")
def login_dialog():
    inject_css()
    st.markdown("<div class='login-box'>", unsafe_allow_html=True)
    st.markdown("<div class='login-title'>REPARA — Login</div>", unsafe_allow_html=True)

    user = st.text_input("Usuário", key="login_user")
    pwd = st.text_input("Senha", type="password", key="login_pwd")

    if st.button("Entrar", key="login_btn"):
        ok, info = authenticate(user, pwd)
        if ok:
            st.session_state.logged = True
            st.session_state.userinfo = info
            st.session_state.page = "main"
            # set rerun flag; actual rerun happens outside the dialog
            st.session_state._rerun = True
            st.success("Login bem-sucedido.")
        else:
            st.error(info)

    if st.button("Esqueci a senha", key="login_forgot"):
        st.session_state.show_recovery = True
        st.session_state._rerun = True

    st.markdown("</div>", unsafe_allow_html=True)


@st.dialog("Recuperação de senha")
def recovery_dialog():
    inject_css()
    st.markdown("<div class='login-box'>", unsafe_allow_html=True)
    st.markdown("<div class='login-title'>Recuperação de senha</div>", unsafe_allow_html=True)

    username = st.text_input("Usuário para recuperação", key="recovery_user")
    if st.button("Gerar token", key="recovery_gen"):
        if username not in USERS:
            st.error("Usuário não encontrado.")
        else:
            token = generate_token(username)
            st.success("Token gerado (válido por 15 minutos).")
            st.info(f"Token: `{token}` — em produção envie por e-mail.")
            st.session_state._rerun = True

    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# CSV reading (robust, tries common delimiters and resets file pointer)
# ----------------------------
def read_csv_any(file):
    if file is None:
        return None

    # check size if available
    size = getattr(file, "size", None)
    if size == 0:
        st.warning("⚠ Arquivo enviado está vazio.")
        return None

    delimiters = [",", ";", "\t", "|"]
    for delim in delimiters:
        try:
            # rewind uploaded file if possible
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
        st.error("⚠ Não foi possível determinar o delimitador do CSV. Verifique se o arquivo é um CSV válido (use , ; \\t ou |).")
        return None

# ----------------------------
# Smart column inference
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
        "hr": find(["gestao", "rh", "motivo", "problema", "challenge", "recrut", "vaga"]),
    }

# ----------------------------
# Gemini analyze helper
# ----------------------------
def gemini_analyse(text_list, title="Análise"):
    if not text_list:
        return "Nenhum texto para análise."

    # ensure gemini configured
    if "GOOGLE_API_KEY" not in st.secrets:
        return "Gemini API key não encontrada em secrets. Configure GOOGLE_API_KEY."

    joined = "\n".join(str(t) for t in text_list)
    prompt = f"""
You are a senior data analyst. Produce a concise but structured analysis for the text below.

Title: {title}

Tasks:
1) Executive summary (one paragraph)
2) Top themes with short bullets
3) Emotions / sentiment overview
4) Actionable recommendations
5) A short table (Theme | Example | Impact | Action)

Text:
{joined}
"""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        resp = model.generate_content(prompt)
        return resp.text
    except Exception as e:
        return f"Erro ao chamar Gemini: {e}"

# ----------------------------
# Chat with Gemini (context from CSV previews)
# ----------------------------
def chat_with_gemini_context(df_cand, df_emp):
    st.header("💬 Chat com Gemini — pergunte sobre os dados")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Build a short context (limit rows)
    context = ""
    if df_cand is not None:
        context += "CANDIDATOS (preview):\n" + df_cand.head(10).to_csv(index=False) + "\n"
    if df_emp is not None:
        context += "EMPRESAS (preview):\n" + df_emp.head(10).to_csv(index=False) + "\n"

    # Show history
    for m in st.session_state.chat_history:
        role = "Você" if m["role"] == "user" else "IA"
        st.markdown(f"**{role}:** {m['text']}")
        st.markdown("---")

    user_q = st.text_input("Sua pergunta sobre os dados", key="chat_question")
    if st.button("Enviar pergunta", key="chat_send"):
        if not user_q.strip():
            st.warning("Escreva algo antes de enviar.")
            return
        st.session_state.chat_history.append({"role": "user", "text": user_q})

        if "GOOGLE_API_KEY" not in st.secrets:
            st.session_state.chat_history.append({"role":"assistant", "text":"Gemini não configurado em secrets."})
            st.experimental_rerun()

        prompt = f"""Você é um analista de dados. Use apenas o contexto abaixo para responder.

Contexto:
{context}

Pergunta:
{user_q}
"""

        try:
            model = genai.GenerativeModel("gemini-2.5-flash")
            resp = model.generate_content(prompt)
            ans = resp.text
        except Exception as e:
            ans = f"Erro ao chamar Gemini: {e}"

        st.session_state.chat_history.append({"role":"assistant","text":ans})
        # safe rerun, flag not required here; we can rerun to update UI
        st.experimental_rerun()

# ----------------------------
# PDF generation (ReportLab -> BytesIO)
# ----------------------------
def generate_pdf_bytes(title, markdown_text):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1, 12)]
    for line in markdown_text.split("\n"):
        if line.strip() == "":
            story.append(Spacer(1,6))
        else:
            story.append(Paragraph(line.replace("&","and"), styles["Normal"]))
    doc.build(story)
    buffer.seek(0)
    return buffer

def pdf_download_button(report_text, title="Relatório REPARA"):
    st.subheader("📄 Relatório (PDF)")
    if st.button("Gerar e baixar PDF", key=f"pdf_{hash(report_text) % 10_000}"):
        buf = generate_pdf_bytes(title, report_text)
        st.download_button("📥 Baixar PDF", data=buf, file_name="relatorio_repara.pdf", mime="application/pdf")

# ----------------------------
# KPIs
# ----------------------------
def dashboard_kpis(df_cand, df_emp):
    st.header("📊 KPIs")
    c1, c2, c3 = st.columns(3)
    if df_cand is not None:
        c1.metric("Candidatos", len(df_cand))
        c2.metric("Colunas (Candidatos)", len(df_cand.columns))
    else:
        c1.metric("Candidatos", "—"); c2.metric("Colunas (Candidatos)", "—")
    if df_emp is not None:
        c3.metric("Empresas", len(df_emp))
    else:
        c3.metric("Empresas", "—")

# ----------------------------
# Admin UI (generates TOML blocks for secrets)
# ----------------------------
def admin_panel_ui():
    st.title("🛡️ Painel Admin")
    st.info("Gere blocos TOML prontos para colar no Streamlit Secrets (secrets.toml).")
    st.subheader("Usuários registrados")
    for u, info in USERS.items():
        st.markdown(f"- **{u}** — {info.get('email')}")
    st.markdown("---")
    st.subheader("Criar novo usuário (gera bloco TOML)")
    nu = st.text_input("Username", key="admin_nu")
    nm = st.text_input("Nome completo", key="admin_nm")
    em = st.text_input("Email", key="admin_em")
    pw = st.text_input("Senha (gera hash)", type="password", key="admin_pw")
    if st.button("Gerar bloco TOML", key="admin_gen"):
        if not nu or not pw:
            st.error("username e senha são obrigatórios.")
        else:
            h = PWD_CTX.hash(pw)
            st.success("Copie e cole o bloco abaixo em secrets.toml")
            st.code(f'[users.{nu}]\nname = "{nm}"\nemail = "{em}"\npassword = "{h}"', language="toml")
    st.markdown("---")
    st.subheader("Gerar hash isolado")
    ph = st.text_input("Senha para hash", type="password", key="admin_hash_pw")
    if st.button("Gerar hash isolado", key="admin_hash_btn"):
        if not ph:
            st.error("Digite a senha.")
        else:
            st.code(PWD_CTX.hash(ph))

# ----------------------------
# Main app (pages & flow)
# ----------------------------
def main_app():
    st.title("📊 REPARA Analytics — v13.1")

    st.sidebar.success(f"Usuário: {st.session_state.userinfo.get('name')}")
    if st.sidebar.button("Painel Admin", key="sidebar_admin"):
        st.session_state.page = "admin"
        st.experimental_rerun()
    if st.sidebar.button("Sair", key="sidebar_logout"):
        st.session_state.logged = False
        st.experimental_rerun()

    if st.session_state.page == "admin":
        # restrict admin access by email (configurable)
        if st.session_state.userinfo.get("email") != "admin@repara.com":
            st.error("Acesso restrito ao administrador.")
            return
        admin_panel_ui()
        return

    st.sidebar.header("📥 Upload CSVs")
    cand_file = st.sidebar.file_uploader("Candidatos (CSV)", type=["csv"], key="up_cand")
    emp_file = st.sidebar.file_uploader("Empresas (CSV)", type=["csv"], key="up_emp")

    df_cand = read_csv_any(cand_file) if cand_file else None
    df_emp = read_csv_any(emp_file) if emp_file else None

    dashboard_kpis(df_cand, df_emp)

    tabs = st.tabs(["👤 Candidatos", "🏢 Empresas", "🔀 Cruzada", "💬 Chat IA"])

    # Candidates tab
    with tabs[0]:
        st.header("👤 Candidatos")
        if df_cand is not None:
            st.dataframe(df_cand)
            cols = infer_cols(df_cand)
            col_name = cols.get("pain")
            if col_name:
                st.subheader(f"Campo detectado: {col_name}")
                text = " ".join(df_cand[col_name].dropna().astype(str))
                if text:
                    wc = WordCloud(width=900, height=400).generate(text)
                    fig, ax = plt.subplots(figsize=(10,4))
                    ax.imshow(wc); ax.axis("off")
                    st.pyplot(fig)
                if st.button("IA — Analisar candidatos", key="analyze_cand"):
                    result = gemini_analyse(df_cand[col_name].dropna().tolist(), title="Candidatos")
                    st.markdown(result)
                    pdf_download_button(result, title="Análise Candidatos")
            else:
                st.info("Nenhuma coluna textual detectada automaticamente.")
        else:
            st.info("Envie o CSV de candidatos pela sidebar.")

    # Companies tab
    with tabs[1]:
        st.header("🏢 Empresas")
        if df_emp is not None:
            st.dataframe(df_emp)
            cols = infer_cols(df_emp)
            col_name = cols.get("hr")
            if col_name:
                st.subheader(f"Campo detectado: {col_name}")
                try:
                    top = df_emp[col_name].dropna().astype(str).value_counts().head(10)
                    fig, ax = plt.subplots()
                    top.plot(kind="barh", ax=ax)
                    st.pyplot(fig)
                except Exception:
                    st.info("Não foi possível plotar o gráfico.")
                if st.button("IA — Analisar empresas", key="analyze_emp"):
                    result = gemini_analyse(df_emp[col_name].dropna().tolist(), title="Empresas")
                    st.markdown(result)
                    pdf_download_button(result, title="Análise Empresas")
            else:
                st.info("Nenhuma coluna textual detectada automaticamente.")
        else:
            st.info("Envie o CSV de empresas pela sidebar.")

    # Cross analysis tab
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
                if st.button("IA — Análise cruzada", key="analyze_cross"):
                    result = gemini_analyse(texts, title="Análise Cruzada")
                    st.markdown(result)
                    pdf_download_button(result, title="Análise Cruzada")

    # Chat tab
    with tabs[3]:
        chat_with_gemini_context(df_cand, df_emp)

    # Reset password footer
    st.markdown("---")
    st.header("🔐 Redefinir senha (se já possui token)")
    token_val = st.text_input("Token de recuperação", key="reset_token_input")
    new_password = st.text_input("Nova senha", type="password", key="reset_new_pw_input")
    if st.button("Redefinir senha", key="reset_submit"):
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
# Execution flow
# ----------------------------
inject_css()
init_tokens()

# Ensure session keys
if "logged" not in st.session_state:
    st.session_state.logged = False
if "page" not in st.session_state:
    st.session_state.page = "main"
if "show_recovery" not in st.session_state:
    st.session_state.show_recovery = False
if "_rerun" not in st.session_state:
    st.session_state._rerun = False

# If flagged by dialog, do rerun outside dialog (safe)
if st.session_state.get("_rerun", False):
    st.session_state._rerun = False
    # use st.rerun (stable) to refresh after dialog closed
    st.rerun()

# Show login / recovery flow
if not st.session_state.logged:
    if st.button("Entrar", key="open_login"):
        login_dialog()
    if st.session_state.show_recovery:
        recovery_dialog()
else:
    main_app()
