# REPARA ANALYTICS — v13.3 (Refatorado: detecção textual robusta + seleção manual)
# Substitua seu app.py por este arquivo.

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
import unicodedata
import re

# ----------------------------
# Configurações
# ----------------------------
st.set_page_config(page_title="Repara Analytics", layout="wide")

# Configure Gemini (opcional; app funciona sem chave, apenas IA retorna erro)
if "GOOGLE_API_KEY" in st.secrets:
    try:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    except Exception:
        pass

PWD_CTX = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
RESET_TOKEN_TTL = 15 * 60

# ----------------------------
# Util: carregar usuários do secrets.toml
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
# Tokens
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
    st.markdown("""
    <style>
    .login-box { background: #ffffff; padding: 18px; border-radius: 10px; box-shadow: 0 6px 18px rgba(0,0,0,0.12); }
    .login-title { font-size: 20px; font-weight:700; color:#0b63ce; text-align:center; margin-bottom:8px; }
    </style>
    """, unsafe_allow_html=True)

# ----------------------------
# Dialogs
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

    username = st.text_input("Usuário para recuperação", key="rec_user")
    if st.button("Gerar token", key="rec_gen"):
        if username not in USERS:
            st.error("Usuário não encontrado.")
        else:
            token = generate_token(username)
            st.success("Token gerado (15 min).")
            st.info(f"Token: `{token}` — em produção envie por e-mail.")
            st.session_state._rerun = True

    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# CSV robust reader (delimiters)
# ----------------------------
def read_csv_any(file):
    if file is None:
        return None
    try:
        size = getattr(file, "size", None)
        if size == 0:
            st.warning("Arquivo vazio.")
            return None
    except Exception:
        pass

    delims = [",", ";", "\t", "|"]
    for d in delims:
        try:
            if hasattr(file, "seek"):
                try: file.seek(0)
                except: pass
            df = pd.read_csv(file, sep=d, engine="python")
            # normalize columns names by stripping
            df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
            if not df.empty:
                return df
        except Exception:
            continue
    # last attempt
    try:
        if hasattr(file, "seek"):
            try: file.seek(0)
            except: pass
        df = pd.read_csv(file, sep=None, engine="python")
        df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
        if not df.empty:
            return df
    except Exception as e:
        st.error("Erro ao ler CSV. Verifique formato e delimitador.")
        return None

# ----------------------------
# Normalização util (remove acentos, lower)
# ----------------------------
def normalize_colname(s):
    s = s.strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s)
    return s

# ----------------------------
# Detect text columns robustly (returns sorted list by score)
# ----------------------------
def detect_text_columns(df, min_pct_text=0.10):
    """
    Returns list of column names likely to contain free-text responses.
    Criterion: % of rows containing alphabetic character (Portuguese included).
    """
    candidates = []
    n = len(df)
    if n == 0:
        return []

    for col in df.columns:
        # convert to string, fillna
        ser = df[col].astype(str).fillna("")
        # percent of cells with at least one letter (A-Za-z + accented)
        pct_text = ser.str.contains(r"[A-Za-zÀ-ÿ]", regex=True).mean()
        # unique values proportion (helps distinguish id columns)
        unique_prop = ser.nunique() / max(1, n)
        # average length
        avg_len = ser.str.len().mean()
        # score: weighted
        score = (pct_text * 0.6) + (min(1.0, avg_len/30) * 0.3) + (min(1.0, unique_prop) * 0.1)
        # consider candidate if pct_text >= threshold OR avg_len > 20
        if pct_text >= min_pct_text or avg_len >= 20:
            candidates.append((col, score, pct_text, avg_len, unique_prop))

    # sort by score desc
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates  # list of tuples (col, score, pct_text, avg_len, unique_prop)

# ----------------------------
# Gemini analysis helper
# ----------------------------
def gemini_analyse(text_list, title="Análise"):
    if not text_list:
        return "Nenhum texto para análise."
    if "GOOGLE_API_KEY" not in st.secrets:
        return "Gemini API não configurada em secrets (GOOGLE_API_KEY)."
    prompt = f"""
Analise este conjunto de respostas e entregue:
- Resumo executivo (curto)
- Principais temas
- Emoções/sentimentos
- Tabela: Tema | Exemplo | Impacto | Ação
- Recomendações práticas

Título: {title}

Texto:
{chr(10).join(map(str, text_list))}
"""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        out = model.generate_content(prompt)
        return out.text
    except Exception as e:
        return f"Erro ao chamar Gemini: {e}"

# ----------------------------
# PDF generation
# ----------------------------
def generate_pdf_bytes(title, text):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1,12)]
    for line in text.split("\n"):
        if line.strip():
            story.append(Paragraph(line.replace("&","and"), styles["Normal"]))
        else:
            story.append(Spacer(1,6))
    doc.build(story)
    buffer.seek(0)
    return buffer

# ----------------------------
# Chat with Gemini (context limited)
# ----------------------------
def chat_with_gemini_context(df_cand, df_emp):
    st.header("💬 Chat com Gemini — pergunte sobre os dados")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    # build short context
    context = ""
    if df_cand is not None:
        context += "Candidatos (preview):\n" + df_cand.head(8).to_csv(index=False) + "\n"
    if df_emp is not None:
        context += "Empresas (preview):\n" + df_emp.head(8).to_csv(index=False) + "\n"
    # show history
    for msg in st.session_state.chat_history:
        role = "Você" if msg["role"]=="user" else "IA"
        st.markdown(f"**{role}:** {msg['text']}")
    user_q = st.text_input("Pergunta sobre os dados", key="chat_input")
    if st.button("Enviar pergunta", key="chat_send"):
        if not user_q.strip():
            st.warning("Escreva algo primeiro.")
            return
        st.session_state.chat_history.append({"role":"user","text":user_q})
        if "GOOGLE_API_KEY" not in st.secrets:
            st.session_state.chat_history.append({"role":"assistant","text":"Gemini não configurado."})
            st.experimental_rerun()
        prompt = f"Contexto:\n{context}\nPergunta: {user_q}"
        try:
            model = genai.GenerativeModel("gemini-2.5-flash")
            resp = model.generate_content(prompt)
            ans = resp.text
        except Exception as e:
            ans = f"Erro: {e}"
        st.session_state.chat_history.append({"role":"assistant","text":ans})
        st.experimental_rerun()

# ----------------------------
# Admin panel UI (generate TOML)
# ----------------------------
def admin_panel_ui():
    st.title("🛡️ Painel Admin")
    st.info("Gera blocos TOML para colar em Streamlit Secrets.")
    st.subheader("Usuários atuais")
    for u, info in USERS.items():
        st.markdown(f"- **{u}** — {info.get('email')}")
    st.markdown("---")
    st.subheader("Criar novo usuário")
    nu = st.text_input("Username", key="admin_nu")
    nm = st.text_input("Nome completo", key="admin_nm")
    em = st.text_input("Email", key="admin_em")
    pw = st.text_input("Senha (gera hash)", type="password", key="admin_pw")
    if st.button("Gerar bloco TOML", key="admin_gen"):
        if not nu or not pw:
            st.error("Preencha username e senha.")
        else:
            h = PWD_CTX.hash(pw)
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
# KPIs
# ----------------------------
def dashboard_kpis(df_cand, df_emp):
    st.header("📊 KPIs Rápidos")
    c1,c2,c3 = st.columns(3)
    c1.metric("Candidatos", len(df_cand) if df_cand is not None else "—")
    c2.metric("Colunas (candidatos)", len(df_cand.columns) if df_cand is not None else "—")
    c3.metric("Empresas", len(df_emp) if df_emp is not None else "—")

# ----------------------------
# MAIN APP
# ----------------------------
def main_app():
    st.title("📊 REPARA Analytics — v13.3")

    st.sidebar.success(f"Usuário: {st.session_state.userinfo.get('name')}")
    if st.sidebar.button("Painel Admin"):
        st.session_state.page = "admin"
        st.session_state._rerun = True
    if st.sidebar.button("Sair"):
        st.session_state.logged = False
        st.session_state._rerun = True

    if st.session_state.page == "admin":
        if st.session_state.userinfo.get("email") != "admin@repara.com":
            st.error("Acesso restrito ao admin.")
            return
        admin_panel_ui()
        return

    st.sidebar.header("📥 Upload CSVs")
    cand_file = st.sidebar.file_uploader("Candidatos (CSV)", type=["csv"])
    emp_file = st.sidebar.file_uploader("Empresas (CSV)", type=["csv"])

    df_cand = read_csv_any(cand_file) if cand_file else None
    df_emp = read_csv_any(emp_file) if emp_file else None

    dashboard_kpis(df_cand, df_emp)

    tabs = st.tabs(["👤 Candidatos", "🏢 Empresas", "🔀 Cruzada", "💬 Chat IA"])

    # CANDIDATOS TAB
    with tabs[0]:
        st.header("👤 Candidatos")
        if df_cand is None:
            st.info("Envie o CSV de candidatos pela sidebar.")
        else:
            st.dataframe(df_cand.head(50))
            # detect text columns
            candidates = detect_text_columns(df_cand)
            if candidates:
                st.subheader("Colunas textuais detectadas (ordem por score):")
                for col, score, pct_text, avg_len, uniq in candidates:
                    st.markdown(f"- **{col}** — score {score:.2f} — pct_text {pct_text:.2f} — avg_len {avg_len:.1f}")
                default_col = candidates[0][0]
            else:
                st.warning("Nenhuma coluna textual detectada automaticamente.")
                default_col = None

            # allow manual selection (always present)
            st.markdown("**Selecione a coluna que contém respostas textuais (p/ análise IA / wordcloud):**")
            col_choice = st.selectbox("Coluna textual", options=[None] + list(df_cand.columns), index=0)
            if col_choice is None and default_col:
                col_choice = default_col

            if col_choice:
                st.subheader(f"Preview — coluna: {col_choice}")
                st.dataframe(df_cand[[col_choice]].head(10))
                # prepare text
                text_series = df_cand[col_choice].dropna().astype(str)
                text_joined = " ".join(text_series.tolist())
                if len(text_joined.strip()) < 10:
                    st.info("Pouco texto nesta coluna — talvez não haja conteúdo para IA.")
                else:
                    if st.button("Gerar wordcloud (candidatos)"):
                        wc = WordCloud(width=900, height=400).generate(text_joined)
                        fig, ax = plt.subplots(figsize=(10,4))
                        ax.imshow(wc); ax.axis("off")
                        st.pyplot(fig)

                    if st.button("IA — Analisar candidatos"):
                        result = gemini_analyse(text_series.tolist(), title="Candidatos")
                        st.markdown(result)
                        buf = generate_pdf_bytes("Análise Candidatos", result)
                        st.download_button("📥 Baixar PDF (Candidatos)", data=buf, file_name="analise_candidatos.pdf", mime="application/pdf")
            else:
                st.info("Selecione uma coluna para ativar a análise IA.")

    # EMPRESAS TAB
    with tabs[1]:
        st.header("🏢 Empresas")
        if df_emp is None:
            st.info("Envie o CSV de empresas pela sidebar.")
        else:
            st.dataframe(df_emp.head(50))
            candidates_e = detect_text_columns(df_emp)
            if candidates_e:
                st.subheader("Colunas textuais detectadas (empresas):")
                for col, score, pct_text, avg_len, uniq in candidates_e:
                    st.markdown(f"- **{col}** — score {score:.2f}")
                default_col_e = candidates_e[0][0]
            else:
                st.warning("Nenhuma coluna textual detectada automaticamente.")
                default_col_e = None

            st.markdown("**Selecione coluna textual (empresas):**")
            col_choice_e = st.selectbox("Coluna empresas", options=[None] + list(df_emp.columns), index=0)
            if col_choice_e is None and default_col_e:
                col_choice_e = default_col_e

            if col_choice_e:
                st.subheader(f"Preview — coluna: {col_choice_e}")
                st.dataframe(df_emp[[col_choice_e]].head(10))
                if st.button("Analisar empresas (IA)"):
                    result = gemini_analyse(df_emp[col_choice_e].dropna().astype(str).tolist(), title="Empresas")
                    st.markdown(result)
                    buf = generate_pdf_bytes("Análise Empresas", result)
                    st.download_button("📥 Baixar PDF (Empresas)", data=buf, file_name="analise_empresas.pdf", mime="application/pdf")
            else:
                st.info("Selecione uma coluna para ativar a análise IA.")

    # CRUZADA TAB
    with tabs[2]:
        st.header("🔀 Análise Cruzada")
        if df_cand is None or df_emp is None:
            st.info("Envie ambos os CSVs (candidatos e empresas).")
        else:
            # ask user which columns to cross
            cand_cands = detect_text_columns(df_cand)
            emp_cands = detect_text_columns(df_emp)
            default_c = cand_cands[0][0] if cand_cands else None
            default_e = emp_cands[0][0] if emp_cands else None
            st.markdown("Selecione coluna textual (candidatos) para cruzamento:")
            ccol = st.selectbox("Candidato (coluna)", options=[None] + list(df_cand.columns), index=0)
            if ccol is None and default_c:
                ccol = default_c
            st.markdown("Selecione coluna textual (empresas) para cruzamento:")
            ecol = st.selectbox("Empresa (coluna)", options=[None] + list(df_emp.columns), index=0)
            if ecol is None and default_e:
                ecol = default_e

            texts = []
            if ccol:
                texts += df_cand[ccol].dropna().astype(str).tolist()
            if ecol:
                texts += df_emp[ecol].dropna().astype(str).tolist()

            if texts:
                if st.button("IA — Análise Cruzada"):
                    result = gemini_analyse(texts, title="Análise Cruzada")
                    st.markdown(result)
                    buf = generate_pdf_bytes("Análise Cruzada", result)
                    st.download_button("📥 Baixar PDF (Cruzada)", data=buf, file_name="analise_cruzada.pdf", mime="application/pdf")
            else:
                st.info("Selecione colunas textuais válidas para cruzar.")

    # CHAT TAB
    with tabs[3]:
        st.header("💬 Chat com Gemini")
        chat_with_gemini_context = globals().get("chat_with_gemini_context")
        # reuse previously defined chat_with_gemini_context if present (keeps compatibility)
        if callable(chat_with_gemini_context):
            chat_with_gemini_context(df_cand, df_emp)
        else:
            st.info("Chat não disponível.")

    # Reset password footer
    st.markdown("---")
    st.header("🔐 Redefinir senha (se tiver token)")
    token_val = st.text_input("Token de recuperação", key="reset_token_v13_3")
    new_password = st.text_input("Nova senha", type="password", key="reset_new_pw_v13_3")
    if st.button("Redefinir senha", key="reset_submit_v13_3"):
        if not token_val:
            st.error("Informe token.")
        else:
            ok, resp = validate_token(token_val)
            if not ok:
                st.error(resp)
            else:
                username = resp
                hashed = PWD_CTX.hash(new_password)
                st.success("Senha atualizada — cole o bloco abaixo no secrets.toml")
                st.code(f'[users.{username}]\nname = "{USERS[username]["name"]}"\nemail = "{USERS[username]["email"]}"\npassword = "{hashed}"', language="toml")

# ----------------------------
# Execução
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

if st.session_state._rerun:
    st.session_state._rerun = False
    st.rerun()

if not st.session_state.logged:
    if st.button("Entrar"):
        login_dialog()
    if st.session_state.show_recovery:
        recovery_dialog()
else:
    main_app()
