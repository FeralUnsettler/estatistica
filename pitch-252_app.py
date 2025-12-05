# REPARA Analytics — v13.4.3 (Streamlit Cloud compatible)
# - Modal login (st.modal)
# - Wordcloud inteligente (NLTK heuristics, sentiment lexicon)
# - Admin panel, CSV robust read, Gemini optional
# Copy as app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import io
import time
import secrets
from passlib.context import CryptContext
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet

# Optional AI
try:
    import google.generativeai as genai
except Exception:
    genai = None

# NLP lightweight
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords", quiet=True)

# ----------------------------
# Config / constants
# ----------------------------
st.set_page_config(page_title="REPARA Analytics v13.4.3", layout="wide")
PWD_CTX = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
RESET_TOKEN_TTL = 15 * 60  # 15 minutes

# If API key present in secrets, configure Gemini (safe try)
if "GOOGLE_API_KEY" in st.secrets and genai is not None:
    try:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    except Exception:
        pass

# ----------------------------
# Helpers: load users from secrets
# ----------------------------
def load_users():
    raw = st.secrets.get("users", {}) or {}
    users = {}
    for uname, info in raw.items():
        users[uname] = {
            "name": info.get("name"),
            "email": info.get("email"),
            "password": info.get("password"),
        }
    return users

USERS = load_users()

# ----------------------------
# Token management (in-session)
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

def authenticate_by_email(email, password):
    # search users by email field
    for uname, info in USERS.items():
        if info.get("email") == email:
            if verify_password(password, info.get("password")):
                return True, {"username": uname, "name": info.get("name"), "email": info.get("email")}
            else:
                return False, "Senha incorreta."
    return False, "Usuário não encontrado."

# ----------------------------
# UI CSS
# ----------------------------
def inject_css():
    st.markdown("""
    <style>
    .login-box { background: #ffffff; padding: 18px; border-radius: 10px; box-shadow: 0 6px 18px rgba(0,0,0,0.12); }
    .login-title { font-size:20px; font-weight:700; color:#0b63ce; text-align:center; margin-bottom:8px; }
    </style>
    """, unsafe_allow_html=True)

# ----------------------------
# CSV robust reader
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
                file.seek(0)
            df = pd.read_csv(file, sep=d, engine="python")
            # normalize column names (strip)
            df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
            if not df.empty:
                return df
        except Exception:
            continue
    # last attempt
    try:
        if hasattr(file, "seek"):
            file.seek(0)
        df = pd.read_csv(file, sep=None, engine="python")
        df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
        if not df.empty:
            return df
    except Exception:
        st.error("Erro ao ler CSV. Verifique o formato e delimitadores.")
        return None

# ----------------------------
# Text column detection
# ----------------------------
def detect_text_columns(df, min_pct_text=0.10):
    candidates = []
    n = len(df)
    if n == 0:
        return []
    for col in df.columns:
        ser = df[col].astype(str).fillna("")
        pct_text = ser.str.contains(r"[A-Za-zÀ-ÿ]", regex=True).mean()
        avg_len = ser.str.len().mean()
        unique_prop = ser.nunique() / max(1, n)
        score = (pct_text * 0.6) + (min(1.0, avg_len/30.0) * 0.3) + (min(1.0, unique_prop) * 0.1)
        if pct_text >= min_pct_text or avg_len >= 20:
            candidates.append((col, score, pct_text, avg_len, unique_prop))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates

# ----------------------------
# Lightweight NLP heuristics + sentiment lexicon
# ----------------------------
STOP_PT = set(stopwords.words("portuguese"))
CUSTOM_STOP = {
    "sim","não","nao","ok","bom","boa","coisa","coisas","dia","mesmo","mesma",
    "gente","pessoa","pessoas","empresa","empresas","acho","acredito","ser",
    "estar","ter","fazer","feito","vou","vai","já","ja","pois","ainda",
    "sobre","também","tambem","etc","tipo","forma","algo","muito","pouco",
    "favor","porfavor","obrigado","agradeço","obrigada","ola","oi"
}
ALL_STOP = STOP_PT.union(CUSTOM_STOP)

_word_pattern = re.compile(r"[A-Za-zÀ-ÿ0-9\-']{2,}", flags=re.UNICODE)

VERB_SUFFIXES = ("ar","er","ir","ando","endo","indo","ado","ido","amos","emos","iram","aria","eria","iria","ava","eva","iva","ou","iu")
ADJ_SUFFIXES = ("oso","osa","avel","ível","ivel","al","ar","ico","ica","nte","ivel","oso","osa","ivo","iva")
NOUN_SUFFIXES = ("ção","ções","dade","ismo","ista","mento","agem","idade","ia","ismo","eza")

def clean_token(t):
    t = t.lower().strip()
    t = re.sub(r"(^['\"-]+)|(['\"-]+$)", "", t)
    t = re.sub(r"[^\wÀ-ÿ\-']+", "", t)
    return t

def is_number(s):
    try:
        float(s.replace(",", "."))
        return True
    except:
        return False

def simple_lemmatize(token):
    t = token
    for suf in ("ando","endo","indo","ado","ido","mente"):
        if t.endswith(suf) and len(t) > len(suf) + 3:
            return t[:-len(suf)]
    for suf in ("ar","er","ir"):
        if t.endswith(suf) and len(t) > len(suf) + 3:
            return t[:-len(suf)]
    if t.endswith("ões"):
        return t[:-3] + "ao"
    if t.endswith("s") and len(t) > 4:
        return t[:-1]
    return t

def classify_token(token):
    t = token.lower()
    if any(t.endswith(suf) for suf in VERB_SUFFIXES):
        return "VERB"
    if any(t.endswith(suf) for suf in ADJ_SUFFIXES):
        return "ADJ"
    if any(t.endswith(suf) for suf in NOUN_SUFFIXES):
        return "NOUN"
    if len(t) > 6:
        return "NOUN"
    return "OTHER"

# sentiment lexicon (light)
SENT_POS = {
    "bom","ótimo","otimo","excelente","maravilhoso","feliz","satisfeito","positivo","gostei","adoro","acertou"
}
SENT_NEG = {
    "ruim","péssimo","péssima","péssimo","pessimo","insatisfeito","triste","difícil","dificil","lento","demorado","injusto"
}

def sentiment_weight(token):
    t = token.lower()
    if t in SENT_POS or t in SENT_NEG:
        return 4
    return 0

def extract_relevant_words_from_text(text, top_k=None):
    if not isinstance(text, str) or not text.strip():
        return []
    tokens = _word_pattern.findall(text)
    cleaned = []
    for tk in tokens:
        tkc = clean_token(tk)
        if not tkc:
            continue
        if is_number(tkc):
            continue
        if tkc in ALL_STOP:
            continue
        if len(tkc) <= 2:
            continue
        cleaned.append(tkc)
    if not cleaned:
        return []
    weighted = []
    for tk in cleaned:
        cls = classify_token(tk)
        lemma = simple_lemmatize(tk)
        pos_weight = 1
        if cls == "VERB":
            pos_weight = 3
        elif cls == "ADJ":
            pos_weight = 2
        elif cls == "NOUN":
            pos_weight = 2
        else:
            pos_weight = 0.5
        s_weight = sentiment_weight(lemma)
        total_weight = pos_weight + s_weight
        weighted.append((lemma, total_weight))
    freq = {}
    for lemma, w in weighted:
        freq[lemma] = freq.get(lemma, 0) + w
    items = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    if top_k:
        items = items[:top_k]
    out = []
    for lemma, score in items:
        repeat = max(1, int(round(score)))
        out.extend([lemma] * repeat)
    return out

# ----------------------------
# Wordcloud theme generator
# ----------------------------
THEMES = {
    "Dark Elegante": ("black", "viridis"),
    "Deep Purple": ("#0d001f", "plasma"),
    "Neon Blue": ("#00111e", "cool"),
    "Gold": ("black", "cividis"),
    "Carbon Gray": ("#1a1a1a", "magma")
}

def generate_wordcloud_from_words(words, theme="Dark Elegante"):
    if not words:
        return None
    text = " ".join(words)
    bg, cmap = THEMES.get(theme, ("black", "viridis"))
    wc = WordCloud(
        width=1000,
        height=500,
        background_color=bg,
        colormap=cmap,
        collocations=False,
        max_words=150
    ).generate(text)
    fig, ax = plt.subplots(figsize=(12,6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig

# ----------------------------
# AI wrapper (Gemini) - safe
# ----------------------------
def gemini_analyse(text_list, title="Análise"):
    if not text_list:
        return "Nenhum texto para análise."
    if "GOOGLE_API_KEY" not in st.secrets or genai is None:
        return "Gemini não configurado em secrets."
    joined = "\n".join(map(str, text_list))
    prompt = f"""Você é um analista de dados. Produza:
1) Resumo executivo (curto)
2) Principais temas
3) Sentimentos
4) Recomendações
5) Tabela (Tema | Exemplo | Impacto | Ação)

Texto:
{joined}
"""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        resp = model.generate_content(prompt)
        return resp.text
    except Exception as e:
        return f"Erro Gemini: {e}"

# ----------------------------
# PDF generator
# ----------------------------
def generate_pdf_bytes(title, markdown_text):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1,12)]
    for line in markdown_text.split("\n"):
        if line.strip():
            story.append(Paragraph(line.replace("&","and"), styles["Normal"]))
        else:
            story.append(Spacer(1,6))
    doc.build(story)
    buffer.seek(0)
    return buffer

# ----------------------------
# Admin panel
# ----------------------------
def admin_panel_ui():
    st.title("🛡️ Painel Admin")
    st.info("Gere blocos TOML para colar no Streamlit Secrets.")
    st.subheader("Usuários atuais")
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
            st.error("Preencha username e senha.")
        else:
            h = PWD_CTX.hash(pw)
            st.success("Copie e cole o bloco abaixo no secrets.toml")
            st.code(f'[users.{nu}]\nname = "{nm}"\nemail = "{em}"\npassword = "{h}"', language="toml")

# ----------------------------
# Login modal (works on Cloud)
# ----------------------------
def open_login_modal():
    inject_css()
    with st.modal("Login"):
        st.markdown("<div class='login-box'>", unsafe_allow_html=True)
        st.markdown("<div class='login-title'>REPARA — Login</div>", unsafe_allow_html=True)
        email = st.text_input("Email", key="modal_email")
        pwd = st.text_input("Senha", type="password", key="modal_pwd")
        if st.button("Entrar", key="modal_enter"):
            ok, info = authenticate_by_email(email, pwd)
            if ok:
                st.session_state.logged = True
                st.session_state.userinfo = info
                st.session_state._rerun = True
                st.success("Login bem-sucedido.")
            else:
                st.error(info)
        if st.button("Esqueci a senha", key="modal_forgot"):
            # set flag to open recovery modal after closing
            st.session_state.show_recovery = True
            st.session_state._rerun = True
        st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Recovery modal
# ----------------------------
def open_recovery_modal():
    inject_css()
    with st.modal("Recuperação de senha"):
        st.markdown("<div class='login-box'>", unsafe_allow_html=True)
        st.markdown("<div class='login-title'>Recuperação de senha</div>", unsafe_allow_html=True)
        username_or_email = st.text_input("Usuário ou Email", key="rec_user")
        if st.button("Gerar token", key="rec_gen_btn"):
            # find username
            found = None
            for uname, info in USERS.items():
                if info.get("email") == username_or_email or uname == username_or_email:
                    found = uname
                    break
            if not found:
                st.error("Usuário não encontrado.")
            else:
                token = generate_token(found)
                st.success("Token gerado (15 min).")
                st.info(f"Token: `{token}` — em produção envie por e-mail.")
                st.session_state._rerun = True
        st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Main application
# ----------------------------
def main_app():
    st.title("📊 REPARA Analytics — v13.4.3")
    st.sidebar.success(f"Usuário: {st.session_state.userinfo.get('name')}")
    if st.sidebar.button("Painel Admin"):
        st.session_state.page = "admin"
        st.session_state._rerun = True
    if st.sidebar.button("Sair"):
        st.session_state.logged = False
        st.session_state._rerun = True
    if st.session_state.page == "admin":
        if st.session_state.userinfo.get("email") != "admin@repara.com":
            st.error("Acesso restrito ao administrador.")
            return
        admin_panel_ui()
        return

    st.sidebar.header("📥 Upload CSVs")
    cand_file = st.sidebar.file_uploader("Candidatos (CSV)", type=["csv"])
    emp_file = st.sidebar.file_uploader("Empresas (CSV)", type=["csv"])

    df_cand = read_csv_any(cand_file) if cand_file else None
    df_emp = read_csv_any(emp_file) if emp_file else None

    # KPIs
    st.header("📊 KPIs Rápidos")
    c1, c2, c3 = st.columns(3)
    c1.metric("Candidatos", len(df_cand) if df_cand is not None else "—")
    c2.metric("Colunas (Candidatos)", len(df_cand.columns) if df_cand is not None else "—")
    c3.metric("Empresas", len(df_emp) if df_emp is not None else "—")

    tabs = st.tabs(["👤 Candidatos", "🏢 Empresas", "🔀 Cruzada", "💬 Chat IA"])

    # CANDIDATOS TAB
    with tabs[0]:
        st.header("👤 Candidatos")
        if df_cand is None:
            st.info("Envie o CSV de candidatos pela sidebar.")
        else:
            st.dataframe(df_cand.head(50))
            candidates = detect_text_columns(df_cand)
            if candidates:
                st.subheader("Colunas textuais detectadas:")
                for col, score, pct, avg_len, uniq in candidates:
                    st.markdown(f"- **{col}** — score {score:.2f} — pct_text {pct:.2f} — avg_len {avg_len:.1f}")
                default_col = candidates[0][0]
            else:
                st.warning("Nenhuma coluna textual detectada automaticamente.")
                default_col = None

            st.markdown("**Selecione a coluna para análise / wordcloud:**")
            col_choice = st.selectbox("Coluna textual", options=[None] + list(df_cand.columns), index=0)
            if col_choice is None and default_col:
                col_choice = default_col

            if col_choice:
                st.subheader(f"Preview — coluna: {col_choice}")
                st.dataframe(df_cand[[col_choice]].head(10))
                text_series = df_cand[col_choice].dropna().astype(str)
                joined = " ".join(text_series.tolist())
                if len(joined.strip()) < 10:
                    st.info("Pouco texto nesta coluna.")
                else:
                    theme = st.selectbox("Tema do Wordcloud", list(THEMES.keys()), index=0)
                    if st.button("Gerar Wordcloud Inteligente"):
                        words = extract_relevant_words_from_text(joined, top_k=200)
                        if not words:
                            st.warning("Nenhuma palavra relevante encontrada.")
                        else:
                            fig = generate_wordcloud_from_words(words, theme=theme)
                            st.pyplot(fig)
                    if st.button("IA — Analisar candidatos"):
                        result = gemini_analyse(text_series.tolist(), title="Candidatos")
                        st.markdown(result)
                        pdfbuf = generate_pdf_bytes("Análise Candidatos", result)
                        st.download_button("📥 Baixar PDF (Candidatos)", data=pdfbuf, file_name="analise_candidatos.pdf", mime="application/pdf")
            else:
                st.info("Selecione uma coluna para ativar IA / Wordcloud.")

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
                for col, score, pct, avg_len, uniq in candidates_e:
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
                if st.button("Gerar Wordcloud Inteligente (empresas)"):
                    words = extract_relevant_words_from_text(" ".join(df_emp[col_choice_e].dropna().astype(str).tolist()), top_k=200)
                    if not words:
                        st.warning("Nenhuma palavra relevante encontrada.")
                    else:
                        fig = generate_wordcloud_from_words(words, theme=list(THEMES.keys())[0])
                        st.pyplot(fig)
                if st.button("IA — Analisar empresas", key="an_emp"):
                    result = gemini_analyse(df_emp[col_choice_e].dropna().astype(str).tolist(), title="Empresas")
                    st.markdown(result)
                    pdfbuf = generate_pdf_bytes("Análise Empresas", result)
                    st.download_button("📥 Baixar PDF (Empresas)", data=pdfbuf, file_name="analise_empresas.pdf", mime="application/pdf")
            else:
                st.info("Selecione uma coluna para ativar a análise IA.")

    # CRUZADA TAB
    with tabs[2]:
        st.header("🔀 Análise Cruzada")
        if df_cand is None or df_emp is None:
            st.info("Envie ambos os CSVs para cruzamento.")
        else:
            cand_cands = detect_text_columns(df_cand)
            emp_cands = detect_text_columns(df_emp)
            default_c = cand_cands[0][0] if cand_cands else None
            default_e = emp_cands[0][0] if emp_cands else None
            ccol = st.selectbox("Coluna candidatos", options=[None] + list(df_cand.columns), index=0)
            if ccol is None and default_c:
                ccol = default_c
            ecol = st.selectbox("Coluna empresas", options=[None] + list(df_emp.columns), index=0)
            if ecol is None and default_e:
                ecol = default_e
            if st.button("IA — Análise Cruzada"):
                texts = []
                if ccol:
                    texts += df_cand[ccol].dropna().astype(str).tolist()
                if ecol:
                    texts += df_emp[ecol].dropna().astype(str).tolist()
                if not texts:
                    st.info("Sem textos para análise cruzada.")
                else:
                    result = gemini_analyse(texts, title="Análise Cruzada")
                    st.markdown(result)
                    pdfbuf = generate_pdf_bytes("Análise Cruzada", result)
                    st.download_button("📥 Baixar PDF (Cruzada)", data=pdfbuf, file_name="analise_cruzada.pdf", mime="application/pdf")

    # CHAT TAB
    with tabs[3]:
        st.header("💬 Chat com Gemini")
        if genai is None or "GOOGLE_API_KEY" not in st.secrets:
            st.info("Gemini não configurado. Configure GOOGLE_API_KEY em secrets.")
        else:
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            for msg in st.session_state.chat_history:
                role = "Você" if msg["role"] == "user" else "IA"
                st.markdown(f"**{role}:** {msg['text']}")
            q = st.text_input("Pergunta sobre os dados", key="chat_q")
            if st.button("Enviar pergunta", key="chat_send"):
                if not q.strip():
                    st.warning("Escreva algo primeiro.")
                else:
                    # build a short preview context
                    ctx = ""
                    if df_cand is not None:
                        ctx += "Candidatos preview:\n" + df_cand.head(8).to_csv(index=False) + "\n"
                    if df_emp is not None:
                        ctx += "Empresas preview:\n" + df_emp.head(8).to_csv(index=False) + "\n"
                    prompt = f"Context:\n{ctx}\nPergunta: {q}"
                    try:
                        model = genai.GenerativeModel("gemini-2.5-flash")
                        resp = model.generate_content(prompt)
                        ans = resp.text
                    except Exception as e:
                        ans = f"Erro Gemini: {e}"
                    st.session_state.chat_history.append({"role":"user","text":q})
                    st.session_state.chat_history.append({"role":"assistant","text":ans})
                    st.experimental_rerun()

    # Footer: reset password
    st.markdown("---")
    st.header("🔐 Redefinir senha (se tiver token)")
    token_val = st.text_input("Token de recuperação", key="reset_token_343")
    new_password = st.text_input("Nova senha", type="password", key="reset_new_pw_343")
    if st.button("Redefinir senha", key="reset_submit_343"):
        if not token_val:
            st.error("Informe token.")
        else:
            ok, resp = validate_token(token_val)
            if not ok:
                st.error(resp)
            else:
                username = resp
                hashed = PWD_CTX.hash(new_password)
                st.success("Senha atualizada — cole o bloco abaixo no secrets.toml:")
                st.code(f'[users.{username}]\nname = "{USERS[username]["name"]}"\nemail = "{USERS[username]["email"]}"\npassword = "{hashed}"', language="toml")

# ----------------------------
# Bootstrap / flow control
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

# safe rerun outside modals
if st.session_state.get("_rerun", False):
    st.session_state._rerun = False
    st.rerun()

# Login / modals
if not st.session_state.logged:
    if st.button("Entrar"):
        open_login_modal()
    if st.session_state.show_recovery:
        open_recovery_modal()
else:
    main_app()
