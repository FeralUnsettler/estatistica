# ===============================================================
# REPARA Analytics — v13.4.5 (LandingPage Edition - Dark MindVision)
# - Full-page login (landing) with MindVision logo (logo.png local)
# - Safe for Streamlit Cloud (no st.modal, no st.dialog, no heavy libs)
# - Wordcloud intelligent, sentiment lexicon, Gemini optional
# ===============================================================

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

# optional
try:
    import google.generativeai as genai
except Exception:
    genai = None

# lightweight NLP
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords", quiet=True)

# -------------------------
# Config / constants
# -------------------------
st.set_page_config(page_title="REPARA Analytics — MindVision", layout="wide", initial_sidebar_state="collapsed")
PWD_CTX = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
RESET_TOKEN_TTL = 15 * 60

# If Google API key present, configure gemini
if "GOOGLE_API_KEY" in st.secrets and genai is not None:
    try:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    except Exception:
        pass

# -------------------------
# Simple helpers to load users from secrets
# -------------------------
def load_users():
    raw = st.secrets.get("users", {}) or {}
    users = {}
    for uname, info in raw.items():
        users[uname] = {
            "name": info.get("name"),
            "email": info.get("email"),
            "password": info.get("password")
        }
    return users

USERS = load_users()

# -------------------------
# Token (password reset) in-session
# -------------------------
def init_tokens():
    if "reset_tokens" not in st.session_state:
        st.session_state.reset_tokens = {}

def generate_token(username):
    token = secrets.token_urlsafe(16)
    st.session_state.reset_tokens[token] = {"username": username, "expire": time.time() + RESET_TOKEN_TTL}
    return token

def validate_token(token):
    entry = st.session_state.reset_tokens.get(token)
    if not entry:
        return False, "Token inválido"
    if time.time() > entry["expire"]:
        del st.session_state.reset_tokens[token]
        return False, "Token expirado"
    return True, entry["username"]

# -------------------------
# Authentication helpers
# -------------------------
def verify_password(plain, hashed):
    try:
        return PWD_CTX.verify(plain, hashed)
    except Exception:
        return False

def authenticate_by_email(email, password):
    for uname, info in USERS.items():
        if info.get("email") == email:
            if verify_password(password, info.get("password")):
                return True, {"username": uname, "name": info.get("name"), "email": info.get("email")}
            else:
                return False, "Senha incorreta."
    return False, "Usuário não encontrado."

# -------------------------
# Styling: Landing Page CSS (dark gradient MindVision)
# -------------------------
def landing_css():
    st.markdown(
        """
        <style>
        .landing {
            height: 90vh;
            display:flex;
            align-items:center;
            justify-content:center;
            background: linear-gradient(135deg, #000000 0%, #002b00 40%, #5a3200 100%);
            color: #ffffff;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial;
        }
        .card {
            background: rgba(255,255,255,0.03);
            border-radius: 12px;
            padding: 28px;
            width: 850px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.5);
            border: 1px solid rgba(255,255,255,0.04);
        }
        .logo {
            display:flex;
            align-items:center;
            gap:16px;
        }
        .title {
            font-size:28px;
            font-weight:700;
            color: #ffd166;
            margin-bottom: 6px;
        }
        .subtitle {
            color: #cfe8c5;
            margin-bottom: 18px;
        }
        .login-row {
            display:flex;
            gap:12px;
            margin-top: 12px;
        }
        .field {
            width:100%;
        }
        .btn {
            background: linear-gradient(90deg,#00ff7a,#ff8c00);
            color: #000;
            border:none;
            padding:10px 18px;
            border-radius:8px;
            font-weight:700;
            cursor:pointer;
        }
        a.link {
            color:#9fd9a5;
            text-decoration:none;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# -------------------------
# CSV reader robust
# -------------------------
def read_csv_any(file):
    if file is None:
        return None
    delims = [",", ";", "\t", "|"]
    for d in delims:
        try:
            if hasattr(file, "seek"):
                file.seek(0)
            df = pd.read_csv(file, sep=d, engine="python")
            df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
            if not df.empty:
                return df
        except Exception:
            continue
    try:
        if hasattr(file, "seek"):
            file.seek(0)
        df = pd.read_csv(file, sep=None, engine="python")
        df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
        return df
    except Exception:
        st.error("Erro ao ler CSV. Verifique delimitador/encoding.")
        return None

# -------------------------
# Detect text columns robust
# -------------------------
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
        score = (pct_text * 0.6) + (min(1.0, avg_len / 30.0) * 0.3) + (min(1.0, unique_prop) * 0.1)
        if pct_text >= min_pct_text or avg_len >= 20:
            candidates.append((col, score, pct_text, avg_len, unique_prop))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates

# -------------------------
# Lightweight NLP heuristics + sentiment lexicon (compatible with Cloud)
# -------------------------
STOP_PT = set(stopwords.words("portuguese"))
CUSTOM_STOP = {
    "sim","não","nao","ok","bom","boa","coisa","coisas","dia","mesmo","mesma",
    "gente","pessoa","pessoas","empresa","empresas","acho","acredito","ser",
    "estar","ter","fazer","feito","vou","vai","já","ja","pois","ainda",
    "sobre","também","tambem","etc","tipo","forma","algo","muito","pouco"
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

# sentiment lexicon (simple)
SENT_POS = {"bom","ótimo","otimo","excelente","maravilhoso","feliz","satisfeito","positivo","gostei","adoro"}
SENT_NEG = {"ruim","péssimo","péssima","pessimo","insatisfeito","triste","difícil","dificil","lento","demorado","injusto"}

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

# -------------------------
# Wordcloud themes
# -------------------------
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
        width=1000, height=500, background_color=bg, colormap=cmap, collocations=False, max_words=150
    ).generate(text)
    fig, ax = plt.subplots(figsize=(12,6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig

# -------------------------
# Gemini wrapper (safe)
# -------------------------
def gemini_analyse(text_list, title="Análise"):
    if not text_list:
        return "Nenhum texto para análise."
    if "GOOGLE_API_KEY" not in st.secrets or genai is None:
        return "Gemini não configurado."
    joined = "\n".join(map(str, text_list))
    prompt = f"""Você é um analista experiente. Analise e gere:
1) Resumo executivo
2) Principais temas
3) Sentimentos
4) Recomendações
5) Tabela: Tema | Exemplo | Impacto | Ação

Texto:
{joined}
"""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        resp = model.generate_content(prompt)
        return resp.text
    except Exception as e:
        return f"Erro Gemini: {e}"

# -------------------------
# PDF generator
# -------------------------
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

# -------------------------
# Admin UI
# -------------------------
def admin_panel_ui():
    st.title("🛡️ Painel Admin")
    st.info("Gere blocos TOML para colar em Streamlit Secrets.")
    st.subheader("Usuários atuais")
    for u, info in USERS.items():
        st.markdown(f"- **{u}** — {info.get('email')}")
    st.markdown("---")
    st.subheader("Criar novo usuário")
    nu = st.text_input("Username", key="admin_nu")
    nm = st.text_input("Nome completo", key="admin_nm")
    em = st.text_input("Email", key="admin_em")
    pw = st.text_input("Senha (gera hash)", type="password", key="admin_pw")
    if st.button("Gerar Bloco TOML"):
        if not nu or not pw:
            st.error("Username e senha obrigatórios.")
        else:
            h = PWD_CTX.hash(pw)
            st.success("Copie para secrets.toml:")
            st.code(f'[users.{nu}]\nname = "{nm}"\nemail = "{em}"\npassword = "{h}"', language="toml")

# -------------------------
# Landing Page (full-page login)
# -------------------------
def landing_page():
    landing_css()
    cols = st.columns([1,1,1])
    with cols[1]:
        st.write("")  # vertical centering
        st.write("")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        # logo (local file)
        try:
            st.image("logo.png", width=600)
        except Exception:
            st.markdown("**MindVision®**")
        st.markdown('<div class="title">MindVision — REPARA Analytics</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Análises de talentos, insights rápidos e IA</div>', unsafe_allow_html=True)
        # login form
        email = st.text_input("Email", key="landing_email")
        pwd = st.text_input("Senha", type="password", key="landing_pwd")
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        col1, col2 = st.columns([2,1])
        with col1:
            if st.button("Entrar", key="landing_enter"):
                ok, info = authenticate_by_email(email, pwd)
                if ok:
                    st.session_state.logged = True
                    st.session_state.userinfo = info
                    st.session_state._rerun = True
                    st.success("Login bem-sucedido — carregando app...")
                else:
                    st.error(info)
        with col2:
            if st.button("Esqueci minha senha", key="landing_forgot"):
                # simple flow: generate token and show
                # find username
                found = None
                for uname, info in USERS.items():
                    if info.get("email") == email:
                        found = uname
                        break
                if not found:
                    st.info("Informe seu email e clique 'Esqueci minha senha' novamente para gerar token.")
                else:
                    token = generate_token(found)
                    st.success("Token gerado (15 min).")
                    st.code(token)
        st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Main app after login
# -------------------------
def main_app():
    st.sidebar.title("REPARA — Menu")
    st.sidebar.write(f"Usuário: {st.session_state.userinfo.get('name')}")
    if st.sidebar.button("Painel Admin"):
        st.session_state.page = "admin"
        st.session_state._rerun = True
    if st.sidebar.button("Sair"):
        st.session_state.logged = False
        st.session_state._rerun = True

    if st.session_state.page == "admin":
        if st.session_state.userinfo.get("email") != "admin@repara.com":
            st.error("Acesso restrito.")
            return
        admin_panel_ui()
        return

    st.title("📊 REPARA Analytics — MindVision")
    st.sidebar.header("Upload CSVs")
    cand_file = st.sidebar.file_uploader("Candidatos CSV", type=["csv"])
    emp_file = st.sidebar.file_uploader("Empresas CSV", type=["csv"])

    df1 = read_csv_any(cand_file) if cand_file else None
    df2 = read_csv_any(emp_file) if emp_file else None

    # KPIs
    st.header("📈 KPIs Rápidos")
    c1, c2, c3 = st.columns(3)
    c1.metric("Candidatos", len(df1) if df1 is not None else "—")
    c2.metric("Colunas (candidatos)", len(df1.columns) if df1 is not None else "—")
    c3.metric("Empresas", len(df2) if df2 is not None else "—")

    tabs = st.tabs(["👤 Candidatos", "🏢 Empresas", "🔀 Cruzada", "💬 Chat IA"])

    # Candidates tab
    with tabs[0]:
        st.header("👤 Candidatos")
        if df1 is None:
            st.info("Envie o CSV de candidatos pela sidebar.")
        else:
            st.dataframe(df1.head(50))
            candidates = detect_text_columns(df1)
            default_col = candidates[0][0] if candidates else None
            st.markdown("Selecione coluna textual:")
            col_choice = st.selectbox("Coluna textual", options=[None] + list(df1.columns), index=0)
            if col_choice is None and default_col:
                col_choice = default_col
            if col_choice:
                st.dataframe(df1[[col_choice]].head(10))
                joined = " ".join(df1[col_choice].dropna().astype(str).tolist())
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
                        result = gemini_analyse(df1[col_choice].dropna().astype(str).tolist(), title="Candidatos")
                        st.markdown(result)
                        pdfbuf = generate_pdf_bytes("Análise Candidatos", result)
                        st.download_button("📥 Baixar PDF (Candidatos)", data=pdfbuf, file_name="analise_candidatos.pdf", mime="application/pdf")
    # Companies tab
    with tabs[1]:
        st.header("🏢 Empresas")
        if df2 is None:
            st.info("Envie o CSV de empresas pela sidebar.")
        else:
            st.dataframe(df2.head(50))
            candidates_e = detect_text_columns(df2)
            default_col_e = candidates_e[0][0] if candidates_e else None
            st.markdown("Selecione coluna textual (empresas):")
            col_choice_e = st.selectbox("Coluna empresas", options=[None] + list(df2.columns), index=0)
            if col_choice_e is None and default_col_e:
                col_choice_e = default_col_e
            if col_choice_e:
                st.dataframe(df2[[col_choice_e]].head(10))
                if st.button("Gerar Wordcloud Inteligente (empresas)"):
                    words = extract_relevant_words_from_text(" ".join(df2[col_choice_e].dropna().astype(str).tolist()), top_k=200)
                    if not words:
                        st.warning("Nenhuma palavra relevante encontrada.")
                    else:
                        fig = generate_wordcloud_from_words(words, theme=list(THEMES.keys())[0])
                        st.pyplot(fig)
                if st.button("IA — Analisar empresas", key="an_emp"):
                    result = gemini_analyse(df2[col_choice_e].dropna().astype(str).tolist(), title="Empresas")
                    st.markdown(result)
                    pdfbuf = generate_pdf_bytes("Análise Empresas", result)
                    st.download_button("📥 Baixar PDF (Empresas)", data=pdfbuf, file_name="analise_empresas.pdf", mime="application/pdf")
    # Cruzada tab
    with tabs[2]:
        st.header("🔀 Análise Cruzada")
        if df1 is None or df2 is None:
            st.info("Carregue os dois CSVs para cruzamento.")
        else:
            cand_cands = detect_text_columns(df1)
            emp_cands = detect_text_columns(df2)
            default_c = cand_cands[0][0] if cand_cands else None
            default_e = emp_cands[0][0] if emp_cands else None
            ccol = st.selectbox("Coluna candidatos", options=[None] + list(df1.columns), index=0)
            if ccol is None and default_c:
                ccol = default_c
            ecol = st.selectbox("Coluna empresas", options=[None] + list(df2.columns), index=0)
            if ecol is None and default_e:
                ecol = default_e
            if st.button("IA — Análise Cruzada"):
                texts = []
                if ccol:
                    texts += df1[ccol].dropna().astype(str).tolist()
                if ecol:
                    texts += df2[ecol].dropna().astype(str).tolist()
                if texts:
                    result = gemini_analyse(texts, title="Análise Cruzada")
                    st.markdown(result)
                    pdfbuf = generate_pdf_bytes("Análise Cruzada", result)
                    st.download_button("📥 Baixar PDF (Cruzada)", data=pdfbuf, file_name="analise_cruzada.pdf", mime="application/pdf")
                else:
                    st.info("Sem textos para análise cruzada.")
    # Chat tab
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
                    ctx = ""
                    if df1 is not None:
                        ctx += "Candidatos preview:\n" + df1.head(8).to_csv(index=False) + "\n"
                    if df2 is not None:
                        ctx += "Empresas preview:\n" + df2.head(8).to_csv(index=False) + "\n"
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
    token_val = st.text_input("Token de recuperação", key="reset_token_345")
    new_password = st.text_input("Nova senha", type="password", key="reset_new_pw_345")
    if st.button("Redefinir senha", key="reset_submit_345"):
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

# -------------------------
# Bootstrap flow
# -------------------------
def inject_css_min():
    st.markdown("<style> .stApp { background-color: #000000; } </style>", unsafe_allow_html=True)

inject_css_min()
init_tokens()

if "logged" not in st.session_state:
    st.session_state.logged = False
if "page" not in st.session_state:
    st.session_state.page = "main"
if "userinfo" not in st.session_state:
    st.session_state.userinfo = {"name": "Visitante", "email": ""}
if "_rerun" not in st.session_state:
    st.session_state._rerun = False

# safe rerun
if st.session_state.get("_rerun", False):
    st.session_state._rerun = False
    st.rerun()

# Show landing page or app
if not st.session_state.logged:
    landing_page()
else:
    main_app()
