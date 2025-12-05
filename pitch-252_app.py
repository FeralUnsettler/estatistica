# REPARA Analytics — v13.5.1
# Correção: st.experimental_rerun() removido do chat; usa flag _chat_rerun rerun seguro.
# Mantém: DEI system prompt, Wordcloud inteligente sem spaCy, aba Recuperação, login modal elegante.

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
import re
import nltk
from collections import Counter
from html import unescape

# ----------------------------
# Config / Setup
# ----------------------------
st.set_page_config(page_title="Repara Analytics", layout="wide")

# Configure Gemini if key present
if "GOOGLE_API_KEY" in st.secrets:
    try:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    except Exception:
        pass

# Ensure NLTK stopwords (quiet)
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

# ----------------------------
# Security & constants
# ----------------------------
PWD_CTX = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
RESET_TOKEN_TTL = 15 * 60

# ----------------------------
# DEI Specialist System Prompt (applied to analyses + chat)
# ----------------------------
DEI_SYSTEM_PROMPT = """
Você é um especialista em Diversidade, Equidade e Inclusão (DEI) com profundo conhecimento em:
- Políticas Públicas de Inclusão no Brasil (ex.: Lei Brasileira de Inclusão)
- Ações afirmativas e práticas de contratação inclusiva
- Diretrizes da OIT, ONU e ODS sobre inclusão social e trabalho decente
- Barreiras estruturais (racial, de gênero, etária, territorial, socioeconômica)
- Boas práticas de acolhimento, requalificação e promoção de oportunidades

Ao analisar respostas de candidatos e informações de empresas, foque em:
1) identificar barreiras de acesso e formas de exclusão;
2) detectar sentimentos de invisibilidade, estigma ou exclusão;
3) priorizar recomendações práticas, alinhadas a políticas públicas e ações internas;
4) sugerir indicadores acionáveis e medidas de mitigação (ex.: ajustes de recrutamento, programas de capacitação, avaliações por competências, parcerias públicas).

Responda sempre em Português do Brasil (pt-BR), de forma clara, empática e acionável.
"""

# ----------------------------
# Load users from secrets
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
# Token helpers
# ----------------------------
def init_tokens():
    if "reset_tokens" not in st.session_state:
        st.session_state.reset_tokens = {}

def generate_token(username):
    token = secrets.token_urlsafe(16)
    st.session_state.reset_tokens[token] = {
        "username": username,
        "expire": time.time() + RESET_TOKEN_TTL,
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
# UI styling helpers
# ----------------------------
def inject_css():
    st.markdown(
        """
    <style>
    .login-card {
        background: linear-gradient(180deg, #ffffff 0%, #f6fbff 100%);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 10px 30px rgba(15, 40, 90, 0.08);
    }
    .login-title { font-size:22px; font-weight:700; color:#0b63ce; text-align:center; margin-bottom:6px; }
    .login-sub { color:#4b5563; font-size:13px; text-align:center; margin-bottom:12px; }
    .small-note { font-size:12px; color:#6b7280; }
    .section-title { font-weight:700; color:#0b63ce; margin-bottom:6px; }
    </style>
    """,
        unsafe_allow_html=True,
    )

# ----------------------------
# Dialogs: LOGIN and Recovery (no experimental rerun)
# ----------------------------
@st.dialog("Login")
def login_dialog():
    inject_css()
    st.markdown("<div class='login-card'>", unsafe_allow_html=True)
    st.markdown("<div class='login-title'>REPARA — Entrar</div>", unsafe_allow_html=True)
    st.markdown("<div class='login-sub'>Plataforma de análise qualitativa com foco em inclusão e políticas públicas</div>", unsafe_allow_html=True)

    st.markdown("**Como usar (rápido):**\n- Faça upload dos CSVs de `Candidatos` e `Empresas` na barra lateral.\n- Selecione a coluna com respostas abertas e gere *Wordcloud* ou peça *Análise IA* (em PT-BR).")
    st.markdown("<hr/>", unsafe_allow_html=True)

    user = st.text_input("Usuário", key="login_user")
    pwd = st.text_input("Senha", type="password", key="login_pwd")

    col1, col2 = st.columns([1,1])
    with col1:
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
    with col2:
        if st.button("Esqueci a senha", key="login_forgot"):
            st.session_state.show_recovery = True
            st.session_state._rerun = True

    st.markdown("<div class='small-note'>Nota: em ambiente de produção, a entrega de tokens deve ser feita por e-mail.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# CSV reader robust
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
                try:
                    file.seek(0)
                except:
                    pass
            df = pd.read_csv(file, sep=d, engine="python")
            df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
            if not df.empty:
                return df
        except Exception:
            continue
    try:
        if hasattr(file, "seek"):
            try:
                file.seek(0)
            except:
                pass
        df = pd.read_csv(file, sep=None, engine="python")
        df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
        if not df.empty:
            return df
    except Exception:
        st.error("Erro ao ler CSV. Verifique delimitadores e formato.")
        return None

# ----------------------------
# Detect text columns (robust)
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
        score = (pct_text * 0.6) + (min(1.0, avg_len / 30.0) * 0.3) + (min(1.0, unique_prop) * 0.1)
        if pct_text >= min_pct_text or avg_len >= 20:
            candidates.append((col, score, pct_text, avg_len, unique_prop))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates

# ----------------------------
# Wordcloud intelligent (NLTK + heuristics)
# ----------------------------
nltk.download("stopwords", quiet=True)
STOP_PT = set(stopwords.words("portuguese"))
CUSTOM_STOP = {
    "sim","não","nao","ok","bom","boa","coisa","coisas","dia","mesmo","mesma",
    "gente","pessoa","pessoas","empresa","empresas","acho","acredito","ser",
    "estar","ter","fazer","feito","vou","vai","ja","já","pois","ainda",
    "sobre","também","tambem","etc","tipo","forma","algo","muito","pouco",
    "favor","porfavor","obrigado","agradeço","obrigada","ola","oi","ok"
}
ALL_STOP = STOP_PT.union(CUSTOM_STOP)
_word_pattern = re.compile(r"[A-Za-zÀ-ÿ0-9\-']{2,}", flags=re.UNICODE)

VERB_SUFFIXES = ("ar","er","ir","ando","endo","indo","ado","ido","arão","aram","emos","aria","eria","iria","ava","eva","iva","ou","iu")
ADJ_SUFFIXES = ("oso","osa","ável","ivel","ível","al","ar","ico","ica","nte","ivo","iva")
NOUN_SUFFIXES = ("ção","ções","dade","ismo","ista","mento","agem","idade","ia","ismo","eza")

def clean_token(t):
    t = unescape(t)
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
        weight = 1
        if cls == "VERB":
            weight = 3
        elif cls == "ADJ":
            weight = 2
        elif cls == "NOUN":
            weight = 2
        else:
            weight = 0.5
        weighted.append((lemma, weight))
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
# Gemini analyse helper (DEI system prompt + PT-BR)
# ----------------------------
def gemini_analyse(text_list, title="Análise"):
    if not text_list:
        return "Nenhum texto para análise."
    if "GOOGLE_API_KEY" not in st.secrets:
        return "Gemini API key não encontrada em secrets. Configure GOOGLE_API_KEY."
    joined = "\n".join(map(str, text_list))
    # Full prompt: DEI system prompt + task instructions (pt-BR)
    prompt = f"""
{DEI_SYSTEM_PROMPT}

Agora, considerando o título: {title}, execute a análise do texto abaixo e entregue, em Português (pt-BR):

1) Resumo executivo (1–2 parágrafos)
2) Principais temas (bullets)
3) Análise de sentimento (positivo / negativo / misto) com justificativa
4) Pontos de dor e oportunidades (bullets)
5) Recomendações práticas, alinhadas a políticas públicas e ações afirmativas
6) Tabela curta: Tema | Exemplo (trecho do texto) | Impacto | Ação recomendada

Texto:
{joined}

IMPORTANTE: Responda de forma objetiva, empática e orientada à ação. Indique métricas/indicadores quando possível.
"""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        resp = model.generate_content(prompt)
        return resp.text
    except Exception as e:
        return f"Erro ao chamar Gemini: {e}"

# ----------------------------
# Chat with Gemini — uses DEI specialist prompt (since option B)
# ----------------------------
def chat_with_gemini_context(df_cand, df_emp):
    st.header("💬 Chat com Gemini (Especialista DEI)")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    context = ""
    if df_cand is not None:
        context += "CANDIDATOS (preview):\n" + df_cand.head(8).to_csv(index=False) + "\n"
    if df_emp is not None:
        context += "EMPRESAS (preview):\n" + df_emp.head(8).to_csv(index=False) + "\n"
    for msg in st.session_state.chat_history:
        role = "Você" if msg["role"] == "user" else "IA"
        st.markdown(f"**{role}:** {msg['text']}")
    q = st.text_input("Pergunte ao especialista DEI sobre os dados", key="chat_q")
    if st.button("Enviar pergunta", key="chat_send"):
        if not q.strip():
            st.warning("Escreva algo antes de enviar.")
            return
        st.session_state.chat_history.append({"role":"user","text":q})
        if "GOOGLE_API_KEY" not in st.secrets:
            st.session_state.chat_history.append({"role":"assistant","text":"Gemini não configurado."})
            # request a safe rerun (outside) to render assistant reply
            st.session_state._chat_rerun = True
            return
        # Compose prompt with DEI system prompt + context + user question
        prompt = f"""
{DEI_SYSTEM_PROMPT}

Contexto:
{context}

Pergunta do usuário:
{q}

Responda em Português do Brasil, com foco em recomendações práticas e políticas públicas quando aplicável.
"""
        try:
            model = genai.GenerativeModel("gemini-2.5-flash")
            resp = model.generate_content(prompt)
            ans = resp.text
        except Exception as e:
            ans = f"Erro: {e}"
        st.session_state.chat_history.append({"role":"assistant","text":ans})
        # mark safe rerun to re-render page with assistant answer
        st.session_state._chat_rerun = True

# ----------------------------
# PDF generator
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
# Admin UI (TOML generator)
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
# Main app (with new '🔐 Recuperação' tab)
# ----------------------------
def main_app():
    st.title("📊 REPARA Analytics — v13.5.1")

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

    # Wordcloud theme selector (global)
    st.sidebar.markdown("### 🎨 Wordcloud")
    wc_theme = st.sidebar.radio("Tema", options=["Light","Dark"], index=0)
    wc_bg = "white" if wc_theme=="Light" else "#0b1220"
    wc_max_words = st.sidebar.slider("Max words", min_value=50, max_value=300, value=150, step=25)

    tabs = st.tabs(["👤 Candidatos", "🏢 Empresas", "🔀 Cruzada", "💬 Chat IA", "🔐 Recuperação"])

    # CANDIDATOS
    with tabs[0]:
        st.header("👤 Candidatos")
        if df_cand is None:
            st.info("Envie o CSV de candidatos pela sidebar.")
        else:
            st.dataframe(df_cand.head(50))
            candidates = detect_text_columns(df_cand)
            if candidates:
                st.subheader("Colunas textuais detectadas (ordem por score):")
                for col, score, pct_text, avg_len, uniq in candidates:
                    st.markdown(f"- **{col}** — score {score:.2f} — pct_text {pct_text:.2f} — avg_len {avg_len:.1f}")
                default_col = candidates[0][0]
            else:
                st.warning("Nenhuma coluna textual detectada automaticamente.")
                default_col = None

            st.markdown("**Selecione a coluna que contém respostas textuais (p/ análise IA / wordcloud):**")
            col_choice = st.selectbox("Coluna textual", options=[None] + list(df_cand.columns), index=0)
            if col_choice is None and default_col:
                col_choice = default_col

            if col_choice:
                st.subheader(f"Preview — coluna: {col_choice}")
                st.dataframe(df_cand[[col_choice]].head(10))
                text_series = df_cand[col_choice].dropna().astype(str)
                text_joined = " ".join(text_series.tolist())
                if len(text_joined.strip()) < 10:
                    st.info("Pouco texto nesta coluna — talvez não haja conteúdo para IA/wordcloud.")
                else:
                    # wordcloud intelligent action
                    if st.button("Gerar Wordcloud Inteligente"):
                        words = extract_relevant_words_from_text(text_joined, top_k=200)
                        if not words:
                            st.warning("Nenhuma palavra relevante encontrada para Wordcloud.")
                        else:
                            cleaned_text = " ".join(words)
                            wc = WordCloud(
                                width=900,
                                height=400,
                                collocations=False,
                                background_color=wc_bg,
                                max_words=wc_max_words,
                                prefer_horizontal=0.9
                            ).generate(cleaned_text)
                            fig, ax = plt.subplots(figsize=(10,4))
                            ax.imshow(wc, interpolation="bilinear")
                            ax.axis("off")
                            if wc_theme == "Dark":
                                fig.patch.set_facecolor("#0b1220")
                            st.pyplot(fig)

                    if st.button("IA — Analisar candidatos"):
                        result = gemini_analyse(text_series.tolist(), title="Candidatos")
                        st.markdown(result)
                        buf = generate_pdf_bytes("Análise Candidatos", result)
                        st.download_button("📥 Baixar PDF (Candidatos)", data=buf, file_name="analise_candidatos.pdf", mime="application/pdf")
            else:
                st.info("Selecione uma coluna para ativar a análise IA / Wordcloud.")

    # EMPRESAS
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
                if st.button("Gerar Wordcloud Inteligente (empresas)"):
                    words = extract_relevant_words_from_text(" ".join(df_emp[col_choice_e].dropna().astype(str).tolist()), top_k=200)
                    if not words:
                        st.warning("Nenhuma palavra relevante encontrada.")
                    else:
                        wc = WordCloud(width=900, height=400, collocations=False, background_color=wc_bg, max_words=wc_max_words).generate(" ".join(words))
                        fig, ax = plt.subplots(figsize=(10,4))
                        ax.imshow(wc, interpolation="bilinear")
                        ax.axis("off")
                        if wc_theme == "Dark":
                            fig.patch.set_facecolor("#0b1220")
                        st.pyplot(fig)

                if st.button("IA — Analisar empresas", key="an_emp"):
                    result = gemini_analyse(df_emp[col_choice_e].dropna().astype(str).tolist(), title="Empresas")
                    st.markdown(result)
                    buf = generate_pdf_bytes("Análise Empresas", result)
                    st.download_button("📥 Baixar PDF (Empresas)", data=buf, file_name="analise_empresas.pdf", mime="application/pdf")
            else:
                st.info("Selecione uma coluna para ativar a análise IA.")

    # CRUZADA
    with tabs[2]:
        st.header("🔀 Análise Cruzada")
        if df_cand is None or df_emp is None:
            st.info("Envie ambos os CSVs (candidatos e empresas).")
        else:
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

    # CHAT IA (DEI specialist)
    with tabs[3]:
        chat_with_gemini_context = globals().get("chat_with_gemini_context")
        if callable(chat_with_gemini_context):
            chat_with_gemini_context(df_cand, df_emp)
        else:
            st.info("Chat não disponível.")

    # RECUPERAÇÃO (nova aba)
    with tabs[4]:
        st.header("🔐 Recuperação de senha")
        st.markdown("Se você recebeu um token de recuperação (gerado via modal de recuperação), insira abaixo para redefinir sua senha.")
        token_val = st.text_input("Token de recuperação", key="recover_token")
        new_password = st.text_input("Nova senha", type="password", key="recover_new_pw")
        if st.button("Redefinir senha", key="recover_submit"):
            if not token_val:
                st.error("Informe o token.")
            else:
                ok, resp = validate_token(token_val)
                if not ok:
                    st.error(resp)
                else:
                    username = resp
                    hashed = PWD_CTX.hash(new_password)
                    st.success("Senha atualizada (local). Copie o bloco abaixo e cole no secrets.toml do Streamlit Cloud:")
                    st.code(f'[users.{username}]\nname = "{USERS[username]["name"]}"\nemail = "{USERS[username]["email"]}"\npassword = "{hashed}"', language="toml")
        st.markdown("---")
        st.markdown("**Gerar token (admin/testes)**")
        gen_user = st.text_input("Gerar token para usuário (username)", key="gen_token_user")
        if st.button("Gerar token", key="gen_token_btn"):
            if not gen_user:
                st.error("Informe o username.")
            elif gen_user not in USERS:
                st.error("Usuário não encontrado.")
            else:
                tk = generate_token(gen_user)
                st.success("Token gerado (visível apenas para testes).")
                st.info(f"Token para {gen_user}: `{tk}` — válido por 15 minutos.")

# ----------------------------
# Execution
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
if "_chat_rerun" not in st.session_state:
    st.session_state._chat_rerun = False

# safe rerun flags handled outside dialogs
if st.session_state._rerun:
    st.session_state._rerun = False
    st.rerun()

if st.session_state._chat_rerun:
    st.session_state._chat_rerun = False
    st.rerun()

if not st.session_state.logged:
    if st.button("Entrar", key="open_login"):
        login_dialog()
    if st.session_state.show_recovery:
        recovery_dialog()
else:
    main_app()
