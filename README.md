# 🧠 REPARA Analytics — Plataforma Inteligente de Análise de Talentos  
**Versão: 13.2**  

Repara Analytics é uma plataforma de análise inteligente que conecta empresas e candidatos, permitindo gerar insights avançados a partir de respostas qualitativas em CSVs.  
Ela utiliza **IA generativa (Gemini 2.5 Flash)**, dashboards visuais, análises automatizadas e um painel admin seguro com autenticação.

Desenvolvida especialmente para o projeto **REPARA – Revela Talentos para Todos**, integrando:

- Análise dos CSVs de candidatos e empresas  
- Insights automáticos via IA  
- Wordclouds, KPIs, gráficos e relatórios PDF  
- Chat com IA usando contexto dos dados  
- Painel administrativo completo  
- Redefinição de senha com token  
- Autenticação robusta com senhas hash (pbkdf2_sha256)  
- Navegação estável sem `experimental_rerun()`  

---

## 🚀 **Funcionalidades Principais**

### 🔐 Autenticação Completa
- Login em modal (UI moderna)  
- Hash seguro de senhas (`pbkdf2_sha256`)  
- Recuperação de senha via token  
- Gerenciamento de usuários via Painel Admin  
- Armazenamento seguro no `secrets.toml`  

### 📊 Análise de Dados
- Leitura de CSV com autodetecção de delimitador  
- Inferência inteligente das colunas textuais  
- Wordcloud dos relatos dos candidatos  
- Dashboard com KPIs  
- Ranking de desafios das empresas  

### 🤖 Inteligência Artificial (Gemini)
- Análise textual automática (temas, sentimentos, recomendações)  
- Análise cruzada candidatos × empresas  
- Chat interativo com contexto dos CSVs  
- Geração de relatórios PDF automáticos  

### 🛡️ Painel Administrativo
- Criar novos usuários  
- Gerar blocos TOML prontos para secrets  
- Hashs de senha com segurança  

---

## 📦 **Tecnologias Utilizadas**

- **Python 3.10+**
- **Streamlit 1.39**
- **Google Generative AI (Gemini 2.5 Flash)**
- **Passlib (pbkdf2_sha256)**
- **Pandas**
- **Matplotlib**
- **WordCloud**
- **ReportLab**
- **Streamlit Dialogs (st.dialog)**

---

## 🗂️ **Estrutura do Projeto**


📁 repara-analytics/
│
├─ app.py                # Aplicação principal (v13.2)
├─ requirements.txt      # Dependências do Streamlit Cloud
├─ README.md             # Este arquivo
└─ data/ (opcional)      # CSVs usados para testes locais

---

# ☁️ Deploy no Streamlit Cloud

## 1️⃣ Criar o repositório no GitHub
- Suba `app.py`
- Suba `requirements.txt`
- Suba este `README.md`

## 2️⃣ Conectar o repositório ao Streamlit Cloud
Entre em:

🔗 https://share.streamlit.io/

Clique em **New App** → selecione o repositório.

## 3️⃣ Configurar Secrets do Streamlit Cloud

Vá em:

**Settings → Secrets**  
e cole:

```toml
GOOGLE_API_KEY = "SUA_CHAVE_GEMINI"

[users.admin]
name = "Administrador"
email = "admin@repara.com"
password = "$pbkdf2-sha256$hash_aqui"
````

Você pode criar outros usuários pelo painel Admin dentro do app.

---

# 🔑 Como criar novas senhas (hash pbkdf2)

Você pode gerar com:

```python
from passlib.context import CryptContext
pwd = CryptContext(schemes=["pbkdf2_sha256"])
print(pwd.hash("SUA_SENHA"))
```

Ou direto no **Painel Admin**.

---

# 🖥️ Como rodar localmente

### 1️⃣ Clonar o repositório

```bash
git clone https://github.com/sua-org/repara-analytics.git
cd repara-analytics
```

### 2️⃣ Criar ambiente virtual

```bash
python -m venv venv
source venv/bin/activate   # Linux/mac
venv\Scripts\activate      # Windows
```

### 3️⃣ Instalar dependências

```bash
pip install -r requirements.txt
```

### 4️⃣ Criar `.streamlit/secrets.toml` localmente

```
mkdir .streamlit
nano .streamlit/secrets.toml
```

Cole:

```toml
GOOGLE_API_KEY = "SUA_CHAVE"

[users.admin]
name = "Administrador"
email = "admin@repara.com"
password = "$pbkdf2-sha256$..."
```

### 5️⃣ Rodar o app

```bash
streamlit run app.py
```

---

# 📸 Screenshots (opcional)

> Substituir imagens pelos seus próprios prints

```
![Login](screenshots/login.png)
![Dashboard](screenshots/dashboard.png)
![Wordcloud](screenshots/wordcloud.png)
![Chat Gemini](screenshots/chat.png)
![Admin](screenshots/admin.png)
```

---

# 🛡️ Segurança

* Senhas sempre armazenadas com hash PBKDF2-SHA256
* Nada fica no cliente (client-side)
* Tokens de recuperação duram 15 minutos
* Gemini jamais recebe dados pessoais sensíveis — apenas trechos dos CSVs
* Dialogs isolados evitam rerun inseguro

---

# 🧭 Roadmap

### 🔜 Futuras Melhorias

* [ ] Suporte a upload múltiplo de CSV
* [ ] Histórico salvo em Supabase
* [ ] Exportação Excel consolidada
* [ ] Painel de BI com Plotly
* [ ] Modo escuro / tema personalizado
* [ ] Autorização por papéis (admin / analista / gestor)
* [ ] Avaliação automática de match candidato–empresa

---

# 📄 Licença

Este projeto é licenciado sob **MIT License** — uso livre com atribuição.

---

# 👥 Equipe

Projeto desenvolvido por Luciano Martins Fagundes
Com suporte técnico via ChatGPT — Build Assist Pro

---
