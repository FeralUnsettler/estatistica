
# 🧠 REPARA Analytics — Plataforma Inteligente de Análise de Talentos  
**Versão:** 13.3 (2025)  
**Tecnologias:** Streamlit + Gemini AI + Python + Wordcloud + Pandas + ReportLab  
**Ambiente:** Compatível com Streamlit Cloud

---

# 🚀 O que é o REPARA Analytics?

O **REPARA Analytics v13.3** é uma plataforma inteligente projetada para analisar dados de candidatos e empresas a partir de arquivos CSV e gerar **insights automáticos** com apoio de IA (Gemini 2.5 Flash).

A aplicação foi criada para o projeto **REPARA — Revela Talentos Para Todos**, com o objetivo de dar visibilidade a grupos sub-representados, gerar análises qualitativas e cruzadas, e auxiliar empresas e instituições educacionais a identificar padrões, dores e oportunidades.

---

# ✨ Principais Funcionalidades

### 🔐 Autenticação Segura  
- Login com UI moderna usando `st.dialog`  
- Senhas com hash PBKDF2-SHA256  
- Painel Admin para criar novos usuários e gerar blocos TOML  
- Recuperação de senha com token temporário (15 minutos)  
- Compatível com `secrets.toml` do Streamlit Cloud  

---

### 📄 Upload e Análise de CSV  
- Leitura *robusta* de CSV com autodetecção de delimitador  
- Normalização automática dos nomes das colunas  
- Preview de até 50 linhas  
- Detecção de colunas textuais usando algoritmo inteligente  
- Suporte total a UTF-8, acentos e textos longos  
- Tratamento de colunas vazias ou inconsistentes  

---

### 🤖 Análises com Inteligência Artificial (Gemini 2.5 Flash)  
Para qualquer coluna textual selecionada:

- Resumo Executivo  
- Principais temas das respostas  
- Quadro “Tema | Exemplo | Impacto | Ação recomendada”  
- Recomendações práticas para o time de RH ou gestão  
- Análises cruzadas (Candidatos × Empresas)  
- Chat com IA usando contexto dos dois CSVs  

---

### 🎨 Visualizações  
- Wordcloud personalizada  
- KPIs básicos (quantidade de candidatos, empresas, colunas, etc.)  
- Gráficos e tabelas dinâmicas  
- Exportação de relatórios em PDF  

---

### 🛡️ Painel Administrador  
- Gerenciamento de usuários  
- Geração de hashes  
- Blocos `TOML` prontos para colar no Streamlit Cloud  
- Exclusivo para admin (ex.: `admin@repara.com`)  

---
---

## 🚀 Novidades da versão 13.4.2
### Streamlit + Gemini + Wordcloud Inteligente + Admin Panel

### ✨ Wordcloud Inteligente Dark Mode
- Fundo escuro premium
- Temas selecionáveis:
  - Dark Elegante
  - Deep Purple
  - Neon Blue
  - Gold
  - Carbon Gray

### ❤️ Sentiment Lexicon PT-BR integrado
Palavras emocionais agora têm peso extra:
- positivas → +4
- negativas → +4

### 🧠 POS Heurística (sem spaCy — compatível com Streamlit Cloud)
- identifica verbos, adjetivos e substantivos por morfologia
- lematização leve
- stopwords PT-BR + customizadas
- compatível com CSVs reais

### 🤖 Gemini 2.5 Flash
- análise textual profunda
- análise cruzada
- chat contextual
- geração de PDF automática

### 🔐 Autenticação completa
- PBKDF2-SHA256
- painel admin para gerar novos usuários
- blocos TOML para Streamlit Cloud

---

## 📦 Estrutura

```
repara/
│── app.py
│── requirements.txt
└── README.md

```

---

## 📥 Dependências (requirements.txt)

```
streamlit
pandas
matplotlib
wordcloud
google-generativeai
passlib
reportlab
python-dotenv
nltk
```

---
---

# 🔥 Novidades da Versão 13.3  

### ✔ DETECÇÃO TEXTUAL 100% REFEITA  
Problema original:  
O app exibia *“Nenhuma coluna textual detectada”* em CSVs válidos.

Agora:

- Detector usa regex avançado para identificar colunas com letras, inclusive acentuadas  
- Mede score baseado em:
  - % de células com texto  
  - tamanho médio das respostas  
  - diversidade de respostas  
- Ordena automaticamente da mais relevante para a menos textual  
- Sempre oferece **seleção manual**  
- IA sempre disponível quando há qualquer coluna válida  

---

### ✔ NORMALIZAÇÃO DE COLUNAS  
- Espaços removidos automaticamente  
- Acentos normalizados internamente para detecção  
- Nomes originais preservados na interface  

---

### ✔ MELHORIAS NO CHAT IA  
O chat agora inclui:

- Preview automático dos CSVs (até 8 linhas)  
- Contexto enxuto para perguntas  
- Histórico persistente  

---

### ✔ WORDCLOUD APRIMORADA  
- Suporte a português  
- Remoção de caracteres indesejados  
- Renderização mais nítida  

---

### ✔ PDF PROFISSIONAL  
- Usando ReportLab  
- Título com estilo  
- Layout limpo  
- Download com um clique  

---

### ✔ PAINEL ADMIN COMPLETO  
- Gerar usuários  
- Gerar hashes  
- TOML pronto  
- Melhor UI  

---

### ✔ SEM MAIS `experimental_rerun()`  
- Toda a navegação usa:

```shell
st.session_state._rerun = True
st.rerun()
```

- Total compatibilidade com `st.dialog`  

---

# 📦 Requisitos

Crie um arquivo **requirements.txt** contendo:

```

streamlit
pandas
matplotlib
wordcloud
reportlab
google-generativeai
passlib
python-dotenv

````

(Esse é exatamente o arquivo recomendado para Streamlit Cloud.)

---

# ☁️ Deploy no Streamlit Cloud

1. Suba para o GitHub:
   - `app.py`
   - `requirements.txt`
   - `README.md`

2. Acesse:  
   https://streamlit.io/cloud

3. Crie um novo app.

4. Em **Settings → Secrets**, coloque:

```toml
GOOGLE_API_KEY = "SUA_CHAVE"

[users.admin]
name = "Administrador"
email = "admin@repara.com"
password = "$pbkdf2-sha256$..."

[users.luciano]
name = "Luciano"
email = "luciano@repara.com"
password = "$pbkdf2-sha256$..."
````

Você pode gerar hashes no painel admin ou com:

```python
from passlib.context import CryptContext
pwd = CryptContext(schemes=["pbkdf2_sha256"])
print(pwd.hash("SUA_SENHA"))
```

---

# 📁 Estrutura do Projeto

```
📦 repara-analytics
│
├── app.py                # aplicativo completo v13.3
├── requirements.txt
└── README.md
```

---

# 🧪 Como rodar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

Crie o arquivo:

```
mkdir .streamlit
nano .streamlit/secrets.toml
```

E coloque suas chaves e usuários.

---

# 💬 Chat com IA

Dentro do app:

1. Vá na aba **“💬 Chat IA”**
2. Pergunte qualquer coisa sobre os CSVs
3. Gemini responde baseado no preview dos dados carregados

---

# 📊 Análises Cruzadas

Na aba **“🔀 Cruzada”**:

1. Selecione uma coluna textual de candidatos
2. Selecione uma coluna textual de empresas
3. Clique **“IA — Análise Cruzada”**

Resultado:

* Tema geral
* Convergência percebida
* Dores comuns
* Recomendações

---

# 📄 Geração de PDF

Todos os relatórios gerados pelo Gemini podem ser baixados em:

```
📥 Baixar PDF
```

Totalmente compatíveis com:

* impressão
* Google Drive
* envio por email

---

# 📌 Segurança

* Senhas nunca são armazenadas em texto plano
* API Key fica em `secrets.toml`
* Tokens de recuperação expiram em 15 minutos
* Nada é armazenado no navegador do usuário
* IA só recebe o mínimo necessário para análise

---

# 🧭 Roadmap da v13.x

* [x] Novo detector de texto (robusto)
* [x] Seleção manual de coluna textual
* [x] Score por relevância
* [x] Chat IA melhorado
* [x] PDF profissional
* [ ] Tema escuro
* [ ] Exportação Excel consolidada
* [ ] Dashboard com Plotly
* [ ] Integração com Supabase
* [ ] Múltiplos perfis: Admin / Analista / Gestor
* [ ] Clusters automáticos nas respostas

---

# 👥 Equipe

**Desenvolvido por:**
Luciano Martins Fagundes

**Assistente técnico:**
ChatGPT — Build Assist Pro (2025)

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
---

# Versões anteriores

## 🧠 **Versão: 13.2**  

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
