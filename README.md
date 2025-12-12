
# 🧠 REPARA Analytics — Plataforma Inteligente de Análise de Talentos  
**Versão:** 13.5.1 (2025)  
**Tecnologias:** Streamlit + Gemini AI + Python + Wordcloud + Pandas + ReportLab  
**Ambiente:** Compatível com Streamlit Cloud

---

# 🚀 O que é o REPARA Analytics?

O **REPARA Analytics** é uma plataforma inteligente projetada para analisar dados de candidatos e empresas a partir de arquivos CSV e gerar **insights automáticos** com apoio de IA (Gemini 2.5 Flash).

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


# 🚀 **O que há de novo na versão 13.5.1**

A versão **13.5.1** é a mais estável e refinada até agora — com grandes melhorias na interface, segurança e experiência do usuário.

---

## 🆕 **Novidades & Alterações da v13.5.1**

### **1️⃣ Remoção total do `st.experimental_rerun()` dentro de diálogos**

Problema recorrente que quebrava o app:

```
st.experimental_rerun() inside st.dialog → ERRO
```

**Correções implementadas:**

* Introdução de flags internas:

  * `st.session_state._rerun`
  * `st.session_state._chat_rerun`
* Rerun seguro ocorre **no fim do app**, nunca dentro de diálogos ou callbacks.
* Chat IA agora funciona **sem travar**.

---

### **2️⃣ Modelo de IA especializado em DEI (Diversidade, Equidade e Inclusão)**

Agora todas as análises seguem um **super prompt mestre DEI**, alinhado a:

* ONU
* ODS
* Políticas Públicas do Brasil
* Ações afirmativas
* OIT
* Barreiras estruturais (racial, gênero, etária, territorial, socioeconômica)

A IA sempre responde em **português do Brasil**, com recomendações práticas e sensíveis.

**Impactos diretos:**

* Insights mais humanos
* Menos vieses
* Análises contextualizadas ao Brasil
* Respostas totalmente PT-BR (corrigido)

---

### **3️⃣ Wordcloud Inteligente 2.0**

Sem SpaCy → totalmente compatível com Streamlit Cloud.

📌 **Vantagens:**

* Extração inteligente de:

  * **Verbros** (peso 3)
  * **Adjetivos** (peso 2)
  * **Substantivos** (peso 2)
* Lematização simples (português)
* Remoção de números, ruídos e palavras inúteis
* Stopwords enriquecidas
* Tokenização robusta

📌 **Novidades da v13.5.1**

* Seletor de tema **Light / Dark**
* Fundo e estética aprimorados
* Wordcloud muito mais limpo e relevante

---

### **4️⃣ Nova ABA: 🔐 Recuperação de Senha**

Agora o fluxo está completo:

#### ➤ **Gerar Token (admin)**

* Cria tokens válidos por 15 minutos
* Ideal para suporte e fluxo de produção

#### ➤ **Redefinir Senha**

* Usuário insere token
* Cria nova senha
* Gera bloco TOML pronto para colar em `secrets.toml`

---

### **5️⃣ Login atualizado com UI elegante**

* Card visual com gradient sutil
* Explicação da ferramenta
* Texto orientativo claro
* Botão de recuperação redireciona corretamente para a nova aba
* Sem reruns inesperados

---

### **6️⃣ Chat IA completamente reestruturado**

* Não utiliza mais `experimental_rerun()` (causava crash)
* Usa rerun seguro via flag
* Histórico persistente e organizado
* Todas as respostas seguem o **modo DEI especialista**

---

### **7️⃣ Detecção de colunas textuais aprimorada**

Nova heurística utilizando:

* % de linhas com texto
* tamanho médio
* variedade de termos
* score combinado

Agora o app:

* detecta mais corretamente colunas relevantes
* evita falsos positivos
* funciona mesmo com CSVs "sujos"

---

### **8️⃣ Painel Admin aprimorado**

* Interface mais clara
* Geração de usuários TOML mais intuitiva
* Geração de hashes isolados

---

### **9️⃣ Código mais estável e seguro**

* Revisão completa das chamadas de rerenderização
* Tratamento de CSVs mais robusto
* Remoção de retornos inesperados nos fluxos de lógica
* Melhorias de performance no Wordcloud

---

# ⚙️ Como Rodar Localmente

### **1. Crie um ambiente virtual**

```
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### **2. Instale as dependências**

```
pip install -r requirements.txt
```

### **3. Adicione suas credenciais**

No arquivo:

```
.streamlit/secrets.toml
```

Exemplo:

```toml
GOOGLE_API_KEY = "sua_chave"

[users.admin]
name = "Administrador"
email = "admin@repara.com"
password = "HASH..."
```

### **4. Rode o app**

```
streamlit run app.py
```

---

# 📁 Estrutura do Projeto

```
📦 repara-analytics
 ┣ 📂 .streamlit
 ┃ ┗ 📜 secrets.toml
 ┣ 📜 requirements.txt
 ┣ 📜 app.py
 ┗ 📜 README.md
```

---

# 🧠 Funcionalidades Principais

### **✔ Upload de CSVs (Candidatos / Empresas)**

Leitura robusta (`,`, `;`, `|`, `\t`)

### **✔ Wordcloud Inteligente**

Focado em verbos, adjetivos e substantivos.

### **✔ Análise IA (DEI Specialist)**

* Candidatos
* Empresas
* Cruzada
* Todos geram PDF

### **✔ Chat IA com contexto**

Memória do chat + dados dos CSVs incluídos no prompt.

### **✔ KPIs automáticos**

### **✔ Login seguro**

Criptografia PBKDF2-SHA256

### **✔ Recuperação de senha com tokens**

### **✔ Painel Admin**

---

# 📦 Deploy no Streamlit Cloud

Basta publicar o repositório e incluir:

* `requirements.txt`
* `secrets.toml`

Tudo 100% compatível.

---

# 🤝 Contribuições

Sinta-se livre para sugerir melhorias UI/UX, novas análises ou integrações.

---

# 🏁 Final

A versão **13.5.1** entrega:

* mais estabilidade
* mais segurança
* mais inteligência
* mais acessibilidade
* mais contexto social aplicado


---
---

# ✨ Novidades da Versão 13.4.1

### 🎨 **1. Wordcloud Inteligente (NLTK + heurísticas + PT-BR)**  
Totalmente reescrita e agora:

- Filtra *somente palavras relevantes*: verbos, substantivos e adjetivos  
- Remove ruídos, termos vazios e pronomes  
- Faz lematização leve  
- Classifica palavras por relevância e peso  
- Inclui seletor de **tema Light ou Dark** elegante  
- 100% compatível com Streamlit Cloud (sem spaCy)

### 💬 **2. IA aprimorada — Responde sempre em Português**  
O prompt foi reescrito, agora 100% PT-BR:

- Resumo executivo  
- Temas principais  
- Análise de sentimentos  
- Pontos de dor e oportunidades  
- Recomendações práticas  
- Tabela “Tema | Exemplo | Impacto | Ação recomendada”  

### 🔐 **3. Tela de Login com UI elegante + explicação simples e clara**  
Interface redesenhada usando:

- Modal `st.dialog`
- Cartão visual moderno
- Explicação minimalista de como a ferramenta funciona
- Suporte a recuperação de senha com token temporário

### 🛡️ **4. Painel Admin refinado**
- Geração de hashes PBKDF2-SHA256  
- Criação de blocos TOML prontos para secrets  
- Gerenciamento simples e seguro

### 🌗 **5. Wordcloud Theme Switch (Light / Dark)**  
- Tema "Light": fundo branco profissional  
- Tema "Dark": fundo #0b1220 elegante, ideal para telões

### 🧩 **6. Refatorações gerais**
- Remoção total de `experimental_rerun()`  
- Navegação 100% estável com `session_state._rerun`  
- Detecção de colunas textuais mais robusta  
- Compatibilidade total com Streamlit Cloud  
- Melhorias no Chat IA e no módulo PDF  

---

# 🛠️ Tecnologias Utilizadas

- **Streamlit 1.39+**
- **Google Gemini 2.5 Flash**
- **NLTK (stopwords e tokenização leve em PT-BR)**
- **Passlib (hash PBKDF2-SHA256)**
- **ReportLab (exportação PDF)**
- **Matplotlib + Wordcloud**
- **Pandas**

---

# 📦 Instalação

### **1. Clone o repositório**
```bash
git clone https://github.com/seu-user/repara-analytics
cd repara-analytics
````

### **2. Instale dependências**

```bash
pip install -r requirements.txt
```

### **3. Execute**

```bash
streamlit run app.py
```

---

# 🧩 Configuração do `secrets.toml` (Streamlit Cloud)

Crie dentro de `.streamlit/secrets.toml`:

```toml
GOOGLE_API_KEY = "SUA_CHAVE"

[users.admin]
name = "Administrador"
email = "admin@repara.com"
password = "$pbkdf2-sha256$..."

[users.luciano]
name = "Luciano Martins"
email = "luciano@exemplo.com"
password = "$pbkdf2-sha256$..."
```

Para gerar hashes:

```python
from passlib.context import CryptContext
PWD = CryptContext(schemes=["pbkdf2_sha256"])
print(PWD.hash("SUA_SENHA"))
```

---

# 🖥️ Deploy no Streamlit Cloud

1. Faça commit de:

   * `app.py`
   * `requirements.txt`
   * `README.md`

2. Acesse: [https://streamlit.io/cloud](https://streamlit.io/cloud)

3. Crie um novo app.

4. Em **Settings → Secrets**, cole seu `secrets.toml`.

5. Rodará automaticamente 🎉

---

# 📁 Estrutura do Projeto

```
📦 repara-analytics
│
├── app.py              # aplicação completa v13.4.1
├── requirements.txt    # dependências para Streamlit Cloud
└── README.md
```

---

# 🧱 Arquitetura da Aplicação

### 🔐 Autenticação

* Login via modal (`st.dialog`)
* Tokens temporários
* Hash PBKDF2-SHA256
* Painel Admin restrito

### 📥 Processamento CSV

* Autodetector de delimitador
* Normalizador de colunas
* Detecção de texto baseada em estatísticas
* Preparação limpa para IA

### 🎨 Wordcloud Inteligente

* Tokenização PT-BR
* Stopwords da NLTK + lista customizada
* Lematização leve
* Classificação gramatical (verbo / adjetivo / substantivo)
* Tema Light / Dark
* Renderização elegante

### 🤖 IA (Gemini 2.5 Flash)

* Prompt 100% português
* Relatórios estruturados
* Exportação PDF

### 🔀 Cruzada

* Junta textos de candidatos e empresas
* IA gera visão integrada
* PDF

### 💬 Chat IA

* Contexto automático dos CSVs
* Histórico persistente

---

# 📊 KPIs

* Contagem de candidatos
* Contagem de empresas
* Colunas textuais detectadas
* Preview dos CSVs

---

# 📄 Exportação PDF

Relatórios profissionais gerados com:

* Títulos padronizados
* Conteúdo rico
* Download instantâneo

---

# 🔒 Segurança

* Senhas nunca armazenadas em texto puro
* Hash PBKDF2-SHA256
* Reset de senha via token
* Segredos isolados no `secrets.toml`
* IA recebe apenas o mínimo necessário

---

# 🧭 Roadmap Futuro (13.5+)

* [ ] Tema escuro completo no app
* [ ] Dashboard com Plotly
* [ ] Agrupamento temático automático (clustering)
* [ ] Exportação consolidada Excel
* [ ] Múltiplos perfis (Analista / Gestor / Admin)
* [ ] Integração Supabase

---

# 👥 Equipe

**Desenvolvido por:**
Luciano Martins Fagundes

**Assistente técnico:**
ChatGPT — Build Assist Pro (2025)

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
