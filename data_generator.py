"""
data_generator.py
Gera uma base sintética para a Plataforma AfirmAção (perfil + satisfação).
Função principal:
    generate_synthetic_afirmacao(n=300, seed=42) -> pd.DataFrame
"""
import numpy as np
import pandas as pd

def generate_synthetic_afirmacao(n=300, seed=42):
    """
    Gera DataFrame com colunas:
    - Idade (int)
    - Gênero (categorical)
    - Renda (categorical)
    - Escolaridade (categorical)
    - Tempo de Uso (meses) (int)
    - Satisfação Geral (1-5)
    - Facilidade de Uso (1-5)
    - Inclusão Percebida (1-5)
    - Confiança na Plataforma (1-5)
    - Engajamento (1-5 float)
    """
    rng = np.random.default_rng(seed)
    idade = rng.normal(30, 8, n).astype(int)
    genero = rng.choice(["Masculino", "Feminino", "Não-binário", "Prefiro não dizer"], size=n, p=[0.45, 0.45, 0.05, 0.05])
    renda = rng.choice(["Até 1 SM", "1–3 SM", "3–6 SM", "Acima de 6 SM"], size=n, p=[0.25, 0.35, 0.25, 0.15])
    escolaridade = rng.choice(["Fundamental", "Médio", "Superior", "Pós-graduação"], size=n, p=[0.1, 0.3, 0.4, 0.2])
    tempo_uso = rng.normal(12, 6, n).clip(1, 36).astype(int)

    # scores 1-5 (inteiros)
    satisfacao_geral = rng.integers(1, 6, n)
    facilidade_uso = rng.integers(1, 6, n)
    inclusao_percebida = rng.integers(1, 6, n)
    confianca_plataforma = rng.integers(1, 6, n)
    # engajamento com valores contínuos entre 1 e 5
    engajamento = rng.normal(3, 1, n).clip(1, 5).round(2)

    df = pd.DataFrame({
        "Idade": idade,
        "Gênero": genero,
        "Renda": renda,
        "Escolaridade": escolaridade,
        "Tempo de Uso (meses)": tempo_uso,
        "Satisfação Geral": satisfacao_geral,
        "Facilidade de Uso": facilidade_uso,
        "Inclusão Percebida": inclusao_percebida,
        "Confiança na Plataforma": confianca_plataforma,
        "Engajamento (1-5)": engajamento
    })

    # pequenas correlações simuladas: usuários com maior tempo de uso tendem a ter maior engajamento/satisfação
    df.loc[df["Tempo de Uso (meses)"] > df["Tempo de Uso (meses)"].median(), "Engajamento (1-5)"] += 0.2
    df["Engajamento (1-5)"] = df["Engajamento (1-5)"].clip(1,5).round(2)

    # Make categorical columns ordered if desired
    df["Escolaridade"] = pd.Categorical(df["Escolaridade"], categories=["Fundamental", "Médio", "Superior", "Pós-graduação"], ordered=True)

    return df

# Quick test when run directly
if __name__ == "__main__":
    df_test = generate_synthetic_afirmacao(10, seed=1)
    print(df_test.head())

