# streamlit_aps_exploratory_app.py
# Streamlit app to generate a fictitious dataset inspired by
# https://id-preview--6b3eb15a-ff95-44dd-bde4-47472b04b24a.lovable.app/
# and run the analyses requested in the APS assignment:
# - Descriptive statistics (mean, median, mode, variance, sd, range, quartiles)
# - Normal distribution overlay
# - Inferential tests: independent t-test, paired t-test, ANOVA, Pearson correlation
# - Export dataset (CSV) and generate a simple PDF report with key results and figures

import io
import base64
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy import stats
import streamlit as st

# ----------------------------- Helpers ---------------------------------

def generate_fictitious_dataset(n=200, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Simulate demographics
    user_id = np.arange(1, n + 1)
    age = rng.normal(32, 8, n).round().astype(int)
    gender = rng.choice(["Male", "Female", "Other"], size=n, p=[0.48, 0.5, 0.02])
    # Three experimental groups (A, B, C) for ANOVA / independent tests
    group = rng.choice(["A", "B", "C"], size=n, p=[0.35, 0.35, 0.3])
    # Hours of practice (skewed): useful for correlation
    hours_practice = np.abs(rng.normal(5, 3, n)).round(1)
    attendance_rate = np.clip(rng.normal(0.87, 0.12, n), 0, 1).round(2)

    # Pre and post scores (paired) with group effects
    base_skill = np.clip(rng.normal(60, 12, n), 20, 100)
    # Group effects A: small improvement, B: medium, C: larger
    group_effect = np.array([5 if g == 'A' else (8 if g == 'B' else 12) for g in group])
    noise = rng.normal(0, 6, n)
    score_pre = np.clip(base_skill + rng.normal(0, 4, n), 0, 100).round(1)
    score_post = np.clip(score_pre + group_effect + (hours_practice * 0.8) / 5 + noise, 0, 100).round(1)

    df = pd.DataFrame({
        "user_id": user_id,
        "age": age,
        "gender": gender,
        "group": group,
        "hours_practice": hours_practice,
        "attendance_rate": attendance_rate,
        "score_pre": score_pre,
        "score_post": score_post,
    })
    return df


def descriptive_statistics(series: pd.Series) -> dict:
    vals = series.dropna()
    desc = {
        "count": int(vals.count()),
        "mean": float(vals.mean()),
        "median": float(vals.median()),
        "mode": list(vals.mode().values) if not vals.mode().empty else [],
        "variance": float(vals.var(ddof=1)),
        "std": float(vals.std(ddof=1)),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "range": float(vals.max() - vals.min()),
        "q1": float(vals.quantile(0.25)),
        "q3": float(vals.quantile(0.75)),
        "iqr": float(vals.quantile(0.75) - vals.quantile(0.25)),
    }
    return desc


def plot_hist_with_normal(series: pd.Series, title: str = "Histogram") -> plt.Figure:
    vals = series.dropna()
    mu, sigma = vals.mean(), vals.std(ddof=1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(vals, bins=20, density=True, alpha=0.6)
    # plot theoretical normal curve
    x = np.linspace(vals.min(), vals.max(), 200)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), linestyle='--')
    ax.set_title(title)
    ax.set_xlabel(series.name)
    ax.set_ylabel('Density')
    plt.tight_layout()
    return fig


def run_inferential_tests(df: pd.DataFrame, group_col: str = "group") -> dict:
    results = {}
    # Independent t-test between group A and B on score_post
    a = df[df[group_col] == 'A']['score_post']
    b = df[df[group_col] == 'B']['score_post']
    t_ind, p_ind = stats.ttest_ind(a, b, equal_var=False, nan_policy='omit')
    results['t_independent_A_vs_B'] = {"t_stat": float(t_ind), "p_value": float(p_ind)}

    # Paired t-test between pre and post (all participants)
    paired = stats.ttest_rel(df['score_post'], df['score_pre'], nan_policy='omit')
    results['t_paired_pre_vs_post'] = {"t_stat": float(paired.statistic), "p_value": float(paired.pvalue)}

    # ANOVA across groups A, B, C on score_post
    groups = [df[df[group_col] == g]['score_post'] for g in ['A', 'B', 'C']]
    f_val, p_anova = stats.f_oneway(*groups)
    results['anova_A_B_C'] = {"f_stat": float(f_val), "p_value": float(p_anova)}

    # Pearson correlation between hours_practice and score_post
    rho, p_corr = stats.pearsonr(df['hours_practice'], df['score_post'])
    results['pearson_hours_scorepost'] = {"r": float(rho), "p_value": float(p_corr)}

    return results


def fig_to_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return buf.read()


def create_simple_pdf_report(df: pd.DataFrame, figs: dict, stats_summary: dict, filename: str = "report.pdf") -> bytes:
    # Create a lightweight PDF containing text and the provided PNG figures
    try:
        from fpdf import FPDF
    except Exception as e:
        raise RuntimeError("fpdf library is required to generate PDF. Install with `pip install fpdf`.")

    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 8, 'Relat\u00f3rio Exploratorio - APS (Fict\u00edcio)', ln=True)
    pdf.ln(4)

    pdf.set_font('Arial', '', 11)
    # Introduction
    pdf.multi_cell(0, 6, f"Base de dados: fict\u00edcia (gerada a partir de https://id-preview--6b3eb15a-ff95-44dd-bde4-47472b04b24a.lovable.app/)\nObserva\u00e7\u00f5es: {len(df)} | Vari\u00e1veis: {df.shape[1]}")
    pdf.ln(3)

    # Insert small descriptive table
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 6, 'Resumo das Estat\u00edsticas Descritivas', ln=True)
    pdf.set_font('Arial', '', 10)
    for k, v in stats_summary.items():
        pdf.multi_cell(0, 5, f"{k}: {v}")

    # Add figures
    for title, fig_bytes in figs.items():
        pdf.add_page()
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 6, title, ln=True)
        pdf.ln(4)
        # Save image to temp then insert
        img_path = f"/tmp/{title.replace(' ', '_')}.png"
        with open(img_path, 'wb') as f:
            f.write(fig_bytes)
        pdf.image(img_path, w=180)

    return pdf.output(dest='S').encode('latin-1')

# ----------------------------- Streamlit app ----------------------------

def main():
    st.set_page_config(page_title='APS - Exploratory Analysis', layout='wide')
    st.title('APS - Relat\u00f3rio de An\u00e1lise Explorat\u00f3ria (fict\u00edcio)')

    st.markdown(
        """
        **Objetivo:** gerar uma base fictícia inspirada na URL fornecida e realizar as análises
        descritas na APS de Fundamentos da Estatística e Análise de Dados.
        """
    )

    # Controls
    st.sidebar.header('Gerar base fict\u00edcia')
    n = st.sidebar.slider('Número de observações', 50, 2000, 200, step=50)
    seed = st.sidebar.number_input('Semente (seed)', value=42)
    if st.sidebar.button('Gerar base'):
        df = generate_fictitious_dataset(n=n, seed=seed)
        st.session_state['df'] = df
    elif 'df' not in st.session_state:
        st.session_state['df'] = generate_fictitious_dataset(n=n, seed=seed)

    df = st.session_state['df']

    st.subheader('Introdução: Base de dados (fict\u00edcia)')
    st.write('Origem: gerada automaticamente a partir do prompt do aluno (inspirada na URL indicada).')
    st.write(f'Observações: {len(df)}  —  Variáveis: {df.shape[1]}')
    with st.expander('Visualizar tabela (primeiras 50 linhas)'):
        st.dataframe(df.head(50))

    # Descriptive analysis
    st.subheader('1) An\u00e1lise Estat\u00edstica Descritiva')
    col1, col2 = st.columns([2, 1])

    with col1:
        var = st.selectbox('Escolha uma variável num\u00e9rica para descrever', options=['age', 'hours_practice', 'attendance_rate', 'score_pre', 'score_post'])
        desc = descriptive_statistics(df[var])
        st.table(pd.DataFrame.from_dict(desc, orient='index', columns=['valor']))

        # Histogram + normal curve
        fig = plot_hist_with_normal(df[var], title=f'Histograma e curva Normal: {var}')
        st.pyplot(fig)

        # Shapiro test for normality (note: for large n shapiro may fail)
        try:
            shapiro_res = stats.shapiro(df[var].dropna())
            st.write(f"Shapiro-Wilk: W={shapiro_res.statistic:.4f}, p-value={shapiro_res.pvalue:.4e}")
        except Exception:
            st.write('Shapiro-Wilk não aplicável para n grande (>5000) ou falha no teste.')

    with col2:
        st.write('Medidas adicionais')
        st.write('Boxplot: identifica outliers')
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        ax2.boxplot(df[var].dropna(), vert=False)
        ax2.set_xlabel(var)
        st.pyplot(fig2)

    # Inferential
    st.subheader('2) An\u00e1lise Estat\u00edstica Inferencial')
    st.write('Testes realizados automaticamente: t-independente (A vs B), t-pareado (pre vs post), ANOVA (A,B,C), Correla\u00e7\u00e3o de Pearson (hours_practice x score_post)')
    if st.button('Executar testes inferenciais'):
        infer = run_inferential_tests(df)
        st.json(infer)

        # Show small plots supporting the inferences
        # Paired differences histogram
        diff = df['score_post'] - df['score_pre']
        fig3, ax3 = plt.subplots(figsize=(6, 3))
        ax3.hist(diff, bins=20)
        ax3.set_title('Distribui\u00e7\u00e3o das diferen\u00e7as (post - pre)')
        st.pyplot(fig3)

        # Boxplot by group
        fig4, ax4 = plt.subplots(figsize=(6, 3))
        groups = [df[df['group'] == g]['score_post'] for g in ['A', 'B', 'C']]
        ax4.boxplot(groups, labels=['A', 'B', 'C'])
        ax4.set_title('Score post por grupo (A, B, C)')
        st.pyplot(fig4)

        # Scatter for correlation
        fig5, ax5 = plt.subplots(figsize=(6, 3))
        ax5.scatter(df['hours_practice'], df['score_post'])
        ax5.set_xlabel('hours_practice')
        ax5.set_ylabel('score_post')
        ax5.set_title('Horas de pr\u00e1tica x score_post')
        # add linear fit
        m, b = np.polyfit(df['hours_practice'], df['score_post'], 1)
        x = np.linspace(df['hours_practice'].min(), df['hours_practice'].max(), 100)
        ax5.plot(x, m * x + b, linestyle='--')
        st.pyplot(fig5)

    # Conclusion and export
    st.subheader('3) Conclus\u00e3o')
    st.markdown(
        """
        - Resuma os principais achados no texto acima (use interpreta\u00e7\u00f5es das sa\u00eddas exibidas).  
        - Limita\u00e7\u00f5es: base fict\u00edcia, poss\u00edvel vi\u00e9s de gera\u00e7\u00e3o, tamanho da amostra, vari\u00e1veis limitadas.  
        """
    )

    st.sidebar.header('Exportar')
    csv = df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button('Baixar CSV da base (fict\u00edcia)', data=csv, file_name='base_ficticia.csv', mime='text/csv')

    # Create PDF report
    if st.sidebar.button('Gerar PDF simples (relat\u00f3rio)'):
        # create figures and stats summary
        figs = {}
        figs['histogram_var'] = fig_to_bytes(plot_hist_with_normal(df['score_post'], 'Histograma - score_post'))
        figs['box_groups'] = fig_to_bytes(plt.figure())
        # simple stats summary text
        stats_summary = {
            'n_obs': len(df),
            'mean_score_post': round(float(df['score_post'].mean()), 3),
            'mean_score_pre': round(float(df['score_pre'].mean()), 3),
        }
        try:
            pdf_bytes = create_simple_pdf_report(df, figs, stats_summary)
            st.sidebar.download_button('Baixar PDF', data=pdf_bytes, file_name='relatorio_aps_ficticio.pdf', mime='application/pdf')
        except RuntimeError as e:
            st.sidebar.error(str(e))

    st.info('Sugest\u00f5es: use os gr\u00e1ficos e tabelas desta aplica\u00e7\u00e3o para montar o relat\u00f3rio final (PDF/Word) seguindo a estrutura exigida pela APS.')

    st.caption('App gerado automaticamente — base fict\u00edcia criada para fins pedag\u00f3gicos.')


if __name__ == '__main__':
    main()
