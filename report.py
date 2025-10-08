"""
report.py
Geração de PDF final com texto e imagens (utiliza FPDF).
create_pdf_report(df, variavel, stats, infer_results, image_bytes_dict, n) -> bytes
"""

import tempfile
import os
from fpdf import FPDF
from io import BytesIO

def create_pdf_report(df, variavel, stats_dict, infer_results, image_bytes_dict, n):
    """
    - df: DataFrame (used for metadata)
    - variavel: variável analisada (string)
    - stats_dict: dict com estatísticas descritivas
    - infer_results: dict com resultados dos testes inferenciais
    - image_bytes_dict: dict {title: png_bytes}
    - n: int, tamanho da amostra
    Retorna: bytes do PDF
    """
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Relatório de Análise - Plataforma AfirmAção", ln=True, align="C")
    pdf.ln(4)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 6, f"Base sintética gerada — n = {n} observações")
    pdf.ln(2)

    # Estatísticas descritivas
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 6, "Estatísticas Descritivas", ln=True)
    pdf.set_font("Arial", size=10)
    for k, v in stats_dict.items():
        pdf.multi_cell(0, 5, f"{k}: {v}")
    pdf.ln(2)

    # Resultados inferenciais
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 6, "Resultados Inferenciais", ln=True)
    pdf.set_font("Arial", size=10)
    # Pretty print infer_results
    for test_name, res in infer_results.items():
        pdf.multi_cell(0, 5, f"{test_name}: {res}")
    pdf.ln(4)

    # Insert images: write bytes to temp files and insert
    tmpdir = tempfile.mkdtemp()
    try:
        for title, img_bytes in image_bytes_dict.items():
            img_path = os.path.join(tmpdir, f"{title[:30].replace(' ','_')}.png")
            with open(img_path, "wb") as f:
                f.write(img_bytes)
            pdf.add_page()
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 6, title, ln=True)
            pdf.ln(4)
            # Calculate width maintain aspect: we'll use w=180mm max
            pdf.image(img_path, w=180)
    finally:
        # cleanup temp files
        for fname in os.listdir(tmpdir):
            try:
                os.remove(os.path.join(tmpdir, fname))
            except Exception:
                pass
        try:
            os.rmdir(tmpdir)
        except Exception:
            pass

    out = BytesIO()
    pdf.output(out)
    out.seek(0)
    return out.read()

