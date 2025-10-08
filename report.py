"""
report.py
Geração de PDF final com texto e imagens (utiliza FPDF).
create_pdf_report(df, variavel, stats, infer_results, image_bytes_dict, n) -> bytes
"""

from fpdf import FPDF
from io import BytesIO

def create_pdf_report(df, variavel, stats_dict, infer_results, image_bytes_dict, n):
    """
    Gera relatório PDF da análise estatística com suporte a UTF-8 (fpdf2).
    """

    pdf = FPDF(orientation="P", unit="mm", format="A4")
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
    for test_name, res in infer_results.items():
        pdf.multi_cell(0, 5, f"{test_name}: {res}")
    pdf.ln(4)

    # Inserção de imagens geradas
    for img_name, img_bytes in image_bytes_dict.items():
        pdf.add_page()
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, img_name, ln=True)
        img_stream = BytesIO(img_bytes)
        pdf.image(img_stream, x=15, y=None, w=170)
        pdf.ln(10)

    # Exporta para bytes
    out = BytesIO()
    pdf_bytes = pdf.output(dest="S").encode("latin-1", "replace")
    out.write(pdf_bytes)
    out.seek(0)
    return out.read()

