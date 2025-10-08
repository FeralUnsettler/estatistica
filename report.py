"""
report.py
Geração de PDF final com texto e imagens (utiliza FPDF).
create_pdf_report(df, variavel, stats, infer_results, image_bytes_dict, n) -> bytes
"""

from fpdf import FPDF
from io import BytesIO
import os

def create_pdf_report(df, variavel, stats_dict, infer_results, image_bytes_dict, n):
    """
    Gera relatório PDF da análise estatística com suporte a Unicode (DejaVuSans.ttf)
    e fallback automático para Arial se a fonte não for encontrada.
    """

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Verifica se a fonte Unicode está disponível
    unicode_font = "DejaVuSans.ttf"
    if os.path.exists(unicode_font):
        pdf.add_font("DejaVu", "", unicode_font, uni=True)
        pdf.set_font("DejaVu", "", 12)
        unicode_enabled = True
    else:
        pdf.set_font("Arial", "", 12)
        unicode_enabled = False

    def safe_text(txt):
        txt = str(txt)
        # substitui travessões por hífens simples
        txt = txt.replace("—", "-").replace("–", "-")
        return txt

    # Cabeçalho
    pdf.set_font("DejaVu" if unicode_enabled else "Arial", "B", 14)
    pdf.cell(0, 10, safe_text("Relatório de Análise - Plataforma AfirmAção"), ln=True, align="C")
    pdf.ln(4)

    pdf.set_font("DejaVu" if unicode_enabled else "Arial", size=11)
    pdf.multi_cell(0, 6, safe_text(f"Base sintética gerada — n = {n} observações"))
    pdf.ln(2)

    # Estatísticas descritivas
    pdf.set_font("DejaVu" if unicode_enabled else "Arial", "B", 12)
    pdf.cell(0, 6, safe_text("Estatísticas Descritivas"), ln=True)
    pdf.set_font("DejaVu" if unicode_enabled else "Arial", size=10)
    for k, v in stats_dict.items():
        pdf.multi_cell(0, 5, safe_text(f"{k}: {v}"))
    pdf.ln(2)

    # Resultados inferenciais
    pdf.set_font("DejaVu" if unicode_enabled else "Arial", "B", 12)
    pdf.cell(0, 6, safe_text("Resultados Inferenciais"), ln=True)
    pdf.set_font("DejaVu" if unicode_enabled else "Arial", size=10)
    for test_name, res in infer_results.items():
        pdf.multi_cell(0, 5, safe_text(f"{test_name}: {res}"))
    pdf.ln(4)

    # Adiciona imagens (gráficos, histogramas etc.)
    for img_name, img_bytes in image_bytes_dict.items():
        pdf.add_page()
        pdf.set_font("DejaVu" if unicode_enabled else "Arial", "B", 12)
        pdf.cell(0, 8, safe_text(img_name), ln=True)
        img_stream = BytesIO(img_bytes)
        pdf.image(img_stream, x=15, y=None, w=170)
        pdf.ln(10)

    # Gera bytes PDF
    out = BytesIO()
    pdf_bytes = pdf.output(dest="S").encode("latin-1", "replace")
    out.write(pdf_bytes)
    out.seek(0)
    return out.read()