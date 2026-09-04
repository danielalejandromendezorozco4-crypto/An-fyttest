import datetime
from fpdf import FPDF
import pandas as pd
from config.settings import sanitizar_para_pdf

class InstitutionalReportPDF(FPDF):
    """
    Clase encapsulada de FPDF para maquetar el reporte institucional de 3 páginas.
    """
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)

    def draw_banner(self, title_text, subtitle_text):
        self.set_fill_color(10, 25, 47)  # Azul Marino Institucional (#0A192F)
        self.rect(0, 0, 210, 28, 'F')
        self.set_y(8)
        self.set_text_color(212, 175, 55)  # Dorado Champagne (#D4AF37)
        self.set_font("Arial", 'B', 15)
        self.cell(0, 6, txt=sanitizar_para_pdf(title_text), ln=True, align='C')
        self.set_text_color(255, 255, 255)
        self.set_font("Arial", 'I', 10)
        self.cell(0, 5, txt=sanitizar_para_pdf(subtitle_text), ln=True, align='C')
        self.set_text_color(51, 65, 85)
        self.ln(8)

    def draw_section_header(self, title):
        self.set_fill_color(226, 232, 240)
        self.set_font("Arial", 'B', 10)
        self.cell(0, 6, txt=sanitizar_para_pdf(f" {title}"), ln=True, fill=True)
        self.ln(2)

    def draw_row_3col(self, col1, val1, col2, val2, col3, val3):
        self.set_font("Arial", 'B', 8.5)
        self.cell(38, 5, txt=sanitizar_para_pdf(col1))
        self.set_font("Arial", '', 8.5)
        self.cell(24, 5, txt=sanitizar_para_pdf(val1))
        self.set_font("Arial", 'B', 8.5)
        self.cell(38, 5, txt=sanitizar_para_pdf(col2))
        self.set_font("Arial", '', 8.5)
        self.cell(24, 5, txt=sanitizar_para_pdf(val2))
        self.set_font("Arial", 'B', 8.5)
        self.cell(38, 5, txt=sanitizar_para_pdf(col3))
        self.set_font("Arial", '', 8.5)
        self.cell(28, 5, txt=sanitizar_para_pdf(val3), ln=True)

def generar_pdf_reporte(analysis_data: dict) -> bytes:
    """
    Función limpia de generación de PDF.
    Recibe un único diccionario estructurado (analysis_data) y devuelve los bytes del PDF.
    """
    pdf = InstitutionalReportPDF()
    
    ticker = analysis_data.get("ticker", "N/A")
    nombre = analysis_data.get("nombre", ticker)
    sector = analysis_data.get("sector", "General")
    pts = analysis_data.get("pts", 0)
    veredicto = analysis_data.get("veredicto", "")
    veredicto_txt = analysis_data.get("veredicto_txt", "")
    v_intr_dcf = analysis_data.get("v_intr_dcf", 0.0)
    precio_actual = analysis_data.get("precio_actual", 0.0)
    precio_max_compra = analysis_data.get("precio_max_compra", 0.0)
    desc_req = analysis_data.get("desc_req", 0.10)
    clase_msg = analysis_data.get("clase_msg", "")
    val_nde = analysis_data.get("val_nde", "N/A")
    val_cob = analysis_data.get("val_cob", "N/A")
    val_de = analysis_data.get("val_de", "N/A")
    roic = analysis_data.get("roic", 0.0)
    fcf_yield = analysis_data.get("fcf_yield", 0.0)
    tasa_libre_riesgo = analysis_data.get("tasa_libre_riesgo", 4.20)
    val_roe = analysis_data.get("val_roe", "N/A")
    roa = analysis_data.get("roa", 0.0)
    mg_op = analysis_data.get("mg_op", 0.0)
    fcf_conv = analysis_data.get("fcf_conv", 0.0)
    val_div_metric = analysis_data.get("val_div_metric", "N/A")
    val_ddm_str = analysis_data.get("val_ddm_str", "N/A")
    pe = analysis_data.get("pe", 0.0)
    p_fcf = analysis_data.get("p_fcf", 0.0)
    peg = analysis_data.get("peg", 0.0)
    texto_ia_final = analysis_data.get("texto_ia_final", "Análisis no disponible.")
    wacc = analysis_data.get("wacc", 9.0)
    g_term = analysis_data.get("g_term", 0.025)
    calcular_dcf_fn = analysis_data.get("calcular_dcf_fn")
    inc = analysis_data.get("inc", pd.DataFrame())
    cf = analysis_data.get("cf", pd.DataFrame())
    perfil_texto = analysis_data.get("perfil_texto", "Perfil corporativo no disponible.")

    # ===== PÁGINA 1: CUADRO DE MANDO =====
    pdf.add_page()
    pdf.draw_banner(
        f"An-FyT - REPORTE INSTITUCIONAL: {ticker}",
        f"{nombre} | Sector: {sector} | Fecha: {datetime.datetime.now().strftime('%Y-%m-%d')}"
    )

    pdf.set_fill_color(241, 245, 249)
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 7, txt=" VEREDICTO DE INVERSIÓN Y SCORE GLOBAL", ln=True, fill=True)
    pdf.set_font("Arial", '', 9)

    v_intr_txt = f"${v_intr_dcf:,.2f}"
    precio_txt = f"${precio_actual:,.2f}"
    recomendacion_txt = veredicto.replace('🟢','').replace('🔴','').replace('🟡','').replace('🟠','').replace('⛔','').strip()

    pdf.ln(2)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 5, txt=sanitizar_para_pdf(f" Puntuación Global: {pts}/100 Pts  |  Recomendación: {recomendacion_txt}"), ln=True)
    pdf.ln(2)

    pdf.set_font("Arial", '', 8.5)
    diag_narrativo = (
        f"DIAGNÓSTICO NARRATIVO INTEGRADO: La empresa {nombre} ({ticker}) cotiza actualmente a {precio_txt}, "
        f"frente a un Valor Intrínseco teórico de {v_intr_txt} (DCF) y un Precio Máximo de Compra sugerido de ${precio_max_compra:,.2f} "
        f"(descuento de seguridad del {desc_req*100:.0f}% por pertenecer a {clase_msg}). "
        f"En términos de solvencia, la firma registra una Deuda Neta/EBITDA de {val_nde} y una Cobertura de Intereses de {val_cob}. "
        f"En cuanto a rentabilidad operativa, genera un ROIC del {roic:.1f}% y un FCF Yield del {fcf_yield:.2f}% frente a una tasa libre de riesgo FRED del {tasa_libre_riesgo:.2f}%. "
        f"\n\nSíntesis Algorítmica: {veredicto_txt}"
    )
    pdf.multi_cell(0, 4.5, txt=sanitizar_para_pdf(diag_narrativo))
    pdf.ln(5)

    pdf.draw_section_header("M1. Solvencia y Liquidez")
    pdf.draw_row_3col("Deuda/EBITDA:", str(val_nde), "Cobertura Int:", str(val_cob), "Deuda/Capital:", str(val_de))
    pdf.ln(3)

    pdf.draw_section_header("M2. Rentabilidad y Dividendos (con FCF Yield vs FRED)")
    pdf.draw_row_3col("ROIC Operativo:", f"{roic:.1f}%", "ROE / ROA:", f"{val_roe} / {roa:.1f}%", "Mg. Operativo:", f"{mg_op:.1f}%")
    pdf.draw_row_3col("FCF Conversion:", f"{fcf_conv:.1f}x", "FCF Yield (FRED):", f"{fcf_yield:.2f}% ({tasa_libre_riesgo:.2f}%)", "Dividendos (Yield):", val_div_metric)
    pdf.ln(3)

    ev_ebitda_val = analysis_data.get("ev_ebitda", 0.0)
    p_s_val = analysis_data.get("p_s", 0.0)
    p_b_val = analysis_data.get("p_b", 0.0)

    pdf.draw_section_header("M3. Valuación y Múltiplos de Mercado")
    pdf.draw_row_3col("V. Int. DCF:", v_intr_txt, "V. Int. DDM:", val_ddm_str, "Precio Máx. C.:", f"${precio_max_compra:,.2f}")
    pdf.draw_row_3col("PER (P/E):", f"{pe:.1f}x" if pe > 0 else "N/A", "P/FCF:", f"{p_fcf:.1f}x" if p_fcf > 0 else "N/A", "PEG Forward:", f"{peg:.2f}x" if peg > 0 else "N/A")
    pdf.draw_row_3col("EV / EBITDA:", f"{ev_ebitda_val:.1f}x" if ev_ebitda_val > 0 else "N/A", "P/S (Ventas):", f"{p_s_val:.2f}x" if p_s_val > 0 else "N/A", "P/B (Libros):", f"{p_b_val:.2f}x" if p_b_val > 0 else "N/A")
    pdf.ln(3)

    pdf.draw_section_header("M5. Análisis Macroeconómico y Geopolítico (IA Gemini)")
    pdf.set_font("Arial", '', 8)
    pdf.multi_cell(0, 4, txt=sanitizar_para_pdf(texto_ia_final))

    # ===== PÁGINA 2: MATRIZ DE SENSIBILIDAD Y AUDITORÍA DE CAPITAL =====
    pdf.add_page()
    pdf.draw_banner(
        f"ANEXO DE SENSIBILIDAD Y ASIGNACIÓN DE CAPITAL ({ticker})",
        f"Matriz de Sensibilidad DCF y Auditoría Histórica (5 Años)"
    )

    pdf.draw_section_header("1. Matriz de Sensibilidad DCF Multiescenario (WACC vs g Terminal)")
    pdf.set_font("Arial", '', 8)
    pdf.cell(0, 4, txt=sanitizar_para_pdf("Escenarios de Valor Intrínseco por Acción ante variaciones macroeconómicas en tasa de descuento y crecimiento:"), ln=True)
    pdf.ln(2)

    wacc_vars = [wacc + 1.0, wacc, wacc - 1.0]
    g_vars = [g_term - 0.005, g_term, g_term + 0.005]

    pdf.set_font("Arial", 'B', 8)
    pdf.set_fill_color(241, 245, 249)
    pdf.cell(45, 6, txt=sanitizar_para_pdf("WACC \ g Terminal"), border=1, fill=True, align='C')
    pdf.cell(45, 6, txt=sanitizar_para_pdf(f"Conservador ({g_vars[0]*100:.1f}%)"), border=1, fill=True, align='C')
    pdf.cell(45, 6, txt=sanitizar_para_pdf(f"Base ({g_vars[1]*100:.1f}%)"), border=1, fill=True, align='C')
    pdf.cell(45, 6, txt=sanitizar_para_pdf(f"Optimista ({g_vars[2]*100:.1f}%)"), border=1, fill=True, align='C')
    pdf.ln(6)

    pdf.set_font("Arial", '', 8)
    wacc_names = [f"Exigente ({wacc+1.0:.1f}%)", f"Base Actual ({wacc:.1f}%)", f"Relajado ({wacc-1.0:.1f}%)"]
    for idx_w, w_v in enumerate(wacc_vars):
        pdf.set_font("Arial", 'B', 8)
        pdf.cell(45, 6, txt=sanitizar_para_pdf(wacc_names[idx_w]), border=1, align='C')
        pdf.set_font("Arial", '', 8)
        for g_v in g_vars:
            val_mat = calcular_dcf_fn(w_v, g_v) if callable(calcular_dcf_fn) else v_intr_dcf
            pdf.cell(45, 6, txt=sanitizar_para_pdf(f"${val_mat:,.2f}"), border=1, align='C')
        pdf.ln(6)
    pdf.ln(6)

    pdf.draw_section_header("2. Auditoría Histórica de Asignación de Capital (5 Años - Millones USD)")
    if isinstance(inc, pd.DataFrame) and isinstance(cf, pd.DataFrame) and not inc.empty and not cf.empty:
        try:
            pdf.set_font("Arial", 'B', 8)
            pdf.set_fill_color(241, 245, 249)
            cols_hist = inc.columns[:5]

            pdf.cell(40, 6, txt=sanitizar_para_pdf("Métrica \ Año"), border=1, fill=True, align='C')
            for c_h in cols_hist:
                yr_str = pd.to_datetime(c_h).year if not pd.isna(c_h) else str(c_h)
                pdf.cell(28, 6, txt=sanitizar_para_pdf(str(yr_str)), border=1, fill=True, align='C')
            pdf.ln(6)

            pdf.set_font("Arial", 'B', 8)
            pdf.cell(40, 5, txt=sanitizar_para_pdf("Ingresos Totales"), border=1)
            pdf.set_font("Arial", '', 8)
            for c_h in cols_hist:
                rev_v = inc.loc['Total Revenue', c_h] / 1e6 if 'Total Revenue' in inc.index and c_h in inc.columns else 0
                pdf.cell(28, 5, txt=sanitizar_para_pdf(f"${rev_v:,.0f}"), border=1, align='C')
            pdf.ln(5)

            pdf.set_font("Arial", 'B', 8)
            pdf.cell(40, 5, txt=sanitizar_para_pdf("Utilidad Neta"), border=1)
            pdf.set_font("Arial", '', 8)
            for c_h in cols_hist:
                ni_v = inc.loc['Net Income', c_h] / 1e6 if 'Net Income' in inc.index and c_h in inc.columns else 0
                pdf.cell(28, 5, txt=sanitizar_para_pdf(f"${ni_v:,.0f}"), border=1, align='C')
            pdf.ln(5)

            pdf.set_font("Arial", 'B', 8)
            pdf.cell(40, 5, txt=sanitizar_para_pdf("Flujo Libre (FCF)"), border=1)
            pdf.set_font("Arial", '', 8)
            for c_h in cols_hist:
                fcf_v = cf.loc['Free Cash Flow', c_h] / 1e6 if 'Free Cash Flow' in cf.index and c_h in cf.columns else 0
                pdf.cell(28, 5, txt=sanitizar_para_pdf(f"${fcf_v:,.0f}"), border=1, align='C')
            pdf.ln(5)
        except Exception:
            pdf.cell(0, 5, txt=sanitizar_para_pdf("Auditoría de asignación de capital no disponible para este ticker."), ln=True)
    pdf.ln(8)

    # ===== PÁGINA 3: TESIS DE NEGOCIO (PERFIL IA) =====
    pdf.add_page()
    pdf.draw_banner(
        f"PERFIL CORPORATIVO Y TESIS DE NEGOCIO ({ticker})",
        f"Investigación Institucional Asistida por Inteligencia Artificial"
    )

    pdf.set_font("Arial", '', 12)
    perfil_saneado = sanitizar_para_pdf(perfil_texto)
    pdf.multi_cell(0, 6, txt=perfil_saneado, align='J')
    pdf.ln(8)
    pdf.set_line_width(0.4)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    pdf.set_font("Arial", 'B', 7.5)
    pdf.cell(0, 4, txt=sanitizar_para_pdf("DESCARGO DE RESPONSABILIDAD LEGAL Y REGULATORIO:"), ln=True)
    pdf.set_font("Arial", '', 6.5)
    disclaimer_text = (
        "Este documento fue generado de forma automatizada mediante algoritmos avanzados de valuación y modelos de inteligencia artificial "
        "basados estrictamente en información financiera pública. Las métricas, calificaciones, rangos de sensibilidad y valores intrínsecos "
        "calculados representan únicamente una evaluación matemática teórica y NO constituyen bajo ninguna circunstancia una asesoría financiera, "
        "recomendación de inversión institucional, ni incitación a la compra o venta de activos bursátiles. Toda inversión en los mercados de valores "
        "conlleva un nivel sustancial de riesgo y la posibilidad de perder parcial o totalmente el capital aportado. Realice su propia investigación "
        "(DYOR) o consulte con un asesor financiero profesional y regulado antes de tomar decisiones de inversión."
    )
    pdf.multi_cell(0, 3, txt=sanitizar_para_pdf(disclaimer_text), align='J')

    raw_pdf = pdf.output(dest='S')
    if isinstance(raw_pdf, bytes):
        return raw_pdf
    elif isinstance(raw_pdf, bytearray):
        return bytes(raw_pdf)
    elif isinstance(raw_pdf, str):
        return raw_pdf.encode('latin-1', 'ignore')
    return bytes(raw_pdf)
