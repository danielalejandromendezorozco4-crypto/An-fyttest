import unittest
import pandas as pd
from reports.pdf_generator import generar_pdf_reporte

def test_generar_pdf_reporte_bytes():
    """
    Verifica que generar_pdf_reporte procese un diccionario de datos completo y devuelva bytes de PDF válidos.
    """
    mock_data = {
        "ticker": "AAPL",
        "nombre": "Apple Inc.",
        "sector": "Technology",
        "pts": 85,
        "veredicto": "🟢 COMPRA FUERTE",
        "veredicto_txt": "Empresa sólida con excelente retorno sobre el capital y bajo endeudamiento.",
        "v_intr_dcf": 220.50,
        "precio_actual": 175.00,
        "precio_max_compra": 198.45,
        "desc_req": 0.10,
        "clase_msg": "Clase A (Alta Calidad)",
        "val_nde": "0.5x",
        "val_cob": "25.0x",
        "val_de": "0.8x",
        "roic": 28.5,
        "fcf_yield": 5.2,
        "tasa_libre_riesgo": 4.20,
        "val_roe": "145.0%",
        "roa": 22.0,
        "mg_op": 30.5,
        "fcf_conv": 1.1,
        "val_div_metric": "$1.00 (0.57%)",
        "val_ddm_str": "$120.00",
        "pe": 28.5,
        "p_fcf": 24.0,
        "peg": 1.8,
        "ev_ebitda": 21.5,
        "p_s": 7.8,
        "p_b": 35.0,
        "gross_margin": 45.0,
        "net_margin": 25.0,
        "texto_ia_final": "Análisis macroeconómico favorable para el sector tecnológico.",
        "wacc": 9.5,
        "g_term": 0.025,
        "calcular_dcf_fn": lambda w, g: 220.50,
        "inc": pd.DataFrame({"2023": [1000, 200]}, index=["Total Revenue", "Net Income"]),
        "cf": pd.DataFrame({"2023": [220]}, index=["Free Cash Flow"]),
        "perfil_texto": "🎯 1. Perfil General: Empresa líder en tecnología de consumo."
    }

    pdf_bytes = generar_pdf_reporte(mock_data)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 0
    assert pdf_bytes.startswith(b"%PDF")

class TestPDFGeneratorUnittest(unittest.TestCase):
    def test_pdf_bytes(self):
        test_generar_pdf_reporte_bytes()

if __name__ == "__main__":
    unittest.main()
