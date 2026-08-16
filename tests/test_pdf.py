import pytest
import unittest
import pandas as pd
from reports.pdf_generator import generar_pdf_reporte
from config.settings import sanitizar_para_pdf, limpiar_texto


def test_sanitizar_para_pdf_polimorfico():
    """
    Verifica que sanitizar_para_pdf y limpiar_texto manejen de forma segura
    objetos de tipo str, bytes, bytearray, None y números sin error.
    """
    assert sanitizar_para_pdf("Hola **Mundo**") == "Hola Mundo"
    assert sanitizar_para_pdf(b"Texto en bytes") == "Texto en bytes"
    assert sanitizar_para_pdf(bytearray(b"Texto en bytearray")) == "Texto en bytearray"
    assert sanitizar_para_pdf(None) == ""
    assert sanitizar_para_pdf(12345) == "12345"

    assert limpiar_texto("Canción") == "Cancion"
    assert limpiar_texto(b"Cancion") == "Cancion"
    assert limpiar_texto(bytearray(b"Cancion")) == "Cancion"
    assert limpiar_texto(None) == ""


def test_generar_pdf_reporte_bytes():
    """
    Verifica que generar_pdf_reporte construya exitosamente un PDF de 3 páginas
    y retorne un objeto de tipo bytes válido que empiece con '%PDF-'.
    """
    inc_mock = pd.DataFrame({
        "2023": [25000e6, 14000e6, 11500e6],
        "2022": [22000e6, 12000e6, 9900e6],
    }, index=["Total Revenue", "Operating Income", "Net Income"])

    cf_mock = pd.DataFrame({
        "2023": [12500e6, 11800e6],
        "2022": [10500e6, 9800e6],
    }, index=["Operating Cash Flow", "Free Cash Flow"])

    analysis_data_mock = {
        "ticker": "MA",
        "nombre": "Mastercard Incorporated",
        "sector": "Financial Services",
        "industria": "Credit Services",
        "precio_actual": 485.50,
        "mcap": 450e9,
        "shares_current": 930e6,
        "pts": 82.5,
        "veredicto": "COMPRA FUERTE",
        "veredicto_txt": "Excelente perfil financiero institucional.",
        "v_intr_dcf": 520.00,
        "precio_max_compra": 468.00,
        "desc_req": 0.10,
        "clase_msg": "Clase A (Alta Calidad)",
        "val_nde": "0.6x",
        "val_cob": "22.5x",
        "val_de": "0.45",
        "roic": 52.0,
        "fcf_yield": 2.8,
        "tasa_libre_riesgo": 4.25,
        "val_roe": "145.0%",
        "roa": 28.5,
        "mg_op": 56.0,
        "fcf_conv": 1.02,
        "val_div_metric": "$2.64 (0.54%)",
        "val_ddm_str": "N/A",
        "pe": 36.5,
        "p_fcf": 38.0,
        "peg": 2.1,
        "ev_ebitda": 26.0,
        "p_s": 18.0,
        "p_b": 60.0,
        "gross_margin": 100.0,
        "net_margin": 45.0,
        "texto_ia_final": "Tesis macroeconómica sólida respaldada por el crecimiento secular de pagos digitales.",
        "wacc": 8.5,
        "g_term": 0.025,
        "calcular_dcf_fn": lambda w, g: 520.0 * (8.5 / w) * (1 + g - 0.025),
        "inc": inc_mock,
        "cf": cf_mock,
        "perfil_texto": "**🎯 1. Modelo de Negocio:** Mastercard opera una red global de pagos...",
        "fcff_wacc": 8.5,
        "fcff_ke": 9.2,
        "fcff_kd": 4.25,
        "fcff_rf": 4.25,
        "fcff_erp": 5.0,
        "fcff_we": 0.95,
        "fcff_wd": 0.05,
        "fcff_tax_rate": 0.18,
        "fcff_ev": 480e9,
        "fcff_equity": 472e9,
        "fcff_margen": 0.07,
        "fcff_historico": [11800e6, 9800e6],
        "fcff_g_term": 0.025,
    }

    pdf_bytes = generar_pdf_reporte(analysis_data_mock)

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF-")


class TestPdfUnittest(unittest.TestCase):
    def test_sanitizar(self):
        test_sanitizar_para_pdf_polimorfico()

    def test_generar_pdf(self):
        test_generar_pdf_reporte_bytes()


if __name__ == "__main__":
    unittest.main()
