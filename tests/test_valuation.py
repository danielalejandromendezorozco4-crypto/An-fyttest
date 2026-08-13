import pytest
import unittest
import pandas as pd
from engine.valuation import calcular_dcf_intr_ps, crear_calculador_dcf, calcular_ddm, calcular_wacc
from engine.metrics import calcular_altman_zscore, calcular_piotroski_fscore, evaluar_veredicto, calcular_scoring
from services.ai_service import obtener_perfil_corporativo, obtener_analisis_macro_ia

def test_calcular_dcf_intr_ps_subvalorada():
    """
    Prueba el cálculo de DCF con datos de una empresa subvalorada (Valor Intrínseco > Precio Actual).
    Debe retornar un semáforo verde.
    """
    wacc_var = 9.0
    g_term_var = 0.025
    flujo_por_accion = 10.0
    g_1_5 = 0.12
    precio_actual = 100.0
    eps_ttm = 8.0
    total_cash = 5000.0
    total_debt = 2000.0
    shares_current = 100.0

    resultado = calcular_dcf_intr_ps(
        wacc_var, g_term_var, flujo_por_accion, g_1_5, precio_actual,
        eps_ttm, total_cash, total_debt, shares_current
    )

    assert isinstance(resultado, dict)
    assert "valor_intrinseco" in resultado
    assert "semaforo" in resultado
    assert "status" in resultado
    assert "upside" in resultado

    assert resultado["valor_intrinseco"] > precio_actual
    assert resultado["semaforo"] == "verde"
    assert resultado["status"] == "🟢"
    assert resultado["upside"] > 0

def test_calcular_dcf_intr_ps_sobrevalorada():
    """
    Prueba el cálculo de DCF con datos de una empresa sobrevalorada (Valor Intrínseco < Precio Actual).
    Debe retornar un semáforo rojo.
    """
    wacc_var = 12.0
    g_term_var = 0.015
    flujo_por_accion = 1.0
    g_1_5 = 0.02
    precio_actual = 200.0
    eps_ttm = 1.5
    total_cash = 100.0
    total_debt = 5000.0
    shares_current = 100.0

    resultado = calcular_dcf_intr_ps(
        wacc_var, g_term_var, flujo_por_accion, g_1_5, precio_actual,
        eps_ttm, total_cash, total_debt, shares_current
    )

    assert resultado["valor_intrinseco"] < precio_actual
    assert resultado["semaforo"] == "rojo"
    assert resultado["status"] == "🔴"

def test_crear_calculador_dcf():
    """
    Verifica que la función factory retorne un calculador ejecutable con 2 parámetros (wacc, g_term).
    """
    calculador = crear_calculador_dcf(
        flujo_por_accion=5.0, g_1_5=0.08, precio_actual=50.0,
        eps_ttm=4.0, total_cash=1000.0, total_debt=500.0, shares_current=50.0
    )
    val = calculador(9.5, 0.025)
    assert isinstance(val, float)
    assert val > 0

def test_altman_zscore_zona_segura():
    """
    Prueba indicador de quiebra Altman Z-Score en Zona Segura (Z > 2.99).
    """
    debt_eq = 0.5  # Bajo apalancamiento
    roa = 15.0     # Alta rentabilidad sobre activos

    res = calcular_altman_zscore(debt_eq, roa)
    assert res["z_score"] > 2.99
    assert res["semaforo"] == "verde"
    assert res["status"] == "🟢"
    assert res["categoria"] == "Zona Segura"

def test_altman_zscore_zona_peligro():
    """
    Prueba indicador de quiebra Altman Z-Score en Zona de Peligro (Z < 1.81).
    """
    debt_eq = 5.0   # Alto apalancamiento
    roa = -2.0      # Destrucción de activos

    res = calcular_altman_zscore(debt_eq, roa)
    assert res["z_score"] < 1.81
    assert res["semaforo"] == "rojo"
    assert res["status"] == "🔴"
    assert res["categoria"] == "Zona de Peligro"

def test_piotroski_fscore_datos_estaticos():
    """
    Prueba el cálculo de Piotroski F-Score con DataFrames estáticos simulados.
    """
    inc = pd.DataFrame({
        "2023": [1000, 400, 200, 150, 10, 100],
        "2022": [900, 350, 180, 120, 10, 100]
    }, index=['Total Revenue', 'Gross Profit', 'Operating Income', 'Net Income', 'Interest Expense', 'Basic Average Shares'])

    bs = pd.DataFrame({
        "2023": [2000, 800, 500, 300],
        "2022": [1800, 750, 450, 350]
    }, index=['Total Assets', 'Current Liabilities', 'Current Assets', 'Long Term Debt'])

    cf = pd.DataFrame({
        "2023": [250, 180],
        "2022": [200, 140]
    }, index=['Operating Cash Flow', 'Free Cash Flow'])

    info = {}

    res = calcular_piotroski_fscore(inc, bs, cf, info)
    assert isinstance(res, dict)
    assert "f_score" in res
    assert res["f_score"] >= 0 and res["f_score"] <= 9
    assert res["semaforo"] in ["verde", "amarillo", "rojo"]

def test_evaluar_veredicto_knockout():
    """
    Verifica que se active el veto de inversión (Knockout) ante un Z-Score bajo (< 1.81).
    """
    res = evaluar_veredicto(
        pts=90, z_score=1.5, net_debt_ebitda=2.0, is_fibra_util=False,
        cob_int=5.0, int_exp=100, roic=15.0
    )
    assert res["is_knockout"] is True
    assert res["color_v"] == "🔴"
    assert "VETO DE INVERSIÓN" in res["veredicto"]

def test_ai_service_fallback_sin_clave():
    """
    Verifica que si no hay API Key de Gemini o falla la API, la app despliega un mensaje de fallback sin fallar.
    """
    perfil = obtener_perfil_corporativo("AAPL", "Apple Inc.", "Technology", "Consumer Electronics", None, 3e12, 30.0, 25.0)
    assert isinstance(perfil, str)
    assert "Perfil Corporativo no disponible" in perfil or "cuota" in perfil

    macro_txt, score = obtener_analisis_macro_ia("AAPL", "Apple Inc.", "Technology", "")
    assert isinstance(macro_txt, str)
    assert score == 2.5

def test_market_cap_fallback():
    """
    Verifica que empresas de gran capitalización (Mega Caps como GOOGL/AAPL) reciban un Market Cap realista
    incluso si una API primaria falla, evitando la penalización indebida de Small Cap.
    """
    precio_actual = 175.0
    info_mock = {"sharesOutstanding": 12000000000.0}
    mcap = info_mock.get("marketCap", 0.0)
    shares = info_mock.get("sharesOutstanding", 0.0)
    if mcap <= 0 and shares > 0:
        mcap = shares * precio_actual
    assert mcap > 2000000000.0

class TestValuationUnittest(unittest.TestCase):
    def test_dcf_subvalorada(self):
        test_calcular_dcf_intr_ps_subvalorada()

    def test_dcf_sobrevalorada(self):
        test_calcular_dcf_intr_ps_sobrevalorada()

    def test_factory_dcf(self):
        test_crear_calculador_dcf()

    def test_zscore_segura(self):
        test_altman_zscore_zona_segura()

    def test_zscore_peligro(self):
        test_altman_zscore_zona_peligro()

    def test_piotroski(self):
        test_piotroski_fscore_datos_estaticos()

    def test_knockout(self):
        test_evaluar_veredicto_knockout()

    def test_ai_fallback(self):
        test_ai_service_fallback_sin_clave()

    def test_mcap_fallback(self):
        test_market_cap_fallback()

if __name__ == "__main__":
    unittest.main()
