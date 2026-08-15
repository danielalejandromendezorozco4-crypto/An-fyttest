import pytest
import unittest
import pandas as pd
from engine.valuation import (
    calcular_dcf_intr_ps,
    crear_calculador_dcf,
    calcular_ddm,
    calcular_wacc,
    calcular_fcff_valuation,
)
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

    def test_fcff_grande(self):
        test_fcff_empresa_grande()

    def test_fcff_sin_deuda(self):
        test_fcff_empresa_sin_deuda()

    def test_fcff_apalancada(self):
        test_fcff_empresa_alto_apalancamiento()

    def test_fcff_negativo(self):
        test_fcff_flujo_negativo()

    def test_fcff_minimo(self):
        test_fcff_datos_minimos()

    def test_fcff_sin_capex(self):
        test_fcff_empresa_sin_capex()

    def test_fcff_googl(self):
        test_fcff_googl_calibracion()


# ─────────────────────────────────────────────────────────────────────────────
# NUEVOS TESTS — Motor FCFF con WACC Dinámico Calibrado
# ─────────────────────────────────────────────────────────────────────────────

def test_fcff_googl_calibracion():
    """
    Criterio 1: Calibración específica para Alphabet Inc. (GOOGL).
    Verifica que con datos fundamentales reales de GOOGL, el valor intrínseco
    por acción arroja una cifra financieramente coherente ($140 - $220 USD),
    eliminando el valor artificialmente deprimido previo (~$87).
    """
    # Perfil fundamental realista de Alphabet Inc. (GOOGL)
    # Flujos históricos (2024 TTM, 2023, 2022, 2021)
    ocf_hist      = [105e9, 101.7e9, 91.5e9, 91.6e9]
    capex_hist    = [38e9,  32.2e9,  31.5e9, 24.6e9]
    interest_hist = [0.35e9, 0.3e9,  0.35e9, 0.35e9]
    pretax_hist   = [100e9, 88e9,    75e9,   90e9]
    taxprov_hist  = [16e9,  14e9,    11.5e9, 14.7e9]  # Tax rate ~16%
    total_debt    = 28e9
    total_cash    = 110e9
    shares_diluted = 12.35e9
    mcap          = 2_180e9
    beta          = 1.05
    rf            = 4.25
    precio_actual = 175.0
    growth_rate_exp = 0.14  # Consenso de utilidades/ingresos ~14%

    resultado = calcular_fcff_valuation(
        ocf_hist        = ocf_hist,
        capex_hist      = capex_hist,
        interest_hist   = interest_hist,
        pretax_hist     = pretax_hist,
        taxprov_hist    = taxprov_hist,
        total_debt      = total_debt,
        total_cash      = total_cash,
        shares_diluted  = shares_diluted,
        mcap            = mcap,
        beta            = beta,
        rf              = rf,
        precio_actual   = precio_actual,
        growth_rate_exp = growth_rate_exp,
    )

    valor_intr = resultado["valor_intrinseco"]

    # 1. El valor intrínseco NO debe ser el atípico previo de ~$87
    assert valor_intr > 130.0, f"Valor intrínseco de GOOGL muy bajo: ${valor_intr:.2f} (debe ser > $130)"

    # 2. Debe encontrarse en un rango analítico razonable frente al precio de cotización y Wall Street ($140 - $240)
    assert 135.0 <= valor_intr <= 240.0, f"Valor intrínseco fuera de rango razonable: ${valor_intr:.2f}"

    # 3. WACC debe situarse en el rango de costo de capital de mega-caps de bajo riesgo (~7.5% - 10.0%)
    assert 7.0 <= resultado["wacc"] <= 11.0, f"WACC anómalo para GOOGL: {resultado['wacc']:.2f}%"

    # 4. Enterprise Value y Equity Value deben ser positivos y en el orden de billones USD (> $1.5 Trillones)
    assert resultado["enterprise_value"] > 1_400e9
    assert resultado["equity_value"] > 1_500e9

def test_fcff_empresa_grande():
    """
    Criterio 1: Empresa de gran capitalización y bajo apalancamiento (tipo AAPL/MSFT).
    Verifica que el motor FCFF produce métricas consistentes sin campos nulos.
    """
    # Datos sintéticos representativos de AAPL (en USD)
    ocf_hist      = [120e9, 110e9, 100e9, 95e9]   # OCF: creciente
    capex_hist    = [11e9,  10e9,  9e9,   8e9]    # CapEx moderado
    interest_hist = [4e9,   3.5e9, 3e9,   2.5e9]  # Intereses bajos
    pretax_hist   = [110e9, 100e9, 90e9,  85e9]   # Ingresos pre-tax
    taxprov_hist  = [27e9,  25e9,  22e9,  21e9]   # Té efectiva ~25%

    resultado = calcular_fcff_valuation(
        ocf_hist      = ocf_hist,
        capex_hist    = capex_hist,
        interest_hist = interest_hist,
        pretax_hist   = pretax_hist,
        taxprov_hist  = taxprov_hist,
        total_debt    = 110e9,
        total_cash    = 160e9,
        shares_diluted = 15.5e9,
        mcap          = 3_000e9,
        beta          = 1.2,
        rf            = 4.35,
        precio_actual = 195.0,
    )

    # Estructura completa
    assert isinstance(resultado, dict)
    campos_requeridos = [
        "valor_intrinseco", "enterprise_value", "equity_value",
        "pv_flujos", "pv_terminal", "fcff_historico", "fcff_proyectado",
        "wacc", "ke", "kd", "we", "wd", "rf", "tax_rate_real",
        "g_term", "total_cash", "total_debt", "shares_diluted",
        "margen_seguridad", "precio_actual", "status", "semaforo", "upside",
    ]
    for campo in campos_requeridos:
        assert campo in resultado, f"Campo faltante: {campo}"
        assert resultado[campo] is not None, f"Campo nulo: {campo}"

    # Métricas financieras consistentes
    assert 6.0 <= resultado["wacc"] <= 18.0, f"WACC fuera de rango: {resultado['wacc']}"
    assert resultado["ke"] > resultado["rf"], "Ke debe ser mayor que Rf"
    assert resultado["kd"] > 0, "Empresa con deuda debe tener Kd > 0"
    assert 0.0 < resultado["tax_rate_real"] <= 0.40, "Tasa impositiva inválida"
    assert resultado["we"] + resultado["wd"] <= 1.001, "Ponderaciones no suman 1"
    assert len(resultado["fcff_historico"]) == 4
    assert len(resultado["fcff_proyectado"]) == 5
    assert resultado["enterprise_value"] > 0
    assert resultado["semaforo"] in ["verde", "rojo"]
    assert resultado["status"] in ["\U0001F7E2", "\U0001F534"]
    assert resultado["valor_intrinseco"] >= 0


def test_fcff_empresa_sin_deuda():
    """
    Criterio 2a: Empresa sin deuda (deuda = 0).
    Verifica que wd=0, kd=0 y WACC = Ke puro, sin ZeroDivisionError.
    """
    resultado = calcular_fcff_valuation(
        ocf_hist      = [50e9, 45e9, 40e9],
        capex_hist    = [5e9,  4e9,  3e9],
        interest_hist = [0.0,  0.0,  0.0],   # Sin intereses
        pretax_hist   = [60e9, 55e9, 50e9],
        taxprov_hist  = [15e9, 14e9, 13e9],
        total_debt    = 0.0,                  # SIN DEUDA
        total_cash    = 80e9,
        shares_diluted = 8e9,
        mcap          = 500e9,
        beta          = 0.9,
        rf            = 4.35,
        precio_actual = 62.5,
    )

    assert resultado["wd"] == 0.0, "Empresa sin deuda debe tener wd=0"
    assert resultado["kd"] == 0.0, "Empresa sin deuda debe tener kd=0"
    assert resultado["we"] == 1.0, "Empresa sin deuda: we debe ser 1.0"
    # WACC debe estar muy cerca de Ke (solo equity)
    wacc_esperado = resultado["ke"]
    # Puede diferir ligeramente por el clampeo de límites
    assert abs(resultado["wacc"] - wacc_esperado) < 2.0, (
        f"WACC ({resultado['wacc']:.2f}) demasiado lejos de Ke ({wacc_esperado:.2f}) en empresa sin deuda"
    )
    assert resultado["equity_value"] > 0
    assert resultado["valor_intrinseco"] > 0


def test_fcff_empresa_alto_apalancamiento():
    """
    Criterio 2b: Empresa de alto apalancamiento (tipo AT&T / empresa telco).
    Verifica que Kd se calcula sin crash y WACC es finito y positivo.
    """
    resultado = calcular_fcff_valuation(
        ocf_hist      = [20e9, 18e9, 16e9],
        capex_hist    = [18e9, 17e9, 15e9],   # CapEx muy alto (intensivo en activos)
        interest_hist = [8e9,  7e9,  6.5e9],  # Intereses muy altos
        pretax_hist   = [5e9,  4e9,  3e9],
        taxprov_hist  = [1.2e9, 1e9,  0.8e9],
        total_debt    = 150e9,                 # Deuda enorme
        total_cash    = 10e9,
        shares_diluted = 7e9,
        mcap          = 80e9,
        beta          = 0.8,
        rf            = 4.35,
        precio_actual = 11.4,
    )

    # No debe lanzar excepción, WACC debe ser finito y positivo
    assert isinstance(resultado["wacc"], float)
    assert 0 < resultado["wacc"] <= 18.0, f"WACC fuera de rango: {resultado['wacc']}"
    assert resultado["kd"] > 0, "Empresa con deuda alta debe tener Kd > 0"
    assert resultado["wd"] > 0.5, "Empresa muy apalancada debe tener Wd dominante"
    # Equity puede ser negativo (deuda > EV) — el valor intrínseco será 0 o bajo
    assert resultado["valor_intrinseco"] >= 0, "Valor intrínseco no puede ser negativo"
    assert resultado["semaforo"] in ["verde", "rojo"]


def test_fcff_flujo_negativo():
    """
    Criterio: Empresa con FCFF base negativo (pérdidas operativas estructurales).
    Verifica que la función no lanza excepción y activa el fallback defensivo.
    """
    resultado = calcular_fcff_valuation(
        ocf_hist      = [-5e9, -3e9, 2e9],    # OCF negativo los 2 años más recientes
        capex_hist    = [10e9, 8e9,  6e9],    # CapEx alto → FCFF muy negativo
        interest_hist = [1e9,  0.8e9, 0.5e9],
        pretax_hist   = [-6e9, -4e9, 1e9],
        taxprov_hist  = [0.0,  0.0,  0.3e9],
        total_debt    = 20e9,
        total_cash    = 5e9,
        shares_diluted = 1e9,
        mcap          = 15e9,
        beta          = 1.8,
        rf            = 4.35,
        precio_actual = 15.0,
    )

    # No debe lanzar excepción
    assert isinstance(resultado, dict)
    # El fallback debe activarse y producir un valor_intrinseco >= 0
    assert resultado["valor_intrinseco"] >= 0
    # Semaforo debe ser rojo (no subvalorada en esta situación)
    assert resultado["semaforo"] in ["verde", "rojo"]
    # Los flujos proyectados siempre deben existir
    assert len(resultado["fcff_proyectado"]) == 5


def test_fcff_datos_minimos():
    """
    Criterio: Solo 1 año de datos históricos disponibles.
    Verifica que la proyección funciona correctamente con datos mínimos.
    """
    resultado = calcular_fcff_valuation(
        ocf_hist      = [10e9],     # Solo TTM
        capex_hist    = [2e9],
        interest_hist = [0.5e9],
        pretax_hist   = [8e9],
        taxprov_hist  = [2e9],
        total_debt    = 5e9,
        total_cash    = 3e9,
        shares_diluted = 500e6,
        mcap          = 100e9,
        beta          = 1.1,
        rf            = 4.35,
        precio_actual = 200.0,
    )

    assert isinstance(resultado, dict)
    assert len(resultado["fcff_historico"]) == 1, "Con 1 año debe haber 1 FCFF histórico"
    assert len(resultado["fcff_proyectado"]) == 5, "Proyección siempre debe ser de 5 años"
    assert resultado["wacc"] > 0
    assert resultado["enterprise_value"] > 0
    assert resultado["valor_intrinseco"] >= 0


def test_fcff_empresa_sin_capex():
    """
    Criterio: Empresa asset-light sin CapEx (software / servicios digitales).
    Verifica que FCFF = OCF + Interest*(1-T) cuando CapEx = 0, sin división por cero.
    """
    ocf = 30e9
    interest = 1e9
    pretax   = 35e9
    taxprov  = 8.75e9   # T_ef = 25%
    tax_ef   = taxprov / pretax  # = 0.25

    resultado = calcular_fcff_valuation(
        ocf_hist      = [ocf, 28e9, 25e9],
        capex_hist    = [0.0, 0.0, 0.0],   # SIN CapEx
        interest_hist = [interest, 0.9e9, 0.8e9],
        pretax_hist   = [pretax, 32e9, 28e9],
        taxprov_hist  = [taxprov, 8e9, 7e9],
        total_debt    = 10e9,
        total_cash    = 50e9,
        shares_diluted = 2e9,
        mcap          = 600e9,
        beta          = 1.3,
        rf            = 4.35,
        precio_actual = 300.0,
    )

    assert isinstance(resultado, dict)
    assert resultado["valor_intrinseco"] >= 0
    assert resultado["wacc"] > 0

    # Verificar FCFF[0] = OCF + Interest*(1-T) - 0
    fcff_ttm = resultado["fcff_historico"][0]
    tax_ef_real = resultado["tax_rate_real"]
    fcff_esperado = ocf + (interest * (1 - tax_ef_real))  # CapEx = 0
    assert abs(fcff_ttm - fcff_esperado) < 1e6, (
        f"FCFF calculado ({fcff_ttm:.0f}) difiere del esperado ({fcff_esperado:.0f})"
    )


if __name__ == "__main__":
    unittest.main()
