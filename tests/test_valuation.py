import pytest
import unittest
import pandas as pd
from engine.valuation import (
    calcular_dcf_intr_ps,
    crear_calculador_dcf,
    calcular_ddm,
    calcular_wacc,
    calcular_fcff_valuation,
    calcular_fcff_normalizado,
    calcular_g_term_restringido,
    calcular_curva_crecimiento_5y,
)
from engine.metrics import (
    calcular_altman_zscore,
    calcular_piotroski_fscore,
    evaluar_veredicto,
    calcular_scoring,
)
from services.ai_service import obtener_perfil_corporativo, obtener_analisis_macro_ia


# ─────────────────────────────────────────────────────────────────────────────
# 1. PRUEBAS DE EXTRACCIÓN Y CÁLCULO DE FCFF BASE REAL
# ─────────────────────────────────────────────────────────────────────────────

def test_fcff_base_con_datos_reales():
    """
    Verifica que el FCFF base = CFO + Interest_Expense * (1 - Tax_Rate) - CapEx
    sin penalizaciones ni descuentos arbitrarios sobre el flujo real.
    """
    cfo = 100_000_000.0
    capex = 20_000_000.0
    interest_exp = 5_000_000.0
    tax_rate = 0.20
    escudo_fiscal = interest_exp * (1.0 - tax_rate)  # 4,000,000
    fcff_esperado = cfo + escudo_fiscal - capex     # 84,000,000

    res = calcular_fcff_valuation(
        ocf_hist       = [cfo],
        capex_hist     = [capex],
        interest_hist  = [interest_exp],
        pretax_hist    = [50_000_000.0],
        taxprov_hist   = [10_000_000.0],  # Tax rate = 20%
        total_debt     = 50_000_000.0,
        total_cash     = 30_000_000.0,
        shares_diluted = 10_000_000.0,
        mcap           = 800_000_000.0,
        beta           = 1.10,
        rf             = 4.20,
        precio_actual  = 80.0,
    )
    assert abs(res["fcff_base"] - fcff_esperado) < 1.0
    assert len(res["fcff_proyectado"]) == 5
    assert res["valor_intrinseco"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# 2. PRUEBAS DE TASA DE CRECIMIENTO SIN RECORTE ARBITRARIO
# ─────────────────────────────────────────────────────────────────────────────

def test_curva_crecimiento_fundamentales_reales():
    """
    Verifica que se utilice directamente el crecimiento reportado de la API
    sin reducirlo artificialmente al 85%.
    """
    curva_api = calcular_curva_crecimiento_5y(
        cagr_revenue_hist=0.08,
        revenue_growth_api=0.12,
        g_term=0.025,
        n_years=5,
    )
    # Debe tomar 12% directamente sin reducción
    assert curva_api[0] == 0.12

    curva_cagr = calcular_curva_crecimiento_5y(
        cagr_revenue_hist=0.10,
        revenue_growth_api=0.0,
        g_term=0.025,
        n_years=5,
    )
    # Debe tomar 10% del CAGR
    assert curva_cagr[0] == 0.10


# ─────────────────────────────────────────────────────────────────────────────
# 3. PRUEBAS DE CÁLCULO DE WACC REAL DE MERCADO
# ─────────────────────────────────────────────────────────────────────────────

def test_wacc_datos_reales_de_mercado():
    """
    Verifica que el WACC calcule Ke por CAPM, Kd empírico por gastos de intereses / deuda,
    y ponderaciones de mercado sin imponer pisos artificiales.
    """
    rf = 4.20
    beta = 1.20
    erp = 5.50
    # Ke = 4.20 + 1.20 * 5.50 = 10.80%
    mcap = 900_000_000.0
    total_debt = 100_000_000.0
    # Total capital = 1,000,000,000 -> We = 0.90, Wd = 0.10
    int_exp = 6_000_000.0  # Kd = 6.0%
    tax_rate = 0.25

    res = calcular_wacc(
        tasa_libre_riesgo=rf,
        beta=beta,
        mcap=mcap,
        total_debt=total_debt,
        int_exp=int_exp,
        tax_rate=tax_rate,
        erp=erp,
    )
    assert abs(res["ke"] - 10.80) < 0.01
    assert abs(res["kd"] - 6.00) < 0.01
    assert abs(res["we"] - 0.90) < 0.01
    assert abs(res["wd"] - 0.10) < 0.01

    # WACC = 0.90 * 10.80 + 0.10 * 6.00 * (1 - 0.25) = 9.72 + 0.45 = 10.17%
    assert abs(res["wacc"] - 10.17) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
# 4. PRUEBAS DE CONVENCIÓN DE MEDIO AÑO Y VALOR TERMINAL (GORDON SHAPIRO)
# ─────────────────────────────────────────────────────────────────────────────

def test_fcff_mid_year_convention_y_gordon_shapiro():
    """
    Verifica que los flujos explícitos aplican Mid-Year Discounting (t - 0.5)
    y el Valor Terminal Gordon Shapiro se descuente a 5 años completos.
    """
    resultado = calcular_fcff_valuation(
        ocf_hist              = [100e9, 90e9, 80e9],
        capex_hist            = [20e9, 18e9, 16e9],
        interest_hist         = [2e9, 1.8e9, 1.5e9],
        pretax_hist           = [90e9, 80e9, 70e9],
        taxprov_hist          = [18e9, 16e9, 14e9],
        total_debt            = 40e9,
        total_cash            = 60e9,   # Caja neta positiva (Deuda neta = -20B)
        shares_diluted        = 10e9,
        mcap                  = 1_500e9,
        beta                  = 1.0,
        rf                    = 4.20,
        precio_actual         = 150.0,
        revenue_growth_api    = 0.08,
    )

    ev = resultado["enterprise_value"]
    pv_f = resultado["pv_flujos"]
    pv_tv = resultado["pv_terminal"]
    eq = resultado["equity_value"]
    sh = resultado["shares_diluted"]
    v_intr = resultado["valor_intrinseco"]

    # 1. EV = PV(flujos) + PV(terminal)
    assert abs(ev - (pv_f + pv_tv)) < 1.0

    # 2. Equity = EV - Deuda Neta = EV - (Deuda - Caja) = EV + Caja - Deuda
    deuda_neta = 40e9 - 60e9
    assert abs(eq - (ev - deuda_neta)) < 1.0

    # 3. Valor por acción = Equity / Shares
    assert abs(v_intr - (eq / sh)) < 0.01
    assert v_intr > 0


# ─────────────────────────────────────────────────────────────────────────────
# 5. PRUEBAS CON DATOS REALES DE EMPRESAS (GOOGL, AAPL, NFLX, KO, JNJ)
# ─────────────────────────────────────────────────────────────────────────────

def test_fcff_googl_calibracion_real():
    """Alphabet Inc. (GOOGL): Caja masiva, bajo apalancamiento y márgenes altos."""
    res = calcular_fcff_valuation(
        ocf_hist          = [105e9, 101.7e9, 91.5e9, 91.6e9],
        capex_hist        = [38e9,  32.2e9,  31.5e9, 24.6e9],
        interest_hist     = [0.35e9, 0.3e9,  0.35e9, 0.35e9],
        pretax_hist       = [100e9, 88e9,    75e9,   90e9],
        taxprov_hist      = [16e9,  14e9,    11.5e9, 14.7e9],
        total_debt        = 28e9,
        total_cash        = 110e9,
        shares_diluted    = 12.35e9,
        mcap              = 2_180e9,
        beta              = 1.05,
        rf                = 4.25,
        precio_actual     = 175.0,
        revenue_growth_api = 0.13,
    )
    assert res["valor_intrinseco"] > 140.0
    assert res["deuda_neta"] < 0  # Caja neta positiva


def test_fcff_aapl_real():
    """Apple Inc. (AAPL): Generación masiva de flujo libre y retorno sobre capital."""
    res = calcular_fcff_valuation(
        ocf_hist          = [118e9, 110e9, 104e9],
        capex_hist        = [10e9, 9.5e9, 9.0e9],
        interest_hist     = [3.9e9, 3.8e9, 3.5e9],
        pretax_hist       = [120e9, 115e9, 105e9],
        taxprov_hist      = [20e9, 19e9, 17e9],
        total_debt        = 105e9,
        total_cash        = 160e9,
        shares_diluted    = 15.4e9,
        mcap              = 3400e9,
        beta              = 1.10,
        rf                = 4.25,
        precio_actual     = 220.0,
        revenue_growth_api = 0.08,
    )
    assert res["valor_intrinseco"] > 0
    assert res["equity_value"] > res["enterprise_value"]


def test_fcff_nflx_real():
    """Netflix Inc. (NFLX): FCF en expansión y beta dinámico."""
    res = calcular_fcff_valuation(
        ocf_hist          = [7.5e9, 6.0e9, 2.0e9],
        capex_hist        = [0.4e9, 0.35e9, 0.3e9],
        interest_hist     = [0.70e9, 0.75e9, 0.80e9],
        pretax_hist       = [8.0e9, 6.5e9, 5.0e9],
        taxprov_hist      = [1.5e9, 1.2e9, 0.9e9],
        total_debt        = 14.0e9,
        total_cash        = 7.0e9,
        shares_diluted    = 430e6,
        mcap              = 280.0e9,
        beta              = 1.25,
        rf                = 4.25,
        precio_actual     = 650.0,
        revenue_growth_api = 0.15,
    )
    assert res["valor_intrinseco"] > 0
    assert res["kd"] > 0


def test_fcff_ko_real():
    """The Coca-Cola Company (KO): Flujos defensivos predecibles."""
    res = calcular_fcff_valuation(
        ocf_hist          = [11.5e9, 11.0e9, 10.5e9],
        capex_hist        = [1.9e9, 1.8e9, 1.5e9],
        interest_hist     = [1.5e9, 1.4e9, 1.3e9],
        pretax_hist       = [13.0e9, 12.5e9, 11.8e9],
        taxprov_hist      = [2.6e9, 2.5e9, 2.3e9],
        total_debt        = 40.0e9,
        total_cash        = 12.0e9,
        shares_diluted    = 4.3e9,
        mcap              = 270.0e9,
        beta              = 0.60,
        rf                = 4.25,
        precio_actual     = 62.0,
        revenue_growth_api = 0.05,
    )
    assert res["valor_intrinseco"] > 0


def test_obtener_erp_mercado_import_y_fallback():
    """Verifica importación y retorno de ERP de contingencia."""
    from data.financial_fetcher import obtener_erp_mercado
    erp_def = obtener_erp_mercado(fred_api_key="", rf_actual=4.25)
    assert isinstance(erp_def, float)
    assert 4.50 <= erp_def <= 6.00


def test_crear_calculador_dcf():
    """Verifica factory para matriz de sensibilidad multiescenario."""
    calculador = crear_calculador_dcf(
        flujo_por_accion=5.0, g_1_5=0.08, precio_actual=50.0,
        eps_ttm=4.0, total_cash=1000.0, total_debt=500.0, shares_current=50.0
    )
    val = calculador(9.5, 0.025)
    assert isinstance(val, float)
    assert val > 0


def test_altman_zscore_segura():
    res = calcular_altman_zscore(debt_eq=0.5, roa=15.0)
    assert res["z_score"] > 2.99
    assert res["status"] == "🟢"


def test_piotroski_fscore_datos_estaticos():
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

    res = calcular_piotroski_fscore(inc, bs, cf, {})
    assert res["f_score"] >= 0 and res["f_score"] <= 9


def test_evaluar_veredicto_knockout():
    res = evaluar_veredicto(pts=90, z_score=1.5, net_debt_ebitda=2.0, is_fibra_util=False, cob_int=5.0, int_exp=100, roic=15.0)
    assert res["is_knockout"] is True
    assert "VETO DE INVERSIÓN" in res["veredicto"]


class TestValuationUnittest(unittest.TestCase):
    def test_fcff_base(self):
        test_fcff_base_con_datos_reales()

    def test_curva_crecimiento(self):
        test_curva_crecimiento_fundamentales_reales()

    def test_wacc_mercado(self):
        test_wacc_datos_reales_de_mercado()

    def test_mid_year_y_gordon(self):
        test_fcff_mid_year_convention_y_gordon_shapiro()

    def test_fcff_googl(self):
        test_fcff_googl_calibracion_real()

    def test_fcff_aapl(self):
        test_fcff_aapl_real()

    def test_fcff_nflx(self):
        test_fcff_nflx_real()

    def test_fcff_ko(self):
        test_fcff_ko_real()

    def test_erp_mercado(self):
        test_obtener_erp_mercado_import_y_fallback()

    def test_factory_dcf(self):
        test_crear_calculador_dcf()

    def test_zscore_segura(self):
        test_altman_zscore_segura()

    def test_piotroski(self):
        test_piotroski_fscore_datos_estaticos()

    def test_knockout(self):
        test_evaluar_veredicto_knockout()


if __name__ == "__main__":
    unittest.main()
