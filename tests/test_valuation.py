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


# ─────────────────────────────────────────────────────────────────────────────
# 1. PRUEBAS DE EXTRACCIÓN Y CÁLCULO DE FCFF BASE REAL
# ─────────────────────────────────────────────────────────────────────────────

def test_fcff_base_con_datos_reales():
    """
    Verifica que el FCFF base = CFO + Interest_Expense * (1 - Tax_Rate) - CapEx
    sin distorsiones sobre el flujo real.
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
    assert res["enterprise_value"] > 0
    assert res["equity_value"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# 2. PRUEBAS DE CÁLCULO DE WACC REAL DE MERCADO
# ─────────────────────────────────────────────────────────────────────────────

def test_wacc_datos_reales_de_mercado():
    """
    Verifica que el WACC calcule Ke por CAPM, Kd empírico por gastos de intereses / deuda,
    y ponderaciones de mercado sin imponer pisos artificiales.
    """
    rf = 4.20
    beta = 1.20
    erp = 5.50
    mcap = 900_000_000.0
    total_debt = 100_000_000.0
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
    assert abs(res["wacc"] - 10.17) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
# 3. PRUEBAS DE VALUACIÓN FCFF POR COMPAÑÍA (MA, AAPL, KO, NVDA)
# ─────────────────────────────────────────────────────────────────────────────

def test_fcff_valuation_mastercard_ma():
    """Valuación institucional FCFF para Mastercard (MA)."""
    res = calcular_fcff_valuation(
        ocf_hist=[12.5e9, 10.5e9, 9.5e9],
        capex_hist=[0.5e9, 0.4e9, 0.4e9],
        interest_hist=[0.5e9, 0.45e9, 0.4e9],
        pretax_hist=[14.0e9, 12.0e9, 10.5e9],
        taxprov_hist=[2.5e9, 2.1e9, 1.8e9],
        total_debt=16e9,
        total_cash=8.5e9,
        shares_diluted=930e6,
        mcap=450e9,
        beta=1.05,
        rf=4.25,
        precio_actual=485.0,
        revenue_growth_api=0.12,
        revenue_ttm=25e9,
    )
    assert res["valor_intrinseco"] > 350.0
    assert res["wacc"] > 0
    assert res["precio_max_compra"] > 0
    assert -1.0 <= res["margen_seguridad"] <= 2.0


def test_fcff_valuation_apple_aapl():
    """Valuación institucional FCFF para Apple Inc. (AAPL)."""
    res = calcular_fcff_valuation(
        ocf_hist=[115e9, 110e9, 122e9],
        capex_hist=[10e9, 11e9, 10.5e9],
        interest_hist=[3.5e9, 3.8e9, 2.9e9],
        pretax_hist=[120e9, 115e9, 119e9],
        taxprov_hist=[18e9, 17e9, 19e9],
        total_debt=105e9,
        total_cash=65e9,
        shares_diluted=15.3e9,
        mcap=3300e9,
        beta=1.10,
        rf=4.20,
        precio_actual=220.0,
        revenue_growth_api=0.08,
        revenue_ttm=390e9,
    )
    assert res["valor_intrinseco"] > 140.0
    assert res["enterprise_value"] > 2000e9


def test_fcff_valuation_coca_cola_ko():
    """Valuación institucional FCFF para The Coca-Cola Company (KO)."""
    res = calcular_fcff_valuation(
        ocf_hist=[11.5e9, 11.0e9, 10.5e9],
        capex_hist=[1.9e9, 1.8e9, 1.5e9],
        interest_hist=[1.5e9, 1.4e9, 1.3e9],
        pretax_hist=[13.0e9, 12.5e9, 11.8e9],
        taxprov_hist=[2.6e9, 2.5e9, 2.3e9],
        total_debt=40e9,
        total_cash=14e9,
        shares_diluted=4.3e9,
        mcap=280e9,
        beta=0.60,
        rf=4.20,
        precio_actual=65.0,
        revenue_growth_api=0.05,
        revenue_ttm=46e9,
    )
    assert res["valor_intrinseco"] > 45.0
    assert res["precio_max_compra"] > 0


def test_fcff_valuation_fcf_negativo_normalizado():
    """Prueba de robustez ante FCF histórico negativo con FCFF normalizado."""
    res = calcular_fcff_valuation(
        ocf_hist=[-200e6],
        capex_hist=[500e6],
        interest_hist=[100e6],
        pretax_hist=[500e6],
        taxprov_hist=[100e6],
        total_debt=2e9,
        total_cash=3e9,
        shares_diluted=500e6,
        mcap=25e9,
        beta=1.35,
        rf=4.20,
        precio_actual=50.0,
        revenue_ttm=12e9,
        operating_margin_hist=0.15,
    )
    assert res["valor_intrinseco"] > 0
    assert res["fcff_base"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# 4. PRUEBAS DE DDM, CURVAS DE CRECIMIENTO Y SENSIBILIDAD
# ─────────────────────────────────────────────────────────────────────────────

def test_ddm_gordon_growth():
    res = calcular_ddm(div_rate=3.0, ke=9.0, g_div=0.03, precio_actual=50.0)
    assert res["viable"] is True
    assert abs(res["valor_intrinseco_ddm"] - 51.50) < 0.10


def test_curva_crecimiento_fundamentales_reales():
    curva_api = calcular_curva_crecimiento_5y(
        cagr_revenue_hist=0.08,
        revenue_growth_api=0.12,
        g_term=0.025,
        n_years=5,
    )
    assert curva_api[0] == 0.12


def test_factory_calculador_dcf():
    calculador = crear_calculador_dcf(
        flujo_por_accion=5.0, g_1_5=0.08, precio_actual=50.0,
        eps_ttm=4.0, total_cash=1000.0, total_debt=500.0, shares_current=50.0
    )
    val = calculador(9.5, 0.025)
    assert isinstance(val, float)
    assert val > 0


class TestValuationUnittest(unittest.TestCase):
    def test_fcff_base(self):
        test_fcff_base_con_datos_reales()

    def test_wacc(self):
        test_wacc_datos_reales_de_mercado()

    def test_ma(self):
        test_fcff_valuation_mastercard_ma()

    def test_aapl(self):
        test_fcff_valuation_apple_aapl()

    def test_ko(self):
        test_fcff_valuation_coca_cola_ko()

    def test_fcf_negativo(self):
        test_fcff_valuation_fcf_negativo_normalizado()

    def test_ddm(self):
        test_ddm_gordon_growth()

    def test_crecimiento(self):
        test_curva_crecimiento_fundamentales_reales()

    def test_factory(self):
        test_factory_calculador_dcf()


if __name__ == "__main__":
    unittest.main()
