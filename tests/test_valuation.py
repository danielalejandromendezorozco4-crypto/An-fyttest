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
    calcular_valoracion_multiplos,
    calcular_valuacion_adaptativa,
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
# 3. PRUEBAS DE VALORACIÓN POR MÚLTIPLOS Y DDM
# ─────────────────────────────────────────────────────────────────────────────

def test_valoracion_multiplos_normalizados():
    res = calcular_valoracion_multiplos(
        precio_actual=100.0,
        mcap=100_000_000.0,
        eps_ttm=5.0,
        forward_eps=6.0,
        fcf_ttm=6_000_000.0,
        ebitda_ttm=10_000_000.0,
        total_debt=10_000_000.0,
        total_cash=20_000_000.0,
        total_equity=50_000_000.0,
        shares_diluted=1_000_000.0,
        sector="Technology",
        benchmark_pe=22.0,
        benchmark_pfcf=18.0,
        benchmark_ev_ebitda=14.0,
    )
    assert res["viable"] is True
    assert res["valor_intrinseco_multiplos"] > 50.0


def test_ddm_gordon_growth():
    res = calcular_ddm(div_rate=3.0, ke=9.0, g_div=0.03, precio_actual=50.0)
    assert res["viable"] is True
    # v = 3.0 * 1.03 / (0.09 - 0.03) = 3.09 / 0.06 = 51.50
    assert abs(res["valor_intrinseco_ddm"] - 51.50) < 0.10


# ─────────────────────────────────────────────────────────────────────────────
# 4. PRUEBAS DEL MOTOR ADAPTATIVO MULTIMODELO
# ─────────────────────────────────────────────────────────────────────────────

def test_valuacion_adaptativa_madura_dividendo_ko():
    """Empresa madura con dividendos estables (The Coca-Cola Company / KO)."""
    res = calcular_valuacion_adaptativa(
        precio_actual=65.0,
        mcap=280e9,
        shares_diluted=4.3e9,
        total_debt=40e9,
        total_cash=14e9,
        beta=0.60,
        rf=4.20,
        erp=5.0,
        ocf_hist=[11.5e9, 11.0e9, 10.5e9],
        capex_hist=[1.9e9, 1.8e9, 1.5e9],
        interest_hist=[1.5e9, 1.4e9, 1.3e9],
        pretax_hist=[13.0e9, 12.5e9, 11.8e9],
        taxprov_hist=[2.6e9, 2.5e9, 2.3e9],
        revenue_ttm=46e9,
        gross_profit_ttm=27e9,
        operating_income_ttm=13e9,
        net_income_ttm=10.5e9,
        ebitda_ttm=15e9,
        fcf_ttm=9.6e9,
        total_equity=28e9,
        eps_ttm=2.50,
        forward_eps=2.85,
        div_rate=1.94,
        div_yield=3.0,
        sector="Consumer Defensive",
        benchmark_pe=22.0,
        benchmark_pfcf=20.0,
    )
    assert res["valor_intrinseco"] > 40.0
    assert res["escenario_pesimista"] < res["escenario_base"] < res["escenario_optimista"]
    assert res["modelos_detalle"]["ddm"]["viable"] is True
    assert res["modelos_detalle"]["dcf"]["viable"] is True
    assert "Madura" in res["perfil_empresa"]


def test_valuacion_adaptativa_alto_crecimiento_nvda():
    """Empresa de alto crecimiento y alto retorno sobre capital (NVIDIA / NVDA)."""
    res = calcular_valuacion_adaptativa(
        precio_actual=125.0,
        mcap=3000e9,
        shares_diluted=24.5e9,
        total_debt=10e9,
        total_cash=35e9,
        beta=1.65,
        rf=4.20,
        erp=5.0,
        ocf_hist=[60e9, 28e9, 5e9],
        capex_hist=[3e9, 1.5e9, 0.8e9],
        interest_hist=[0.3e9, 0.25e9, 0.2e9],
        pretax_hist=[65e9, 32e9, 6e9],
        taxprov_hist=[8e9, 4e9, 0.8e9],
        revenue_ttm=100e9,
        gross_profit_ttm=75e9,
        operating_income_ttm=62e9,
        net_income_ttm=55e9,
        ebitda_ttm=65e9,
        fcf_ttm=57e9,
        total_equity=60e9,
        eps_ttm=2.25,
        forward_eps=3.80,
        div_rate=0.04,
        div_yield=0.03,
        revenue_growth_api=0.35,
        sector="Technology",
        benchmark_pe=32.0,
        benchmark_pfcf=30.0,
    )
    assert res["valor_intrinseco"] > 70.0
    assert res["escenario_pesimista"] < res["escenario_optimista"]
    assert res["modelos_detalle"]["dcf"]["viable"] is True


def test_valuacion_adaptativa_financiera_ma():
    """Institución financiera / procesamiento de pagos (Mastercard / MA)."""
    res = calcular_valuacion_adaptativa(
        precio_actual=485.0,
        mcap=450e9,
        shares_diluted=930e6,
        total_debt=16e9,
        total_cash=8.5e9,
        beta=1.05,
        rf=4.25,
        erp=5.0,
        ocf_hist=[12.5e9, 10.5e9, 9.5e9],
        capex_hist=[0.5e9, 0.4e9, 0.4e9],
        interest_hist=[0.5e9, 0.45e9, 0.4e9],
        pretax_hist=[14.0e9, 12.0e9, 10.5e9],
        taxprov_hist=[2.5e9, 2.1e9, 1.8e9],
        revenue_ttm=25e9,
        gross_profit_ttm=25e9,
        operating_income_ttm=14e9,
        net_income_ttm=11.5e9,
        ebitda_ttm=15e9,
        fcf_ttm=12.0e9,
        total_equity=7.0e9,
        eps_ttm=13.50,
        forward_eps=15.80,
        div_rate=2.64,
        div_yield=0.55,
        revenue_growth_api=0.12,
        sector="Financial Services",
        benchmark_pe=28.0,
        benchmark_pb=10.0,
    )
    assert res["valor_intrinseco"] > 350.0
    assert res["modelos_detalle"]["multiplos"]["viable"] is True


def test_valuacion_adaptativa_fcf_negativo_defensivo():
    """Empresa con FCF negativo transitorio donde el motor repondera hacia múltiplos."""
    res = calcular_valuacion_adaptativa(
        precio_actual=50.0,
        mcap=20e9,
        shares_diluted=400e6,
        total_debt=5e9,
        total_cash=3e9,
        beta=1.30,
        rf=4.20,
        erp=5.0,
        ocf_hist=[-0.2e9, 0.1e9],
        capex_hist=[0.8e9, 0.5e9],
        interest_hist=[0.2e9, 0.15e9],
        pretax_hist=[0.5e9, 0.2e9],
        taxprov_hist=[0.1e9, 0.05e9],
        revenue_ttm=10e9,
        gross_profit_ttm=4e9,
        operating_income_ttm=0.8e9,
        net_income_ttm=0.4e9,
        ebitda_ttm=1.2e9,
        fcf_ttm=-1.0e9,  # FCF negativo
        total_equity=8e9,
        eps_ttm=1.00,
        forward_eps=1.50,
        div_rate=0.0,
        div_yield=0.0,
        revenue_growth_api=0.20,
        sector="Technology",
        benchmark_pe=25.0,
    )
    assert res["valor_intrinseco"] > 0
    assert res["advertencia_calidad"] != ""
    assert res["modelos_detalle"]["multiplos"]["viable"] is True


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

    def test_multiplos(self):
        test_valoracion_multiplos_normalizados()

    def test_ddm(self):
        test_ddm_gordon_growth()

    def test_ko(self):
        test_valuacion_adaptativa_madura_dividendo_ko()

    def test_nvda(self):
        test_valuacion_adaptativa_alto_crecimiento_nvda()

    def test_ma(self):
        test_valuacion_adaptativa_financiera_ma()

    def test_fcf_negativo(self):
        test_valuacion_adaptativa_fcf_negativo_defensivo()

    def test_factory(self):
        test_factory_calculador_dcf()


if __name__ == "__main__":
    unittest.main()
