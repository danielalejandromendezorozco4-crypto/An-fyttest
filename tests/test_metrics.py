import pytest
import unittest
import pandas as pd
import numpy as np

from engine.metrics import (
    calcular_multiplos_valuacion,
    calcular_ratios_rentabilidad,
    calcular_ratios_solvencia,
    calcular_altman_zscore,
    calcular_piotroski_fscore,
    calcular_scoring,
    evaluar_veredicto,
)
from data.financial_fetcher import (
    extraer_metricas_ttm,
    obtener_capex_historico,
)


def test_multiplos_valuacion_estandar():
    """
    Verifica que los múltiplos de valuación se calculen según la metodología
    institucional estándar (Finviz / Yahoo Finance).
    """
    precio_actual = 200.0
    mcap = 3_000_000_000_000.0  # 3T USD
    eps_ttm = 6.50
    forward_eps = 7.50
    fcf_ttm = 100_000_000_000.0  # 100B USD
    ebitda_ttm = 130_000_000_000.0  # 130B USD
    total_debt = 100_000_000_000.0
    total_cash = 150_000_000_000.0
    revenue_ttm = 400_000_000_000.0
    total_equity = 80_000_000_000.0
    peg_info = 1.8
    earnings_growth = 0.12

    res = calcular_multiplos_valuacion(
        precio_actual=precio_actual,
        mcap=mcap,
        eps_ttm=eps_ttm,
        forward_eps=forward_eps,
        fcf_ttm=fcf_ttm,
        ebitda_ttm=ebitda_ttm,
        total_debt=total_debt,
        total_cash=total_cash,
        revenue_ttm=revenue_ttm,
        total_equity=total_equity,
        peg_info=peg_info,
        earnings_growth=earnings_growth,
    )

    assert isinstance(res, dict)

    # 1. PER Trailing = 200 / 6.50 = 30.769...
    pe_esperado = precio_actual / eps_ttm
    assert abs(res["pe"] - pe_esperado) < 0.01

    # 2. PER Forward = 200 / 7.50 = 26.666...
    pe_fwd_esperado = precio_actual / forward_eps
    assert abs(res["pe_forward"] - pe_fwd_esperado) < 0.01

    # 3. P/FCF = 3T / 100B = 30.0
    p_fcf_esperado = mcap / fcf_ttm
    assert abs(res["p_fcf"] - p_fcf_esperado) < 0.01

    # 4. EV / EBITDA = (3T + 100B - 150B) / 130B = 2.95T / 130B = 22.692...
    ev_esperado = mcap + total_debt - total_cash
    assert abs(res["enterprise_value"] - ev_esperado) < 1.0
    ev_ebitda_esperado = ev_esperado / ebitda_ttm
    assert abs(res["ev_ebitda"] - ev_ebitda_esperado) < 0.01

    # 5. P/S = 3T / 400B = 7.50
    p_s_esperado = mcap / revenue_ttm
    assert abs(res["p_s"] - p_s_esperado) < 0.01

    # 6. P/B = 3T / 80B = 37.50
    p_b_esperado = mcap / total_equity
    assert abs(res["p_b"] - p_b_esperado) < 0.01

    # 7. Formatos strings
    assert "x" in res["pe_str"]
    assert "x" in res["p_fcf_str"]
    assert "x" in res["ev_ebitda_str"]
    assert "x" in res["p_s_str"]
    assert "x" in res["p_b_str"]


def test_multiplos_valuacion_casos_limite():
    """
    Verifica el tratamiento defensivo de casos límite:
    - EPS negativo / cero
    - FCF negativo
    - EBITDA cero
    - Patrimonio negativo
    """
    res = calcular_multiplos_valuacion(
        precio_actual=50.0,
        mcap=500_000_000.0,
        eps_ttm=-2.0,       # Pérdidas
        forward_eps=0.0,
        fcf_ttm=-50_000_000.0,  # Quema de caja
        ebitda_ttm=0.0,
        total_debt=100_000_000.0,
        total_cash=10_000_000.0,
        revenue_ttm=0.0,
        total_equity=-20_000_000.0,  # Patrimonio negativo
        peg_info=0.0,
        earnings_growth=0.0,
    )

    assert res["pe"] == 0.0
    assert res["pe_str"] == "N/A"
    assert res["p_fcf"] == 0.0
    assert res["p_fcf_str"] == "N/A"
    assert res["ev_ebitda"] == 0.0
    assert res["ev_ebitda_str"] == "N/A"
    assert res["p_s"] == 0.0
    assert res["p_b"] == 0.0
    assert res["col_pe"] == "🔴"


def test_ratios_rentabilidad_estandar():
    """
    Verifica que los márgenes (Bruto, Operativo, Neto), ROE, ROA y ROIC
    se calculen de forma consistente con los reportes financieros institucionales.
    """
    rev = 100_000.0
    gross_prof = 45_000.0   # 45%
    ebit = 25_000.0         # 25%
    net_inc = 20_000.0      # 20%
    ta = 150_000.0          # ROA = 20 / 150 = 13.33%
    te = 80_000.0           # ROE = 20 / 80 = 25.0%
    cl = 30_000.0
    short_debt = 5_000.0
    tax_rate = 0.20

    res = calcular_ratios_rentabilidad(
        revenue_ttm=rev,
        gross_profit_ttm=gross_prof,
        operating_income_ttm=ebit,
        net_income_ttm=net_inc,
        total_assets=ta,
        total_equity=te,
        total_debt=20_000.0,
        total_cash=10_000.0,
        current_liabilities=cl,
        short_term_debt=short_debt,
        tax_rate=tax_rate,
        is_asset_light=False,
    )

    assert isinstance(res, dict)
    assert abs(res["gross_margin"] - 45.0) < 0.01
    assert abs(res["mg_op"] - 25.0) < 0.01
    assert abs(res["net_margin"] - 20.0) < 0.01
    assert abs(res["roe"] - 25.0) < 0.01
    assert abs(res["roa"] - (20_000.0 / 150_000.0 * 100.0)) < 0.01

    # NOPAT = 25,000 * (1 - 0.20) = 20,000
    assert abs(res["nopat"] - 20_000.0) < 0.01

    # Invested Capital = 150,000 - (30,000 - 5,000) = 125,000
    assert abs(res["invested_capital"] - 125_000.0) < 0.01

    # ROIC = 20,000 / 125,000 * 100 = 16.0%
    assert abs(res["roic"] - 16.0) < 0.01
    assert "%" in res["roic_str"]


def test_ratios_solvencia_estandar():
    """
    Verifica que Deuda Neta, Net Debt/EBITDA, Cobertura de Intereses,
    Razón Corriente y Debt/Equity se calculen con precisión institucional.
    """
    total_debt = 50_000.0
    total_cash = 20_000.0
    total_eq = 40_000.0
    ebitda = 15_000.0
    ebit = 12_000.0
    int_exp = 2_000.0
    cur_assets = 35_000.0
    cur_liab = 25_000.0
    fcf = 10_000.0
    shares = 1_000.0

    res = calcular_ratios_solvencia(
        total_debt=total_debt,
        total_cash=total_cash,
        total_equity=total_eq,
        ebitda_ttm=ebitda,
        ebit_ttm=ebit,
        interest_expense=int_exp,
        current_assets=cur_assets,
        current_liabilities=cur_liab,
        fcf_ttm=fcf,
        shares_current=shares,
        is_fibra_util=False,
    )

    # Net Debt = 50,000 - 20,000 = 30,000
    assert res["net_debt"] == 30_000.0

    # Net Debt / EBITDA = 30,000 / 15,000 = 2.0x
    assert abs(res["net_debt_ebitda"] - 2.0) < 0.01
    assert "2.00x" in res["val_nde"]

    # Interest Coverage = 12,000 / 2,000 = 6.0x
    assert abs(res["cob_int"] - 6.0) < 0.01
    assert "6.0x" in res["val_cob"]

    # Current Ratio = 35,000 / 25,000 = 1.40x
    assert abs(res["cur_ratio"] - 1.40) < 0.01

    # Debt / Equity = 50,000 / 40,000 = 1.25x
    assert abs(res["debt_eq"] - 1.25) < 0.01

    # FCF / Debt = 10,000 / 50,000 * 100 = 20.0%
    assert abs(res["fcf_debt_ratio"] - 20.0) < 0.01

    # Net Cash / Share = (20,000 - 50,000) / 1,000 = -30.0
    assert abs(res["net_cash_per_share"] - (-30.0)) < 0.01


def test_fcf_ttm_sin_descalce_de_signos():
    """
    Criterio 2: Verifica que FCF TTM se calcule como OCF - |CapEx|
    sin descalces de signos, tanto si CapEx viene negativo (yfinance)
    como positivo (FMP).
    """
    # Caso 1: CapEx reportado como negativo (estilo yfinance)
    cf_yf = pd.DataFrame({
        "2023": [100_000.0, -15_000.0],
    }, index=["Operating Cash Flow", "Capital Expenditure"])

    capex_yf = obtener_capex_historico(cf_yf)
    assert capex_yf[0] == 15_000.0  # Normalizado a positivo

    m_yf = extraer_metricas_ttm(
        info={"operatingCashflow": 100_000.0},
        inc=pd.DataFrame(),
        bs=pd.DataFrame(),
        cf=cf_yf,
        precio_actual=100.0,
    )
    # FCF = 100,000 - 15,000 = 85,000
    assert m_yf["fcf_ttm"] == 85_000.0

    # Caso 2: CapEx reportado como positivo (estilo FMP)
    cf_fmp = pd.DataFrame({
        "2023": [120_000.0, 20_000.0],
    }, index=["Operating Cash Flow", "Capital Expenditure"])

    capex_fmp = obtener_capex_historico(cf_fmp)
    assert capex_fmp[0] == 20_000.0

    m_fmp = extraer_metricas_ttm(
        info={"operatingCashflow": 120_000.0},
        inc=pd.DataFrame(),
        bs=pd.DataFrame(),
        cf=cf_fmp,
        precio_actual=100.0,
    )
    # FCF = 120,000 - 20,000 = 100,000
    assert m_fmp["fcf_ttm"] == 100_000.0


def test_tolerancia_activos_clave_finviz_yahoo():
    """
    Criterio 1: Verifica que para perfiles sintéticos representativos de
    Mega Caps (AAPL, MSFT, GOOGL, NVDA), los múltiplos PER, P/FCF, EV/EBITDA
    y márgenes calculados coincidan dentro del rango de tolerancia del ±2%.
    """
    perfiles = {
        "AAPL": {
            "precio": 195.0, "mcap": 3_000e9, "eps": 6.50, "fcf": 105e9,
            "ebitda": 130e9, "debt": 110e9, "cash": 160e9, "rev": 390e9,
            "gross_prof": 175e9, "ebit": 120e9, "net_inc": 100e9, "equity": 75e9,
        },
        "MSFT": {
            "precio": 420.0, "mcap": 3_120e9, "eps": 11.50, "fcf": 70e9,
            "ebitda": 125e9, "debt": 80e9, "cash": 110e9, "rev": 245e9,
            "gross_prof": 170e9, "ebit": 110e9, "net_inc": 88e9, "equity": 250e9,
        },
        "GOOGL": {
            "precio": 175.0, "mcap": 2_180e9, "eps": 7.20, "fcf": 68e9,
            "ebitda": 115e9, "debt": 30e9, "cash": 120e9, "rev": 320e9,
            "gross_prof": 180e9, "ebit": 100e9, "net_inc": 85e9, "equity": 300e9,
        },
        "NVDA": {
            "precio": 125.0, "mcap": 3_050e9, "eps": 2.50, "fcf": 55e9,
            "ebitda": 80e9, "debt": 12e9, "cash": 35e9, "rev": 100e9,
            "gross_prof": 75e9, "ebit": 65e9, "net_inc": 60e9, "equity": 60e9,
        },
    }

    for ticker, d in perfiles.items():
        res_mult = calcular_multiplos_valuacion(
            precio_actual=d["precio"],
            mcap=d["mcap"],
            eps_ttm=d["eps"],
            forward_eps=d["eps"] * 1.15,
            fcf_ttm=d["fcf"],
            ebitda_ttm=d["ebitda"],
            total_debt=d["debt"],
            total_cash=d["cash"],
            revenue_ttm=d["rev"],
            total_equity=d["equity"],
            peg_info=1.5,
            earnings_growth=0.15,
        )

        res_rent = calcular_ratios_rentabilidad(
            revenue_ttm=d["rev"],
            gross_profit_ttm=d["gross_prof"],
            operating_income_ttm=d["ebit"],
            net_income_ttm=d["net_inc"],
            total_assets=d["equity"] + d["debt"],
            total_equity=d["equity"],
            total_debt=d["debt"],
            total_cash=d["cash"],
            current_liabilities=20e9,
            short_term_debt=5e9,
            tax_rate=0.20,
        )

        # Validación PER (±2% tolerancia)
        pe_ref = d["precio"] / d["eps"]
        assert abs(res_mult["pe"] - pe_ref) / pe_ref <= 0.02, f"Fallo PER en {ticker}"

        # Validación P/FCF (±2% tolerancia)
        pfcf_ref = d["mcap"] / d["fcf"]
        assert abs(res_mult["p_fcf"] - pfcf_ref) / pfcf_ref <= 0.02, f"Fallo P/FCF en {ticker}"

        # Validación EV/EBITDA (±2% tolerancia)
        ev_ref = (d["mcap"] + d["debt"] - d["cash"]) / d["ebitda"]
        assert abs(res_mult["ev_ebitda"] - ev_ref) / ev_ref <= 0.02, f"Fallo EV/EBITDA en {ticker}"

        # Validación Margen Operativo (±2% tolerancia)
        mg_op_ref = (d["ebit"] / d["rev"]) * 100.0
        assert abs(res_rent["mg_op"] - mg_op_ref) / mg_op_ref <= 0.02, f"Fallo Margen Op en {ticker}"


def test_extraer_metricas_ttm_nflx_defensivo():
    """
    Verifica que la extracción de métricas TTM para tickers como NFLX o empresas
    sin datos de earningsGrowth no lance NameError ni KeyError y maneje valores nulos de forma segura.
    """
    info_nflx_incompleto = {
        "symbol": "NFLX",
        "longName": "Netflix Inc.",
        "marketCap": 280_000_000_000.0,
        "sharesOutstanding": 430_000_000.0,
        "totalDebt": 14_000_000_000.0,
        "totalCash": 7_000_000_000.0,
        "totalRevenue": 36_000_000_000.0,
        "operatingIncome": 8_000_000_000.0,
        "netIncomeToCommon": 6_500_000_000.0,
        "operatingCashflow": 7_500_000_000.0,
        "freeCashflow": 6_500_000_000.0,
        # earningsGrowth omitido intencionalmente para probar resiliencia
        "beta": 1.25,
    }

    inc_nflx = pd.DataFrame({
        "2023": [36e9, 8e9, 6.5e9],
        "2022": [31e9, 5.6e9, 4.4e9],
    }, index=["Total Revenue", "Operating Income", "Net Income"])

    bs_nflx = pd.DataFrame({
        "2023": [14e9, 7e9, 48e9],
        "2022": [14.5e9, 5.1e9, 45e9],
    }, index=["Total Debt", "Cash And Cash Equivalents", "Total Assets"])

    cf_nflx = pd.DataFrame({
        "2023": [7.5e9, -1.0e9, 6.5e9],
        "2022": [2.0e9, -0.4e9, 1.6e9],
    }, index=["Operating Cash Flow", "Capital Expenditure", "Free Cash Flow"])

    # No debe lanzar NameError ni KeyError
    m = extraer_metricas_ttm(info_nflx_incompleto, inc_nflx, bs_nflx, cf_nflx, precio_actual=650.0)

    assert isinstance(m, dict)
    assert "earnings_growth" in m
    assert "revenue_growth" in m
    assert "cagr_revenue_3_5y" in m
    assert "op_margin_hist" in m
    assert "shares_diluted" in m
    assert m["mcap"] == 280_000_000_000.0
    assert m["earnings_growth"] >= 0.0  # Ingerido de Net Income CAGR


class TestMetricsUnittest(unittest.TestCase):
    def test_multiplos_estandar(self):
        test_multiplos_valuacion_estandar()

    def test_multiplos_casos_limite(self):
        test_multiplos_valuacion_casos_limite()

    def test_rentabilidad_estandar(self):
        test_ratios_rentabilidad_estandar()

    def test_solvencia_estandar(self):
        test_ratios_solvencia_estandar()

    def test_fcf_sin_descalce(self):
        test_fcf_ttm_sin_descalce_de_signos()

    def test_tolerancia_activos(self):
        test_tolerancia_activos_clave_finviz_yahoo()

    def test_nflx_defensivo(self):
        test_extraer_metricas_ttm_nflx_defensivo()


if __name__ == "__main__":
    unittest.main()

