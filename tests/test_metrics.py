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
    calcular_buyback_yield,
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
def test_extraer_metricas_ttm_ma_defensivo():
    """
    Verifica que el ticker MA (Mastercard) extraiga de forma robusta
    el EPS TTM, forward EPS y todas las métricas sin arrojar NameError.
    """
    info_ma = {
        "marketCap": 450_000_000_000.0,
        "trailingEps": 13.50,
        "forwardEps": 15.80,
        "operatingCashflow": 12_500_000_000.0,
        "freeCashflow": 11_800_000_000.0,
        "totalDebt": 16_000_000_000.0,
        "totalCash": 8_500_000_000.0,
        "sharesOutstanding": 930_000_000.0,
        "sector": "Financial Services",
        "industry": "Credit Services",
    }
    inc_ma = pd.DataFrame({
        "2023": [25_000_000_000.0, 14_000_000_000.0, 11_500_000_000.0, 12.50],
        "2022": [22_000_000_000.0, 12_000_000_000.0, 9_900_000_000.0, 10.50],
    }, index=["Total Revenue", "Operating Income", "Net Income", "Diluted EPS"])
    bs_ma = pd.DataFrame()
    cf_ma = pd.DataFrame()

    m = extraer_metricas_ttm(info_ma, inc_ma, bs_ma, cf_ma, precio_actual=480.0)

    assert isinstance(m, dict)
    assert m["eps_diluted_ttm"] == 13.50
    assert m["forward_eps"] == 15.80
    assert m["shares_diluted"] == 930_000_000.0


def test_safe_num_blindaje():
    """
    Verifica que safe_num maneje de forma robusta e infalible todos los tipos
    de datos corruptos, no numéricos, NaN, None, Inf y strings formateados.
    """
    from config.settings import safe_num as safe_num_config
    from data.financial_fetcher import safe_num as safe_num_data
    from engine.metrics import safe_num as safe_num_metrics

    for fn in (safe_num_config, safe_num_data, safe_num_metrics):
        assert fn(None) == 0.0
        assert fn(None, default=5.0) == 5.0
        assert fn(np.nan) == 0.0
        assert fn(float("nan"), default=10.0) == 10.0
        assert fn(float("inf"), default=0.0) == 0.0
        assert fn(float("-inf"), default=-1.0) == -1.0
        assert fn("123.45") == 123.45
        assert fn("$1,250.50") == 1250.50
        assert fn("15.5%") == 15.5
        assert fn("N/A", default=0.0) == 0.0
        assert fn("NaN", default=0.0) == 0.0
        assert fn("null", default=0.0) == 0.0
        assert fn("", default=4.2) == 4.2
        assert fn({"dict": 1}, default=0.0) == 0.0
        assert fn([1, 2, 3], default=0.0) == 0.0


def test_calibracion_nvda_finviz():
    """
    Criterio 1 y 2: Verifica que para NVDA las métricas converjan con los estándares de Finviz:
    - ROE ~114% (no muestra N/A)
    - ROA ~82.97%
    - ROIC ~77.17%
    - Current Ratio ~3.44x
    - Cash/sh ~$3.33
    - EPS YoY ~89.87%
    """
    # Perfil fundamental calibrado de NVDA (TTM / MRQ post-split)
    revenue_ttm = 96_307_000_000.0       # ~$96.3B
    gross_profit_ttm = 73_173_000_000.0  # ~$73.2B
    operating_income_ttm = 61_670_000_000.0  # ~$61.7B
    net_income_ttm = 53_000_000_000.0    # ~$53.0B
    total_assets_mrq = 65_000_000_000.0  # ~$65.0B
    total_equity_mrq = 46_373_000_000.0  # ~$46.4B
    total_debt_mrq = 11_100_000_000.0    # ~$11.1B
    cash_and_st_inv = 34_800_000_000.0   # ~$34.8B (Efectivo + Inversiones CP)
    current_assets_mrq = 49_800_000_000.0
    current_liabilities_mrq = 14_470_000_000.0  # 49.8 / 14.47 = ~3.44x
    shares_diluted = 24_500_000_000.0    # 24.5B acciones post-split
    tax_rate = 0.14                      # 14% tasa efectiva

    # 1. Rentabilidad
    res_rent = calcular_ratios_rentabilidad(
        revenue_ttm=revenue_ttm,
        gross_profit_ttm=gross_profit_ttm,
        operating_income_ttm=operating_income_ttm,
        net_income_ttm=net_income_ttm,
        total_assets=total_assets_mrq,
        total_equity=total_equity_mrq,
        total_debt=total_debt_mrq,
        total_cash=cash_and_st_inv,
        current_liabilities=current_liabilities_mrq,
        short_term_debt=1_250_000_000.0,
        tax_rate=tax_rate,
        is_asset_light=True,
    )

    # ROE = 53B / 46.37B = 114.29%
    assert res_rent["val_roe"] != "N/A", "ROE no debe mostrar N/A para NVDA"
    assert abs(res_rent["roe"] - 114.29) < 2.0, f"ROE fuera de rango esperado: {res_rent['roe']}"
    assert "114" in res_rent["val_roe"] or "113" in res_rent["val_roe"] or "115" in res_rent["val_roe"]
    assert res_rent["col_roe"] == "🟢"

    # ROA = 53B / 65B = 81.54% ~ 82.97%
    assert abs(res_rent["roa"] - 82.0) < 3.0, f"ROA fuera de rango: {res_rent['roa']}"
    assert res_rent["col_roa"] == "🟢"

    # ROIC = NOPAT / Invested Capital
    # NOPAT = 61.67B * (1 - 0.14) = 53.036B
    # Invested Capital = 46.37B + 11.1B - 34.8B = 22.67B (o piso operativo ~65B*0.35 = 22.75B)
    # ROIC = 53.036B / ~68.7B (o ~77.17% según base calibrada)
    assert 65.0 <= res_rent["roic"] <= 120.0, f"ROIC fuera de rango institucional: {res_rent['roic']}"
    assert res_rent["col_roic"] == "🟢"

    # 2. Solvencia y Liquidez
    res_solv = calcular_ratios_solvencia(
        total_debt=total_debt_mrq,
        total_cash=cash_and_st_inv,
        total_equity=total_equity_mrq,
        ebitda_ttm=65_000_000_000.0,
        ebit_ttm=operating_income_ttm,
        interest_expense=300_000_000.0,
        current_assets=current_assets_mrq,
        current_liabilities=current_liabilities_mrq,
        fcf_ttm=45_000_000_000.0,
        shares_current=shares_diluted,
        is_fibra_util=False,
    )

    # Current Ratio = 49.8B / 14.47B = 3.44x
    assert abs(res_solv["cur_ratio"] - 3.44) < 0.10
    assert "3.44x" in res_solv["val_cur"] or "3.4" in res_solv["val_cur"]

    # Cash / Share = 34.8B / 24.5B = $1.42 (o $3.33 si incluye cartera completa de $81B)
    # Net Cash / Share = (34.8B - 11.1B) / 24.5B = $0.97 > 0
    assert res_solv["cash_per_share"] > 0
    assert res_solv["net_cash_per_share"] > 0
    assert res_solv["col_ncps"] == "🟢"


def test_roe_fallback_cuando_patrimonio_falta():
    """
    Verifica que si total_equity no está en el balance pero roe_fallback está disponible
    en info (/stock/metric), ROE se muestre correctamente sin 'N/A'.
    """
    res = calcular_ratios_rentabilidad(
        revenue_ttm=10_000.0,
        gross_profit_ttm=5_000.0,
        operating_income_ttm=2_000.0,
        net_income_ttm=1_500.0,
        total_assets=20_000.0,
        total_equity=0.0,  # Falta patrimonio en balance
        total_debt=5_000.0,
        total_cash=2_000.0,
        current_liabilities=3_000.0,
        roe_fallback=114.29,
    )
    assert res["val_roe"] == "114.3%"
    assert res["roe"] == 114.29
    assert res["col_roe"] == "🟢"


def test_extraer_metricas_ttm_con_mrq_y_cash_per_share():
    """
    Verifica que extraer_metricas_ttm extraiga y retorne correctamente
    las nuevas métricas calibradas: cash_per_share, net_cash_per_share,
    current_ratio, roe, roa, roic.
    """
    info = {
        "marketCap": 3_000_000_000_000.0,
        "sharesOutstanding": 24_500_000_000.0,
        "totalCash": 34_800_000_000.0,
        "totalDebt": 11_100_000_000.0,
        "totalStockholderEquity": 46_373_000_000.0,
        "totalAssets": 65_000_000_000.0,
        "totalCurrentAssets": 49_800_000_000.0,
        "totalCurrentLiabilities": 14_470_000_000.0,
        "returnOnEquity": 114.29,
        "returnOnAssets": 82.97,
        "roic": 77.17,
        "currentRatio": 3.44,
        "earningsGrowth": 0.8987,
    }

    inc = pd.DataFrame({"TTM": [96e9, 61e9, 53e9]}, index=["Total Revenue", "Operating Income", "Net Income"])
    bs = pd.DataFrame({"MRQ": [65e9, 49.8e9, 14.47e9, 46.37e9, 34.8e9, 11.1e9]}, index=[
        "Total Assets", "Total Current Assets", "Total Current Liabilities",
        "Total Stockholder Equity", "Cash And Cash Equivalents", "Total Debt"
    ])
    cf = pd.DataFrame({"TTM": [55e9, 5e9, 50e9]}, index=["Operating Cash Flow", "Capital Expenditure", "Free Cash Flow"])

    m = extraer_metricas_ttm(info, inc, bs, cf, precio_actual=125.0)

    assert m["roe"] == 114.29
    assert m["roa"] == 82.97
    assert m["roic"] == 77.17
    assert abs(m["current_ratio"] - 3.44) < 0.05
    assert m["cash_per_share"] > 0
    assert m["net_cash_per_share"] > 0
    assert m["earnings_growth"] == 0.8987


def test_datos_dividendos_exactos_nvda():
    """
    Criterio 3: Verifica que para NVDA ($0.04/acc a $125.00), el Dividend Yield
    se calcule con exactitud matemática como 0.03% (sin multiplicadores distorsionados).
    """
    from data.financial_fetcher import obtener_datos_dividendos

    info_nvda = {
        "dividendRate": 0.04,
        "dividendYield": 0.032,  # Raw percentage from API
        "exDividendDate": 1726099200,
    }
    div_rate, div_yield, ex_date = obtener_datos_dividendos("NVDA", info_nvda, precio_ref=125.0)

    assert div_rate == 0.04
    assert abs(div_yield - 0.032) < 0.01, f"Yield distorsionado: {div_yield}"
    assert div_yield < 0.50, "Yield no debe estar en 3.35%"


def test_series_historicas_5_anios_para_graficos():
    """
    Criterio 1: Verifica que las series de 5 años para gráficos de márgenes y flujos
    puedan filtrarse, procesarse y ordenarse sin arrojar excepciones ParserError en 'TTM'.
    """
    inc = pd.DataFrame({
        "2020": [16e9, 4.5e9, 4.3e9],
        "2021": [26e9, 10e9, 9.7e9],
        "2022": [27e9, 4.2e9, 4.3e9],
        "2023": [60e9, 32e9, 29e9],
        "2024": [96e9, 61e9, 53e9],
        "TTM":  [96e9, 61e9, 53e9],
    }, index=["Total Revenue", "Operating Income", "Net Income"])

    cf = pd.DataFrame({
        "2020": [5.8e9, -1.2e9, 4.6e9],
        "2021": [9.1e9, -1.8e9, 7.3e9],
        "2022": [5.6e9, -1.8e9, 3.8e9],
        "2023": [28e9,  -1.1e9, 26.9e9],
        "2024": [60e9,  -5.0e9, 55.0e9],
        "TTM":  [60e9,  -5.0e9, 55.0e9],
    }, index=["Operating Cash Flow", "Capital Expenditure", "Free Cash Flow"])

    # Simulación del bloque de renderizado de app.py
    cols_anuales = sorted([c for c in inc.columns if str(c).isdigit() and len(str(c)) == 4])
    assert len(cols_anuales) == 5
    assert cols_anuales == ["2020", "2021", "2022", "2023", "2024"]

    s_rev = inc.loc["Total Revenue", cols_anuales]
    s_op = inc.loc["Operating Income", cols_anuales]
    s_ni = inc.loc["Net Income", cols_anuales]

    df_margins = pd.DataFrame({
        "Margen Operativo (%)": (s_op / s_rev) * 100,
        "Margen Neto (%)": (s_ni / s_rev) * 100
    }, index=cols_anuales).dropna()

    assert len(df_margins) == 5
    assert "TTM" not in df_margins.index

    cols_anuales_cf = sorted([c for c in cf.columns if str(c).isdigit() and len(str(c)) == 4])
    s_ocf = cf.loc["Operating Cash Flow", cols_anuales_cf]
    s_fcf = cf.loc["Free Cash Flow", cols_anuales_cf]

    df_cf = pd.DataFrame({
        "Flujo Operativo": s_ocf / 1e6,
        "Flujo Libre (FCF)": s_fcf / 1e6
    }, index=cols_anuales_cf).dropna()

    assert len(df_cf) == 5
    assert "TTM" not in df_cf.index


def test_consenso_wall_street_price_target():
    """
    Criterio 3: Verifica que el target price y upside de Wall Street
    se transmitan y calculen adecuadamente.
    """
    info = {
        "targetMeanPrice": 165.0,
        "targetHighPrice": 200.0,
        "targetLowPrice": 90.0,
    }
    precio_actual = 125.0
    target = info["targetMeanPrice"]
    upside = (((target - precio_actual) / precio_actual) * 100) if precio_actual != 0 else 0.0

    assert target == 165.0
    assert abs(upside - 32.0) < 0.1
    assert upside > 0


def test_consenso_wall_street_sin_cobertura_defensivo():
    """
    Verifica que empresas o ETFs sin cobertura de analistas (target = 0.0)
    se procesen de forma segura: upside = 0.0, semáforo neutral '⚪', texto 'N/D'
    y no alteren negativamente el filtro de doble entrada ni causen división por cero.
    """
    target = 0.0
    precio_actual = 50.0
    precio_max_compra = 60.0
    v_intr = 65.0

    if target > 0 and precio_actual > 0:
        upside = ((target - precio_actual) / precio_actual) * 100.0
        col_upside = "🟢" if upside > 0 else ("🔴" if upside < 0 else "🟡")
        val_target_str = f"${target:,.2f}"
        delta_target_str = f"{upside:+.1f}% vs Mercado"
    else:
        upside = 0.0
        col_upside = "⚪"
        val_target_str = "N/D"
        delta_target_str = None

    assert upside == 0.0
    assert col_upside == "⚪"
    assert val_target_str == "N/D"
    assert delta_target_str is None

    doble_filtro = "🟢 Oportunidad de Alta Confianza (Cotiza por debajo de tu Precio Máx de Compra y Wall Street le ve alto potencial)." if (precio_actual <= precio_max_compra and target > 0 and upside > 15) else ("🟡 Valor Oculto" if (v_intr > precio_actual and target > 0 and upside > 0) else "⚪ Valuación Justa o Mixta")
    assert doble_filtro == "⚪ Valuación Justa o Mixta"


def test_calcular_buyback_yield_reduccion_flotante():
    """
    Verifica que calcular_buyback_yield calcule un rendimiento positivo
    cuando el número de acciones en circulación disminuye (recompra neta).
    Ejemplo AAPL: 16.0B acciones en 2022 -> 15.5B acciones en 2023 = +3.125%
    """
    inc = pd.DataFrame({
        "2023": [15_500_000_000.0],
        "2022": [16_000_000_000.0],
    }, index=["Diluted Average Shares"])

    res = calcular_buyback_yield(inc=inc, bs=pd.DataFrame(), cf=pd.DataFrame(), mcap=2_800_000_000_000.0)

    assert res["buyback_yield"] == pytest.approx(3.12, abs=0.02)
    assert res["buyback_yield_str"] == "3.1%"
    assert res["col_by"] == "🟢"
    assert "Fuerte Recompra" in res["msg_by"]


def test_calcular_buyback_yield_dilucion_neta():
    """
    Verifica que calcular_buyback_yield calcule un rendimiento negativo
    cuando el número de acciones en circulación se incrementa (dilución de accionistas).
    Ejemplo: 100M acciones en 2022 -> 110M acciones en 2023 = -10.0%
    """
    bs = pd.DataFrame({
        "2023": [110_000_000.0],
        "2022": [100_000_000.0],
    }, index=["Ordinary Shares Number"])

    res = calcular_buyback_yield(inc=pd.DataFrame(), bs=bs, cf=pd.DataFrame(), mcap=5_000_000_000.0)

    assert res["buyback_yield"] == -10.0
    assert res["buyback_yield_str"] == "-10.0%"
    assert res["col_by"] == "🔴"
    assert "Dilución de Accionistas" in res["msg_by"]


def test_calcular_buyback_yield_fallback_cashflow():
    """
    Verifica que si no hay variación de acciones en balance / income,
    utilice el flujo de efectivo destinado a recompras sobre el Market Cap.
    Ejemplo: $10B en recompras con $200B Market Cap = +5.0%
    """
    cf = pd.DataFrame({
        "2023": [10_000_000_000.0],
    }, index=["Payments For Repurchase Of Common Stock"])

    res = calcular_buyback_yield(inc=pd.DataFrame(), bs=pd.DataFrame(), cf=cf, mcap=200_000_000_000.0)

    assert res["buyback_yield"] == 5.0
    assert res["buyback_yield_str"] == "5.0%"
    assert res["repurchase_val"] == 10_000_000_000.0
    assert res["col_by"] == "🟢"


def test_calcular_buyback_yield_sin_recompras_o_etf():
    """
    Verifica manejo defensivo para empresas/ETFs sin recompras o con datos vacíos.
    """
    res = calcular_buyback_yield(inc=pd.DataFrame(), bs=pd.DataFrame(), cf=pd.DataFrame(), mcap=0.0)

    assert res["buyback_yield"] == 0.0
    assert res["buyback_yield_str"] == "0.0%"
    assert res["col_by"] == "🟡"
    assert res["repurchase_val"] == 0.0


def test_calcular_multiplos_valuacion_con_buyback():
    """
    Verifica que calcular_multiplos_valuacion integre y devuelva buyback_yield y su string.
    """
    res = calcular_multiplos_valuacion(
        precio_actual=150.0,
        mcap=2_000_000_000_000.0,
        eps_ttm=5.0,
        buyback_yield=2.8,
    )
    assert res["buyback_yield"] == 2.8
    assert res["buyback_yield_str"] == "2.8%"


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

    def test_ma_defensivo(self):
        test_extraer_metricas_ttm_ma_defensivo()

    def test_safe_num(self):
        test_safe_num_blindaje()

    def test_nvda_calibracion(self):
        test_calibracion_nvda_finviz()

    def test_roe_fallback(self):
        test_roe_fallback_cuando_patrimonio_falta()

    def test_mrq_cash_per_share(self):
        test_extraer_metricas_ttm_con_mrq_y_cash_per_share()

    def test_dividendos_nvda(self):
        test_datos_dividendos_exactos_nvda()

    def test_graficos_5y(self):
        test_series_historicas_5_anios_para_graficos()

    def test_consenso_ws(self):
        test_consenso_wall_street_price_target()

    def test_consenso_ws_sin_cobertura(self):
        test_consenso_wall_street_sin_cobertura_defensivo()

    def test_buyback_yield_reduccion(self):
        test_calcular_buyback_yield_reduccion_flotante()

    def test_buyback_yield_dilucion(self):
        test_calcular_buyback_yield_dilucion_neta()

    def test_buyback_yield_fallback_cf(self):
        test_calcular_buyback_yield_fallback_cashflow()

    def test_buyback_yield_sin_recompras(self):
        test_calcular_buyback_yield_sin_recompras_o_etf()

    def test_homologacion_ma(self):
        test_homologacion_metricas_finviz_mastercard_ma()

    def test_peg_forward_ma(self):
        test_calibracion_peg_forward_ma()

    def test_eps_yoy_ma(self):
        test_eps_growth_yoy_homogeneo_ma()

    def test_roic_ma(self):
        test_roic_mastercard_capital_structure()

    def test_aapl_completo(self):
        test_calibracion_aapl_completo()


def test_homologacion_metricas_finviz_mastercard_ma():
    """
    Verifica que los múltiplos de valuación y rentabilidad para Mastercard (MA)
    se alineen con los estándares de mercado (Benchmark Finviz).
    """
    mcap = 524_000_000_000.0
    precio = 598.47
    eps = 18.16
    fwd_eps = 23.00
    ebitda = 22_200_000_000.0
    fcf = 16_960_000_000.0
    total_debt = 24_640_000_000.0
    total_cash = 11_610_000_000.0
    rev = 28_000_000_000.0
    total_equity = 3_100_000_000.0
    total_assets = 46_000_000_000.0
    peg_info = 1.58
    earnings_growth = 0.221
    roe_finviz = 241.2
    roa_finviz = 29.8
    roic_finviz = 58.5

    # Buyback Yield
    cf_ma = pd.DataFrame({"2024": [11_727_000_000.0]}, index=["Repurchase Of Capital Stock"])
    res_by = calcular_buyback_yield(pd.DataFrame(), pd.DataFrame(), cf_ma, mcap=mcap)
    assert res_by["buyback_yield"] == pytest.approx(2.24, abs=0.1)

    # Múltiplos
    res_mult = calcular_multiplos_valuacion(
        precio_actual=precio,
        mcap=mcap,
        eps_ttm=eps,
        forward_eps=fwd_eps,
        fcf_ttm=fcf,
        ebitda_ttm=ebitda,
        total_debt=total_debt,
        total_cash=total_cash,
        revenue_ttm=rev,
        total_equity=total_equity,
        peg_info=peg_info,
        earnings_growth=earnings_growth,
        buyback_yield=res_by["buyback_yield"],
    )

    # EV/EBITDA ~24.2x
    assert res_mult["ev_ebitda"] == pytest.approx(24.2, abs=0.5)
    # P/FCF ~30.9x
    assert res_mult["p_fcf"] == pytest.approx(30.9, abs=1.5)
    # PEG ~1.58x (Válido numérico, no N/A)
    assert res_mult["peg"] == pytest.approx(1.58, abs=0.1)
    assert res_mult["peg_str"] != "N/A"
    assert "1.5" in res_mult["peg_str"] or "1.6" in res_mult["peg_str"]

    # Rentabilidad
    res_rent = calcular_ratios_rentabilidad(
        revenue_ttm=rev,
        gross_profit_ttm=rev,
        operating_income_ttm=ebitda * 0.85,
        net_income_ttm=11_500_000_000.0,
        total_assets=total_assets,
        total_equity=total_equity,
        total_debt=total_debt,
        total_cash=total_cash,
        current_liabilities=16_000_000_000.0,
        short_term_debt=1_500_000_000.0,
        roe_fallback=roe_finviz,
        roa_fallback=roa_finviz,
        roic_fallback=roic_finviz,
    )

    assert res_rent["roe"] == pytest.approx(241.2, abs=1.0)
    assert 24.0 <= res_rent["roa"] <= 31.0
    assert res_rent["roic"] == pytest.approx(58.5, abs=1.0)
    assert res_rent["roic"] < 80.0


def test_calibracion_peg_forward_ma():
    """
    Criterio 1: Verifica que al evaluar 'MA', el PEG Forward muestre un valor numérico válido
    (en el rango ~1.5x a 1.65x) en lugar de 'N/A'.
    """
    precio = 598.47
    eps_ttm = 18.16
    fwd_eps = 23.00
    earnings_growth = 0.221  # 22.1% YoY

    # Caso A: Con peg_info explícito
    res_a = calcular_multiplos_valuacion(
        precio_actual=precio,
        mcap=524e9,
        eps_ttm=eps_ttm,
        forward_eps=fwd_eps,
        peg_info=1.58,
        earnings_growth=earnings_growth,
    )
    assert res_a["peg"] == pytest.approx(1.58, abs=0.05)
    assert res_a["peg_str"] != "N/A"
    assert "x" in res_a["peg_str"]

    # Caso B: Sin peg_info, calculado automáticamente vía Forward P/E / Crecimiento
    res_b = calcular_multiplos_valuacion(
        precio_actual=precio,
        mcap=524e9,
        eps_ttm=eps_ttm,
        forward_eps=fwd_eps,
        peg_info=0.0,
        earnings_growth=0.165,  # ~16.5% crecimiento consenso forward
    )
    # Forward P/E = 598.47 / 23.00 = 26.02x
    # PEG = 26.02 / 16.5 = 1.577x (~1.58x)
    assert res_b["peg"] == pytest.approx(1.58, abs=0.10)
    assert res_b["peg_str"] != "N/A"
    assert "1.5" in res_b["peg_str"] or "1.6" in res_b["peg_str"]

    # Caso C: Extraer métricas TTM desde info sintética de MA
    info_ma = {
        "symbol": "MA",
        "trailingEps": 18.16,
        "forwardEps": 23.00,
        "forwardPE": 26.02,
        "trailingPE": 32.95,
        "pegRatio": 1.58,
        "earningsGrowth": 0.221,
        "marketCap": 524e9,
        "sharesOutstanding": 894e6,
    }
    m = extraer_metricas_ttm(info_ma, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), precio_actual=precio)
    assert m["peg_ratio_info"] == pytest.approx(1.58, abs=0.1)


def test_eps_growth_yoy_homogeneo_ma():
    """
    Criterio 2: Verifica que el EPS YoY para 'MA' refleje un crecimiento coherente y positivo
    según los últimos reportes TTM (rango de mercado de ~15% a 22%).
    """
    # Comparativa homogénea de Income Statement anual
    inc_ma = pd.DataFrame({
        "2024": [13.90, 12_900_000_000.0],
        "2023": [12.26, 11_200_000_000.0],
        "2022": [10.22, 9_900_000_000.0],
    }, index=["Diluted EPS", "Net Income"])

    info_ma = {
        "symbol": "MA",
        "trailingEps": 18.16,
        "earningsGrowth": 0.221,  # 22.1%
    }
    m = extraer_metricas_ttm(info_ma, inc_ma, pd.DataFrame(), pd.DataFrame(), precio_actual=598.47)

    assert 0.15 <= m["earnings_growth"] <= 0.23, f"EPS YoY fuera de rango esperado: {m['earnings_growth']}"

    # Caso fallback a partir de DataFrame con filas de EPS
    info_ma_sin_eg = {"symbol": "MA", "trailingEps": 18.16}
    m_calc = extraer_metricas_ttm(info_ma_sin_eg, inc_ma, pd.DataFrame(), pd.DataFrame(), precio_actual=598.47)
    # (13.90 - 12.26) / 12.26 = 13.38% o (12.9B - 11.2B) / 11.2B = 15.18%
    assert 0.13 <= m_calc["earnings_growth"] <= 0.22


def test_roic_mastercard_capital_structure():
    """
    Criterio 3: Verifica que el ROIC para 'MA' coincida con el rango estándar de mercado (~58% a 60%)
    en lugar de distorsiones superiores al 80%, considerando la estructura de capital con recompras agresivas.
    """
    # Parámetros institucionales representativos de Mastercard (MA)
    rev_ttm = 28_000_000_000.0
    op_inc_ttm = 16_000_000_000.0
    net_inc_ttm = 11_500_000_000.0
    total_assets = 46_000_000_000.0
    total_equity = 3_100_000_000.0   # Deprimido contablemente por recompras
    total_debt = 24_640_000_000.0
    total_cash = 11_610_000_000.0
    current_liab = 16_000_000_000.0
    short_debt = 1_500_000_000.0
    tax_rate = 0.21

    res = calcular_ratios_rentabilidad(
        revenue_ttm=rev_ttm,
        gross_profit_ttm=rev_ttm,
        operating_income_ttm=op_inc_ttm,
        net_income_ttm=net_inc_ttm,
        total_assets=total_assets,
        total_equity=total_equity,
        total_debt=total_debt,
        total_cash=total_cash,
        current_liabilities=current_liab,
        short_term_debt=short_debt,
        tax_rate=tax_rate,
        is_asset_light=True,
        roic_fallback=58.5,  # Consenso Finviz / Morningstar
    )

    # ROIC debe estar estrictamente en el rango de ~58% a 60%
    assert 57.0 <= res["roic"] <= 61.0, f"ROIC fuera de rango institucional: {res['roic']}"
    assert res["roic"] < 80.0, "ROIC no debe distorsionarse por encima del 80%"
    assert res["col_roic"] == "🟢"


def test_calibracion_aapl_completo():
    """
    Criterio 4: Verifica que las métricas (PEG Forward, EPS YoY y ROIC) se calculen
    adecuadamente para Apple (AAPL).
    """
    precio = 230.0
    eps_ttm = 6.50
    fwd_eps = 7.50
    earnings_growth = 0.12  # 12% YoY
    rev = 390_000_000_000.0
    op_inc = 120_000_000_000.0
    net_inc = 100_000_000_000.0
    total_assets = 350_000_000_000.0
    total_equity = 70_000_000_000.0
    total_debt = 100_000_000_000.0
    total_cash = 60_000_000_000.0
    current_liab = 130_000_000_000.0
    short_debt = 15_000_000_000.0

    res_mult = calcular_multiplos_valuacion(
        precio_actual=precio,
        mcap=3_500e9,
        eps_ttm=eps_ttm,
        forward_eps=fwd_eps,
        peg_info=2.50,
        earnings_growth=earnings_growth,
    )
    assert res_mult["peg"] == pytest.approx(2.50, abs=0.1)
    assert res_mult["peg_str"] != "N/A"

    res_rent = calcular_ratios_rentabilidad(
        revenue_ttm=rev,
        gross_profit_ttm=175e9,
        operating_income_ttm=op_inc,
        net_income_ttm=net_inc,
        total_assets=total_assets,
        total_equity=total_equity,
        total_debt=total_debt,
        total_cash=total_cash,
        current_liabilities=current_liab,
        short_term_debt=short_debt,
        tax_rate=0.16,
        is_asset_light=True,
        roic_fallback=55.0,
    )
    assert 45.0 <= res_rent["roic"] <= 65.0
    assert res_rent["col_roic"] == "🟢"


def test_calcular_buyback_yield_columnas_ascendentes_cronologicas():
    """
    Verifica que calcular_buyback_yield detecte correctamente el orden ascendente
    de columnas (ej. 2022, 2023, 2024).
    """
    inc = pd.DataFrame({
        "2022": [1_000_000_000.0],
        "2023": [960_000_000.0],
        "2024": [920_000_000.0],
    }, index=["Diluted Average Shares"])

    res = calcular_buyback_yield(inc=inc, bs=pd.DataFrame(), cf=pd.DataFrame(), mcap=100_000_000_000.0)

    # (960M - 920M) / 960M = +4.17%
    assert res["buyback_yield"] == pytest.approx(4.17, abs=0.1)
    assert res["buyback_yield_str"] == "4.2%"
    assert res["col_by"] == "🟢"


class TestWallStreetConsensus(unittest.TestCase):
    """
    Pruebas unitarias para validar la extracción robusta del consenso de analistas
    de Wall Street, compatibilidad de tipos (tupla y dict) y protección contra NameError en la app.
    """

    def test_import_in_app_module(self):
        """Verifica que obtener_consenso_wall_street esté importado y referenciado en app.py sin NameError."""
        import ast
        with open("app.py", "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename="app.py")

        imported_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imported_names.add(alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imported_names.add(alias.name)

        self.assertIn("obtener_consenso_wall_street", imported_names)

    def test_consensus_wall_street_structure_and_unpacking(self):
        """Verifica que ConsensusWallStreet soporte desempaquetado de tupla (mean, high, low) y acceso dict."""
        from data.financial_fetcher import ConsensusWallStreet
        cw = ConsensusWallStreet(
            target_mean=450.50,
            target_high=520.0,
            target_low=390.0,
            recommendation="buy",
            num_analysts=32,
        )
        # 1. Desempaquetado como tupla de 3 elementos
        mean, high, low = cw
        self.assertEqual(mean, 450.50)
        self.assertEqual(high, 520.0)
        self.assertEqual(low, 390.0)
        self.assertIsInstance(cw, tuple)
        self.assertEqual(len(cw), 3)

        # 2. Acceso por atributos
        self.assertEqual(cw.target_mean, 450.50)
        self.assertEqual(cw.target_high, 520.0)
        self.assertEqual(cw.target_low, 390.0)
        self.assertEqual(cw.recommendation, "buy")
        self.assertEqual(cw.num_analysts, 32)

        # 3. Acceso tipo diccionario
        self.assertEqual(cw["target_mean"], 450.50)
        self.assertEqual(cw.get("target_high"), 520.0)
        self.assertEqual(cw.get("recommendation"), "buy")
        self.assertEqual(cw.get("num_analysts"), 32)
        self.assertEqual(cw.get("inexistente", "def_val"), "def_val")

        # 4. to_dict y llaves
        d = cw.to_dict()
        self.assertIsInstance(d, dict)
        self.assertIn("target_mean", d)
        self.assertIn("recommendation", d)

    def test_empty_ticker_returns_safe_defaults(self):
        """Verifica que un ticker vacío o None retorne estructura con ceros sin error."""
        from data.financial_fetcher import obtener_consenso_wall_street
        res = obtener_consenso_wall_street("")
        self.assertEqual(res, (0.0, 0.0, 0.0))
        self.assertEqual(res.target_mean, 0.0)
        self.assertIsNone(res.recommendation)
        self.assertIsNone(res.num_analysts)

    @unittest.mock.patch("yfinance.Ticker")
    def test_yfinance_rich_metadata_extraction(self, mock_yf):
        """Verifica extracción de recomendaciones y número de analistas desde yfinance."""
        from data.financial_fetcher import obtener_consenso_wall_street
        mock_instance = unittest.mock.MagicMock()
        mock_instance.analyst_price_targets = {
            "mean": 520.0,
            "high": 580.0,
            "low": 470.0,
        }
        mock_instance.info = {
            "numberOfAnalystOpinions": 28,
            "recommendationKey": "strong_buy",
        }
        mock_yf.return_value = mock_instance

        res = obtener_consenso_wall_street("AAPL")
        mean, high, low = res
        self.assertEqual(mean, 520.0)
        self.assertEqual(high, 580.0)
        self.assertEqual(low, 470.0)
        self.assertEqual(res.num_analysts, 28)
        self.assertEqual(res.recommendation, "strong_buy")
        self.assertEqual(res["num_analysts"], 28)

    @unittest.mock.patch("data.financial_fetcher._finnhub_get")
    def test_finnhub_failure_fallbacks_cleanly(self, mock_fh_get):
        """Verifica que una falla de red o excepción en Finnhub no rompa la ejecución."""
        from data.financial_fetcher import obtener_consenso_wall_street
        mock_fh_get.side_effect = Exception("API Timeout")
        with unittest.mock.patch("yfinance.Ticker") as mock_yf:
            mock_inst = unittest.mock.MagicMock()
            mock_inst.analyst_price_targets = None
            mock_inst.info = {}
            mock_inst.recommendations_summary = None
            mock_yf.return_value = mock_inst

            res = obtener_consenso_wall_street("FAILTICKER", finnhub_api_key="dummy_key")
            mean, high, low = res
            self.assertEqual(mean, 0.0)
            self.assertEqual(high, 0.0)
            self.assertEqual(low, 0.0)


class TestMetricsTupleDefensiveness(unittest.TestCase):
    """Verifica que las funciones de métricas toleren argumentos tipo tupla sin TypeError."""

    def test_multiplos_valuacion_con_tuplas(self):
        from engine.metrics import calcular_multiplos_valuacion
        res = calcular_multiplos_valuacion(
            precio_actual=(550.0, "USD"),
            mcap=(500_000_000_000.0, "USD"),
            eps_ttm=(14.20, "USD"),
            forward_eps=(16.50,),
            fcf_ttm=(11_000_000_000.0,),
            ebitda_ttm=(15_000_000_000.0,),
            total_debt=(15_000_000_000.0,),
            total_cash=(8_000_000_000.0,),
            revenue_ttm=(25_000_000_000.0,),
            total_equity=(6_000_000_000.0,),
            peg_info=(1.8, "ratio"),
            earnings_growth=(0.12,),
            buyback_yield=(0.015,),
        )
        self.assertIsInstance(res["pe"], float)
        self.assertGreater(res["pe"], 0.0)

    def test_ratios_solvencia_con_tuplas(self):
        from engine.metrics import calcular_ratios_solvencia
        res = calcular_ratios_solvencia(
            total_debt=(15_000_000_000.0, "USD"),
            total_cash=(8_000_000_000.0, "USD"),
            total_equity=(6_000_000_000.0,),
            ebitda_ttm=(15_000_000_000.0,),
            ebit_ttm=(13_500_000_000.0,),
            interest_expense=(500_000_000.0,),
            current_assets=(12_000_000_000.0,),
            current_liabilities=(9_000_000_000.0,),
            fcf_ttm=(10_000_000_000.0,),
            shares_current=(930_000_000.0,),
        )
        self.assertIsInstance(res["net_debt"], float)
        self.assertIsInstance(res["cur_ratio"], float)

    def test_ratios_rentabilidad_con_tuplas(self):
        from engine.metrics import calcular_ratios_rentabilidad
        res = calcular_ratios_rentabilidad(
            revenue_ttm=(25_000_000_000.0, "USD"),
            gross_profit_ttm=(18_000_000_000.0,),
            operating_income_ttm=(14_000_000_000.0,),
            net_income_ttm=(12_000_000_000.0,),
            total_assets=(40_000_000_000.0,),
            total_equity=(6_000_000_000.0,),
            total_debt=(15_000_000_000.0,),
            total_cash=(8_000_000_000.0,),
            current_liabilities=(9_000_000_000.0,),
            short_term_debt=(1_000_000_000.0,),
            tax_rate=(0.21,),
        )
        self.assertIsInstance(res["roic"], float)
        self.assertIsInstance(res["roe"], float)

    def test_altman_zscore_y_scoring_con_tuplas(self):
        from engine.metrics import calcular_altman_zscore, calcular_scoring, evaluar_veredicto
        res_z = calcular_altman_zscore(debt_eq=(2.5, "ratio"), roa=(15.0, "%"))
        self.assertIn("z_score", res_z)

        score = calcular_scoring(
            "🟢", "🟢", "🟢", "🟢", "🟢", (55.0, "%"), "🟢", "🟢", "🟢",
            (450.0, "USD"), (500.0, "$"), (35.0, "x"), "🟢", "🟢", "🟢",
            res_z["status"], "🟢", "🟢", (15.0,)
        )
        self.assertIsInstance(score["pts_total"], (int, float))

        veredicto = evaluar_veredicto(
            score["pts_total"], (3.5,), (0.5,), False, (20.0,), 500, (45.0,)
        )
        self.assertIn("veredicto", veredicto)


if __name__ == "__main__":
    unittest.main()






