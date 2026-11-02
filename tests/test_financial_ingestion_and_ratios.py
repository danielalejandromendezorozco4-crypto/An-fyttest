"""
Pruebas integrales de ingesta financiera, ratios fundamentales (referencia Finviz),
gráfico de histórico de precios a 24 meses y blindaje de variables de múltiplos (p_b).
"""
import pytest
import pandas as pd
from data.financial_fetcher import (
    fetch_datos_fundamentales,
    fetch_cotizacion_intradia,
    extraer_metricas_ttm,
    extraer_fcff_desapalancado,
)
from engine.metrics import (
    calcular_multiplos_valuacion,
    calcular_ratios_rentabilidad,
    calcular_ratios_solvencia,
)
from engine.valuation import calcular_fcff_valuation


def test_fundamental_ratios_finviz_reference_ma():
    """
    Criterio 1: Para el ticker 'MA', los múltiplos y márgenes principales son consistentes
    con los fundamentales del mercado de referencia (Finviz):
    - PER ~30-35
    - ROE > 100%
    - Márgenes operativos ~55-60%
    - Múltiplos P/FCF, EV/EBITDA, P/B, PEG disponibles.
    """
    precio, prev_c, hist = fetch_cotizacion_intradia("MA")
    info, inc, bs, cf = fetch_datos_fundamentales("MA")
    m_ttm = extraer_metricas_ttm(info, inc, bs, cf, precio)

    assert precio > 0, "El precio de MA debe ser positivo"
    assert m_ttm["shares_diluted"] > 100_000_000, "Las acciones diluidas deben reflejar la escala corporativa (~866M)"

    # Múltiplos
    res_mult = calcular_multiplos_valuacion(
        precio_actual=precio,
        mcap=m_ttm["mcap"],
        eps_ttm=m_ttm["eps_diluted_ttm"],
        forward_eps=m_ttm["forward_eps"],
        fcf_ttm=m_ttm["fcf_ttm"],
        ebitda_ttm=m_ttm["ebitda_ttm"],
        total_debt=m_ttm["total_debt"],
        total_cash=m_ttm["total_cash"],
        revenue_ttm=m_ttm["revenue_ttm"],
        total_equity=m_ttm["total_equity"],
        peg_info=m_ttm["peg_ratio_info"],
        earnings_growth=m_ttm["earnings_growth"],
        buyback_yield=1.5,
    )

    # PER ~30-35
    assert 25.0 <= res_mult["pe"] <= 40.0, f"PER ({res_mult['pe']}) debe estar entre 25 y 40"
    # P/B y Forward P/E definidos
    assert res_mult["p_b"] > 0, "P/B debe ser mayor a 0"
    assert res_mult["pe_forward"] > 0, "Forward P/E debe ser mayor a 0"
    assert res_mult["ev_ebitda"] > 0, "EV/EBITDA debe ser mayor a 0"
    assert res_mult["p_fcf"] > 0, "P/FCF debe ser mayor a 0"
    assert res_mult["peg"] > 0, "PEG debe ser mayor a 0"

    # Rentabilidad
    res_rent = calcular_ratios_rentabilidad(
        revenue_ttm=m_ttm["revenue_ttm"],
        gross_profit_ttm=m_ttm["gross_profit_ttm"],
        operating_income_ttm=m_ttm["operating_income_ttm"],
        net_income_ttm=m_ttm["net_income_ttm"],
        total_assets=m_ttm["total_assets"],
        total_equity=m_ttm["total_equity"],
        total_debt=m_ttm["total_debt"],
        total_cash=m_ttm["total_cash"],
        roe_fallback=m_ttm.get("roe", 0.0),
        roa_fallback=m_ttm.get("roa", 0.0),
        roic_fallback=m_ttm.get("roic", 0.0),
    )

    assert res_rent["roe"] > 100.0, f"ROE ({res_rent['roe']}%) debe ser superior al 100% para MA"
    assert 50.0 <= res_rent["mg_op"] <= 65.0, f"Margen operativo ({res_rent['mg_op']}%) debe rondar 55-60%"

    # Solvencia
    res_solv = calcular_ratios_solvencia(
        total_debt=m_ttm["total_debt"],
        total_cash=m_ttm["total_cash"],
        total_equity=m_ttm["total_equity"],
        ebitda_ttm=m_ttm["ebitda_ttm"],
        ebit_ttm=m_ttm["operating_income_ttm"],
        interest_expense=m_ttm["interest_expense_ttm"],
        current_assets=m_ttm["current_assets"],
        current_liabilities=m_ttm["current_liabilities"],
    )
    assert res_solv["cur_ratio"] > 0.8, "Razón corriente debe ser razonable (~1.2x)"
    assert res_solv["val_nde"] != "N/A", "Deuda Neta / EBITDA debe ser calculable"


def test_dcf_valuation_uses_real_shares():
    """
    Criterio 2: La valuación intrínseca DCF utiliza las acciones reales en circulación
    y arroja un valor por acción dentro de órdenes de magnitud razonables (no millones de USD).
    """
    precio, _, _ = fetch_cotizacion_intradia("MA")
    info, inc, bs, cf = fetch_datos_fundamentales("MA")
    comp_fcff = extraer_fcff_desapalancado(cf, inc, bs, info)
    m_ttm = extraer_metricas_ttm(info, inc, bs, cf, precio)

    # Verificar shares en comp_fcff
    assert comp_fcff["shares_diluted"] >= 800_000_000, "comp_fcff debe usar ~866M acciones para MA"

    res_dcf = calcular_fcff_valuation(
        ocf_hist=comp_fcff["ocf_hist"],
        capex_hist=comp_fcff["capex_hist"],
        interest_hist=comp_fcff["interest_hist"],
        pretax_hist=comp_fcff["pretax_hist"],
        taxprov_hist=comp_fcff["taxprov_hist"],
        total_debt=comp_fcff["total_debt"],
        total_cash=comp_fcff["total_cash"],
        shares_diluted=comp_fcff["shares_diluted"],
        precio_actual=precio,
        rf=4.20,
        beta=1.05,
        mcap=m_ttm["mcap"],
        cagr_revenue_hist=0.12,
        operating_margin_hist=0.57,
        ebit_hist=comp_fcff.get("ebit_hist", []),
        da_hist=comp_fcff.get("da_hist", []),
        delta_nwc_hist=comp_fcff.get("delta_nwc_hist", []),
        ticker="MA",
    )

    v_intr = res_dcf["valor_intrinseco"]
    # El valor por acción debe estar en un rango razonable ($50 a $1500 USD por acción, nunca $19M)
    assert 50.0 <= v_intr <= 1500.0, f"Valor intrínseco DCF (${v_intr}) fuera del orden de magnitud real"


def test_historical_price_series_24_months():
    """
    Criterio 3: La gráfica de tendencia mensual muestra la curva continua de 24 meses.
    """
    precio, prev_c, hist = fetch_cotizacion_intradia("MA")
    assert not hist.empty, "Historial de precios no debe estar vacío"
    assert len(hist) >= 24, f"Se esperaban al menos 24 registros en hist, se obtuvieron {len(hist)}"

    df_m = hist["Close"].resample("ME").last().dropna().reset_index()
    assert len(df_m) >= 24, f"La serie mensual de 24 meses debe tener al menos 24 puntos, se obtuvieron {len(df_m)}"


def test_no_name_error_p_b():
    """
    Criterio 4: No ocurren excepciones 'name p_b is not defined' y los múltiplos se desglosan correctamente.
    """
    precio, _, _ = fetch_cotizacion_intradia("MA")
    info, inc, bs, cf = fetch_datos_fundamentales("MA")
    m_ttm = extraer_metricas_ttm(info, inc, bs, cf, precio)

    res_mult = calcular_multiplos_valuacion(
        precio_actual=precio,
        mcap=m_ttm["mcap"],
        eps_ttm=m_ttm["eps_diluted_ttm"],
        forward_eps=m_ttm["forward_eps"],
        fcf_ttm=m_ttm["fcf_ttm"],
        ebitda_ttm=m_ttm["ebitda_ttm"],
        total_debt=m_ttm["total_debt"],
        total_cash=m_ttm["total_cash"],
        revenue_ttm=m_ttm["revenue_ttm"],
        total_equity=m_ttm["total_equity"],
        peg_info=m_ttm["peg_ratio_info"],
        earnings_growth=m_ttm["earnings_growth"],
        buyback_yield=1.5,
    )

    p_b = res_mult.get("p_b", 0.0)
    p_b_str = res_mult.get("p_b_str", "N/A")
    assert p_b is not None
    assert p_b > 0
    assert p_b_str != "N/A"


def test_financial_statements_have_historical_series():
    """
    Criterio 5: Los estados financieros de MA disponen de 5 años históricos,
    permitiendo que los gráficos de márgenes y flujos se rendericen sin mostrar 'no disponible'.
    """
    info, inc, bs, cf = fetch_datos_fundamentales("MA")
    assert not inc.empty, "Income statement no debe estar vacío para MA"
    assert not cf.empty, "Cash flow statement no debe estar vacío para MA"
    assert not bs.empty, "Balance sheet no debe estar vacío para MA"

    # Verificar filas canónicas necesarias para el gráfico de márgenes
    assert any(k in inc.index for k in ["Total Revenue", "Revenue", "Operating Revenue"])
    assert any(k in inc.index for k in ["Operating Income", "Operating Profit", "EBIT"])
    assert any(k in inc.index for k in ["Net Income", "Net Income Common Stockholders"])

    # Verificar filas canónicas para gráfico de flujos
    assert any(k in cf.index for k in ["Operating Cash Flow", "Total Cash From Operating Activities"])
    assert any(k in cf.index for k in ["Free Cash Flow", "Capital Expenditure", "CapitalExpenditures"])

    # Columnas anuales numéricas
    cols_anuales_inc = [c for c in inc.columns if str(c).isdigit()]
    assert len(cols_anuales_inc) >= 4, f"Se esperaban al menos 4 años en inc, se tienen {cols_anuales_inc}"
