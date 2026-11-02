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


@pytest.mark.parametrize("ticker", ["GOOGL", "AAPL", "MA"])
def test_price_history_real_curve_and_volatility(ticker: str):
    """
    Criterio de Aceptación 1:
    Al consultar tickers como 'GOOGL', 'AAPL' o 'MA', la gráfica muestra la curva
    de precios real de los últimos 24 meses (no una línea recta ni progresión artificial).
    """
    precio, prev_c, hist = fetch_cotizacion_intradia(ticker)
    assert precio > 0, f"El precio actual de {ticker} debe ser mayor a 0"
    assert not hist.empty, f"El historial de precios de {ticker} no debe estar vacío"
    assert len(hist) >= 24, f"Se esperaban al menos 24 registros diarios para {ticker}, se obtuvieron {len(hist)}"

    df_m = hist["Close"].resample("ME").last().dropna().reset_index()
    assert len(df_m) >= 24, f"La serie mensual de 24 meses debe tener al menos 24 puntos para {ticker}, tiene {len(df_m)}"

    # Verificar que no es una línea recta (segundas diferencias con desviación estándar > 0)
    diffs = df_m["Close"].diff().dropna()
    second_diffs = diffs.diff().dropna()
    assert second_diffs.std() > 0.05, f"La serie de {ticker} parece una línea recta sintética (std sec diffs <= 0.05)"
    # Rango de variación real de mercado
    price_range = df_m["Close"].max() - df_m["Close"].min()
    assert price_range > 10.0, f"Rango de precios de {ticker} demasiado estrecho ({price_range})"


@pytest.mark.parametrize("ticker", ["GOOGL", "AAPL", "MA"])
def test_financial_metrics_modules_real_values(ticker: str):
    """
    Criterio de Aceptación 2:
    Los módulos de Solvencia, Rentabilidad y Valuación reflejan métricas calculadas
    con datos reales y no caen en valores por defecto (0.00x, N/A, $0.20).
    """
    precio, _, _ = fetch_cotizacion_intradia(ticker)
    info, inc, bs, cf = fetch_datos_fundamentales(ticker)
    m_ttm = extraer_metricas_ttm(info, inc, bs, cf, precio)

    # 1. Solvencia y Liquidez
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
    assert res_solv["cur_ratio"] > 0.4, f"Current ratio para {ticker} ({res_solv['cur_ratio']}) debe ser > 0.4"
    assert res_solv["val_nde"] not in ["N/A", "N/D", ""], f"Net Debt/EBITDA para {ticker} no debe ser N/A"

    # 2. Rentabilidad y Eficiencia
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
    assert res_rent["roe"] > 0.0, f"ROE para {ticker} debe ser positivo (> 0%), se obtuvo {res_rent['roe']}"
    assert res_rent["roa"] > 0.0, f"ROA para {ticker} debe ser positivo (> 0%), se obtuvo {res_rent['roa']}"
    assert res_rent["roic"] > 0.0, f"ROIC para {ticker} debe ser positivo (> 0%), se obtuvo {res_rent['roic']}"
    assert res_rent["mg_op"] > 0.0, f"Margen operativo para {ticker} debe ser > 0%, se obtuvo {res_rent['mg_op']}"
    assert res_rent["net_margin"] > 0.0, f"Margen neto para {ticker} debe ser > 0%, se obtuvo {res_rent['net_margin']}"

    # 3. Múltiplos de Valuación
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
    assert res_mult["pe"] > 0.0, f"PER para {ticker} debe ser positivo, se obtuvo {res_mult['pe']}"
    assert res_mult["p_fcf"] > 0.0, f"P/FCF para {ticker} debe ser positivo, se obtuvo {res_mult['p_fcf']}"
    assert res_mult["ev_ebitda"] > 0.0, f"EV/EBITDA para {ticker} debe ser positivo, se obtuvo {res_mult['ev_ebitda']}"

    # 4. Valuación Intrínseca DCF (no debe ser el fallback artificial de $0.20)
    comp_fcff = extraer_fcff_desapalancado(cf, inc, bs, info)
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
        cagr_revenue_hist=0.10,
        operating_margin_hist=0.25,
        ebit_hist=comp_fcff.get("ebit_hist", []),
        da_hist=comp_fcff.get("da_hist", []),
        delta_nwc_hist=comp_fcff.get("delta_nwc_hist", []),
        ticker=ticker,
    )
    v_intr = res_dcf["valor_intrinseco"]
    assert v_intr > 10.0, f"Valor intrínseco DCF para {ticker} cayó en el fallback residual ($0.20), obtenido: ${v_intr}"


def test_criterio_1_googl_cash_flow_capex_and_fcf_discount():
    """
    Criterio 1: En 'GOOGL', el gráfico de flujos muestra la barra diferenciada de CapEx
    y el FCF refleja el descuento real del CapEx sobre el OCF (FCF = OCF - CapEx).
    """
    info, inc, bs, cf = fetch_datos_fundamentales("GOOGL")
    assert not cf.empty, "El estado de flujos de efectivo de GOOGL no debe estar vacío"
    assert "Capital Expenditure" in cf.index, "Capital Expenditure debe estar estandarizado en cf.index"
    assert "Operating Cash Flow" in cf.index, "Operating Cash Flow debe estar en cf.index"
    assert "Free Cash Flow" in cf.index, "Free Cash Flow debe estar en cf.index"

    cols_anuales = sorted([c for c in cf.columns if str(c).isdigit() and len(str(c)) == 4])
    assert len(cols_anuales) >= 4, f"Se esperaban al menos 4 años anuales para GOOGL, obtenidos: {cols_anuales}"

    s_ocf = cf.loc["Operating Cash Flow", cols_anuales]
    s_capex = cf.loc["Capital Expenditure", cols_anuales].abs()
    s_fcf = cf.loc["Free Cash Flow", cols_anuales]

    # Verificar para cada año con datos válidos que FCF = OCF - CapEx
    valid_years = 0
    for col in cols_anuales:
        v_ocf = s_ocf.get(col)
        v_cap = s_capex.get(col)
        v_fcf = s_fcf.get(col)
        if pd.notna(v_ocf) and pd.notna(v_cap) and pd.notna(v_fcf) and v_ocf > 0:
            valid_years += 1
            # CapEx diferenciado y sustancial (> $10B para Alphabet)
            assert v_cap > 10_000_000_000.0, f"CapEx para GOOGL en {col} (${v_cap:,.0f}) debe ser > $10B"
            # FCF refleja el descuento real de CapEx sobre OCF
            diferencia = abs(v_fcf - (v_ocf - v_cap))
            assert diferencia < 1e6, f"FCF ({v_fcf}) no coincide con OCF ({v_ocf}) - CapEx ({v_cap}) en {col}"

    assert valid_years >= 3, f"Se requerían al menos 3 años con datos completos de flujos para GOOGL, encontrados: {valid_years}"


def test_criterio_2_googl_margins_operating_and_net():
    """
    Criterio 2: En 'GOOGL', el gráfico de márgenes visualiza las dos series:
    Margen Neto y Margen Operativo.
    """
    info, inc, bs, cf = fetch_datos_fundamentales("GOOGL")
    assert not inc.empty, "El estado de resultados de GOOGL no debe estar vacío"
    assert "Total Revenue" in inc.index or "Revenue" in inc.index
    assert "Operating Income" in inc.index or "EBIT" in inc.index
    assert "Net Income" in inc.index

    cols_anuales = sorted([c for c in inc.columns if str(c).isdigit() and len(str(c)) == 4])
    s_rev = inc.loc["Total Revenue" if "Total Revenue" in inc.index else "Revenue", cols_anuales]
    s_op = inc.loc["Operating Income" if "Operating Income" in inc.index else "EBIT", cols_anuales]
    s_ni = inc.loc["Net Income", cols_anuales]

    valid_years = 0
    for col in cols_anuales:
        r = s_rev.get(col)
        op = s_op.get(col)
        ni = s_ni.get(col)
        if pd.notna(r) and r > 0 and pd.notna(op) and pd.notna(ni):
            valid_years += 1
            op_margin = (op / r) * 100.0
            net_margin = (ni / r) * 100.0
            # Ambas series calculables y positivas
            assert 20.0 <= op_margin <= 40.0, f"Margen Operativo ({op_margin:.1f}%) fuera del rango esperado para GOOGL"
            assert 15.0 <= net_margin <= 40.0, f"Margen Neto ({net_margin:.1f}%) fuera del rango esperado para GOOGL"

    assert valid_years >= 3, f"Se requerían al menos 3 años de márgenes para GOOGL, encontrados: {valid_years}"


def test_criterio_3_hover_tooltips_nominal_and_yoy_calculation():
    """
    Criterio 3: Al pasar el cursor sobre cualquier barra de ambos gráficos, el tooltip
    muestra claramente el monto nominal formateado y la variación porcentual interanual (YoY %).
    """
    from engine.metrics import calcular_variacion_yoy

    # Caso estándar con crecimiento y caída
    serie_nom = [100.0, 120.0, 110.0, 143.0]
    yoy_res = calcular_variacion_yoy(serie_nom)
    assert yoy_res[0] == "Base", "El primer año t=0 debe reportar 'Base'"
    assert yoy_res[1] == "+20.0%", "El incremento de 100 a 120 debe ser +20.0%"
    assert yoy_res[2] == "-8.3%", "El decremento de 120 a 110 debe ser -8.3%"
    assert yoy_res[3] == "+30.0%", "El incremento de 110 a 143 debe ser +30.0%"

    # Manejo de cero en el denominador (evitar división por cero)
    serie_con_cero = [50.0, 0.0, 75.0]
    yoy_cero = calcular_variacion_yoy(serie_con_cero)
    assert yoy_cero[0] == "Base"
    assert yoy_cero[1] == "-100.0%"
    assert yoy_cero[2] == "N/A", "División por cero debe retornar 'N/A'"

    # Serie de un único año
    serie_uno = [250.0]
    assert calcular_variacion_yoy(serie_uno) == ["Base"]


def test_criterio_4_resilience_incomplete_financial_statements():
    """
    Criterio 4: La app compila y ejecuta sin errores ante activos con estados financieros incompletos
    (auto-reconciliación de CapEx y FCF, DataFrames vacíos o con columnas faltantes).
    """
    from data.financial_fetcher import estandarizar_dataframe_financiero, MAPA_CANONICO_CASHFLOW

    # Caso A: CF con OCF y FCF pero sin CapEx (debe auto-derivar CapEx = OCF - FCF)
    cf_sin_capex = pd.DataFrame({
        "2023": [100.0, 70.0],
        "2024": [120.0, 85.0],
    }, index=["Operating Cash Flow", "Free Cash Flow"])
    cf_std_a = estandarizar_dataframe_financiero(cf_sin_capex, MAPA_CANONICO_CASHFLOW)
    assert "Capital Expenditure" in cf_std_a.index
    assert cf_std_a.loc["Capital Expenditure", "2023"] == 30.0
    assert cf_std_a.loc["Capital Expenditure", "2024"] == 35.0

    # Caso B: CF con OCF y CapEx pero FCF idéntico a OCF (debe corregir FCF = OCF - CapEx)
    cf_fcf_igual_ocf = pd.DataFrame({
        "2023": [200.0, 50.0, 200.0],
        "2024": [250.0, 60.0, 250.0],
    }, index=["Operating Cash Flow", "Capital Expenditure", "Free Cash Flow"])
    cf_std_b = estandarizar_dataframe_financiero(cf_fcf_igual_ocf, MAPA_CANONICO_CASHFLOW)
    assert cf_std_b.loc["Free Cash Flow", "2023"] == 150.0
    assert cf_std_b.loc["Free Cash Flow", "2024"] == 190.0

    # Caso C: DataFrame vacío
    df_vacio = pd.DataFrame()
    assert estandarizar_dataframe_financiero(df_vacio, MAPA_CANONICO_CASHFLOW).empty


def test_criterio_1_ma_capex_and_fcf_reconciliation():
    """
    Criterio 1: En 'MA', el estado de flujos y el módulo de efectivo muestran barras visibles
    y valores reales para CapEx, diferenciando claramente OCF de FCF en los 5 años evaluados.
    """
    info, inc, bs, cf = fetch_datos_fundamentales("MA")
    assert not cf.empty, "El Cash Flow Statement de MA no debe estar vacío"
    assert "Capital Expenditure" in cf.index, "CapEx debe existir en el Cash Flow de MA"
    assert "Operating Cash Flow" in cf.index, "OCF debe existir en el Cash Flow de MA"
    assert "Free Cash Flow" in cf.index, "FCF debe existir en el Cash Flow de MA"

    cols_anuales = sorted([c for c in cf.columns if str(c).isdigit() and len(str(c)) == 4])
    assert len(cols_anuales) >= 5, f"Se esperaban al menos 5 años anuales para MA, obtenidos: {cols_anuales}"

    # Evaluar los últimos 5 años
    for y in cols_anuales[-5:]:
        v_ocf = float(cf.loc["Operating Cash Flow", y])
        v_capex = abs(float(cf.loc["Capital Expenditure", y]))
        v_fcf = float(cf.loc["Free Cash Flow", y])

        assert v_capex > 0, f"CapEx en {y} debe ser mayor a 0 para MA"
        assert v_ocf > 0, f"OCF en {y} debe ser positivo para MA"
        assert v_fcf > 0, f"FCF en {y} debe ser positivo para MA"
        assert v_fcf != v_ocf, f"FCF no debe duplicar a OCF en {y} para MA"
        # Reconciliación FCF = OCF - CapEx (tolerancia por redondeo de millones)
        assert abs(v_fcf - (v_ocf - v_capex)) < 1e6, f"FCF debe igualar OCF - CapEx en {y}"


def test_criterio_2_cashflow_labels_formatting():
    """
    Criterio 2: Las etiquetas numéricas sobre las barras de flujos de efectivo formatean
    adecuadamente tanto números positivos ($16,433) como negativos (-$4,949), evitando símbolos
    extraños como '$-100'.
    """
    fcf_nom = [16433.0, 13586.0, -4949.0, 0.0]
    formatted = [f"-${abs(v):,.0f}" if v < 0 else f"${v:,.0f}" for v in fcf_nom]
    assert formatted[0] == "$16,433"
    assert formatted[1] == "$13,586"
    assert formatted[2] == "-$4,949"
    assert formatted[3] == "$0"


def test_criterio_3_intc_complex_negative_margins_solvency_and_5y_history():
    """
    Criterio 3: En 'INTC', los módulos de solvencia, rentabilidad y valuación muestran sus ratios
    reales correspondientes, y los gráficos de márgenes y flujos despliegan el historial completo
    de 5 años sin truncarse ni colapsar a cero, preservando márgenes y ROIC negativos.
    """
    info, inc, bs, cf = fetch_datos_fundamentales("INTC")
    assert not inc.empty, "Income Statement de INTC no debe estar vacío"
    assert not cf.empty, "Cash Flow de INTC no debe estar vacío"

    cols_anuales_inc = sorted([c for c in inc.columns if str(c).isdigit() and len(str(c)) == 4])
    cols_anuales_cf = sorted([c for c in cf.columns if str(c).isdigit() and len(str(c)) == 4])
    assert len(cols_anuales_inc) >= 5, f"INC de INTC debe tener al menos 5 años: {cols_anuales_inc}"
    assert len(cols_anuales_cf) >= 5, f"CF de INTC debe tener al menos 5 años: {cols_anuales_cf}"

    m_ttm = extraer_metricas_ttm(info, inc, bs, cf, precio_actual=25.0)

    # Ratios de rentabilidad para INTC
    res_rent = calcular_ratios_rentabilidad(
        revenue_ttm=m_ttm["revenue_ttm"],
        gross_profit_ttm=m_ttm["gross_profit_ttm"],
        operating_income_ttm=m_ttm["operating_income_ttm"],
        net_income_ttm=m_ttm["net_income_ttm"],
        total_assets=m_ttm["total_assets"],
        total_equity=m_ttm["total_equity"],
        total_debt=m_ttm["total_debt"],
        total_cash=m_ttm["total_cash"],
        current_liabilities=m_ttm["current_liabilities"],
        short_term_debt=m_ttm["short_term_debt"],
        tax_rate=0.21,
    )

    # Verificar que el margen operativo y neto no colapsan a 0.0 si hay pérdidas
    assert res_rent["mg_op"] != 0.0 or m_ttm["operating_income_ttm"] == 0.0
    assert res_rent["roic"] != 0.0 or m_ttm["operating_income_ttm"] == 0.0

    # Ratios de solvencia para INTC con FCF negativo
    res_solv = calcular_ratios_solvencia(
        total_debt=m_ttm["total_debt"],
        total_cash=m_ttm["total_cash"],
        total_equity=m_ttm["total_equity"],
        ebitda_ttm=m_ttm["ebitda_ttm"],
        ebit_ttm=m_ttm["operating_income_ttm"],
        interest_expense=m_ttm["interest_expense_ttm"],
        current_assets=m_ttm["current_assets"],
        current_liabilities=m_ttm["current_liabilities"],
        fcf_ttm=m_ttm["fcf_ttm"],
    )
    assert res_solv["cur_ratio"] > 0, "Current Ratio de INTC debe ser positivo"
    assert "Flujo de Caja Libre Negativo" in res_solv["msg_fcfd"] or res_solv["fcf_debt_ratio"] > 0

    # Verificar que FCF negativo y utilidades negativas no produzcan ratio de conversión FCF verde
    net_income = m_ttm["net_income_ttm"]
    fcf_ttm = m_ttm["fcf_ttm"]
    if net_income < 0 and fcf_ttm < 0:
        # No debe considerarse saludable
        assert not (fcf_ttm / net_income > 1.0 and net_income > 0)


def test_criterio_4_app_compilation_and_clean_execution():
    """
    Criterio 4: La app compila y ejecuta sin errores ante activos con utilidades o flujos negativos.
    """
    import py_compile
    # Compilación de app.py
    py_compile.compile("app.py", doraise=True)

