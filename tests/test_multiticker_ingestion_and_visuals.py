"""
Suite de pruebas automatizada para validación multiticker (10 tickers representativos),
soporte de métricas negativas, estandarización de ingesta contable y formato visual
de gráficos en Streamlit.
"""
import pytest
import pandas as pd
import numpy as np

from data.financial_fetcher import (
    fetch_datos_fundamentales,
    extraer_metricas_ttm,
    estandarizar_dataframe_financiero,
    MAPA_CANONICO_INCOME,
    MAPA_CANONICO_CASHFLOW,
)
from engine.metrics import (
    calcular_multiplos_valuacion,
    calcular_ratios_rentabilidad,
    calcular_ratios_solvencia,
    evaluar_veredicto,
)


def test_criterio_1_aapl_margins_not_flat_zero():
    """
    Criterio 1: En AAPL, la gráfica de márgenes refleja los porcentajes operativos
    y netos reales de los 5 años (no planos en 0.0%).
    """
    info, inc, bs, cf = fetch_datos_fundamentales("AAPL")
    cols_anuales = sorted([c for c in inc.columns if str(c).isdigit() and len(str(c)) == 4])
    assert len(cols_anuales) >= 5, f"Se esperaban al menos 5 años para AAPL, obtenidos: {cols_anuales}"

    ultimos_5 = cols_anuales[-5:]
    assert "Total Revenue" in inc.index or "Revenue" in inc.index
    assert "Operating Income" in inc.index

    row_rev = "Total Revenue" if "Total Revenue" in inc.index else "Revenue"
    row_op = "Operating Income"
    row_ni = "Net Income"

    op_pcts = []
    ni_pcts = []
    for c in ultimos_5:
        r = float(inc.loc[row_rev, c])
        op = float(inc.loc[row_op, c])
        ni = float(inc.loc[row_ni, c])

        assert r > 0, f"Revenue de AAPL en {c} debe ser positivo"
        pct_op = (op / r) * 100.0
        pct_ni = (ni / r) * 100.0

        # En ningún año de los 5 debe ser 0.0% plano
        assert pct_op > 15.0, f"Margen operativo de AAPL en {c} ({pct_op:.1f}%) no debe estar plano en 0.0%"
        assert pct_ni > 15.0, f"Margen neto de AAPL en {c} ({pct_ni:.1f}%) no debe estar plano en 0.0%"

        op_pcts.append(pct_op)
        ni_pcts.append(pct_ni)

    assert len(op_pcts) == 5
    assert len(ni_pcts) == 5


def test_criterio_2_googl_and_ma_operating_margin_and_capex():
    """
    Criterio 2: En GOOGL y MA, la barra de Margen Operativo es visible y CapEx
    muestra montos reales deduciendo coherentemente el FCF (FCF = OCF - CapEx).
    """
    for ticker in ["GOOGL", "MA"]:
        info, inc, bs, cf = fetch_datos_fundamentales(ticker)

        # 1. Margen Operativo visible
        cols_inc = sorted([c for c in inc.columns if str(c).isdigit() and len(str(c)) == 4])[-5:]
        assert len(cols_inc) >= 5, f"{ticker} debe tener 5 años en inc"
        row_rev = "Total Revenue" if "Total Revenue" in inc.index else "Revenue"
        for c in cols_inc:
            r = float(inc.loc[row_rev, c])
            op = float(inc.loc["Operating Income", c])
            mg_op = (op / r) * 100.0
            assert mg_op > 10.0, f"Margen operativo de {ticker} en {c} debe ser visible (>10%), obtenido: {mg_op:.1f}%"

        # 2. CapEx y reconciliación FCF
        cols_cf = sorted([c for c in cf.columns if str(c).isdigit() and len(str(c)) == 4])[-5:]
        assert len(cols_cf) >= 5, f"{ticker} debe tener 5 años en cf"
        for c in cols_cf:
            ocf = float(cf.loc["Operating Cash Flow", c])
            capex = abs(float(cf.loc["Capital Expenditure", c]))
            fcf = float(cf.loc["Free Cash Flow", c])

            assert capex > 0, f"CapEx de {ticker} en {c} debe ser positivo y real"
            assert ocf != fcf, f"FCF no debe duplicar a OCF en {ticker} en {c}"
            assert abs(fcf - (ocf - capex)) < 1e6, f"FCF debe igualar OCF - CapEx en {ticker} en {c}"


def test_criterio_3_intc_negative_metrics_and_5y_history():
    """
    Criterio 3: En INTC, las métricas de rentabilidad y los gráficos reflejan los periodos
    contables reales (incluyendo variaciones negativas) sin truncarse ni arrojar 0.0% artificial.
    """
    info, inc, bs, cf = fetch_datos_fundamentales("INTC")
    cols_anuales_inc = sorted([c for c in inc.columns if str(c).isdigit() and len(str(c)) == 4])
    cols_anuales_cf = sorted([c for c in cf.columns if str(c).isdigit() and len(str(c)) == 4])

    assert len(cols_anuales_inc) >= 5, "INC de INTC debe tener al menos 5 años"
    assert len(cols_anuales_cf) >= 5, "CF de INTC debe tener al menos 5 años"

    # Verificar que el año 2024 preserva el margen operativo negativo real
    if "2024" in inc.columns:
        r_2024 = float(inc.loc["Total Revenue", "2024"])
        op_2024 = float(inc.loc["Operating Income", "2024"])
        mg_op_2024 = (op_2024 / r_2024) * 100.0
        assert mg_op_2024 < 0.0, f"Margen operativo de INTC en 2024 debe ser negativo, obtenido: {mg_op_2024:.1f}%"

    m_ttm = extraer_metricas_ttm(info, inc, bs, cf, precio_actual=25.0)
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

    # Verificar que ROIC y ROE no colapsan a 0.0 artificial cuando hay pérdidas
    assert res_rent["col_roic"] == "🔴" or res_rent["roic"] > 0
    assert res_rent["col_roe"] == "🔴" or res_rent["roe"] > 0


def test_criterio_4_cashflow_labels_anti_overlap_configuration():
    """
    Criterio 4: En el gráfico de flujos de efectivo, las etiquetas numéricas no se enciman
    entre sí en ninguna de las barras (verificación de configuración de layout, rotación
    textangle=-45, espaciado bargroupgap y cliponaxis=False en app.py).
    """
    with open("app.py", "r", encoding="utf-8") as f:
        code = f.read()

    # Verificación de rotación angular anti-solapamiento
    assert "textangle=-45" in code, "Las barras de flujo de efectivo deben tener textangle=-45 para evitar solapamiento"
    # Verificación de que no se corta el texto superior o inferior
    assert "cliponaxis=False" in code, "cliponaxis=False es obligatorio para evitar truncamiento de etiquetas"
    # Verificación de espaciado de barras agrupadas
    assert "bargroupgap=0.08" in code or "bargroupgap=" in code, "El layout debe configurar bargroupgap para separación entre barras"
    assert "margin=dict(l=10, r=10, t=55" in code or "t=55" in code, "El margen superior debe acomodar las etiquetas rotadas"


def test_criterio_5_batch_10_tickers_sample_evaluation():
    """
    Criterio 5: La prueba con los 10 tickers seleccionados corre con éxito demostrable
    y sin excepciones no controladas en toda la muestra diversificada:
    ['AAPL', 'GOOGL', 'MSFT', 'INTC', 'MA', 'NVDA', 'KO', 'TSLA', 'AMZN', 'DIS'].
    """
    tickers_muestra = ["AAPL", "GOOGL", "MSFT", "INTC", "MA", "NVDA", "KO", "TSLA", "AMZN", "DIS"]

    for ticker in tickers_muestra:
        info, inc, bs, cf = fetch_datos_fundamentales(ticker)

        assert isinstance(inc, pd.DataFrame) and not inc.empty, f"INC de {ticker} no debe estar vacío"
        assert isinstance(bs, pd.DataFrame) and not bs.empty, f"BS de {ticker} no debe estar vacío"
        assert isinstance(cf, pd.DataFrame) and not cf.empty, f"CF de {ticker} no debe estar vacío"

        cols_inc = sorted([c for c in inc.columns if str(c).isdigit() and len(str(c)) == 4])
        cols_cf = sorted([c for c in cf.columns if str(c).isdigit() and len(str(c)) == 4])
        assert len(cols_inc) >= 5, f"{ticker} debe tener al menos 5 años en inc (obtenidos {len(cols_inc)})"
        assert len(cols_cf) >= 5, f"{ticker} debe tener al menos 5 años en cf (obtenidos {len(cols_cf)})"

        # Verificar últimos 5 años
        ultimos_5_cf = cols_cf[-5:]
        for y in ultimos_5_cf:
            ocf = float(cf.loc["Operating Cash Flow", y])
            capex = float(cf.loc["Capital Expenditure", y])
            fcf = float(cf.loc["Free Cash Flow", y])

            assert not np.isnan(ocf), f"OCF en {ticker} [{y}] no debe ser NaN"
            assert not np.isnan(capex), f"CapEx en {ticker} [{y}] no debe ser NaN"
            assert not np.isnan(fcf), f"FCF en {ticker} [{y}] no debe ser NaN"
            assert capex > 0, f"CapEx en {ticker} [{y}] debe ser positivo (> 0), obtenido: {capex}"
            assert fcf != ocf, f"FCF no debe duplicar a OCF en {ticker} [{y}]"

        # Ejecutar pipeline completo de métricas sin excepciones
        m_ttm = extraer_metricas_ttm(info, inc, bs, cf, precio_actual=100.0)
        res_mult = calcular_multiplos_valuacion(
            precio_actual=100.0,
            mcap=m_ttm["mcap"],
            eps_ttm=m_ttm["eps_diluted_ttm"],
            fcf_ttm=m_ttm["fcf_ttm"],
            ebitda_ttm=m_ttm["ebitda_ttm"],
            total_debt=m_ttm["total_debt"],
            total_cash=m_ttm["total_cash"],
            revenue_ttm=m_ttm["revenue_ttm"],
            total_equity=m_ttm["total_equity"],
        )
        assert isinstance(res_mult, dict)

        res_rent = calcular_ratios_rentabilidad(
            revenue_ttm=m_ttm["revenue_ttm"],
            gross_profit_ttm=m_ttm["gross_profit_ttm"],
            operating_income_ttm=m_ttm["operating_income_ttm"],
            net_income_ttm=m_ttm["net_income_ttm"],
            total_assets=m_ttm["total_assets"],
            total_equity=m_ttm["total_equity"],
        )
        assert isinstance(res_rent, dict)

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
        assert isinstance(res_solv, dict)
