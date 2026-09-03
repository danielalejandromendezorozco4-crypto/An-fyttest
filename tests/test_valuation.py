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
    assert len(res["fcff_proyectado"]) == res["n_total"]
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
    assert res["valor_intrinseco"] > 300.0
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
    assert res["valor_intrinseco"] > 130.0
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


# ─────────────────────────────────────────────────────────────────────────────
# NUEVOS TESTS — Refactorización Metodológica v2
# ─────────────────────────────────────────────────────────────────────────────

def _base_params(**overrides):
    """Parámetros base compartidos para tests de valuación."""
    params = {
        "ocf_hist":      [80_000_000.0, 75_000_000.0, 70_000_000.0],
        "capex_hist":    [15_000_000.0, 14_000_000.0, 13_000_000.0],
        "interest_hist": [3_000_000.0,  2_800_000.0,  2_600_000.0],
        "pretax_hist":   [60_000_000.0, 55_000_000.0, 50_000_000.0],
        "taxprov_hist":  [12_600_000.0, 11_550_000.0, 10_500_000.0],
        "total_debt":    50_000_000.0,
        "total_cash":    20_000_000.0,
        "shares_diluted": 10_000_000.0,
        "mcap":          500_000_000.0,
        "beta":          1.1,
        "rf":            4.20,
        "precio_actual": 50.0,
    }
    params.update(overrides)
    return params


def test_fcff_dual_path_ebit_primario():
    """
    Cuando ebit_hist, da_hist y delta_nwc_hist son provistos con EBIT > 0,
    el path primario desapalancado debe producir un FCFF histórico válido
    mayor que cero y el método debe reportarse como 'ebit'.
    """
    ebit = 60_000_000.0
    da   = 12_000_000.0
    capex = 15_000_000.0
    dnwc  = 3_000_000.0
    tax   = 0.21
    # FCFF esperado vía EBIT: 60M*(1-0.21) + 12M - 15M - 3M = 47.4M + 12M - 18M = 41.4M
    fcff_esperado = ebit * (1.0 - tax) + da - capex - dnwc

    res = calcular_fcff_valuation(
        **_base_params(),
        ebit_hist      = [ebit],
        da_hist        = [da],
        delta_nwc_hist = [dnwc],
    )
    assert res["fcff_method"] == "ebit", f"Se esperaba path EBIT pero fue: {res['fcff_method']}"
    assert res["fcff_historico"][0] == pytest.approx(fcff_esperado, rel=0.01), \
        f"FCFF vía EBIT incorrecto: {res['fcff_historico'][0]:.0f} vs {fcff_esperado:.0f}"


def test_fcff_dual_path_ocf_fallback_cuando_ebit_negativo():
    """
    Cuando EBIT es negativo, el motor debe caer al path OCF.
    El método reportado debe ser 'ocf'.
    """
    ocf   = 80_000_000.0
    capex = 15_000_000.0
    interest = 3_000_000.0
    tax = 0.21

    res = calcular_fcff_valuation(
        **_base_params(),
        ebit_hist      = [-5_000_000.0],  # EBIT negativo → fallback
        da_hist        = [10_000_000.0],
        delta_nwc_hist = [2_000_000.0],
    )
    assert res["fcff_method"] == "ocf", f"Se esperaba fallback OCF pero fue: {res['fcff_method']}"
    fcff_ocf = ocf + interest * (1.0 - tax) - capex
    assert res["fcff_historico"][0] == pytest.approx(fcff_ocf, rel=0.02)


def test_wacc_clamping_invariante_minimo():
    """
    El WACC resultante debe ser siempre >= g_term + 1.5%.
    Se verifica con un caso donde beta muy bajo produciría WACC < g + 1.5%.
    """
    from config.settings import WACC_MIN_SPREAD_OVER_G
    res = calcular_fcff_valuation(
        **_base_params(beta=0.10, rf=2.0),  # beta muy bajo → WACC muy bajo
    )
    wacc_dec  = res["wacc"] / 100.0
    g_term    = res["g_term"]
    assert wacc_dec >= g_term + WACC_MIN_SPREAD_OVER_G - 1e-9, \
        f"Invariante violada: WACC={wacc_dec:.4f}, g_term={g_term:.4f}, spread={wacc_dec-g_term:.4f}"


def test_fade_period_tasas_decrecientes():
    """
    El fade period debe producir tasas de crecimiento estrictamente
    decrecientes desde g_1_5 hacia g_term en los años del fade.
    """
    res = calcular_fcff_valuation(
        **_base_params(),
        fade_years         = 3,
        g_term_override    = 0.025,
        revenue_growth_api = 0.12,  # g_1_5 = 12%
    )
    detalle = res["fcff_pv_detalle"]
    assert len(detalle) == 8, f"Esperado 5 + 3 = 8 años, obtenido: {len(detalle)}"
    # Verificar que los años del fade tienen g decreciente
    fade_rows = [r for r in detalle if r["Fase"] == "Fade"]
    assert len(fade_rows) == 3
    tasas_fade = [r["g Aplicada (%)"] for r in fade_rows]
    for i in range(len(tasas_fade) - 1):
        assert tasas_fade[i] >= tasas_fade[i + 1], \
            f"g no es decreciente en fade: año {i} = {tasas_fade[i]}, año {i+1} = {tasas_fade[i+1]}"


def test_mid_year_convention_pv_detalle():
    """
    Cada fila de fcff_pv_detalle debe cumplir:
        VP = FCFF / (1+WACC)^(t-0.5)
    La suma de los VP debe ser igual a pv_flujos reportado (tolerancia 1%).
    """
    import math
    res = calcular_fcff_valuation(**_base_params(), fade_years=2)
    detalle = res["fcff_pv_detalle"]
    wacc_dec = res["wacc"] / 100.0

    suma_pv_detalle = sum(r["VP (M USD)"] for r in detalle) * 1e6
    assert abs(suma_pv_detalle - res["pv_flujos"]) / res["pv_flujos"] < 0.01, \
        f"Discrepancia en suma VP: {suma_pv_detalle:.0f} vs pv_flujos={res['pv_flujos']:.0f}"

    for row in detalle:
        t = row["Ano"]
        factor_esperado = (1.0 + wacc_dec) ** (t - 0.5)
        factor_real     = row["Factor Desc (t-0.5)"]
        assert abs(factor_real - factor_esperado) < 0.0001, \
            f"Factor de descuento incorrecto en año {t}: {factor_real} vs {factor_esperado}"


def test_buyback_reduce_acciones_efectivas():
    """
    Con buyback_rate = 2% anual y n_total = 8 años,
    shares_efectivas = shares * (1 - 0.02)^8 < shares_diluted.
    El valor intrínseco debe ser mayor que sin buyback.
    """
    params = _base_params()

    res_sin = calcular_fcff_valuation(**params, buyback_rate=0.0, fade_years=3)
    res_con = calcular_fcff_valuation(**params, buyback_rate=0.02, fade_years=3)

    n_total = res_con["n_total"]
    shares_orig = params["shares_diluted"]
    shares_ef_esperado = shares_orig * (0.98 ** n_total)

    assert res_con["shares_efectivas"] == pytest.approx(shares_ef_esperado, rel=0.01)
    assert res_con["valor_intrinseco"] > res_sin["valor_intrinseco"], \
        "Buyback positivo debe aumentar el valor intrínseco por acción"


def test_wacc_igual_a_g_edge_case():
    """
    Cuando el WACC calculado se acercaría a g_term, el motor debe forzar
    el spread mínimo evitando NaN o ZeroDivision en el Valor Terminal.
    """
    # Beta muy bajo + rf muy bajo → WACC podría ser ~2.5% ≈ g_term
    res = calcular_fcff_valuation(
        **_base_params(beta=0.05, rf=1.5, precio_actual=50.0),
        g_term_override = 0.025,
        fade_years = 2,
    )
    assert res["terminal_value"] > 0, "Terminal Value no debe ser cero o negativo"
    assert not (res["valor_intrinseco"] != res["valor_intrinseco"]), "NaN detectado en valor intrínseco"
    assert not (res["terminal_value"] != res["terminal_value"]), "NaN detectado en terminal_value"
    wacc_dec = res["wacc"] / 100.0
    g_term   = res["g_term"]
    spread   = wacc_dec - g_term
    assert spread > 0.010, f"Spread WACC-g demasiado pequeño: {spread:.4f}"


def test_msft_megacap_sin_nan_ni_error():
    """
    Simula condiciones típicas de un megacap tipo MSFT.
    Verifica que el motor no produce NaN, ZeroDivision ni valores negativos
    en enterprise_value, pv_terminal, valor_intrinseco.
    """
    res = calcular_fcff_valuation(
        ocf_hist         = [87e9, 79e9, 70e9, 60e9],
        capex_hist       = [20e9, 18e9, 16e9, 14e9],
        interest_hist    = [2.5e9, 2.2e9, 2.0e9, 1.8e9],
        pretax_hist      = [100e9, 90e9, 80e9, 70e9],
        taxprov_hist     = [17e9,  15e9, 14e9, 12e9],
        total_debt       = 80e9,
        total_cash       = 60e9,
        shares_diluted   = 7.4e9,
        mcap             = 3_200e9,
        beta             = 0.90,
        rf               = 4.30,
        precio_actual    = 450.0,
        erp              = 4.5,
        revenue_growth_api = 0.13,
        ebit_hist        = [70e9, 65e9, 58e9, 50e9],
        da_hist          = [14e9, 13e9, 12e9, 11e9],
        delta_nwc_hist   = [3e9,  2e9,  2e9,  1e9],
        buyback_rate     = 0.015,
        fade_years       = 3,
        g_term_override  = 0.028,
    )

    for campo in ["valor_intrinseco", "enterprise_value", "pv_flujos", "pv_terminal", "terminal_value"]:
        val = res[campo]
        assert val == val, f"NaN detectado en {campo}"  # NaN != NaN
        assert val >= 0, f"Valor negativo en {campo}: {val}"

    assert res["fcff_pv_detalle"], "fcff_pv_detalle no debe estar vacío para MSFT"
    assert len(res["fcff_pv_detalle"]) == 8, "Esperado 5 años + 3 fade = 8 filas"


def test_fcff_valuation_nvda_calibracion():
    """
    Criterio 2: Verifica que para NVDA con FCF TTM (~$55B) y tasa de crecimiento
    prospectiva del ~38.5% con Fade Period de 4 años, el modelo DCF/FCFF arroje
    un Valor Intrínseco coherente en el rango de ~$210 - $218 por acción.
    """
    res = calcular_fcff_valuation(
        ocf_hist         = [60e9, 28e9, 5.6e9, 9.1e9],
        capex_hist       = [5e9, 1.1e9, 1.8e9, 1.0e9],  # FCF TTM = 55e9
        interest_hist    = [300e6, 250e6, 200e6],
        pretax_hist      = [61e9, 32e9, 5e9],
        taxprov_hist     = [8.5e9, 4.5e9, 600e6],       # Tax rate ~14%
        total_debt       = 11.1e9,
        total_cash       = 34.8e9,                      # Caja Neta Positiva
        shares_diluted   = 24.5e9,
        mcap             = 3_050e9,
        beta             = 1.15,                        # Beta fundamental ajustada
        rf               = 4.25,
        precio_actual    = 125.0,
        erp              = 4.5,
        growth_rate_exp  = 0.39,                        # Crecimiento prospectivo Fase 1 (39%)
        revenue_growth_api = 0.89,
        revenue_ttm      = 96.3e9,
        operating_margin_hist = 0.62,
        fade_years       = 4,
        g_term_override  = 0.025,
    )

    v_intr = res["valor_intrinseco"]
    assert 205.0 <= v_intr <= 225.0, f"Valor Intrínseco NVDA fuera de rango esperado: {v_intr}"
    assert res["fcff_base"] >= 50e9
    assert res["wacc"] > 0
    assert res["enterprise_value"] > 4_500e9


# ─────────────────────────────────────────────────────────────────────────────
# 6. PRUEBAS DE CALIBRACIÓN DE VALUACIÓN DCF/WACC Y CONSENSO WALL STREET
# ─────────────────────────────────────────────────────────────────────────────

def test_fcff_valuation_mastercard_mercado_congruente():
    """
    Criterio 1: Verifica que para Mastercard (MA) con parámetros de mercado estándar
    (g=2.5%, WACC ~7.5%-9.5%, OCF ~$13.5B, CapEx ~$0.6B, EBIT ~$16B, Debt ~$16B, Cash ~$8.5B),
    el modelo FCFF genere un Valor Intrínseco congruente con el mercado (~$300-$750)
    y no valores inflados por recompras históricas.
    """
    res = calcular_fcff_valuation(
        ocf_hist=[13.5e9, 11.5e9, 10.0e9],
        capex_hist=[0.6e9, 0.5e9, 0.45e9],
        interest_hist=[0.55e9, 0.45e9, 0.4e9],
        pretax_hist=[15.0e9, 13.0e9, 11.5e9],
        taxprov_hist=[2.8e9, 2.3e9, 2.0e9],
        total_debt=16.0e9,
        total_cash=8.5e9,
        shares_diluted=894e6,
        mcap=520e9,
        beta=1.05,
        rf=4.25,
        precio_actual=598.47,
        revenue_ttm=27.0e9,
        revenue_growth_api=0.12,
        operating_margin_hist=0.58,
        ebit_hist=[16.0e9, 14.0e9, 12.5e9],
        da_hist=[0.9e9, 0.8e9, 0.7e9],
        delta_nwc_hist=[0.2e9, 0.15e9, 0.1e9],
        buyback_rate=0.0,
        fade_years=3,
        g_term_override=0.025,
    )
    v_intr = res["valor_intrinseco"]
    wacc = res["wacc"]
    assert 300.0 <= v_intr <= 750.0, f"Valor intrínseco de MA fuera de rango: {v_intr}"
    assert 7.0 <= wacc <= 10.0, f"WACC de MA fuera de rango: {wacc}"
    assert res["enterprise_value"] > res["equity_value"], "EV debe ser mayor a Equity cuando Deuda Neta > 0"
    assert abs(res["deuda_neta"] - 7.5e9) < 1e6


def test_fcff_valuation_apple_mercado_congruente():
    """
    Criterio 1: Verifica que para Apple (AAPL) con WACC ~9%-10.5%, g=2.5%, OCF ~$118B, CapEx ~$10B,
    el Valor Intrínseco se ubique en rango congruente de mercado (~$120-$280).
    """
    res = calcular_fcff_valuation(
        ocf_hist=[118e9, 110e9, 122e9],
        capex_hist=[10.5e9, 11.0e9, 10.0e9],
        interest_hist=[3.6e9, 3.8e9, 2.9e9],
        pretax_hist=[125e9, 115e9, 119e9],
        taxprov_hist=[19e9, 17e9, 18e9],
        total_debt=105e9,
        total_cash=65e9,
        shares_diluted=15.2e9,
        mcap=3400e9,
        beta=1.10,
        rf=4.20,
        precio_actual=230.0,
        revenue_ttm=395e9,
        revenue_growth_api=0.08,
        operating_margin_hist=0.31,
        ebit_hist=[123e9, 114e9, 119e9],
        da_hist=[11.5e9, 11.0e9, 11.5e9],
        delta_nwc_hist=[1.0e9, 0.5e9, 0.8e9],
        buyback_rate=0.0,
        fade_years=3,
        g_term_override=0.025,
    )
    v_intr = res["valor_intrinseco"]
    wacc = res["wacc"]
    assert 120.0 <= v_intr <= 280.0, f"Valor intrínseco de AAPL fuera de rango: {v_intr}"
    assert 8.5 <= wacc <= 10.5, f"WACC de AAPL fuera de rango: {wacc}"
    assert res["equity_value"] == round(res["enterprise_value"] - res["deuda_neta"], 2)


def test_ev_to_equity_bridge_deuda_neta_exacto():
    """
    Criterio 2: Verifica el puente financiero exacto:
    Enterprise Value = PV(Flujos) + PV(Terminal Value)
    Equity Value = Enterprise Value - (Total Debt - Total Cash)
    Valor Intrínseco por acción = Equity Value / Shares
    """
    res = calcular_fcff_valuation(
        ocf_hist=[100e6],
        capex_hist=[20e6],
        interest_hist=[10e6],
        pretax_hist=[80e6],
        taxprov_hist=[16e6],
        total_debt=50e6,
        total_cash=20e6,      # Deuda neta = 30e6
        shares_diluted=10e6,
        mcap=800e6,
        beta=1.0,
        rf=4.0,
        precio_actual=80.0,
        g_term_override=0.025,
    )
    ev = res["enterprise_value"]
    dn = res["deuda_neta"]
    eq = res["equity_value"]
    vi = res["valor_intrinseco"]
    sh = res["shares_efectivas"]

    assert abs(dn - 30e6) < 1.0
    assert abs(eq - (ev - dn)) < 1.0
    assert abs(vi - round(eq / sh, 2)) < 0.05


def test_wacc_ponderacion_estricta_market_cap():
    """
    Verifica que las ponderaciones We y Wd se calculen sobre Market Cap + Total Debt
    y nunca sobre el valor contable de capital (Book Value).
    """
    mcap = 500e9
    total_debt = 25e9
    res = calcular_wacc(
        tasa_libre_riesgo=4.0,
        beta=1.0,
        mcap=mcap,
        total_debt=total_debt,
        int_exp=1.25e9,  # Kd = 5.0%
        tax_rate=0.20,
        erp=5.0,
    )
    assert abs(res["we"] - (500.0 / 525.0)) < 0.001
    assert abs(res["wd"] - (25.0 / 525.0)) < 0.001
    assert abs(res["ke"] - 9.0) < 0.01
    assert abs(res["kd"] - 5.0) < 0.01


def test_buyback_rate_normalizacion_y_limites():
    """
    Verifica que buyback_rate acepte tanto porcentajes (2.5) como decimales (0.025)
    y no produzca saltos desproporcionados ni divisiones por cero.
    """
    res_dec = calcular_fcff_valuation(
        ocf_hist=[100e6], capex_hist=[20e6], interest_hist=[5e6], pretax_hist=[80e6],
        taxprov_hist=[16e6], total_debt=10e6, total_cash=10e6, shares_diluted=10e6,
        mcap=500e6, beta=1.0, rf=4.0, precio_actual=50.0, buyback_rate=0.025
    )
    res_pct = calcular_fcff_valuation(
        ocf_hist=[100e6], capex_hist=[20e6], interest_hist=[5e6], pretax_hist=[80e6],
        taxprov_hist=[16e6], total_debt=10e6, total_cash=10e6, shares_diluted=10e6,
        mcap=500e6, beta=1.0, rf=4.0, precio_actual=50.0, buyback_rate=2.5
    )
    assert abs(res_dec["valor_intrinseco"] - res_pct["valor_intrinseco"]) < 0.05
    assert res_dec["shares_efectivas"] < 10e6


class TestRefactorizacionV2(unittest.TestCase):
    """Suite unittest que envuelve los tests de refactorización v2."""

    def test_dual_path_ebit(self):
        test_fcff_dual_path_ebit_primario()

    def test_dual_path_ocf_fallback(self):
        test_fcff_dual_path_ocf_fallback_cuando_ebit_negativo()

    def test_wacc_invariante(self):
        test_wacc_clamping_invariante_minimo()

    def test_fade_decreciente(self):
        test_fade_period_tasas_decrecientes()

    def test_mid_year_pv(self):
        test_mid_year_convention_pv_detalle()

    def test_buyback(self):
        test_buyback_reduce_acciones_efectivas()

    def test_wacc_g_edge(self):
        test_wacc_igual_a_g_edge_case()

    def test_msft_megacap(self):
        test_msft_megacap_sin_nan_ni_error()

    def test_nvda_calibracion_dcf(self):
        test_fcff_valuation_nvda_calibracion()

    def test_ma_mercado_congruente(self):
        test_fcff_valuation_mastercard_mercado_congruente()

    def test_aapl_mercado_congruente(self):
        test_fcff_valuation_apple_mercado_congruente()

    def test_bridge_deuda_neta(self):
        test_ev_to_equity_bridge_deuda_neta_exacto()

    def test_wacc_market_cap_we_wd(self):
        test_wacc_ponderacion_estricta_market_cap()

    def test_buyback_norm(self):
        test_buyback_rate_normalizacion_y_limites()


class TestDefensiveNumericUnpacking(unittest.TestCase):
    """Pruebas para blindaje contra TypeError: float() argument must be a string or a real number, not 'tuple'."""

    def test_safe_num_tupla_con_moneda_o_status(self):
        from data.financial_fetcher import safe_num, ConsensusWallStreet
        # Tuplas tipo (precio, moneda)
        self.assertEqual(safe_num((150.0, "USD")), 150.0)
        self.assertEqual(safe_num(("USD", 150.0)), 150.0)
        self.assertEqual(safe_num(("$150.50", "USD")), 150.50)
        # Tuplas tipo (valor, status)
        self.assertEqual(safe_num((200.0, "OK")), 200.0)
        self.assertEqual(safe_num((200.0, True)), 200.0)
        self.assertEqual(safe_num((None, 42.0)), 42.0)
        # Tupla 1-elemento
        self.assertEqual(safe_num((550.0,)), 550.0)
        # ConsensusWallStreet
        cw = ConsensusWallStreet(667.30, 735.0, 550.0)
        self.assertEqual(safe_num(cw), 667.30)
        # pd.Series / np.ndarray
        import numpy as np
        self.assertEqual(safe_num(pd.Series([99.9])), 99.9)
        self.assertEqual(safe_num(np.array([88.8])), 88.8)
        # Fallbacks seguros en multi-elemento
        self.assertEqual(safe_num([1, 2, 3], default=0.0), 0.0)
        self.assertEqual(safe_num({"dict": 1}, default=0.0), 0.0)

    def test_calcular_wacc_con_tuplas(self):
        # Asegurar que calcular_wacc nunca falle si se le pasan tuplas
        res = calcular_wacc(
            tasa_libre_riesgo=(4.20, "USD"),
            beta=(1.10, "5Y"),
            mcap=(500_000_000.0, "USD"),
            total_debt=(100_000_000.0, "USD"),
            int_exp=(5_000_000.0, "USD"),
            tax_rate=(0.21, "statutory"),
            erp=(5.0,),
        )
        self.assertIsInstance(res["wacc"], float)
        self.assertGreater(res["wacc"], 0.0)

    def test_calcular_fcff_valuation_con_tuplas(self):
        # Simular entradas con tuplas para verificar ausencia total de TypeError
        res = calcular_fcff_valuation(
            ocf_hist=[100_000_000.0],
            capex_hist=[20_000_000.0],
            interest_hist=[5_000_000.0],
            pretax_hist=[50_000_000.0],
            taxprov_hist=[10_000_000.0],
            total_debt=(50_000_000.0, "USD"),
            total_cash=(30_000_000.0, "USD"),
            shares_diluted=(10_000_000.0,),
            mcap=(800_000_000.0, "USD"),
            beta=(1.10,),
            rf=(4.20,),
            precio_actual=(80.0, "$"),
            erp=(5.0,),
            buyback_rate=(0.01,),
            g_term_override=(0.025,),
        )
        self.assertIsInstance(res["valor_intrinseco"], float)
        self.assertGreater(res["valor_intrinseco"], 0.0)

    def test_calcular_ddm_con_tuplas(self):
        res = calcular_ddm(
            div_rate=(2.50, "USD"),
            ke=(8.5, "%"),
            g_div=(0.025,),
            precio_actual=(100.0,),
        )
        self.assertIsInstance(res["valor_intrinseco_ddm"], float)
        self.assertGreater(res["valor_intrinseco_ddm"], 0.0)


class TestRecalibracionFCFFYSupuestosAvanzados(unittest.TestCase):
    """Pruebas de consistencia y reactividad para Supuestos Avanzados DCF y Consenso Wall Street."""

    def setUp(self):
        self.ma_params = dict(
            ocf_hist=[17.5e9, 15.0e9, 13.0e9],
            capex_hist=[0.5e9, 0.45e9, 0.4e9],
            interest_hist=[0.6e9, 0.5e9, 0.4e9],
            pretax_hist=[18.0e9, 15.5e9, 13.5e9],
            taxprov_hist=[3.5e9, 3.0e9, 2.6e9],
            total_debt=24.0e9,
            total_cash=11.5e9,
            shares_diluted=870e6,
            mcap=510e9,
            beta=1.05,
            rf=4.25,
            precio_actual=588.0,
            revenue_ttm=28.0e9,
            revenue_growth_api=0.12,
            operating_margin_hist=0.58,
            ebit_hist=[19.0e9, 16.5e9, 14.5e9],
            da_hist=[1.0e9, 0.9e9, 0.8e9],
            delta_nwc_hist=[0.2e9, 0.15e9, 0.1e9],
            ticker="MA",
        )

    def test_sensibilidad_recompra_neta_anual(self):
        """Verifica que el aumento paulatino de recompras incremente el valor por acción de forma monótona."""
        res_0 = calcular_fcff_valuation(**self.ma_params, buyback_rate=0.0, fade_years=3, g_term_override=0.025)
        res_2 = calcular_fcff_valuation(**self.ma_params, buyback_rate=0.02, fade_years=3, g_term_override=0.025)
        res_4 = calcular_fcff_valuation(**self.ma_params, buyback_rate=0.04, fade_years=3, g_term_override=0.025)

        self.assertGreater(res_2["valor_intrinseco"], res_0["valor_intrinseco"])
        self.assertGreater(res_4["valor_intrinseco"], res_2["valor_intrinseco"])
        self.assertLess(res_2["shares_efectivas"], res_0["shares_efectivas"])
        self.assertLess(res_4["shares_efectivas"], res_2["shares_efectivas"])

    def test_sensibilidad_fade_period(self):
        """Verifica que el Fade Period extienda adecuadamente el horizonte y la transición a tasa terminal."""
        res_1 = calcular_fcff_valuation(**self.ma_params, buyback_rate=0.0, fade_years=1, g_term_override=0.025)
        res_5 = calcular_fcff_valuation(**self.ma_params, buyback_rate=0.0, fade_years=5, g_term_override=0.025)

        self.assertEqual(res_1["n_total"], 6)
        self.assertEqual(res_5["n_total"], 10)
        self.assertGreater(res_5["valor_intrinseco"], res_1["valor_intrinseco"])

    def test_sensibilidad_tasa_terminal_g(self):
        """Verifica que la tasa terminal aumente el valor intrínseco respetando el spread mínimo WACC > g."""
        res_g15 = calcular_fcff_valuation(**self.ma_params, buyback_rate=0.0, fade_years=3, g_term_override=0.015)
        res_g30 = calcular_fcff_valuation(**self.ma_params, buyback_rate=0.0, fade_years=3, g_term_override=0.030)

        self.assertGreater(res_g30["valor_intrinseco"], res_g15["valor_intrinseco"])
        self.assertGreater(res_g30["wacc"] / 100.0, res_g30["g_term"])

    def test_consenso_yfinance_exclusivo(self):
        """Verifica que el consenso de analistas para MA extraiga un valor numérico positivo desde yfinance."""
        from data.financial_fetcher import obtener_consenso_wall_street
        mock_info = {
            "targetMeanPrice": 670.92,
            "targetHighPrice": 740.0,
            "targetLowPrice": 550.0,
            "numberOfAnalystOpinions": 37,
            "recommendationKey": "strong_buy",
        }
        res = obtener_consenso_wall_street("MA", yf_info=mock_info)
        self.assertEqual(res.target_mean, 670.92)
        self.assertEqual(res.target_high, 740.0)
        self.assertEqual(res.target_low, 550.0)
        self.assertEqual(res.num_analysts, 37)
        self.assertEqual(res.recommendation, "strong_buy")

    def test_ma_fcff_no_sobrevalorado_evita_986(self):
        """Verifica que con CapEx anual real (~1.2B) MA se valore entre $400-$650 y no en $986+."""
        # Parámetros agresivos (fade=4, bb=2%, g=3%)
        res_agr = calcular_fcff_valuation(
            ocf_hist=[17.43e9, 17.65e9, 14.78e9],
            capex_hist=[1.215e9, 1.194e9, 1.088e9],
            interest_hist=[0.55e9, 0.45e9, 0.4e9],
            pretax_hist=[18.0e9, 15.5e9, 13.5e9],
            taxprov_hist=[3.5e9, 3.0e9, 2.6e9],
            total_debt=24.64e9,
            total_cash=11.61e9,
            shares_diluted=869.46e6,
            mcap=511e9,
            beta=1.05,
            rf=4.25,
            precio_actual=588.0,
            erp=5.0,
            cagr_revenue_hist=0.12,
            revenue_growth_api=0.12,
            revenue_ttm=28.0e9,
            operating_margin_hist=0.58,
            ticker="MA",
            buyback_rate=0.02,
            fade_years=4,
            g_term_override=0.03,
            ebit_hist=[19.51e9, 16.33e9, 14.63e9],
            da_hist=[1.1e9, 0.95e9, 0.85e9],
            delta_nwc_hist=[0.1e9, 0.1e9, 0.1e9],
        )
        self.assertLess(res_agr["valor_intrinseco"], 650.0, "El valor intrínseco de MA no debe sobrevalorarse a $986")
        self.assertGreater(res_agr["valor_intrinseco"], 500.0)

        # Parámetros base (fade=3, bb=0%, g=2.5%)
        res_base = calcular_fcff_valuation(
            ocf_hist=[17.43e9, 17.65e9, 14.78e9],
            capex_hist=[1.215e9, 1.194e9, 1.088e9],
            interest_hist=[0.55e9, 0.45e9, 0.4e9],
            pretax_hist=[18.0e9, 15.5e9, 13.5e9],
            taxprov_hist=[3.5e9, 3.0e9, 2.6e9],
            total_debt=24.64e9,
            total_cash=11.61e9,
            shares_diluted=869.46e6,
            mcap=511e9,
            beta=1.05,
            rf=4.25,
            precio_actual=588.0,
            erp=5.0,
            cagr_revenue_hist=0.12,
            revenue_growth_api=0.12,
            revenue_ttm=28.0e9,
            operating_margin_hist=0.58,
            ticker="MA",
            buyback_rate=0.0,
            fade_years=3,
            g_term_override=0.025,
            ebit_hist=[19.51e9, 16.33e9, 14.63e9],
            da_hist=[1.1e9, 0.95e9, 0.85e9],
            delta_nwc_hist=[0.1e9, 0.1e9, 0.1e9],
        )
        self.assertLess(res_base["valor_intrinseco"], 500.0)
        self.assertGreater(res_base["valor_intrinseco"], 400.0)

    def test_capex_ttm_implicito_extraer_fcff(self):
        """Verifica que extraer_fcff_desapalancado obtenga CapEx TTM exacto de OCF - FCF."""
        from data.financial_fetcher import extraer_fcff_desapalancado
        import pandas as pd
        mock_info = {
            "operatingCashflow": 17_437_000_000.0,
            "freeCashflow": 16_222_000_000.0,
            "capitalExpenditures": 475_749_376.0,  # Trimestral anómalo
            "operatingIncome": 19_514_000_000.0,
        }
        mock_cf = pd.DataFrame({
            "2024": {"Operating Cash Flow": 17.43e9, "Capital Expenditure": -1.215e9},
            "2023": {"Operating Cash Flow": 17.65e9, "Capital Expenditure": -1.194e9},
        })
        mock_inc = pd.DataFrame()
        mock_bs = pd.DataFrame()

        comp = extraer_fcff_desapalancado(mock_cf, mock_inc, mock_bs, mock_info)
        # CapEx debe ser ~1.215B y no 475M
        self.assertGreater(comp["capex_hist"][0], 1.0e9)
        self.assertEqual(len(comp["ocf_hist"]), len(comp["capex_hist"]))

    def test_multiplos_calibrados_sin_na(self):
        """Verifica que los 5 múltiplos de valuación se calculen correctamente sin 'N/A'."""
        from engine.metrics import calcular_multiplos_valuacion, calcular_buyback_yield
        import pandas as pd

        # Simular métricas para Mastercard (MA)
        res_m = calcular_multiplos_valuacion(
            precio_actual=588.0,
            mcap=511e9,
            eps_ttm=18.41,
            forward_eps=23.03,
            fcf_ttm=16.22e9,
            ebitda_ttm=22.20e9,
            total_debt=24.64e9,
            total_cash=11.61e9,
            revenue_ttm=28.0e9,
            total_equity=6.5e9,
            peg_info=1.52,
            earnings_growth=0.15,
            buyback_yield=2.2,
        )
        self.assertNotEqual(res_m["pe_str"], "N/A")
        self.assertNotEqual(res_m["p_fcf_str"], "N/A")
        self.assertNotEqual(res_m["peg_str"], "N/A")
        self.assertNotEqual(res_m["ev_ebitda_str"], "N/A")
        self.assertEqual(res_m["buyback_yield_str"], "2.2%")

        self.assertAlmostEqual(res_m["pe"], 588.0 / 18.41, delta=0.2)
        self.assertAlmostEqual(res_m["p_fcf"], 511e9 / 16.22e9, delta=0.5)
        self.assertEqual(res_m["peg"], 1.52)
        self.assertAlmostEqual(res_m["ev_ebitda"], (511e9 + 24.64e9 - 11.61e9) / 22.20e9, delta=0.5)

    def test_reactividad_sliders_supuestos_avanzados(self):
        """Verifica que los sliders de supuestos avanzados (fade, recompra, g) reaccionen armónicamente en tiempo real."""
        # Base
        res_base = calcular_fcff_valuation(**self.ma_params, buyback_rate=0.0, fade_years=3, g_term_override=0.025)
        # Slider Fade: 1 a 5 años
        res_fade_min = calcular_fcff_valuation(**self.ma_params, buyback_rate=0.0, fade_years=1, g_term_override=0.025)
        res_fade_max = calcular_fcff_valuation(**self.ma_params, buyback_rate=0.0, fade_years=5, g_term_override=0.025)
        self.assertLess(res_fade_min["valor_intrinseco"], res_fade_max["valor_intrinseco"])

        # Slider Recompra: 0% a 3%
        res_bb_0 = calcular_fcff_valuation(**self.ma_params, buyback_rate=0.0, fade_years=3, g_term_override=0.025)
        res_bb_3 = calcular_fcff_valuation(**self.ma_params, buyback_rate=0.03, fade_years=3, g_term_override=0.025)
        self.assertLess(res_bb_0["valor_intrinseco"], res_bb_3["valor_intrinseco"])
        self.assertGreater(res_bb_0["shares_efectivas"], res_bb_3["shares_efectivas"])

        # Slider Tasa Terminal g: 1.5% a 3.0%
        res_g_low = calcular_fcff_valuation(**self.ma_params, buyback_rate=0.0, fade_years=3, g_term_override=0.015)
        res_g_high = calcular_fcff_valuation(**self.ma_params, buyback_rate=0.0, fade_years=3, g_term_override=0.030)
        self.assertLess(res_g_low["valor_intrinseco"], res_g_high["valor_intrinseco"])

        # Precio Máximo de Compra siempre aplica el margen de descuento proporcional
        self.assertAlmostEqual(res_base["precio_max_compra"], round(res_base["valor_intrinseco"] * 0.90, 2), delta=0.05)


if __name__ == "__main__":
    unittest.main()
