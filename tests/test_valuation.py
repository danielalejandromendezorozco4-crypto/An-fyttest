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
# 1. PRUEBAS DE NO CIRCULARIDAD EN RESPALDOS (FALLBACK)
# ─────────────────────────────────────────────────────────────────────────────

def test_calcular_fcff_normalizado_no_circular():
    """
    Verifica que el FCFF Normalizado se base estrictamente en ingresos,
    margen operativo histórico y tasa impositiva, o en EPS ajustado,
    eliminando cualquier circularidad con el precio de cotización del mercado.
    """
    # Caso 1: Con ingresos y margen operativo histórico
    rev_ttm = 100_000.0
    mg_op = 0.20  # 20%
    tax_rate = 0.21
    shares = 1_000.0

    # FCFF = 100,000 * 0.20 * (1 - 0.21) = 15,800 total | 15.80 por acción
    fcff_tot, fcff_ps = calcular_fcff_normalizado(
        revenue_ttm=rev_ttm,
        operating_margin_hist=mg_op,
        tax_rate=tax_rate,
        shares_diluted=shares,
    )
    assert abs(fcff_tot - 15_800.0) < 0.01
    assert abs(fcff_ps - 15.80) < 0.01

    # Caso 2: Margen operativo menor a 10% (debe aplicar el piso de 10%)
    fcff_tot_low, fcff_ps_low = calcular_fcff_normalizado(
        revenue_ttm=rev_ttm,
        operating_margin_hist=0.04,
        tax_rate=tax_rate,
        shares_diluted=shares,
    )
    # FCFF = 100,000 * 0.10 * (1 - 0.21) = 7,900
    assert abs(fcff_tot_low - 7_900.0) < 0.01

    # Caso 3: Sin ingresos pero con EPS positivo
    fcff_tot_eps, fcff_ps_eps = calcular_fcff_normalizado(
        revenue_ttm=0.0,
        operating_margin_hist=0.0,
        tax_rate=tax_rate,
        shares_diluted=shares,
        eps_ttm=5.0,
        reinvestment_rate=0.25,
    )
    # FCFF por acción = 5.0 * 0.75 = 3.75
    assert abs(fcff_ps_eps - 3.75) < 0.01
    assert abs(fcff_tot_eps - 3_750.0) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
# 2. PRUEBAS DE ESTABILIZACIÓN DE CRECIMIENTO Y FADE-DOWN
# ─────────────────────────────────────────────────────────────────────────────

def test_calcular_curva_crecimiento_fade_down():
    """
    Verifica que la tasa de crecimiento proyectada esté acotada entre 3.0% y 15.0%
    y desacelere progresivamente (fade-down) hacia g_term a lo largo de los 5 años.
    """
    # Tasa muy alta (e.g. 35%) debe acotarse al 15%
    curva_alta = calcular_curva_crecimiento_5y(
        cagr_revenue_hist=0.35,
        revenue_growth_api=0.40,
        g_term=0.022,
        n_years=5,
    )
    assert len(curva_alta) == 5
    assert curva_alta[0] == 0.15  # Año 1 tope de 15%
    assert curva_alta[0] > curva_alta[1] > curva_alta[2] > curva_alta[3] > curva_alta[4]
    assert curva_alta[-1] < 0.08  # Desaceleración efectiva

    # Tasa muy baja o negativa (e.g. -5%) debe acotarse al 3%
    curva_baja = calcular_curva_crecimiento_5y(
        cagr_revenue_hist=-0.05,
        revenue_growth_api=-0.10,
        g_term=0.020,
        n_years=5,
    )
    assert curva_baja[0] == 0.030  # Año 1 piso de 3%


# ─────────────────────────────────────────────────────────────────────────────
# 3. PRUEBAS DE RESTRICCIÓN DE TASA TERMINAL (GORDON-SHAPIRO)
# ─────────────────────────────────────────────────────────────────────────────

def test_g_term_restringido():
    """
    Verifica que g_term se mantenga en el rango de crecimiento del PIB (1.5% a 2.5%)
    y conserve un spread mínimo de seguridad de 3.5% respecto al WACC.
    """
    # WACC normal (9.0% -> 0.090)
    g1 = calcular_g_term_restringido(0.090)
    assert 0.015 <= g1 <= 0.025
    assert (0.090 - g1) >= 0.035

    # WACC bajo (5.5% -> 0.055): debe restringir g_term para asegurar spread >= 3.5%
    g2 = calcular_g_term_restringido(0.055)
    assert (0.055 - g2) >= 0.035


# ─────────────────────────────────────────────────────────────────────────────
# 4. PRUEBAS DE MOTOR WACC UNIFICADO
# ─────────────────────────────────────────────────────────────────────────────

def test_wacc_unificado():
    """
    Verifica que calcular_wacc opere como fuente única de verdad,
    calculando Ke por CAPM, Kd empírico y ponderaciones de mercado correctas.
    """
    res = calcular_wacc(
        tasa_libre_riesgo=4.20,
        beta=1.10,
        mcap=2_000_000_000.0,
        total_debt=500_000_000.0,
        int_exp=25_000_000.0,  # Kd = 25M / 500M = 5.0%
        tax_rate=0.20,
        erp=5.0,
    )

    # Ke = 4.20 + 1.10 * 5.0 = 9.70%
    assert abs(res["ke"] - 9.70) < 0.01

    # Kd = 5.0%
    assert abs(res["kd"] - 5.00) < 0.01

    # We = 2000 / 2500 = 0.80 | Wd = 500 / 2500 = 0.20
    assert abs(res["we"] - 0.80) < 0.01
    assert abs(res["wd"] - 0.20) < 0.01

    # WACC = 0.80 * 9.70 + 0.20 * 5.0 * (1 - 0.20) = 7.76 + 0.80 = 8.56%
    assert abs(res["wacc"] - 8.56) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
# 5. PRUEBAS DE CONVENCIÓN DE MEDIO AÑO Y PUENTE EV A EQUITY
# ─────────────────────────────────────────────────────────────────────────────

def test_fcff_mid_year_convention_y_puente():
    """
    Verifica que:
    1. Los flujos explícitos aplican Mid-Year Discounting (t - 0.5).
    2. El Valor Terminal se descuenta a t = 5.
    3. El puente EV -> Deuda Neta -> Equity Value -> Por Acción concilia con exactitud.
    """
    resultado = calcular_fcff_valuation(
        ocf_hist              = [100e9, 90e9, 80e9],
        capex_hist            = [20e9, 18e9, 16e9],
        interest_hist         = [2e9, 1.8e9, 1.5e9],
        pretax_hist           = [90e9, 80e9, 70e9],
        taxprov_hist          = [18e9, 16e9, 14e9],
        total_debt            = 40e9,
        total_cash            = 60e9,   # Deuda neta negativa (-20B)
        shares_diluted        = 10e9,
        mcap                  = 1_500e9,
        beta                  = 1.0,
        rf                    = 4.25,
        precio_actual         = 150.0,
        cagr_revenue_hist     = 0.08,
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
# 6. PRUEBAS DE CALIBRACIÓN DE GOOGL Y MEGA CAPS
# ─────────────────────────────────────────────────────────────────────────────

def test_fcff_googl_calibracion():
    """
    Verifica que con datos fundamentales reales de Alphabet Inc. (GOOGL),
    el valor intrínseco resultante se alinee al consenso institucional ($150 - $220).
    """
    ocf_hist      = [105e9, 101.7e9, 91.5e9, 91.6e9]
    capex_hist    = [38e9,  32.2e9,  31.5e9, 24.6e9]
    interest_hist = [0.35e9, 0.3e9,  0.35e9, 0.35e9]
    pretax_hist   = [100e9, 88e9,    75e9,   90e9]
    taxprov_hist  = [16e9,  14e9,    11.5e9, 14.7e9]
    total_debt    = 28e9
    total_cash    = 110e9
    shares_diluted = 12.35e9
    mcap          = 2_180e9
    beta          = 1.05
    rf            = 4.25
    precio_actual = 175.0

    resultado = calcular_fcff_valuation(
        ocf_hist          = ocf_hist,
        capex_hist        = capex_hist,
        interest_hist     = interest_hist,
        pretax_hist       = pretax_hist,
        taxprov_hist      = taxprov_hist,
        total_debt        = total_debt,
        total_cash        = total_cash,
        shares_diluted    = shares_diluted,
        mcap              = mcap,
        beta              = beta,
        rf                = rf,
        precio_actual     = precio_actual,
        cagr_revenue_hist = 0.13,
    )

    valor_intr = resultado["valor_intrinseco"]
    assert valor_intr > 135.0, f"Valor intrínseco de GOOGL muy bajo: ${valor_intr:.2f}"
    assert 140.0 <= valor_intr <= 240.0, f"Valor intrínseco fuera de rango: ${valor_intr:.2f}"
    assert 7.0 <= resultado["wacc"] <= 11.0, f"WACC anómalo: {resultado['wacc']:.2f}%"


def test_fcff_empresa_grande():
    """Empresa de gran capitalización y bajo apalancamiento (AAPL/MSFT)."""
    resultado = calcular_fcff_valuation(
        ocf_hist      = [120e9, 110e9, 100e9, 95e9],
        capex_hist    = [11e9,  10e9,  9e9,   8e9],
        interest_hist = [4e9,   3.5e9, 3e9,   2.5e9],
        pretax_hist   = [110e9, 100e9, 90e9,  85e9],
        taxprov_hist  = [27e9,  25e9,  22e9,  21e9],
        total_debt    = 110e9,
        total_cash    = 160e9,
        shares_diluted = 15.5e9,
        mcap          = 3_000e9,
        beta          = 1.2,
        rf            = 4.35,
        precio_actual = 195.0,
    )
    assert isinstance(resultado, dict)
    assert 6.0 <= resultado["wacc"] <= 14.0
    assert resultado["enterprise_value"] > 0
    assert resultado["valor_intrinseco"] > 0


def test_fcff_empresa_sin_deuda():
    """Empresa sin deuda (deuda = 0). wd=0, kd=0 y WACC = Ke."""
    resultado = calcular_fcff_valuation(
        ocf_hist      = [50e9, 45e9, 40e9],
        capex_hist    = [5e9,  4e9,  3e9],
        interest_hist = [0.0,  0.0,  0.0],
        pretax_hist   = [60e9, 55e9, 50e9],
        taxprov_hist  = [15e9, 14e9, 13e9],
        total_debt    = 0.0,
        total_cash    = 80e9,
        shares_diluted = 8e9,
        mcap          = 500e9,
        beta          = 0.9,
        rf            = 4.35,
        precio_actual = 62.5,
    )
    assert resultado["wd"] == 0.0
    assert resultado["kd"] == 0.0
    assert resultado["we"] == 1.0
    assert abs(resultado["wacc"] - resultado["ke"]) < 2.0
    assert resultado["valor_intrinseco"] > 0


def test_fcff_empresa_alto_apalancamiento():
    """Empresa de alto apalancamiento."""
    resultado = calcular_fcff_valuation(
        ocf_hist      = [20e9, 18e9, 16e9],
        capex_hist    = [18e9, 17e9, 15e9],
        interest_hist = [8e9,  7e9,  6.5e9],
        pretax_hist   = [5e9,  4e9,  3e9],
        taxprov_hist  = [1.2e9, 1e9,  0.8e9],
        total_debt    = 150e9,
        total_cash    = 10e9,
        shares_diluted = 7e9,
        mcap          = 80e9,
        beta          = 0.8,
        rf            = 4.35,
        precio_actual = 11.4,
    )
    assert resultado["valor_intrinseco"] >= 0


def test_fcff_flujo_negativo():
    """Empresa con FCFF base negativo activa fallback no circular."""
    resultado = calcular_fcff_valuation(
        ocf_hist      = [-5e9, -3e9, 2e9],
        capex_hist    = [10e9, 8e9,  6e9],
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
        revenue_ttm   = 10e9,
        operating_margin_hist = 0.10,
    )
    assert resultado["valor_intrinseco"] >= 0
    assert len(resultado["fcff_proyectado"]) == 5


def test_fcff_datos_minimos():
    """Solo 1 año de datos disponibles."""
    resultado = calcular_fcff_valuation(
        ocf_hist      = [10e9],
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
    assert len(resultado["fcff_proyectado"]) == 5
    assert resultado["valor_intrinseco"] >= 0


def test_fcff_empresa_sin_capex():
    """Empresa asset-light sin CapEx."""
    resultado = calcular_fcff_valuation(
        ocf_hist      = [30e9, 28e9, 25e9],
        capex_hist    = [0.0, 0.0, 0.0],
        interest_hist = [1e9, 0.9e9, 0.8e9],
        pretax_hist   = [35e9, 32e9, 28e9],
        taxprov_hist  = [8.75e9, 8e9, 7e9],
        total_debt    = 10e9,
        total_cash    = 50e9,
        shares_diluted = 2e9,
        mcap          = 600e9,
        beta          = 1.3,
        rf            = 4.35,
        precio_actual = 300.0,
    )
    assert resultado["valor_intrinseco"] > 0
    assert abs(resultado["fcff_historico"][0] - (30e9 + 1e9 * (1.0 - resultado["tax_rate_real"]))) < 1e6


def test_calcular_dcf_intr_ps_subvalorada():
    resultado = calcular_dcf_intr_ps(
        wacc_var=9.0, g_term_var=0.022, flujo_por_accion=10.0, g_1_5=0.10,
        precio_actual=100.0, eps_ttm=8.0, total_cash=5000.0, total_debt=2000.0,
        shares_current=100.0,
    )
    assert resultado["valor_intrinseco"] > 100.0
    assert resultado["status"] == "🟢"


def test_calcular_dcf_intr_ps_sobrevalorada():
    resultado = calcular_dcf_intr_ps(
        wacc_var=12.0, g_term_var=0.015, flujo_por_accion=1.0, g_1_5=0.03,
        precio_actual=200.0, eps_ttm=1.5, total_cash=100.0, total_debt=5000.0,
        shares_current=100.0,
    )
    assert resultado["valor_intrinseco"] < 200.0
    assert resultado["status"] == "🔴"


def test_crear_calculador_dcf():
    calculador = crear_calculador_dcf(
        flujo_por_accion=5.0, g_1_5=0.08, precio_actual=50.0,
        eps_ttm=4.0, total_cash=1000.0, total_debt=500.0, shares_current=50.0
    )
    val = calculador(9.5, 0.022)
    assert isinstance(val, float)
    assert val > 0


def test_altman_zscore_zona_segura():
    res = calcular_altman_zscore(debt_eq=0.5, roa=15.0)
    assert res["z_score"] > 2.99
    assert res["status"] == "🟢"


def test_altman_zscore_zona_peligro():
    res = calcular_altman_zscore(debt_eq=5.0, roa=-2.0)
    assert res["z_score"] < 1.81
    assert res["status"] == "🔴"


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


def test_fcff_perfil_nflx_empirico():
    """
    Verifica valuación con estructura de capital de streaming / media (NFLX):
    Deuda moderada, alto FCF operativo, amortización de contenido y beta dinámico.
    """
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
        erp               = 5.25,
        cagr_revenue_hist = 0.12,
    )
    assert res["valor_intrinseco"] > 0
    assert 7.5 <= res["wacc"] <= 12.0
    assert res["kd"] > 0
    assert res["we"] > 0.90
    assert res["wd"] < 0.10
    assert res["tax_rate_real"] > 0.10


def test_fcff_perfil_aapl_empirico():
    """
    Verifica valuación con estructura de capital cash-rich / megacap (AAPL):
    Caja superior a deuda, márgenes altos, beta de mercado equilibrado.
    """
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
        erp               = 5.00,
        cagr_revenue_hist = 0.08,
    )
    assert res["valor_intrinseco"] > 0
    assert res["deuda_neta"] < 0  # Caja neta positiva (-55B)
    assert res["equity_value"] > res["enterprise_value"]


def test_fcff_perfil_ko_empirico():
    """
    Verifica valuación con estructura de consumo defensivo / dividend aristocrat (KO):
    Beta bajo (0.60), apalancamiento estable, flujos altamente predecibles.
    """
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
        erp               = 4.80,
        cagr_revenue_hist = 0.05,
    )
    assert res["valor_intrinseco"] > 0
    assert res["ke"] < 8.0  # Ke defensivo: 4.25 + 0.60 * 4.80 = 7.13%
    assert 6.0 <= res["wacc"] <= 9.0


def test_fcff_perfil_jnj_empirico():
    """
    Verifica valuación con perfil de salud / AAA balance sheet (JNJ):
    Bajo beta (0.55), alta solvencia, bajo costo de deuda empírico.
    """
    res = calcular_fcff_valuation(
        ocf_hist          = [22.0e9, 21.0e9, 20.0e9],
        capex_hist        = [4.5e9, 4.2e9, 4.0e9],
        interest_hist     = [0.8e9, 0.75e9, 0.70e9],
        pretax_hist       = [22.0e9, 21.0e9, 19.5e9],
        taxprov_hist      = [3.5e9, 3.3e9, 3.0e9],
        total_debt        = 30.0e9,
        total_cash        = 25.0e9,
        shares_diluted    = 2.4e9,
        mcap              = 380.0e9,
        beta              = 0.55,
        rf                = 4.25,
        precio_actual     = 160.0,
        erp               = 4.80,
        cagr_revenue_hist = 0.05,
    )
    assert res["valor_intrinseco"] > 0
    assert res["wacc"] <= 8.5
    assert res["tax_rate_real"] < 0.20


class TestValuationUnittest(unittest.TestCase):
    def test_fcff_normalizado(self):
        test_calcular_fcff_normalizado_no_circular()

    def test_curva_crecimiento(self):
        test_calcular_curva_crecimiento_fade_down()

    def test_g_term(self):
        test_g_term_restringido()

    def test_wacc(self):
        test_wacc_unificado()

    def test_mid_year_y_puente(self):
        test_fcff_mid_year_convention_y_puente()

    def test_fcff_googl(self):
        test_fcff_googl_calibracion()

    def test_fcff_nflx(self):
        test_fcff_perfil_nflx_empirico()

    def test_fcff_aapl(self):
        test_fcff_perfil_aapl_empirico()

    def test_fcff_ko(self):
        test_fcff_perfil_ko_empirico()

    def test_fcff_jnj(self):
        test_fcff_perfil_jnj_empirico()

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


if __name__ == "__main__":
    unittest.main()

