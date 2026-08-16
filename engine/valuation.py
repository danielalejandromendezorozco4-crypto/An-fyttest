from __future__ import annotations

import statistics
from typing import Callable, Dict, List, Optional, Tuple, Union

# NOTE: Imports de módulos con dependencia de streamlit (financial_fetcher,
# config.settings) se hacen de forma diferida (lazy) dentro de cada función
# para evitar arrastrar streamlit al nivel de módulo. Esto previene
# ImportError en pytest y en Streamlit Cloud.


# ─────────────────────────────────────────────────────────────────────────────
# 1. MOTOR UNIFICADO DE WACC (ÚNICA FUENTE DE VERDAD)
# ─────────────────────────────────────────────────────────────────────────────

def calcular_wacc(
    tasa_libre_riesgo: float,
    beta: float,
    mcap: float,
    total_debt: float,
    int_exp: float,
    tax_rate: float,
    fmp_key: str = "",
    fred_key: str = "",
    ticker: str = "",
    erp: float = 5.0,
) -> Dict[str, float]:
    """
    Motor centralizado y unificado para el cálculo del Costo Promedio Ponderado
    de Capital (WACC), garantizando que una única fuente de verdad alimente
    el Dashboard, el motor FCFF, las matrices de sensibilidad y los reportes PDF.

    .. math::

        K_e = R_f + \\beta \\times ERP
        K_d = \\frac{\\text{Gastos por Intereses}}{\\text{Deuda Total}} \\times 100
        WACC = \\frac{E}{V} K_e + \\frac{D}{V} K_d (1 - T_{ef})

    Args:
        tasa_libre_riesgo: Tasa libre de riesgo en % (e.g. 4.35 para 4.35%).
        beta:              Coeficiente de volatilidad beta del activo.
        mcap:              Capitalización de mercado actual en USD.
        total_debt:        Deuda financiera total en USD.
        int_exp:           Gastos anuales por intereses en USD.
        tax_rate:          Tasa impositiva efectiva real (decimal, e.g. 0.21).
        fmp_key:           API Key opcional de Financial Modeling Prep.
        fred_key:          API Key opcional de FRED.
        ticker:            Símbolo bursátil.
        erp:               Prima de riesgo de mercado en % (default: 5.0%).

    Returns:
        Diccionario tipado con ``wacc``, ``ke``, ``kd``, ``we``, ``wd``, ``rf``, ``erp``, ``tax_rate``.
    """
    rf = max(float(tasa_libre_riesgo), 0.0)
    beta_val = max(float(beta), 0.1)
    erp_val = max(float(erp), 3.0)
    t_ef = max(min(float(tax_rate), 0.35), 0.0)

    # 1. Costo de capital propio (CAPM)
    ke = rf + (beta_val * erp_val)

    # 2. Costo de deuda (Kd) empírico
    kd = 0.0
    if total_debt > 0:
        if int_exp > 0:
            kd_raw = (int_exp / total_debt) * 100.0
            kd = min(max(kd_raw, rf), 12.0)
        elif ticker and (fmp_key or fred_key):
            try:
                from data.financial_fetcher import obtener_kd_fmp_fred  # noqa: PLC0415
                kd_fmp = obtener_kd_fmp_fred(ticker, fmp_key, fred_key, int_exp, total_debt)
                kd = min(max(float(kd_fmp), rf), 12.0)
            except Exception:
                kd = max(rf + 1.0, 5.50)
        else:
            kd = max(rf + 1.0, 5.50)

    # 3. Ponderaciones a valor de mercado
    total_capital = mcap + total_debt
    if total_capital > 0:
        we = mcap / total_capital
        wd = total_debt / total_capital
    else:
        we, wd = 1.0, 0.0

    # 4. WACC institucional
    wacc_raw = (we * ke) + (wd * kd * (1.0 - t_ef))
    # Límites institucionales: spread mínimo sobre Rf (+0.5%), piso 6.5%, techo 14.0%
    wacc_min = max(rf + 0.5, 6.5)
    wacc = min(max(wacc_raw, wacc_min), 14.0)

    return {
        "wacc": round(wacc, 4),
        "ke": round(ke, 4),
        "kd": round(kd, 4),
        "we": round(we, 4),
        "wd": round(wd, 4),
        "rf": round(rf, 4),
        "erp": round(erp_val, 4),
        "tax_rate": round(t_ef, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. AUXILIARES: NORMALIZACIÓN DE FLUJOS Y RESTRICCIONES METODOLÓGICAS
# ─────────────────────────────────────────────────────────────────────────────

def calcular_fcff_normalizado(
    revenue_ttm: float,
    operating_margin_hist: float,
    tax_rate: float,
    shares_diluted: float,
    eps_ttm: float = 0.0,
    reinvestment_rate: float = 0.25,
) -> Tuple[float, float]:
    """
    Calcula el FCFF Operativo Normalizado eliminando toda circularidad con el
    precio de cotización actual del mercado.

    .. math::

        \\text{FCFF Normalizado Total} = \\text{Revenue}_{TTM} \\times \\max(Mg_{op}, 0.10) \\times (1 - T_{ef})
        \\text{FCFF Normalizado por Acción} = \\frac{\\text{FCFF Normalizado Total}}{\\text{Shares}}

    Si Revenue <= 0, utiliza el EPS TTM ajustado por reinversión:

    .. math::

        \\text{FCFF Normalizado por Acción} = \\text{EPS}_{TTM} \\times (1 - \\text{ReinvestmentRate})

    Returns:
        Tupla ``(fcff_normalizado_total, fcff_normalizado_por_accion)``.
    """
    t_ef = max(min(float(tax_rate), 0.35), 0.0)
    mg_op = max(float(operating_margin_hist), 0.10)

    if revenue_ttm > 0:
        fcff_total = revenue_ttm * mg_op * (1.0 - t_ef)
        fcff_ps = fcff_total / shares_diluted if shares_diluted > 0 else 0.0
        return fcff_total, fcff_ps

    if eps_ttm > 0:
        fcff_ps = eps_ttm * (1.0 - max(min(reinvestment_rate, 0.50), 0.10))
        fcff_total = fcff_ps * shares_diluted if shares_diluted > 0 else 0.0
        return fcff_total, fcff_ps

    return 0.0, 0.0


def calcular_g_term_restringido(wacc_decimal: float) -> float:
    """
    Restringe la tasa terminal de Gordon-Shapiro estrictamente al rango de
    crecimiento de largo plazo del PIB (1.5% a 2.5%), asegurando un spread mínimo
    de seguridad de 3.5% respecto al WACC para evitar distorsiones en el Valor Terminal.

    .. math::

        g_{term} = \\min(0.025, \\max(0.015, WACC_{decimal} - 0.035))
    """
    g_term = min(0.025, max(0.015, wacc_decimal - 0.035))
    if (wacc_decimal - g_term) < 0.035:
        g_term = max(wacc_decimal - 0.035, 0.010)
    return round(g_term, 5)


def calcular_curva_crecimiento_5y(
    cagr_revenue_hist: float,
    revenue_growth_api: float,
    g_term: float,
    n_years: int = 5,
) -> List[float]:
    """
    Estabiliza la tasa de crecimiento proyectada a 5 años dando prioridad a la
    CAGR histórica de ingresos (3 a 5 años) con límites duros entre 3.0% y 15.0%,
    e implementa una desaceleración gradual (fade-down) año a año hacia g_term.

    Args:
        cagr_revenue_hist:  CAGR de ingresos históricos (e.g. 0.12 para 12%).
        revenue_growth_api: Crecimiento de ingresos reportado por API.
        g_term:             Tasa terminal de convergencia (e.g. 0.022).
        n_years:            Horizonte explícito (default: 5 años).

    Returns:
        Lista de tasas de crecimiento anuales para cada año $t \\in [1, n]$.
    """
    # 1. Priorización de tasa base
    if cagr_revenue_hist > 0:
        g_base_raw = cagr_revenue_hist
    elif revenue_growth_api > 0:
        g_base_raw = revenue_growth_api
    else:
        g_base_raw = 0.06

    # 2. Límites duros institucionales: [3.0%, 15.0%]
    g_base = min(max(g_base_raw, 0.030), 0.150)

    # 3. Fade-down lineal suave hacia g_term
    curva: List[float] = []
    for yr in range(1, n_years + 1):
        # En el año 1 es g_base; en el año n converge hacia g_term
        peso_terminal = (yr - 1) / max(n_years - 1, 1)
        g_yr = (g_base * (1.0 - peso_terminal * 0.70)) + (g_term * (peso_terminal * 0.70))
        curva.append(round(g_yr, 5))

    return curva


# ─────────────────────────────────────────────────────────────────────────────
# 3. MOTOR DE VALUACIÓN FCFF INSTITUCIONAL (CON MID-YEAR CONVENTION)
# ─────────────────────────────────────────────────────────────────────────────

def calcular_fcff_valuation(
    ocf_hist: List[float],
    capex_hist: List[float],
    interest_hist: List[float],
    pretax_hist: List[float],
    taxprov_hist: List[float],
    total_debt: float,
    total_cash: float,
    shares_diluted: float,
    mcap: float,
    beta: float,
    rf: float,
    precio_actual: float,
    n_years: int = 5,
    erp: float = 5.0,
    growth_rate_exp: Optional[float] = None,
    revenue_ttm: float = 0.0,
    operating_margin_hist: float = 0.0,
    cagr_revenue_hist: float = 0.0,
    revenue_growth_api: float = 0.0,
    fmp_key: str = "",
    fred_key: str = "",
    ticker: str = "",
) -> Dict[str, Union[float, str, list]]:
    """
    Motor de valuación FCFF de grado analítico con:
    1. Respaldo normalizado operativo no circular.
    2. Curva de crecimiento a 5 años estabilizada con fade-down.
    3. Tasa terminal Gordon-Shapiro restringida con spread >= 3.5% vs WACC.
    4. Motor unificado de WACC.
    5. Convención de medio año (Mid-Year Convention) para flujos explícitos.
    6. Puente institucional de reconciliación EV → Deuda Neta → Equity Value → Por Acción.
    """
    # ── Guardia de datos mínimos ─────────────────────────────────────────────
    if shares_diluted <= 0 or precio_actual <= 0:
        return _resultado_fcff_vacio(precio_actual, total_cash, total_debt, rf, beta, erp)

    # ── 1. Tasa impositiva efectiva real ─────────────────────────────────────
    tasas_impositivas: list[float] = []
    for pt, tp in zip(pretax_hist, taxprov_hist):
        if pt > 0 and tp >= 0:
            t_i = tp / pt
            if 0.0 < t_i <= 0.60:
                tasas_impositivas.append(t_i)

    tax_rate_real: float = (
        statistics.mean(tasas_impositivas) if tasas_impositivas else 0.21
    )
    tax_rate_real = max(min(tax_rate_real, 0.35), 0.0)

    # ── 2. Motor WACC unificado ──────────────────────────────────────────────
    int_exp_ultimo = interest_hist[0] if interest_hist else 0.0
    res_wacc = calcular_wacc(
        tasa_libre_riesgo = rf,
        beta              = beta,
        mcap              = mcap,
        total_debt        = total_debt,
        int_exp           = int_exp_ultimo,
        tax_rate          = tax_rate_real,
        fmp_key           = fmp_key,
        fred_key          = fred_key,
        ticker            = ticker,
        erp               = erp,
    )
    wacc = res_wacc["wacc"]
    ke = res_wacc["ke"]
    kd = res_wacc["kd"]
    we = res_wacc["we"]
    wd = res_wacc["wd"]
    wacc_decimal = wacc / 100.0

    # ── 3. FCFF histórico ────────────────────────────────────────────────────
    n_hist = len(ocf_hist)
    fcff_historico: list[float] = []
    for i in range(n_hist):
        ocf_i   = ocf_hist[i] if i < len(ocf_hist) else 0.0
        capex_i = capex_hist[i] if i < len(capex_hist) else 0.0
        int_i   = interest_hist[i] if i < len(interest_hist) else 0.0
        fcff_i  = ocf_i + (int_i * (1.0 - tax_rate_real)) - capex_i
        fcff_historico.append(fcff_i)

    # ── 4. FCFF base para proyección (sin circularidad de precio) ───────────
    fcff_reciente = fcff_historico[0] if fcff_historico else 0.0
    periodos_base = fcff_historico[:min(3, len(fcff_historico))]
    pesos = [3, 2, 1][:len(periodos_base)]
    fcff_media_pond = (
        sum(f * p for f, p in zip(periodos_base, pesos)) / sum(pesos)
        if periodos_base else 0.0
    )

    if fcff_reciente > 0:
        fcff_base = max(fcff_reciente, fcff_media_pond)
    elif fcff_media_pond > 0:
        fcff_base = fcff_media_pond
    else:
        # Respaldo FCFF normalizado basado en ingresos / margen contable
        fcff_norm_total, _ = calcular_fcff_normalizado(
            revenue_ttm           = revenue_ttm,
            operating_margin_hist = operating_margin_hist,
            tax_rate              = tax_rate_real,
            shares_diluted        = shares_diluted,
            eps_ttm               = pretax_hist[0] / shares_diluted if (pretax_hist and shares_diluted > 0) else 0.0,
        )
        if fcff_norm_total > 0:
            fcff_base = fcff_norm_total
        else:
            ocf_prom = statistics.mean(ocf_hist) if ocf_hist else 0.0
            fcff_base = max(ocf_prom * 0.05, 0.0)

    # ── 5. Restricción de Tasa Terminal g_term ───────────────────────────────
    g_term = calcular_g_term_restringido(wacc_decimal)

    # ── 6. Curva de crecimiento a 5 años (fade-down) ─────────────────────────
    if growth_rate_exp is not None and growth_rate_exp > 0:
        cagr_utilizada = growth_rate_exp
    else:
        cagr_utilizada = cagr_revenue_hist

    curva_g = calcular_curva_crecimiento_5y(
        cagr_revenue_hist  = cagr_utilizada,
        revenue_growth_api = revenue_growth_api,
        g_term             = g_term,
        n_years            = n_years,
    )

    # ── 7. Proyección y Descuento con Convención de Medio Año ───────────────
    fcff_proyectado: list[float] = []
    pv_flujos: float = 0.0
    f_actual = fcff_base

    for yr, g_yr in enumerate(curva_g, start=1):
        f_actual = f_actual * (1.0 + g_yr)
        fcff_proyectado.append(round(f_actual, 2))
        # Mid-year convention: factor = 1 / ((1 + wacc) ^ (t - 0.5))
        factor_descuento_my = 1.0 / ((1.0 + wacc_decimal) ** (yr - 0.5))
        pv_flujos += f_actual * factor_descuento_my

    # ── 8. Valor Terminal y Descuento a 5 Años Completos ────────────────────
    f_terminal = fcff_proyectado[-1] * (1.0 + g_term) if fcff_proyectado else fcff_base * (1.0 + g_term)
    denominador_tv = max(wacc_decimal - g_term, 0.035)

    tv_gordon = f_terminal / denominador_tv
    exit_multiple = min(max(1.0 / denominador_tv, 15.0), 22.0)
    tv_exit = (fcff_proyectado[-1] if fcff_proyectado else fcff_base) * exit_multiple
    valor_terminal = (tv_gordon + tv_exit) / 2.0

    # Descuento a término del horizonte explícito (5 años completos)
    pv_terminal = valor_terminal / ((1.0 + wacc_decimal) ** n_years)

    # ── 9. Puente Financiero: Enterprise Value → Equity Value → Valor/Acción ─
    enterprise_value = pv_flujos + pv_terminal
    deuda_neta = total_debt - total_cash
    equity_value = enterprise_value - deuda_neta
    valor_intrinseco = max(equity_value / shares_diluted, 0.0) if shares_diluted > 0 else 0.0

    # ── 10. Semáforo y Margen de Seguridad ───────────────────────────────────
    margen_seguridad = (
        (valor_intrinseco - precio_actual) / precio_actual
        if precio_actual > 0 else 0.0
    )
    es_atractivo = valor_intrinseco > 0 and valor_intrinseco >= precio_actual
    semaforo = "verde" if es_atractivo else "rojo"
    status = "🟢" if es_atractivo else "🔴"
    upside = margen_seguridad * 100.0

    return {
        "valor_intrinseco": round(valor_intrinseco, 2),
        "enterprise_value": round(enterprise_value, 2),
        "equity_value": round(equity_value, 2),
        "pv_flujos": round(pv_flujos, 2),
        "pv_terminal": round(pv_terminal, 2),
        "deuda_neta": round(deuda_neta, 2),
        "fcff_historico": fcff_historico,
        "fcff_proyectado": fcff_proyectado,
        "wacc": round(wacc, 2),
        "ke": round(ke, 2),
        "kd": round(kd, 2),
        "we": round(we, 4),
        "wd": round(wd, 4),
        "rf": round(rf, 2),
        "tax_rate_real": round(tax_rate_real, 4),
        "g_term": round(g_term, 4),
        "g_fase1": round(curva_g[0] if curva_g else 0.06, 4),
        "curva_g": curva_g,
        "total_cash": total_cash,
        "total_debt": total_debt,
        "shares_diluted": shares_diluted,
        "margen_seguridad": round(margen_seguridad, 4),
        "precio_actual": precio_actual,
        "status": status,
        "semaforo": semaforo,
        "upside": round(upside, 2),
    }


def _resultado_fcff_vacio(
    precio_actual: float,
    total_cash: float,
    total_debt: float,
    rf: float,
    beta: float,
    erp: float,
) -> Dict[str, Union[float, str, list]]:
    ke = rf + (beta * erp)
    return {
        "valor_intrinseco": 0.0,
        "enterprise_value": 0.0,
        "equity_value": 0.0,
        "pv_flujos": 0.0,
        "pv_terminal": 0.0,
        "deuda_neta": total_debt - total_cash,
        "fcff_historico": [],
        "fcff_proyectado": [],
        "wacc": ke,
        "ke": ke,
        "kd": 0.0,
        "we": 1.0,
        "wd": 0.0,
        "rf": rf,
        "tax_rate_real": 0.21,
        "g_term": 0.020,
        "g_fase1": 0.06,
        "curva_g": [0.06] * 5,
        "total_cash": total_cash,
        "total_debt": total_debt,
        "shares_diluted": 0.0,
        "margen_seguridad": -1.0,
        "precio_actual": precio_actual,
        "status": "🔴",
        "semaforo": "rojo",
        "upside": -100.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. DCF SIMPLIFICADO Y MATRIZ DE SENSIBILIDAD
# ─────────────────────────────────────────────────────────────────────────────

def calcular_dcf_intr_ps(
    wacc_var: float,
    g_term_var: float,
    flujo_por_accion: float,
    g_1_5: float,
    precio_actual: float,
    eps_ttm: float,
    total_cash: float,
    total_debt: float,
    shares_current: float,
) -> Dict[str, Union[float, str]]:
    """
    Cálculo simplificado de Descuento de Flujos (DCF) por acción para matrices
    de sensibilidad y respaldos de scoring, usando Mid-Year Convention y
    restricción terminal sin circularidad con el precio.
    """
    wacc_dec = (wacc_var / 100.0) if wacc_var > 1.0 else wacc_var
    g_term_eff = calcular_g_term_restringido(wacc_dec) if g_term_var > 0.035 else max(g_term_var, 0.015)
    g_base = min(max(g_1_5, 0.030), 0.150)

    # Curva de crecimiento con fade-down
    curva_g = calcular_curva_crecimiento_5y(g_base, g_base, g_term_eff, n_years=5)

    pv_flujos = 0.0
    f_ps = max(flujo_por_accion, 0.0)
    for yr, g_yr in enumerate(curva_g, start=1):
        f_ps *= (1.0 + g_yr)
        pv_flujos += f_ps / ((1.0 + wacc_dec) ** (yr - 0.5))

    denominador_tv = max(wacc_dec - g_term_eff, 0.035)
    tv_gordon = (f_ps * (1.0 + g_term_eff)) / denominador_tv
    exit_multiple = min(max(1.0 / denominador_tv, 15.0), 22.0)
    tv_exit = f_ps * exit_multiple
    tv_hibrido = (tv_gordon + tv_exit) / 2.0

    pv_terminal = tv_hibrido / ((1.0 + wacc_dec) ** 5)
    ev_ps = pv_flujos + pv_terminal
    caja_neta_ps = ((total_cash - total_debt) / shares_current) if shares_current > 0 else 0.0
    v_calc = max(ev_ps + caja_neta_ps, 0.0)

    es_atractivo = v_calc > 0 and v_calc >= precio_actual
    semaforo = "verde" if es_atractivo else "rojo"
    status = "🟢" if es_atractivo else "🔴"
    upside = (((v_calc - precio_actual) / precio_actual) * 100) if precio_actual > 0 else 0.0

    return {
        "valor_intrinseco": round(v_calc, 2),
        "status": status,
        "semaforo": semaforo,
        "upside": round(upside, 2),
    }


def crear_calculador_dcf(
    flujo_por_accion: float,
    g_1_5: float,
    precio_actual: float,
    eps_ttm: float,
    total_cash: float,
    total_debt: float,
    shares_current: float,
) -> Callable[[float, float], float]:
    """
    Factory que retorna una función ``calculador(wacc_var, g_term_var) -> float``
    utilizada por la matriz de sensibilidad multiescenario del reporte PDF.
    """
    def calculador(wacc_var: float, g_term_var: float) -> float:
        res = calcular_dcf_intr_ps(
            wacc_var         = wacc_var,
            g_term_var       = g_term_var,
            flujo_por_accion = flujo_por_accion,
            g_1_5            = g_1_5,
            precio_actual    = precio_actual,
            eps_ttm          = eps_ttm,
            total_cash       = total_cash,
            total_debt       = total_debt,
            shares_current   = shares_current,
        )
        return float(res["valor_intrinseco"])
    return calculador


# ─────────────────────────────────────────────────────────────────────────────
# 5. MODELO DE DESCUENTO DE DIVIDENDOS (DDM / GORDON GROWTH)
# ─────────────────────────────────────────────────────────────────────────────

def calcular_ddm(
    div_rate: float,
    ke: float,
    g_div: float,
    precio_actual: float = 0.0,
) -> Dict[str, Union[float, str]]:
    """
    Calcula el Valor Intrínseco mediante el Modelo Gordon Growth (DDM)
    para emisoras con dividendos significativos o FIBRAs/REITs.
    """
    ke_dec = (ke / 100.0) if ke > 1.0 else ke
    g_div_eff = min(max(g_div, 0.015), 0.040)
    denominador = ke_dec - g_div_eff

    v_intr_ddm = (
        (div_rate * (1.0 + g_div_eff)) / denominador
        if (div_rate > 0 and denominador > 0.015) else 0.0
    )
    val_ddm_str = f"${v_intr_ddm:,.2f}" if v_intr_ddm > 0 else "N/A"

    if v_intr_ddm == 0:
        semaforo, status = "gris", "⚪"
    elif v_intr_ddm >= precio_actual:
        semaforo, status = "verde", "🟢"
    else:
        semaforo, status = "rojo", "🔴"

    return {
        "valor_intrinseco_ddm": round(v_intr_ddm, 2),
        "val_ddm_str": val_ddm_str,
        "status": status,
        "semaforo": semaforo,
    }
