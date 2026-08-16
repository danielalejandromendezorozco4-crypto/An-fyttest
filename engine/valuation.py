from __future__ import annotations

import statistics
from typing import Callable, Dict, List, Optional, Tuple, Union

# NOTE: Imports de módulos con dependencia de streamlit (financial_fetcher,
# config.settings) se hacen de forma diferida (lazy) dentro de cada función
# para evitar arrastrar streamlit al nivel de módulo. Esto previene
# ImportError en pytest y en Streamlit Cloud.


# ─────────────────────────────────────────────────────────────────────────────
# 1. CÁLCULO DEL WACC Y ESTRUCTURA DE CAPITAL
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
    Motor unificado para el cálculo del Costo Promedio Ponderado de Capital (WACC):
    - Tasa Libre de Riesgo (Rf): obtenida de FRED (DGS10) o 4.20% por defecto.
    - Beta: beta del ticker (1.0 por defecto).
    - Equity Risk Premium (ERP): prima observada o spread S&P 500 - Rf.
    - Coste del Equity (Ke): Ke = Rf + (Beta * ERP).
    - Ponderaciones:
        Total Capital = Market Cap + Total Debt
        We = Market Cap / Total Capital (si Total Capital > 0 sino 1.0)
        Wd = Total Debt / Total Capital (si Total Capital > 0 sino 0.0)
    - Coste de la Deuda (Kd):
        Si Total Debt > 0 e Interest Expense > 0: Kd_real = (Interest Expense / Total Debt) * 100
        Si no, consultar serie BAMLC0A0CM en FRED o 5.50%
        Kd = min(max(Kd_real, Rf), 15.0)
    - WACC Final:
        WACC_calculado = (We * Ke) + (Wd * Kd * (1 - Tax_Rate))
        Acotar WACC entre 7.5% y 15.0%: WACC = max(min(WACC_calculado, 15.0), 7.5)

    Returns:
        Diccionario tipado con wacc, ke, kd, we, wd, rf, erp, tax_rate.
    """
    rf = max(float(tasa_libre_riesgo) if tasa_libre_riesgo is not None else 4.20, 0.0)
    beta_val = max(float(beta) if beta is not None else 1.0, 0.1)
    erp_val = max(float(erp) if erp is not None else 5.0, 3.0)
    t_ef = max(min(float(tax_rate) if tax_rate is not None else 0.21, 0.35), 0.0)

    # 1. Coste del Equity (Ke por CAPM)
    ke = rf + (beta_val * erp_val)

    # 2. Ponderaciones de capital
    total_capital = mcap + total_debt
    if total_capital > 0:
        we = mcap / total_capital
        wd = total_debt / total_capital
    else:
        we, wd = 1.0, 0.0

    # 3. Coste de la Deuda (Kd)
    if total_debt > 0:
        if int_exp > 0:
            kd_real = (int_exp / total_debt) * 100.0
        elif ticker and (fmp_key or fred_key):
            try:
                from data.financial_fetcher import obtener_kd_fmp_fred  # noqa: PLC0415
                kd_fmp = obtener_kd_fmp_fred(ticker, fmp_key, fred_key, int_exp, total_debt)
                kd_real = float(kd_fmp)
            except Exception:
                kd_real = 5.50
        else:
            kd_real = 5.50
        kd = min(max(kd_real, rf), 15.0)
    else:
        kd = 0.0

    # 4. WACC Final acotado en [7.5%, 15.0%]
    wacc_calculado = (we * ke) + (wd * kd * (1.0 - t_ef))
    wacc = max(min(wacc_calculado, 15.0), 7.5)

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
# 2. DETERMINACIÓN DE TASAS DE CRECIMIENTO Y FLUJO BASE POR ACCIÓN
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
    Calcula el FCFF Operativo Normalizado:
    FCFF Normalizado = (Revenue_TTM * max(Margen_Operativo_Historico, 0.10) * (1 - Tax_Rate))
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
    Tasa terminal (g_term):
    g_term = 0.025 si 0.025 < (WACC / 100) sino (WACC / 100) - 0.015.
    """
    if 0.025 < wacc_decimal:
        return 0.025
    return max(wacc_decimal - 0.015, 0.010)


def calcular_curva_crecimiento_5y(
    cagr_revenue_hist: float,
    revenue_growth_api: float = 0.0,
    g_term: float = 0.025,
    n_years: int = 5,
) -> List[float]:
    """
    g_1_5 = min(max(crecimiento * 0.85, 0.04), 0.18)
    """
    crecimiento = cagr_revenue_hist if cagr_revenue_hist > 0 else (revenue_growth_api if revenue_growth_api > 0 else 0.08)
    g_1_5 = min(max(crecimiento * 0.85, 0.04), 0.18)
    return [round(g_1_5, 5)] * n_years


# ─────────────────────────────────────────────────────────────────────────────
# 3. MOTOR DE VALUACIÓN DCF HÍBRIDO (POR ACCIÓN)
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
    Motor de Valuación DCF Híbrido por Acción:
    1. Proyección y Descuento de Flujos (Años 1 a 5):
       f_ps = flujo_por_accion
       Para cada año i de 1 a 5:
           f_ps = f_ps * (1 + g_1_5)
           pv_flujos += f_ps / ((1 + (wacc_var / 100)) ** i)
    2. Valor Terminal Híbrido (Gordon Growth + Múltiplo de Salida P/E):
       tv_gordon = (f_ps * (1 + g_term_var)) / ((wacc_var / 100) - g_term_var) si (wacc_var / 100) > g_term_var sino 0
       pe_dinamico_actual = (precio_actual / eps_ttm) si eps_ttm > 0 sino 15
       terminal_pe = max(min(pe_dinamico_actual * 0.75, 20.0), 10.0)
       tv_multiplo = f_ps * terminal_pe
       tv_hibrido = (tv_gordon + tv_multiplo) / 2.
       pv_terminal = tv_hibrido / ((1 + (wacc_var / 100)) ** 5)
    3. Ajuste por Caja Neta por Acción:
       caja_neta_por_accion = (Total Cash - Total Debt) / Shares Outstanding
       valor_calculado = pv_flujos + pv_terminal + caja_neta_por_accion
       Si valor_calculado > 0 devolver valor_calculado, de lo contrario devolver (precio_actual * 0.85).
    """
    wacc_dec = (wacc_var / 100.0) if wacc_var > 1.0 else wacc_var

    # 1. Proyección y Descuento de Flujos (Años 1 a 5)
    pv_flujos = 0.0
    f_ps = float(flujo_por_accion)
    for i in range(1, 6):
        f_ps = f_ps * (1.0 + g_1_5)
        pv_flujos += f_ps / ((1.0 + wacc_dec) ** i)

    # 2. Valor Terminal Híbrido
    if wacc_dec > g_term_var:
        tv_gordon = (f_ps * (1.0 + g_term_var)) / (wacc_dec - g_term_var)
    else:
        tv_gordon = 0.0

    pe_dinamico_actual = (precio_actual / eps_ttm) if eps_ttm > 0 else 15.0
    terminal_pe = max(min(pe_dinamico_actual * 0.75, 20.0), 10.0)
    tv_multiplo = f_ps * terminal_pe
    tv_hibrido = (tv_gordon + tv_multiplo) / 2.0
    pv_terminal = tv_hibrido / ((1.0 + wacc_dec) ** 5)

    # 3. Ajuste por Caja Neta por Acción
    caja_neta_por_accion = ((total_cash - total_debt) / shares_current) if shares_current > 0 else 0.0
    valor_calculado = pv_flujos + pv_terminal + caja_neta_por_accion

    if valor_calculado > 0:
        valor_final = valor_calculado
    else:
        valor_final = precio_actual * 0.85

    es_atractivo = valor_final >= precio_actual
    semaforo = "verde" if es_atractivo else "rojo"
    status = "🟢" if es_atractivo else "🔴"
    upside = (((valor_final - precio_actual) / precio_actual) * 100.0) if precio_actual > 0 else 0.0

    return {
        "valor_intrinseco": round(valor_final, 2),
        "pv_flujos": round(pv_flujos, 2),
        "pv_terminal": round(pv_terminal, 2),
        "caja_neta_por_accion": round(caja_neta_por_accion, 2),
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
    Retorna una función calculador(wacc_var, g_term_var) para matrices de sensibilidad.
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
# 4. MODELO DE DESCUENTO DE DIVIDENDOS (DDM GORDON GROWTH)
# ─────────────────────────────────────────────────────────────────────────────

def calcular_ddm(
    div_rate: float,
    ke: float,
    g_div: float,
    precio_actual: float = 0.0,
) -> Dict[str, Union[float, str]]:
    """
    Modelo Gordon Growth (DDM):
    - g_div = min(max(g_1_5 * 0.5, 0.02), 0.04)
    - v_intr_ddm = (Dividend Rate * (1 + g_div)) / ((Ke / 100) - g_div) si (Dividend Rate > 0 y (Ke / 100) > g_div) sino 0.
    """
    ke_dec = (ke / 100.0) if ke > 1.0 else ke
    g_div_eff = min(max(g_div, 0.02), 0.04)

    if div_rate > 0 and ke_dec > g_div_eff:
        v_intr_ddm = (div_rate * (1.0 + g_div_eff)) / (ke_dec - g_div_eff)
    else:
        v_intr_ddm = 0.0

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


# ─────────────────────────────────────────────────────────────────────────────
# 5. MOTOR FCFF INSTITUCIONAL
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
    Motor FCFF integrado con WACC, DCF híbrido por acción y puente EV -> Equity.
    """
    if shares_diluted <= 0 or precio_actual <= 0:
        return _resultado_fcff_vacio(precio_actual, total_cash, total_debt, rf, beta, erp)

    # 1. Tasa impositiva efectiva real
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

    # 2. WACC unificado
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

    # 3. FCFF histórico
    n_hist = len(ocf_hist)
    fcff_historico: list[float] = []
    for i in range(n_hist):
        ocf_i   = ocf_hist[i] if i < len(ocf_hist) else 0.0
        capex_i = capex_hist[i] if i < len(capex_hist) else 0.0
        int_i   = interest_hist[i] if i < len(interest_hist) else 0.0
        fcff_i  = ocf_i + (int_i * (1.0 - tax_rate_real)) - capex_i
        fcff_historico.append(fcff_i)

    # 4. Flujo base por acción y crecimiento
    crecimiento = growth_rate_exp if (growth_rate_exp and growth_rate_exp > 0) else (
        cagr_revenue_hist if cagr_revenue_hist > 0 else (revenue_growth_api if revenue_growth_api > 0 else 0.08)
    )
    g_1_5 = min(max(crecimiento * 0.85, 0.04), 0.18)
    g_term = 0.025 if 0.025 < wacc_decimal else max(wacc_decimal - 0.015, 0.010)

    fcf_base = fcff_historico[0] if fcff_historico else 0.0
    fcf_base_per_share = (fcf_base / shares_diluted) if shares_diluted > 0 else 0.0
    eps_ttm = (pretax_hist[0] * (1.0 - tax_rate_real) / shares_diluted) if (pretax_hist and shares_diluted > 0) else 0.0
    flujo_por_accion = max(fcf_base_per_share, eps_ttm * 0.85) or (precio_actual * 0.035)

    # 5. Cálculo DCF Híbrido por Acción
    res_dcf = calcular_dcf_intr_ps(
        wacc_var         = wacc,
        g_term_var       = g_term,
        flujo_por_accion = flujo_por_accion,
        g_1_5            = g_1_5,
        precio_actual    = precio_actual,
        eps_ttm          = eps_ttm,
        total_cash       = total_cash,
        total_debt       = total_debt,
        shares_current   = shares_diluted,
    )
    valor_intrinseco = float(res_dcf["valor_intrinseco"])

    # 6. Proyección total de flujos para gráficos
    fcff_proyectado = []
    f_p = flujo_por_accion * shares_diluted
    for _ in range(5):
        f_p *= (1.0 + g_1_5)
        fcff_proyectado.append(round(f_p, 2))

    deuda_neta = total_debt - total_cash
    equity_value = valor_intrinseco * shares_diluted
    enterprise_value = equity_value + deuda_neta

    margen_seguridad = ((valor_intrinseco - precio_actual) / precio_actual) if precio_actual > 0 else 0.0
    es_atractivo = valor_intrinseco >= precio_actual
    semaforo = "verde" if es_atractivo else "rojo"
    status = "🟢" if es_atractivo else "🔴"
    upside = margen_seguridad * 100.0

    return {
        "valor_intrinseco": round(valor_intrinseco, 2),
        "enterprise_value": round(enterprise_value, 2),
        "equity_value": round(equity_value, 2),
        "pv_flujos": float(res_dcf["pv_flujos"]),
        "pv_terminal": float(res_dcf["pv_terminal"]),
        "deuda_neta": round(deuda_neta, 2),
        "fcff_historico": fcff_historico,
        "fcff_proyectado": fcff_proyectado,
        "wacc": round(wacc, 2),
        "ke": round(ke, 2),
        "kd": round(kd, 2),
        "we": round(we, 4),
        "wd": round(wd, 4),
        "rf": round(rf, 2),
        "erp": round(erp, 2),
        "tax_rate_real": round(tax_rate_real, 4),
        "g_term": round(g_term, 4),
        "g_fase1": round(g_1_5, 4),
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
        "erp": erp,
        "tax_rate_real": 0.21,
        "g_term": 0.020,
        "g_fase1": 0.06,
        "total_cash": total_cash,
        "total_debt": total_debt,
        "shares_diluted": 0.0,
        "margen_seguridad": -1.0,
        "precio_actual": precio_actual,
        "status": "🔴",
        "semaforo": "rojo",
        "upside": -100.0,
    }
