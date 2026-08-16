from __future__ import annotations

import math
import statistics
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


# ─────────────────────────────────────────────────────────────────────────────
# 0. UTILITARIOS NUMÉRICOS DEFENSIVOS
# ─────────────────────────────────────────────────────────────────────────────

def safe_num(val: Any, default: float = 0.0) -> float:
    """
    Convierte cualquier valor de forma segura a float o al valor por defecto especificado.
    Maneja None, np.nan, float('nan'), inf, -inf, strings no numéricos y tipos corruptos.
    """
    if val is None:
        return float(default) if default is not None else 0.0
    try:
        if isinstance(val, (int, float)):
            if math.isnan(val) or math.isinf(val):
                return float(default) if default is not None else 0.0
            return float(val)
        if isinstance(val, str):
            clean_str = val.replace(',', '').replace('$', '').replace('%', '').strip()
            if not clean_str or clean_str.lower() in ('nan', 'none', 'n/a', 'null', 'inf', '-inf'):
                return float(default) if default is not None else 0.0
            return float(clean_str)
        return float(val)
    except (ValueError, TypeError, Exception):
        return float(default) if default is not None else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 1. CÁLCULO DEL WACC CON DATOS REALES DE MERCADO
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
    Cálculo del Costo Promedio Ponderado de Capital (WACC) con datos reales de mercado:
    - Tasa Libre de Riesgo (Rf): Rendimiento del Bono del Tesoro a 10 años (FRED: DGS10 / ~4.20%).
    - Beta: Beta reportado por la API frente al S&P 500.
    - Prima de Riesgo de Mercado (ERP): Rentabilidad esperada del mercado - Rf.
    - Costo del Capital Propio (Ke): Ke = Rf + (Beta * ERP).
    - Costo de la Deuda (Kd):
        Kd_real = (Interest_Expense / Total_Debt) * 100 si Total_Debt > 0 e Interest_Expense > 0.
        Si no hay deuda o no reporta intereses, Kd = Rf.
    - Ponderaciones a Valor de Mercado:
        Total_Capital = Market_Cap + Total_Debt
        We = Market_Cap / Total_Capital (si Total_Capital > 0 sino 1.0)
        Wd = Total_Debt / Total_Capital (si Total_Capital > 0 sino 0.0)
    - WACC = (We * Ke) + (Wd * Kd * (1 - Tax_Rate)) (sin pisos ni techos arbitrarios).

    Returns:
        Diccionario tipado con wacc, ke, kd, we, wd, rf, erp, tax_rate.
    """
    rf = max(float(tasa_libre_riesgo) if tasa_libre_riesgo is not None else 4.20, 0.0)
    beta_val = max(float(beta) if beta is not None else 1.0, 0.05)
    erp_val = max(float(erp) if erp is not None else 5.0, 2.0)
    t_ef = max(min(float(tax_rate) if tax_rate is not None else 0.21, 0.35), 0.0)

    # 1. Costo del Capital Propio (Ke por CAPM)
    ke = rf + (beta_val * erp_val)

    # 2. Ponderaciones a valor de mercado
    total_capital = mcap + total_debt
    if total_capital > 0:
        we = mcap / total_capital
        wd = total_debt / total_capital
    else:
        we, wd = 1.0, 0.0

    # 3. Costo de la Deuda (Kd real)
    if total_debt > 0:
        if int_exp > 0:
            kd = (int_exp / total_debt) * 100.0
        elif ticker and (fmp_key or fred_key):
            try:
                from data.financial_fetcher import obtener_kd_fmp_fred  # noqa: PLC0415
                kd_fmp = obtener_kd_fmp_fred(ticker, fmp_key, fred_key, int_exp, total_debt)
                kd = float(kd_fmp)
            except Exception:
                kd = rf
        else:
            kd = rf
    else:
        kd = rf

    # 4. WACC real
    wacc_real = (we * ke) + (wd * kd * (1.0 - t_ef))
    wacc = max(wacc_real, 1.0)

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
# 2. MODELO 1: DCF MULTIETAPA FCFF (FREE CASH FLOW TO FIRM)
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
    Tasa terminal (g_term): 2.5% anual alineada al PIB global de largo plazo.
    Asegura que g_term < WACC_decimal con spread mínimo de seguridad.
    """
    g_term = 0.025
    if wacc_decimal <= g_term:
        g_term = max(wacc_decimal - 0.015, 0.010)
    return round(g_term, 4)


def calcular_curva_crecimiento_5y(
    cagr_revenue_hist: float,
    revenue_growth_api: float = 0.0,
    g_term: float = 0.025,
    n_years: int = 5,
) -> List[float]:
    """
    Determina la tasa de crecimiento proyectada para los años 1 al 5:
    - Prioridad 1: revenue_growth o earnings_growth positivos reportados por la API.
    - Prioridad 2: CAGR histórico de 3 a 5 años de ingresos/EBITDA.
    - Fallback: 6.0% - 8.0%.
    """
    if revenue_growth_api > 0:
        g_1_5 = revenue_growth_api
    elif cagr_revenue_hist > 0:
        g_1_5 = cagr_revenue_hist
    else:
        g_1_5 = 0.08

    # Acotar en rango realista institucional [2.0%, 25.0%]
    g_1_5_clamped = min(max(g_1_5, 0.02), 0.25)
    return [round(g_1_5_clamped, 5)] * n_years


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
    Motor FCFF Institucional con datos financieros reales y convención de medio año.
    """
    if shares_diluted <= 0 or precio_actual <= 0:
        return _resultado_fcff_vacio(precio_actual, total_cash, total_debt, rf, beta, erp)

    # 1. Tasa Impositiva Efectiva Real
    tasas_impositivas: list[float] = []
    for pt, tp in zip(pretax_hist, taxprov_hist):
        if pt > 0 and tp >= 0:
            t_i = tp / pt
            if 0.0 <= t_i <= 0.60:
                tasas_impositivas.append(t_i)

    tax_rate_real: float = (
        statistics.mean(tasas_impositivas) if tasas_impositivas else 0.21
    )
    tax_rate_real = max(min(tax_rate_real, 0.35), 0.0)

    # 2. Cálculo del WACC con datos reales
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

    # 3. FCFF Histórico Real (TTM y períodos previos)
    fcff_historico: list[float] = []
    for i in range(len(ocf_hist)):
        cfo_i   = ocf_hist[i] if i < len(ocf_hist) else 0.0
        capex_i = abs(capex_hist[i]) if i < len(capex_hist) else 0.0
        int_i   = interest_hist[i] if i < len(interest_hist) else 0.0
        escudo_fiscal_i = int_i * (1.0 - tax_rate_real)
        fcff_i  = cfo_i + escudo_fiscal_i - capex_i
        fcff_historico.append(fcff_i)

    # 4. FCFF Base para la Proyección
    if fcff_historico and fcff_historico[0] > 0:
        fcff_base = fcff_historico[0]
    else:
        fcff_norm_tot, _ = calcular_fcff_normalizado(
            revenue_ttm           = revenue_ttm,
            operating_margin_hist = operating_margin_hist,
            tax_rate              = tax_rate_real,
            shares_diluted        = shares_diluted,
            eps_ttm               = pretax_hist[0] * (1.0 - tax_rate_real) / shares_diluted if (pretax_hist and shares_diluted > 0) else 0.0,
        )
        if fcff_norm_tot > 0:
            fcff_base = fcff_norm_tot
        else:
            fcff_base = max(ocf_hist[0] * 0.50 if ocf_hist else 0.0, 1e6)

    # 5. Tasa de Crecimiento a 5 Años (g_1_5)
    if growth_rate_exp is not None and growth_rate_exp > 0:
        g_1_5 = growth_rate_exp
    elif revenue_growth_api > 0:
        g_1_5 = revenue_growth_api
    elif cagr_revenue_hist > 0:
        g_1_5 = cagr_revenue_hist
    else:
        g_1_5 = 0.08
    g_1_5 = min(max(g_1_5, 0.02), 0.25)

    # 6. Tasa Terminal (g_term = 2.5%)
    g_term = 0.025
    if wacc_decimal <= g_term:
        g_term = max(wacc_decimal - 0.015, 0.010)

    # 7. Proyección y Descuento con Convención de Medio Año
    fcff_proyectado: list[float] = []
    pv_flujos: float = 0.0
    f_t = fcff_base

    for t in range(1, n_years + 1):
        f_t = f_t * (1.0 + g_1_5)
        fcff_proyectado.append(round(f_t, 2))
        factor_descuento_my = (1.0 + wacc_decimal) ** (t - 0.5)
        pv_flujos += f_t / factor_descuento_my

    # 8. Valor Terminal de Gordon Shapiro
    f_terminal = fcff_proyectado[-1] * (1.0 + g_term)
    denominador_tv = max(wacc_decimal - g_term, 0.010)
    terminal_value = f_terminal / denominador_tv
    pv_terminal = terminal_value / ((1.0 + wacc_decimal) ** n_years)

    # 9. Puente Financiero: Enterprise Value → Equity Value → Por Acción
    enterprise_value = pv_flujos + pv_terminal
    deuda_neta = total_debt - total_cash
    equity_value = enterprise_value - deuda_neta
    valor_intrinseco = max(equity_value / shares_diluted, 0.0) if shares_diluted > 0 else 0.0

    # 10. Margen de Seguridad y Semáforo
    margen_seguridad = (
        (valor_intrinseco - precio_actual) / precio_actual
        if precio_actual > 0 else 0.0
    )
    es_atractivo = valor_intrinseco >= precio_actual
    semaforo = "verde" if es_atractivo else "rojo"
    status = "🟢" if es_atractivo else "🔴"
    upside = margen_seguridad * 100.0

    return {
        "valor_intrinseco": round(valor_intrinseco, 2),
        "enterprise_value": round(enterprise_value, 2),
        "equity_value": round(equity_value, 2),
        "pv_flujos": round(pv_flujos, 2),
        "pv_terminal": round(pv_terminal, 2),
        "terminal_value": round(terminal_value, 2),
        "deuda_neta": round(deuda_neta, 2),
        "fcff_base": round(fcff_base, 2),
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
        "terminal_value": 0.0,
        "deuda_neta": total_debt - total_cash,
        "fcff_base": 0.0,
        "fcff_historico": [],
        "fcff_proyectado": [],
        "wacc": ke,
        "ke": ke,
        "kd": rf,
        "we": 1.0,
        "wd": 0.0,
        "rf": rf,
        "erp": erp,
        "tax_rate_real": 0.21,
        "g_term": 0.025,
        "g_fase1": 0.08,
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
# 3. MODELO 2: MODELO DE DESCUENTO DE DIVIDENDOS (DDM / GORDON GROWTH)
# ─────────────────────────────────────────────────────────────────────────────

def calcular_ddm(
    div_rate: float,
    ke: float,
    g_div: float,
    precio_actual: float = 0.0,
) -> Dict[str, Union[float, str, bool]]:
    """
    Modelo Gordon Growth para Dividendos (DDM):
    v_intr_ddm = (Dividend_Rate * (1 + g_div)) / (Ke_dec - g_div)
    """
    ke_dec = (ke / 100.0) if ke > 1.0 else ke
    g_div_eff = min(max(g_div, 0.015), 0.04)

    if div_rate > 0 and ke_dec > g_div_eff:
        v_intr_ddm = (div_rate * (1.0 + g_div_eff)) / (ke_dec - g_div_eff)
        viable = True
    else:
        v_intr_ddm = 0.0
        viable = False

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
        "viable": viable,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. MODELO 3: MÚLTIPLOS HISTÓRICOS Y RELATIVOS NORMALIZADOS
# ─────────────────────────────────────────────────────────────────────────────

def calcular_valoracion_multiplos(
    precio_actual: float,
    mcap: float,
    eps_ttm: float,
    forward_eps: float,
    fcf_ttm: float,
    ebitda_ttm: float,
    total_debt: float,
    total_cash: float,
    total_equity: float,
    shares_diluted: float,
    sector: str = "Technology",
    benchmark_pe: float = 20.0,
    benchmark_pfcf: float = 18.0,
    benchmark_ev_ebitda: float = 14.0,
    benchmark_pb: float = 3.5,
) -> Dict[str, Union[float, str, bool]]:
    """
    Estima el valor intrínseco mediante múltiplos de mercado normalizados:
    1. PER Normalizado: Target_PE * EPS_Normalizado.
    2. P/FCF Normalizado: Target_PFCF * FCF_por_accion.
    3. EV/EBITDA Normalizado: (Target_EV_EBITDA * EBITDA - Net_Debt) / Shares.
    4. P/B Normalizado (clave para sector financiero / inmobiliario).
    """
    if shares_diluted <= 0:
        return {"valor_intrinseco_multiplos": 0.0, "viable": False, "status": "⚪"}

    valores_candidatos: list[tuple[float, float]] = []  # (valor, peso)

    # 1. Componente PER Normalizado
    eps_base = max(eps_ttm, forward_eps * 0.90 if forward_eps > 0 else 0.0)
    if eps_base > 0:
        pe_target = min(max(benchmark_pe, 12.0), 32.0)
        v_pe = eps_base * pe_target
        valores_candidatos.append((v_pe, 0.40))

    # 2. Componente P/FCF Normalizado
    fcf_ps = (fcf_ttm / shares_diluted) if fcf_ttm > 0 else 0.0
    if fcf_ps > 0:
        pfcf_target = min(max(benchmark_pfcf, 10.0), 25.0)
        v_pfcf = fcf_ps * pfcf_target
        valores_candidatos.append((v_pfcf, 0.35))

    # 3. Componente EV/EBITDA
    if ebitda_ttm > 0:
        ev_target = min(max(benchmark_ev_ebitda, 8.0), 20.0)
        ev_est = ebitda_ttm * ev_target
        deuda_neta = total_debt - total_cash
        equity_est = max(ev_est - deuda_neta, 0.0)
        v_evebitda = equity_est / shares_diluted
        valores_candidatos.append((v_evebitda, 0.25))

    # 4. Componente P/B para financieras o REITs
    if sector in ("Financial Services", "Real Estate") and total_equity > 0:
        bvps = total_equity / shares_diluted
        pb_target = min(max(benchmark_pb, 1.0), 4.5)
        v_pb = bvps * pb_target
        valores_candidatos.append((v_pb, 0.50))

    if not valores_candidatos:
        return {"valor_intrinseco_multiplos": 0.0, "viable": False, "status": "⚪"}

    suma_pesos = sum(p for _, p in valores_candidatos)
    v_multiplos = sum(v * p for v, p in valores_candidatos) / suma_pesos

    return {
        "valor_intrinseco_multiplos": round(v_multiplos, 2),
        "viable": True,
        "status": "🟢" if v_multiplos >= precio_actual else "🔴",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. MOTOR ADAPTATIVO MULTIMODELO (COMPOSITE INTRINSIC VALUE)
# ─────────────────────────────────────────────────────────────────────────────

def calcular_valuacion_adaptativa(
    # Métricas cuantitativas
    precio_actual: float,
    mcap: float,
    shares_diluted: float,
    total_debt: float,
    total_cash: float,
    beta: float,
    rf: float,
    erp: float,
    # Estados financieros
    ocf_hist: List[float],
    capex_hist: List[float],
    interest_hist: List[float],
    pretax_hist: List[float],
    taxprov_hist: List[float],
    revenue_ttm: float,
    gross_profit_ttm: float,
    operating_income_ttm: float,
    net_income_ttm: float,
    ebitda_ttm: float,
    fcf_ttm: float,
    total_equity: float,
    eps_ttm: float,
    forward_eps: float,
    # Dividendos y crecimiento
    div_rate: float,
    div_yield: float,
    growth_rate_exp: Optional[float] = None,
    cagr_revenue_hist: float = 0.0,
    revenue_growth_api: float = 0.0,
    earnings_growth_api: float = 0.0,
    # Contexto sectorial
    sector: str = "Technology",
    benchmark_pe: float = 20.0,
    benchmark_pfcf: float = 18.0,
    benchmark_ev_ebitda: float = 14.0,
    benchmark_pb: float = 3.5,
    # Credenciales opcionales
    fmp_key: str = "",
    fred_key: str = "",
    ticker: str = "",
) -> Dict[str, Any]:
    """
    Motor Adaptativo Multimodelo de Valuación Institucional:
    1. Ejecuta los 3 modelos clave: DCF FCFF, DDM Gordon Growth y Múltiplos Normalizados.
    2. Clasifica el perfil de la empresa (Madura dividendo, Crecimiento sólido, Reinversión intensiva, Financiera).
    3. Asigna ponderaciones dinámicas según viabilidad y calidad de datos.
    4. Genera la matriz de escenarios (Pesimista - Base - Optimista) y margen de seguridad.
    """
    if shares_diluted <= 0 or precio_actual <= 0:
        return _resultado_adaptativo_vacio(precio_actual, total_cash, total_debt, rf, beta, erp)

    # ── A. Modelo 1: DCF FCFF Multietapa ─────────────────────────────────────
    crecimiento_base = growth_rate_exp if (growth_rate_exp and growth_rate_exp > 0) else (
        revenue_growth_api if revenue_growth_api > 0 else (
            earnings_growth_api if earnings_growth_api > 0 else (
                cagr_revenue_hist if cagr_revenue_hist > 0 else 0.08
            )
        )
    )
    g_1_5 = min(max(crecimiento_base, 0.02), 0.25)

    res_fcff = calcular_fcff_valuation(
        ocf_hist              = ocf_hist,
        capex_hist            = capex_hist,
        interest_hist         = interest_hist,
        pretax_hist           = pretax_hist,
        taxprov_hist          = taxprov_hist,
        total_debt            = total_debt,
        total_cash            = total_cash,
        shares_diluted        = shares_diluted,
        mcap                  = mcap,
        beta                  = beta,
        rf                    = rf,
        precio_actual         = precio_actual,
        erp                   = erp,
        growth_rate_exp       = g_1_5,
        revenue_ttm           = revenue_ttm,
        operating_margin_hist = (operating_income_ttm / revenue_ttm) if revenue_ttm > 0 else 0.15,
        cagr_revenue_hist     = cagr_revenue_hist,
        revenue_growth_api    = revenue_growth_api,
        fmp_key               = fmp_key,
        fred_key              = fred_key,
        ticker                = ticker,
    )

    val_dcf = float(res_fcff["valor_intrinseco"])
    dcf_viable = val_dcf > 0 and (fcf_ttm > 0 or (ocf_hist and ocf_hist[0] > 0))

    # ── B. Modelo 2: DDM Gordon Growth ───────────────────────────────────────
    ke = float(res_fcff["ke"])
    wacc = float(res_fcff["wacc"])
    g_term = float(res_fcff["g_term"])
    g_div = min(max(g_1_5 * 0.5, 0.015), 0.035)

    res_ddm = calcular_ddm(div_rate=div_rate, ke=ke, g_div=g_div, precio_actual=precio_actual)
    val_ddm = float(res_ddm["valor_intrinseco_ddm"])
    ddm_viable = bool(res_ddm["viable"]) and div_rate > 0 and (div_yield >= 1.0 or sector in ("Financial Services", "Real Estate", "Utilities", "Consumer Defensive"))

    # ── C. Modelo 3: Múltiplos Históricos Normalizados ───────────────────────
    res_mult = calcular_valoracion_multiplos(
        precio_actual       = precio_actual,
        mcap                = mcap,
        eps_ttm             = eps_ttm,
        forward_eps         = forward_eps,
        fcf_ttm             = fcf_ttm,
        ebitda_ttm          = ebitda_ttm,
        total_debt          = total_debt,
        total_cash          = total_cash,
        total_equity        = total_equity,
        shares_diluted      = shares_diluted,
        sector              = sector,
        benchmark_pe        = benchmark_pe,
        benchmark_pfcf      = benchmark_pfcf,
        benchmark_ev_ebitda = benchmark_ev_ebitda,
        benchmark_pb        = benchmark_pb,
    )
    val_mult = float(res_mult["valor_intrinseco_multiplos"])
    mult_viable = bool(res_mult["viable"]) and val_mult > 0

    # ── D. Clasificación de Perfil y Ponderación Dinámica ────────────────────
    es_financiera_reit = sector in ("Financial Services", "Real Estate")
    es_madura_dividendo = div_rate > 0 and div_yield >= 2.0 and dcf_viable
    es_alto_crecimiento = g_1_5 >= 0.15 or fcf_ttm <= 0

    if es_financiera_reit:
        perfil = "Institución Financiera / Inmobiliaria (Balance Intensivo)"
        peso_dcf, peso_ddm, peso_mult = 0.20, 0.40, 0.40
    elif es_madura_dividendo:
        perfil = "Empresa Madura con Flujo y Dividendo Estable"
        peso_dcf, peso_ddm, peso_mult = 0.45, 0.35, 0.20
    elif es_alto_crecimiento:
        perfil = "Empresa de Alto Crecimiento / Flujos Irregulares"
        peso_dcf, peso_ddm, peso_mult = 0.35, 0.00, 0.65
    else:
        perfil = "Crecimiento Sólido / Rentabilidad Sostenible"
        peso_dcf, peso_ddm, peso_mult = 0.60, (0.10 if ddm_viable else 0.0), (0.30 if ddm_viable else 0.40)

    # Anular pesos de modelos no viables y re-normalizar a 100%
    if not dcf_viable: peso_dcf = 0.0
    if not ddm_viable: peso_ddm = 0.0
    if not mult_viable: peso_mult = 0.0

    suma_pesos = peso_dcf + peso_ddm + peso_mult
    if suma_pesos > 0:
        peso_dcf  /= suma_pesos
        peso_ddm  /= suma_pesos
        peso_mult /= suma_pesos
    else:
        # Fallback de emergencia
        peso_dcf, peso_ddm, peso_mult = 1.0, 0.0, 0.0
        val_dcf = precio_actual * 0.90

    valor_compuesto_base = (val_dcf * peso_dcf) + (val_ddm * peso_ddm) + (val_mult * peso_mult)
    valor_compuesto_base = round(max(valor_compuesto_base, 0.0), 2)

    # ── E. Generación de Escenarios (Pesimista - Base - Optimista) ───────────
    # Escenario Pesimista (Bearish): Crecimiento -30%, WACC +1.0%, Terminal 2.0%
    res_fcff_bear = calcular_fcff_valuation(
        ocf_hist              = [ocf * 0.85 for ocf in ocf_hist],
        capex_hist            = capex_hist,
        interest_hist         = interest_hist,
        pretax_hist           = pretax_hist,
        taxprov_hist          = taxprov_hist,
        total_debt            = total_debt,
        total_cash            = total_cash,
        shares_diluted        = shares_diluted,
        mcap                  = mcap,
        beta                  = beta * 1.10,
        rf                    = rf + 0.50,
        precio_actual         = precio_actual,
        erp                   = erp,
        growth_rate_exp       = max(g_1_5 * 0.70, 0.015),
        revenue_ttm           = revenue_ttm * 0.90,
        operating_margin_hist = (operating_income_ttm / revenue_ttm * 0.85) if revenue_ttm > 0 else 0.12,
        fmp_key               = fmp_key,
        fred_key              = fred_key,
        ticker                = ticker,
    )
    val_dcf_bear = float(res_fcff_bear["valor_intrinseco"]) if dcf_viable else val_dcf * 0.75
    val_ddm_bear = val_ddm * 0.80 if ddm_viable else 0.0
    val_mult_bear = val_mult * 0.80 if mult_viable else 0.0
    escenario_pesimista = round((val_dcf_bear * peso_dcf) + (val_ddm_bear * peso_ddm) + (val_mult_bear * peso_mult), 2)

    # Escenario Optimista (Bullish): Crecimiento +20%, WACC -0.75%, Terminal 3.0%
    res_fcff_bull = calcular_fcff_valuation(
        ocf_hist              = [ocf * 1.15 for ocf in ocf_hist],
        capex_hist            = capex_hist,
        interest_hist         = interest_hist,
        pretax_hist           = pretax_hist,
        taxprov_hist          = taxprov_hist,
        total_debt            = total_debt,
        total_cash            = total_cash,
        shares_diluted        = shares_diluted,
        mcap                  = mcap,
        beta                  = max(beta * 0.95, 0.50),
        rf                    = rf,
        precio_actual         = precio_actual,
        erp                   = max(erp - 0.50, 4.0),
        growth_rate_exp       = min(g_1_5 * 1.20, 0.30),
        revenue_ttm           = revenue_ttm * 1.10,
        operating_margin_hist = (operating_income_ttm / revenue_ttm * 1.10) if revenue_ttm > 0 else 0.18,
        fmp_key               = fmp_key,
        fred_key              = fred_key,
        ticker                = ticker,
    )
    val_dcf_bull = float(res_fcff_bull["valor_intrinseco"]) if dcf_viable else val_dcf * 1.25
    val_ddm_bull = val_ddm * 1.20 if ddm_viable else 0.0
    val_mult_bull = val_mult * 1.20 if mult_viable else 0.0
    escenario_optimista = round((val_dcf_bull * peso_dcf) + (val_ddm_bull * peso_ddm) + (val_mult_bull * peso_mult), 2)

    # ── F. Margen de Seguridad, Veredicto y Advertencias ─────────────────────
    margen_seguridad = ((valor_compuesto_base - precio_actual) / precio_actual) if precio_actual > 0 else 0.0
    desc_req = 0.20 if (mcap > 0 and mcap < 2e9) else 0.10
    precio_max_compra = round(valor_compuesto_base * (1.0 - desc_req), 2)

    if valor_compuesto_base >= precio_actual * 1.15:
        status, semaforo = "🟢", "verde"
    elif valor_compuesto_base >= precio_actual * 0.90:
        status, semaforo = "🟡", "amarillo"
    else:
        status, semaforo = "🔴", "rojo"

    advertencia_calidad = ""
    if fcf_ttm <= 0:
        advertencia_calidad = "⚠️ Flujo de Caja Libre TTM negativo: la valoración descansa primordialmente en múltiplos normalizados y proyección operativa futura."
    elif not dcf_viable:
        advertencia_calidad = "ℹ️ Datos de flujo históricos con alta dispersión: modelo DCF reponderado."

    return {
        "valor_intrinseco": valor_compuesto_base,
        "escenario_pesimista": escenario_pesimista,
        "escenario_base": valor_compuesto_base,
        "escenario_optimista": escenario_optimista,
        "margen_seguridad": round(margen_seguridad, 4),
        "upside": round(margen_seguridad * 100.0, 2),
        "precio_max_compra": precio_max_compra,
        "desc_req": desc_req,
        "perfil_empresa": perfil,
        "status": status,
        "semaforo": semaforo,
        "modelos_detalle": {
            "dcf": {
                "nombre": "DCF Multietapa (FCFF)",
                "valor": round(val_dcf, 2),
                "peso": round(peso_dcf * 100.0, 1),
                "viable": dcf_viable,
            },
            "ddm": {
                "nombre": "Descuento de Dividendos (DDM)",
                "valor": round(val_ddm, 2),
                "peso": round(peso_ddm * 100.0, 1),
                "viable": ddm_viable,
            },
            "multiplos": {
                "nombre": "Múltiplos Históricos Normalizados",
                "valor": round(val_mult, 2),
                "peso": round(peso_mult * 100.0, 1),
                "viable": mult_viable,
            },
        },
        "supuestos_clave": {
            "wacc": wacc,
            "ke": ke,
            "kd": float(res_fcff["kd"]),
            "rf": rf,
            "erp": erp,
            "g_fase1": round(g_1_5 * 100.0, 2),
            "g_term": round(g_term * 100.0, 2),
            "tax_rate": float(res_fcff["tax_rate_real"]),
        },
        "advertencia_calidad": advertencia_calidad,
        # Compatibilidad con PDF y módulos
        "enterprise_value": float(res_fcff["enterprise_value"]),
        "equity_value": float(res_fcff["equity_value"]),
        "deuda_neta": float(res_fcff["deuda_neta"]),
        "fcff_historico": res_fcff["fcff_historico"],
        "fcff_proyectado": res_fcff["fcff_proyectado"],
        "wacc": wacc,
        "ke": ke,
        "kd": float(res_fcff["kd"]),
        "rf": rf,
        "erp": erp,
        "g_term": g_term,
        "v_intr_dcf": round(val_dcf, 2),
        "v_intr_ddm": round(val_ddm, 2),
        "v_intr_mult": round(val_mult, 2),
    }


def _resultado_adaptativo_vacio(
    precio_actual: float,
    total_cash: float,
    total_debt: float,
    rf: float,
    beta: float,
    erp: float,
) -> Dict[str, Any]:
    ke = rf + (beta * erp)
    return {
        "valor_intrinseco": 0.0,
        "escenario_pesimista": 0.0,
        "escenario_base": 0.0,
        "escenario_optimista": 0.0,
        "margen_seguridad": -1.0,
        "upside": -100.0,
        "precio_max_compra": 0.0,
        "desc_req": 0.10,
        "perfil_empresa": "Datos Insuficientes",
        "status": "🔴",
        "semaforo": "rojo",
        "modelos_detalle": {},
        "supuestos_clave": {
            "wacc": ke, "ke": ke, "kd": rf, "rf": rf, "erp": erp,
            "g_fase1": 8.0, "g_term": 2.5, "tax_rate": 0.21,
        },
        "advertencia_calidad": "⚠️ Datos insuficientes para completar la valoración adaptativa.",
        "enterprise_value": 0.0,
        "equity_value": 0.0,
        "deuda_neta": total_debt - total_cash,
        "fcff_historico": [],
        "fcff_proyectado": [],
        "wacc": ke,
        "ke": ke,
        "kd": rf,
        "rf": rf,
        "erp": erp,
        "g_term": 0.025,
        "v_intr_dcf": 0.0,
        "v_intr_ddm": 0.0,
        "v_intr_mult": 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. FACTORY PARA MATRIZ DE SENSIBILIDAD
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
    Función de valuación DCF por acción para matrices de sensibilidad:
    1. Proyección y Descuento de Flujos con Convención de Medio Año (Años 1 a 5).
    2. Valor Terminal Gordon Shapiro.
    3. Ajuste por Caja Neta por Acción: (Total_Cash - Total_Debt) / Shares.
    4. Enterprise Value -> Equity Value por acción.
    """
    wacc_dec = (wacc_var / 100.0) if wacc_var > 1.0 else wacc_var
    g_term_eff = g_term_var if g_term_var < wacc_dec else max(wacc_dec - 0.015, 0.010)

    # 1. Proyección y Descuento de Flujos (Mid-Year)
    pv_flujos = 0.0
    f_ps = float(flujo_por_accion)
    for t in range(1, 6):
        f_ps = f_ps * (1.0 + g_1_5)
        factor_my = (1.0 + wacc_dec) ** (t - 0.5)
        pv_flujos += f_ps / factor_my

    # 2. Valor Terminal Gordon Shapiro
    f_term = f_ps * (1.0 + g_term_eff)
    denominador_tv = max(wacc_dec - g_term_eff, 0.010)
    tv = f_term / denominador_tv
    pv_terminal = tv / ((1.0 + wacc_dec) ** 5)

    # 3. Puente Enterprise Value a Equity Value por acción
    caja_neta_ps = ((total_cash - total_debt) / shares_current) if shares_current > 0 else 0.0
    valor_intrinseco = max(pv_flujos + pv_terminal + caja_neta_ps, 0.0)

    es_atractivo = valor_intrinseco >= precio_actual
    semaforo = "verde" if es_atractivo else "rojo"
    status = "🟢" if es_atractivo else "🔴"
    upside = (((valor_intrinseco - precio_actual) / precio_actual) * 100.0) if precio_actual > 0 else 0.0

    return {
        "valor_intrinseco": round(valor_intrinseco, 2),
        "pv_flujos": round(pv_flujos, 2),
        "pv_terminal": round(pv_terminal, 2),
        "caja_neta_por_accion": round(caja_neta_ps, 2),
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
    Factory que retorna una función calculador(wacc_var, g_term_var) para matrices de sensibilidad.
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
