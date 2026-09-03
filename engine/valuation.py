from __future__ import annotations

import math
import statistics
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from config.settings import (
    DEFAULT_BUYBACK_RATE,
    DEFAULT_FADE_YEARS,
    G_TERM_DEFAULT,
    WACC_CEILING,
    WACC_FLOOR,
    WACC_MIN_SPREAD_OVER_G,
)


# -----------------------------------------------------------------------------
# 0. UTILITARIOS DEFENSIVOS
# -----------------------------------------------------------------------------

def safe_num(val: Any, default: float = 0.0) -> float:
    """
    Convierte cualquier valor de forma segura a float o al valor por defecto especificado.
    Maneja defensivamente:
    - Escalares numéricos (int, float, np.number).
    - Objetos de consenso (ConsensusWallStreet) con atributo target_mean.
    - Tuplas y listas de 1 elemento, o tuplas con 1 escalar numérico y metadatos (ej. (precio, moneda), (valor, status)).
    - pd.Series, np.ndarray de tamaño 1.
    - Diccionarios con claves numéricas estándar ('value', 'target_mean', etc.).
    - Strings formateados ('$1,250.50', '15.5%', '550.00 USD').
    - None, np.nan, float('nan'), inf, -inf, cadenas no numéricas y tipos corruptos.
    """
    if val is None:
        return float(default) if default is not None else 0.0

    try:
        if hasattr(val, "target_mean"):
            val = getattr(val, "target_mean")

        while isinstance(val, (tuple, list, set)):
            if len(val) == 0:
                return float(default) if default is not None else 0.0
            if len(val) == 1:
                val = next(iter(val))
                if val is None:
                    return float(default) if default is not None else 0.0
                continue
            numerics = []
            for item in val:
                if item is not None and not isinstance(item, (dict, list, tuple, set, bool)):
                    if isinstance(item, (int, float, np.number)):
                        numerics.append(item)
                    elif isinstance(item, str):
                        clean_item = item.replace(',', '').replace('$', '').replace('%', '').strip()
                        parts = clean_item.split()
                        if len(parts) > 1:
                            clean_item = parts[0]
                        try:
                            float(clean_item)
                            numerics.append(item)
                        except ValueError:
                            pass
            if len(numerics) == 1:
                val = numerics[0]
            else:
                return float(default) if default is not None else 0.0

        if isinstance(val, (pd.Series, np.ndarray)):
            if val.size == 0 or val.size > 1:
                return float(default) if default is not None else 0.0
            val = val.flat[0] if isinstance(val, np.ndarray) else val.iloc[0]
            if val is None or pd.isna(val):
                return float(default) if default is not None else 0.0

        if isinstance(val, dict):
            candidatos = [
                val.get("value"), val.get("val"), val.get("target_mean"),
                val.get("target_mean_price"), val.get("mean"), val.get("price"),
                val.get("close"), val.get("current"),
            ]
            found = False
            for cand in candidatos:
                if cand is not None and not isinstance(cand, (dict, list, tuple)):
                    val = cand
                    found = True
                    break
            if not found:
                return float(default) if default is not None else 0.0

        if isinstance(val, (int, float, np.number)):
            f_val = float(val)
            if math.isnan(f_val) or math.isinf(f_val):
                return float(default) if default is not None else 0.0
            return f_val

        if isinstance(val, str):
            clean_str = val.replace(',', '').replace('$', '').replace('%', '').strip()
            parts = clean_str.split()
            if len(parts) > 1:
                clean_str = parts[0]
            if not clean_str or clean_str.lower() in ('nan', 'none', 'n/a', 'null', 'inf', '-inf', 'n/d'):
                return float(default) if default is not None else 0.0
            return float(clean_str)

        if pd.isna(val):
            return float(default) if default is not None else 0.0

        return float(val)
    except (ValueError, TypeError, Exception):
        return float(default) if default is not None else 0.0


# -----------------------------------------------------------------------------
# 1. CALCULO DEL WACC CON DATOS REALES DE MERCADO
# -----------------------------------------------------------------------------

def calcular_wacc(
    tasa_libre_riesgo: float,
    beta: float,
    mcap: float,
    total_debt: float,
    int_exp: float,
    tax_rate: float,
    finnhub_key: str = "",
    fred_key: str = "",
    ticker: str = "",
    erp: float = 5.0,
    fmp_key: str = "",
) -> Dict[str, float]:
    """
    Costo Promedio Ponderado de Capital (WACC) con datos reales de mercado.

    Metodologia:
    - Ke = Rf + beta * ERP  (CAPM).
    - Kd = Interest_Expense / Total_Debt * 100  (empirico).
    - We = MarketCap / (MarketCap + Debt),  Wd = 1 - We.
    - WACC = We*Ke + Wd*Kd*(1-Tax_Rate).
    - Bandas de control: WACC_FLOOR <= WACC% <= WACC_CEILING.
      La invariante WACC > g + 1.5% se aplica en calcular_fcff_valuation.

    Returns:
        Diccionario con: wacc, ke, kd, we, wd, rf, erp, tax_rate.
    """
    api_k = finnhub_key or fmp_key
    rf       = max(safe_num(tasa_libre_riesgo, default=4.20), 0.0)
    beta_val = max(safe_num(beta, default=1.0), 0.05)
    erp_val  = max(safe_num(erp, default=5.0), 2.0)
    t_ef     = max(min(safe_num(tax_rate, default=0.21), 0.35), 0.0)

    ke = rf + (beta_val * erp_val)

    mcap_val = safe_num(mcap, default=0.0)
    debt_val = safe_num(total_debt, default=0.0)
    total_capital = mcap_val + debt_val
    if total_capital > 0:
        we = mcap_val / total_capital
        wd = debt_val / total_capital
    else:
        we, wd = 1.0, 0.0

    if debt_val > 0:
        int_exp_val = safe_num(int_exp, default=0.0)
        if int_exp_val > 0:
            kd = (int_exp_val / debt_val) * 100.0
        elif ticker and (api_k or fred_key):
            try:
                from data.financial_fetcher import obtener_kd_finnhub_fred  # noqa: PLC0415
                kd = safe_num(obtener_kd_finnhub_fred(ticker, api_k, fred_key, int_exp_val, debt_val), default=rf)
            except Exception:
                kd = rf
        else:
            kd = rf
    else:
        kd = rf

    wacc_real = (we * ke) + (wd * kd * (1.0 - t_ef))
    wacc = max(min(wacc_real, WACC_CEILING), WACC_FLOOR)

    return {
        "wacc":     round(wacc, 4),
        "ke":       round(ke, 4),
        "kd":       round(kd, 4),
        "we":       round(we, 4),
        "wd":       round(wd, 4),
        "rf":       round(rf, 4),
        "erp":      round(erp_val, 4),
        "tax_rate": round(t_ef, 4),
    }


# -----------------------------------------------------------------------------
# 2. NORMALIZACION Y AUXILIARES DE CRECIMIENTO
# -----------------------------------------------------------------------------

def calcular_fcff_normalizado(
    revenue_ttm: float,
    operating_margin_hist: float,
    tax_rate: float,
    shares_diluted: float,
    eps_ttm: float = 0.0,
    reinvestment_rate: float = 0.25,
) -> Tuple[float, float]:
    """
    FCFF Operativo Normalizado:
        FCFF_norm = Revenue_TTM x max(Margen_Op, 10%) x (1 - Tax_Rate)

    Returns:
        Tupla (fcff_total, fcff_por_accion).
    """
    t_ef  = max(min(safe_num(tax_rate, default=0.21), 0.35), 0.0)
    mg_op = max(safe_num(operating_margin_hist, default=0.10), 0.10)
    rev_val = safe_num(revenue_ttm, default=0.0)
    shares_val = safe_num(shares_diluted, default=0.0)
    eps_val = safe_num(eps_ttm, default=0.0)

    if rev_val > 0:
        fcff_total = rev_val * mg_op * (1.0 - t_ef)
        fcff_ps    = fcff_total / shares_val if shares_val > 0 else 0.0
        return fcff_total, fcff_ps

    if eps_val > 0:
        fcff_ps    = eps_val * (1.0 - max(min(safe_num(reinvestment_rate, default=0.25), 0.50), 0.10))
        fcff_total = fcff_ps * shares_val if shares_val > 0 else 0.0
        return fcff_total, fcff_ps

    return 0.0, 0.0


def calcular_g_term_restringido(wacc_decimal: float) -> float:
    """
    Tasa terminal alineada al crecimiento sostenible del PIB (~2.5%).
    Garantiza spread minimo WACC - g >= WACC_MIN_SPREAD_OVER_G.

    Args:
        wacc_decimal: WACC en decimal (ej. 0.09 = 9%).

    Returns:
        g_term en decimal.
    """
    g_term = G_TERM_DEFAULT
    if wacc_decimal <= g_term + WACC_MIN_SPREAD_OVER_G:
        g_term = max(wacc_decimal - WACC_MIN_SPREAD_OVER_G, 0.010)
    return round(g_term, 4)


def calcular_curva_crecimiento_5y(
    cagr_revenue_hist: float,
    revenue_growth_api: float = 0.0,
    g_term: float = 0.025,
    n_years: int = 5,
) -> List[float]:
    """
    Tasa de crecimiento proyectada para los anios explicitos.
    Prioridad: revenue_growth_api > cagr_revenue_hist > 8.0% fallback.
    Acotada en rango institucional [2%, 22%].
    """
    if revenue_growth_api > 0:
        g_1_5 = revenue_growth_api
    elif cagr_revenue_hist > 0:
        g_1_5 = cagr_revenue_hist
    else:
        g_1_5 = 0.08

    g_max = 0.45 if g_1_5 > 0.20 else 0.25
    g_1_5_clamped = min(max(g_1_5, 0.02), g_max)
    return [round(g_1_5_clamped, 5)] * n_years


# -----------------------------------------------------------------------------
# 3. MOTOR PRINCIPAL DE VALUACION FCFF / DCF INSTITUCIONAL
# -----------------------------------------------------------------------------

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
    finnhub_key: str = "",
    fred_key: str = "",
    ticker: str = "",
    buyback_rate: float = DEFAULT_BUYBACK_RATE,
    fade_years: int = DEFAULT_FADE_YEARS,
    g_term_override: Optional[float] = None,
    ebit_hist: Optional[List[float]] = None,
    da_hist: Optional[List[float]] = None,
    delta_nwc_hist: Optional[List[float]] = None,
    fmp_key: str = "",
) -> Dict[str, Union[float, str, list]]:
    """
    Motor de Valuacion FCFF (Free Cash Flow to Firm) Institucional.

    Jerarquia dual-path de FCFF base:
    1. Path primario (EBIT): FCFF = EBIT*(1-t) + D&A - CapEx - DNWC
       Elimina doble conteo de deuda al descontar con WACC.
    2. Path fallback (OCF): FCFF = OCF + Interest*(1-t) - CapEx
    3. Ultimo recurso: Normalizacion Revenue x Margen Operativo.

    Modelo 2 etapas + Fade Period lineal:
    - Fase 1 (anios 1..n_years): tasa g_1_5 constante.
    - Fade (anios n_years+1..n_total): transicion lineal g_1_5 -> g_term.

    Convencion de Medio Anio consistente:
    - Flujos: VP_t = FCFF_t / (1+WACC)^(t-0.5)
    - Terminal: VP_TV = TV / (1+WACC)^(n_total-0.5)

    Ajuste por recompras: shares_efectivas = shares * (1-buyback_rate)^n_total
    Invariante WACC > g + 1.5% forzado antes del TV.

    Args:
        ocf_hist, capex_hist, interest_hist, pretax_hist, taxprov_hist:
            Listas historicas (mas reciente primero).
        buyback_rate: Recompra neta anual (fraccion). Default: 0.0.
        fade_years:   Anios del fade period. Default: 3.
        g_term_override: Tasa terminal en decimal. Default: None (auto 2.5%).
        ebit_hist, da_hist, delta_nwc_hist: Insumos para path primario.

    Returns:
        Diccionario con metricas de valuacion, fcff_pv_detalle (tabla
        anio a anio) y metadatos del modelo.
    """
    total_debt = safe_num(total_debt, 0.0)
    total_cash = safe_num(total_cash, 0.0)
    shares_diluted = safe_num(shares_diluted, 0.0)
    mcap = safe_num(mcap, 0.0)
    beta = safe_num(beta, 1.0)
    rf = safe_num(rf, 4.20)
    precio_actual = safe_num(precio_actual, 0.0)
    erp = safe_num(erp, 5.0)
    revenue_ttm = safe_num(revenue_ttm, 0.0)
    operating_margin_hist = safe_num(operating_margin_hist, 0.0)
    cagr_revenue_hist = safe_num(cagr_revenue_hist, 0.0)
    revenue_growth_api = safe_num(revenue_growth_api, 0.0)
    fade_years = max(int(safe_num(fade_years, DEFAULT_FADE_YEARS)), 1)
    n_years = max(int(safe_num(n_years, 5)), 1)

    # Reconciliación defensiva de unidades entre Deuda/Efectivo y Market Cap
    if mcap > 1e9:
        if 0 < total_debt < 1e5:
            total_debt = total_debt * 1e6
        if 0 < total_cash < 1e5:
            total_cash = total_cash * 1e6

    if shares_diluted <= 0 or precio_actual <= 0:
        return _resultado_fcff_vacio(precio_actual, total_cash, total_debt, rf, beta, erp)

    ebit_hist_      = list(ebit_hist or [])
    da_hist_        = list(da_hist or [])
    delta_nwc_hist_ = list(delta_nwc_hist or [])

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

    # 2. WACC de mercado
    int_exp_ultimo = interest_hist[0] if interest_hist else 0.0
    res_wacc = calcular_wacc(
        tasa_libre_riesgo = rf,
        beta              = beta,
        mcap              = mcap,
        total_debt        = total_debt,
        int_exp           = int_exp_ultimo,
        tax_rate          = tax_rate_real,
        finnhub_key       = finnhub_key or fmp_key,
        fred_key          = fred_key,
        ticker            = ticker,
        erp               = erp,
    )
    wacc         = round(res_wacc["wacc"], 2)
    ke           = round(res_wacc["ke"], 2)
    kd           = round(res_wacc["kd"], 2)
    we           = round(res_wacc["we"], 4)
    wd           = round(res_wacc["wd"], 4)
    wacc_decimal = wacc / 100.0

    # 3. Tasa Terminal y WACC clamping con invariante financiero
    if g_term_override is not None:
        g_term = max(min(safe_num(g_term_override, default=0.025), 0.04), 0.005)
    else:
        g_term = G_TERM_DEFAULT

    min_wacc_dec = g_term + WACC_MIN_SPREAD_OVER_G
    if wacc_decimal < min_wacc_dec:
        wacc_decimal = min_wacc_dec
        wacc = round(wacc_decimal * 100.0, 2)
    wacc_decimal = min(wacc_decimal, WACC_CEILING / 100.0)

    # 4. FCFF Historico con Dual-Path
    fcff_historico: list[float] = []
    fcff_method_used = "ocf"

    for i in range(len(ocf_hist)):
        ebit_i  = ebit_hist_[i]      if i < len(ebit_hist_)      else 0.0
        da_i    = da_hist_[i]        if i < len(da_hist_)         else 0.0
        dnwc_i  = delta_nwc_hist_[i] if i < len(delta_nwc_hist_) else 0.0
        capex_i = abs(capex_hist[i]) if i < len(capex_hist)       else 0.0
        ocf_i   = ocf_hist[i]        if i < len(ocf_hist)         else 0.0
        int_i   = interest_hist[i]   if i < len(interest_hist)    else 0.0

        if ebit_i > 0 and da_i >= 0:
            nopat_i = ebit_i * (1.0 - tax_rate_real)
            # Inversión en capital de trabajo operativo (absorción de caja dnwc >= 0)
            # Evita que variaciones negativas o contables agreguen flujo extraordinario no recurrente
            dnwc_clamped = max(min(dnwc_i, nopat_i * 0.25), 0.0)
            fcff_ebit = nopat_i + da_i - capex_i - dnwc_clamped
            fcf_cash = ocf_i - capex_i
            escudo_i = int_i * (1.0 - tax_rate_real)
            # Validación contable: evitar sobrestimación sobre el flujo físico de caja + escudo fiscal
            if fcf_cash > 0:
                fcff_ebit = min(fcff_ebit, fcf_cash + escudo_i)
            if fcff_ebit > 0 and (fcf_cash <= 0 or fcff_ebit >= fcf_cash * 0.4):
                fcff_historico.append(fcff_ebit)
                if i == 0:
                    fcff_method_used = "ebit"
                continue

        escudo_fiscal = int_i * (1.0 - tax_rate_real)
        fcff_historico.append(max(ocf_i + escudo_fiscal - capex_i, 0.0))

    # 5. FCFF Base
    if fcff_historico and fcff_historico[0] > 0:
        fcff_base = fcff_historico[0]
    else:
        fcff_norm_tot, _ = calcular_fcff_normalizado(
            revenue_ttm           = revenue_ttm,
            operating_margin_hist = operating_margin_hist,
            tax_rate              = tax_rate_real,
            shares_diluted        = shares_diluted,
            eps_ttm               = (
                pretax_hist[0] * (1.0 - tax_rate_real) / shares_diluted
                if (pretax_hist and shares_diluted > 0) else 0.0
            ),
        )
        if fcff_norm_tot > 0:
            fcff_base = fcff_norm_tot
            if fcff_method_used == "ocf":
                fcff_method_used = "normalizado"
        else:
            fcff_base = max(ocf_hist[0] * 0.50 if ocf_hist else 0.0, 1e6)

    # 6. Tasa de Crecimiento Fase 1 (Prospectiva y Dinámica)
    if growth_rate_exp is not None and growth_rate_exp > 0:
        g_1_5 = growth_rate_exp
    elif revenue_growth_api > 0:
        g_1_5 = revenue_growth_api
    elif cagr_revenue_hist > 0:
        g_1_5 = cagr_revenue_hist
    else:
        g_1_5 = 0.08

    # Para empresas de hipercrecimiento probado, permitir tasas prospectivas de hasta 40%-45% en Fase 1
    g_max_fase1 = 0.45 if (growth_rate_exp is not None or revenue_growth_api > 0.20 or cagr_revenue_hist > 0.20) else 0.25
    if mcap > 200e9 and growth_rate_exp is None:
        g_max_fase1 = min(g_max_fase1, 0.20)
    g_1_5 = min(max(g_1_5, 0.02), g_max_fase1)

    # 7. Proyeccion 2 etapas + Fade Period con Mid-Year Convention
    fcff_proyectado: list[float] = []
    fcff_pv_detalle: list[dict]  = []
    pv_flujos: float = 0.0
    f_t     = fcff_base
    n_total = n_years + fade_years

    for t in range(1, n_total + 1):
        if t <= n_years:
            g_t = g_1_5
        else:
            alpha = (t - n_years) / fade_years
            g_t   = g_1_5 * (1.0 - alpha) + g_term * alpha

        f_t = f_t * (1.0 + g_t)
        fcff_proyectado.append(round(f_t, 2))

        factor_descuento = (1.0 + wacc_decimal) ** (t - 0.5)
        pv_t = f_t / factor_descuento
        pv_flujos += pv_t

        fase = "Fase 1" if t <= n_years else "Fade"
        fcff_pv_detalle.append({
            "Ano":               t,
            "Fase":              fase,
            "g Aplicada (%)":    round(g_t * 100, 2),
            "FCFF (M USD)":      round(f_t / 1e6, 2),
            "Factor Desc (t-0.5)": round(factor_descuento, 4),
            "VP (M USD)":        round(pv_t / 1e6, 2),
        })

    # 8. Valor Terminal (Mid-Year en n_total - 0.5)
    f_terminal     = fcff_proyectado[-1] * (1.0 + g_term)
    denominador_tv = max(wacc_decimal - g_term, WACC_MIN_SPREAD_OVER_G)
    terminal_value = f_terminal / denominador_tv
    pv_terminal    = terminal_value / ((1.0 + wacc_decimal) ** (n_total - 0.5))

    # 9. Puente Financiero EV -> Equity -> Por Accion
    enterprise_value = pv_flujos + pv_terminal
    deuda_neta       = total_debt - total_cash
    equity_value     = enterprise_value - deuda_neta

    buyback_rate_val = safe_num(buyback_rate, default=0.0)
    if buyback_rate_val > 1.0:
        buyback_rate_val = buyback_rate_val / 100.0
    buyback_rate_ = max(min(buyback_rate_val, 0.05), -0.05)

    shares_count = max(safe_num(shares_diluted, default=1.0), 1.0)
    if mcap > 10_000_000 and precio_actual > 0:
        implied_sh = mcap / precio_actual
        if shares_count <= 1000 or shares_count < implied_sh * 0.01 or shares_count > implied_sh * 100:
            shares_count = implied_sh

    if buyback_rate_ != 0.0:
        shares_efectivas = max(shares_count * ((1.0 - buyback_rate_) ** n_total), 1.0)
    else:
        shares_efectivas = max(shares_count, 1.0)
    valor_intrinseco = max(equity_value / shares_efectivas, 0.0)

    # 10. Margen de Seguridad y Precios Objetivo
    margen_seguridad  = (
        (valor_intrinseco - precio_actual) / precio_actual
        if precio_actual > 0 else 0.0
    )
    desc_req          = 0.20 if (mcap > 0 and mcap < 2e9) else 0.10
    precio_max_compra = round(valor_intrinseco * (1.0 - desc_req), 2)

    es_atractivo = valor_intrinseco >= precio_actual
    semaforo     = "verde" if es_atractivo else "rojo"
    status       = "\U0001f7e2" if es_atractivo else "\U0001f534"
    upside       = margen_seguridad * 100.0

    return {
        "valor_intrinseco":  round(valor_intrinseco, 2),
        "enterprise_value":  round(enterprise_value, 2),
        "equity_value":      round(equity_value, 2),
        "pv_flujos":         round(pv_flujos, 2),
        "pv_terminal":       round(pv_terminal, 2),
        "terminal_value":    round(terminal_value, 2),
        "deuda_neta":        round(deuda_neta, 2),
        "fcff_base":         round(fcff_base, 2),
        "fcff_historico":    fcff_historico,
        "fcff_proyectado":   fcff_proyectado,
        "fcff_pv_detalle":   fcff_pv_detalle,
        "fcff_method":       fcff_method_used,
        "wacc":              round(wacc, 2),
        "ke":                round(ke, 2),
        "kd":                round(kd, 2),
        "we":                round(we, 4),
        "wd":                round(wd, 4),
        "rf":                round(rf, 2),
        "erp":               round(erp, 2),
        "tax_rate_real":     round(tax_rate_real, 4),
        "g_term":            round(g_term, 4),
        "g_fase1":           round(g_1_5, 4),
        "total_cash":        total_cash,
        "total_debt":        total_debt,
        "shares_diluted":    shares_diluted,
        "shares_efectivas":  round(shares_efectivas, 0),
        "buyback_rate":      round(buyback_rate_, 4),
        "fade_years":        fade_years,
        "n_total":           n_total,
        "margen_seguridad":  round(margen_seguridad, 4),
        "precio_max_compra": precio_max_compra,
        "desc_req":          desc_req,
        "precio_actual":     precio_actual,
        "status":            status,
        "semaforo":          semaforo,
        "upside":            round(upside, 2),
    }


def _resultado_fcff_vacio(
    precio_actual: float,
    total_cash: float,
    total_debt: float,
    rf: float,
    beta: float,
    erp: float,
) -> Dict[str, Union[float, str, list]]:
    """Resultado neutro cuando faltan datos minimos de valuacion."""
    precio_actual = safe_num(precio_actual, 0.0)
    total_cash = safe_num(total_cash, 0.0)
    total_debt = safe_num(total_debt, 0.0)
    rf = safe_num(rf, 4.20)
    beta = safe_num(beta, 1.0)
    erp = safe_num(erp, 5.0)
    ke = rf + (beta * erp)
    return {
        "valor_intrinseco":  0.0,
        "enterprise_value":  0.0,
        "equity_value":      0.0,
        "pv_flujos":         0.0,
        "pv_terminal":       0.0,
        "terminal_value":    0.0,
        "deuda_neta":        total_debt - total_cash,
        "fcff_base":         0.0,
        "fcff_historico":    [],
        "fcff_proyectado":   [],
        "fcff_pv_detalle":   [],
        "fcff_method":       "sin_datos",
        "wacc":              ke,
        "ke":                ke,
        "kd":                rf,
        "we":                1.0,
        "wd":                0.0,
        "rf":                rf,
        "erp":               erp,
        "tax_rate_real":     0.21,
        "g_term":            G_TERM_DEFAULT,
        "g_fase1":           0.08,
        "total_cash":        total_cash,
        "total_debt":        total_debt,
        "shares_diluted":    0.0,
        "shares_efectivas":  0.0,
        "buyback_rate":      0.0,
        "fade_years":        DEFAULT_FADE_YEARS,
        "n_total":           5 + DEFAULT_FADE_YEARS,
        "margen_seguridad":  -1.0,
        "precio_max_compra": 0.0,
        "desc_req":          0.10,
        "precio_actual":     precio_actual,
        "status":            "\U0001f534",
        "semaforo":          "rojo",
        "upside":            -100.0,
    }


# -----------------------------------------------------------------------------
# 4. MODELO DE DESCUENTO DE DIVIDENDOS (DDM / GORDON GROWTH)
# -----------------------------------------------------------------------------

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
    div_rate_val = safe_num(div_rate, default=0.0)
    ke_val       = safe_num(ke, default=0.0)
    g_div_val    = safe_num(g_div, default=0.02)
    p_act        = safe_num(precio_actual, default=0.0)

    ke_dec    = (ke_val / 100.0) if ke_val > 1.0 else ke_val
    g_div_eff = min(max(g_div_val, 0.015), 0.04)

    if div_rate_val > 0 and ke_dec > g_div_eff:
        v_intr_ddm = (div_rate_val * (1.0 + g_div_eff)) / (ke_dec - g_div_eff)
        viable = True
    else:
        v_intr_ddm = 0.0
        viable     = False

    val_ddm_str = f"${v_intr_ddm:,.2f}" if v_intr_ddm > 0 else "N/A"

    if v_intr_ddm == 0:
        semaforo, status = "gris", "\u26aa"
    elif v_intr_ddm >= p_act:
        semaforo, status = "verde", "\U0001f7e2"
    else:
        semaforo, status = "rojo", "\U0001f534"

    return {
        "valor_intrinseco_ddm": round(v_intr_ddm, 2),
        "val_ddm_str":          val_ddm_str,
        "status":               status,
        "semaforo":             semaforo,
        "viable":               viable,
    }


# -----------------------------------------------------------------------------
# 5. FACTORY DE SENSIBILIDAD DCF
# -----------------------------------------------------------------------------

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
    Valuacion DCF por accion para matrices de sensibilidad.
    Aplica Mid-Year Convention y puente EV -> Equity por accion.
    """
    wacc_num   = safe_num(wacc_var, default=9.0)
    wacc_dec   = (wacc_num / 100.0) if wacc_num > 1.0 else wacc_num
    g_term_num = safe_num(g_term_var, default=0.025)
    g_term_eff = g_term_num if g_term_num < wacc_dec else max(wacc_dec - WACC_MIN_SPREAD_OVER_G, 0.010)

    pv_flujos = 0.0
    f_ps      = safe_num(flujo_por_accion, default=0.0)
    g_1_5_val = safe_num(g_1_5, default=0.08)
    for t in range(1, 6):
        f_ps      = f_ps * (1.0 + g_1_5_val)
        factor_my = (1.0 + wacc_dec) ** (t - 0.5)
        pv_flujos += f_ps / factor_my

    f_term         = f_ps * (1.0 + g_term_eff)
    denominador_tv = max(wacc_dec - g_term_eff, WACC_MIN_SPREAD_OVER_G)
    tv             = f_term / denominador_tv
    pv_terminal    = tv / ((1.0 + wacc_dec) ** (5 - 0.5))

    p_act = safe_num(precio_actual, default=0.0)
    t_cash = safe_num(total_cash, default=0.0)
    t_debt = safe_num(total_debt, default=0.0)
    sh_curr = max(safe_num(shares_current, default=1.0), 1.0)

    caja_neta_ps     = ((t_cash - t_debt) / sh_curr) if sh_curr > 0 else 0.0
    valor_intrinseco = max(pv_flujos + pv_terminal + caja_neta_ps, 0.0)

    es_atractivo = valor_intrinseco >= p_act
    semaforo     = "verde" if es_atractivo else "rojo"
    status       = "\U0001f7e2" if es_atractivo else "\U0001f534"
    upside       = (
        ((valor_intrinseco - p_act) / p_act) * 100.0
        if p_act > 0 else 0.0
    )

    return {
        "valor_intrinseco":     round(valor_intrinseco, 2),
        "pv_flujos":            round(pv_flujos, 2),
        "pv_terminal":          round(pv_terminal, 2),
        "caja_neta_por_accion": round(caja_neta_ps, 2),
        "status":               status,
        "semaforo":             semaforo,
        "upside":               round(upside, 2),
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
    Factory que retorna calculador(wacc_var, g_term_var) -> valor_intrinseco
    para matrices de sensibilidad bidimensionales.
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
        return safe_num(res.get("valor_intrinseco"), default=0.0)
    return calculador