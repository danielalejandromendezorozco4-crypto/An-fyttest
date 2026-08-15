from __future__ import annotations

import statistics
from typing import List, Optional

# NOTE: Imports de módulos con dependencia de streamlit (financial_fetcher,
# config.settings) se hacen de forma diferida (lazy) dentro de cada función
# para evitar arrastrar streamlit al nivel de módulo. Esto previene
# ImportError en pytest y en Streamlit Cloud cuando el grafo de dependencias
# se resuelve antes de que streamlit esté disponible.


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES HEREDADAS — se mantienen sin cambio para backward-compatibility
# ─────────────────────────────────────────────────────────────────────────────

def calcular_wacc(
    tasa_libre_riesgo: float,
    beta: float,
    mcap: float,
    total_debt: float,
    int_exp: float,
    fmp_key: str,
    fred_key: str,
    tax_rate: float,
    ticker: str,
) -> dict:
    """
    Calcula el WACC (Weighted Average Cost of Capital) usando CAPM y la
    función centralizada de costo de deuda (FRED / FMP).

    Args:
        tasa_libre_riesgo: Tasa libre de riesgo en % (e.g. 4.35).
        beta:              Coeficiente beta del activo.
        mcap:              Capitalización de mercado en USD.
        total_debt:        Deuda total en USD.
        int_exp:           Gastos por intereses anuales en USD.
        fmp_key:           API Key de Financial Modeling Prep.
        fred_key:          API Key de FRED.
        tax_rate:          Tasa impositiva efectiva (decimal, e.g. 0.21).
        ticker:            Símbolo bursátil.

    Returns:
        Diccionario con ``ke``, ``kd``, ``wacc``, ``we``, ``wd`` (todos en %).
    """
    erp = 5.5
    total_capital = mcap + total_debt
    we, wd = (mcap / total_capital, total_debt / total_capital) if total_capital > 0 else (1.0, 0.0)
    ke = tasa_libre_riesgo + (beta * erp)
    # Import diferido para evitar arrastrar `streamlit` al nivel de módulo
    from data.financial_fetcher import obtener_kd_fmp_fred  # noqa: PLC0415
    kd_real = obtener_kd_fmp_fred(ticker, fmp_key, fred_key, int_exp, total_debt)
    kd = min(max(kd_real, tasa_libre_riesgo), 15.0)
    wacc = max(min((we * ke) + (wd * kd * (1 - tax_rate)), 15.0), 7.5)

    return {"ke": ke, "kd": kd, "wacc": wacc, "we": we, "wd": wd}


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
) -> dict:
    """
    Calcula el Valor Intrínseco por acción mediante Descuento de Flujos (DCF)
    simplificado, usando un modelo híbrido Gordon Growth / múltiplo PE terminal.

    Args:
        wacc_var:        WACC en % (e.g. 9.5).
        g_term_var:      Tasa de crecimiento terminal (decimal).
        flujo_por_accion: Flujo de caja base por acción.
        g_1_5:           Tasa de crecimiento para el período explícito (decimal).
        precio_actual:   Precio de mercado actual por acción.
        eps_ttm:         EPS de los últimos 12 meses.
        total_cash:      Efectivo total de la empresa.
        total_debt:      Deuda total de la empresa.
        shares_current:  Acciones en circulación.

    Returns:
        Diccionario con ``valor_intrinseco``, ``status``, ``semaforo``, ``upside``.
    """
    pv_f_ps = 0.0
    f_ps_var = flujo_por_accion
    for i in range(1, 6):
        f_ps_var *= (1 + g_1_5)
        pv_f_ps += f_ps_var / ((1 + (wacc_var / 100)) ** i)

    tv_gordon_var = (
        (f_ps_var * (1 + g_term_var)) / ((wacc_var / 100) - g_term_var)
        if (wacc_var / 100) > g_term_var else 0
    )
    pe_dinamico_actual = (precio_actual / eps_ttm) if eps_ttm > 0 else 15
    terminal_pe_var = max(min(pe_dinamico_actual * 0.75, 20.0), 10.0)
    tv_hibrido_var = (tv_gordon_var + (f_ps_var * terminal_pe_var)) / 2
    pv_terminal_var = tv_hibrido_var / ((1 + (wacc_var / 100)) ** 5)
    v_calc = pv_f_ps + pv_terminal_var + (
        (total_cash - total_debt) / shares_current if shares_current > 0 else 0
    )
    v_final = v_calc if v_calc > 0 else (precio_actual * 0.85)

    es_atractivo = v_final >= precio_actual
    semaforo = "verde" if es_atractivo else "rojo"
    status = "🟢" if es_atractivo else "🔴"
    upside = (((v_final - precio_actual) / precio_actual) * 100) if precio_actual > 0 else 0.0

    return {"valor_intrinseco": v_final, "status": status, "semaforo": semaforo, "upside": upside}


def crear_calculador_dcf(
    flujo_por_accion: float,
    g_1_5: float,
    precio_actual: float,
    eps_ttm: float,
    total_cash: float,
    total_debt: float,
    shares_current: float,
):
    """
    Factory que retorna una función ``calculador(wacc_var, g_term_var) -> float``
    para simplificar iteraciones en matrices de sensibilidad y exportación PDF.
    """
    def calculador(wacc_var: float, g_term_var: float) -> float:
        res = calcular_dcf_intr_ps(
            wacc_var, g_term_var, flujo_por_accion, g_1_5,
            precio_actual, eps_ttm, total_cash, total_debt, shares_current
        )
        return res["valor_intrinseco"]
    return calculador


def calcular_ddm(
    div_rate: float,
    ke: float,
    g_div: float,
    precio_actual: float = 0.0,
) -> dict:
    """
    Calcula el Valor Intrínseco mediante el Modelo Gordon Growth (DDM).

    Args:
        div_rate:      Dividendo anualizado por acción.
        ke:            Costo de capital propio en % (e.g. 9.0).
        g_div:         Tasa de crecimiento del dividendo (decimal).
        precio_actual: Precio de mercado actual por acción.

    Returns:
        Diccionario con ``valor_intrinseco_ddm``, ``val_ddm_str``, ``status``, ``semaforo``.
    """
    v_intr_ddm = (
        (div_rate * (1 + g_div)) / ((ke / 100) - g_div)
        if (div_rate > 0 and (ke / 100) > g_div) else 0.0
    )
    val_ddm_str = f"${v_intr_ddm:,.2f}" if v_intr_ddm > 0 else "N/A"

    if v_intr_ddm == 0:
        semaforo, status = "gris", "⚪"
    elif v_intr_ddm >= precio_actual:
        semaforo, status = "verde", "🟢"
    else:
        semaforo, status = "rojo", "🔴"

    return {
        "valor_intrinseco_ddm": v_intr_ddm,
        "val_ddm_str": val_ddm_str,
        "status": status,
        "semaforo": semaforo,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NUEVO MOTOR DE VALUACIÓN: FCFF CON WACC DINÁMICO EMPÍRICO
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
) -> dict:
    """
    Motor de valuación de grado analítico basado en Flujo de Caja Libre para la
    Firma (FCFF) con WACC dinámico empírico y puente completo EV → Equity Value.

    Calibración institucional:
    - Base FCFF anclada al período más reciente sin penalizar crecimiento histórico.
    - Curva de crecimiento explícita (Fase 1 años 1–3, Fase 2 convergencia años 4–5).
    - Valor Terminal Híbrido: Gordon Growth con $g_{term}$ balanceado y múltiplo de salida.
    - WACC acotado a rangos empíricos institucionales (6.5% - 14.0%).
    """
    # ── Guardia: acciones diluidas válidas ───────────────────────────────────
    if shares_diluted <= 0 or precio_actual <= 0:
        return _resultado_fcff_vacio(precio_actual, total_cash, total_debt, rf, beta, erp)

    # ── 1. Tasa impositiva efectiva real ─────────────────────────────────────
    tasas_impositivas: list[float] = []
    for pt, tp in zip(pretax_hist, taxprov_hist):
        if pt > 0 and tp >= 0:
            t_i = tp / pt
            if 0.0 < t_i <= 0.60:  # Filtrar tasas anómalas
                tasas_impositivas.append(t_i)

    tax_rate_real: float = (
        statistics.mean(tasas_impositivas) if tasas_impositivas else 0.21
    )
    tax_rate_real = max(min(tax_rate_real, 0.35), 0.0)

    # ── 2. FCFF histórico ────────────────────────────────────────────────────
    n_hist = len(ocf_hist)
    fcff_historico: list[float] = []
    for i in range(n_hist):
        ocf_i    = ocf_hist[i] if i < len(ocf_hist) else 0.0
        capex_i  = capex_hist[i] if i < len(capex_hist) else 0.0
        int_i    = interest_hist[i] if i < len(interest_hist) else 0.0
        fcff_i   = ocf_i + (int_i * (1.0 - tax_rate_real)) - capex_i
        fcff_historico.append(fcff_i)

    # ── 3. FCFF base para proyección (anclado en flujo reciente) ─────────────
    fcff_reciente = fcff_historico[0] if fcff_historico else 0.0
    periodos_base = fcff_historico[:min(3, len(fcff_historico))]
    pesos = [3, 2, 1][:len(periodos_base)]
    fcff_media_pond = (
        sum(f * p for f, p in zip(periodos_base, pesos)) / sum(pesos)
        if periodos_base else 0.0
    )

    if fcff_reciente > 0:
        # Si la empresa está en expansión, priorizar el flujo reciente
        fcff_base = max(fcff_reciente, fcff_media_pond)
    elif fcff_media_pond > 0:
        fcff_base = fcff_media_pond
    else:
        # Fallback defensivo si FCFF base es negativo o nulo
        ocf_prom = statistics.mean(ocf_hist) if ocf_hist else 0.0
        fcff_base = max(ocf_prom * 0.05, precio_actual * shares_diluted * 0.02, 1.0)

    # ── 4. Tasa de crecimiento implícita del FCFF histórico / fundamental ───
    if growth_rate_exp is not None and growth_rate_exp > 0:
        g_hist = growth_rate_exp
    elif len(fcff_historico) >= 2 and fcff_historico[-1] > 0 and fcff_historico[0] > 0:
        n_periodos = len(fcff_historico) - 1
        g_hist_raw = (fcff_historico[0] / fcff_historico[-1]) ** (1.0 / n_periodos) - 1.0
        # Acotar a un rango institucional razonable [4%, 18%]
        g_hist = max(min(g_hist_raw, 0.18), 0.04)
    else:
        g_hist = 0.08  # Crecimiento moderado institucional por defecto

    # ── 5. WACC dinámico ─────────────────────────────────────────────────────
    ke: float = rf + (beta * erp)  # CAPM en %

    # Kd efectivo basado en gastos de intereses reales sobre deuda total
    if total_debt > 0 and interest_hist and interest_hist[0] > 0:
        int_reciente = interest_hist[0]
        kd_raw = (int_reciente / total_debt) * 100.0  # En %
        kd: float = max(min(kd_raw, 12.0), rf)
    else:
        kd = 0.0  # Sin deuda o intereses mínimos

    # Ponderaciones de mercado
    total_capital = mcap + total_debt
    if total_capital > 0:
        we: float = mcap / total_capital
        wd: float = total_debt / total_capital
    else:
        we, wd = 1.0, 0.0

    # WACC final (en %)
    wacc_raw = (we * ke) + (wd * kd * (1.0 - tax_rate_real))
    # Acotar a límites realistas: piso 6.5%, techo 14.0%
    wacc: float = max(min(wacc_raw, 14.0), max(rf + 0.5, 6.5))

    wacc_decimal = wacc / 100.0

    # ── 6. Tasa de crecimiento terminal ──────────────────────────────────────
    # g_term institucional: 2.2% a 3.0%, asegurando spread >= 4.0% vs WACC
    g_term: float = min(0.030, max(0.022, (rf / 100.0) * 0.60))
    if (wacc_decimal - g_term) < 0.040:
        g_term = max(wacc_decimal - 0.040, 0.015)

    # ── 7. Proyección de flujos explícitos (n_years) ─────────────────────────
    # Fase 1 (Años 1–3): crecimiento dinámico; Fase 2 (Años 4–5): convergencia suave
    fcff_proyectado: list[float] = []
    g_fase1 = max(min(g_hist, 0.16), 0.04)
    g_fase2 = (g_fase1 + g_term * 2.0) / 3.0

    pv_flujos: float = 0.0
    f_actual = fcff_base
    for yr in range(1, n_years + 1):
        g_yr = g_fase1 if yr <= 3 else g_fase2
        f_actual = f_actual * (1.0 + g_yr)
        fcff_proyectado.append(f_actual)
        pv_flujos += f_actual / ((1.0 + wacc_decimal) ** yr)

    # ── 8. Valor Terminal Híbrido (Gordon Growth + Múltiplo de Salida) ───────
    f_terminal = fcff_proyectado[-1] * (1.0 + g_term)
    denominador_tv = max(wacc_decimal - g_term, 0.040)
    tv_gordon = f_terminal / denominador_tv

    # Múltiplo de salida calibrado (16x - 22x FCF terminal según WACC/calidad)
    exit_multiple = min(max(1.0 / denominador_tv, 16.0), 22.0)
    tv_exit = fcff_proyectado[-1] * exit_multiple

    valor_terminal = (tv_gordon + tv_exit) / 2.0
    pv_terminal: float = valor_terminal / ((1.0 + wacc_decimal) ** n_years)

    # ── 9. Puente: Enterprise Value → Equity Value → Valor por Acción ────────
    enterprise_value: float = pv_flujos + pv_terminal
    equity_value: float = enterprise_value + total_cash - total_debt
    valor_intrinseco: float = equity_value / shares_diluted if shares_diluted > 0 else 0.0

    if valor_intrinseco <= 0:
        valor_intrinseco = 0.0

    # ── 10. Métricas de semáforo y margen de seguridad ───────────────────────
    margen_seguridad: float = (
        (valor_intrinseco - precio_actual) / precio_actual
        if precio_actual > 0 else 0.0
    )
    es_atractivo = valor_intrinseco > 0 and valor_intrinseco >= precio_actual
    semaforo = "verde" if es_atractivo else "rojo"
    status = "🟢" if es_atractivo else "🔴"
    upside = margen_seguridad * 100.0

    return {
        "valor_intrinseco":  valor_intrinseco,
        "enterprise_value":  enterprise_value,
        "equity_value":      equity_value,
        "pv_flujos":         pv_flujos,
        "pv_terminal":       pv_terminal,
        "fcff_historico":    fcff_historico,
        "fcff_proyectado":   fcff_proyectado,
        "wacc":              wacc,
        "ke":                ke,
        "kd":                kd,
        "we":                we,
        "wd":                wd,
        "rf":                rf,
        "tax_rate_real":     tax_rate_real,
        "g_term":            g_term,
        "g_fase1":           g_fase1,
        "total_cash":        total_cash,
        "total_debt":        total_debt,
        "shares_diluted":    shares_diluted,
        "margen_seguridad":  margen_seguridad,
        "precio_actual":     precio_actual,
        "status":            status,
        "semaforo":          semaforo,
        "upside":            upside,
    }


def _resultado_fcff_vacio(
    precio_actual: float,
    total_cash: float,
    total_debt: float,
    rf: float,
    beta: float,
    erp: float,
) -> dict:
    """
    Retorna un resultado FCFF con valor intrínseco 0 y semáforo rojo cuando
    los datos mínimos necesarios no están disponibles (e.g., acciones = 0).
    """
    ke = rf + (beta * erp)
    return {
        "valor_intrinseco":  0.0,
        "enterprise_value":  0.0,
        "equity_value":      0.0,
        "pv_flujos":         0.0,
        "pv_terminal":       0.0,
        "fcff_historico":    [],
        "fcff_proyectado":   [],
        "wacc":              ke,
        "ke":                ke,
        "kd":                0.0,
        "we":                1.0,
        "wd":                0.0,
        "rf":                rf,
        "tax_rate_real":     0.21,
        "g_term":            0.025,
        "g_fase1":           0.06,
        "total_cash":        total_cash,
        "total_debt":        total_debt,
        "shares_diluted":    0.0,
        "margen_seguridad":  -1.0,
        "precio_actual":     precio_actual,
        "status":            "🔴",
        "semaforo":          "rojo",
        "upside":            -100.0,
    }
