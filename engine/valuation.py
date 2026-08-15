from __future__ import annotations

import statistics
from typing import Optional

from data.financial_fetcher import obtener_kd_fmp_fred
from config.settings import safe_get


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
    ocf_hist: list[float],
    capex_hist: list[float],
    interest_hist: list[float],
    pretax_hist: list[float],
    taxprov_hist: list[float],
    total_debt: float,
    total_cash: float,
    shares_diluted: float,
    mcap: float,
    beta: float,
    rf: float,
    precio_actual: float,
    n_years: int = 5,
    erp: float = 5.5,
) -> dict:
    """
    Motor de valuación de grado analítico basado en Flujo de Caja Libre para la
    Firma (FCFF) con WACC dinámico empírico y puente completo EV → Equity Value.

    **Fórmula FCFF:**

    .. math::

        FCFF_i = OCF_i + (InterestExpense_i \\times (1 - T_{ef})) - CapEx_i

    **WACC dinámico:**

    .. math::

        K_e = R_f + \\beta \\times ERP \\qquad K_d = \\frac{\\sum Interest}{Deuda}

        WACC = \\frac{E}{V} K_e + \\frac{D}{V} K_d (1 - T_{ef})

    **Reconciliación de valor:**

    .. math::

        EV = PV_{flujos} + PV_{TV} \\qquad
        Equity = EV + Caja - Deuda \\qquad
        V_{acc} = \\frac{Equity}{Shares}

    Manejo defensivo:
    - ``total_debt = 0`` → ``wd = 0``, WACC = Ke (sin división por cero en Kd).
    - FCFF base negativo → se usa un fallback conservador del 2% del OCF promedio.
    - Tasas impositivas derivadas de cero o negativas → fallback a 21%.
    - ``shares_diluted <= 0`` → retorna ``valor_intrinseco = 0.0`` con status rojo.

    Args:
        ocf_hist:       Operating Cash Flow histórico por período (más reciente = índice 0).
        capex_hist:     CapEx histórico positivo por período (más reciente = índice 0).
        interest_hist:  Interest Expense histórico absoluto (más reciente = índice 0).
        pretax_hist:    Pre-tax Income histórico (más reciente = índice 0).
        taxprov_hist:   Tax Provision histórica (más reciente = índice 0).
        total_debt:     Deuda total más reciente en USD.
        total_cash:     Efectivo + equivalentes más reciente en USD.
        shares_diluted: Acciones diluidas en circulación.
        mcap:           Capitalización de mercado actual en USD.
        beta:           Coeficiente beta del activo.
        rf:             Tasa libre de riesgo en % (e.g. 4.35).
        precio_actual:  Precio de mercado actual por acción.
        n_years:        Años de proyección explícita (default: 5).
        erp:            Prima de riesgo de mercado en % (default: 5.5).

    Returns:
        Diccionario rico con todos los componentes del modelo:

        - ``valor_intrinseco``  (float): Valor por acción calculado.
        - ``enterprise_value``  (float): EV total en USD.
        - ``equity_value``      (float): Equity Value total en USD.
        - ``pv_flujos``         (float): VP de flujos proyectados.
        - ``pv_terminal``       (float): VP del Valor Terminal.
        - ``fcff_historico``    (list):  FCFF real calculado por período.
        - ``fcff_proyectado``   (list):  FCFF proyectado año 1–N.
        - ``wacc``              (float): WACC resultante en %.
        - ``ke``                (float): Costo de capital propio (CAPM) en %.
        - ``kd``                (float): Costo de deuda efectivo en %.
        - ``we``                (float): Peso de equity en estructura de capital.
        - ``wd``                (float): Peso de deuda en estructura de capital.
        - ``rf``                (float): Tasa libre de riesgo usada en %.
        - ``tax_rate_real``     (float): Tasa impositiva efectiva real.
        - ``g_term``            (float): Tasa de crecimiento terminal.
        - ``total_cash``        (float): Efectivo total.
        - ``total_debt``        (float): Deuda total.
        - ``shares_diluted``    (float): Acciones diluidas.
        - ``margen_seguridad``  (float): (V - P) / P en decimal.
        - ``precio_actual``     (float): Precio de mercado actual.
        - ``status``            (str):   Emoji semáforo.
        - ``semaforo``          (str):   "verde" | "rojo".
        - ``upside``            (float): Upside/downside en %.
    """
    # ── Guardia: acciones dilluidas válidas ───────────────────────────────────
    if shares_diluted <= 0:
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
    tax_rate_real = max(min(tax_rate_real, 0.40), 0.0)

    # ── 2. FCFF histórico ────────────────────────────────────────────────────
    n_hist = len(ocf_hist)
    fcff_historico: list[float] = []
    for i in range(n_hist):
        ocf_i    = ocf_hist[i] if i < len(ocf_hist) else 0.0
        capex_i  = capex_hist[i] if i < len(capex_hist) else 0.0
        int_i    = interest_hist[i] if i < len(interest_hist) else 0.0
        fcff_i   = ocf_i + (int_i * (1.0 - tax_rate_real)) - capex_i
        fcff_historico.append(fcff_i)

    # ── 3. FCFF base para proyección (media últimos 3 años, ponderada) ───────
    periodos_base = fcff_historico[:min(3, len(fcff_historico))]
    pesos = [3, 2, 1][:len(periodos_base)]
    fcff_base: float = (
        sum(f * p for f, p in zip(periodos_base, pesos)) / sum(pesos)
        if periodos_base else 0.0
    )

    # Fallback defensivo si FCFF base es negativo o nulo
    if fcff_base <= 0:
        ocf_prom = statistics.mean(ocf_hist) if ocf_hist else 0.0
        fcff_base = max(ocf_prom * 0.02, 1.0)  # 2% del OCF promedio como mínimo

    # ── 4. Tasa de crecimiento implícita del FCFF histórico ──────────────────
    if len(fcff_historico) >= 2 and fcff_historico[-1] > 0 and fcff_historico[0] > 0:
        n_periodos = len(fcff_historico) - 1
        g_hist_raw = (fcff_historico[0] / fcff_historico[-1]) ** (1.0 / n_periodos) - 1.0
        # Acotar a un rango razonable para evitar distorsiones por un año atípico
        g_hist = max(min(g_hist_raw, 0.20), -0.05)
    else:
        g_hist = 0.06  # Crecimiento conservador por defecto

    # ── 5. WACC dinámico ─────────────────────────────────────────────────────
    ke: float = rf + (beta * erp)  # CAPM en %

    # Kd efectivo basado en gastos de intereses reales sobre deuda total
    if total_debt > 0 and interest_hist:
        int_reciente = interest_hist[0]
        kd_raw = (int_reciente / total_debt) * 100.0  # En %
        # Clampear entre Rf y 20% para evitar valores irracionales
        kd: float = max(min(kd_raw, 20.0), rf)
    else:
        kd = 0.0  # Sin deuda → costo de deuda irrelevante

    # Ponderaciones de mercado
    total_capital = mcap + total_debt
    if total_capital > 0:
        we: float = mcap / total_capital
        wd: float = total_debt / total_capital
    else:
        we, wd = 1.0, 0.0

    # WACC final (en %)
    wacc_raw = (we * ke) + (wd * kd * (1.0 - tax_rate_real))
    # Límites realistas: mínimo Rf + 1%, máximo 18%
    wacc: float = max(min(wacc_raw, 18.0), max(rf + 1.0, 6.0))

    wacc_decimal = wacc / 100.0

    # ── 6. Tasa de crecimiento terminal ──────────────────────────────────────
    # g_term conservador: menor entre Rf/2 y 3.0%, siempre < WACC
    g_term: float = min(rf / 2.0 / 100.0, 0.030, wacc_decimal - 0.01)
    g_term = max(g_term, 0.010)  # Mínimo 1% (economía nominal mínima)

    # ── 7. Proyección de flujos explícitos (n_years) ─────────────────────────
    # Año 1-3: crecimiento histórico ajustado; Año 4-5: convergencia a g_term
    fcff_proyectado: list[float] = []
    g_fase1 = max(min(g_hist, 0.15), 0.02)  # Fase 1 acotada [2%, 15%]
    g_fase2 = max(min((g_fase1 + g_term * 100.0 * 3.0) / 4.0 / 100.0, 0.08), g_term)

    pv_flujos: float = 0.0
    f_actual = fcff_base
    for yr in range(1, n_years + 1):
        g_yr = g_fase1 if yr <= 3 else g_fase2
        f_actual = f_actual * (1.0 + g_yr)
        fcff_proyectado.append(f_actual)
        pv_flujos += f_actual / ((1.0 + wacc_decimal) ** yr)

    # ── 8. Valor Terminal (Gordon Growth) ────────────────────────────────────
    f_terminal = fcff_proyectado[-1] * (1.0 + g_term) if fcff_proyectado else fcff_base * (1.0 + g_term)
    denominador_tv = wacc_decimal - g_term
    if denominador_tv > 0:
        valor_terminal = f_terminal / denominador_tv
    else:
        # Fallback: múltiplo de salida conservador (15x FCFF terminal)
        valor_terminal = f_terminal * 15.0

    pv_terminal: float = valor_terminal / ((1.0 + wacc_decimal) ** n_years)

    # ── 9. Puente: Enterprise Value → Equity Value → Valor por Acción ────────
    enterprise_value: float = pv_flujos + pv_terminal
    equity_value: float = enterprise_value + total_cash - total_debt
    valor_intrinseco: float = equity_value / shares_diluted if shares_diluted > 0 else 0.0

    # Guardia: si el equity_value es negativo (empresa muy endeudada), se reporta
    # sin lanzar excepción — el semáforo indicará la situación.
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
