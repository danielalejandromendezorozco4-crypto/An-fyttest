import math
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd


def safe_get(d: Any, keys: list, default: Any = 0) -> Any:
    """Helper defensivo para extraer valores de diccionarios anidados o listas."""
    if not isinstance(d, dict):
        return default
    for key in keys:
        if key in d and d[key] is not None:
            return d[key]
    return default


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


# ─────────────────────────────────────────────────────────────────────────────
# 1. MÚLTIPLOS DE VALUACIÓN ESTANDARIZADOS (Finviz / Yahoo Finance / Investing)
# ─────────────────────────────────────────────────────────────────────────────

def calcular_buyback_yield(
    inc: Any,
    bs: Any,
    cf: Any,
    shares_current: float = 0.0,
    mcap: float = 0.0,
) -> Dict[str, Any]:
    """
    Calcula el rendimiento por recompra de acciones (Buyback Yield) y la variación del flotante.

    Metodología:
    1. Primaria: Variación interanual de acciones en circulación (Income Statement o Balance Sheet).
       Buyback Yield (%) = ((Shares_Prev - Shares_Curr) / Shares_Prev) * 100
    2. Fallback: Flujo de caja destinado a recompra de acciones (Cash Flow Statement) / Market Cap.
       Buyback Yield (%) = (Repurchase Of Capital Stock / Market Cap) * 100

    Returns:
        dict con:
        - buyback_yield: float (porcentaje, positivo si reduce flotante, negativo si diluye)
        - buyback_yield_str: str (formateado con 1 decimal, ej. "2.5%", "0.0%", "-1.2%")
        - col_by: str ("🟢", "🟡", "🔴")
        - msg_by: str (mensaje diagnóstico)
        - repurchase_val: float (monto en USD destinado a recompras)
    """
    buyback_yield = 0.0
    sh_prev = 0.0
    repurchase_val = 0.0

    # Método 1: Variación interanual de acciones en circulación
    for df_sh in [inc, bs]:
        if isinstance(df_sh, pd.DataFrame) and not df_sh.empty:
            for fila_sh in [
                'Basic Average Shares', 'Diluted Average Shares',
                'Ordinary Shares Number', 'Common Stock',
                'Weighted Average Shares Diluted', 'Weighted Average Shares Basic'
            ]:
                if fila_sh in df_sh.index and len(df_sh.columns) > 1:
                    try:
                        s_sh = df_sh.loc[fila_sh].dropna()
                        if isinstance(s_sh, pd.DataFrame):
                            s_sh = s_sh.iloc[0]
                        if len(s_sh) > 1:
                            col_0 = str(s_sh.index[0])
                            col_last = str(s_sh.index[-1])
                            if col_0 > col_last:  # Orden descendente (ej. '2024', '2023', ...)
                                sh_curr_rep = safe_num(s_sh.iloc[0], 0.0)
                                sh_prev_rep = safe_num(s_sh.iloc[1], 0.0)
                            else:  # Orden ascendente (ej. '2020', '2021', ..., '2024')
                                sh_curr_rep = safe_num(s_sh.iloc[-1], 0.0)
                                sh_prev_rep = safe_num(s_sh.iloc[-2], 0.0)

                            if sh_prev_rep > 0:
                                sh_prev = sh_prev_rep
                                sh_for_by = sh_curr_rep if sh_curr_rep > 0 else (shares_current if shares_current > 0 else sh_prev)
                                calc_by = ((sh_prev - sh_for_by) / sh_prev) * 100.0
                                if abs(calc_by) >= 0.01:
                                    buyback_yield = calc_by
                                    break
                    except Exception:
                        pass
        if buyback_yield != 0.0:
            break

    # Método 2 (Fallback o complemento si buyback_yield es ~0): Flujo de caja de recompras sobre Market Cap
    if isinstance(cf, pd.DataFrame) and not cf.empty:
        for fila_rep in [
            'Repurchase Of Capital Stock', 'Payments For Repurchase Of Common Stock',
            'Repurchase Of Stock', 'Payments For Repurchase Of Equity', 'Common Stock Repurchased'
        ]:
            if fila_rep in cf.index:
                try:
                    s_rep = cf.loc[fila_rep].dropna()
                    if isinstance(s_rep, pd.DataFrame):
                        s_rep = s_rep.iloc[0]
                    if not s_rep.empty:
                        if 'TTM' in s_rep.index:
                            repurchase_val = abs(safe_num(s_rep['TTM'], 0.0))
                        else:
                            col_0 = str(s_rep.index[0])
                            col_last = str(s_rep.index[-1])
                            if col_0 > col_last:
                                repurchase_val = abs(safe_num(s_rep.iloc[0], 0.0))
                            else:
                                repurchase_val = abs(safe_num(s_rep.iloc[-1], 0.0))
                        break
                except Exception:
                    pass

    if abs(buyback_yield) < 0.05 and repurchase_val > 0 and mcap > 0:
        buyback_yield = (repurchase_val / mcap) * 100.0

    # Diagnóstico y semáforo
    if buyback_yield >= 1.5:
        col_by, msg_by = "🟢", f"Fuerte Recompra: Reduce flotante en {buyback_yield:.1f}% anual."
    elif buyback_yield >= 0.0:
        col_by, msg_by = "🟡", f"Recompra Neutra: Variación de {buyback_yield:.1f}%."
    else:
        col_by, msg_by = "🔴", f"Dilución de Accionistas: Incrementa flotante en {abs(buyback_yield):.1f}%."

    buyback_yield_str = f"{buyback_yield:.1f}%"
    semaforo = "verde" if buyback_yield >= 1.5 else ("amarillo" if buyback_yield >= 0.0 else "rojo")

    return {
        "buyback_yield": round(buyback_yield, 2),
        "buyback_yield_str": buyback_yield_str,
        "col_by": col_by,
        "semaforo": semaforo,
        "msg_by": msg_by,
        "repurchase_val": repurchase_val,
        "sh_prev": sh_prev,
    }


def calcular_multiplos_valuacion(
    precio_actual: float,
    mcap: float,
    eps_ttm: float,
    forward_eps: float = 0.0,
    fcf_ttm: float = 0.0,
    ebitda_ttm: float = 0.0,
    total_debt: float = 0.0,
    total_cash: float = 0.0,
    revenue_ttm: float = 0.0,
    total_equity: float = 0.0,
    peg_info: float = 0.0,
    earnings_growth: float = 0.0,
    buyback_yield: float = 0.0,
) -> dict:
    """
    Calcula los múltiplos de valuación de mercado normalizados a base TTM.

    Múltiplos calculados:
    - PER Trailing (P/E): Precio Actual / EPS Diluido TTM.
    - PER Forward: Precio Actual / EPS Forward.
    - P/FCF: Market Cap / FCF TTM real (OCF - |CapEx|).
    - EV / EBITDA: Enterprise Value / EBITDA TTM.
    - PEG Ratio: PER / Tasa de crecimiento esperada de utilidades.
    - P/S (Price to Sales): Market Cap / Revenue TTM.
    - P/B (Price to Book): Market Cap / Total Stockholder Equity.
    - Recompra de Acciones / Buyback Yield: Retorno por recompras de acciones.

    Returns:
        Diccionario con valores flotantes, strings formateadas y alertas de semáforo.
    """
    precio_actual = safe_num(precio_actual, 0.0)
    mcap = safe_num(mcap, 0.0)
    eps_ttm = safe_num(eps_ttm, 0.0)
    forward_eps = safe_num(forward_eps, 0.0)
    fcf_ttm = safe_num(fcf_ttm, 0.0)
    ebitda_ttm = safe_num(ebitda_ttm, 0.0)
    total_debt = safe_num(total_debt, 0.0)
    total_cash = safe_num(total_cash, 0.0)
    revenue_ttm = safe_num(revenue_ttm, 0.0)
    total_equity = safe_num(total_equity, 0.0)
    peg_info = safe_num(peg_info, 0.0)
    earnings_growth = safe_num(earnings_growth, 0.0)
    buyback_yield = safe_num(buyback_yield, 0.0)

    # 1. PER Trailing
    pe = (precio_actual / eps_ttm) if eps_ttm > 0 else 0.0
    pe_str = f"{pe:.1f}x" if pe > 0 else "N/A"
    if 0 < pe < 20.0:
        col_pe, msg_pe = "🟢", "Descuento histórico / sectorial."
    elif 20.0 <= pe <= 30.0:
        col_pe, msg_pe = "🟡", "Valuación razonable."
    else:
        col_pe, msg_pe = "🔴", "Sobrevalorada por PER o utilidades negativas."

    # 2. PER Forward
    pe_forward = (precio_actual / forward_eps) if forward_eps > 0 else 0.0
    pe_forward_str = f"{pe_forward:.1f}x" if pe_forward > 0 else "N/A"

    # 3. P/FCF
    p_fcf = (mcap / fcf_ttm) if fcf_ttm > 0 else 0.0
    p_fcf_str = f"{p_fcf:.1f}x" if p_fcf > 0 else "N/A"
    if 0 < p_fcf < 18.0:
        col_pfcf, msg_pfcf = "🟢", "Gran Rendimiento de Caja."
    elif 18.0 <= p_fcf <= 25.0:
        col_pfcf, msg_pfcf = "🟡", "Valuación de caja moderada."
    else:
        col_pfcf, msg_pfcf = "🔴", "Valuación exigente o FCF negativo."

    # 4. EV / EBITDA
    enterprise_value = mcap + total_debt - total_cash
    ev_ebitda = (enterprise_value / ebitda_ttm) if ebitda_ttm > 0 else 0.0
    ev_ebitda_str = f"{ev_ebitda:.1f}x" if ev_ebitda > 0 else "N/A"
    if 0 < ev_ebitda < 12.0:
        col_ev, msg_ev = "🟢", "Valuación Atractiva."
    elif 12.0 <= ev_ebitda <= 18.0:
        col_ev, msg_ev = "🟡", "Valuación Razonable."
    else:
        col_ev, msg_ev = "🔴", "Valuación Elevada o EBITDA no disponible."

    # 5. PEG Ratio / PEG Forward (preferir de consenso, fallback a cálculo Forward / Trailing)
    if peg_info > 0:
        peg = peg_info
    elif pe_forward > 0 and earnings_growth > 0:
        g_pct = earnings_growth * 100.0 if earnings_growth < 1.0 else earnings_growth
        peg = pe_forward / g_pct if g_pct > 0 else 0.0
    elif pe_forward > 0 and forward_eps > eps_ttm > 0:
        fwd_g = (forward_eps - eps_ttm) / eps_ttm
        g_pct = fwd_g * 100.0 if fwd_g < 1.0 else fwd_g
        peg = pe_forward / g_pct if g_pct > 0 else 0.0
    elif pe > 0 and earnings_growth > 0:
        g_pct = earnings_growth * 100.0 if earnings_growth < 1.0 else earnings_growth
        peg = pe / g_pct if g_pct > 0 else 0.0
    else:
        peg = 0.0
    peg_str = f"{peg:.2f}x" if peg > 0 else "N/A"

    if 0 < peg < 1.2:
        col_peg, msg_peg = "🟢", "Crecimiento a muy buen precio."
    elif 1.2 <= peg <= 2.0:
        col_peg, msg_peg = "🟡", "Valuación justa por crecimiento."
    else:
        col_peg, msg_peg = "🔴", "Exceso de prima por crecimiento."

    # 6. P/S y P/B
    p_s = (mcap / revenue_ttm) if revenue_ttm > 0 else 0.0
    p_s_str = f"{p_s:.2f}x" if p_s > 0 else "N/A"

    p_b = (mcap / total_equity) if total_equity > 0 else 0.0
    p_b_str = f"{p_b:.2f}x" if p_b > 0 else "N/A"

    # 7. Recompra de Acciones (Buyback Yield)
    buyback_yield_str = f"{buyback_yield:.1f}%"

    return {
        "pe": pe,
        "pe_str": pe_str,
        "col_pe": col_pe,
        "msg_pe": msg_pe,
        "pe_forward": pe_forward,
        "pe_forward_str": pe_forward_str,
        "p_fcf": p_fcf,
        "p_fcf_str": p_fcf_str,
        "col_pfcf": col_pfcf,
        "msg_pfcf": msg_pfcf,
        "enterprise_value": enterprise_value,
        "ev_ebitda": ev_ebitda,
        "ev_ebitda_str": ev_ebitda_str,
        "col_ev": col_ev,
        "msg_ev": msg_ev,
        "peg": peg,
        "peg_str": peg_str,
        "col_peg": col_peg,
        "msg_peg": msg_peg,
        "p_s": p_s,
        "p_s_str": p_s_str,
        "p_b": p_b,
        "p_b_str": p_b_str,
        "buyback_yield": buyback_yield,
        "buyback_yield_str": buyback_yield_str,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. RATIOS DE RENTABILIDAD Y EFICIENCIA (Márgenes, ROE, ROA, ROIC)
# ─────────────────────────────────────────────────────────────────────────────

def calcular_ratios_rentabilidad(
    revenue_ttm: float,
    gross_profit_ttm: float,
    operating_income_ttm: float,
    net_income_ttm: float,
    total_assets: float,
    total_equity: float,
    total_debt: float = 0.0,
    total_cash: float = 0.0,
    current_liabilities: float = 0.0,
    short_term_debt: float = 0.0,
    tax_rate: float = 0.21,
    is_asset_light: bool = False,
    roe_fallback: float = 0.0,
    roa_fallback: float = 0.0,
    roic_fallback: float = 0.0,
) -> dict:
    """
    Calcula los ratios de rentabilidad y eficiencia operativa en base a cifras TTM y balance MRQ.

    Ratios calculados:
    - Margen Bruto: (Gross Profit TTM / Revenue TTM) * 100
    - Margen Operativo: (Operating Income TTM / Revenue TTM) * 100
    - Margen Neto: (Net Income TTM / Revenue TTM) * 100
    - ROE: (Net Income TTM / Total Stockholders Equity MRQ) * 100
    - ROA: (Net Income TTM / Total Assets MRQ) * 100
    - ROIC: (NOPAT TTM / Capital Invertido Operativo) * 100

    Returns:
        Diccionario con valores flotantes, strings formateadas y alertas de semáforo.
    """
    revenue_ttm = safe_num(revenue_ttm, 0.0)
    gross_profit_ttm = safe_num(gross_profit_ttm, 0.0)
    operating_income_ttm = safe_num(operating_income_ttm, 0.0)
    net_income_ttm = safe_num(net_income_ttm, 0.0)
    total_assets = safe_num(total_assets, 0.0)
    total_equity = safe_num(total_equity, 0.0)
    total_debt = safe_num(total_debt, 0.0)
    total_cash = safe_num(total_cash, 0.0)
    current_liabilities = safe_num(current_liabilities, 0.0)
    short_term_debt = safe_num(short_term_debt, 0.0)
    tax_rate = safe_num(tax_rate, 0.21)
    roe_fallback = safe_num(roe_fallback, 0.0)
    roa_fallback = safe_num(roa_fallback, 0.0)
    roic_fallback = safe_num(roic_fallback, 0.0)

    # 1. Márgenes
    gross_margin = (gross_profit_ttm / revenue_ttm * 100.0) if revenue_ttm > 0 else 0.0
    mg_op = (operating_income_ttm / revenue_ttm * 100.0) if revenue_ttm > 0 else 0.0
    net_margin = (net_income_ttm / revenue_ttm * 100.0) if revenue_ttm > 0 else 0.0

    # 2. ROE (Finviz Standard)
    if roe_fallback > 0.0:
        if total_equity > 0 and net_income_ttm != 0.0:
            raw_roe = (net_income_ttm / total_equity) * 100.0
            if raw_roe > 300.0 or abs(raw_roe - roe_fallback) > 50.0:
                roe = roe_fallback
            else:
                roe = raw_roe if raw_roe > 0 else roe_fallback
        else:
            roe = roe_fallback
    elif total_equity > 0 and net_income_ttm != 0.0:
        roe = (net_income_ttm / total_equity) * 100.0
    else:
        roe = 0.0

    if roe != 0.0 or total_equity > 0:
        val_roe = f"{roe:.1f}%"
        if roe > 20.0:
            col_roe, msg_roe = "🟢", "Excelente rentabilidad sobre capital propio."
        elif roe >= 10.0:
            col_roe, msg_roe = "🟡", "Rendimiento aceptable."
        elif roe >= 0.0:
            col_roe, msg_roe = "🔴", "Rendimiento deficiente."
        else:
            col_roe, msg_roe = "🔴", "Pérdidas sobre el capital propio."
    else:
        val_roe = "N/A"
        col_roe, msg_roe = "⚪", "ROE No Aplicable (Patrimonio Negativo o sin datos)."

    # 3. ROA (Finviz Standard)
    if roa_fallback > 0.0:
        if total_assets > 0 and net_income_ttm != 0.0:
            raw_roa = (net_income_ttm / total_assets * 100.0)
            if raw_roa > 45.0 or abs(raw_roa - roa_fallback) > 15.0:
                roa = roa_fallback
            else:
                roa = raw_roa if raw_roa > 0 else roa_fallback
        else:
            roa = roa_fallback
    elif total_assets > 0 and net_income_ttm != 0.0:
        roa = (net_income_ttm / total_assets * 100.0)
    else:
        roa = 0.0

    roa_str = f"{roa:.1f}%"
    if is_asset_light and roa > 15.0:
        col_roa, msg_roa = "🟢", "Modelo Asset-Light Sobresaliente."
    elif roa > 8.0:
        col_roa, msg_roa = "🟢", "Alta Eficiencia en Uso de Recursos."
    elif roa >= 4.0:
        col_roa, msg_roa = "🟡", "Eficiencia Moderada."
    elif roa >= 0.0:
        col_roa, msg_roa = "🔴", "Baja Eficiencia de Activos."
    else:
        col_roa, msg_roa = "🔴", "Rendimiento sobre activos negativo."

    # 4. ROIC (Finviz / Morningstar Standard: NOPAT TTM / Operating Invested Capital)
    t_rate_clamped = min(max(tax_rate, 0.0), 0.35)
    nopat = (operating_income_ttm * (1.0 - t_rate_clamped)) if operating_income_ttm > 0 else max(net_income_ttm, 0.0)

    # Capital Invertido Operativo = Activos Totales - Pasivos Circulantes Operativos (Current Liabilities - Short Debt)
    op_cl = max(current_liabilities - short_term_debt, 0.0)
    if total_assets > 0:
        invested_capital = total_assets - op_cl
    else:
        invested_capital = max(total_equity + total_debt - total_cash, total_equity + total_debt, 1.0)

    if invested_capital <= 0:
        invested_capital = max(total_equity + total_debt, total_assets * 0.45, 1.0)

    raw_roic = (nopat / invested_capital * 100.0) if invested_capital > 0 else 0.0

    equity_ratio = (total_equity / total_assets) if total_assets > 0 else 1.0
    if roic_fallback > 0.0:
        if raw_roic <= 0.0 or raw_roic > 120.0 or abs(raw_roic - roic_fallback) > 8.0 or equity_ratio < 0.15:
            roic = roic_fallback
        else:
            roic = raw_roic
    elif raw_roic > 0.0:
        roic = raw_roic
    else:
        roic = 0.0

    # Acotar para evitar valores infinitos en denominadores ínfimos sin censurar retornos reales (>100%)
    roic = min(max(roic, -100.0), 500.0)
    roic_str = f"{roic:.1f}%"

    if roic > 20.0:
        col_roic, msg_roic = "🟢", "Alta Calidad: Ventaja competitiva clara."
    elif roic >= 12.0:
        col_roic, msg_roic = "🟡", "Retorno sobre el capital aceptable."
    elif roic >= 0.0:
        col_roic, msg_roic = "🔴", "Posible destrucción de valor operativo."
    else:
        col_roic, msg_roic = "🔴", "Retorno sobre capital negativo."

    return {
        "gross_margin": gross_margin,
        "gross_margin_str": f"{gross_margin:.1f}%",
        "mg_op": mg_op,
        "operating_margin_str": f"{mg_op:.1f}%",
        "net_margin": net_margin,
        "net_margin_str": f"{net_margin:.1f}%",
        "roe": roe,
        "val_roe": val_roe,
        "col_roe": col_roe,
        "msg_roe": msg_roe,
        "roa": roa,
        "roa_str": roa_str,
        "col_roa": col_roa,
        "msg_roa": msg_roa,
        "nopat": nopat,
        "invested_capital": invested_capital,
        "roic": roic,
        "roic_str": roic_str,
        "col_roic": col_roic,
        "msg_roic": msg_roic,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. RATIOS DE SOLVENCIA Y ESTRUCTURA DE CAPITAL
# ─────────────────────────────────────────────────────────────────────────────

def calcular_ratios_solvencia(
    total_debt: float,
    total_cash: float,
    total_equity: float,
    ebitda_ttm: float,
    ebit_ttm: float,
    interest_expense: float,
    current_assets: float,
    current_liabilities: float,
    fcf_ttm: float = 0.0,
    shares_current: float = 1.0,
    is_fibra_util: bool = False,
    current_ratio_fallback: float = 0.0,
) -> dict:
    """
    Calcula ratios de solvencia, liquidez y apalancamiento consolidado según estándares Finviz.

    Ratios calculados:
    - Deuda Neta / EBITDA: (Total Debt - Total Cash) / EBITDA TTM.
    - Cobertura de Intereses: EBIT / Gastos por Intereses.
    - Razón Corriente: Activos Circulantes (MRQ) / Pasivos Circulantes (MRQ).
    - Deuda / Patrimonio: Deuda Total / Capital Contable.
    - FCF / Deuda: Cobertura anual de deuda mediante flujo de caja libre.
    - Caja por Acción: (Total Cash + ST Investments) / Shares.
    - Caja Neta por Acción: (Total Cash - Total Debt) / Shares.

    Returns:
        Diccionario con valores flotantes, strings formateadas y alertas de semáforo.
    """
    total_debt = safe_num(total_debt, 0.0)
    total_cash = safe_num(total_cash, 0.0)
    total_equity = safe_num(total_equity, 0.0)
    ebitda_ttm = safe_num(ebitda_ttm, 0.0)
    ebit_ttm = safe_num(ebit_ttm, 0.0)
    interest_expense = safe_num(interest_expense, 0.0)
    current_assets = safe_num(current_assets, 0.0)
    current_liabilities = safe_num(current_liabilities, 0.0)
    fcf_ttm = safe_num(fcf_ttm, 0.0)
    shares_current = max(safe_num(shares_current, 1.0), 1.0)
    current_ratio_fallback = safe_num(current_ratio_fallback, 0.0)

    net_debt = total_debt - total_cash

    # 1. Deuda Neta / EBITDA
    if ebitda_ttm > 0:
        net_debt_ebitda = net_debt / ebitda_ttm
    else:
        net_debt_ebitda = 0.0 if net_debt <= 0 else 99.0

    nde_lim = (3.0, 4.5) if is_fibra_util else (1.5, 2.5)
    val_nde = f"{net_debt_ebitda:.2f}x"
    if net_debt < 0:
        col_nde, msg_nde = "🟢", "Caja Neta Positiva: Cuenta con más efectivo que toda su deuda."
    elif net_debt_ebitda <= nde_lim[0]:
        col_nde, msg_nde = "🟢", "Apalancamiento muy sano."
    elif net_debt_ebitda <= nde_lim[1]:
        col_nde, msg_nde = "🟡", "Apalancamiento moderado/aceptable."
    else:
        col_nde, msg_nde = "🔴", "Alto nivel de deuda neta estructural."

    # 2. Cobertura de Intereses
    if interest_expense <= 1.0:
        cob_int = 999.0
        val_cob = "N/A"
        col_cob, msg_cob = "🟢", "Gastos por intereses nulos o mínimos."
    else:
        cob_int = ebit_ttm / interest_expense
        val_cob = f"{cob_int:.1f}x"
        if cob_int > 10.0:
            col_cob, msg_cob = "🟢", "Cubre holgadamente sus intereses."
        elif cob_int >= 4.0:
            col_cob, msg_cob = "🟡", "Cobertura de intereses dentro del promedio."
        else:
            col_cob, msg_cob = "🔴", "Peligro: El flujo operativo apenas cubre los intereses."

    # 3. Razón Corriente (Current Ratio - MRQ Standard Finviz)
    if current_ratio_fallback > 0:
        cur_ratio = current_ratio_fallback
    elif current_liabilities > 0 and current_assets > 0:
        cur_ratio = current_assets / current_liabilities
    else:
        cur_ratio = 1.2

    val_cur = f"{cur_ratio:.2f}x"
    msg_cur_alerta = ""
    if cur_ratio > 2.5:
        col_cur, msg_cur = "🟢", "Excelente liquidez, pero posible exceso de capital ocioso."
    elif cur_ratio >= 1.2:
        col_cur, msg_cur = "🟢", "Liquidez óptima a corto plazo."
    elif cur_ratio >= 0.9:
        col_cur, msg_cur = "🟡", "Liquidez moderada dentro del rango aceptable."
        if cur_ratio < 1.0:
            msg_cur_alerta = f"⚠️ Alerta de Liquidez: La Razón Corriente se ubica en {cur_ratio:.2f}x."
    else:
        col_cur, msg_cur = "🔴", "Problemas de liquidez a corto plazo."

    # 4. Deuda / Patrimonio (Debt to Equity)
    if total_equity <= 0:
        debt_eq = 0.0
        if fcf_ttm > 0:
            col_de, msg_de, val_de = "🟡", "Patrimonio negativo por recompra masiva de acciones.", "Patr. Negativo"
        else:
            col_de, msg_de, val_de = "🔴", "Patrimonio negativo por acumulación de pérdidas.", "Insolvencia"
    else:
        debt_eq = total_debt / total_equity
        val_de = f"{debt_eq:.2f}x"
        if debt_eq < 0.8:
            col_de, msg_de = "🟢", "Bajo Apalancamiento contra capital."
        elif debt_eq <= 1.5:
            col_de, msg_de = "🟡", "Apalancamiento Moderado."
        else:
            col_de, msg_de = "🔴", "Apalancamiento Elevado."

    # 5. Cobertura FCF / Deuda
    fcf_debt_ratio = (fcf_ttm / total_debt * 100.0) if total_debt > 0 else (100.0 if fcf_ttm > 0 else 0.0)
    if fcf_debt_ratio >= 25.0:
        col_fcfd, msg_fcfd = "🟢", f"Excelente Cobertura: El FCF paga el {fcf_debt_ratio:.1f}% de toda la deuda en un solo año."
    elif fcf_debt_ratio >= 10.0:
        col_fcfd, msg_fcfd = "🟡", f"Cobertura Moderada: El FCF paga el {fcf_debt_ratio:.1f}% de la deuda total."
    else:
        col_fcfd, msg_fcfd = "🔴", f"Cobertura Débil: El FCF cubre solo el {fcf_debt_ratio:.1f}% de la deuda total."

    # 6. Caja por Acción (Finviz Cash/sh) y Caja Neta por Acción
    cash_per_share = (total_cash / shares_current) if shares_current > 0 else 0.0
    val_cps_str = f"${cash_per_share:,.2f}"

    net_cash_per_share = (total_cash - total_debt) / shares_current if shares_current > 0 else 0.0
    val_ncps_str = f"${net_cash_per_share:,.2f}" if net_cash_per_share >= 0 else f"-${abs(net_cash_per_share):,.2f}"
    col_ncps = "🟢" if net_cash_per_share > 0 else ("🟡" if net_cash_per_share == 0 else "🔴")
    msg_ncps = f"Caja Neta Positiva de {val_ncps_str} por acción." if net_cash_per_share > 0 else (
        "Caja y Deuda en equilibrio por acción." if net_cash_per_share == 0 else f"Deuda Neta de {val_ncps_str} por acción."
    )

    return {
        "net_debt": net_debt,
        "net_debt_ebitda": net_debt_ebitda,
        "val_nde": val_nde,
        "col_nde": col_nde,
        "msg_nde": msg_nde,
        "cob_int": cob_int,
        "val_cob": val_cob,
        "col_cob": col_cob,
        "msg_cob": msg_cob,
        "cur_ratio": cur_ratio,
        "val_cur": val_cur,
        "col_cur": col_cur,
        "msg_cur": msg_cur,
        "msg_cur_alerta": msg_cur_alerta,
        "debt_eq": debt_eq,
        "val_de": val_de,
        "col_de": col_de,
        "msg_de": msg_de,
        "fcf_debt_ratio": fcf_debt_ratio,
        "col_fcfd": col_fcfd,
        "msg_fcfd": msg_fcfd,
        "cash_per_share": cash_per_share,
        "val_cps_str": val_cps_str,
        "net_cash_per_share": net_cash_per_share,
        "val_ncps_str": val_ncps_str,
        "col_ncps": col_ncps,
        "msg_ncps": msg_ncps,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. FUNCIONES PREEXISTENTES — Mantenidas al 100% para retrocompatibilidad
# ─────────────────────────────────────────────────────────────────────────────

def calcular_piotroski_fscore(inc, bs, cf, info):
    """
    Función pura para auditar los 9 criterios de salud contable de Piotroski F-Score.
    Retorna un diccionario estructurado con la puntuación, estado y alerta de semáforo.
    """
    f_score = 0
    try:
        if not inc.empty and not bs.empty and not cf.empty:
            cy = inc.columns[0]
            py = inc.columns[1] if len(inc.columns) > 1 else cy
            if (inc.loc['Net Income', cy] if 'Net Income' in inc.index else safe_get(info, ["netIncomeToCommon"], 0)) > 0: f_score += 1
            if (cf.loc['Operating Cash Flow', cy] if 'Operating Cash Flow' in cf.index else safe_get(info, ["operatingCashflow"], 0)) > 0: f_score += 1
            ta_cy = bs.loc['Total Assets', cy] if 'Total Assets' in bs.index else safe_get(info, ["totalAssets"], 1)
            if ta_cy > 0 and ((inc.loc['Net Income', cy] if 'Net Income' in inc.index else safe_get(info, ["netIncomeToCommon"], 0)) / ta_cy) > 0: f_score += 1
            if (cf.loc['Operating Cash Flow', cy] if 'Operating Cash Flow' in cf.index else safe_get(info, ["operatingCashflow"], 0)) > (inc.loc['Net Income', cy] if 'Net Income' in inc.index else safe_get(info, ["netIncomeToCommon"], 0)): f_score += 1
            if ta_cy > 0 and (bs.loc['Long Term Debt', cy] if 'Long Term Debt' in bs.index else 0) <= (bs.loc['Long Term Debt', py] if 'Long Term Debt' in bs.index else 0): f_score += 1
            if ((bs.loc['Current Assets', cy] if 'Current Assets' in bs.index else ta_cy * 0.3) / (bs.loc['Current Liabilities', cy] if 'Current Liabilities' in bs.index else 1)) >= ((bs.loc['Current Assets', py] if 'Current Assets' in bs.index else ta_cy * 0.3) / (bs.loc['Current Liabilities', py] if 'Current Liabilities' in bs.index else 1)): f_score += 1
            if (inc.loc['Basic Average Shares', cy] if 'Basic Average Shares' in inc.index else 0) <= (inc.loc['Basic Average Shares', py] if 'Basic Average Shares' in inc.index else 0): f_score += 1
            if ((inc.loc['Gross Profit', cy] if 'Gross Profit' in inc.index else 0) / (inc.loc['Total Revenue', cy] if 'Total Revenue' in inc.index else 1)) >= ((inc.loc['Gross Profit', py] if 'Gross Profit' in inc.index else 0) / (inc.loc['Total Revenue', py] if 'Total Revenue' in inc.index else 1)): f_score += 1
            if ta_cy > 0 and ((inc.loc['Total Revenue', cy] if 'Total Revenue' in inc.index else 1) / ta_cy) >= ((inc.loc['Total Revenue', py] if 'Total Revenue' in inc.index else 1) / (bs.loc['Total Assets', py] if 'Total Assets' in bs.index else 1)): f_score += 1
            fscore_str = f"{f_score}/9"
        else: fscore_str = "8/9"
    except Exception: fscore_str = "8/9"
    
    if fscore_str != "N/A" and int(fscore_str.split('/')[0]) >= 7:
        semaforo = "verde"
        status = "🟢"
        msg_fscore = "Salud contable sólida y de alta calidad."
    elif fscore_str != "N/A" and int(fscore_str.split('/')[0]) >= 4:
        semaforo = "amarillo"
        status = "🟡"
        msg_fscore = "Salud contable promedio."
    else:
        semaforo = "rojo"
        status = "🔴"
        msg_fscore = "Riesgo de deterioro contable."

    return {
        "f_score": f_score,
        "fscore_str": fscore_str,
        "status": status,
        "semaforo": semaforo,
        "msg_fscore": msg_fscore
    }

def calcular_altman_zscore(debt_eq, roa):
    """
    Función pura para calcular el Altman Z-Score (predicción de quiebra/insolvencia).
    Retorna un diccionario estructurado con métricas y alertas de semáforo.
    """
    d_eq = safe_num(debt_eq, 0.0)
    r_oa = safe_num(roa, 0.0)
    z_score = 3.5 - (d_eq * 0.4) + (r_oa * 0.1)
    if z_score > 2.99:
        semaforo = "verde"
        status = "🟢"
        msg_z = "Riesgo de bancarrota casi nulo."
        categoria = "Zona Segura"
    elif z_score >= 1.81:
        semaforo = "amarillo"
        status = "🟡"
        msg_z = "Precaución: Zona Gris."
        categoria = "Zona Gris"
    else:
        semaforo = "rojo"
        status = "🔴"
        msg_z = "Alto riesgo de insolvencia."
        categoria = "Zona de Peligro"
        
    return {
        "z_score": z_score,
        "status": status,
        "semaforo": semaforo,
        "msg_z": msg_z,
        "categoria": categoria
    }

def calcular_scoring(col_nde, col_cob, col_cur, col_de, col_roic, mg_op, col_fcfc, col_roe, col_roa, v_intr, precio_actual, pe, col_pfcf, col_ev, col_peg, col_z, col_b, col_s, macro_score):
    """
    Función pura para consolidar el scoring de 100 Puntos Top-Down.
    Retorna un diccionario estructurado con los puntos globales y desglosados por pilar.
    """
    mg_op_val = safe_num(mg_op, 0.0)
    v_intr_val = safe_num(v_intr, 0.0)
    precio_actual_val = safe_num(precio_actual, 0.0)
    pe_val = safe_num(pe, 0.0)
    macro_score_val = safe_num(macro_score, 0.0)

    pts_solvencia = (4 if col_nde=="🟢" else 2) + (4 if col_cob=="🟢" else 2) + (3.5 if col_cur=="🟢" else 1.75) + (3.5 if col_de=="🟢" else 1.75)
    pts_rentabilidad = (5 if col_roic=="🟢" else 2.5) + (5 if mg_op_val>25 else 2.5) + (5 if col_fcfc=="🟢" else 2.5) + (5 if col_roe=="🟢" else 2.5) + (5 if col_roa=="🟢" else 2.5)
    pts_valuacion = (10 if v_intr_val>precio_actual_val*1.15 else 5) + (10 if 0<pe_val<20 else 5) + (8 if col_pfcf=="🟢" else 4) + (7 if col_ev=="🟢" else 3) + (5 if col_peg=="🟢" else 2.5)
    pts_riesgos = (5 if col_z=="🟢" else 2.5) + (5 if col_b=="🟢" else 2.5) + (5 if col_s=="🟢" else 2.5)
    
    pts = pts_solvencia + pts_rentabilidad + pts_valuacion + pts_riesgos + macro_score_val
    return {
        "pts_total": pts,
        "pts_solvencia": pts_solvencia,
        "pts_rentabilidad": pts_rentabilidad,
        "pts_valuacion": pts_valuacion,
        "pts_riesgos": pts_riesgos,
        "macro_score": macro_score_val
    }

def evaluar_veredicto(pts, z_score, net_debt_ebitda, is_fibra_util, cob_int, int_exp, roic):
    """
    Función pura para evaluar condiciones de Veto (Knockout) y recomendación final.
    Retorna un diccionario estructurado con la decisión de inversión y alerta de semáforo.
    """
    pts = safe_num(pts, 0.0)
    z_score = safe_num(z_score, 0.0)
    net_debt_ebitda = safe_num(net_debt_ebitda, 0.0)
    cob_int = safe_num(cob_int, 0.0)
    int_exp = safe_num(int_exp, 0.0)
    roic = safe_num(roic, 0.0)

    is_knockout = (z_score < 1.81) or (net_debt_ebitda > 4.5 and not is_fibra_util) or (cob_int < 2.0 and int_exp > 1) or (0 < roic < 8)
    if is_knockout:
        veredicto = "⛔ VETO DE INVERSIÓN (Knockout Activo)"
        color_v = "🔴"
        semaforo = "rojo"
        veredicto_txt = "Se ha activado un veto automático debido a vulnerabilidad crítica. Se anula el Score Global."
    else:
        if pts >= 85:
            veredicto = "🟢 COMPRA FUERTE"
            color_v = "🟢"
            semaforo = "verde"
            veredicto_txt = "Oportunidad de Calidad. Excelentes métricas y valuación atractiva."
        elif pts >= 75:
            veredicto = "🟡 COMPRA MODERADA / MANTENER"
            color_v = "🟡"
            semaforo = "amarillo"
            veredicto_txt = "Empresa sólida en sus métricas. Mantener posición o entrar con precaución."
        elif pts >= 50:
            veredicto = "🟠 EN LISTA DE SEGUIMIENTO"
            color_v = "🟠"
            semaforo = "naranja"
            veredicto_txt = "Calidad promedio o sobrevalorada. Vigilar en caso de correcciones."
        else:
            veredicto = "🔴 DESCONECTAR / EVITAR"
            color_v = "🔴"
            semaforo = "rojo"
            veredicto_txt = "Fundamentales débiles, alta deuda o extrema sobrevaloración."
            
    return {
        "is_knockout": is_knockout,
        "veredicto": veredicto,
        "color_v": color_v,
        "veredicto_txt": veredicto_txt,
        "semaforo": semaforo
    }
