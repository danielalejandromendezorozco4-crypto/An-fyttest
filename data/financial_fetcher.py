from __future__ import annotations

import datetime
import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests
import streamlit as st

from config.settings import (
    FINNHUB_BASE_URL,
    FINNHUB_CACHE_TTL_METRICS,
    FINNHUB_CACHE_TTL_NEWS,
    FINNHUB_CACHE_TTL_QUOTE,
    FRED_CACHE_TTL,
    safe_get,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 1. CLIENTE Y SESIÓN HTTP DEFENSIVA PARA FINNHUB API
# ─────────────────────────────────────────────────────────────────────────────

def obtener_session_finnhub(api_key: str = "") -> requests.Session:
    """
    Crea y configura una sesión de requests optimizada con headers estándar,
    timeouts y autenticación para Finnhub API.
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": "An-FyT/2.0 (Institutional Equity Research & Fundamental Valuation Engine)",
        "Accept": "application/json",
    })
    if api_key:
        session.headers["X-Finnhub-Token"] = api_key.strip()
    return session


def _finnhub_get(
    endpoint: str,
    params: Optional[Dict[str, Any]] = None,
    api_key: str = "",
    timeout: float = 8.0,
) -> Optional[Union[Dict[str, Any], List[Any]]]:
    """
    Realiza una petición GET segura y resiliente a la API de Finnhub.
    Maneja defensivamente códigos HTTP 429 (Rate Limit), 401/403 (Auth) y timeouts.
    """
    url = f"{FINNHUB_BASE_URL}/{endpoint.lstrip('/')}"
    query_params = dict(params or {})
    if api_key and "token" not in query_params:
        query_params["token"] = api_key.strip()

    try:
        session = obtener_session_finnhub(api_key=api_key)
        response = session.get(url, params=query_params, timeout=timeout)

        if response.status_code == 200:
            try:
                return response.json()
            except Exception:
                return None
        elif response.status_code == 429:
            logger.warning("Finnhub API Rate Limit Exceeded (HTTP 429).")
            return None
        elif response.status_code in (401, 403):
            logger.warning("Finnhub API Key no válida o no autorizada (HTTP %s).", response.status_code)
            return None
        else:
            logger.warning("Finnhub API respondió con código HTTP %s para %s", response.status_code, endpoint)
            return None
    except (requests.RequestException, Exception) as exc:
        logger.debug("Excepción durante petición GET a Finnhub [%s]: %s", endpoint, exc)
        return None


def safe_num(val: Any, default: float = 0.0) -> float:
    """
    Convierte cualquier valor de forma segura a float o al valor por defecto especificado.
    Maneja None, np.nan, float('nan'), inf, -inf, strings formateados ('$1,250.50', '15.5%') y tipos corruptos.
    """
    if val is None:
        return float(default) if default is not None else 0.0
    try:
        if isinstance(val, (int, float)):
            if math.isnan(val) or math.isinf(val) or np.isnan(val) or np.isinf(val):
                return float(default) if default is not None else 0.0
            return float(val)
        if isinstance(val, str):
            clean_str = val.replace(',', '').replace('$', '').replace('%', '').strip()
            if not clean_str or clean_str.lower() in ('nan', 'none', 'n/a', 'null', 'inf', '-inf'):
                return float(default) if default is not None else 0.0
            return float(clean_str)
        if pd.isna(val):
            return float(default) if default is not None else 0.0
        return float(val)
    except (ValueError, TypeError, Exception):
        return float(default) if default is not None else 0.0


def _normalizar_nombre_fila(nombre: str) -> str:
    """Normaliza un nombre de fila eliminando espacios, guiones y mayúsculas para matching flexible."""
    if not isinstance(nombre, str):
        return str(nombre).lower().strip()
    return nombre.lower().replace(" ", "").replace("_", "").replace("-", "").strip()


def _extraer_val_df(df: Any, posibles_filas: List[str], default: float = 0.0) -> float:
    """
    Extrae el valor más reciente de un DataFrame buscando en las filas candidatas
    usando primero búsqueda exacta y luego búsqueda normalizada insensible a mayúsculas/espacios.
    """
    if isinstance(df, pd.DataFrame) and not df.empty:
        # 1. Búsqueda exacta
        for f in posibles_filas:
            if f in df.index:
                try:
                    elem = df.loc[f]
                    if isinstance(elem, pd.DataFrame):
                        elem = elem.iloc[0]
                    serie = elem.dropna()
                    if not serie.empty:
                        val = safe_num(serie.iloc[0], default=None)
                        if val is not None:
                            return val
                except Exception:
                    pass
        # 2. Búsqueda normalizada
        index_map = {_normalizar_nombre_fila(idx): idx for idx in df.index}
        for f in posibles_filas:
            norm_f = _normalizar_nombre_fila(f)
            if norm_f in index_map:
                try:
                    elem = df.loc[index_map[norm_f]]
                    if isinstance(elem, pd.DataFrame):
                        elem = elem.iloc[0]
                    serie = elem.dropna()
                    if not serie.empty:
                        val = safe_num(serie.iloc[0], default=None)
                        if val is not None:
                            return val
                except Exception:
                    pass
    return float(default)


def _extraer_serie(
    df: pd.DataFrame,
    posibles_filas: List[str],
    absval: bool = False,
) -> List[float]:
    """
    Extrae una serie histórica de un DataFrame (índice = conceptos, columnas = fechas)
    buscando secuencialmente los nombres de fila en ``posibles_filas`` mediante matching exacto y normalizado.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return []
    # 1. Búsqueda exacta
    for fila in posibles_filas:
        if fila in df.index:
            try:
                elem = df.loc[fila]
                if isinstance(elem, pd.DataFrame):
                    elem = elem.iloc[0]
                serie = elem.dropna()
                if not serie.empty:
                    return [
                        abs(safe_num(v, 0.0)) if absval else safe_num(v, 0.0)
                        for v in serie.values
                    ]
            except Exception:
                pass
    # 2. Búsqueda normalizada
    index_map = {_normalizar_nombre_fila(idx): idx for idx in df.index}
    for fila in posibles_filas:
        norm_f = _normalizar_nombre_fila(fila)
        if norm_f in index_map:
            try:
                elem = df.loc[index_map[norm_f]]
                if isinstance(elem, pd.DataFrame):
                    elem = elem.iloc[0]
                serie = elem.dropna()
                if not serie.empty:
                    return [
                        abs(safe_num(v, 0.0)) if absval else safe_num(v, 0.0)
                        for v in serie.values
                    ]
            except Exception:
                pass
    return []


# ─────────────────────────────────────────────────────────────────────────────
# 2. COTIZACIONES Y VELAS HISTÓRICAS (FINNHUB)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=FINNHUB_CACHE_TTL_QUOTE)
def fetch_cotizacion_intradia(ticker: str, finnhub_api_key: str = "") -> Tuple[float, float, pd.DataFrame]:
    """
    Obtiene la cotización en tiempo real y el histórico de 5 años de velas diarias (OHLCV)
    directamente desde los endpoints `/quote` y `/stock/candle` de Finnhub API.

    Normaliza la respuesta a un pandas.DataFrame con columnas estandarizadas:
    ['Open', 'High', 'Low', 'Close', 'Volume'] y DatetimeIndex ('Date').
    """
    ticker = str(ticker).upper().strip()
    precio_actual = 0.0
    prev_close = 0.0
    hist = pd.DataFrame()

    if not ticker:
        return precio_actual, prev_close, hist

    # CAPA 1: Endpoint Quote en tiempo real
    quote_res = _finnhub_get("quote", params={"symbol": ticker}, api_key=finnhub_api_key)
    if isinstance(quote_res, dict) and "c" in quote_res:
        precio_actual = safe_num(quote_res.get("c", 0.0))
        prev_close = safe_num(quote_res.get("pc", 0.0))
        if prev_close == 0.0 and precio_actual > 0:
            diff_d = safe_num(quote_res.get("d", 0.0))
            prev_close = precio_actual - diff_d

    # CAPA 2: Endpoint Stock Candle (5 Años de Velas Diarias)
    now_dt = datetime.datetime.now(datetime.timezone.utc)
    to_ts = int(now_dt.timestamp())
    from_ts = int((now_dt - datetime.timedelta(days=5 * 365 + 10)).timestamp())

    candle_res = _finnhub_get(
        "stock/candle",
        params={
            "symbol": ticker,
            "resolution": "D",
            "from": from_ts,
            "to": to_ts,
        },
        api_key=finnhub_api_key,
        timeout=10.0,
    )

    if isinstance(candle_res, dict) and candle_res.get("s") == "ok" and "t" in candle_res:
        try:
            timestamps = candle_res.get("t", [])
            opens = candle_res.get("o", [])
            highs = candle_res.get("h", [])
            lows = candle_res.get("l", [])
            closes = candle_res.get("c", [])
            volumes = candle_res.get("v", [])

            if timestamps and closes:
                df_c = pd.DataFrame({
                    "Open": [safe_num(x) for x in opens],
                    "High": [safe_num(x) for x in highs],
                    "Low": [safe_num(x) for x in lows],
                    "Close": [safe_num(x) for x in closes],
                    "Volume": [safe_num(x) for x in volumes],
                })
                df_c["Date"] = pd.to_datetime(timestamps, unit="s")
                df_c = df_c.set_index("Date").sort_index()
                hist = df_c

                if precio_actual == 0.0 and not hist.empty:
                    precio_actual = safe_num(hist["Close"].iloc[-1], 0.0)
                if prev_close == 0.0 and len(hist) > 1:
                    prev_close = safe_num(hist["Close"].iloc[-2], 0.0)
        except Exception as e:
            logger.debug("Error procesando velas Finnhub para %s: %s", ticker, e)

    return precio_actual, prev_close, hist


# ─────────────────────────────────────────────────────────────────────────────
# 3. DATOS FUNDAMENTALES Y ESTADOS FINANCIEROS (FINNHUB)
# ─────────────────────────────────────────────────────────────────────────────

def _map_gics_sector(finnhub_industry: str) -> str:
    """Mapea la industria/sector de Finnhub a los 11 sectores estándar GICS de An-FyT."""
    ind = str(finnhub_industry or "").lower().strip()
    if any(k in ind for k in ["health", "pharma", "biotech", "medical", "care", "drug", "therapeutic"]):
        return "Healthcare"
    if any(k in ind for k in ["tech", "software", "semiconductor", "hardware", "it services"]):
        return "Technology"
    if any(k in ind for k in ["bank", "financial", "insurance", "credit", "capital market", "asset management"]):
        return "Financial Services"
    if any(k in ind for k in ["auto", "retail", "apparel", "hotel", "restaurant", "cyclical", "luxury"]):
        return "Consumer Cyclical"
    if any(k in ind for k in ["beverage", "food", "tobacco", "household", "defensive", "staples"]):
        return "Consumer Defensive"
    if any(k in ind for k in ["industrial", "aerospace", "machinery", "transport", "defense", "logistics"]):
        return "Industrials"
    if any(k in ind for k in ["energy", "oil", "gas", "petroleum"]):
        return "Energy"
    if any(k in ind for k in ["utility", "utilities", "electric", "water"]):
        return "Utilities"
    if any(k in ind for k in ["real estate", "reit", "reits"]):
        return "Real Estate"
    if any(k in ind for k in ["communication", "telecom", "media", "entertainment", "internet"]):
        return "Communication Services"
    if any(k in ind for k in ["material", "chemical", "mining", "metal", "steel"]):
        return "Basic Materials"
    return finnhub_industry.title() if finnhub_industry else "General"


@st.cache_data(ttl=FINNHUB_CACHE_TTL_METRICS)
def fetch_datos_fundamentales(
    ticker: str, finnhub_api_key: str = ""
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Extrae, homogeneiza y estructura el perfil corporativo, múltiplos clave y estados financieros
    históricos (Income Statement, Balance Sheet, Cash Flow) a partir de Finnhub API.

    Endpoints consumidos:
    - `/stock/profile2`: Perfil, market cap en millones, shares outstanding en millones.
    - `/stock/metric?metric=all`: Ratios financieros TTM y series históricas anuales.
    - `/stock/financials-reported`: Estados financieros as-reported SEC (10-K).
    """
    ticker = str(ticker).upper().strip()
    info: Dict[str, Any] = {}
    inc = pd.DataFrame()
    bs = pd.DataFrame()
    cf = pd.DataFrame()

    if not ticker:
        return info, inc, bs, cf

    # 1. Consultas concurrentes a Finnhub
    with ThreadPoolExecutor(max_workers=3) as executor:
        f_prof = executor.submit(
            _finnhub_get, "stock/profile2", {"symbol": ticker}, finnhub_api_key
        )
        f_metric = executor.submit(
            _finnhub_get, "stock/metric", {"symbol": ticker, "metric": "all"}, finnhub_api_key
        )
        f_rep = executor.submit(
            _finnhub_get, "stock/financials-reported", {"symbol": ticker, "freq": "annual"}, finnhub_api_key
        )

        try:
            prof_data = f_prof.result(timeout=10.0) or {}
        except Exception:
            prof_data = {}

        try:
            metric_data = f_metric.result(timeout=12.0) or {}
        except Exception:
            metric_data = {}

        try:
            rep_data = f_rep.result(timeout=12.0) or {}
        except Exception:
            rep_data = {}

    # 2. Procesamiento de Perfil (/stock/profile2)
    # Nota Finnhub: marketCapitalization y shareOutstanding vienen expresados en MILLONES.
    mcap_m = safe_num(prof_data.get("marketCapitalization", 0.0))
    shares_m = safe_num(prof_data.get("shareOutstanding", 0.0))

    mcap = mcap_m * 1e6 if mcap_m > 0 else 0.0
    shares_outstanding = shares_m * 1e6 if shares_m > 0 else 0.0
    long_name = prof_data.get("name", ticker)
    industry_raw = prof_data.get("finnhubIndustry", "General")
    sector_std = _map_gics_sector(industry_raw)

    # 3. Procesamiento de Métricas Básicas (/stock/metric)
    metrics_dict = metric_data.get("metric", {}) if isinstance(metric_data, dict) else {}
    series_dict = metric_data.get("series", {}).get("annual", {}) if isinstance(metric_data, dict) else {}

    beta = safe_num(metrics_dict.get("beta", 1.0), default=1.0)
    eps_ttm = safe_num(metrics_dict.get("epsTTM", metrics_dict.get("epsNormalizedAnnual", 0.0)))
    pe_ttm = safe_num(metrics_dict.get("peTTM", metrics_dict.get("peAnnual", 0.0)))
    div_rate = safe_num(metrics_dict.get("dividendPerShareAnnual", metrics_dict.get("dividendPerShareTTM", 0.0)))
    current_ratio = safe_num(metrics_dict.get("currentRatioQuarterly", metrics_dict.get("currentRatioAnnual", 1.0)), default=1.0)
    debt_to_equity = safe_num(metrics_dict.get("totalDebt/totalEquityQuarterly", metrics_dict.get("totalDebt/totalEquityAnnual", 0.0)))
    roe_val = safe_num(metrics_dict.get("roeTTM", metrics_dict.get("roeAnnual", 0.0)))
    roa_val = safe_num(metrics_dict.get("roaTTM", metrics_dict.get("roaAnnual", 0.0)))
    roic_val = safe_num(metrics_dict.get("roicTTM", metrics_dict.get("roicAnnual", 0.0)))
    op_margins_raw = safe_num(metrics_dict.get("operatingMarginTTM", metrics_dict.get("operatingMarginAnnual", 0.0)))
    op_margins = op_margins_raw / 100.0 if op_margins_raw > 1.0 else op_margins_raw
    rev_growth_raw = safe_num(metrics_dict.get("revenueGrowthTTMYoy", metrics_dict.get("revenueGrowth3Y", 0.0)))
    rev_growth = rev_growth_raw / 100.0 if abs(rev_growth_raw) > 1.0 else rev_growth_raw
    eps_growth_raw = safe_num(metrics_dict.get("epsGrowthTTMYoy", metrics_dict.get("epsGrowth3Y", 0.0)))
    eps_growth = eps_growth_raw / 100.0 if abs(eps_growth_raw) > 1.0 else eps_growth_raw

    # 4. Construcción de DataFrames Financieros desde /stock/financials-reported
    dict_inc: Dict[str, Dict[str, float]] = {}
    dict_bs: Dict[str, Dict[str, float]] = {}
    dict_cf: Dict[str, Dict[str, float]] = {}

    rep_list = rep_data.get("data", []) if isinstance(rep_data, dict) else []
    if isinstance(rep_list, list) and len(rep_list) > 0:
        for filing in rep_list:
            year_val = filing.get("year") or filing.get("endDate", "")[:4]
            if not year_val:
                continue
            period_label = str(year_val)
            report_items = filing.get("report", {})

            # ── A. Income Statement (ic) ────────────────────────────────────
            ic_items = report_items.get("ic", [])
            ic_map = {item.get("concept", ""): safe_num(item.get("value")) for item in ic_items if "concept" in item}
            ic_labels = {item.get("label", "").lower(): safe_num(item.get("value")) for item in ic_items if "label" in item}

            def _get_val(concepts: List[str], label_kw: List[str], source_map: Dict[str, float], label_map: Dict[str, float]) -> float:
                for c in concepts:
                    if c in source_map and source_map[c] != 0.0:
                        return source_map[c]
                for kw in label_kw:
                    for l_key, val in label_map.items():
                        if kw.lower() in l_key and val != 0.0:
                            return val
                return 0.0

            rev = _get_val(
                ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet", "Revenue"],
                ["total net sales", "revenue", "total revenue", "sales"],
                ic_map, ic_labels
            )
            gross_p = _get_val(["GrossProfit"], ["gross profit"], ic_map, ic_labels) or rev
            op_inc = _get_val(["OperatingIncomeLoss", "OperatingProfit"], ["operating income", "operating profit"], ic_map, ic_labels)
            net_inc = _get_val(["NetIncomeLoss", "ProfitLoss"], ["net income", "net loss"], ic_map, ic_labels)
            int_exp = abs(_get_val(["InterestExpense", "InterestExpenseNonoperating"], ["interest expense"], ic_map, ic_labels))
            pretax_inc = _get_val(["IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments", "PretaxIncome"], ["income before tax"], ic_map, ic_labels)
            tax_prov = _get_val(["IncomeTaxExpenseBenefit"], ["income tax"], ic_map, ic_labels)
            shares_dil = _get_val(["WeightedAverageNumberOfDilutedSharesOutstanding", "WeightedAverageNumberOfSharesOutstandingDiluted"], ["diluted shares", "shares diluted"], ic_map, ic_labels)

            dict_inc[period_label] = {
                "Total Revenue": rev,
                "Gross Profit": gross_p,
                "Operating Income": op_inc,
                "Net Income": net_inc,
                "Interest Expense": int_exp,
                "Pretax Income": pretax_inc,
                "Tax Provision": tax_prov,
                "Diluted Average Shares": shares_dil or shares_outstanding,
                "Basic Average Shares": shares_dil or shares_outstanding,
            }

            # ── B. Balance Sheet (bs) ────────────────────────────────────────
            bs_items = report_items.get("bs", [])
            bs_map = {item.get("concept", ""): safe_num(item.get("value")) for item in bs_items if "concept" in item}
            bs_labels = {item.get("label", "").lower(): safe_num(item.get("value")) for item in bs_items if "label" in item}

            t_assets = _get_val(["Assets"], ["total assets"], bs_map, bs_labels)
            c_assets = _get_val(["AssetsCurrent"], ["total current assets", "current assets"], bs_map, bs_labels)
            c_liab = _get_val(["LiabilitiesCurrent"], ["total current liabilities", "current liabilities"], bs_map, bs_labels)
            t_equity = _get_val(["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"], ["total shareholders equity", "stockholders equity", "total equity"], bs_map, bs_labels)
            cash_val = _get_val(["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsAndShortTermInvestments", "CashAndCashEquivalents"], ["cash and cash equivalents", "cash"], bs_map, bs_labels)
            lt_debt = _get_val(["LongTermDebtNoncurrent", "LongTermDebt"], ["long-term debt", "long term debt"], bs_map, bs_labels)
            st_debt = _get_val(["DebtCurrent", "ShortTermBorrowings"], ["current debt", "short-term debt"], bs_map, bs_labels)
            t_debt = lt_debt + st_debt if (lt_debt > 0 or st_debt > 0) else _get_val(["TotalDebt"], ["total debt"], bs_map, bs_labels)

            dict_bs[period_label] = {
                "Total Assets": t_assets,
                "Current Assets": c_assets,
                "Total Current Assets": c_assets,
                "Current Liabilities": c_liab,
                "Total Current Liabilities": c_liab,
                "Total Stockholder Equity": t_equity,
                "Cash And Cash Equivalents": cash_val,
                "Long Term Debt": lt_debt,
                "Current Debt": st_debt,
                "Total Debt": t_debt,
            }

            # ── C. Cash Flow (cf) ────────────────────────────────────────────
            cf_items = report_items.get("cf", [])
            cf_map = {item.get("concept", ""): safe_num(item.get("value")) for item in cf_items if "concept" in item}
            cf_labels = {item.get("label", "").lower(): safe_num(item.get("value")) for item in cf_items if "label" in item}

            ocf = _get_val(["NetCashProvidedByUsedInOperatingActivities"], ["operating activities", "operating cash flow"], cf_map, cf_labels)
            capex = abs(_get_val(["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"], ["capital expenditure", "property, plant and equipment", "additions to property"], cf_map, cf_labels))
            repurchase = abs(_get_val(["PaymentsForRepurchaseOfCommonStock", "PaymentsForRepurchaseOfStock"], ["repurchase of common stock", "repurchases of stock"], cf_map, cf_labels))
            fcf = ocf - capex

            dict_cf[period_label] = {
                "Operating Cash Flow": ocf,
                "Capital Expenditure": capex,
                "Free Cash Flow": fcf,
                "Repurchase Of Capital Stock": repurchase,
            }

    if dict_inc:
        inc = pd.DataFrame(dict_inc)
    if dict_bs:
        bs = pd.DataFrame(dict_bs)
    if dict_cf:
        cf = pd.DataFrame(dict_cf)

    # 5. Respaldo / Fallback de Series Históricas desde /stock/metric si filings SEC están incompletos
    if inc.empty and "series" in metric_data:
        try:
            ebitda_series = series_dict.get("ebitda", [])
            eps_series = series_dict.get("eps", [])
            years_s = [str(item.get("period", ""))[:4] for item in ebitda_series if "period" in item]
            if years_s:
                dict_synth_inc = {}
                dict_synth_cf = {}
                for idx_y, y_str in enumerate(years_s):
                    ebitda_val = safe_num(ebitda_series[idx_y].get("v", 0.0)) if idx_y < len(ebitda_series) else 0.0
                    eps_val = safe_num(eps_series[idx_y].get("v", 0.0)) if idx_y < len(eps_series) else 0.0
                    est_rev = ebitda_val / max(op_margins, 0.15) if ebitda_val > 0 else 0.0
                    est_ni = eps_val * shares_outstanding if (eps_val > 0 and shares_outstanding > 0) else ebitda_val * 0.7

                    dict_synth_inc[y_str] = {
                        "Total Revenue": est_rev,
                        "Gross Profit": est_rev,
                        "Operating Income": ebitda_val * 0.85,
                        "Net Income": est_ni,
                        "Interest Expense": 0.0,
                        "Diluted Average Shares": shares_outstanding,
                    }
                    dict_synth_cf[y_str] = {
                        "Operating Cash Flow": est_ni * 1.1,
                        "Capital Expenditure": est_rev * 0.04,
                        "Free Cash Flow": (est_ni * 1.1) - (est_rev * 0.04),
                    }
                inc = pd.DataFrame(dict_synth_inc)
                cf = pd.DataFrame(dict_synth_cf)
        except Exception:
            pass

    # 6. Inyección de valores consolidados a info
    total_assets_val = _extraer_val_df(bs, ["Total Assets", "totalAssets"])
    total_debt_val = _extraer_val_df(bs, ["Total Debt", "totalDebt", "Long Term Debt"]) or safe_num(metrics_dict.get("netDebtAnnual", 0.0))
    total_cash_val = _extraer_val_df(bs, ["Cash And Cash Equivalents", "Cash"])
    total_equity_val = _extraer_val_df(bs, ["Total Stockholder Equity", "Stockholders Equity"])
    cur_assets_val = _extraer_val_df(bs, ["Total Current Assets", "Current Assets"])
    cur_liab_val = _extraer_val_df(bs, ["Total Current Liabilities", "Current Liabilities"])
    ebit_val = _extraer_val_df(inc, ["Operating Income", "Operating Profit", "EBIT"])
    net_income_val = _extraer_val_df(inc, ["Net Income"])
    rev_val = _extraer_val_df(inc, ["Total Revenue", "Revenue"])
    gross_profit_val = _extraer_val_df(inc, ["Gross Profit"]) or rev_val
    ocf_val = _extraer_val_df(cf, ["Operating Cash Flow"])
    fcf_val = _extraer_val_df(cf, ["Free Cash Flow"]) or safe_num(metrics_dict.get("fcfAnnual", 0.0))
    int_exp_val = abs(_extraer_val_df(inc, ["Interest Expense"]))
    pretax_val = _extraer_val_df(inc, ["Pretax Income"])
    tax_prov_val = _extraer_val_df(inc, ["Tax Provision"])
    ebitda_val = safe_num(metrics_dict.get("ebitdaAnnual", 0.0)) or (ebit_val * 1.15)

    info.update({
        "symbol": ticker,
        "longName": long_name,
        "sector": sector_std,
        "industry": industry_raw,
        "marketCap": mcap,
        "sharesOutstanding": shares_outstanding,
        "beta": beta,
        "dividendRate": div_rate,
        "trailingEps": eps_ttm,
        "forwardEps": eps_ttm * (1.0 + max(eps_growth, 0.05)),
        "currentRatio": current_ratio,
        "debtToEquity": debt_to_equity,
        "returnOnEquity": roe_val,
        "returnOnAssets": roa_val,
        "roic": roic_val,
        "operatingMargins": op_margins,
        "revenueGrowth": rev_growth,
        "earningsGrowth": eps_growth,
        "totalAssets": total_assets_val,
        "totalDebt": total_debt_val,
        "totalCash": total_cash_val,
        "totalStockholderEquity": total_equity_val,
        "totalCurrentAssets": cur_assets_val,
        "totalCurrentLiabilities": cur_liab_val,
        "operatingIncome": ebit_val,
        "netIncomeToCommon": net_income_val,
        "netIncome": net_income_val,
        "totalRevenue": rev_val,
        "grossProfit": gross_profit_val,
        "grossProfits": gross_profit_val,
        "operatingCashflow": ocf_val,
        "freeCashflow": fcf_val,
        "interestExpense": int_exp_val,
        "pretaxIncome": pretax_val,
        "taxProvision": tax_prov_val,
        "ebitda": ebitda_val,
        "pegRatio": safe_num(pe_ttm / (eps_growth * 100.0)) if (pe_ttm > 0 and eps_growth > 0) else 0.0,
        "targetMeanPrice": 0.0,
        "shortPercentOfFloat": 0.0,
    })

    claves_criticas = [
        "marketCap", "sharesOutstanding", "totalDebt", "totalCash", "ebitda",
        "operatingIncome", "netIncomeToCommon", "freeCashflow", "operatingCashflow",
        "interestExpense", "totalAssets", "totalStockholderEquity", "totalCurrentAssets",
        "totalCurrentLiabilities", "currentRatio", "debtToEquity", "returnOnEquity",
        "returnOnAssets", "operatingMargins", "pretaxIncome", "taxProvision", "beta"
    ]
    for k in claves_criticas:
        info[k] = safe_num(info.get(k), default=0.0)

    return info, inc, bs, cf


# ─────────────────────────────────────────────────────────────────────────────
# 4. NOTICIAS CORPORATIVAS Y MACROECONOMÍA (FINNHUB & FRED)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=FINNHUB_CACHE_TTL_NEWS)
def obtener_noticias_financieras(ticker: str, finnhub_api_key: str = "") -> List[Dict[str, str]]:
    """
    Obtiene los titulares y noticias corporativas más recientes para un ticker
    desde el endpoint `/company-news` de Finnhub API.
    """
    ticker = str(ticker).upper().strip()
    clean_news: List[Dict[str, str]] = []
    if not ticker:
        return clean_news

    now = datetime.date.today()
    to_date = now.strftime("%Y-%m-%d")
    from_date = (now - datetime.timedelta(days=30)).strftime("%Y-%m-%d")

    raw_news = _finnhub_get(
        "company-news",
        params={
            "symbol": ticker,
            "from": from_date,
            "to": to_date,
        },
        api_key=finnhub_api_key,
        timeout=8.0,
    )

    if isinstance(raw_news, list):
        for item in raw_news[:8]:
            if not isinstance(item, dict):
                continue
            title = item.get("headline", "").strip()
            link = item.get("url", "").strip()
            if not title or not link or link == "#":
                continue

            pub_ts = item.get("datetime")
            if pub_ts and isinstance(pub_ts, (int, float)):
                try:
                    date_str = datetime.datetime.fromtimestamp(pub_ts).strftime("%Y-%m-%d %H:%M")
                except Exception:
                    date_str = "Fecha reciente"
            else:
                date_str = "Fecha reciente"

            publisher = item.get("source", "Finnhub")

            clean_news.append({
                "title": title,
                "publisher": publisher,
                "link": link,
                "date": date_str,
            })

    return clean_news


@st.cache_data(ttl=FRED_CACHE_TTL)
def obtener_tasa_fred(api_key: str) -> float:
    """Obtiene la tasa del bono del tesoro a 10 años (DGS10) desde la API de FRED."""
    if api_key:
        try:
            url = f"https://api.stlouisfed.org/fred/series/observations?series_id=DGS10&api_key={api_key.strip()}&file_type=json&sort_order=desc&limit=1"
            res = requests.get(url, timeout=5).json()
            return safe_num(res['observations'][0]['value'], default=4.20)
        except Exception:
            pass
    return 4.20


@st.cache_data(ttl=FRED_CACHE_TTL)
def obtener_erp_mercado(fred_api_key: str = "", rf_actual: float = 4.25) -> float:
    """
    Calcula la Prima de Riesgo de Mercado (ERP) empírica observada a partir de
    series macroeconómicas oficiales de FRED (spread de crédito High Yield BAMLH0A0HYM2).
    """
    if fred_api_key:
        try:
            url = f"https://api.stlouisfed.org/fred/series/observations?series_id=BAMLH0A0HYM2&api_key={fred_api_key.strip()}&file_type=json&sort_order=desc&limit=1"
            res = requests.get(url, timeout=5).json()
            spread_val = safe_num(res['observations'][0]['value'], default=3.50)
            erp_calc = 2.0 + (spread_val * 0.75)
            return round(min(max(erp_calc, 4.50), 6.00), 2)
        except Exception:
            pass
    return 5.00


@st.cache_data(ttl=FRED_CACHE_TTL)
def obtener_rf_tnx(fallback_fred: float = 4.20, finnhub_api_key: str = "") -> float:
    """
    Obtiene la tasa libre de riesgo (R_f).
    """
    return float(fallback_fred) if fallback_fred > 0 else 4.20


@st.cache_data(ttl=FRED_CACHE_TTL)
def obtener_kd_finnhub_fred(
    ticker: str,
    finnhub_api_key: str = "",
    fred_api_key: str = "",
    int_expense: float = 0.0,
    total_debt_val: float = 0.0,
) -> float:
    """
    Calcula el costo empírico de la deuda (Kd).
    """
    if total_debt_val > 0 and int_expense > 0:
        return (int_expense / total_debt_val) * 100.0
    if fred_api_key:
        try:
            url_corp = f"https://api.stlouisfed.org/fred/series/observations?series_id=BAMLC0A0CM&api_key={fred_api_key.strip()}&file_type=json&sort_order=desc&limit=1"
            res_corp = requests.get(url_corp, timeout=4).json()
            return safe_num(res_corp['observations'][0]['value'], default=5.50)
        except Exception:
            pass
    return 5.50


# Alias para retrocompatibilidad
obtener_kd_fmp_fred = obtener_kd_finnhub_fred


@st.cache_data(ttl=FINNHUB_CACHE_TTL_METRICS)
def obtener_datos_dividendos(
    ticker: str, info_dict: Dict[str, Any], finnhub_api_key: str = "", precio_ref: float = 0.0
) -> Tuple[float, float, str]:
    """Calcula el dividendo anualizado, el yield (%) y la fecha ex-dividendo."""
    div_rate = safe_get(info_dict, ["dividendRate", "trailingAnnualDividendRate", "lastDiv"], 0.0)
    div_rate = safe_num(div_rate, 0.0)
    div_yield = (div_rate / precio_ref * 100.0) if (div_rate > 0 and precio_ref > 0) else 0.0
    ex_div_ts = safe_get(info_dict, ["exDividendDate"], None)
    if ex_div_ts and isinstance(ex_div_ts, (int, float)):
        try:
            next_div_date = datetime.datetime.fromtimestamp(ex_div_ts).strftime('%Y-%m-%d')
        except Exception:
            next_div_date = "N/A"
    else:
        next_div_date = "N/A"
    return div_rate, div_yield, next_div_date


# ─────────────────────────────────────────────────────────────────────────────
# 5. ORQUESTADOR CONCURRENTE MAESTRO
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=FINNHUB_CACHE_TTL_QUOTE)
def fetch_datos_concurrente(
    ticker: str, finnhub_key: str = "", fred_key: str = ""
) -> Dict[str, Any]:
    """
    Función concurrente maestra que dispara simultáneamente:
    - Cotización intradía y velas 5Y (Finnhub)
    - Datos fundamentales, métricas y estados financieros (Finnhub)
    - Tasa de bono FRED
    - Feed de noticias (Finnhub)
    vía ThreadPoolExecutor para máxima velocidad de respuesta en UI.
    """
    with ThreadPoolExecutor(max_workers=4) as executor:
        f_quote = executor.submit(fetch_cotizacion_intradia, ticker, finnhub_key)
        f_funda = executor.submit(fetch_datos_fundamentales, ticker, finnhub_key)
        f_fred = executor.submit(obtener_tasa_fred, fred_key)
        f_news = executor.submit(obtener_noticias_financieras, ticker, finnhub_key)

        try:
            precio_actual, prev_close, hist = f_quote.result(timeout=14)
        except Exception:
            precio_actual, prev_close, hist = 0.0, 0.0, pd.DataFrame()

        try:
            info, inc, bs, cf = f_funda.result(timeout=16)
        except Exception:
            info, inc, bs, cf = {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        try:
            tasa_fred = f_fred.result(timeout=5)
        except Exception:
            tasa_fred = 4.20

        try:
            news_data = f_news.result(timeout=8)
        except Exception:
            news_data = []

    return {
        "precio_actual": precio_actual,
        "prev_close": prev_close,
        "hist": hist,
        "info": info,
        "inc": inc,
        "bs": bs,
        "cf": cf,
        "tasa_fred": tasa_fred,
        "news_data": news_data,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. EXTRACTORES ANALÍTICOS PURAMENTE DETERMINÍSTICOS (TTM Y FCFF)
# ─────────────────────────────────────────────────────────────────────────────

def obtener_capex_historico(cf: pd.DataFrame) -> List[float]:
    """
    Extrae el Gasto de Capital (CapEx) del estado de flujos de efectivo y lo
    retorna como lista de valores positivos (gasto real), ordenados del más
    reciente al más antiguo.
    """
    posibles_filas = [
        "Capital Expenditure",
        "CapitalExpenditure",
        "Capital Expenditures",
        "Purchase Of Property Plant And Equipment",
        "Purchases Of Property, Plant And Equipment",
        "Capital Expenditure Reported",
        "Purchase Of PPE",
        "Net PPE Purchase And Sale",
        "capitalExpenditures",
    ]
    vals = _extraer_serie(cf, posibles_filas, absval=True)
    return vals if vals else []


def extraer_fcff_desapalancado(
    cf: pd.DataFrame,
    inc: pd.DataFrame,
    bs: pd.DataFrame,
    info: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extrae y sanitiza los insumos para el cálculo de FCFF histórico siguiendo
    una jerarquía dual-path estricta que elimina el doble conteo de la deuda.
    """
    ocf_hist = _extraer_serie(cf, [
        "Operating Cash Flow", "OperatingCashFlow",
        "Cash Flow From Continuing Operating Activities",
        "Total Cash From Operating Activities",
        "Net Cash Provided By Operating Activities"
    ])
    if not ocf_hist:
        ocf_ttm = safe_num(info.get("operatingCashflow", 0.0), 0.0)
        ocf_hist = [ocf_ttm] if ocf_ttm != 0.0 else [0.0]

    capex_hist = obtener_capex_historico(cf)
    if not capex_hist:
        fcf_ttm = safe_num(info.get("freeCashflow", 0.0), 0.0)
        ocf_ttm = safe_num(info.get("operatingCashflow", 0.0), 0.0)
        capex_hist = [abs(ocf_ttm - fcf_ttm)] if (ocf_ttm != 0.0 and fcf_ttm != 0.0) else [0.0]

    interest_hist = _extraer_serie(inc, [
        "Interest Expense", "InterestExpense",
        "Interest Expense Non Operating", "Net Non Operating Interest Income Expense"
    ], absval=True)
    if not interest_hist:
        interest_hist = [abs(safe_num(info.get("interestExpense", 0.0), 0.0))]

    pretax_hist = _extraer_serie(inc, [
        "Pretax Income", "Income Before Tax", "IncomeBeforeTax", "PretaxIncome"
    ])
    if not pretax_hist:
        pretax_hist = [safe_num(info.get("pretaxIncome", 0.0), 0.0)]

    taxprov_hist = _extraer_serie(inc, [
        "Tax Provision", "IncomeTaxExpense", "Income Tax Expense", "TaxProvision"
    ])
    if not taxprov_hist:
        taxprov_hist = [safe_num(info.get("taxProvision", 0.0), 0.0)]

    ebit_hist = _extraer_serie(inc, [
        "Operating Income", "OperatingIncome", "EBIT",
        "Normalized EBIT", "Normalized Operating Profit", "Operating Profit"
    ])
    if not ebit_hist:
        ebit_ttm = safe_num(info.get("operatingIncome", 0.0), 0.0)
        ebit_hist = [ebit_ttm] if ebit_ttm != 0.0 else []

    da_hist = _extraer_serie(cf, [
        "Depreciation Amortization Depletion",
        "Depreciation And Amortization",
        "Depreciation",
        "DepreciationAmortization",
        "Depreciation & Amortization",
    ], absval=True)
    if not da_hist:
        da_hist = _extraer_serie(inc, [
            "Reconciled Depreciation",
            "Depreciation And Amortization In Income Statement",
        ], absval=True)

    nwc_hist: List[float] = []
    cur_assets_hist = _extraer_serie(bs, ["Total Current Assets", "Current Assets", "CurrentAssets"])
    cur_liab_hist = _extraer_serie(bs, ["Total Current Liabilities", "Current Liabilities", "CurrentLiabilities"])
    cash_hist_bs = _extraer_serie(bs, [
        "Cash And Cash Equivalents", "CashCashEquivalentsAndShortTermInvestments",
        "Cash Financial", "Cash And Short Term Investments",
        "Cash, Cash Equivalents & Short Term Investments", "Cash"
    ])
    stdebt_hist = _extraer_serie(bs, [
        "Current Debt", "Current Debt And Capital Lease Obligation", "Short Term Debt", "CurrentDebt"
    ])

    n_nwc = min(len(cur_assets_hist) if cur_assets_hist else 0, len(cur_liab_hist) if cur_liab_hist else 0)
    for i in range(n_nwc):
        ca = safe_num(cur_assets_hist[i], 0.0)
        cl = safe_num(cur_liab_hist[i], 0.0)
        cash_i = safe_num(cash_hist_bs[i], 0.0) if i < len(cash_hist_bs) else 0.0
        std_i = safe_num(stdebt_hist[i], 0.0) if i < len(stdebt_hist) else 0.0
        nwc_hist.append((ca - cash_i) - (cl - std_i))

    delta_nwc_hist: List[float] = []
    for i in range(len(nwc_hist) - 1):
        delta_nwc_hist.append(nwc_hist[i] - nwc_hist[i + 1])
    if nwc_hist and not delta_nwc_hist:
        delta_nwc_hist = [0.0]

    total_debt = safe_num(info.get("totalDebt", 0.0), 0.0)
    if total_debt == 0.0:
        total_debt = _extraer_val_df(bs, [
            "Total Debt", "TotalDebt", "Long Term Debt And Capital Lease Obligation",
            "Long Term Debt", "LongTermDebt"
        ], default=0.0)

    total_cash = safe_num(info.get("totalCash", 0.0), 0.0)
    if total_cash == 0.0:
        total_cash = _extraer_val_df(bs, [
            "Cash And Cash Equivalents", "CashCashEquivalentsAndShortTermInvestments",
            "Cash Financial", "Cash And Short Term Investments",
            "Cash, Cash Equivalents & Short Term Investments", "Cash"
        ], default=0.0)

    shares_diluted = safe_num(info.get("sharesOutstanding", 0.0), 0.0)
    if shares_diluted <= 0:
        shares_diluted = _extraer_val_df(inc, [
            "Diluted Average Shares", "Basic Average Shares", "Ordinary Shares Number",
            "Weighted Average Shares Diluted"
        ], default=0.0)

    mcap_val = safe_num(info.get("marketCap", 0.0), 0.0)
    cur_p = safe_num(info.get("currentPrice", 0.0), 0.0)
    if mcap_val > 10_000_000 and cur_p > 0:
        implied_sh = mcap_val / cur_p
        if shares_diluted <= 1000 or shares_diluted < implied_sh * 0.01 or shares_diluted > implied_sh * 100:
            shares_diluted = implied_sh

    n_max = max(len(ocf_hist), 1)

    def _pad(lst: List[float], n: int, fill: float = 0.0) -> List[float]:
        if len(lst) >= n:
            return lst[:n]
        return lst + [fill] * (n - len(lst))

    return {
        "ocf_hist": _pad(ocf_hist, n_max),
        "capex_hist": _pad(capex_hist, n_max),
        "interest_hist": _pad(interest_hist, n_max),
        "pretax_hist": _pad(pretax_hist, n_max),
        "taxprov_hist": _pad(taxprov_hist, n_max),
        "ebit_hist": _pad(ebit_hist, n_max),
        "da_hist": _pad(da_hist, n_max),
        "delta_nwc_hist": _pad(delta_nwc_hist, n_max),
        "total_debt": total_debt,
        "total_cash": total_cash,
        "shares_diluted": shares_diluted,
        "n_periodos": n_max,
    }


def extraer_componentes_fcff(
    cf: pd.DataFrame,
    inc: pd.DataFrame,
    bs: pd.DataFrame,
    info: Dict[str, Any],
) -> Dict[str, Any]:
    return extraer_fcff_desapalancado(cf, inc, bs, info)


def extraer_metricas_ttm(
    info: Dict[str, Any],
    inc: pd.DataFrame,
    bs: pd.DataFrame,
    cf: pd.DataFrame,
    precio_actual: float = 0.0,
) -> Dict[str, Any]:
    """
    Extrae, normaliza y consolida todas las métricas y cifras de los estados financieros
    en base a los últimos 12 meses (TTM) y balance consolidado más reciente.
    """
    shares_diluted = safe_num(info.get("sharesOutstanding", 0.0), 0.0)
    if shares_diluted <= 0:
        shares_diluted = _extraer_val_df(inc, [
            "Diluted Average Shares", "Basic Average Shares", "Ordinary Shares Number",
            "Weighted Average Shares Diluted"
        ], default=0.0)
    if shares_diluted <= 0:
        shares_diluted = _extraer_val_df(bs, ["Ordinary Shares Number", "Share Issued", "Common Stock"], default=0.0)

    mcap = safe_num(info.get("marketCap", 0.0), 0.0)
    if mcap <= 0 and precio_actual > 0 and shares_diluted > 0:
        mcap = shares_diluted * precio_actual
    if shares_diluted <= 0 and mcap > 0 and precio_actual > 0:
        shares_diluted = mcap / precio_actual

    if mcap > 10_000_000 and precio_actual > 0:
        implied_sh = mcap / precio_actual
        if shares_diluted <= 1000 or shares_diluted < implied_sh * 0.01 or shares_diluted > implied_sh * 100:
            shares_diluted = implied_sh

    revenue_ttm = safe_num(info.get("totalRevenue", 0.0), 0.0)
    if revenue_ttm <= 0:
        revenue_ttm = _extraer_val_df(inc, ["Total Revenue", "Operating Revenue", "Revenue", "TotalRevenue"], default=0.0)

    gross_profit_ttm = safe_num(info.get("grossProfits", info.get("grossProfit", 0.0)), 0.0)
    if gross_profit_ttm <= 0:
        gross_profit_ttm = _extraer_val_df(inc, ["Gross Profit", "GrossProfit"], default=0.0)
    if gross_profit_ttm <= 0 and revenue_ttm > 0:
        gross_profit_ttm = revenue_ttm

    operating_income_ttm = safe_num(info.get("operatingIncome", 0.0), 0.0)
    if operating_income_ttm == 0.0:
        operating_income_ttm = _extraer_val_df(inc, [
            "Operating Income", "OperatingIncome", "Operating Profit", "EBIT",
            "Normalized EBIT", "Normalized Operating Profit"
        ], default=0.0)

    ebitda_ttm = safe_num(info.get("ebitda", 0.0), 0.0)
    if ebitda_ttm <= 0:
        ebitda_ttm = _extraer_val_df(inc, ["EBITDA", "Normalized EBITDA", "ebitda"], default=0.0)
    if ebitda_ttm <= 0 and operating_income_ttm > 0:
        ebitda_ttm = operating_income_ttm * 1.15

    net_income_ttm = safe_num(info.get("netIncomeToCommon", 0.0), 0.0)
    if net_income_ttm == 0.0:
        net_income_ttm = safe_num(info.get("netIncome", 0.0), 0.0)
    if net_income_ttm == 0.0:
        net_income_ttm = _extraer_val_df(inc, [
            "Net Income Common Stockholders", "Net Income", "NetIncome",
            "Net Income To Common", "Net Income Continuous Operations"
        ], default=0.0)

    eps_diluted_ttm = safe_num(info.get("trailingEps", 0.0), 0.0)
    if eps_diluted_ttm == 0.0:
        eps_diluted_ttm = safe_num(info.get("epsTrailingTwelveMonths", 0.0), 0.0)
    if eps_diluted_ttm == 0.0:
        eps_diluted_ttm = _extraer_val_df(inc, [
            "Diluted EPS", "DilutedEPS", "Basic EPS", "BasicEPS",
            "Diluted EPS from Continuing Operations", "EPS"
        ], default=0.0)
    if eps_diluted_ttm == 0.0 and shares_diluted > 0 and net_income_ttm != 0.0:
        eps_diluted_ttm = net_income_ttm / shares_diluted

    forward_eps = safe_num(info.get("forwardEps", 0.0), 0.0)
    if forward_eps == 0.0:
        forward_eps = safe_num(info.get("epsForward", 0.0), 0.0)

    pretax_income_ttm = safe_num(info.get("pretaxIncome", 0.0), 0.0)
    if pretax_income_ttm == 0.0:
        pretax_income_ttm = _extraer_val_df(inc, ["Pretax Income", "Income Before Tax", "IncomeBeforeTax", "PretaxIncome"], default=0.0)

    tax_provision_ttm = safe_num(info.get("taxProvision", 0.0), 0.0)
    if tax_provision_ttm == 0.0:
        tax_provision_ttm = _extraer_val_df(inc, ["Tax Provision", "IncomeTaxExpense", "Income Tax Expense", "TaxProvision"], default=0.0)

    interest_expense_ttm = abs(safe_num(info.get("interestExpense", 0.0), 0.0))
    if interest_expense_ttm == 0.0:
        interest_expense_ttm = abs(_extraer_val_df(inc, [
            "Interest Expense", "InterestExpense", "Interest Expense Non Operating",
            "Net Non Operating Interest Income Expense"
        ], default=0.0))

    ocf_ttm = safe_num(info.get("operatingCashflow", 0.0), 0.0)
    if ocf_ttm == 0.0:
        ocf_ttm = _extraer_val_df(cf, [
            "Operating Cash Flow", "OperatingCashFlow",
            "Cash Flow From Continuing Operating Activities",
            "Total Cash From Operating Activities",
            "Net Cash Provided By Operating Activities"
        ], default=0.0)

    capex_list = obtener_capex_historico(cf)
    capex_ttm = capex_list[0] if capex_list else 0.0
    if capex_ttm == 0.0:
        capex_ttm = abs(safe_num(info.get("capitalExpenditures", 0.0), 0.0))

    fcf_ttm = ocf_ttm - capex_ttm
    if fcf_ttm == 0.0 and ocf_ttm == 0.0:
        fcf_ttm = safe_num(info.get("freeCashflow", 0.0), 0.0)

    total_debt = safe_num(info.get("totalDebt", 0.0), 0.0)
    if total_debt == 0.0:
        total_debt = _extraer_val_df(bs, [
            "Total Debt", "TotalDebt", "Long Term Debt And Capital Lease Obligation",
            "Long Term Debt", "LongTermDebt", "Total Non Current Liabilities Net Minority Interest"
        ], default=0.0)

    total_cash = safe_num(info.get("totalCash", 0.0), 0.0)
    if total_cash == 0.0:
        total_cash = _extraer_val_df(bs, [
            "Cash And Cash Equivalents",
            "CashCashEquivalentsAndShortTermInvestments",
            "Cash Financial", "Cash And Short Term Investments",
            "Cash, Cash Equivalents & Short Term Investments", "Cash"
        ], default=0.0)

    total_equity = safe_num(info.get("totalStockholderEquity", 0.0), 0.0)
    if total_equity == 0.0:
        total_equity = _extraer_val_df(bs, [
            "Total Stockholder Equity", "Stockholders Equity",
            "StockholdersEquity", "TotalStockholderEquity",
            "Common Stock Equity", "CommonStockEquity",
            "Total Equity", "Total Equity Gross Minority Interest"
        ], default=0.0)

    total_assets = safe_num(info.get("totalAssets", 0.0), 0.0)
    if total_assets == 0.0:
        total_assets = _extraer_val_df(bs, ["Total Assets", "TotalAssets", "Total Assets Net", "totalAssets"], default=0.0)

    current_assets = safe_num(info.get("totalCurrentAssets", 0.0), 0.0)
    if current_assets == 0.0:
        current_assets = _extraer_val_df(bs, ["Total Current Assets", "Current Assets", "CurrentAssets"], default=0.0)

    current_liabilities = safe_num(info.get("totalCurrentLiabilities", 0.0), 0.0)
    if current_liabilities == 0.0:
        current_liabilities = _extraer_val_df(bs, ["Total Current Liabilities", "Current Liabilities", "CurrentLiabilities"], default=0.0)

    short_term_debt = _extraer_val_df(bs, [
        "Current Debt", "Current Debt And Capital Lease Obligation", "Short Term Debt", "CurrentDebt"
    ], default=0.0)

    cagr_revenue_3_5y = 0.0
    op_margin_hist = 0.0
    if isinstance(inc, pd.DataFrame) and not inc.empty:
        for fila_rev in ["Total Revenue", "Operating Revenue", "Revenue", "TotalRevenue"]:
            if fila_rev in inc.index:
                try:
                    s_rev = inc.loc[fila_rev].dropna()
                    if isinstance(s_rev, pd.DataFrame):
                        s_rev = s_rev.iloc[0]
                    vals_rev = [safe_num(v, 0.0) for v in s_rev.values if safe_num(v, 0.0) > 0]
                    if len(vals_rev) >= 2:
                        n_anios = min(len(vals_rev) - 1, 4)
                        r_reciente = vals_rev[0]
                        r_antiguo = vals_rev[n_anios]
                        if r_reciente > 0 and r_antiguo > 0:
                            cagr_raw = (r_reciente / r_antiguo) ** (1.0 / n_anios) - 1.0
                            if -0.30 <= cagr_raw <= 0.80:
                                cagr_revenue_3_5y = cagr_raw
                    break
                except Exception:
                    pass

        for fila_op in ["Operating Income", "OperatingIncome", "EBIT", "Operating Profit"]:
            if fila_op in inc.index:
                try:
                    s_op = inc.loc[fila_op].dropna()
                    if isinstance(s_op, pd.DataFrame):
                        s_op = s_op.iloc[0]
                    for fila_rev in ["Total Revenue", "Operating Revenue", "Revenue", "TotalRevenue"]:
                        if fila_rev in inc.index:
                            s_rev = inc.loc[fila_rev].dropna()
                            if isinstance(s_rev, pd.DataFrame):
                                s_rev = s_rev.iloc[0]
                            common_cols = [c for c in s_rev.index if c in s_op.index]
                            mgs = []
                            for col_c in common_cols:
                                r_val = safe_num(s_rev[col_c], 0.0)
                                o_val = safe_num(s_op[col_c], 0.0)
                                if r_val > 0:
                                    mgs.append(o_val / r_val)
                            if mgs:
                                op_margin_hist = max(float(np.mean(mgs)), 0.0)
                            break
                    if op_margin_hist > 0:
                        break
                except Exception:
                    pass

    if op_margin_hist == 0.0 and revenue_ttm > 0 and operating_income_ttm > 0:
        op_margin_hist = operating_income_ttm / revenue_ttm

    earnings_growth = safe_num(info.get("earningsGrowth", 0.0), 0.0)
    if earnings_growth == 0.0 and isinstance(inc, pd.DataFrame) and not inc.empty:
        for fila_ni in ["Net Income", "NetIncome", "Net Income Common Stockholders"]:
            if fila_ni in inc.index:
                try:
                    s_ni = inc.loc[fila_ni].dropna()
                    if isinstance(s_ni, pd.DataFrame):
                        s_ni = s_ni.iloc[0]
                    vals_ni = [safe_num(v, 0.0) for v in s_ni.values if safe_num(v, 0.0) > 0]
                    if len(vals_ni) >= 2 and vals_ni[0] > 0 and vals_ni[-1] > 0:
                        n_anios_ni = min(len(vals_ni) - 1, 3)
                        ni_cagr = (vals_ni[0] / vals_ni[n_anios_ni]) ** (1.0 / n_anios_ni) - 1.0
                        if -0.50 <= ni_cagr <= 1.0:
                            earnings_growth = ni_cagr
                    break
                except Exception:
                    pass

    revenue_growth = safe_num(info.get("revenueGrowth", 0.0), 0.0)
    if revenue_growth == 0.0 and cagr_revenue_3_5y != 0.0:
        revenue_growth = cagr_revenue_3_5y

    peg_ratio_info = safe_num(info.get("pegRatio", 0.0), 0.0)
    beta = safe_num(info.get("beta", 1.0), 1.0)
    short_percent_of_float = safe_num(info.get("shortPercentOfFloat", 0.0), 0.0)
    target_mean_price = safe_num(info.get("targetMeanPrice", 0.0), 0.0)

    return {
        "shares_diluted": shares_diluted,
        "mcap": mcap,
        "revenue_ttm": revenue_ttm,
        "gross_profit_ttm": gross_profit_ttm,
        "operating_income_ttm": operating_income_ttm,
        "ebitda_ttm": ebitda_ttm,
        "net_income_ttm": net_income_ttm,
        "eps_diluted_ttm": eps_diluted_ttm,
        "forward_eps": forward_eps,
        "pretax_income_ttm": pretax_income_ttm,
        "tax_provision_ttm": tax_provision_ttm,
        "interest_expense_ttm": interest_expense_ttm,
        "ocf_ttm": ocf_ttm,
        "capex_ttm": capex_ttm,
        "fcf_ttm": fcf_ttm,
        "total_debt": total_debt,
        "total_cash": total_cash,
        "net_debt": total_debt - total_cash,
        "total_equity": total_equity,
        "total_assets": total_assets,
        "current_assets": current_assets,
        "current_liabilities": current_liabilities,
        "short_term_debt": short_term_debt,
        "earnings_growth": earnings_growth,
        "revenue_growth": revenue_growth,
        "cagr_revenue_3_5y": cagr_revenue_3_5y,
        "op_margin_hist": op_margin_hist,
        "peg_ratio_info": peg_ratio_info,
        "beta": beta,
        "short_percent_of_float": short_percent_of_float,
        "target_mean_price": target_mean_price,
    }
