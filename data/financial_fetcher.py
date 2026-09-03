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
    return safe_num(default, 0.0)


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

    # CAPA 3 (Fallback yfinance): Si Finnhub no devolvió cotización o velas
    if precio_actual == 0.0 or hist.empty:
        try:
            import yfinance as yf
            ticker_yf = yf.Ticker(ticker)
            df_yf = ticker_yf.history(period="5y")
            if df_yf is not None and not df_yf.empty:
                hist = df_yf[["Open", "High", "Low", "Close", "Volume"]].copy()
                if precio_actual == 0.0:
                    precio_actual = safe_num(hist["Close"].iloc[-1], 0.0)
                if prev_close == 0.0:
                    prev_close = safe_num(hist["Close"].iloc[-2], 0.0) if len(hist) > 1 else precio_actual
        except Exception as e_yf:
            logger.debug("yfinance history fallback error para %s: %s", ticker, e_yf)

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


class ConsensusWallStreet(tuple):
    """
    Estructura híbrida inmutable (target_mean, target_high, target_low) que
    soporta tanto desempaquetado posicional de tupla de 3 elementos como acceso por clave tipo dict
    y por atributo.
    Garantiza compatibilidad absoluta con:
      - mean, high, low = obtener_consenso_wall_street(ticker)
      - res = obtener_consenso_wall_street(ticker) -> res['target_mean'], res.get('recommendation')
    """
    def __new__(
        cls,
        target_mean: Any = 0.0,
        target_high: Any = 0.0,
        target_low: Any = 0.0,
        recommendation: Optional[str] = None,
        num_analysts: Optional[int] = None,
        raw_data: Optional[Dict[str, Any]] = None,
    ) -> ConsensusWallStreet:
        m = safe_num(target_mean, default=0.0)
        h = safe_num(target_high, default=0.0)
        l = safe_num(target_low, default=0.0)
        return super().__new__(cls, (m, h, l))

    def __init__(
        self,
        target_mean: Any = 0.0,
        target_high: Any = 0.0,
        target_low: Any = 0.0,
        recommendation: Optional[str] = None,
        num_analysts: Optional[int] = None,
        raw_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.target_mean: float = safe_num(target_mean, default=0.0)
        self.target_high: float = safe_num(target_high, default=0.0)
        self.target_low: float = safe_num(target_low, default=0.0)
        self.recommendation: Optional[str] = recommendation
        self.num_analysts: Optional[int] = num_analysts
        self._dict: Dict[str, Any] = {
            "target_mean": self.target_mean,
            "target_high": self.target_high,
            "target_low": self.target_low,
            "target_mean_price": self.target_mean,
            "target_high_price": self.target_high,
            "target_low_price": self.target_low,
            "targetMeanPrice": self.target_mean,
            "targetHighPrice": self.target_high,
            "targetLowPrice": self.target_low,
            "recommendation": self.recommendation,
            "recommendation_key": self.recommendation,
            "num_analysts": self.num_analysts,
            "number_of_analysts": self.num_analysts,
            "raw_data": raw_data or {},
        }

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, str):
            return self._dict[item]
        return super().__getitem__(item)

    def get(self, key: str, default: Any = None) -> Any:
        return self._dict.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._dict)

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()

    def __contains__(self, key: object) -> bool:
        return key in self._dict or super().__contains__(key)

    def __repr__(self) -> str:
        return (
            f"ConsensusWallStreet(target_mean={self.target_mean}, "
            f"target_high={self.target_high}, target_low={self.target_low}, "
            f"recommendation={self.recommendation!r}, num_analysts={self.num_analysts!r})"
        )


@st.cache_data(ttl=FINNHUB_CACHE_TTL_METRICS)
def obtener_consenso_wall_street(
    ticker: str,
    finnhub_api_key: str = "",
    target_data: Optional[Any] = None,
    metrics_dict: Optional[Dict[str, Any]] = None,
    _yf_info: Optional[Dict[str, Any]] = None,
    yf_info: Optional[Dict[str, Any]] = None,
) -> ConsensusWallStreet:
    """
    Extrae el precio objetivo del consenso de analistas de Wall Street
    exclusivamente desde los datos de Yahoo Finance (yfinance), con respaldo
    defensivo ante activos sin cobertura o fallos de conexión.

    Jerarquía de extracción Yahoo Finance (yfinance):
    1. Objeto `_yf_info` o `yf_info` pre-cargado si es provisto (máxima velocidad sin latencia).
    2. Atributo `ticker_yf.analyst_price_targets` ('mean', 'median', 'high', 'low').
    3. Diccionario `ticker_yf.info` ('targetMeanPrice', 'targetMedianPrice', 'targetHighPrice', etc.).
    4. DataFrame `ticker_yf.recommendations_summary` (estimaciones agregadas).
    5. Fallback secundario controlado si el activo carece de cobertura en Yahoo Finance.

    Returns:
        ConsensusWallStreet: Estructura híbrida compatible con tupla de 3 elementos (mean, high, low)
        y con acceso tipo diccionario (res['target_mean'], res.get('recommendation'), res.to_dict()).
        Si el activo no tiene cobertura, retorna (0.0, 0.0, 0.0).
    """
    ticker = str(ticker).upper().strip()
    if not ticker:
        return ConsensusWallStreet(0.0, 0.0, 0.0, None, None)

    target_mean_price = 0.0
    target_high_price = 0.0
    target_low_price = 0.0
    recommendation: Optional[str] = None
    num_analysts: Optional[int] = None
    raw_payload: Dict[str, Any] = {}

    info_dict = _yf_info or yf_info

    # ── CAPA 1: Insumos directos provistos por el llamador ──
    # 1.1 Intentar con yf_info pre-cargado si existe
    if isinstance(info_dict, dict) and info_dict:
        mean_c = safe_num(
            info_dict.get("targetMeanPrice")
            or info_dict.get("targetMedianPrice")
            or info_dict.get("targetPrice")
            or info_dict.get("target_mean_price"),
            0.0
        )
        if mean_c > 0.0:
            target_mean_price = mean_c
            target_high_price = safe_num(info_dict.get("targetHighPrice") or info_dict.get("target_high_price"), 0.0)
            target_low_price = safe_num(info_dict.get("targetLowPrice") or info_dict.get("target_low_price"), 0.0)
            n_opinions = info_dict.get("numberOfAnalystOpinions") or info_dict.get("num_analysts")
            if n_opinions is not None:
                try:
                    num_analysts = int(n_opinions)
                except (ValueError, TypeError):
                    pass
            recommendation = info_dict.get("recommendationKey") or info_dict.get("recommendation")
            raw_payload["source"] = "yf_info_cache"

    # 1.2 Si se proveyó target_data explícito (ej. pruebas o endpoint previo)
    if target_mean_price <= 0.0 and target_data:
        target_info = target_data[0] if (isinstance(target_data, list) and len(target_data) > 0 and isinstance(target_data[0], dict)) else (target_data if isinstance(target_data, dict) else {})
        raw_payload["price_target_data"] = target_info
        target_mean_val = (
            target_info.get("targetMean")
            or target_info.get("targetMedian")
            or target_info.get("targetMeanPrice")
            or target_info.get("targetMedianPrice")
            or target_info.get("target_mean")
            or target_info.get("target_median")
            or target_info.get("priceTarget")
            or target_info.get("targetPrice")
        )
        target_mean_price = safe_num(target_mean_val, 0.0)
        target_high_price = safe_num(target_info.get("targetHigh", target_info.get("target_high", 0.0)))
        target_low_price = safe_num(target_info.get("targetLow", target_info.get("target_low", 0.0)))
        n_an = target_info.get("numberAnalysts") or target_info.get("numberOfAnalysts")
        if n_an is not None:
            try:
                num_analysts = int(n_an)
            except (ValueError, TypeError):
                pass
        if target_mean_price <= 0.0 and target_high_price > 0.0 and target_low_price > 0.0:
            target_mean_price = round((target_high_price + target_low_price) / 2.0, 2)

    # 1.3 Si se proveyó metrics_dict explícito
    if target_mean_price <= 0.0 and metrics_dict:
        raw_payload["metrics_dict"] = metrics_dict
        target_mean_price = safe_num(
            metrics_dict.get("targetPrice")
            or metrics_dict.get("targetMeanPrice")
            or metrics_dict.get("priceTarget")
            or metrics_dict.get("targetMean")
            or metrics_dict.get("targetMedian")
            or metrics_dict.get("consensusPriceTarget")
            or metrics_dict.get("targetPriceMean"),
            0.0
        )
        if num_analysts is None:
            n_an = metrics_dict.get("numberOfAnalystOpinions") or metrics_dict.get("numberAnalysts")
            if n_an is not None:
                try:
                    num_analysts = int(n_an)
                except (ValueError, TypeError):
                    pass

    # ── CAPA 2: yfinance en vivo (Fuente Primaria Exclusiva de Mercado) ──
    if target_mean_price <= 0.0:
        try:
            import yfinance as yf
            ticker_yf = yf.Ticker(ticker)

            # Prioridad 2.1: analyst_price_targets
            try:
                apt = getattr(ticker_yf, "analyst_price_targets", None)
                if isinstance(apt, dict) and apt:
                    raw_payload["yf_analyst_price_targets"] = apt
                    mean_val = safe_num(apt.get("mean") or apt.get("median") or apt.get("current"), 0.0)
                    if mean_val > 0.0:
                        target_mean_price = mean_val
                    if target_high_price <= 0.0:
                        target_high_price = safe_num(apt.get("high"), 0.0)
                    if target_low_price <= 0.0:
                        target_low_price = safe_num(apt.get("low"), 0.0)
            except Exception as e_apt:
                logger.debug("yfinance analyst_price_targets no disponible para %s: %s", ticker, e_apt)

            # Prioridad 2.2: info de yfinance
            try:
                live_info = getattr(ticker_yf, "info", None) or {}
                if isinstance(live_info, dict) and live_info:
                    raw_payload["yf_info"] = {
                        k: live_info[k] for k in ["targetMeanPrice", "targetMedianPrice", "targetHighPrice", "targetLowPrice", "numberOfAnalystOpinions", "recommendationKey"] if k in live_info
                    }
                    if target_mean_price <= 0.0:
                        mean_val = safe_num(
                            live_info.get("targetMeanPrice")
                            or live_info.get("targetMedianPrice")
                            or live_info.get("targetPrice")
                            or live_info.get("target_mean_price"),
                            0.0
                        )
                        if mean_val > 0.0:
                            target_mean_price = mean_val
                    if target_high_price <= 0.0:
                        target_high_price = safe_num(live_info.get("targetHighPrice"), 0.0)
                    if target_low_price <= 0.0:
                        target_low_price = safe_num(live_info.get("targetLowPrice"), 0.0)

                    if num_analysts is None and "numberOfAnalystOpinions" in live_info:
                        try:
                            num_analysts = int(live_info["numberOfAnalystOpinions"])
                        except (ValueError, TypeError):
                            pass

                    if recommendation is None and "recommendationKey" in live_info:
                        recommendation = str(live_info["recommendationKey"])
            except Exception as e_info:
                logger.debug("yfinance info targetMeanPrice no disponible para %s: %s", ticker, e_info)

            # Prioridad 2.3: recommendations_summary
            if target_mean_price <= 0.0:
                try:
                    rec_sum = getattr(ticker_yf, "recommendations_summary", None)
                    if isinstance(rec_sum, pd.DataFrame) and not rec_sum.empty:
                        for col in ["targetMeanPrice", "targetMedianPrice", "mean", "targetPrice", "target"]:
                            if col in rec_sum.columns:
                                val = safe_num(rec_sum[col].iloc[0], 0.0)
                                if val > 0.0:
                                    target_mean_price = val
                                    break
                except Exception as e_rec:
                    logger.debug("yfinance recommendations_summary no disponible para %s: %s", ticker, e_rec)
        except Exception as e_yf:
            logger.debug("Consulta yfinance falló para %s: %s", ticker, e_yf)

    # ── CAPA 3: Fallback Finnhub remoto solo si finnhub_api_key presente y no se obtuvo nada ──
    if target_mean_price <= 0.0 and finnhub_api_key and target_data is None:
        try:
            fh_data = _finnhub_get("stock/price-target", {"symbol": ticker}, api_key=finnhub_api_key)
            if fh_data:
                target_info = fh_data[0] if (isinstance(fh_data, list) and len(fh_data) > 0 and isinstance(fh_data[0], dict)) else (fh_data if isinstance(fh_data, dict) else {})
                target_mean_val = (
                    target_info.get("targetMean")
                    or target_info.get("targetMedian")
                    or target_info.get("targetMeanPrice")
                    or target_info.get("targetMedianPrice")
                    or target_info.get("priceTarget")
                )
                target_mean_price = safe_num(target_mean_val, 0.0)
                if target_high_price <= 0.0:
                    target_high_price = safe_num(target_info.get("targetHigh", 0.0))
                if target_low_price <= 0.0:
                    target_low_price = safe_num(target_info.get("targetLow", 0.0))
        except Exception:
            pass

    if target_mean_price <= 0.0 and target_high_price > 0.0 and target_low_price > 0.0:
        target_mean_price = round((target_high_price + target_low_price) / 2.0, 2)

    return ConsensusWallStreet(
        target_mean=round(target_mean_price, 2),
        target_high=round(target_high_price, 2),
        target_low=round(target_low_price, 2),
        recommendation=recommendation,
        num_analysts=num_analysts,
        raw_data=raw_payload,
    )


@st.cache_data(ttl=FINNHUB_CACHE_TTL_METRICS)
def fetch_datos_fundamentales(
    ticker: str, finnhub_api_key: str = ""
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Extrae, homogeneiza y estructura el perfil corporativo, múltiplos clave y estados financieros
    históricos y más recientes (MRQ / TTM) a partir de Finnhub API y fallbacks institucionales.

    Endpoints consumidos:
    - `/stock/profile2`: Perfil, market cap en millones, shares outstanding en millones.
    - `/stock/metric?metric=all`: Ratios financieros TTM y series históricas anuales.
    - `/stock/financials-reported`: Estados financieros as-reported SEC (10-K y 10-Q).
    """
    ticker = str(ticker).upper().strip()
    info: Dict[str, Any] = {}
    inc = pd.DataFrame()
    bs = pd.DataFrame()
    cf = pd.DataFrame()

    if not ticker:
        return info, inc, bs, cf

    # 1. Consultas concurrentes a Finnhub
    with ThreadPoolExecutor(max_workers=5) as executor:
        f_prof = executor.submit(
            _finnhub_get, "stock/profile2", {"symbol": ticker}, finnhub_api_key
        )
        f_metric = executor.submit(
            _finnhub_get, "stock/metric", {"symbol": ticker, "metric": "all"}, finnhub_api_key
        )
        f_rep_a = executor.submit(
            _finnhub_get, "stock/financials-reported", {"symbol": ticker, "freq": "annual"}, finnhub_api_key
        )
        f_rep_q = executor.submit(
            _finnhub_get, "stock/financials-reported", {"symbol": ticker, "freq": "quarterly"}, finnhub_api_key
        )
        f_target = executor.submit(
            _finnhub_get, "stock/price-target", {"symbol": ticker}, finnhub_api_key
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
            rep_data_a = f_rep_a.result(timeout=12.0) or {}
        except Exception:
            rep_data_a = {}

        try:
            rep_data_q = f_rep_q.result(timeout=12.0) or {}
        except Exception:
            rep_data_q = {}

        try:
            target_data = f_target.result(timeout=8.0) or {}
        except Exception:
            target_data = {}

    # 2. Procesamiento de Perfil (/stock/profile2)
    mcap_m = safe_num(prof_data.get("marketCapitalization", 0.0))
    shares_m = safe_num(prof_data.get("shareOutstanding", 0.0))

    mcap = mcap_m * 1e6 if mcap_m > 0 else 0.0
    shares_outstanding = shares_m * 1e6 if shares_m > 0 else 0.0
    long_name = prof_data.get("name", ticker)
    industry_raw = prof_data.get("finnhubIndustry", "General")
    sector_std = _map_gics_sector(industry_raw)

    # 3. Procesamiento de Métricas Básicas (/stock/metric) y Precio Objetivo
    metrics_dict = metric_data.get("metric", {}) if isinstance(metric_data, dict) else {}
    series_dict = metric_data.get("series", {}).get("annual", {}) if isinstance(metric_data, dict) else {}
    yf_short = safe_num(metrics_dict.get("shortPercentOfFloat"), 0.0)

    # Extracción jerárquica robusta de Consenso de Wall Street (Finnhub + Fallback yfinance)
    target_mean_price, target_high_price, target_low_price = obtener_consenso_wall_street(
        ticker=ticker,
        finnhub_api_key=finnhub_api_key,
        target_data=target_data,
        metrics_dict=metrics_dict,
    )

    beta = safe_num(metrics_dict.get("beta", 1.0), default=1.0)
    eps_ttm = safe_num(metrics_dict.get("epsTTM", metrics_dict.get("epsNormalizedAnnual", 0.0)))
    pe_ttm = safe_num(metrics_dict.get("peTTM", metrics_dict.get("peAnnual", 0.0)))
    forward_eps = safe_num(metrics_dict.get("epsForward", 0.0))
    pe_fwd = safe_num(metrics_dict.get("peForward", metrics_dict.get("forwardPE", 0.0)))
    peg_val = safe_num(metrics_dict.get("pegTTM", metrics_dict.get("pegAnnual", metrics_dict.get("pegNormalizedAnnual", 0.0))))
    div_rate = safe_num(metrics_dict.get("dividendPerShareTTM", metrics_dict.get("dividendPerShareAnnual", 0.0)))
    div_yield_ind = safe_num(metrics_dict.get("dividendYieldIndicatedAnnual", metrics_dict.get("dividendYieldTTM", 0.0)))
    current_ratio = safe_num(metrics_dict.get("currentRatioQuarterly", metrics_dict.get("currentRatioAnnual", 0.0)), default=0.0)
    debt_to_equity = safe_num(metrics_dict.get("totalDebt/totalEquityQuarterly", metrics_dict.get("totalDebt/totalEquityAnnual", 0.0)))
    roe_val = safe_num(metrics_dict.get("roeTTM", metrics_dict.get("roeAnnual", metrics_dict.get("roeRfy", 0.0))))
    roa_val = safe_num(metrics_dict.get("roaTTM", metrics_dict.get("roaAnnual", metrics_dict.get("roaRfy", 0.0))))
    roic_val = safe_num(metrics_dict.get("roicTTM", metrics_dict.get("roicAnnual", metrics_dict.get("roicRfy", 0.0))))
    cash_per_share_metric = safe_num(metrics_dict.get("cashPerSharePerShareQuarterly", metrics_dict.get("cashPerSharePerShareAnnual", 0.0)))

    op_margins_raw = safe_num(metrics_dict.get("operatingMarginTTM", metrics_dict.get("operatingMarginAnnual", 0.0)))
    op_margins = op_margins_raw / 100.0 if op_margins_raw > 1.0 else op_margins_raw
    rev_growth_raw = safe_num(metrics_dict.get("revenueGrowthTTMYoy", metrics_dict.get("revenueGrowthQuarterlyYoy", metrics_dict.get("revenueGrowth3Y", 0.0))))
    rev_growth = rev_growth_raw / 100.0 if abs(rev_growth_raw) > 1.0 else rev_growth_raw
    eps_growth_raw = safe_num(metrics_dict.get("epsGrowthTTMYoy", metrics_dict.get("epsGrowthQuarterlyYoy", metrics_dict.get("epsGrowthAnnual", metrics_dict.get("epsGrowth3Y", 0.0)))))
    eps_growth = eps_growth_raw / 100.0 if abs(eps_growth_raw) > 1.0 else eps_growth_raw

    # 4. Funciones auxiliares de parseo XBRL
    def _get_val(concepts: List[str], label_kw: List[str], source_map: Dict[str, float], label_map: Dict[str, float]) -> float:
        for c in concepts:
            if c in source_map and source_map[c] != 0.0:
                return source_map[c]
        for kw in label_kw:
            kw_clean = kw.lower().replace("'", "").replace("’", "").replace(" ", "").strip()
            for l_key, val in label_map.items():
                l_clean = l_key.lower().replace("'", "").replace("’", "").replace(" ", "").strip()
                if kw_clean in l_clean and val != 0.0:
                    return val
        return 0.0

    def _get_total_cash_and_investments(bs_map: Dict[str, float], bs_labels: Dict[str, float]) -> float:
        comb_val = _get_val(
            ["CashCashEquivalentsAndShortTermInvestments", "CashAndShortTermInvestments", "CashCashEquivalentsAndMarketableSecurities"],
            ["cash, cash equivalents and short-term investments", "cash and short term investments", "cash and short-term investments"],
            bs_map, bs_labels
        )
        if comb_val > 0:
            return comb_val
        cash_pure = _get_val(
            ["CashAndCashEquivalentsAtCarryingValue", "CashAndCashEquivalents", "Cash", "CashEquivalentsAtCarryingValue"],
            ["cash and cash equivalents", "cash & cash equivalents", "cash"],
            bs_map, bs_labels
        )
        st_inv = _get_val(
            ["MarketableSecuritiesCurrent", "ShortTermInvestments", "AvailableForSaleSecuritiesCurrent", "OtherShortTermInvestments", "MarketableSecurities"],
            ["marketable securities", "short-term investments", "short term investments", "short-term marketable securities"],
            bs_map, bs_labels
        )
        return cash_pure + st_inv

    def _parse_filing_report(filing: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        report_items = filing.get("report", {})
        # ic
        ic_items = report_items.get("ic", [])
        ic_map = {item.get("concept", ""): safe_num(item.get("value")) for item in ic_items if "concept" in item}
        ic_labels = {item.get("label", "").lower(): safe_num(item.get("value")) for item in ic_items if "label" in item}

        rev = _get_val(
            ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet", "Revenue", "TotalRevenue"],
            ["total net sales", "revenue", "total revenue", "sales"],
            ic_map, ic_labels
        )
        gross_p = _get_val(["GrossProfit"], ["gross profit"], ic_map, ic_labels) or rev
        op_inc = _get_val(["OperatingIncomeLoss", "OperatingProfit", "OperatingIncome"], ["operating income", "operating profit"], ic_map, ic_labels)
        net_inc = _get_val(["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic", "NetIncome"], ["net income", "net loss", "net income available to common"], ic_map, ic_labels)
        int_exp = abs(_get_val(["InterestExpense", "InterestExpenseNonoperating", "InterestAndDebtExpense"], ["interest expense"], ic_map, ic_labels))
        pretax_inc = _get_val(["IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments", "PretaxIncome"], ["income before tax", "pretax income"], ic_map, ic_labels)
        tax_prov = _get_val(["IncomeTaxExpenseBenefit", "IncomeTaxExpense"], ["income tax"], ic_map, ic_labels)
        shares_dil = _get_val(["WeightedAverageNumberOfDilutedSharesOutstanding", "WeightedAverageNumberOfSharesOutstandingDiluted"], ["diluted shares", "shares diluted"], ic_map, ic_labels)
        eps_dil = _get_val(
            ["EarningsPerShareDiluted", "DilutedEarningsPerShareBasicAndDiluted", "DilutedEPS", "EarningsPerShareBasic", "BasicEPS"],
            ["diluted earnings per share", "diluted eps", "basic eps", "earnings per share"],
            ic_map, ic_labels
        )
        if eps_dil == 0.0 and net_inc != 0.0 and (shares_dil or shares_outstanding) > 0:
            eps_dil = net_inc / (shares_dil or shares_outstanding)

        inc_d = {
            "Total Revenue": rev,
            "Gross Profit": gross_p,
            "Operating Income": op_inc,
            "Net Income": net_inc,
            "Interest Expense": int_exp,
            "Pretax Income": pretax_inc,
            "Tax Provision": tax_prov,
            "Diluted EPS": eps_dil,
            "Diluted Average Shares": shares_dil or shares_outstanding,
            "Basic Average Shares": shares_dil or shares_outstanding,
        }

        # bs
        bs_items = report_items.get("bs", [])
        bs_map = {item.get("concept", ""): safe_num(item.get("value")) for item in bs_items if "concept" in item}
        bs_labels = {item.get("label", "").lower(): safe_num(item.get("value")) for item in bs_items if "label" in item}

        t_assets = _get_val(["Assets", "TotalAssets"], ["total assets"], bs_map, bs_labels)
        c_assets = _get_val(["AssetsCurrent", "TotalAssetsCurrent"], ["total current assets", "current assets"], bs_map, bs_labels)
        c_liab = _get_val(["LiabilitiesCurrent", "TotalLiabilitiesCurrent"], ["total current liabilities", "current liabilities"], bs_map, bs_labels)
        t_equity = _get_val(
            [
                "StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
                "CommonStockholdersEquity", "TotalStockholdersEquity", "TotalEquity", "CommonEquity", "TotalStockholderEquity"
            ],
            [
                "total stockholders equity", "total shareholders equity", "total equity",
                "stockholders equity", "shareholders equity", "total shareowners equity", "common stock equity"
            ],
            bs_map, bs_labels
        )
        l_debt = _get_val(
            [
                "LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations", "LongTermDebt",
                "LongTermDebtCurrent", "DebtLongtermAndShorttermCombinedTotal"
            ],
            ["long-term debt", "long term debt", "total debt"],
            bs_map, bs_labels
        )
        s_debt = _get_val(
            ["ShortTermBorrowings", "CurrentDebt", "DebtCurrent", "CommercialPaper"],
            ["short-term debt", "current portion of long-term debt", "commercial paper"],
            bs_map, bs_labels
        )
        tot_cash_inv = _get_total_cash_and_investments(bs_map, bs_labels)

        bs_d = {
            "Total Assets": t_assets,
            "Current Assets": c_assets,
            "Current Liabilities": c_liab,
            "Total Stockholder Equity": t_equity,
            "Total Debt": l_debt + s_debt,
            "Long Term Debt": l_debt,
            "Current Debt": s_debt,
            "Cash And Cash Equivalents": tot_cash_inv,
        }

        # cf
        cf_items = report_items.get("cf", [])
        cf_map = {item.get("concept", ""): safe_num(item.get("value")) for item in cf_items if "concept" in item}
        cf_labels = {item.get("label", "").lower(): safe_num(item.get("value")) for item in cf_items if "label" in item}

        ocf = _get_val(["NetCashProvidedByUsedInOperatingActivities"], ["operating activities", "operating cash flow"], cf_map, cf_labels)
        capex = abs(_get_val(["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"], ["capital expenditure", "property, plant and equipment", "additions to property"], cf_map, cf_labels))
        repurchase = abs(_get_val(
            ["PaymentsForRepurchaseOfCommonStock", "PaymentsForRepurchaseOfStock", "PaymentsForRepurchaseOfEquity", "PaymentsForRepurchaseOfPreferredStockAndPreferenceStock"],
            ["repurchase of common stock", "repurchases of stock", "common stock repurchased"],
            cf_map, cf_labels
        ))
        da_val = abs(_get_val(
            ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization", "Depreciation", "AccumulatedDepreciationDepletionAndAmortizationPropertyPlantAndEquipment"],
            ["depreciation and amortization", "depreciation", "amortization"],
            cf_map, cf_labels
        ))
        fcf = ocf - capex

        cf_d = {
            "Operating Cash Flow": ocf,
            "Capital Expenditure": capex,
            "Free Cash Flow": fcf,
            "Repurchase Of Capital Stock": repurchase,
            "Depreciation & Amortization": da_val,
        }

        return inc_d, bs_d, cf_d

    # 5. Procesamiento de Filings SEC Anuales y Trimestrales (MRQ / TTM)
    dict_inc: Dict[str, Dict[str, float]] = {}
    dict_bs: Dict[str, Dict[str, float]] = {}
    dict_cf: Dict[str, Dict[str, float]] = {}

    rep_list_a = rep_data_a.get("data", []) if isinstance(rep_data_a, dict) else []
    rep_list_q = rep_data_q.get("data", []) if isinstance(rep_data_q, dict) else []

    # Variables MRQ y TTM derivadas
    mrq_bs_d: Dict[str, float] = {}
    ttm_inc_d: Dict[str, float] = {}
    ttm_cf_d: Dict[str, float] = {}

    # Procesar MRQ (Quarterly más reciente)
    if isinstance(rep_list_q, list) and len(rep_list_q) > 0:
        latest_q = rep_list_q[0]
        _, q_bs, _ = _parse_filing_report(latest_q)
        if q_bs.get("Total Assets", 0) > 0 or q_bs.get("Current Assets", 0) > 0:
            mrq_bs_d = q_bs

        # Calcular TTM a partir de los 4 trimestres más recientes
        if len(rep_list_q) >= 4:
            q_inc_list = []
            q_cf_list = []
            for filing_q in rep_list_q[:4]:
                qi, _, qc = _parse_filing_report(filing_q)
                q_inc_list.append(qi)
                q_cf_list.append(qc)

            ttm_rev = sum(q.get("Total Revenue", 0.0) for q in q_inc_list)
            ttm_gp = sum(q.get("Gross Profit", 0.0) for q in q_inc_list)
            ttm_op = sum(q.get("Operating Income", 0.0) for q in q_inc_list)
            ttm_ni = sum(q.get("Net Income", 0.0) for q in q_inc_list)
            ttm_int = sum(q.get("Interest Expense", 0.0) for q in q_inc_list)
            ttm_pretax = sum(q.get("Pretax Income", 0.0) for q in q_inc_list)
            ttm_tax = sum(q.get("Tax Provision", 0.0) for q in q_inc_list)
            ttm_ocf = sum(q.get("Operating Cash Flow", 0.0) for q in q_cf_list)
            ttm_capex = sum(q.get("Capital Expenditure", 0.0) for q in q_cf_list)

            if ttm_rev > 0 or ttm_ni != 0:
                ttm_inc_d = {
                    "Total Revenue": ttm_rev,
                    "Gross Profit": ttm_gp or ttm_rev,
                    "Operating Income": ttm_op,
                    "Net Income": ttm_ni,
                    "Interest Expense": ttm_int,
                    "Pretax Income": ttm_pretax,
                    "Tax Provision": ttm_tax,
                    "Diluted Average Shares": shares_outstanding,
                    "Basic Average Shares": shares_outstanding,
                }
                ttm_cf_d = {
                    "Operating Cash Flow": ttm_ocf,
                    "Capital Expenditure": ttm_capex,
                    "Free Cash Flow": ttm_ocf - ttm_capex,
                    "Repurchase Of Capital Stock": sum(q.get("Repurchase Of Capital Stock", 0.0) for q in q_cf_list),
                }

            # Calcular EPS YoY y Revenue YoY interanual comparando 4Q actuales vs 4Q anteriores o t vs t-4
            if len(rep_list_q) >= 8:
                q_inc_prev = [_parse_filing_report(fq)[0] for fq in rep_list_q[4:8]]
                prev_ttm_ni = sum(q.get("Net Income", 0.0) for q in q_inc_prev)
                prev_ttm_rev = sum(q.get("Total Revenue", 0.0) for q in q_inc_prev)
                if prev_ttm_ni > 0 and ttm_ni > 0 and eps_growth == 0.0:
                    eps_growth = (ttm_ni - prev_ttm_ni) / prev_ttm_ni
                if prev_ttm_rev > 0 and ttm_rev > 0 and rev_growth == 0.0:
                    rev_growth = (ttm_rev - prev_ttm_rev) / prev_ttm_rev
            elif len(rep_list_q) >= 5 and eps_growth == 0.0:
                q_curr_ic = _parse_filing_report(rep_list_q[0])[0]
                q_prev_ic = _parse_filing_report(rep_list_q[4])[0]
                q_curr_eps = q_curr_ic.get("Diluted EPS", 0.0) or (q_curr_ic.get("Net Income", 0.0) / shares_outstanding if shares_outstanding > 0 else 0.0)
                q_prev_eps = q_prev_ic.get("Diluted EPS", 0.0) or (q_prev_ic.get("Net Income", 0.0) / shares_outstanding if shares_outstanding > 0 else 0.0)
                if q_curr_eps > 0 and q_prev_eps > 0:
                    eps_growth = (q_curr_eps - q_prev_eps) / q_prev_eps

    # Procesar serie anual (hasta 6 años para garantizar al menos 5 períodos completos)
    if isinstance(rep_list_a, list) and len(rep_list_a) > 0:
        for filing in rep_list_a[:6]:
            year_val = filing.get("year") or filing.get("endDate", "")[:4]
            if not year_val:
                continue
            period_label = str(year_val)
            inc_item, bs_item, cf_item = _parse_filing_report(filing)
            dict_inc[period_label] = inc_item
            dict_bs[period_label] = bs_item
            dict_cf[period_label] = cf_item

    # Si tenemos balance MRQ y TTM, agregarlos a los diccionarios
    if mrq_bs_d:
        dict_bs["MRQ"] = mrq_bs_d

    if ttm_inc_d:
        dict_inc["TTM"] = ttm_inc_d
    if ttm_cf_d:
        dict_cf["TTM"] = ttm_cf_d

    if dict_inc:
        inc = pd.DataFrame(dict_inc)
    if dict_bs:
        bs = pd.DataFrame(dict_bs)
    if dict_cf:
        cf = pd.DataFrame(dict_cf)

    # 6. Fallback de Series Históricas si filings SEC están vacíos
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

    # 7. Inyección y calibración de valores consolidados a info (TTM + MRQ)
    total_assets_val = mrq_bs_d.get("Total Assets", 0.0) or _extraer_val_df(bs, ["Total Assets", "totalAssets"])
    total_debt_val = mrq_bs_d.get("Total Debt", 0.0) or _extraer_val_df(bs, ["Total Debt", "totalDebt", "Long Term Debt"]) or safe_num(metrics_dict.get("totalDebtQuarterly", metrics_dict.get("netDebtAnnual", 0.0)))
    total_cash_val = mrq_bs_d.get("Cash And Cash Equivalents", 0.0) or _extraer_val_df(bs, ["Cash And Cash Equivalents", "Cash"])
    total_equity_val = mrq_bs_d.get("Total Stockholder Equity", 0.0) or _extraer_val_df(bs, ["Total Stockholder Equity", "Stockholders Equity"])
    cur_assets_val = mrq_bs_d.get("Current Assets", 0.0) or _extraer_val_df(bs, ["Total Current Assets", "Current Assets"])
    cur_liab_val = mrq_bs_d.get("Current Liabilities", 0.0) or _extraer_val_df(bs, ["Total Current Liabilities", "Current Liabilities"])
    
    ebit_val = ttm_inc_d.get("Operating Income", 0.0) or _extraer_val_df(inc, ["Operating Income", "Operating Profit", "EBIT"])
    net_income_val = ttm_inc_d.get("Net Income", 0.0) or _extraer_val_df(inc, ["Net Income"])
    rev_val = ttm_inc_d.get("Total Revenue", 0.0) or _extraer_val_df(inc, ["Total Revenue", "Revenue"])
    gross_profit_val = ttm_inc_d.get("Gross Profit", 0.0) or _extraer_val_df(inc, ["Gross Profit"]) or rev_val
    ocf_val = ttm_cf_d.get("Operating Cash Flow", 0.0) or _extraer_val_df(cf, ["Operating Cash Flow"])
    fcf_val = ttm_cf_d.get("Free Cash Flow", 0.0) or _extraer_val_df(cf, ["Free Cash Flow"]) or safe_num(metrics_dict.get("fcfTTM", metrics_dict.get("fcfAnnual", 0.0)))
    int_exp_val = abs(ttm_inc_d.get("Interest Expense", 0.0) or _extraer_val_df(inc, ["Interest Expense"]))
    pretax_val = ttm_inc_d.get("Pretax Income", 0.0) or _extraer_val_df(inc, ["Pretax Income"])
    tax_prov_val = ttm_inc_d.get("Tax Provision", 0.0) or _extraer_val_df(inc, ["Tax Provision"])
    ebitda_val = safe_num(metrics_dict.get("ebitdaTTM", metrics_dict.get("ebitdaAnnual", 0.0))) or (ebit_val * 1.15)

    # ── CAPA 4 (Enriquecimiento y Fallback Institucional con yfinance) ──
    if (mcap <= 0.0 or inc.empty or bs.empty or cf.empty or ebitda_val <= 0.0 or roe_val <= 0.0 or eps_ttm <= 0.0 or rev_val <= 0.0):
        try:
            import yfinance as yf
            ticker_yf = yf.Ticker(ticker)
            yf_info = getattr(ticker_yf, "info", None) or {}

            if isinstance(yf_info, dict) and yf_info:
                if mcap <= 0.0:
                    mcap = safe_num(yf_info.get("marketCap"), 0.0)
                if shares_outstanding <= 0.0:
                    shares_outstanding = safe_num(yf_info.get("sharesOutstanding"), 0.0)
                if not long_name or long_name == ticker:
                    long_name = yf_info.get("longName") or yf_info.get("shortName") or ticker
                if not sector_std or sector_std == "General":
                    sector_std = _map_gics_sector(yf_info.get("sector") or yf_info.get("industry") or "")
                    industry_raw = yf_info.get("industry") or industry_raw

                if beta == 1.0 or beta <= 0.0:
                    beta = safe_num(yf_info.get("beta"), 1.0)
                if eps_ttm <= 0.0:
                    eps_ttm = safe_num(yf_info.get("trailingEps"), 0.0)
                if forward_eps <= 0.0:
                    forward_eps = safe_num(yf_info.get("forwardEps"), 0.0)
                if pe_ttm <= 0.0:
                    pe_ttm = safe_num(yf_info.get("trailingPE"), 0.0)
                if pe_fwd <= 0.0:
                    pe_fwd = safe_num(yf_info.get("forwardPE"), 0.0)
                if div_rate <= 0.0:
                    div_rate = safe_num(yf_info.get("dividendRate"), 0.0)
                if div_yield_ind <= 0.0:
                    div_yield_ind = safe_num(yf_info.get("dividendYield"), 0.0)
                if debt_to_equity <= 0.0:
                    debt_to_equity = safe_num(yf_info.get("debtToEquity"), 0.0)

                # Current Ratio Finviz/MRQ
                yf_cr = safe_num(yf_info.get("currentRatio"), 0.0)
                if yf_cr > 0.0:
                    current_ratio = yf_cr

                # ROE / ROA (yfinance viene en decimales, ej. 2.412 = 241.2%)
                raw_roe = safe_num(yf_info.get("returnOnEquity"), 0.0)
                if raw_roe > 0.0:
                    raw_roe_pct = raw_roe * 100.0 if raw_roe <= 10.0 else raw_roe
                    if roe_val <= 0.0 or roe_val > 300.0 or abs(roe_val - raw_roe_pct) > 50.0:
                        roe_val = raw_roe_pct

                raw_roa = safe_num(yf_info.get("returnOnAssets"), 0.0)
                if raw_roa > 0.0:
                    raw_roa_pct = raw_roa * 100.0 if raw_roa <= 10.0 else raw_roa
                    if roa_val <= 0.0 or roa_val > 45.0 or abs(roa_val - raw_roa_pct) > 15.0:
                        roa_val = raw_roa_pct

                yf_roic = safe_num(yf_info.get("roic"), 0.0)
                if yf_roic > 0.0:
                    roic_val = yf_roic * 100.0 if yf_roic <= 1.0 else yf_roic

                if total_assets_val <= 0.0:
                    total_assets_val = safe_num(yf_info.get("totalAssets"), 0.0)
                if total_debt_val <= 0.0:
                    total_debt_val = safe_num(yf_info.get("totalDebt"), 0.0)
                if total_cash_val <= 0.0:
                    total_cash_val = safe_num(yf_info.get("totalCash"), 0.0)
                if total_equity_val <= 0.0:
                    bv = safe_num(yf_info.get("bookValue"), 0.0)
                    total_equity_val = (bv * shares_outstanding) if (bv > 0 and shares_outstanding > 0) else safe_num(yf_info.get("totalStockholderEquity"), 0.0)

                yf_ebitda = safe_num(yf_info.get("ebitda"), 0.0)
                if ebitda_val <= 0.0 and yf_ebitda > 0.0:
                    ebitda_val = yf_ebitda

                yf_ocf = safe_num(yf_info.get("operatingCashflow"), 0.0)
                if ocf_val <= 0.0 and yf_ocf > 0.0:
                    ocf_val = yf_ocf

                yf_fcf = safe_num(yf_info.get("freeCashflow"), 0.0)
                if fcf_val <= 0.0 and yf_fcf > 0.0:
                    fcf_val = yf_fcf

                if rev_val <= 0.0:
                    rev_val = safe_num(yf_info.get("totalRevenue"), 0.0)
                if net_income_val == 0.0:
                    net_income_val = safe_num(yf_info.get("netIncomeToCommon") or yf_info.get("netIncome"), 0.0)
                if ebit_val <= 0.0:
                    ebit_val = safe_num(yf_info.get("operatingIncome"), ebitda_val * 0.85 if ebitda_val > 0 else 0.0)

                if roic_val <= 0.0 and total_equity_val > 0 and total_debt_val >= 0 and ebit_val > 0:
                    inv_c = total_equity_val + total_debt_val
                    if inv_c > 0:
                        roic_val = (ebit_val * (1.0 - 0.21) / inv_c) * 100.0

                yf_eg = safe_num(yf_info.get("earningsGrowth"), safe_num(yf_info.get("earningsQuarterlyGrowth"), 0.0))
                if yf_eg != 0.0 and (eps_growth == 0.0 or (abs(eps_growth) > 1.0 and abs(yf_eg) < 1.0)):
                    eps_growth = yf_eg

                yf_rg = safe_num(yf_info.get("revenueGrowth"), safe_num(yf_info.get("revenueQuarterlyGrowth"), 0.0))
                if yf_rg != 0.0 and rev_growth == 0.0:
                    rev_growth = yf_rg

                yf_peg = safe_num(yf_info.get("pegRatio"), 0.0)
                if yf_peg > 0.0 and peg_val <= 0.0:
                    peg_val = yf_peg

                # Target Price
                if target_mean_price <= 0.0:
                    t_mean_yf, t_high_yf, t_low_yf = obtener_consenso_wall_street(ticker)
                    if t_mean_yf > 0.0:
                        target_mean_price = t_mean_yf
                        target_high_price = t_high_yf
                        target_low_price = t_low_yf

                yf_short = safe_num(
                    yf_info.get("shortPercentOfFloat")
                    or yf_info.get("sharesPercentSharesOut")
                    or yf_info.get("shortPercentOfSharesOutstanding"),
                    0.0
                )

            # Enriquecer DataFrames si están vacíos
            if inc.empty:
                yf_inc = getattr(ticker_yf, "financials", None)
                if yf_inc is None or (isinstance(yf_inc, pd.DataFrame) and yf_inc.empty):
                    yf_inc = getattr(ticker_yf, "income_stmt", None)
                if isinstance(yf_inc, pd.DataFrame) and not yf_inc.empty:
                    inc = yf_inc.copy()
                    inc.columns = [str(c)[:4] for c in inc.columns]

            if bs.empty:
                yf_bs = getattr(ticker_yf, "balance_sheet", None)
                if isinstance(yf_bs, pd.DataFrame) and not yf_bs.empty:
                    bs = yf_bs.copy()
                    bs.columns = [str(c)[:4] for c in bs.columns]
                yf_q_bs = getattr(ticker_yf, "quarterly_balance_sheet", None)
                if isinstance(yf_q_bs, pd.DataFrame) and not yf_q_bs.empty:
                    bs["MRQ"] = yf_q_bs.iloc[:, 0]

            if cf.empty:
                yf_cf = getattr(ticker_yf, "cashflow", None)
                if isinstance(yf_cf, pd.DataFrame) and not yf_cf.empty:
                    cf = yf_cf.copy()
                    cf.columns = [str(c)[:4] for c in cf.columns]
                yf_q_cf = getattr(ticker_yf, "quarterly_cashflow", None)
                if isinstance(yf_q_cf, pd.DataFrame) and not yf_q_cf.empty and len(yf_q_cf.columns) >= 4:
                    cf["TTM"] = yf_q_cf.iloc[:, :4].sum(axis=1)

        except Exception as e_enrich:
            logger.debug("Error en enriquecimiento yfinance para %s: %s", ticker, e_enrich)

    # Garantizar extracción de Target Price de Consenso Wall Street desde yfinance
    if target_mean_price <= 0.0:
        try:
            cw_fund = obtener_consenso_wall_street(ticker)
            if cw_fund.target_mean > 0:
                target_mean_price = cw_fund.target_mean
                target_high_price = cw_fund.target_high
                target_low_price = cw_fund.target_low
        except Exception as e_cw:
            logger.debug("Error extrayendo consenso en fetch_datos_fundamentales para %s: %s", ticker, e_cw)

    # Calibración de métricas de mercado (Finviz Standards)
    if current_ratio <= 0.0 and cur_assets_val > 0 and cur_liab_val > 0:
        current_ratio = cur_assets_val / cur_liab_val

    if cur_assets_val > 0 and cur_liab_val > 0:
        cr_calculated = cur_assets_val / cur_liab_val
        if current_ratio <= 0.0 or abs(cr_calculated - current_ratio) < 0.5:
            current_ratio = cr_calculated

    if roe_val <= 0.0 and total_equity_val > 0 and net_income_val != 0.0:
        roe_val = (net_income_val / total_equity_val) * 100.0

    if roa_val <= 0.0 and total_assets_val > 0 and net_income_val != 0.0:
        roa_val = (net_income_val / total_assets_val) * 100.0

    if total_equity_val <= 0.0 and roe_val > 0.0 and net_income_val > 0.0:
        total_equity_val = net_income_val / (roe_val / 100.0)

    if forward_eps <= 0.0 and eps_ttm > 0:
        forward_eps = eps_ttm * (1.0 + max(eps_growth if eps_growth > 0 else 0.08, 0.05))

    # Cálculo defensivo de PEG Forward si no vino directo de API
    if peg_val <= 0.0:
        fwd_growth_calc = ((forward_eps - eps_ttm) / eps_ttm) if (forward_eps > eps_ttm and eps_ttm > 0) else (eps_growth if eps_growth > 0 else 0.0)
        if pe_fwd > 0 and fwd_growth_calc > 0:
            g_pct_fwd = fwd_growth_calc * 100.0 if fwd_growth_calc < 1.0 else fwd_growth_calc
            if g_pct_fwd > 0:
                peg_val = pe_fwd / g_pct_fwd
        elif pe_ttm > 0 and eps_growth > 0:
            g_pct_ttm = eps_growth * 100.0 if eps_growth < 1.0 else eps_growth
            if g_pct_ttm > 0:
                peg_val = pe_ttm / g_pct_ttm

    # Caja por acción y Caja Neta por acción
    cash_per_share = (total_cash_val / shares_outstanding) if shares_outstanding > 0 else cash_per_share_metric
    net_cash_per_share = ((total_cash_val - total_debt_val) / shares_outstanding) if shares_outstanding > 0 else 0.0

    info.update({
        "symbol": ticker,
        "longName": long_name,
        "sector": sector_std,
        "industry": industry_raw,
        "marketCap": mcap,
        "sharesOutstanding": shares_outstanding,
        "beta": beta,
        "dividendRate": div_rate,
        "dividendYield": div_yield_ind,
        "trailingEps": eps_ttm,
        "forwardEps": forward_eps,
        "forwardPE": pe_fwd,
        "trailingPE": pe_ttm,
        "currentRatio": current_ratio,
        "debtToEquity": debt_to_equity,
        "returnOnEquity": roe_val,
        "returnOnAssets": roa_val,
        "roic": roic_val,
        "cashPerShare": cash_per_share,
        "netCashPerShare": net_cash_per_share,
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
        "pegRatio": peg_val if peg_val > 0 else (safe_num(pe_ttm / (eps_growth * 100.0)) if (pe_ttm > 0 and eps_growth > 0) else 0.0),
        "targetMeanPrice": target_mean_price,
        "targetHighPrice": target_high_price,
        "targetLowPrice": target_low_price,
        "shortPercentOfFloat": yf_short if yf_short > 0 else safe_num(metrics_dict.get("shortPercentOfFloat"), 0.0),
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
    val = safe_num(fallback_fred, default=4.20)
    return val if val > 0 else 4.20


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
    """Calcula el dividendo anualizado, el yield (%) y la fecha ex-dividendo con fallbacks TTM."""
    div_rate = safe_get(info_dict, ["dividendRate", "dividendPerShareTTM", "dividendPerShareAnnual", "trailingAnnualDividendRate", "lastDiv"], 0.0)
    div_rate = safe_num(div_rate, 0.0)
    
    if div_rate > 0 and precio_ref > 0:
        div_yield = (div_rate / precio_ref) * 100.0
    else:
        div_yield = safe_get(info_dict, ["dividendYield", "dividendYieldIndicatedAnnual", "dividendYieldTTM"], 0.0)
        div_yield = safe_num(div_yield, 0.0)
        if div_yield > 0 and precio_ref > 0 and div_rate == 0.0:
            div_rate = (div_yield / 100.0) * precio_ref

    ex_div_ts = safe_get(info_dict, ["exDividendDate"], None)
    if ex_div_ts and isinstance(ex_div_ts, (int, float)):
        try:
            next_div_date = datetime.datetime.fromtimestamp(ex_div_ts).strftime('%Y-%m-%d')
        except Exception:
            next_div_date = "N/A"
    else:
        next_div_date = "N/A"
    return round(div_rate, 4), round(div_yield, 3), next_div_date


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
    ocf_ttm = safe_num(info.get("operatingCashflow", 0.0), 0.0)
    fcf_ttm = safe_num(info.get("freeCashflow", 0.0), 0.0)
    capex_ttm = abs(safe_num(info.get("capitalExpenditures", 0.0), 0.0))
    if capex_ttm == 0.0 and ocf_ttm > 0 and fcf_ttm > 0 and ocf_ttm >= fcf_ttm:
        capex_ttm = ocf_ttm - fcf_ttm

    ebit_ttm = safe_num(info.get("operatingIncome", 0.0), 0.0)

    ocf_hist = _extraer_serie(cf, [
        "Operating Cash Flow", "OperatingCashFlow",
        "Cash Flow From Continuing Operating Activities",
        "Total Cash From Operating Activities",
        "Net Cash Provided By Operating Activities"
    ])
    capex_hist = obtener_capex_historico(cf)

    # Si cf carece de datos históricos, usar TTM como fallback
    if not ocf_hist:
        ocf_hist = [ocf_ttm] if ocf_ttm > 0 else [0.0]
    elif ocf_ttm > 0 and "TTM" in cf.columns:
        ocf_hist[0] = ocf_ttm

    if not capex_hist:
        capex_hist = [capex_ttm] if capex_ttm > 0 else [0.0]
    elif capex_ttm > 0 and "TTM" in cf.columns:
        capex_hist[0] = capex_ttm

    # Alinear longitud de series para consistencia período a período
    if ocf_hist and capex_hist:
        min_c = min(len(ocf_hist), len(capex_hist))
        ocf_hist = ocf_hist[:min_c]
        capex_hist = capex_hist[:min_c]

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
    if ebit_ttm > 0:
        if not ebit_hist:
            ebit_hist = [ebit_ttm]
        elif abs(ebit_hist[0] - ebit_ttm) > 1000 and "TTM" not in inc.columns:
            ebit_hist = [ebit_ttm] + ebit_hist
    elif not ebit_hist:
        ebit_hist = []

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

    da_ttm = _extraer_val_df(cf, [
        "Depreciation & Amortization", "Depreciation And Amortization",
        "Depreciation Amortization Depletion", "Depreciation", "DepreciationAmortization"
    ], default=0.0)
    if da_ttm == 0.0:
        da_ttm = _extraer_val_df(inc, [
            "Reconciled Depreciation", "Depreciation And Amortization In Income Statement",
            "Depreciation & Amortization", "Depreciation"
        ], default=0.0)

    ebitda_ttm = safe_num(info.get("ebitda", 0.0), 0.0)
    if ebitda_ttm <= 0:
        ebitda_ttm = _extraer_val_df(inc, ["EBITDA", "Normalized EBITDA", "ebitda"], default=0.0)
    if ebitda_ttm <= 0 and operating_income_ttm > 0:
        ebitda_ttm = operating_income_ttm + (da_ttm if da_ttm > 0 else operating_income_ttm * 0.15)

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
    fcf_info = safe_num(info.get("freeCashflow", 0.0), 0.0)
    if fcf_info > 0.0 and (fcf_ttm <= 0.0 or abs(fcf_ttm - fcf_info) > fcf_info * 0.35):
        fcf_ttm = fcf_info
    elif fcf_ttm == 0.0:
        fcf_ttm = fcf_info

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
            "Cash, Cash Equivalents & Short Term Investments",
            "Marketable Securities", "Short Term Investments", "Cash"
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

    if total_equity == 0.0 and total_assets > 0:
        total_liabilities = _extraer_val_df(bs, ["Total Liabilities", "TotalLiabilities", "Liabilities"], default=0.0)
        if total_liabilities > 0:
            total_equity = max(total_assets - total_liabilities, 0.0)

    current_assets = safe_num(info.get("totalCurrentAssets", 0.0), 0.0)
    if current_assets == 0.0:
        current_assets = _extraer_val_df(bs, ["Total Current Assets", "Current Assets", "CurrentAssets"], default=0.0)

    current_liabilities = safe_num(info.get("totalCurrentLiabilities", 0.0), 0.0)
    if current_liabilities == 0.0:
        current_liabilities = _extraer_val_df(bs, ["Total Current Liabilities", "Current Liabilities", "CurrentLiabilities"], default=0.0)

    short_term_debt = _extraer_val_df(bs, [
        "Current Debt", "Current Debt And Capital Lease Obligation", "Short Term Debt", "CurrentDebt"
    ], default=0.0)

    # Calibración de métricas fundamentales
    roe_info = safe_num(info.get("returnOnEquity", 0.0), 0.0)
    if roe_info <= 0.0 and total_equity > 0 and net_income_ttm != 0.0:
        roe_info = (net_income_ttm / total_equity) * 100.0

    roa_info = safe_num(info.get("returnOnAssets", 0.0), 0.0)
    if roa_info <= 0.0 and total_assets > 0 and net_income_ttm != 0.0:
        roa_info = (net_income_ttm / total_assets) * 100.0

    roic_info = safe_num(info.get("roic", 0.0), 0.0)
    if roic_info <= 0.0 and total_equity > 0 and total_debt >= 0 and operating_income_ttm > 0:
        inv_cap_std = total_equity + total_debt
        if inv_cap_std > 0:
            nopat_calc = operating_income_ttm * (1.0 - 0.21)
            roic_info = (nopat_calc / inv_cap_std) * 100.0

    current_ratio_info = safe_num(info.get("currentRatio", 0.0), 0.0)
    if current_ratio_info <= 0.0 and current_assets > 0 and current_liabilities > 0:
        current_ratio_info = current_assets / current_liabilities

    cash_per_share = (total_cash / shares_diluted) if shares_diluted > 0 else safe_num(info.get("cashPerShare", 0.0), 0.0)
    net_cash_per_share = ((total_cash - total_debt) / shares_diluted) if shares_diluted > 0 else safe_num(info.get("netCashPerShare", 0.0), 0.0)

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
                                op_margin_hist = max(safe_num(np.mean(mgs), default=0.0), 0.0)
                            break
                    if op_margin_hist > 0:
                        break
                except Exception:
                    pass

    if op_margin_hist == 0.0 and revenue_ttm > 0 and operating_income_ttm > 0:
        op_margin_hist = operating_income_ttm / revenue_ttm

    earnings_growth = safe_num(info.get("earningsGrowth", 0.0), 0.0)
    if earnings_growth == 0.0 and isinstance(inc, pd.DataFrame) and not inc.empty:
        for fila_ni in [
            "Diluted EPS", "DilutedEPS", "Basic EPS", "BasicEPS",
            "Net Income", "NetIncome", "Net Income Common Stockholders",
            "Net Income To Common", "Operating Income", "OperatingIncome"
        ]:
            if fila_ni in inc.index:
                try:
                    s_ni = inc.loc[fila_ni].dropna()
                    if isinstance(s_ni, pd.DataFrame):
                        s_ni = s_ni.iloc[0]
                    vals_ni = [safe_num(v, 0.0) for v in s_ni.values if safe_num(v, 0.0) > 0]
                    if len(vals_ni) >= 2 and vals_ni[0] > 0 and vals_ni[1] > 0:
                        yoy_ni = (vals_ni[0] - vals_ni[1]) / vals_ni[1]
                        if -0.60 <= yoy_ni <= 2.5:
                            earnings_growth = yoy_ni
                    elif len(vals_ni) >= 2 and vals_ni[0] > 0 and vals_ni[-1] > 0:
                        n_anios_ni = min(len(vals_ni) - 1, 3)
                        ni_cagr = (vals_ni[0] / vals_ni[n_anios_ni]) ** (1.0 / n_anios_ni) - 1.0
                        if -0.50 <= ni_cagr <= 1.0:
                            earnings_growth = ni_cagr
                    if earnings_growth != 0.0:
                        break
                except Exception:
                    pass

    revenue_growth = safe_num(info.get("revenueGrowth", 0.0), 0.0)
    if revenue_growth == 0.0 and cagr_revenue_3_5y != 0.0:
        revenue_growth = cagr_revenue_3_5y

    peg_ratio_info = safe_num(info.get("pegRatio", 0.0), 0.0)
    if peg_ratio_info <= 0.0:
        pe_calc = (precio_actual / eps_diluted_ttm) if (eps_diluted_ttm > 0 and precio_actual > 0) else 0.0
        pe_fwd_calc = (precio_actual / forward_eps) if (forward_eps > 0 and precio_actual > 0) else 0.0
        g_for_peg = earnings_growth
        if g_for_peg <= 0.0 and forward_eps > eps_diluted_ttm > 0:
            g_for_peg = (forward_eps - eps_diluted_ttm) / eps_diluted_ttm

        if pe_fwd_calc > 0 and g_for_peg > 0:
            g_pct = g_for_peg * 100.0 if g_for_peg < 1.0 else g_for_peg
            peg_ratio_info = (pe_fwd_calc / g_pct) if g_pct > 0 else 0.0
        elif pe_calc > 0 and g_for_peg > 0:
            g_pct = g_for_peg * 100.0 if g_for_peg < 1.0 else g_for_peg
            peg_ratio_info = (pe_calc / g_pct) if g_pct > 0 else 0.0

    beta = safe_num(info.get("beta", 1.0), 1.0)
    short_percent_of_float = safe_num(
        info.get("shortPercentOfFloat")
        or info.get("short_percent_of_float")
        or info.get("sharesPercentSharesOut"),
        0.0
    )
    target_mean_price = safe_num(
        info.get("targetMeanPrice")
        or info.get("target_mean_price")
        or info.get("targetMean")
        or info.get("targetMedian")
        or info.get("targetPrice")
        or info.get("priceTarget")
        or info.get("targetMedianPrice")
        or info.get("target_median")
        or info.get("target_mean"),
        0.0
    )
    if target_mean_price <= 0.0 and info.get("symbol"):
        sym = str(info.get("symbol")).upper().strip()
        t_mean, _, _ = obtener_consenso_wall_street(sym)
        if t_mean > 0.0:
            target_mean_price = t_mean

    return {
        "shares_diluted": shares_diluted,
        "shares_current": shares_diluted,
        "mcap": mcap,
        "market_cap": mcap,
        "revenue_ttm": revenue_ttm,
        "gross_profit_ttm": gross_profit_ttm,
        "operating_income_ttm": operating_income_ttm,
        "ebitda_ttm": ebitda_ttm,
        "ebitda": ebitda_ttm,
        "net_income_ttm": net_income_ttm,
        "eps_diluted_ttm": eps_diluted_ttm,
        "eps_ttm": eps_diluted_ttm,
        "eps": eps_diluted_ttm,
        "forward_eps": forward_eps,
        "pretax_income_ttm": pretax_income_ttm,
        "tax_provision_ttm": tax_provision_ttm,
        "interest_expense_ttm": interest_expense_ttm,
        "ocf_ttm": ocf_ttm,
        "operating_cashflow": ocf_ttm,
        "capex_ttm": capex_ttm,
        "fcf_ttm": fcf_ttm,
        "free_cashflow": fcf_ttm,
        "total_debt": total_debt,
        "total_cash": total_cash,
        "net_debt": total_debt - total_cash,
        "total_equity": total_equity,
        "total_assets": total_assets,
        "current_assets": current_assets,
        "current_liabilities": current_liabilities,
        "short_term_debt": short_term_debt,
        "roe": roe_info,
        "roa": roa_info,
        "roic": roic_info,
        "current_ratio": current_ratio_info,
        "cash_per_share": cash_per_share,
        "net_cash_per_share": net_cash_per_share,
        "earnings_growth": earnings_growth,
        "revenue_growth": revenue_growth,
        "cagr_revenue_3_5y": cagr_revenue_3_5y,
        "op_margin_hist": op_margin_hist,
        "peg_ratio_info": peg_ratio_info,
        "beta": beta,
        "short_percent_of_float": short_percent_of_float,
        "target_mean_price": target_mean_price,
    }
