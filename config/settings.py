import math
import os
from typing import Any
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES GLOBALES DEL MOTOR DCF / FCFF
# ─────────────────────────────────────────────────────────────────────────────

#: Recompra neta anual por defecto (fracción; 0.0 = sin ajuste).
DEFAULT_BUYBACK_RATE: float = 0.0

#: Años de transición lineal desde g_inicial hasta g_terminal (fade period).
DEFAULT_FADE_YEARS: int = 3

#: Spread mínimo que debe mantener el WACC sobre la tasa terminal para
#: evitar denominadores negativos o asíntotas en el Valor Terminal de Gordon.
WACC_MIN_SPREAD_OVER_G: float = 0.015

#: Piso absoluto del WACC expresado en porcentaje (%).
WACC_FLOOR: float = 3.0

#: Techo absoluto del WACC expresado en porcentaje (%).
WACC_CEILING: float = 25.0

#: Tasa terminal de crecimiento perpetuo por defecto (fracción).
G_TERM_DEFAULT: float = 0.025

#: Tooltip explicativo para el control de Años Fade Period en la interfaz.
TOOLTIP_FADE_YEARS: str = (
    "Periodo de convergencia lineal desde el crecimiento inicial hacia la tasa terminal (g_term).\n\n"
    "Guía de calibración según foso económico:\n"
    "• 1 a 2 años: Empresas cíclicas, materias primas o ensambladoras de hardware.\n"
    "• 3 a 4 años: Empresas maduras con ventajas competitivas sólidas (ej. Apple, Google).\n"
    "• 5 años: Monopolios naturales, duopolios o efectos de red críticos (ej. Visa, Mastercard, ASML)."
)

SECTOR_BENCHMARKS = {
    "Technology": {"PE": 28.0, "PEG": 1.5, "PFCF": 25.0, "ROA": 10.0, "ROE": 20.0, "ROI": 15.0},
    "Healthcare": {"PE": 22.0, "PEG": 1.8, "PFCF": 20.0, "ROA": 8.0, "ROE": 15.0, "ROI": 10.0},
    "Financial Services": {"PE": 14.0, "PEG": 1.2, "PFCF": 12.0, "ROA": 2.0, "ROE": 12.0, "ROI": 8.0},
    "Consumer Cyclical": {"PE": 20.0, "PEG": 1.4, "PFCF": 18.0, "ROA": 6.0, "ROE": 14.0, "ROI": 10.0},
    "Consumer Defensive": {"PE": 22.0, "PEG": 2.0, "PFCF": 20.0, "ROA": 7.0, "ROE": 18.0, "ROI": 12.0},
    "Industrials": {"PE": 18.0, "PEG": 1.5, "PFCF": 16.0, "ROA": 5.0, "ROE": 13.0, "ROI": 9.0},
    "Energy": {"PE": 12.0, "PEG": 1.0, "PFCF": 10.0, "ROA": 8.0, "ROE": 18.0, "ROI": 12.0},
    "Utilities": {"PE": 16.0, "PEG": 2.2, "PFCF": 15.0, "ROA": 3.0, "ROE": 10.0, "ROI": 6.0},
    "Real Estate": {"PE": 30.0, "PEG": 2.5, "PFCF": 22.0, "ROA": 3.0, "ROE": 8.0, "ROI": 5.0},
    "Communication Services": {"PE": 20.0, "PEG": 1.3, "PFCF": 18.0, "ROA": 6.0, "ROE": 15.0, "ROI": 10.0},
    "Basic Materials": {"PE": 18.0, "PEG": 1.6, "PFCF": 15.0, "ROA": 5.0, "ROE": 14.0, "ROI": 8.0},
}

def limpiar_texto(texto):
    if texto is None:
        return ""
    if isinstance(texto, (bytes, bytearray)):
        try:
            texto = texto.decode('utf-8', 'ignore')
        except Exception:
            texto = str(texto)
    elif not isinstance(texto, str):
        texto = str(texto)
    try:
        return unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('ascii')
    except Exception:
        return str(texto)

def sanitizar_para_pdf(texto):
    if texto is None:
        return ""
    if isinstance(texto, (bytes, bytearray)):
        try:
            texto = texto.decode('utf-8', 'ignore')
        except Exception:
            texto = str(texto)
    elif not isinstance(texto, str):
        texto = str(texto)
        
    texto = (
        texto.replace('**', '')
        .replace('•', '-')
        .replace('–', '-')
        .replace('—', '-')
        .replace('”', '"')
        .replace('“', '"')
        .replace('’', "'")
        .replace('‘', "'")
        .replace('…', '...')
        .replace('€', 'EUR')
    )
    try:
        return texto.encode('latin-1', 'ignore').decode('latin-1').strip()
    except Exception:
        return str(texto).strip()

def safe_get(d, keys, default=0):
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
# CONSTANTES DE INTEGRACIÓN FINNHUB API
# ─────────────────────────────────────────────────────────────────────────────

#: URL base de la API oficial de Finnhub (v1)
FINNHUB_BASE_URL: str = "https://finnhub.io/api/v1"

#: TTL de caché para cotizaciones intradía y velas (segundos)
FINNHUB_CACHE_TTL_QUOTE: int = 60

#: TTL de caché para métricas fundamentales, estados financieros y perfiles (segundos)
FINNHUB_CACHE_TTL_METRICS: int = 43200

#: TTL de caché para noticias corporativas recientes (segundos)
FINNHUB_CACHE_TTL_NEWS: int = 900

#: TTL de caché para series macroeconómicas de FRED (segundos)
FRED_CACHE_TTL: int = 3600


def obtener_ruta_logo():
    posibles_rutas = ["logo.png", "logo.jpg", "logo.jpeg", "assets/logo.png", "images/logo.png"]
    for r in posibles_rutas:
        if os.path.exists(r):
            return r
    return None

def cargar_secrets():
    """
    Carga las claves de API necesarias desde st.secrets o variables de entorno (os.environ).
    Retorna: (gemini_key, fred_key, finnhub_key).
    """
    def _obtener_clave(nombres: list[str]) -> str:
        for nom in nombres:
            try:
                if hasattr(st, "secrets") and nom in st.secrets and st.secrets[nom]:
                    return str(st.secrets[nom]).strip()
            except Exception:
                pass
            env_val = os.getenv(nom)
            if env_val and env_val.strip():
                return env_val.strip()
        return ""

    gemini_key = _obtener_clave(["GEMINI_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"])
    fred_key = _obtener_clave(["FRED_KEY", "FRED_API_KEY"])
    finnhub_key = _obtener_clave(["FINNHUB_API_KEY", "FINNHUB_KEY", "FMP_KEY"])

    if not gemini_key and not fred_key and not finnhub_key:
        try:
            st.error(
                "⚠️ Faltan claves en la configuración. Asegúrate de configurar 'FINNHUB_API_KEY', 'GEMINI_KEY' y 'FRED_KEY' "
                "en st.secrets o en tu archivo .env."
            )
            st.stop()
        except Exception:
            pass

    return gemini_key, fred_key, finnhub_key
