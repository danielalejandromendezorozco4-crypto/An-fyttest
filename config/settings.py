import os
import unicodedata
import streamlit as st

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
    return unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('ascii')

def sanitizar_para_pdf(texto):
    if not isinstance(texto, str): return str(texto)
    texto = texto.replace('**', '').replace('•', '-').replace('–', '-').replace('—', '-').replace('”', '"').replace('“', '"')
    return texto.encode('latin-1', 'ignore').decode('latin-1').strip()

def safe_get(d, keys, default=0):
    for key in keys:
        if key in d and d[key] is not None:
            return d[key]
    return default

def obtener_ruta_logo():
    posibles_rutas = ["logo.png", "logo.jpg", "logo.jpeg", "assets/logo.png", "images/logo.png"]
    for r in posibles_rutas:
        if os.path.exists(r):
            return r
    return None

def cargar_secrets():
    try:
        gemini_key = st.secrets["GEMINI_KEY"]
        fred_key = st.secrets["FRED_KEY"]
        fmp_key = st.secrets.get("FMP_KEY", None)
        return gemini_key, fred_key, fmp_key
    except KeyError:
        st.error("⚠️ Faltan claves en st.secrets. Asegúrate de tener 'GEMINI_KEY' y 'FRED_KEY' guardadas en el menú de Streamlit.")
        st.stop()
