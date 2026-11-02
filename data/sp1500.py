import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from typing import Any, List, Optional
import streamlit as st
import pandas as pd
import requests

@st.cache_data(ttl=86400)
def obtener_directorio_sp1500():
    df_sp500 = pd.DataFrame()
    df_mid = pd.DataFrame()
    df_sml = pd.DataFrame()
    
    # CAPA 1: Extraer S&P 500 desde CDN Sólido
    try:
        url_sp500 = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        df_sp500 = pd.read_csv(url_sp500)
        df_sp500 = df_sp500[['Symbol', 'Security', 'GICS Sector', 'GICS Sub Industry']]
        df_sp500.rename(columns={'GICS Sub Industry': 'GICS Sub-Industry'}, inplace=True)
    except Exception:
        pass
        
    # CAPA 2: Extraer Mid y Small Caps de Wikipedia con Stealth Mode
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
    }
    try:
        req_mid = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_400_companies', headers=headers, timeout=5)
        df_mid = pd.read_html(req_mid.text)[0][['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry']]
    except Exception:
        pass
        
    try:
        req_sml = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_600_companies', headers=headers, timeout=5)
        df_sml = pd.read_html(req_sml.text)[0][['Symbol', 'Company', 'GICS Sector', 'GICS Sub-Industry']]
        df_sml.rename(columns={'Company': 'Security'}, inplace=True)
    except Exception:
        pass
        
    # FUSIÓN DE CAPAS
    df_list = [df for df in [df_sp500, df_mid, df_sml] if not df.empty]
    
    if len(df_list) > 0:
        df_total = pd.concat(df_list, ignore_index=True)
        df_total['Symbol'] = df_total['Symbol'].astype(str).str.replace('.', '-', regex=False)
        return df_total
    else:
        # CAPA 3 (Absolute Fallback): Diccionario de Emergencia Mínimo
        datos_fallback = {
            'Symbol': ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'JPM', 'V', 'JNJ'],
            'Security': ['Apple Inc.', 'Microsoft Corp.', 'NVIDIA Corp.', 'Alphabet Inc.', 'Amazon.com', 'Meta Platforms', 'Tesla Inc.', 'JPMorgan Chase', 'Visa Inc.', 'Johnson & Johnson'],
            'GICS Sector': ['Information Technology', 'Information Technology', 'Information Technology', 'Communication Services', 'Consumer Discretionary', 'Communication Services', 'Consumer Discretionary', 'Financials', 'Financials', 'Health Care'],
            'GICS Sub-Industry': ['Technology Hardware', 'Systems Software', 'Semiconductors', 'Interactive Media', 'Broadline Retail', 'Interactive Media', 'Automobile Manufacturers', 'Diversified Banks', 'Transaction & Payment', 'Pharmaceuticals']
        }
        return pd.DataFrame(datos_fallback)

_PEERS_CACHE: dict = {}

@st.cache_data(ttl=86400, show_spinner=False)
def _obtener_peers_cached(
    ticker: str,
    max_peers: int = 3,
    finnhub_api_key: str = "",
    _finnhub_client: Any = None,
) -> List[str]:
    """
    Función interna cacheada para buscar competidores relacionados.
    Parámetro _finnhub_client prefijado con guión bajo para ser ignorado en el hashing de Streamlit.
    """
    ticker = str(ticker or "").upper().strip()
    if not ticker:
        return []

    cache_k = f"{ticker}_{max_peers}"
    if cache_k in _PEERS_CACHE:
        return list(_PEERS_CACHE[cache_k])

    # Prioridad 1: Finnhub API vía finnhub_client.company_peers
    try:
        if _finnhub_client is not None and hasattr(_finnhub_client, "company_peers"):
            peers_fh = _finnhub_client.company_peers(ticker)
        else:
            from data.financial_fetcher import FinnhubClient
            client = FinnhubClient(api_key=finnhub_api_key)
            peers_fh = client.company_peers(ticker)

        if peers_fh and isinstance(peers_fh, list):
            clean_peers = [str(p).upper().strip() for p in peers_fh if p and str(p).upper().strip() != ticker]
            if clean_peers:
                res = clean_peers[:max_peers]
                _PEERS_CACHE[cache_k] = res
                return res
    except Exception:
        pass

    # Prioridad 2: Directorio S&P 1500 (Fallback)
    df_sp = obtener_directorio_sp1500()
    empresa = df_sp[df_sp['Symbol'] == ticker]
    if not empresa.empty:
        sub_ind = empresa['GICS Sub-Industry'].iloc[0]
        sector = empresa['GICS Sector'].iloc[0]
        
        # Prioridad 2.1: Misma Sub-Industria
        peers = df_sp[(df_sp['GICS Sub-Industry'] == sub_ind) & (df_sp['Symbol'] != ticker)]
        
        # Prioridad 2.2: Si no logramos obtener 3, rellenamos con empresas del mismo Sector general
        if len(peers) < max_peers:
            peers_extra = df_sp[(df_sp['GICS Sector'] == sector) & (df_sp['Symbol'] != ticker) & (df_sp['GICS Sub-Industry'] != sub_ind)]
            peers = pd.concat([peers, peers_extra])
            
        return peers['Symbol'].head(max_peers).tolist()
    return []


def obtener_peers(
    ticker: str,
    max_peers: int = 3,
    finnhub_api_key: str = "",
    finnhub_client: Any = None,
    **kwargs: Any,
) -> List[str]:
    """
    Busca competidores relacionados consumiendo finnhub_client.company_peers(symbol) con caché persistente de 24h.
    Soporta argumentos posicionales y nominales ('finnhub_client' o '_finnhub_client').
    Si no hay respuesta de la API o ocurre un error, usa el directorio S&P 1500 como fallback defensivo.
    """
    client = finnhub_client or kwargs.get("_finnhub_client")
    api_key = finnhub_api_key or kwargs.get("api_key", "")
    return _obtener_peers_cached(
        ticker=str(ticker or "").upper().strip(),
        max_peers=max_peers,
        finnhub_api_key=str(api_key or "").strip(),
        _finnhub_client=client,
    )

