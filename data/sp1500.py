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

def obtener_peers(ticker, max_peers=3):
    """Busca competidores relacionados basándose en la Sub-Industria y Sector del S&P 1500"""
    df_sp = obtener_directorio_sp1500()
    empresa = df_sp[df_sp['Symbol'] == ticker]
    if not empresa.empty:
        sub_ind = empresa['GICS Sub-Industry'].iloc[0]
        sector = empresa['GICS Sector'].iloc[0]
        
        # Prioridad 1: Misma Sub-Industria
        peers = df_sp[(df_sp['GICS Sub-Industry'] == sub_ind) & (df_sp['Symbol'] != ticker)]
        
        # Prioridad 2: Si no logramos obtener 3, rellenamos con empresas del mismo Sector general
        if len(peers) < max_peers:
            peers_extra = df_sp[(df_sp['GICS Sector'] == sector) & (df_sp['Symbol'] != ticker) & (df_sp['GICS Sub-Industry'] != sub_ind)]
            peers = pd.concat([peers, peers_extra])
            
        return peers['Symbol'].head(max_peers).tolist()
    return []
