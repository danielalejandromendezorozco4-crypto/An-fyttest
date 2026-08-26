import datetime
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from config.settings import safe_get

def obtener_session_yfinance():
    """
    Crea una sesión de requests con User-Agent de navegador real para mitigar bloqueos HTTP 429/403 en Streamlit Cloud.
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,es;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive"
    })
    return session

def safe_num(val, default=0.0):
    if val is None:
        return float(default)
    try:
        if pd.isna(val) or np.isinf(val):
            return float(default)
        return float(val)
    except (ValueError, TypeError):
        return float(default)

def _extraer_val_df(df, posibles_filas, default=0.0):
    if isinstance(df, pd.DataFrame) and not df.empty:
        for f in posibles_filas:
            if f in df.index:
                try:
                    serie = df.loc[f].dropna()
                    if not serie.empty:
                        val = safe_num(serie.iloc[0], default=None)
                        if val is not None:
                            return val
                except Exception:
                    pass
    return float(default)

@st.cache_data(ttl=3600)
def obtener_tasa_fred(api_key):
    try:
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id=DGS10&api_key={api_key}&file_type=json&sort_order=desc&limit=1"
        session = obtener_session_yfinance()
        res = session.get(url, timeout=5).json()
        return safe_num(res['observations'][0]['value'], default=4.20)
    except Exception:
        return 4.20

@st.cache_data(ttl=3600)
def obtener_erp_mercado(fred_api_key: str = "", rf_actual: float = 4.25) -> float:
    """
    Calcula la Prima de Riesgo de Mercado (ERP) empírica observada a partir de
    series macroeconómicas oficiales (FRED / diferenciales de riesgo soberano y de crédito).
    """
    if fred_api_key:
        try:
            url = f"https://api.stlouisfed.org/fred/series/observations?series_id=BAMLH0A0HYM2&api_key={fred_api_key}&file_type=json&sort_order=desc&limit=1"
            session = obtener_session_yfinance()
            res = session.get(url, timeout=5).json()
            spread_val = safe_num(res['observations'][0]['value'], default=3.50)
            erp_calc = 2.0 + (spread_val * 0.75)
            return round(min(max(erp_calc, 4.50), 6.00), 2)
        except Exception:
            pass
    return 5.00

@st.cache_data(ttl=60)
def fetch_cotizacion_intradia(ticker, fmp_api_key):
    precio_actual = 0.0
    prev_close = 0.0
    hist = pd.DataFrame()
    session = obtener_session_yfinance()
    
    # CAPA 1: yfinance Ticker con sesión personalizada
    try:
        accion = yf.Ticker(ticker, session=session)
        try:
            precio_actual = safe_num(accion.fast_info.last_price, 0.0)
            prev_close = safe_num(accion.fast_info.previous_close, 0.0)
        except Exception:
            pass

        if precio_actual == 0.0:
            info_yf = accion.info or {}
            precio_actual = safe_num(info_yf.get("currentPrice", info_yf.get("regularMarketPrice", 0.0)), 0.0)
            prev_close = safe_num(info_yf.get("previousClose", info_yf.get("regularMarketPreviousClose", 0.0)), 0.0)
        
        hist = accion.history(period="5y")
    except Exception:
        pass

    # CAPA 2: yf.download() como respaldo
    if precio_actual == 0.0 or hist.empty:
        try:
            hist_dl = yf.download(ticker, period="5y", progress=False, session=session)
            if isinstance(hist_dl.columns, pd.MultiIndex):
                hist_dl = hist_dl.xs(ticker, level=1, axis=1) if ticker in hist_dl.columns.levels[1] else hist_dl.droplevel(1, axis=1)
            
            if not hist_dl.empty and 'Close' in hist_dl.columns:
                hist = hist_dl
                close_series = hist['Close'].dropna()
                if precio_actual == 0.0 and len(close_series) > 0:
                    precio_actual = safe_num(close_series.iloc[-1], 0.0)
                if prev_close == 0.0 and len(close_series) > 1:
                    prev_close = safe_num(close_series.iloc[-2], 0.0)
        except Exception:
            pass

    # CAPA 3: Financial Modeling Prep (FMP)
    if precio_actual == 0.0 and fmp_api_key:
        try:
            url_prof = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={fmp_api_key}"
            res_p = session.get(url_prof, timeout=5).json()
            if isinstance(res_p, list) and len(res_p) > 0:
                p_data = res_p[0]
                precio_actual = safe_num(p_data.get("price", 0.0))
                prev_close = precio_actual - safe_num(p_data.get("changes", 0.0))
            
            url_hist = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?seriestype=line&apikey={fmp_api_key}"
            res_hist = session.get(url_hist, timeout=5).json()
            if isinstance(res_hist, dict) and "historical" in res_hist:
                df_h = pd.DataFrame(res_hist["historical"])
                if not df_h.empty:
                    df_h["Date"] = pd.to_datetime(df_h["date"])
                    df_h = df_h.rename(columns={"close": "Close"}).sort_values("Date").set_index("Date")
                    hist = df_h
        except Exception:
            pass
            
    return precio_actual, prev_close, hist

def _fetch_url_json(url):
    try:
        session = obtener_session_yfinance()
        res = session.get(url, timeout=5)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return None

@st.cache_data(ttl=43200)
def fetch_datos_fundamentales(ticker, fmp_api_key):
    info, inc, bs, cf = {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    fmp_exitoso = False
    session = obtener_session_yfinance()
    
    # 1. Intento primario vía FMP concurrente
    if fmp_api_key:
        try:
            url_prof = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={fmp_api_key}"
            url_inc = f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}?limit=5&apikey={fmp_api_key}"
            url_bs = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}?limit=5&apikey={fmp_api_key}"
            url_cf = f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{ticker}?limit=5&apikey={fmp_api_key}"
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                f_p = executor.submit(_fetch_url_json, url_prof)
                f_inc = executor.submit(_fetch_url_json, url_inc)
                f_bs = executor.submit(_fetch_url_json, url_bs)
                f_cf = executor.submit(_fetch_url_json, url_cf)
                
                res_p = f_p.result()
                res_inc = f_inc.result()
                res_bs = f_bs.result()
                res_cf = f_cf.result()
            
            if isinstance(res_inc, list) and len(res_inc) > 0 and isinstance(res_bs, list) and len(res_bs) > 0 and isinstance(res_cf, list) and len(res_cf) > 0:
                p_data = res_p[0] if isinstance(res_p, list) and len(res_p) > 0 else {}
                i_data = res_inc[0]
                b_data = res_bs[0]
                c_data = res_cf[0]
                
                fmp_price = safe_num(p_data.get("price", 1.0))
                fmp_mcap = safe_num(p_data.get("mktCap", 0.0))
                shares_out = fmp_mcap / fmp_price if fmp_price > 0 else safe_num(b_data.get("weightedAverageShsOut", 1.0))
                
                info.update({
                    "longName": p_data.get("companyName", ticker),
                    "sector": p_data.get("sector", "General"),
                    "industry": p_data.get("industry", "General"),
                    "marketCap": fmp_mcap,
                    "beta": safe_num(p_data.get("beta", 1.0)),
                    "dividendRate": safe_num(p_data.get("lastDiv", 0.0)),
                    "sharesOutstanding": shares_out,
                    "totalDebt": safe_num(b_data.get("totalDebt", 0.0)),
                    "totalCash": safe_num(b_data.get("cashAndCashEquivalents", 0.0)),
                    "ebitda": safe_num(i_data.get("ebitda", 0.0)),
                    "operatingIncome": safe_num(i_data.get("operatingIncome", 0.0)),
                    "netIncomeToCommon": safe_num(i_data.get("netIncome", 0.0)),
                    "freeCashflow": safe_num(c_data.get("freeCashFlow", 0.0)),
                    "operatingCashflow": safe_num(c_data.get("operatingCashFlow", 0.0)),
                    "interestExpense": abs(safe_num(i_data.get("interestExpense", 0.0))),
                    "totalAssets": safe_num(b_data.get("totalAssets", 0.0)),
                    "totalStockholderEquity": safe_num(b_data.get("totalStockholdersEquity", 0.0)),
                    "totalCurrentAssets": safe_num(b_data.get("totalCurrentAssets", 0.0)),
                    "totalCurrentLiabilities": safe_num(b_data.get("totalCurrentLiabilities", 0.0)),
                    "currentRatio": (safe_num(b_data.get("totalCurrentAssets")) / safe_num(b_data.get("totalCurrentLiabilities"), 1.0)) if safe_num(b_data.get("totalCurrentLiabilities")) != 0 else 1.0,
                    "debtToEquity": (safe_num(b_data.get("totalDebt")) / safe_num(b_data.get("totalStockholdersEquity"), 1.0)) * 100 if safe_num(b_data.get("totalStockholdersEquity")) > 0 else 0.0,
                    "returnOnEquity": (safe_num(i_data.get("netIncome")) / safe_num(b_data.get("totalStockholdersEquity"), 1.0)) if safe_num(b_data.get("totalStockholdersEquity")) > 0 else 0.0,
                    "returnOnAssets": (safe_num(i_data.get("netIncome")) / safe_num(b_data.get("totalAssets"), 1.0)) if safe_num(b_data.get("totalAssets")) > 0 else 0.0,
                    "operatingMargins": (safe_num(i_data.get("operatingIncome")) / safe_num(i_data.get("revenue"), 1.0)) if safe_num(i_data.get("revenue")) > 0 else 0.0,
                    "pretaxIncome": safe_num(i_data.get("incomeBeforeTax", 0.0)),
                    "taxProvision": safe_num(i_data.get("incomeTaxExpense", 0.0))
                })
                
                df_inc_t = pd.DataFrame(res_inc).set_index("date").T
                inc = df_inc_t.rename(index={
                    "revenue": "Total Revenue", "grossProfit": "Gross Profit", 
                    "operatingIncome": "Operating Income", "netIncome": "Net Income", 
                    "interestExpense": "Interest Expense", "weightedAverageShsOut": "Basic Average Shares"
                })
                df_bs_t = pd.DataFrame(res_bs).set_index("date").T
                bs = df_bs_t.rename(index={
                    "totalAssets": "Total Assets", "totalCurrentLiabilities": "Current Liabilities", 
                    "totalCurrentAssets": "Current Assets", "longTermDebt": "Long Term Debt",
                    "shortTermDebt": "Current Debt"
                })
                df_cf_t = pd.DataFrame(res_cf).set_index("date").T
                cf = df_cf_t.rename(index={
                    "operatingCashFlow": "Operating Cash Flow", "freeCashFlow": "Free Cash Flow"
                })
                fmp_exitoso = True
        except Exception:
            fmp_exitoso = False

    # 2. Extracción vía yfinance con sesión personalizada de navegador
    try:
        accion = yf.Ticker(ticker, session=session)
        info_yf = accion.info or {}
        if isinstance(info_yf, dict) and len(info_yf) > 0:
            for k, v in info_yf.items():
                if k not in info or info[k] is None or info[k] == 0:
                    info[k] = v

        if inc.empty and hasattr(accion, "financials") and accion.financials is not None:
            inc = accion.financials
        if bs.empty and hasattr(accion, "balance_sheet") and accion.balance_sheet is not None:
            bs = accion.balance_sheet
        if cf.empty and hasattr(accion, "cashflow") and accion.cashflow is not None:
            cf = accion.cashflow
    except Exception:
        pass

    # 3. Mapeo y saneamiento de campos numéricos esenciales desde DataFrames si no están en info
    if not info.get("totalAssets") or info.get("totalAssets") == 0:
        info["totalAssets"] = _extraer_val_df(bs, ['Total Assets', 'TotalAssets', 'Total Assets Net'])
        
    if not info.get("totalStockholderEquity") or info.get("totalStockholderEquity") == 0:
        info["totalStockholderEquity"] = _extraer_val_df(bs, ['Total Stockholder Equity', 'Stockholders Equity', 'TotalStockholderEquity', 'Common Stock Equity'])
        
    if not info.get("totalDebt") or info.get("totalDebt") == 0:
        info["totalDebt"] = _extraer_val_df(bs, ['Total Debt', 'TotalDebt', 'Long Term Debt'])
        
    if not info.get("totalCash") or info.get("totalCash") == 0:
        info["totalCash"] = _extraer_val_df(bs, ['Cash And Cash Equivalents', 'CashCashEquivalentsAndShortTermInvestments', 'Cash Financial'])
        
    if not info.get("operatingIncome") or info.get("operatingIncome") == 0:
        info["operatingIncome"] = _extraer_val_df(inc, ['Operating Income', 'OperatingIncome', 'EBIT'])
        
    if not info.get("netIncomeToCommon") or info.get("netIncomeToCommon") == 0:
        info["netIncomeToCommon"] = _extraer_val_df(inc, ['Net Income', 'NetIncome', 'Net Income Common Stockholders'])

    if not info.get("freeCashflow") or info.get("freeCashflow") == 0:
        info["freeCashflow"] = _extraer_val_df(cf, ['Free Cash Flow', 'FreeCashFlow'])
        
    if not info.get("operatingCashflow") or info.get("operatingCashflow") == 0:
        info["operatingCashflow"] = _extraer_val_df(cf, ['Operating Cash Flow', 'OperatingCashFlow', 'Cash Flow From Continuing Operating Activities'])
        
    if not info.get("interestExpense") or info.get("interestExpense") == 0:
        info["interestExpense"] = abs(_extraer_val_df(inc, ['Interest Expense', 'InterestExpense', 'Interest Expense Non Operating']))
        
    if not info.get("ebitda") or info.get("ebitda") == 0:
        info["ebitda"] = _extraer_val_df(inc, ['EBITDA', 'Ebitda']) or (info.get("operatingIncome", 0.0) * 1.15)

    # 4. Asegurar que NINGUNA clave crítica en info sea None o NaN
    claves_criticas = [
        "marketCap", "sharesOutstanding", "totalDebt", "totalCash", "ebitda",
        "operatingIncome", "netIncomeToCommon", "freeCashflow", "operatingCashflow",
        "interestExpense", "totalAssets", "totalStockholderEquity", "totalCurrentAssets",
        "totalCurrentLiabilities", "currentRatio", "debtToEquity", "returnOnEquity",
        "returnOnAssets", "operatingMargins", "pretaxIncome", "taxProvision", "beta"
    ]
    for k in claves_criticas:
        info[k] = safe_num(info.get(k), default=0.0)

    if not info.get("longName"): info["longName"] = ticker
    if not info.get("sector"): info["sector"] = "General"
    if not info.get("industry"): info["industry"] = "General"

    return info, inc, bs, cf

@st.cache_data(ttl=3600)
def obtener_kd_fmp_fred(ticker, fmp_api_key, fred_api_key, int_expense, total_debt_val):
    if total_debt_val > 0 and int_expense > 0:
        return (int_expense / total_debt_val) * 100
    try:
        session = obtener_session_yfinance()
        url_corp = f"https://api.stlouisfed.org/fred/series/observations?series_id=BAMLC0A0CM&api_key={fred_api_key}&file_type=json&sort_order=desc&limit=1"
        res_corp = session.get(url_corp, timeout=4).json()
        return safe_num(res_corp['observations'][0]['value'], default=5.50)
    except Exception:
        return 5.50

@st.cache_data(ttl=3600)
def obtener_datos_dividendos(ticker, info_dict, fmp_api_key, precio_ref):
    div_rate = safe_get(info_dict, ["dividendRate", "trailingAnnualDividendRate", "lastDiv"], 0.0)
    div_rate = safe_num(div_rate, 0.0)
    div_yield = (div_rate / precio_ref * 100) if (div_rate > 0 and precio_ref > 0) else 0.0
    ex_div_ts = safe_get(info_dict, ["exDividendDate"], None)
    if ex_div_ts:
        try:
            next_div_date = datetime.datetime.fromtimestamp(ex_div_ts).strftime('%Y-%m-%d')
        except Exception:
            next_div_date = "N/A"
    else:
        next_div_date = "N/A"
    return div_rate, div_yield, next_div_date

@st.cache_data(ttl=900)
def obtener_noticias_financieras(ticker):
    clean_news = []
    try:
        session = obtener_session_yfinance()
        accion = yf.Ticker(ticker, session=session)
        news = accion.news
        if news and isinstance(news, list):
            for item in news[:8]:
                title = item.get("title", "")
                link = item.get("link", "")
                
                if not title and 'content' in item:
                    title = item['content'].get("title", "")
                    link = item['content'].get("canonicalUrl", {}).get("url", "")
                    
                if not title or not link or link == "#":
                    continue
                    
                pub_time = item.get("providerPublishTime")
                if pub_time and isinstance(pub_time, (int, float)):
                    date_str = datetime.datetime.fromtimestamp(pub_time).strftime("%Y-%m-%d %H:%M")
                else:
                    date_str = "Fecha reciente"
                    
                publisher = item.get("publisher", "Yahoo Finance")
                
                clean_news.append({
                    "title": title,
                    "publisher": publisher,
                    "link": link,
                    "date": date_str
                })
        return clean_news
    except Exception:
        return clean_news

@st.cache_data(ttl=60)
def fetch_datos_concurrente(ticker, fmp_key, fred_key):
    """
    Función concurrente maestra que dispara simultáneamente:
    - Cotización intradía
    - Datos fundamentales históricos
    - Tasa del bono FRED
    - Feed de noticias
    vía ThreadPoolExecutor para optimizar la velocidad de carga.
    """
    with ThreadPoolExecutor(max_workers=4) as executor:
        f_quote = executor.submit(fetch_cotizacion_intradia, ticker, fmp_key)
        f_funda = executor.submit(fetch_datos_fundamentales, ticker, fmp_key)
        f_fred  = executor.submit(obtener_tasa_fred, fred_key)
        f_news  = executor.submit(obtener_noticias_financieras, ticker)
        
        try:
            precio_actual, prev_close, hist = f_quote.result(timeout=12)
        except Exception:
            precio_actual, prev_close, hist = 0.0, 0.0, pd.DataFrame()
            
        try:
            info, inc, bs, cf = f_funda.result(timeout=18)
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
        "news_data": news_data
    }


# ─────────────────────────────────────────────────────────────────────────────
# NUEVAS FUNCIONES PARA EL MOTOR FCFF
# ─────────────────────────────────────────────────────────────────────────────

def obtener_capex_historico(cf: pd.DataFrame) -> list[float]:
    """
    Extrae el Gasto de Capital (CapEx) del estado de flujos de efectivo y lo
    retorna como lista de valores **positivos** (gasto real), ordenados del más
    reciente al más antiguo.

    yfinance reporta CapEx con signo negativo; FMP lo reporta positivo.
    Esta función normaliza ambas convenciones con `abs()`.

    Args:
        cf: DataFrame del estado de flujos de efectivo (índice = conceptos, columnas = fechas).

    Returns:
        Lista de floats con CapEx positivo por período. Lista vacía si no se encuentra.
    """
    posibles_filas = [
        "Capital Expenditure",
        "CapitalExpenditure",
        "Capital Expenditures",
        "Purchase Of Property Plant And Equipment",
        "Purchases Of Property, Plant And Equipment",
        "Capital Expenditure Reported",
    ]
    if isinstance(cf, pd.DataFrame) and not cf.empty:
        for fila in posibles_filas:
            if fila in cf.index:
                try:
                    serie = cf.loc[fila].dropna()
                    if not serie.empty:
                        return [abs(safe_num(v, 0.0)) for v in serie.values]
                except Exception:
                    pass
    return []


@st.cache_data(ttl=300)
def obtener_rf_tnx(fallback_fred: float = 4.20) -> float:
    """
    Obtiene la tasa libre de riesgo (R_f) del bono del Tesoro a 10 años vía
    yfinance (`^TNX`) con fallback automático al valor FRED proporcionado.

    El ticker `^TNX` cotiza en puntos porcentuales (e.g., 4.35 = 4.35%),
    por lo que se retorna directamente como porcentaje.

    Args:
        fallback_fred: Tasa FRED de respaldo a usar si falla la consulta (%).

    Returns:
        Tasa libre de riesgo en porcentaje (e.g., 4.35 representa 4.35%).
    """
    try:
        session = obtener_session_yfinance()
        tnx = yf.Ticker("^TNX", session=session)
        rf_val = safe_num(tnx.fast_info.last_price, default=None)
        if rf_val is not None and 0.5 < rf_val < 20.0:
            return rf_val
    except Exception:
        pass
    return float(fallback_fred)


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACCIÓN FCFF DESAPALANCADO — DUAL-PATH (EBIT primario / OCF fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _extraer_serie(
    df: pd.DataFrame,
    posibles_filas: list[str],
    absval: bool = False,
) -> list[float]:
    """
    Extrae una serie histórica de un DataFrame (índice = conceptos, columnas = fechas)
    buscando secuencialmente los nombres de fila en ``posibles_filas``.

    Args:
        df:             DataFrame de estados financieros.
        posibles_filas: Lista de nombres de fila candidatos (orden de prioridad).
        absval:         Si True aplica ``abs()`` a cada valor extraído.

    Returns:
        Lista de floats ordenados del más reciente al más antiguo.
        Lista vacía si ninguna fila fue encontrada o el DataFrame está vacío.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return []
    for fila in posibles_filas:
        if fila in df.index:
            try:
                serie = df.loc[fila].dropna()
                if not serie.empty:
                    vals = [abs(safe_num(v, 0.0)) if absval else safe_num(v, 0.0)
                            for v in serie.values]
                    return vals
            except Exception:
                pass
    return []


def extraer_fcff_desapalancado(
    cf: pd.DataFrame,
    inc: pd.DataFrame,
    bs: pd.DataFrame,
    info: dict,
) -> dict:
    """
    Extrae y sanitiza los insumos para el cálculo de FCFF histórico siguiendo
    una jerarquía dual-path estricta que elimina el doble conteo de la deuda:

    **Path primario (vía EBIT)** — desapalancado puro:
        ``FCFF = EBIT × (1 − Tax_Rate) + D&A − CapEx − ΔCapital_Trabajo``

    **Path fallback (vía OCF)** — requiere ajuste de escudo fiscal:
        ``FCFF = OCF + Interest_Expense × (1 − Tax_Rate) − CapEx``

    El path primario se utiliza cuando EBIT y D&A están disponibles y producen
    un FCFF positivo. En caso contrario se cae al path OCF. Si ambos fallan,
    se activa la normalización mediante Revenue × Margen Operativo.

    Manejo defensivo:
    - D&A ausente → imputa 0 (conservador; no infla NOPAT).
    - ΔCapital de Trabajo ausente → imputa 0.
    - Arrays de distinta longitud → alineados al largo máximo de OCF.
    - DataFrames vacíos o con datos parciales → escalares desde ``info``.

    Args:
        cf:   DataFrame del estado de flujos de efectivo.
        inc:  DataFrame del estado de resultados.
        bs:   DataFrame del balance general.
        info: Diccionario con metadatos y métricas del ticker.

    Returns:
        Diccionario con las siguientes claves (compatible con
        ``calcular_fcff_valuation``):

        - ``ocf_hist``         (list[float]): Operating Cash Flow histórico.
        - ``capex_hist``       (list[float]): CapEx positivo histórico.
        - ``interest_hist``    (list[float]): Interest Expense (absoluto) histórico.
        - ``pretax_hist``      (list[float]): Pre-tax Income histórico.
        - ``taxprov_hist``     (list[float]): Tax Provision histórica.
        - ``ebit_hist``        (list[float]): EBIT histórico (path primario).
        - ``da_hist``          (list[float]): Depreciación & Amortización histórica.
        - ``delta_nwc_hist``   (list[float]): Variación de Capital de Trabajo histórica.
        - ``total_debt``       (float):       Deuda total más reciente.
        - ``total_cash``       (float):       Efectivo + equivalentes más reciente.
        - ``shares_diluted``   (float):       Acciones diluidas en circulación.
        - ``n_periodos``       (int):         Períodos históricos con datos.
    """
    # ── OCF histórico ────────────────────────────────────────────────────────
    ocf_hist = _extraer_serie(cf, [
        "Operating Cash Flow", "OperatingCashFlow",
        "Cash Flow From Continuing Operating Activities",
    ])
    if not ocf_hist:
        ocf_ttm = safe_num(info.get("operatingCashflow", 0.0), 0.0)
        ocf_hist = [ocf_ttm] if ocf_ttm != 0.0 else [0.0]

    # ── CapEx histórico ──────────────────────────────────────────────────────
    capex_hist = obtener_capex_historico(cf)
    if not capex_hist:
        fcf_ttm = safe_num(info.get("freeCashflow", 0.0), 0.0)
        ocf_ttm = safe_num(info.get("operatingCashflow", 0.0), 0.0)
        capex_hist = [abs(ocf_ttm - fcf_ttm)] if (ocf_ttm != 0.0 and fcf_ttm != 0.0) else [0.0]

    # ── Interest Expense histórico (absoluto) ────────────────────────────────
    interest_hist = _extraer_serie(inc, [
        "Interest Expense", "InterestExpense",
        "Interest Expense Non Operating",
    ], absval=True)
    if not interest_hist:
        interest_hist = [abs(safe_num(info.get("interestExpense", 0.0), 0.0))]

    # ── Pre-tax Income histórico ─────────────────────────────────────────────
    pretax_hist = _extraer_serie(inc, [
        "Pretax Income", "Income Before Tax", "IncomeBeforeTax",
    ])
    if not pretax_hist:
        pretax_hist = [safe_num(info.get("pretaxIncome", 0.0), 0.0)]

    # ── Tax Provision histórica ──────────────────────────────────────────────
    taxprov_hist = _extraer_serie(inc, [
        "Tax Provision", "IncomeTaxExpense", "Income Tax Expense",
    ])
    if not taxprov_hist:
        taxprov_hist = [safe_num(info.get("taxProvision", 0.0), 0.0)]

    # ── EBIT histórico (path primario desapalancado) ──────────────────────────
    ebit_hist = _extraer_serie(inc, [
        "Operating Income", "OperatingIncome", "EBIT",
        "Normalized EBIT", "Normalized Operating Profit",
    ])
    if not ebit_hist:
        ebit_ttm = safe_num(info.get("operatingIncome", 0.0), 0.0)
        ebit_hist = [ebit_ttm] if ebit_ttm != 0.0 else []

    # ── Depreciación y Amortización (D&A) histórica ──────────────────────────
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
    # Si no se encuentra D&A, se imputa 0 al rellenar los arrays más adelante.

    # ── Capital de Trabajo Neto (NWC) histórico ──────────────────────────────
    # NWC = (Activos Corrientes − Efectivo) − (Pasivos Corrientes − Deuda CP)
    # ΔNWC = NWC_t − NWC_{t+1}  (incremento de NWC = uso de efectivo → resta)
    nwc_hist: list[float] = []
    cur_assets_hist = _extraer_serie(bs, [
        "Total Current Assets", "Current Assets", "CurrentAssets",
    ])
    cur_liab_hist = _extraer_serie(bs, [
        "Total Current Liabilities", "Current Liabilities", "CurrentLiabilities",
    ])
    cash_hist_bs = _extraer_serie(bs, [
        "Cash And Cash Equivalents", "CashCashEquivalentsAndShortTermInvestments",
        "Cash Financial", "Cash And Short Term Investments",
    ])
    stdebt_hist = _extraer_serie(bs, [
        "Current Debt", "Current Debt And Capital Lease Obligation", "Short Term Debt",
    ])

    n_nwc = min(
        len(cur_assets_hist) if cur_assets_hist else 0,
        len(cur_liab_hist) if cur_liab_hist else 0,
    )
    for i in range(n_nwc):
        ca    = safe_num(cur_assets_hist[i], 0.0)
        cl    = safe_num(cur_liab_hist[i],  0.0)
        cash_i = safe_num(cash_hist_bs[i],  0.0) if i < len(cash_hist_bs) else 0.0
        std_i  = safe_num(stdebt_hist[i],   0.0) if i < len(stdebt_hist)  else 0.0
        nwc_hist.append((ca - cash_i) - (cl - std_i))

    delta_nwc_hist: list[float] = []
    for i in range(len(nwc_hist) - 1):
        # Positivo → capital de trabajo creció → usa efectivo → resta en FCFF
        delta_nwc_hist.append(nwc_hist[i] - nwc_hist[i + 1])
    if nwc_hist and not delta_nwc_hist:
        delta_nwc_hist = [0.0]

    # ── Balance: Deuda Total, Efectivo, Acciones Diluidas ────────────────────
    total_debt = safe_num(info.get("totalDebt", 0.0), 0.0)
    if total_debt == 0.0:
        total_debt = _extraer_val_df(bs, [
            "Total Debt", "TotalDebt", "Long Term Debt",
            "Long Term Debt And Capital Lease Obligation",
        ], default=0.0)

    total_cash = safe_num(info.get("totalCash", 0.0), 0.0)
    if total_cash == 0.0:
        total_cash = _extraer_val_df(bs, [
            "Cash And Cash Equivalents",
            "CashCashEquivalentsAndShortTermInvestments",
            "Cash Financial", "Cash And Short Term Investments",
        ], default=0.0)

    shares_diluted = safe_num(info.get("sharesOutstanding", 0.0), 0.0)
    for fila in ["Diluted Average Shares", "Ordinary Shares Number", "Basic Average Shares"]:
        if isinstance(inc, pd.DataFrame) and not inc.empty and fila in inc.index:
            try:
                val = safe_num(inc.loc[fila].dropna().iloc[0], default=None)
                if val is not None and val > 0:
                    shares_diluted = val
                    break
            except Exception:
                pass

    # ── Alineación de arrays al largo de OCF (n_max) ─────────────────────────
    n_max = max(len(ocf_hist), 1)

    def _pad(lst: list[float], n: int, fill: float = 0.0) -> list[float]:
        """Extiende o recorta una lista a ``n`` elementos usando ``fill``."""
        if len(lst) >= n:
            return lst[:n]
        return lst + [fill] * (n - len(lst))

    return {
        "ocf_hist":        _pad(ocf_hist,        n_max),
        "capex_hist":      _pad(capex_hist,      n_max),
        "interest_hist":   _pad(interest_hist,   n_max),
        "pretax_hist":     _pad(pretax_hist,     n_max),
        "taxprov_hist":    _pad(taxprov_hist,    n_max),
        "ebit_hist":       _pad(ebit_hist,       n_max),
        "da_hist":         _pad(da_hist,         n_max),
        "delta_nwc_hist":  _pad(delta_nwc_hist,  n_max),
        "total_debt":      total_debt,
        "total_cash":      total_cash,
        "shares_diluted":  shares_diluted,
        "n_periodos":      n_max,
    }


def extraer_componentes_fcff(
    cf: pd.DataFrame,
    inc: pd.DataFrame,
    bs: pd.DataFrame,
    info: dict,
) -> dict:
    """
    Extrae y sanitiza todos los insumos necesarios para el cálculo de FCFF
    histórico desde los DataFrames de estados financieros con jerarquía dual-path.

    Delega en ``extraer_fcff_desapalancado`` garantizando compatibilidad total.
    """
    return extraer_fcff_desapalancado(cf, inc, bs, info)



def extraer_metricas_ttm(
    info: dict,
    inc: pd.DataFrame,
    bs: pd.DataFrame,
    cf: pd.DataFrame,
    precio_actual: float = 0.0,
) -> dict:
    """
    Extrae, normaliza y consolida todas las métricas y cifras de los estados financieros
    en base a los últimos 12 meses (TTM) y balance consolidado más reciente, alineado
    a la metodología institucional (Finviz, Yahoo Finance, Investing.com).

    Garantiza:
    - FCF TTM = OCF TTM - |CapEx TTM| (sin descalces de signos).
    - Acciones diluidas consolidadas para evitar distorsiones con tickers multiclase.
    - Manejo seguro de campos nulos o no disponibles.
    """
    # ── 1. Acciones y Market Cap ─────────────────────────────────────────────
    shares_diluted = safe_num(info.get("impliedSharesOutstanding", 0.0), 0.0)
    if shares_diluted <= 0:
        shares_diluted = _extraer_val_df(inc, [
            "Diluted Average Shares", "Ordinary Shares Number", "Basic Average Shares"
        ], default=0.0)
    if shares_diluted <= 0:
        shares_diluted = safe_num(info.get("sharesOutstanding", 0.0), 0.0)
    if shares_diluted <= 0:
        shares_diluted = safe_num(info.get("floatShares", 0.0), 0.0)

    mcap = safe_num(info.get("marketCap", 0.0), 0.0)
    if mcap <= 0 and precio_actual > 0 and shares_diluted > 0:
        mcap = shares_diluted * precio_actual
    if shares_diluted <= 0 and mcap > 0 and precio_actual > 0:
        shares_diluted = mcap / precio_actual

    # ── 2. Estado de Resultados TTM ──────────────────────────────────────────
    revenue_ttm = safe_num(info.get("totalRevenue", 0.0), 0.0)
    if revenue_ttm <= 0:
        revenue_ttm = _extraer_val_df(inc, ["Total Revenue", "TotalRevenue", "Revenue"], default=0.0)

    gross_profit_ttm = safe_num(info.get("grossProfits", 0.0), 0.0)
    if gross_profit_ttm <= 0:
        gross_profit_ttm = _extraer_val_df(inc, ["Gross Profit", "GrossProfit"], default=0.0)

    operating_income_ttm = safe_num(info.get("operatingIncome", 0.0), 0.0)
    if operating_income_ttm == 0.0:
        operating_income_ttm = _extraer_val_df(inc, ["Operating Income", "OperatingIncome", "EBIT"], default=0.0)

    ebitda_ttm = safe_num(info.get("ebitda", 0.0), 0.0)
    if ebitda_ttm <= 0:
        ebitda_ttm = _extraer_val_df(inc, ["EBITDA", "Normalized EBITDA"], default=0.0)
    if ebitda_ttm <= 0 and operating_income_ttm > 0:
        ebitda_ttm = operating_income_ttm * 1.15

    net_income_ttm = safe_num(info.get("netIncomeToCommon", 0.0), 0.0)
    if net_income_ttm == 0.0:
        net_income_ttm = safe_num(info.get("netIncome", 0.0), 0.0)
    if net_income_ttm == 0.0:
        net_income_ttm = _extraer_val_df(inc, [
            "Net Income Common Stockholders", "Net Income", "NetIncome"
        ], default=0.0)

    eps_diluted_ttm = safe_num(info.get("trailingEps", 0.0), 0.0)
    if eps_diluted_ttm == 0.0:
        eps_diluted_ttm = safe_num(info.get("epsTrailingTwelveMonths", 0.0), 0.0)
    if eps_diluted_ttm == 0.0 and shares_diluted > 0 and net_income_ttm != 0.0:
        eps_diluted_ttm = net_income_ttm / shares_diluted
    if eps_diluted_ttm == 0.0:
        eps_diluted_ttm = _extraer_val_df(inc, [
            "Diluted EPS", "DilutedEPS", "Basic EPS", "BasicEPS",
            "Diluted EPS from Continuing Operations", "EPS"
        ], default=0.0)

    forward_eps = safe_num(info.get("forwardEps", 0.0), 0.0)
    if forward_eps == 0.0:
        forward_eps = safe_num(info.get("epsForward", 0.0), 0.0)

    pretax_income_ttm = safe_num(info.get("pretaxIncome", 0.0), 0.0)
    if pretax_income_ttm == 0.0:
        pretax_income_ttm = _extraer_val_df(inc, ["Pretax Income", "Income Before Tax", "IncomeBeforeTax"], default=0.0)

    tax_provision_ttm = safe_num(info.get("taxProvision", 0.0), 0.0)
    if tax_provision_ttm == 0.0:
        tax_provision_ttm = _extraer_val_df(inc, ["Tax Provision", "IncomeTaxExpense", "Income Tax Expense"], default=0.0)

    interest_expense_ttm = abs(safe_num(info.get("interestExpense", 0.0), 0.0))
    if interest_expense_ttm == 0.0:
        interest_expense_ttm = abs(_extraer_val_df(inc, ["Interest Expense", "InterestExpense"], default=0.0))

    # ── 3. Flujo de Caja TTM ─────────────────────────────────────────────────
    ocf_ttm = safe_num(info.get("operatingCashflow", 0.0), 0.0)
    if ocf_ttm == 0.0:
        ocf_ttm = _extraer_val_df(cf, [
            "Operating Cash Flow", "OperatingCashFlow",
            "Cash Flow From Continuing Operating Activities"
        ], default=0.0)

    capex_list = obtener_capex_historico(cf)
    capex_ttm = capex_list[0] if capex_list else 0.0
    if capex_ttm == 0.0:
        capex_ttm = abs(safe_num(info.get("capitalExpenditures", 0.0), 0.0))

    # FCF TTM = OCF - CapEx (estrictamente normalizado)
    fcf_ttm = ocf_ttm - capex_ttm
    if fcf_ttm == 0.0 and ocf_ttm == 0.0:
        fcf_ttm = safe_num(info.get("freeCashflow", 0.0), 0.0)

    # ── 4. Balance Consolidado ───────────────────────────────────────────────
    total_debt = safe_num(info.get("totalDebt", 0.0), 0.0)
    if total_debt == 0.0:
        total_debt = _extraer_val_df(bs, [
            "Total Debt", "TotalDebt", "Long Term Debt",
            "Long Term Debt And Capital Lease Obligation"
        ], default=0.0)

    total_cash = safe_num(info.get("totalCash", 0.0), 0.0)
    if total_cash == 0.0:
        total_cash = _extraer_val_df(bs, [
            "Cash And Cash Equivalents",
            "CashCashEquivalentsAndShortTermInvestments",
            "Cash Financial", "Cash And Short Term Investments"
        ], default=0.0)

    total_equity = safe_num(info.get("totalStockholderEquity", 0.0), 0.0)
    if total_equity == 0.0:
        total_equity = _extraer_val_df(bs, [
            "Total Stockholder Equity", "Stockholders Equity",
            "TotalStockholderEquity", "Common Stock Equity"
        ], default=0.0)

    total_assets = safe_num(info.get("totalAssets", 0.0), 0.0)
    if total_assets == 0.0:
        total_assets = _extraer_val_df(bs, ["Total Assets", "TotalAssets", "Total Assets Net"], default=0.0)

    current_assets = safe_num(info.get("totalCurrentAssets", 0.0), 0.0)
    if current_assets == 0.0:
        current_assets = _extraer_val_df(bs, ["Total Current Assets", "Current Assets", "CurrentAssets"], default=0.0)

    current_liabilities = safe_num(info.get("totalCurrentLiabilities", 0.0), 0.0)
    if current_liabilities == 0.0:
        current_liabilities = _extraer_val_df(bs, ["Total Current Liabilities", "Current Liabilities", "CurrentLiabilities"], default=0.0)

    short_term_debt = _extraer_val_df(bs, [
        "Current Debt", "Current Debt And Capital Lease Obligation", "Short Term Debt"
    ], default=0.0)

    # ── 6. CAGR Histórico de Ingresos (3-5 años) y Margen Operativo Medio ───
    cagr_revenue_3_5y = 0.0
    op_margin_hist = 0.0
    if isinstance(inc, pd.DataFrame) and not inc.empty:
        for fila_rev in ["Total Revenue", "TotalRevenue", "Revenue"]:
            if fila_rev in inc.index:
                try:
                    s_rev = inc.loc[fila_rev].dropna()
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

        for fila_op in ["Operating Income", "OperatingIncome", "EBIT"]:
            if fila_op in inc.index:
                try:
                    s_op = inc.loc[fila_op].dropna()
                    for fila_rev in ["Total Revenue", "TotalRevenue", "Revenue"]:
                        if fila_rev in inc.index:
                            s_rev = inc.loc[fila_rev].dropna()
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

    # ── 7. Crecimiento de Ganancias, Ingresos y Consenso ────────────────────
    earnings_growth = safe_num(info.get("earningsGrowth", 0.0), 0.0)
    if earnings_growth == 0.0 and isinstance(inc, pd.DataFrame) and not inc.empty:
        for fila_ni in ["Net Income", "NetIncome", "Net Income Common Stockholders"]:
            if fila_ni in inc.index:
                try:
                    s_ni = inc.loc[fila_ni].dropna()
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
