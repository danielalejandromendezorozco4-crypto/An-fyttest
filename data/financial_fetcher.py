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


def extraer_componentes_fcff(
    cf: pd.DataFrame,
    inc: pd.DataFrame,
    bs: pd.DataFrame,
    info: dict,
) -> dict:
    """
    Extrae y sanitiza todos los insumos necesarios para el cálculo de FCFF
    histórico desde los DataFrames de estados financieros.

    Maneja defensivamente:
    - CapEx reportado como negativo (yfinance) o positivo (FMP).
    - Interest Expense reportado con/sin signo.
    - Empresas sin deuda (total_debt = 0).
    - DataFrames vacíos o con datos parciales.

    Args:
        cf:   DataFrame del estado de flujos de efectivo.
        inc:  DataFrame del estado de resultados.
        bs:   DataFrame del balance general.
        info: Diccionario con metadatos y métricas del ticker.

    Returns:
        Diccionario con las siguientes claves:
        - ``ocf_hist``        (list[float]): Operating Cash Flow por período (más reciente primero).
        - ``capex_hist``      (list[float]): CapEx positivo por período.
        - ``interest_hist``   (list[float]): Interest Expense (absoluto) por período.
        - ``pretax_hist``     (list[float]): Pre-tax Income por período.
        - ``taxprov_hist``    (list[float]): Tax Provision por período.
        - ``total_debt``      (float):       Deuda total más reciente.
        - ``total_cash``      (float):       Efectivo + equivalentes más reciente.
        - ``shares_diluted``  (float):       Acciones diluidas en circulación.
        - ``n_periodos``      (int):         Número de períodos históricos con datos completos.
    """
    # ── 1. Operating Cash Flow histórico ────────────────────────────────────
    ocf_hist: list[float] = []
    for fila in ["Operating Cash Flow", "OperatingCashFlow",
                 "Cash Flow From Continuing Operating Activities"]:
        if isinstance(cf, pd.DataFrame) and not cf.empty and fila in cf.index:
            try:
                serie = cf.loc[fila].dropna()
                if not serie.empty:
                    ocf_hist = [safe_num(v, 0.0) for v in serie.values]
                    break
            except Exception:
                pass
    if not ocf_hist:
        # Fallback escalar desde info
        ocf_ttm = safe_num(info.get("operatingCashflow", 0.0), 0.0)
        if ocf_ttm != 0.0:
            ocf_hist = [ocf_ttm]

    # ── 2. CapEx histórico (normalizado como gasto positivo) ────────────────
    capex_hist = obtener_capex_historico(cf)
    if not capex_hist:
        # Inferir desde FCF y OCF escalares
        fcf_ttm = safe_num(info.get("freeCashflow", 0.0), 0.0)
        ocf_ttm = safe_num(info.get("operatingCashflow", 0.0), 0.0)
        if ocf_ttm != 0.0 and fcf_ttm != 0.0:
            capex_hist = [abs(ocf_ttm - fcf_ttm)]
        else:
            capex_hist = [0.0]

    # ── 3. Interest Expense histórico (absoluto) ────────────────────────────
    interest_hist: list[float] = []
    for fila in ["Interest Expense", "InterestExpense",
                 "Interest Expense Non Operating"]:
        if isinstance(inc, pd.DataFrame) and not inc.empty and fila in inc.index:
            try:
                serie = inc.loc[fila].dropna()
                if not serie.empty:
                    interest_hist = [abs(safe_num(v, 0.0)) for v in serie.values]
                    break
            except Exception:
                pass
    if not interest_hist:
        int_scalar = abs(safe_num(info.get("interestExpense", 0.0), 0.0))
        interest_hist = [int_scalar]

    # ── 4. Pre-tax Income histórico ─────────────────────────────────────────
    pretax_hist: list[float] = []
    for fila in ["Pretax Income", "Income Before Tax",
                 "IncomeBeforeTax", "Pretax Income"]:
        if isinstance(inc, pd.DataFrame) and not inc.empty and fila in inc.index:
            try:
                serie = inc.loc[fila].dropna()
                if not serie.empty:
                    pretax_hist = [safe_num(v, 0.0) for v in serie.values]
                    break
            except Exception:
                pass
    if not pretax_hist:
        pretax_hist = [safe_num(info.get("pretaxIncome", 0.0), 0.0)]

    # ── 5. Tax Provision histórica ──────────────────────────────────────────
    taxprov_hist: list[float] = []
    for fila in ["Tax Provision", "IncomeTaxExpense",
                 "Income Tax Expense", "Tax Effect Of Unusual Items"]:
        if isinstance(inc, pd.DataFrame) and not inc.empty and fila in inc.index:
            try:
                serie = inc.loc[fila].dropna()
                if not serie.empty:
                    taxprov_hist = [safe_num(v, 0.0) for v in serie.values]
                    break
            except Exception:
                pass
    if not taxprov_hist:
        taxprov_hist = [safe_num(info.get("taxProvision", 0.0), 0.0)]

    # ── 6. Balance: Deuda Total, Efectivo, Acciones Diluidas ────────────────
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

    # Acciones diluidas: preferir diluted shares sobre basic
    shares_diluted = safe_num(info.get("sharesOutstanding", 0.0), 0.0)
    for fila in ["Diluted Average Shares", "Ordinary Shares Number",
                 "Basic Average Shares"]:
        if isinstance(inc, pd.DataFrame) and not inc.empty and fila in inc.index:
            try:
                val = safe_num(inc.loc[fila].dropna().iloc[0], default=None)
                if val is not None and val > 0:
                    shares_diluted = val
                    break
            except Exception:
                pass

    # ── 7. Alinear longitudes de arrays al mínimo disponible ────────────────
    n_max = max(len(ocf_hist), 1)
    def _pad(lst: list[float], n: int, fill: float = 0.0) -> list[float]:
        """Extiende o recorta una lista a `n` elementos con valor `fill`."""
        if len(lst) >= n:
            return lst[:n]
        return lst + [fill] * (n - len(lst))

    ocf_hist    = _pad(ocf_hist,      n_max)
    capex_hist  = _pad(capex_hist,    n_max)
    interest_hist = _pad(interest_hist, n_max)
    pretax_hist = _pad(pretax_hist,   n_max)
    taxprov_hist = _pad(taxprov_hist, n_max)

    return {
        "ocf_hist":      ocf_hist,
        "capex_hist":    capex_hist,
        "interest_hist": interest_hist,
        "pretax_hist":   pretax_hist,
        "taxprov_hist":  taxprov_hist,
        "total_debt":    total_debt,
        "total_cash":    total_cash,
        "shares_diluted": shares_diluted,
        "n_periodos":    n_max,
    }
