import datetime
import requests
import pandas as pd
import yfinance as yf
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from config.settings import safe_get

@st.cache_data(ttl=3600)
def obtener_tasa_fred(api_key):
    try:
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id=DGS10&api_key={api_key}&file_type=json&sort_order=desc&limit=1"
        res = requests.get(url, timeout=4).json()
        return float(res['observations'][0]['value'])
    except Exception:
        return 4.20

@st.cache_data(ttl=60)
def fetch_cotizacion_intradia(ticker, fmp_api_key):
    precio_actual = 0.0
    prev_close = 0.0
    hist = pd.DataFrame()
    
    # CAPA 1: Extracción ligera vía fast_info
    try:
        accion = yf.Ticker(ticker)
        try:
            precio_actual = float(accion.fast_info.last_price)
            prev_close = float(accion.fast_info.previous_close)
        except Exception:
            pass

        if precio_actual == 0.0:
            info_yf = accion.info or {}
            precio_actual = float(info_yf.get("currentPrice", info_yf.get("regularMarketPrice", 0.0)))
            prev_close = float(info_yf.get("previousClose", info_yf.get("regularMarketPreviousClose", 0.0)))
        
        hist = accion.history(period="5y")
    except Exception:
        pass

    # CAPA 2: Respaldo directo mediante yf.download()
    if precio_actual == 0.0 or hist.empty:
        try:
            hist_dl = yf.download(ticker, period="5y", progress=False)
            if isinstance(hist_dl.columns, pd.MultiIndex):
                hist_dl = hist_dl.xs(ticker, level=1, axis=1) if ticker in hist_dl.columns.levels[1] else hist_dl.droplevel(1, axis=1)
            
            if not hist_dl.empty and 'Close' in hist_dl.columns:
                hist = hist_dl
                close_series = hist['Close'].dropna()
                if precio_actual == 0.0 and len(close_series) > 0:
                    precio_actual = float(close_series.iloc[-1])
                if prev_close == 0.0 and len(close_series) > 1:
                    precio_actual = float(close_series.iloc[-2])
        except Exception:
            pass

    # CAPA 3: Respaldo vía Financial Modeling Prep (FMP)
    if precio_actual == 0.0 and fmp_api_key:
        try:
            url_prof = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={fmp_api_key}"
            res_p = requests.get(url_prof, timeout=4).json()
            if isinstance(res_p, list) and len(res_p) > 0:
                p_data = res_p[0]
                precio_actual = float(p_data.get("price", 0.0))
                prev_close = precio_actual - float(p_data.get("changes", 0.0))
            
            url_hist = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?seriestype=line&apikey={fmp_api_key}"
            res_hist = requests.get(url_hist, timeout=4).json()
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
        res = requests.get(url, timeout=4)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return None

@st.cache_data(ttl=43200)
def fetch_datos_fundamentales(ticker, fmp_api_key):
    info, inc, bs, cf = {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    fmp_exitoso = False
    
    # Intentar vía FMP concurrente si la API Key está activa
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
                fmp_price = float(p_data.get("price", 1.0))
                fmp_mcap = float(p_data.get("mktCap", 0.0))
                shares_out = fmp_mcap / fmp_price if fmp_price > 0 else float(b_data.get("weightedAverageShsOut", 1.0))
                info.update({
                    "longName": p_data.get("companyName", ticker),
                    "sector": p_data.get("sector", "General"),
                    "industry": p_data.get("industry", "General"),
                    "marketCap": fmp_mcap,
                    "beta": float(p_data.get("beta", info.get("beta", 1.0))),
                    "dividendRate": float(p_data.get("lastDiv", 0.0)),
                    "sharesOutstanding": shares_out,
                    "totalDebt": float(b_data.get("totalDebt", 0.0)),
                    "totalCash": float(b_data.get("cashAndCashEquivalents", 0.0)),
                    "ebitda": float(i_data.get("ebitda", 1.0)),
                    "operatingIncome": float(i_data.get("operatingIncome", 0.0)),
                    "netIncomeToCommon": float(i_data.get("netIncome", 0.0)),
                    "freeCashflow": float(c_data.get("freeCashFlow", 0.0)),
                    "interestExpense": abs(float(i_data.get("interestExpense", 0.0))),
                    "totalAssets": float(b_data.get("totalAssets", 1.0)),
                    "totalStockholderEquity": float(b_data.get("totalStockholdersEquity", 1.0)),
                    "currentRatio": (float(b_data.get("totalCurrentAssets", 0.0)) / float(b_data.get("totalCurrentLiabilities", 1.0))) if float(b_data.get("totalCurrentLiabilities", 1.0)) != 0 else 1.0,
                    "debtToEquity": (float(b_data.get("totalDebt", 0.0)) / float(b_data.get("totalStockholdersEquity", 1.0))) * 100 if float(b_data.get("totalStockholdersEquity", 1.0)) > 0 else 0.0,
                    "returnOnEquity": (float(i_data.get("netIncome", 0.0)) / float(b_data.get("totalStockholdersEquity", 1.0))) if float(b_data.get("totalStockholdersEquity", 1.0)) > 0 else 0.0,
                    "returnOnAssets": (float(i_data.get("netIncome", 0.0)) / float(b_data.get("totalAssets", 1.0))) if float(b_data.get("totalAssets", 1.0)) > 0 else 0.0,
                    "operatingMargins": (float(i_data.get("operatingIncome", 0.0)) / float(i_data.get("revenue", 1.0))) if float(i_data.get("revenue", 1.0)) > 0 else 0.0,
                    "pretaxIncome": float(i_data.get("incomeBeforeTax", 1.0)),
                    "taxProvision": float(i_data.get("incomeTaxExpense", 0.0))
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

    # Fallback con yfinance
    if not fmp_exitoso:
        try:
            accion = yf.Ticker(ticker)
            info_fall = accion.info or {}
            info.update(info_fall)
            
            # Extraer de forma garantizada marketCap y sharesOutstanding
            if not info.get("marketCap"):
                mcap_y = safe_get(info_fall, ["marketCap", "regularMarketMarketCap", "enterpriseValue"], 0.0)
                if mcap_y > 0: info["marketCap"] = float(mcap_y)
                
            if not info.get("sharesOutstanding"):
                sh_y = safe_get(info_fall, ["sharesOutstanding", "impliedSharesOutstanding", "floatShares"], 0.0)
                if sh_y > 0: info["sharesOutstanding"] = float(sh_y)
            
            if not info.get("longName"): info["longName"] = ticker
            if not info.get("sector"): info["sector"] = "General"
            if not info.get("industry"): info["industry"] = "General"
            
            inc = accion.financials if hasattr(accion, "financials") and accion.financials is not None else pd.DataFrame()
            bs = accion.balance_sheet if hasattr(accion, "balance_sheet") and accion.balance_sheet is not None else pd.DataFrame()
            cf = accion.cashflow if hasattr(accion, "cashflow") and accion.cashflow is not None else pd.DataFrame()
        except Exception:
            pass

    return info, inc, bs, cf

@st.cache_data(ttl=3600)
def obtener_kd_fmp_fred(ticker, fmp_api_key, fred_api_key, int_expense, total_debt_val):
    if total_debt_val > 0 and int_expense > 0:
        return (int_expense / total_debt_val) * 100
    try:
        url_corp = f"https://api.stlouisfed.org/fred/series/observations?series_id=BAMLC0A0CM&api_key={fred_api_key}&file_type=json&sort_order=desc&limit=1"
        res_corp = requests.get(url_corp, timeout=4).json()
        return float(res_corp['observations'][0]['value'])
    except Exception:
        return 5.50

@st.cache_data(ttl=3600)
def obtener_datos_dividendos(ticker, info_dict, fmp_api_key, precio_ref):
    div_rate = safe_get(info_dict, ["dividendRate", "trailingAnnualDividendRate", "lastDiv"], 0.0)
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
        accion = yf.Ticker(ticker)
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
            precio_actual, prev_close, hist = f_quote.result(timeout=10)
        except Exception:
            precio_actual, prev_close, hist = 0.0, 0.0, pd.DataFrame()
            
        try:
            info, inc, bs, cf = f_funda.result(timeout=15)
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
