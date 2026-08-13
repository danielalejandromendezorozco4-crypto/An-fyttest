import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Custom module imports
from config.settings import (
    SECTOR_BENCHMARKS,
    cargar_secrets,
    obtener_ruta_logo,
    safe_get,
)
from config.styles import inyectar_estilos
from data.sp1500 import obtener_directorio_sp1500, obtener_peers
from data.financial_fetcher import (
    fetch_datos_concurrente,
    obtener_datos_dividendos,
    obtener_noticias_financieras,
    obtener_tasa_fred,
)
from engine.valuation import (
    calcular_ddm,
    calcular_wacc,
    crear_calculador_dcf,
)
from engine.metrics import (
    calcular_altman_zscore,
    calcular_piotroski_fscore,
    calcular_scoring,
    evaluar_veredicto,
)
from services.ai_service import (
    obtener_analisis_macro_ia,
    obtener_perfil_corporativo,
)
from reports.pdf_generator import generar_pdf_reporte

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="An-FyT - Análisis Fundamental Top-Down", layout="wide", page_icon="📈")

# --- MANEJO DE ESTADO GLOBAL PARA DIRECTORIO SECTORIAL ---
if "ticker_search" not in st.session_state:
    st.session_state["ticker_search"] = ""

def seleccionar_ticker(t):
    st.session_state["ticker_search"] = t

# --- INYECCIÓN DE CSS PERSONALIZADO ---
inyectar_estilos()

# --- AUTENTICACIÓN SECRETS ---
gemini_key, fred_key, fmp_key = cargar_secrets()

# --- BÚSQUEDA DE LOGO Y RENDERIZADO SIDEBAR ---
ruta_logo_detectada = obtener_ruta_logo()
if ruta_logo_detectada:
    st.sidebar.image(ruta_logo_detectada, use_container_width=True)
else:
    st.sidebar.markdown('<div class="brand-title">An-FyT</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="brand-subtitle">ANÁLISIS FUNDAMENTAL BURSÁTIL</div>', unsafe_allow_html=True)
st.sidebar.markdown("---")

# Tooltip Implementado en el Text Input
ticker_input = st.sidebar.text_input(
    "📌 Ticker (ej. MSFT, NVDA, AAPL, MA, KO)", 
    key="ticker_search",
    help="Un Ticker es el código corto con el que cotiza una empresa en bolsa. Ejemplo: AAPL (Apple), MSFT (Microsoft), NVDA (NVIDIA), KO (Coca-Cola)."
).upper().strip()

col_btn1, col_btn2 = st.sidebar.columns([3, 1])
with col_btn1:
    analizar_btn = st.button("🚀 Iniciar Evaluación", use_container_width=True)
with col_btn2:
    limpiar_btn = st.button("🧹", help="Limpiar búsqueda actual", use_container_width=True)

if limpiar_btn:
    st.session_state["ticker_search"] = ""
    st.rerun()

# INYECCIÓN DE TICKERS RELACIONADOS (PEERS)
if ticker_input:
    peers_list = obtener_peers(ticker_input, max_peers=3)
    if peers_list:
        st.sidebar.caption("🏢 Empresas relacionadas:")
        cols_peers = st.sidebar.columns(len(peers_list))
        for i, peer in enumerate(peers_list):
            with cols_peers[i]:
                st.button(peer, key=f"btn_peer_{peer}", on_click=seleccionar_ticker, args=(peer,), use_container_width=True)

# --- FUNCIÓN DE RENDERIZADO DEL DIRECTORIO SECTORIAL ---
def render_directorio():
    df_sp = obtener_directorio_sp1500()
    
    st.markdown('<h3 style="color: #0A192F;">🗂️ Explorador de Mercado y Directorio Sectorial (Universo S&P 1500)</h3>', unsafe_allow_html=True)
    st.write("Selecciona un sector y sub-industria para descubrir empresas líderes (Large, Mid y Small Caps) y evalúalas con un solo clic.")
    
    sectores = sorted(df_sp['GICS Sector'].dropna().unique())
    sec_col, sub_col = st.columns(2)
    
    with sec_col:
        sel_sector = st.selectbox("1️⃣ Filtra por Sector GICS:", ["Selecciona un Sector..."] + list(sectores))
    
    if sel_sector != "Selecciona un Sector...":
        df_sec = df_sp[df_sp['GICS Sector'] == sel_sector]
        sub_inds = sorted(df_sec['GICS Sub-Industry'].dropna().unique())
        
        with sub_col:
            sel_sub = st.selectbox("2️⃣ Filtra por Sub-Industria:", ["Todas las Sub-Industrias..."] + list(sub_inds))
        
        if sel_sub != "Todas las Sub-Industrias...":
            df_final = df_sec[df_sec['GICS Sub-Industry'] == sel_sub]
        else:
            df_final = df_sec
            
        st.markdown(f"**Empresas encontradas:** {len(df_final)}")
        st.markdown("<br>", unsafe_allow_html=True)
        
        columnas_grid = st.columns(4)
        for idx, row in enumerate(df_final.itertuples()):
            with columnas_grid[idx % 4]:
                st.button(
                    label=f"[{row.Symbol}] {str(row.Security)[:20]}", 
                    key=f"btn_dir_{row.Symbol}_{idx}", 
                    on_click=seleccionar_ticker, 
                    args=(row.Symbol,),
                    use_container_width=True
                )

tasa_libre_riesgo = obtener_tasa_fred(fred_key)

# --- FLUJO PRINCIPAL DE RENDERIZADO BIFURCADO ---
if not ticker_input:
    # 1. PANTALLA DE BIENVENIDA
    col_h1, col_h2 = st.columns([1.5, 4])
    with col_h1:
        if ruta_logo_detectada:
            st.image(ruta_logo_detectada, width=220)
        else:
            st.markdown('<div class="brand-title">An-FyT</div>', unsafe_allow_html=True)
            
    with col_h2:
        st.markdown('<h1 style="color: #0A192F; margin-bottom: 5px; font-size: 32px;">An-FyT - Sistema de Análisis Fundamental Top-Down</h1>', unsafe_allow_html=True)
        st.markdown('<p style="color: #475569; font-size: 15px; margin-bottom: 20px;">Evaluación cuantitativa e institucional de empresas cotizadas en bolsa mediante 100 Puntos de Scoring.</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🎯 Dimensiones Estratégicas de Evaluación (100 Puntos)")
    col_p1, col_p2, col_p3, col_p4, col_p5 = st.columns(5)
    
    with col_p1:
        st.markdown('<div class="pillar-card"><h4 style="color: #0284C7; margin-top:0;">🏦 Solvencia (15%)</h4><p style="color: #334155; font-size: 12px; line-height: 1.5;">Auditoría de deuda neta, EBITDA, razón corriente, cobertura de FCF y caja neta por acción.</p></div>', unsafe_allow_html=True)
    with col_p2:
        st.markdown('<div class="pillar-card"><h4 style="color: #059669; margin-top:0;">📈 Rentabilidad (25%)</h4><p style="color: #334155; font-size: 12px; line-height: 1.5;">Retorno sobre capital (ROIC, ROE, ROA), Buyback Yield y FCF Yield frente a Tasa FRED.</p></div>', unsafe_allow_html=True)
    with col_p3:
        st.markdown('<div class="pillar-card"><h4 style="color: #B45309; margin-top:0;">🏷️ Valuación (40%)</h4><p style="color: #334155; font-size: 12px; line-height: 1.5;">Modelo DCF por acción con WACC dinámico, DDM Gordon Growth y múltiplos de mercado.</p></div>', unsafe_allow_html=True)
    with col_p4:
        st.markdown('<div class="pillar-card"><h4 style="color: #D97706; margin-top:0;">🛡️ Riesgos (15%)</h4><p style="color: #334155; font-size: 12px; line-height: 1.5;">Riesgo de quiebra (Altman Z-Score), volatilidad (Beta) y auditoría contable Piotroski F-Score.</p></div>', unsafe_allow_html=True)
    with col_p5:
        st.markdown('<div class="pillar-card"><h4 style="color: #7E22CE; margin-top:0;">🌍 Macro (5%)</h4><p style="color: #334155; font-size: 12px; line-height: 1.5;">Análisis de entorno macroeconómico, tasas e inflación impulsado por Inteligencia Artificial.</p></div>', unsafe_allow_html=True)
    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
    st.markdown("### 🚥 Código de Semáforos e Indicadores")
    
    col_sem1, col_sem2, col_sem3 = st.columns(3)
    with col_sem1:
        st.markdown('<div style="background: #ECFDF5; padding: 16px; border-radius: 8px; border: 1px solid #A7F3D0;"><b style="color: #047857; font-size: 16px;">🟢 Verde (Saludable / Atractivo)</b><p style="color: #065F46; font-size: 13px; margin-top: 6px; margin-bottom: 0;">Métrica sobresaliente o cotización por debajo del precio máximo de compra sugerido.</p></div>', unsafe_allow_html=True)
    with col_sem2:
        st.markdown('<div style="background: #FFFBEB; padding: 16px; border-radius: 8px; border: 1px solid #FDE68A;"><b style="color: #B45309; font-size: 16px;">🟡 Amarillo (Aceptable / Precaución)</b><p style="color: #92400E; font-size: 13px; margin-top: 6px; margin-bottom: 0;">Métrica dentro del promedio sectorial o liquidez moderada que requiere monitoreo.</p></div>', unsafe_allow_html=True)
    with col_sem3:
        st.markdown('<div style="background: #FEF2F2; padding: 16px; border-radius: 8px; border: 1px solid #FCA5A5;"><b style="color: #B91C1C; font-size: 16px;">🔴 Rojo (Riesgo / Exigente)</b><p style="color: #991B1B; font-size: 13px; margin-top: 6px; margin-bottom: 0;">Alerta de endeudamiento elevado, destrucción de valor operativo o sobrevaloración sobre flujos.</p></div>', unsafe_allow_html=True)
    st.markdown("---")
    render_directorio()

# 2. SISTEMA DE EVALUACIÓN (CUANDO HAY UN TICKER ACTIVO)
else:
    st.markdown('<h1 style="color: #0A192F;">📈 Sistema Avanzado de Análisis Fundamental Top-Down</h1>', unsafe_allow_html=True)
    with st.spinner(f"📡 Compilando métricas y evaluando macroeconomía para {ticker_input}..."):
        try:
            # Extracción concurrente optimizada
            datos_mercado = fetch_datos_concurrente(ticker_input, fmp_key, fred_key)
            precio_actual = datos_mercado["precio_actual"]
            prev_close = datos_mercado["prev_close"]
            hist = datos_mercado["hist"]
            info_estatica = datos_mercado["info"]
            inc = datos_mercado["inc"]
            bs = datos_mercado["bs"]
            cf = datos_mercado["cf"]
            tasa_libre_riesgo = datos_mercado["tasa_fred"]
            news_data_cache = datos_mercado["news_data"]
            
            info = info_estatica.copy()
            info["currentPrice"] = precio_actual
            info["previousClose"] = prev_close
            
            if not info or (hist.empty and inc.empty) or precio_actual == 0:
                st.error("❌ Ticker no encontrado o problemas de conexión con el proveedor financiero. Verifica el símbolo ingresado.")
                st.stop()

            # --- MÓDULO 1: SOLVENCIA Y DEUDA (15%) ---
            mcap = safe_get(info, ["marketCap", "mktCap", "regularMarketMarketCap"], 0.0)
            shares_current = safe_get(info, ["sharesOutstanding", "impliedSharesOutstanding", "floatShares"], 0.0)

            if mcap <= 0 and shares_current > 0 and precio_actual > 0:
                mcap = shares_current * precio_actual

            if mcap <= 0 and not inc.empty and 'Basic Average Shares' in inc.index:
                try:
                    sh_val = inc.loc['Basic Average Shares'].iloc[0]
                    if not pd.isna(sh_val) and float(sh_val) > 0:
                        shares_current = float(sh_val)
                        mcap = shares_current * precio_actual
                except Exception:
                    pass

            if mcap <= 0 and precio_actual > 0:
                mcap = 10_000_000_000.0 # Fallback neutro 10B USD para evitar castigos Small Cap falsos en Mega Caps

            if shares_current <= 0:
                shares_current = (mcap / precio_actual) if (mcap > 0 and precio_actual > 0) else 1.0

            # EVALUACIÓN DE INTEGRIDAD DE DATOS (Solo marca incompleto si faltan insumos críticos de valuación)
            net_income_val = safe_get(info, ["netIncomeToCommon", "netIncome"], 0.0)
            op_cash_val = safe_get(info, ["operatingCashflow", "freeCashflow"], 0.0)

            datos_completos = True
            if mcap <= 0 or (net_income_val == 0.0 and op_cash_val == 0.0 and (inc.empty and bs.empty)):
                datos_completos = False

            nombre = info.get("longName", ticker_input)
            sector = info.get("sector", "General")
            industria = info.get("industry", "General")
            
            st.markdown(f"<h2 style='color: #0A192F; font-weight: 700; margin-bottom: 2px;'>🏢 {nombre} ({ticker_input})</h2>", unsafe_allow_html=True)
            
            pct_change = ((precio_actual - prev_close) / prev_close * 100) if prev_close else 0.0
            if pct_change >= 0:
                clase_pct = "positivo"
                texto_valia = "Plusvalía al día de hoy"
            else:
                clase_pct = "negativo"
                texto_valia = "Minusvalía al día de hoy"
            
            st.markdown(f"### Cotización: ${precio_actual:,.2f} | <span class='badge-plusvalia {clase_pct}' style='font-size: 0.85em; font-weight: 700;'>{pct_change:.2f}% ({texto_valia})</span>", unsafe_allow_html=True)
            st.caption(f"🏷️ Sector: {sector} | 🏭 Industria: {industria}")
            
            try:
                hist_2y = hist.tail(504) if not hist.empty else pd.DataFrame()
                if not hist_2y.empty and 'Close' in hist_2y.columns:
                    df_m = hist_2y['Close'].resample('ME').last().reset_index()
                    df_m['Date'] = pd.to_datetime(df_m['Date'].dt.strftime('%Y-%m-01'))
                    
                    precio_inicial_2y = df_m['Close'].iloc[0]
                    precio_final_2y = df_m['Close'].iloc[-1]
                    color_linea_2y = '#059669' if precio_final_2y >= precio_inicial_2y else '#DC2626'
                    
                    fig_monthly = go.Figure()
                    fig_monthly.add_trace(go.Scatter(
                        x=df_m['Date'], y=df_m['Close'],
                        mode='lines+markers', name='Precio Cierre',
                        line=dict(color=color_linea_2y, width=2.5),
                        marker=dict(size=6, color=color_linea_2y),
                        hovertemplate='<b>Fecha:</b> %{x|%b %Y}<br><b>Precio:</b> $%{y:,.2f}<extra></extra>'
                    ))
                    
                    fig_monthly.update_layout(
                        title=f"📈 Tendencia Mensual del Precio ({ticker_input} - Últimos 24 Meses)",
                        height=280, margin=dict(l=0, r=0, t=35, b=0),
                        font=dict(color='#0A192F', family='Trebuchet MS', size=12),
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(title="", showgrid=True, gridcolor='rgba(10, 25, 47, 0.15)', color='#0A192F', tickfont=dict(color='#0A192F', size=11), tickformat='%b %Y'),
                        yaxis=dict(title=dict(text="Precio ($USD)", font=dict(color='#0A192F', size=12)), range=[df_m['Close'].min() * 0.95, df_m['Close'].max() * 1.05], showgrid=True, gridcolor='rgba(10, 25, 47, 0.15)', color='#0A192F', tickfont=dict(color='#0A192F', size=11))
                    )
                    st.plotly_chart(fig_monthly, use_container_width=True)
            except Exception:
                pass
            is_fibra_util = sector in ["Real Estate", "Utilities", "Communication Services"]
            is_asset_light = sector in ["Technology", "Communication Services", "Financial Services"]
            
            total_debt = safe_get(info, ["totalDebt"], 0.0)
            total_cash = safe_get(info, ["totalCash"], 0.0)
            ebitda = safe_get(info, ["ebitda"], 1.0)
            net_debt = total_debt - total_cash
            
            net_cash_per_share = (total_cash - total_debt) / shares_current if shares_current > 0 else 0.0
            val_ncps_str = f"${net_cash_per_share:,.2f}" if net_cash_per_share > 0 else f"-${abs(net_cash_per_share):,.2f}"
            col_ncps = "🟢" if net_cash_per_share > 0 else "🔴"
            msg_ncps = f"Caja Neta Positiva de {val_ncps_str} por acción." if net_cash_per_share > 0 else f"Deuda Neta de {val_ncps_str} por acción."
            
            fcf_ttm = safe_get(info, ["freeCashflow", "operatingCashflow"], 0.0)
            fcf_debt_ratio = (fcf_ttm / total_debt * 100) if total_debt > 0 else (100.0 if fcf_ttm > 0 else 0.0)
            if fcf_debt_ratio >= 25.0: col_fcfd, msg_fcfd = "🟢", f"Excelente Cobertura: El FCF paga el {fcf_debt_ratio:.1f}% de toda la deuda en un solo año."
            elif fcf_debt_ratio >= 10.0: col_fcfd, msg_fcfd = "🟡", f"Cobertura Moderada: El FCF paga el {fcf_debt_ratio:.1f}% de la deuda total."
            else: col_fcfd, msg_fcfd = "🔴", f"Cobertura Débil: El FCF cubre solo el {fcf_debt_ratio:.1f}% de la deuda total."
            net_debt_ebitda = (net_debt / ebitda) if ebitda != 0 else 99
            nde_lim = (3.0, 4.5) if is_fibra_util else (1.5, 2.5)
            if net_debt < 0:
                col_nde, msg_nde, val_nde = "🟢", "Caja Neta Positiva: Cuenta con más efectivo que toda su deuda.", f"{net_debt_ebitda:.2f}x"
            elif net_debt_ebitda <= nde_lim[0]: col_nde, msg_nde, val_nde = "🟢", "Apalancamiento muy sano.", f"{net_debt_ebitda:.2f}x"
            elif net_debt_ebitda <= nde_lim[1]: col_nde, msg_nde, val_nde = "🟡", "Apalancamiento moderado/aceptable.", f"{net_debt_ebitda:.2f}x"
            else: col_nde, msg_nde, val_nde = "🔴", "Alto nivel de deuda neta estructural.", f"{net_debt_ebitda:.2f}x"
            int_exp = abs(safe_get(info, ["interestExpense"], 0.0))
            ebit = safe_get(info, ["operatingIncome", "ebitda"], ebitda)
            
            if int_exp <= 1:
                cob_int = 999.0
                col_cob, msg_cob, val_cob = "🟢", "Gastos por intereses nulos o mínimos.", "N/A"
            else:
                cob_int = ebit / int_exp
                if cob_int > 10.0: col_cob, msg_cob, val_cob = "🟢", "Cubre holgadamente sus intereses.", f"{cob_int:.1f}x"
                elif cob_int >= 4.0: col_cob, msg_cob, val_cob = "🟡", "Cobertura de intereses dentro del promedio.", f"{cob_int:.1f}x"
                else: col_cob, msg_cob, val_cob = "🔴", "Peligro: El flujo operativo apenas cubre los intereses.", f"{cob_int:.1f}x"
            cur_ratio = safe_get(info, ["currentRatio"], 1.2)
            msg_cur_alerta = ""
            if cur_ratio > 2.5: col_cur, msg_cur = "🟢", "Excelente liquidez, pero posible exceso de capital ocioso."
            elif cur_ratio >= 1.2: col_cur, msg_cur = "🟢", "Liquidez óptima a corto plazo."
            elif cur_ratio >= 0.9:
                col_cur, msg_cur = "🟡", "Liquidez moderada dentro del rango aceptable."
                if cur_ratio < 1.0: msg_cur_alerta = f"⚠️ Alerta de Liquidez: La Razón Corriente se ubica en {cur_ratio:.2f}x."
            else: col_cur, msg_cur = "🔴", "Problemas de liquidez a corto plazo."
                
            debt_eq = safe_get(info, ["debtToEquity"], 0.0) / 100
            total_eq = safe_get(info, ["totalStockholderEquity"], 1.0)
            op_cf = safe_get(info, ["operatingCashflow"], 0.0)
            
            if total_eq < 0:
                if fcf_ttm > 0 and op_cf > 0: col_de, msg_de, val_de = "🟡", "Patrimonio negativo por recompra masiva de acciones.", "Patr. Negativo"
                else: col_de, msg_de, val_de = "🔴", "Patrimonio negativo por acumulación de pérdidas.", "Insolvencia"
            else:
                if debt_eq < 0.8: col_de, msg_de, val_de = "🟢", "Bajo Apalancamiento contra capital.", f"{debt_eq:.2f}x"
                elif debt_eq <= 1.5: col_de, msg_de, val_de = "🟡", "Apalancamiento Moderado.", f"{debt_eq:.2f}x"
                else: col_de, msg_de, val_de = "🔴", "Apalancamiento Elevado.", f"{debt_eq:.2f}x"
                
            # --- MÓDULO 2: RENTABILIDAD Y EFICIENCIA (25%) ---
            net_income = safe_get(info, ["netIncomeToCommon"], 0.0)
            total_assets_val = safe_get(info, ["totalAssets"], 1.0)

            roe = safe_get(info, ["returnOnEquity"], 0.0) * 100
            if roe == 0.0 and net_income != 0.0 and total_eq > 0:
                roe = (net_income / total_eq) * 100

            roa = safe_get(info, ["returnOnAssets"], 0.0) * 100
            if roa == 0.0 and net_income != 0.0 and total_assets_val > 0:
                roa = (net_income / total_assets_val) * 100

            mg_op = safe_get(info, ["operatingMargins"], 0.0) * 100
            
            ebt = safe_get(info, ["pretaxIncome"], 1.0)
            tax_exp = safe_get(info, ["taxProvision"], 0.0)
            tax_rate = min(max(tax_exp / ebt, 0.0), 0.35) if ebt > 0 else 0.21
            nopat = ebit * (1 - tax_rate)
            
            def get_bs_val(metric):
                if not bs.empty and metric in bs.index:
                    val = bs.loc[metric].iloc[0]
                    return val if not pd.isna(val) else 0
                return 0
            ta = get_bs_val('Total Assets') or total_assets_val
            cl = get_bs_val('Current Liabilities') or safe_get(info, ["totalCurrentLiabilities"], 0.0)
            short_term_debt = max(get_bs_val('Current Debt'), get_bs_val('Current Debt And Capital Lease Obligation'))
            
            op_cl = max(cl - short_term_debt, 0)
            invested_capital_op = ta - op_cl
            if invested_capital_op <= 0:
                invested_capital_op = max(ta * 0.6, total_eq + total_debt - total_cash, 1.0)

            roic = (nopat / invested_capital_op) * 100 if invested_capital_op > 0 else (roe * 0.85)
            roic = min(max(roic, -50.0), 95.0) # Acotar ROIC a límites financieros realistas (-50% a 95%)

            if roic > 20: col_roic, msg_roic = "🟢", "Alta Calidad: Ventaja competitiva clara."
            elif roic >= 12: col_roic, msg_roic = "🟡", "Retorno sobre el capital aceptable."
            else: col_roic, msg_roic = "🔴", "Posible destrucción de valor operativo."
            if total_eq < 0: col_roe, msg_roe, val_roe = "⚪", "ROE No Aplicable (Patrimonio Negativo).", "N/A"
            else:
                if roe > 20: col_roe, msg_roe, val_roe = "🟢", "Excelente rentabilidad sobre capital propio.", f"{roe:.1f}%"
                elif roe >= 10: col_roe, msg_roe, val_roe = "🟡", "Rendimiento aceptable.", f"{roe:.1f}%"
                else: col_roe, msg_roe, val_roe = "🔴", "Rendimiento deficiente.", f"{roe:.1f}%"
            if is_asset_light and roa > 15: col_roa, msg_roa = "🟢", "Modelo Asset-Light Sobresaliente."
            elif roa > 8: col_roa, msg_roa = "🟢", "Alta Eficiencia en Uso de Recursos."
            elif roa >= 4: col_roa, msg_roa = "🟡", "Eficiencia Moderada."
            else: col_roa, msg_roa = "🔴", "Baja Eficiencia de Activos."
            divergencia_roa = "⚠️ Alerta de Apalancamiento: Brecha extrema entre ROE y ROA." if (roa > 0 and (roe / roa) > 4.0) else ""
            
            fcf_conv = (fcf_ttm / net_income) if net_income != 0 else 0.0
            if fcf_conv > 1.0: col_fcfc, msg_fcfc = "🟢", "Genera más dinero líquido que ganancia contable."
            elif fcf_conv >= 0.8: col_fcfc, msg_fcfc = "🟢", "Flujo de Caja Sano."
            else: col_fcfc, msg_fcfc = "🔴", "Baja conversión contable a flujo de caja."
            
            alerta_fcf_calidad = f"⚠️ Alerta de Calidad de Flujo: Convierte solo el {fcf_conv*100:.1f}% de utilidades en FCF." if (fcf_conv < 0.8 and net_income > 0) else ""
            eps_growth = safe_get(info, ["earningsGrowth"], 0.0) * 100
            ni_growth = safe_get(info, ["netIncomeGrowth"], eps_growth/100) * 100
            msg_eps = ""
            if eps_growth > 10 and ni_growth < 3: col_eps, msg_eps = "🟡", "Alerta: EPS impulsado por recompra de acciones."
            elif eps_growth > 10: col_eps, msg_eps = "🟢", "Crecimiento robusto de beneficios."
            elif eps_growth >= 6: col_eps, msg_eps = "🟡", "Crecimiento moderado."
            else: col_eps, msg_eps = "🔴", "Crecimiento lento o estancado."
            # BUYBACK YIELD
            sh_prev = bs.loc['Basic Average Shares'].iloc[1] if (not bs.empty and 'Basic Average Shares' in bs.index and len(bs.columns) > 1 and not pd.isna(bs.loc['Basic Average Shares'].iloc[1])) else shares_current
            buyback_yield = ((sh_prev - shares_current) / sh_prev * 100) if sh_prev > 0 else 0.0
            if buyback_yield >= 1.5: col_by, msg_by = "🟢", f"Fuerte Recompra: Reduce flotante en {buyback_yield:.1f}% anual."
            elif buyback_yield >= 0.0: col_by, msg_by = "🟡", f"Recompra Neutra: Variación de {buyback_yield:.1f}%."
            else: col_by, msg_by = "🔴", f"Dilución de Accionistas: Incrementa flotante en {abs(buyback_yield):.1f}%."
            div_rate, div_yield, next_div_date = obtener_datos_dividendos(ticker_input, info, fmp_key, precio_actual)
            val_div_metric = f"${div_rate:.2f} ({div_yield:.2f}%)" if div_rate > 0 else "N/A"
            msg_div_tooltip = f"Rendimiento anualizado por dividendo: {div_yield:.2f}%\nPróxima fecha ex-dividendo estimada: {next_div_date}"
            fcf_yield = (fcf_ttm / mcap * 100) if mcap > 0 else 0.0
            if fcf_yield >= (tasa_libre_riesgo + 3.0): col_fcfy, msg_fcfy = "🟢", f"Excelente Rendimiento FCF ({fcf_yield:.2f}% vs FRED {tasa_libre_riesgo:.2f}%)."
            elif fcf_yield >= tasa_libre_riesgo: col_fcfy, msg_fcfy = "🟡", f"Rendimiento de Caja Aceptable ({fcf_yield:.2f}% vs FRED {tasa_libre_riesgo:.2f}%)."
            else: col_fcfy, msg_fcfy = "🔴", f"Rendimiento Exigente ({fcf_yield:.2f}% vs FRED {tasa_libre_riesgo:.2f}%)."
            
            # --- MÓDULO 3: VALUACIÓN, DCF Y DDM (40%) ---
            beta = safe_get(info, ["beta"], 1.0)
            res_wacc = calcular_wacc(tasa_libre_riesgo, beta, mcap, total_debt, int_exp, fmp_key, fred_key, tax_rate, ticker_input)
            ke, kd, wacc, we, wd = res_wacc["ke"], res_wacc["kd"], res_wacc["wacc"], res_wacc["we"], res_wacc["wd"]
            
            g_1_5 = min(max(safe_get(info, ["earningsGrowth"], safe_get(info, ["revenueGrowth"], 0.08)) * 0.85, 0.04), 0.18) 
            g_term = 0.025 if 0.025 < (wacc/100) else (wacc/100) - 0.015
            
            fcf_base_per_share = (fcf_ttm / shares_current) if shares_current > 0 else 0
            eps_ttm = net_income / shares_current if shares_current > 0 else 0
            flujo_por_accion = max(fcf_base_per_share, eps_ttm * 0.85) or (precio_actual * 0.035)
            
            calcular_dcf_fn = crear_calculador_dcf(flujo_por_accion, g_1_5, precio_actual, eps_ttm, total_cash, total_debt, shares_current)
            v_intr_dcf = calcular_dcf_fn(wacc, g_term)
            
            g_div = min(max(g_1_5 * 0.5, 0.02), 0.04)
            res_ddm = calcular_ddm(div_rate, ke, g_div, precio_actual)
            v_intr_ddm = res_ddm["valor_intrinseco_ddm"]
            val_ddm_str = res_ddm["val_ddm_str"]
            col_ddm = res_ddm["status"]
            mostrar_ddm = (div_rate > 0 and (div_yield >= 1.8 or is_fibra_util))
            v_intr = v_intr_dcf
                
            pe = (precio_actual / eps_ttm) if eps_ttm > 0 else safe_get(info, ["trailingPE"], 0.0)
            p_fcf = (mcap / fcf_ttm) if fcf_ttm != 0 else 0.0
            ev_ebitda = ((mcap + total_debt - total_cash) / ebitda) if ebitda > 0 else 0.0
            peg = safe_get(info, ["pegRatio"], 0.0)
            target = safe_get(info, ["targetMeanPrice"], 0.0)
            upside = (((target - precio_actual) / precio_actual) * 100) if precio_actual != 0 else 0.0
            
            col_pfcf, msg_pfcf = ("🟢", "Gran Rendimiento de Caja.") if (0 < p_fcf < 18) else (("🟡", "Valuación de caja moderada.") if p_fcf <= 25 else ("🔴", "Valuación exigente."))
            col_ev, msg_ev = ("🟢", "Valuación Atractiva.") if (0 < ev_ebitda < 12) else (("🟡", "Valuación Razonable.") if ev_ebitda <= 18 else ("🔴", "Valuación Elevada."))
            col_peg, msg_peg = ("🟢", "Crecimiento a muy buen precio.") if (0 < peg < 1.2) else (("🟡", "Valuación justa por crecimiento.") if peg <= 2.0 else ("🔴", "Exceso de prima por crecimiento."))
            col_pe, msg_pe = ("🟢", "Descuento histórico/sectorial.") if (0 < pe < 20) else (("🟡", "Valuación razonable.") if (0 < pe <= 30) else ("🔴", "Sobrevalorada por PER."))
            col_upside = "🟢" if upside > 0 else "🔴"
            col_vintr = "🟢" if v_intr_dcf >= precio_actual else "🔴"
            
            # --- MÓDULO 4: RIESGOS Y SALUD CONTABLE (15%) ---
            res_z = calcular_altman_zscore(debt_eq, roa)
            z_score, col_z, msg_z = res_z["z_score"], res_z["status"], res_z["msg_z"]
            short_int = safe_get(info, ["shortPercentOfFloat"], 0.0) * 100
            
            col_b, msg_b = ("🟢", "Volatilidad baja (Defensiva).") if beta < 0.8 else (("🟡", "Volatilidad moderada.") if beta <= 1.4 else ("🔴", "Alta volatilidad sistémica."))
            col_s, msg_s = ("🟢", "Bajo interés en corto.") if short_int < 5 else (("🟡", "Posicionamiento en corto moderado.") if short_int <= 10 else ("🔴", "Fuerte pesimismo en corto."))
            
            res_fs = calcular_piotroski_fscore(inc, bs, cf, info)
            f_score, fscore_str, col_fscore, msg_fscore = res_fs["f_score"], res_fs["fscore_str"], res_fs["status"], res_fs["msg_fscore"]
            
            # --- EXTRACCIÓN MACROECONÓMICA Y GEOPOLÍTICA (5%) ---
            texto_ia_final, macro_score = obtener_analisis_macro_ia(ticker_input, nombre, sector, gemini_key)
            
            # --- MÓDULO 5: SCORING MATEMÁTICO FINAL Y RADAR (100 Pts Total) ---
            res_score = calcular_scoring(
                col_nde, col_cob, col_cur, col_de, col_roic, mg_op, col_fcfc, col_roe, col_roa,
                v_intr, precio_actual, pe, col_pfcf, col_ev, col_peg, col_z, col_b, col_s, macro_score
            )
            pts, pts_solvencia, pts_rentabilidad, pts_valuacion, pts_riesgos = (
                res_score["pts_total"], res_score["pts_solvencia"], res_score["pts_rentabilidad"],
                res_score["pts_valuacion"], res_score["pts_riesgos"]
            )
            
            # --- AJUSTE FINANCIERO DINÁMICO (SMALL CAPS) ---
            if mcap > 0 and mcap < 2000000000:
                desc_req = 0.20 # 20% Margen de Seguridad por Riesgo de Liquidez en Small Caps reales
            else:
                desc_req = 0.10 # 10% Margen Normal
                
            clase_msg = "Clase A (Alta Calidad)" if pts > 75 else ("Clase B (Calidad Sólida)" if pts >= 50 else "Clase C (Cíclica / Riesgo)")
            
            precio_max_compra = v_intr * (1 - desc_req)
            col_pmax = "🟢" if precio_max_compra >= precio_actual else "🔴"
            
            res_veredicto = evaluar_veredicto(pts, z_score, net_debt_ebitda, is_fibra_util, cob_int, int_exp, roic)
            is_knockout, veredicto, color_v, veredicto_txt = (
                res_veredicto["is_knockout"], res_veredicto["veredicto"], res_veredicto["color_v"], res_veredicto["veredicto_txt"]
            )
            
            doble_filtro = "🟢 Oportunidad de Alta Confianza (Cotiza por debajo de tu Precio Máx de Compra y Wall Street le ve alto potencial)." if (precio_actual <= precio_max_compra and upside > 15) else ("🟡 Valor Oculto" if (v_intr > precio_actual and upside > 0) else "⚪ Valuación Justa o Mixta")

            # --- DEFINICIÓN DE TOOLTIPS MÓDULOS 1 Y 2 ---
            h_nde = f"¿Qué es? Deuda neta dividida por EBITDA.\n¿Para qué sirve? Mide cuántos años tardaría en pagar su deuda.\nDiagnóstico: {msg_nde}"
            h_cob = f"¿Qué es? Beneficio operativo entre gastos por intereses.\n¿Para qué sirve? Indica capacidad para pagar los intereses de su deuda.\nDiagnóstico: {msg_cob}"
            h_cur = f"¿Qué es? Activos a corto plazo entre pasivos a corto plazo.\n¿Para qué sirve? Mide la liquidez para pagar deudas de menos de un año.\nDiagnóstico: {msg_cur}"
            h_ncps = f"¿Qué es? Efectivo menos deuda total dividido entre acciones circulantes.\n¿Para qué sirve? Mide el respaldo líquido real por acción.\nDiagnóstico: {msg_ncps}"
            h_fcfd = f"¿Qué es? Flujo de Caja Libre dividido entre la Deuda Total.\n¿Para qué sirve? Mide la capacidad de pago acelerado de deuda mediante caja líquida.\nDiagnóstico: {msg_fcfd}"
            
            h_roic = f"¿Qué es? Retorno sobre el Capital Invertido Operativo.\n¿Para qué sirve? Mide la eficiencia del negocio principal para generar dinero.\nDiagnóstico: {msg_roic}"
            h_roe  = f"¿Qué es? Retorno sobre el Capital.\n¿Para qué sirve? Mide la rentabilidad generada con el dinero de los accionistas.\nDiagnóstico: {msg_roe}"
            h_roa  = f"¿Qué es? Retorno sobre Activos Totales.\n¿Para qué sirve? Mide la eficiencia directiva usando todos los activos.\nDiagnóstico: {msg_roa}"
            h_fcfc = f"¿Qué es? Flujo de Caja Libre entre Utilidad Neta.\n¿Para qué sirve? Valida que las ganancias contables se conviertan en dinero real.\nDiagnóstico: {msg_fcfc}"
            h_fcfy = f"¿Qué es? Flujo de Caja Libre entre Capitalización de Mercado.\n¿Para qué sirve? Compara la rentabilidad real de caja frente al Bono FRED ({tasa_libre_riesgo:.2f}%).\nDiagnóstico: {msg_fcfy}"
            h_eps  = f"¿Qué es? Crecimiento de Ganancias por Acción.\n¿Para qué sirve? Ritmo al que crecen las utilidades para el dueño.\nDiagnóstico: {msg_eps}"
            h_by   = f"¿Qué es? Variación porcentual interanual de acciones en circulación.\n¿Para qué sirve? Audita la reducción de flotante por recompras masivas.\nDiagnóstico: {msg_by}"
            
            # --- DEFINICIÓN DE TOOLTIPS DINÁMICOS MÓDULO 3 ---
            h_vint = "¿Qué es? Modelo DCF (Flujos de Caja Descontados).\n¿Para qué sirve? Estima el valor presente de la caja libre futura que generará el negocio, descontada al WACC.\nDiagnóstico: Representa el valor intrínseco fundamental."
            h_ddm  = "¿Qué es? Modelo Gordon Growth (DDM).\n¿Para qué sirve? Valúa la acción basándose en el valor presente de sus dividendos futuros asumiendo crecimiento constante.\nDiagnóstico: Complementario para empresas maduras o REITs."
            h_pmax = f"¿Qué es? Valor Intrínseco menos Margen de Seguridad.\n¿Para qué sirve? Define el umbral máximo de precio para mitigar riesgos de error en la proyección.\nDiagnóstico: Descuento de seguridad exigido del {desc_req*100:.0f}%."
            h_ws   = f"¿Qué es? Precio Objetivo Consenso (12 meses).\n¿Para qué sirve? Indica la valoración promedio proyectada por analistas de Wall Street.\nDiagnóstico: Upside esperado del {upside:.1f}%."
            
            h_pe   = f"¿Qué es? Múltiplo Precio / Beneficio Neto (P/E).\n¿Para qué sirve? Mide la valoración de las utilidades contables del último año (TTM).\nDiagnóstico: La empresa cotiza a {pe:.1f} veces sus ganancias anuales, lo que requiere un crecimiento continuo de beneficios para sostener la valoración de mercado."
            h_pfcf = f"¿Qué es? Precio / Flujo de Caja Libre.\n¿Para qué sirve? Valora la empresa en función del efectivo real generado, aislando posibles trucos contables.\nDiagnóstico: Con un P/FCF de {p_fcf:.1f}x, la empresa requiere {p_fcf:.1f} años de generación de flujo libre de caja para cubrir su precio actual de mercado."
            
            if peg > 1:
                peg_msg = f"una prima del {(peg-1)*100:.0f}% sobre"
            elif 0 < peg <= 1:
                peg_msg = f"un descuento del {(1-peg)*100:.0f}% frente a"
            else:
                peg_msg = "una prima sobre"
            h_peg  = f"¿Qué es? PER / Tasa de Crecimiento Esperado.\n¿Para qué sirve? Ajusta el múltiplo de valuación según la velocidad a la que crecen las utilidades.\nDiagnóstico: Con un PEG de {peg:.2f}x, la acción cotiza con {peg_msg} la tasa de crecimiento esperada de sus ganancias."
            
            h_ev   = f"¿Qué es? Valor de Empresa (EV) / EBITDA.\n¿Para qué sirve? Valora la operación del negocio neutralizando su estructura de deuda y carga fiscal.\nDiagnóstico: El Valor total de la Empresa (EV) equivale a {ev_ebitda:.1f} años de su beneficio operativo EBITDA actual."
            
            h_z    = f"¿Qué es? Modelo Altman Z-Score.\n¿Para qué sirve? Predice riesgo de quiebra.\nDiagnóstico: {msg_z}"
            h_b    = f"¿Qué es? Coeficiente Beta.\n¿Para qué sirve? Mide volatilidad bursátil.\nDiagnóstico: {msg_b}"
            h_s    = f"¿Qué es? Short Interest (% Float).\n¿Para qué sirve? Apuestas institucionales en contra.\nDiagnóstico: {msg_s}"
            h_fs   = f"¿Qué es? Piotroski F-Score.\n¿Para qué sirve? Audita 9 criterios contables.\nDiagnóstico: {msg_fscore}"

            # --- GENERAR PERFIL CORPORATIVO CON GEMINI ---
            perfil_texto = obtener_perfil_corporativo(ticker_input, nombre, sector, industria, gemini_key, mcap, mg_op, roic)

            # --- RENDERIZADO VISUAL STREAMLIT ---
            st.markdown("---")
            tab_dashboard, tab_perfil, tab_noticias, tab_directorio = st.tabs(["📊 Dashboard de Evaluación", "🏢 Perfil Corporativo", "📰 Noticias", "🗂️ Directorio"])
            
            with tab_dashboard:
                st.markdown(f"## {color_v} Score Final: {pts}/100 Pts - {veredicto}")
                
                radar_col1, radar_col2 = st.columns([1, 2])
                with radar_col1:
                    st.write(f"**Razonamiento:** {veredicto_txt}")
                    st.info(f"**Doble Filtro de Entrada:** {doble_filtro}")
                    if mcap > 0 and mcap < 2000000000:
                        st.info("ℹ️ **Ajuste de Riesgo DCF (Small Cap):** Se ha aplicado un Margen de Seguridad exigente del 20% debido a que la emisora posee un Market Cap inferior a $2B USD.")
                with radar_col2:
                    categories = ['Solvencia (Max 15)', 'Rentabilidad (Max 25)', 'Valuación (Max 40)', 'Riesgos (Max 15)', 'Macro (Max 5)']
                    scores = [pts_solvencia, pts_rentabilidad, pts_valuacion, pts_riesgos, macro_score]
                    max_scores = [15, 25, 40, 15, 5]
                    
                    scores_pct = [(s/m)*100 for s, m in zip(scores, max_scores)]
                    
                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(
                        r=scores_pct, theta=categories, fill='toself', name='Fortalezas Top-Down',
                        line_color='rgba(10, 25, 47, 0.8)', fillcolor='rgba(10, 25, 47, 0.2)'
                    ))
                    fig_radar.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                        showlegend=False, margin=dict(l=40, r=40, t=20, b=20),
                        font=dict(color='#0A192F', family='Trebuchet MS', size=11),
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=250
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)
                st.markdown("---")
                
                # MÓDULO 1: SOLVENCIA
                st.markdown("### 🏦 Módulo 1: Solvencia y Liquidez (15%)")
                if msg_cur_alerta: st.warning(msg_cur_alerta)
                c1, c2, c3, c4, c4_fcfd = st.columns(5)
                
                c1.metric(f"{col_nde} Deuda Neta / EBITDA", val_nde, help=h_nde)
                c2.metric(f"{col_cob} Cobertura Intereses", val_cob, help=h_cob)
                c3.metric(f"{col_cur} Razón Corriente", f"{cur_ratio:.2f}x", help=h_cur)
                c4.metric(f"{col_ncps} Caja Neta / Acc", val_ncps_str, help=h_ncps)
                c4_fcfd.metric(f"{col_fcfd} Cobertura FCF/Deuda", f"{fcf_debt_ratio:.1f}%", help=h_fcfd)
                
                # MÓDULO 2: RENTABILIDAD
                st.markdown("### 📈 Módulo 2: Rentabilidad y Eficiencia (25%)")
                if divergencia_roa: st.warning(divergencia_roa)
                if alerta_fcf_calidad: st.warning(alerta_fcf_calidad)
                if col_eps == "🟡" and msg_eps.startswith("Alerta"): st.warning(msg_eps)
                
                c5, c6, c7, c8 = st.columns(4)
                c5.metric(f"{col_roic} ROIC", f"{roic:.1f}%", help=h_roic)
                c6.metric(f"{col_roe} ROE", val_roe, help=h_roe)
                c7.metric(f"{col_roa} ROA", f"{roa:.1f}%", help=h_roa)
                c8.metric(f"{col_fcfc} Conv. FCF", f"{fcf_conv:.1f}x", help=h_fcfc)
                c8_fcfy, c9, c9_by, c9_div = st.columns(4)
                c8_fcfy.metric(f"{col_fcfy} FCF Yield", f"{fcf_yield:.2f}%", help=h_fcfy)
                c9.metric(f"{col_eps} EPS YoY", f"{eps_growth:.1f}%", help=h_eps)
                c9_by.metric(f"{col_by} Buyback Yield", f"{buyback_yield:.1f}%", help=h_by)
                c9_div.metric("💵 Dividendos", val_div_metric, help=msg_div_tooltip)
                
                # --- GRÁFICOS MÓDULO 2 ---
                g_col1, g_col2 = st.columns(2)
                
                with g_col1:
                    st.write("**Comparativa de Márgenes (5 Años)**")
                    if 'Gross Profit' in inc.index and 'Operating Income' in inc.index and 'Total Revenue' in inc.index:
                        try:
                            rev_seguro = inc.loc['Total Revenue'].replace(0, np.nan)
                            df_margins = pd.DataFrame({
                                "Margen Operativo (%)": (inc.loc['Operating Income'] / rev_seguro) * 100,
                                "Margen Neto (%)": (inc.loc['Net Income'] / rev_seguro) * 100
                            }).dropna()
                            
                            df_margins.index = pd.to_datetime(df_margins.index).year.astype(str)
                            df_margins = df_margins.sort_index()
                            fig_m = px.bar(df_margins, barmode='group', text_auto='.1f', color_discrete_sequence=['#0A192F', '#0284C7'])
                            fig_m.update_layout(
                                xaxis_title="", yaxis_title="Porcentaje (%)", legend_title="",
                                font=dict(color='#0A192F', family='Trebuchet MS', size=11),
                                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color='#0A192F', size=11)),
                                margin=dict(l=0, r=0, t=30, b=0),
                                xaxis=dict(color='#0A192F', showgrid=True, gridcolor='rgba(10, 25, 47, 0.15)', tickfont=dict(color='#0A192F', size=11)),
                                yaxis=dict(color='#0A192F', showgrid=True, gridcolor='rgba(10, 25, 47, 0.15)', tickfont=dict(color='#0A192F', size=11))
                            )
                            st.plotly_chart(fig_m, use_container_width=True)
                        except Exception:
                            st.write("Gráfico de márgenes no disponible para este ticker.")
                with g_col2:
                    st.write("**Comparativa de Flujos de Efectivo (5 Años - Millones USD)**")
                    if not cf.empty and 'Operating Cash Flow' in cf.index and 'Free Cash Flow' in cf.index:
                        try:
                            df_cf = pd.DataFrame({
                                "Flujo Operativo": cf.loc['Operating Cash Flow'] / 1e6,
                                "Flujo Libre (FCF)": cf.loc['Free Cash Flow'] / 1e6
                            }).dropna()
                            
                            df_cf.index = pd.to_datetime(df_cf.index).year.astype(str)
                            df_cf = df_cf.sort_index()
                            fig_cf = px.bar(df_cf, barmode='group', text_auto=',.0f', color_discrete_sequence=['#0284C7', '#059669'])
                            fig_cf.update_layout(
                                xaxis_title="", yaxis_title="Millones USD ($)", legend_title="",
                                font=dict(color='#0A192F', family='Trebuchet MS', size=11),
                                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color='#0A192F', size=11)),
                                margin=dict(l=0, r=0, t=30, b=0),
                                xaxis=dict(color='#0A192F', showgrid=True, gridcolor='rgba(10, 25, 47, 0.15)', tickfont=dict(color='#0A192F', size=11)),
                                yaxis=dict(color='#0A192F', showgrid=True, gridcolor='rgba(10, 25, 47, 0.15)', tickfont=dict(color='#0A192F', size=11))
                            )
                            st.plotly_chart(fig_cf, use_container_width=True)
                        except Exception:
                            st.write("Gráfico de flujos no disponible para este ticker.")
                st.markdown("### 🏷️ Módulo 3: Valuación y Valor Intrínseco (40%)")
                
                if mostrar_ddm:
                    row1_c1, row1_c2, row1_c3, row1_c4 = st.columns(4)
                    row1_c1.metric(f"{col_vintr} V. Intrínseco (DCF)", f"${v_intr_dcf:,.2f}", help=h_vint)
                    row1_c2.metric(f"{col_ddm} V. Intrínseco (DDM)", val_ddm_str, help=h_ddm)
                    row1_c3.metric(f"{col_pmax} P. Máx Compra", f"${precio_max_compra:,.2f}", help=h_pmax)
                    row1_c4.metric(f"{col_upside} Consenso W.St", f"${target:,.2f}", f"{upside:.1f}%", help=h_ws)
                else:
                    row1_c1, row1_c2, row1_c3 = st.columns(3)
                    row1_c1.metric(f"{col_vintr} V. Intrínseco (DCF)", f"${v_intr_dcf:,.2f}", help=h_vint)
                    row1_c2.metric(f"{col_pmax} P. Máx Compra", f"${precio_max_compra:,.2f}", help=h_pmax)
                    row1_c3.metric(f"{col_upside} Consenso W.St", f"${target:,.2f}", f"{upside:.1f}%", help=h_ws)
                st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
                row2_c1, row2_c2, row2_c3, row2_c4 = st.columns(4)
                row2_c1.metric(f"{col_pe} PER (P/E)", f"{pe:.1f}x", help=h_pe)
                row2_c2.metric(f"{col_pfcf} P/FCF", f"{p_fcf:.1f}x", help=h_pfcf)
                row2_c3.metric(f"{col_peg} PEG Forward", f"{peg:.2f}x", help=h_peg)
                row2_c4.metric(f"{col_ev} EV / EBITDA", f"{ev_ebitda:.1f}x", help=h_ev)
                if mostrar_ddm:
                    explicacion_modelo = (
                        f"💡 **Análisis Comparativo de Valuación (DCF vs. DDM):**\n\n"
                        f"* **Modelo DCF:** Valúa el 100% de la caja libre generada por {nombre} (operaciones, reinversión y recompras de acciones).\n"
                        f"* **Modelo DDM (Gordon Growth):** Valúa únicamente la corriente directa de dividendos en efectivo distribuidos al accionista (Yield: {div_yield:.2f}%).\n\n"
                        f"**Criterio:** Al ser una empresa madura o de alto dividendo, ambos modelos ofrecen una perspectiva complementaria."
                    )
                else:
                    explicacion_modelo = (
                        f"💡 **Criterio de Valuación Seleccionado (DCF):**\n\n"
                        f"Para **{nombre}** (empresa orientada a crecimiento o con bajo dividendo del {div_yield:.2f}%), el **Modelo de Descuento de Flujos de Caja (DCF)** es el estándar de oro utilizado por el sistema. "
                        f"Este método valúa la caja libre operativa total y el beneficio de las recompras de acciones, representando con precisión la capacidad total del negocio."
                    )
                st.info(explicacion_modelo)
                st.markdown("### 🛡️ Módulo 4: Capa de Riesgos y Salud Contable (15%)")
                c16, c17, c18, c19 = st.columns(4)
                c16.metric(f"{col_z} Altman Z-Score", f"{z_score:.2f}", help=h_z)
                c17.metric(f"{col_b} Beta (Volatilidad)", f"{beta:.2f}", help=h_b)
                c18.metric(f"{col_s} Short Interest", f"{short_int:.2f}%", help=h_s)
                c19.metric(f"{col_fscore} Piotroski F-Score", fscore_str, help=h_fs)
                with st.expander("📖 Guía de Interpretación de Riesgos (Desplegar)"):
                    st.markdown('''
                    **1. Altman Z-Score (Probabilidad de Insolvencia)**
                    Algoritmo que mide la probabilidad matemática de quiebra en los próximos 2 años evaluando liquidez, rentabilidad y deuda.
                    * 🟢 **Zona Segura (> 2.99):** Finanzas robustas, riesgo de insolvencia estadísticamente nulo.
                    * 🟡 **Zona Gris (1.81 - 2.99):** Precaución moderada. Requiere vigilancia de obligaciones a corto plazo.
                    * 🔴 **Zona de Peligro (< 1.81):** Alto riesgo de reestructuración financiera o quiebra.
                    **2. Beta (Volatilidad Sistémica)**
                    Indica cómo reacciona la acción frente a los movimientos bruscos del mercado general (S&P 500).
                    * 🟢 **Defensivo (< 0.8):** Acción refugio. Sube o baja menos agresivamente que el mercado.
                    * 🟡 **Promedio (0.8 - 1.4):** Se mueve a la par o ligeramente más rápido que el mercado global.
                    * 🔴 **Alta Volatilidad (> 1.4):** Movimientos extremos. Mayor sensibilidad a caídas en pánico.
                    **3. Short Interest (Interés en Corto)**
                    Porcentaje de acciones circulantes que los inversores institucionales han tomado prestadas apostando a que el precio caerá.
                    * 🟢 **Saludable (< 5%):** Nivel normal de mercado. No hay pesimismo evidente.
                    * 🟡 **Moderado (5% - 10%):** Especulación creciente. Algunas dudas institucionales sobre el corto plazo.
                    * 🔴 **Pesimismo Extremo (> 10%):** Fuerte presión vendedora en Wall Street. Alerta de problemas fundamentales ocultos.
                    **4. Piotroski F-Score (Calidad Contable)**
                    Metodología que audita 9 criterios de los estados financieros para garantizar que las ganancias son reales y no producto de contabilidad creativa.
                    * 🟢 **Fuerte (7 - 9):** Excelente calidad. Las ganancias se respaldan con flujo de caja real y la deuda disminuye.
                    * 🟡 **Promedio (4 - 6):** Situación estable pero sin crecimiento orgánico destacable.
                    * 🔴 **Débil (0 - 3):** Riesgo severo. Destrucción de liquidez, dilución de acciones o posible manipulación de ingresos.
                    ''')
                st.markdown("---")
                st.markdown("### 🌍 Análisis Macroeconómico y Geopolítico (IA Gemini)")
                st.write(texto_ia_final)
                st.caption(f"**Puntuación IA aportada al modelo:** {macro_score}/5 pts")

            # --- CONTENIDO DE LA PESTAÑA: PERFIL CORPORATIVO ---
            with tab_perfil:
                st.markdown(f"<h2 style='color: #0A192F; font-weight: 700;'>🏢 Investigación Institucional: {nombre} ({ticker_input})</h2>", unsafe_allow_html=True)
                st.write(perfil_texto)

            # --- CONTENIDO DE LA PESTAÑA: NOTICIAS Y CATALIZADORES ---
            with tab_noticias:
                st.markdown(f"<h2 style='color: #0A192F; font-weight: 700;'>📰 Noticias y Catalizadores Recientes ({ticker_input})</h2>", unsafe_allow_html=True)
                news_data = news_data_cache if news_data_cache else obtener_noticias_financieras(ticker_input)
                
                if news_data:
                    for n in news_data:
                        link_href = n['link'] if n['link'] and n['link'].startswith('http') else f"https://finance.yahoo.com/quote/{ticker_input}/news"
                        st.markdown(f"""
                        <div style='border: 1px solid #C5A059; border-radius: 8px; padding: 15px; margin-bottom: 15px; background-color: #FFFFFF; box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>
                            <p style='color: #64748B; font-size: 12px; margin-bottom: 5px; font-weight: 700;'>🏢 {n['publisher']} &nbsp;|&nbsp; 🕒 {n['date']}</p>
                            <h4 style='color: #0A192F; margin-top: 0px; margin-bottom: 8px; font-size: 16px;'>{n['title']}</h4>
                            <a href='{link_href}' target='_blank' style='color: #0284C7; text-decoration: none; font-weight: 600; font-size: 13px;'>🔗 Leer noticia completa en fuente original</a>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info(f"En este momento no hay titulares financieros recientes disponibles en los feeds primarios para {ticker_input}.")

            # --- CONTENIDO DE LA PESTAÑA: DIRECTORIO SECTORIAL ---
            with tab_directorio:
                render_directorio()

            # --- RENDERIZADO SIDEBAR: BENCHMARKS ---
            st.sidebar.markdown("---")
            st.sidebar.subheader(f"📊 Benchmarks Institucionales")
            
            if datos_completos:
                st.sidebar.caption("🟢 **Integridad de Datos: 100%**")
            else:
                st.sidebar.caption("🟡 **Datos Parciales Detectados**")
                
            st.sidebar.metric("Tasa Libre Riesgo (FRED)", f"{tasa_libre_riesgo:.2f}%", help="Rendimiento actual del Bono del Tesoro a 10 años (FRED). Define el WACC base.")
            
            sec_b = SECTOR_BENCHMARKS.get(sector, {"PE": 20.0, "PEG": 1.5, "PFCF": 18.0, "ROA": 5.0, "ROE": 14.0, "ROI": 10.0})
            
            def eval_val(val, target): 
                if val <= 0: return "⚪"
                if val <= target * 1.10: return "🟢"
                elif val <= target * 1.35: return "🟡"
                else: return "🔴"
            def eval_rent(val, target): 
                if val >= target * 0.90: return "🟢"
                elif val >= target * 0.70: return "🟡"
                else: return "🔴"
                
            def format_val_multiplo(name, val, target, is_peg=False):
                semaforo = eval_val(val, target)
                val_str = f"{val:.2f}x" if is_peg else f"{val:.1f}x"
                tgt_str = f"{target:.2f}x" if is_peg else f"{target:.1f}x"
                if target > 0 and val > 0:
                    diff = ((val / target) - 1) * 100
                    extra = f"Cotiza con {diff:.1f}% de Prima" if diff > 0 else f"Cotiza con {abs(diff):.1f}% de Descuento"
                else:
                    extra = "Datos insuficientes"
                return f"<div style='line-height: 1.4; margin-bottom: 8px;'>{semaforo} <b>{name}:</b> {val_str} vs Sec {tgt_str}<br><span style='font-size: 11.5px; color: #475569;'><i>({extra})</i></span></div>"
                
            def format_val_rentabilidad(name, val, target):
                semaforo = eval_rent(val, target)
                if target > 0:
                    diff = ((val / target) - 1) * 100
                    extra = f"+{diff:.1f}% superior al sector" if diff > 0 else f"{diff:.1f}% inferior al sector"
                else:
                    extra = "Datos insuficientes"
                return f"<div style='line-height: 1.4; margin-bottom: 8px;'>{semaforo} <b>{name}:</b> {val:.1f}% vs Sec {target:.1f}%<br><span style='font-size: 11.5px; color: #475569;'><i>({extra})</i></span></div>"
                
            st.sidebar.markdown(f"**Comparativa Sector ({sector}):**")
            st.sidebar.markdown(format_val_multiplo("PER", pe, sec_b['PE']), unsafe_allow_html=True)
            st.sidebar.markdown(format_val_multiplo("PEG", peg, sec_b['PEG'], is_peg=True), unsafe_allow_html=True)
            st.sidebar.markdown(format_val_multiplo("P/FCF", p_fcf, sec_b['PFCF']), unsafe_allow_html=True)
            st.sidebar.markdown(format_val_rentabilidad("ROA", roa, sec_b['ROA']), unsafe_allow_html=True)
            st.sidebar.markdown(format_val_rentabilidad("ROE", roe, sec_b['ROE']), unsafe_allow_html=True)
            st.sidebar.markdown(format_val_rentabilidad("ROI/ROIC", roic, sec_b['ROI']), unsafe_allow_html=True)
            
            # --- GENERADOR DE PDF INSTITUCIONAL PREMIUM (3 PÁGINAS) ---
            st.markdown("---")
            analysis_data = {
                "ticker": ticker_input,
                "nombre": nombre,
                "sector": sector,
                "pts": pts,
                "veredicto": veredicto,
                "veredicto_txt": veredicto_txt,
                "v_intr_dcf": v_intr_dcf,
                "precio_actual": precio_actual,
                "precio_max_compra": precio_max_compra,
                "desc_req": desc_req,
                "clase_msg": clase_msg,
                "val_nde": val_nde,
                "val_cob": val_cob,
                "val_de": val_de,
                "roic": roic,
                "fcf_yield": fcf_yield,
                "tasa_libre_riesgo": tasa_libre_riesgo,
                "val_roe": val_roe,
                "roa": roa,
                "mg_op": mg_op,
                "fcf_conv": fcf_conv,
                "val_div_metric": val_div_metric,
                "val_ddm_str": val_ddm_str,
                "pe": pe,
                "p_fcf": p_fcf,
                "peg": peg,
                "texto_ia_final": texto_ia_final,
                "wacc": wacc,
                "g_term": g_term,
                "calcular_dcf_fn": calcular_dcf_fn,
                "inc": inc,
                "cf": cf,
                "perfil_texto": perfil_texto
            }
            pdf_bytes = generar_pdf_reporte(analysis_data)
            st.download_button(
                label="📄 Exportar Reporte Institucional Premium (PDF 3 Páginas)",
                data=pdf_bytes,
                file_name=f"Reporte_Premium_{ticker_input}.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"❌ Ocurrió un error procesando los datos: {e}. Revisa si el ticker me es correcto.")
