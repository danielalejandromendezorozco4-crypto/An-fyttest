import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Custom module imports
from config.settings import (
    SECTOR_BENCHMARKS,
    TOOLTIP_FADE_YEARS,
    cargar_secrets,
    validar_secrets_configurados,
    obtener_ruta_logo,
    safe_get,
    safe_num,
)
from config.styles import inyectar_estilos
from data.sp1500 import obtener_directorio_sp1500, obtener_peers
from engine.valuation import (
    calcular_ddm,
    calcular_wacc,
    calcular_fcff_valuation,
    crear_calculador_dcf,
)
from data.financial_fetcher import (
    FinnhubClient,
    fetch_datos_concurrente,
    obtener_datos_dividendos,
    obtener_noticias_financieras,
    obtener_tasa_fred,
    obtener_erp_mercado,
    extraer_componentes_fcff,
    extraer_fcff_desapalancado,
    extraer_metricas_ttm,
    obtener_rf_tnx,
    obtener_consenso_wall_street,
)
from engine.metrics import (
    calcular_altman_zscore,
    calcular_piotroski_fscore,
    calcular_multiplos_valuacion,
    calcular_ratios_rentabilidad,
    calcular_ratios_solvencia,
    calcular_scoring,
    evaluar_veredicto,
    calcular_buyback_yield,
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

# --- AUTENTICACIÓN Y AUDITORÍA DE SECRETS ---
secrets_actuales = cargar_secrets()
gemini_key, fred_key, finnhub_key, fmp_key = secrets_actuales
finnhub_client = FinnhubClient(api_key=finnhub_key)

diag_secrets = validar_secrets_configurados(secrets_actuales)
if diag_secrets.get("errores"):
    for err in diag_secrets["errores"]:
        st.sidebar.error(f"⚠️ {err}")
if diag_secrets.get("avisos"):
    for av in diag_secrets["avisos"]:
        st.sidebar.info(f"ℹ️ {av}")

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
    peers_list = obtener_peers(ticker_input, max_peers=3, finnhub_client=finnhub_client)
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

# --- SUPUESTOS AVANZADOS DCF (Sidebar Expander) ---
st.sidebar.markdown("---")
with st.sidebar.expander("⚙️ Supuestos Avanzados DCF", expanded=False):
    buyback_rate_ui = st.slider(
        "Recompra Neta Anual (%)",
        min_value=0.0, max_value=10.0, value=0.0, step=0.5,
        help="Porcentaje anual de reducción neta de acciones por recompras. "
             "0% = sin ajuste. Aumenta el valor por acción si > 0.",
    ) / 100.0
    fade_years_ui = st.slider(
        "Años Fade Period",
        min_value=1, max_value=5, value=3, step=1,
        help=TOOLTIP_FADE_YEARS,
    )
    g_term_ui = st.slider(
        "Tasa Terminal g (%)",
        min_value=1.0, max_value=4.0, value=2.5, step=0.25,
        help="Tasa de crecimiento perpetuo del Valor Terminal (Gordon Shapiro). "
             "Recomendado: 2.0–3.0% (alineado al PIB nominal global).",
    ) / 100.0
st.sidebar.markdown("---")

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
            # Extracción concurrente optimizada con caché en sesión para evitar llamadas al mover sliders
            session_cache_key = f"market_data_{ticker_input}"
            if session_cache_key in st.session_state and st.session_state.get("active_ticker") == ticker_input:
                datos_mercado = st.session_state[session_cache_key]
            else:
                datos_mercado = fetch_datos_concurrente(ticker_input, finnhub_key, fred_key, fmp_key=fmp_key)
                st.session_state[session_cache_key] = datos_mercado
                st.session_state["active_ticker"] = ticker_input

            precio_actual = datos_mercado["precio_actual"]
            prev_close = datos_mercado["prev_close"]
            hist = datos_mercado["hist"]
            info_estatica = datos_mercado["info"]
            inc = datos_mercado["inc"]
            bs = datos_mercado["bs"]
            cf = datos_mercado["cf"]
            tasa_libre_riesgo = datos_mercado["tasa_fred"]
            news_data_cache = datos_mercado["news_data"]
            diagnosticos_api = datos_mercado.get("diagnosticos", [])
            
            # Avisos específicos de degradación o límites temporales de API
            for diag in diagnosticos_api:
                st.warning(f"⚠️ {diag.get('detalle', '')}")

            # Fallback defensivo para precio actual si el endpoint quote de Finnhub falló
            if precio_actual == 0.0 and isinstance(info_estatica, dict):
                precio_actual = safe_num(
                    info_estatica.get("stockPrice")
                    or info_estatica.get("price")
                    or info_estatica.get("currentPrice")
                    or (
                        info_estatica.get("marketCap", 0.0) / info_estatica.get("sharesOutstanding", 1.0)
                        if safe_num(info_estatica.get("sharesOutstanding", 0.0)) > 0
                        else 0.0
                    ),
                    0.0
                )

            # Blindaje para histórico si está vacío pero disponemos de precio actual
            if hist.empty and precio_actual > 0:
                today_dt = pd.to_datetime(datetime.date.today())
                hist = pd.DataFrame([{
                    "Open": precio_actual,
                    "High": precio_actual,
                    "Low": precio_actual,
                    "Close": precio_actual,
                    "Volume": 0,
                }], index=pd.DatetimeIndex([today_dt], name="Date"))

            info = info_estatica.copy()
            info["currentPrice"] = precio_actual
            info["previousClose"] = prev_close
            
            # Verificación defensiva elegante: solo detener si NO hay ningún dato fundamental disponible
            if (not info and inc.empty and bs.empty) or (precio_actual == 0.0 and inc.empty and bs.empty):
                st.error(
                    f"❌ No se pudieron obtener estados financieros ni cotización para '{ticker_input}'. "
                    "Verifica que el símbolo sea válido o revisa la conectividad y límites de cuota de tus APIs."
                )
                st.stop()

            # --- EXTRACCIÓN Y NORMALIZACIÓN INSTITUCIONAL TTM ---
            m_ttm = extraer_metricas_ttm(info, inc, bs, cf, precio_actual)
            mcap = safe_num(m_ttm.get("mcap", 0.0), 0.0)
            shares_current = safe_num(m_ttm.get("shares_diluted", 0.0), 0.0)
            if shares_current <= 0:
                shares_current = (mcap / precio_actual) if (mcap > 0 and precio_actual > 0) else 1.0

            net_income_val = safe_num(m_ttm.get("net_income_ttm", 0.0), 0.0)
            op_cash_val = safe_num(m_ttm.get("ocf_ttm", 0.0), 0.0)
            
            # Inicialización defensiva universal de EPS TTM
            eps_ttm = safe_num(m_ttm.get("eps_diluted_ttm", 0.0), 0.0)
            if eps_ttm == 0.0 and shares_current > 0 and net_income_val != 0.0:
                eps_ttm = net_income_val / shares_current

            # EVALUACIÓN DE INTEGRIDAD DE DATOS (Solo marca incompleto si faltan insumos críticos de valuación)
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
            
            # --- MÓDULO 1: SOLVENCIA Y DEUDA (15%) ---
            total_debt = m_ttm["total_debt"]
            total_cash = m_ttm["total_cash"]
            ebitda = m_ttm["ebitda_ttm"]
            ebit = m_ttm["operating_income_ttm"]
            int_exp = m_ttm["interest_expense_ttm"]
            total_eq = m_ttm["total_equity"]
            cur_assets = m_ttm["current_assets"]
            cur_liab = m_ttm["current_liabilities"]
            fcf_ttm = m_ttm["fcf_ttm"]

            res_solv = calcular_ratios_solvencia(
                total_debt=total_debt,
                total_cash=total_cash,
                total_equity=total_eq,
                ebitda_ttm=ebitda,
                ebit_ttm=ebit,
                interest_expense=int_exp,
                current_assets=cur_assets,
                current_liabilities=cur_liab,
                fcf_ttm=fcf_ttm,
                shares_current=shares_current,
                is_fibra_util=is_fibra_util,
                current_ratio_fallback=safe_num(m_ttm.get("current_ratio", 0.0)),
            )

            net_debt = res_solv["net_debt"]
            net_debt_ebitda = res_solv["net_debt_ebitda"]
            val_nde = res_solv["val_nde"]
            col_nde = res_solv["col_nde"]
            msg_nde = res_solv["msg_nde"]

            cob_int = res_solv["cob_int"]
            val_cob = res_solv["val_cob"]
            col_cob = res_solv["col_cob"]
            msg_cob = res_solv["msg_cob"]

            cur_ratio = res_solv["cur_ratio"]
            val_cur = res_solv["val_cur"]
            col_cur = res_solv["col_cur"]
            msg_cur = res_solv["msg_cur"]
            msg_cur_alerta = res_solv["msg_cur_alerta"]

            debt_eq = res_solv["debt_eq"]
            val_de = res_solv["val_de"]
            col_de = res_solv["col_de"]
            msg_de = res_solv["msg_de"]

            fcf_debt_ratio = res_solv["fcf_debt_ratio"]
            col_fcfd = res_solv["col_fcfd"]
            msg_fcfd = res_solv["msg_fcfd"]

            net_cash_per_share = res_solv["net_cash_per_share"]
            val_ncps_str = res_solv["val_ncps_str"]
            col_ncps = res_solv["col_ncps"]
            msg_ncps = res_solv["msg_ncps"]
                
            # --- MÓDULO 2: RENTABILIDAD Y EFICIENCIA (25%) ---
            net_income = m_ttm["net_income_ttm"]
            total_assets_val = m_ttm["total_assets"]
            rev_ttm = m_ttm["revenue_ttm"]
            gross_prof_ttm = m_ttm["gross_profit_ttm"]
            sh_debt = m_ttm["short_term_debt"]
            ebt = m_ttm["pretax_income_ttm"]
            tax_exp = m_ttm["tax_provision_ttm"]
            tax_rate = min(max(tax_exp / ebt, 0.0), 0.35) if ebt > 0 else 0.21

            res_rent = calcular_ratios_rentabilidad(
                revenue_ttm=rev_ttm,
                gross_profit_ttm=gross_prof_ttm,
                operating_income_ttm=ebit,
                net_income_ttm=net_income,
                total_assets=total_assets_val,
                total_equity=total_eq,
                total_debt=total_debt,
                total_cash=total_cash,
                current_liabilities=cur_liab,
                short_term_debt=sh_debt,
                tax_rate=tax_rate,
                is_asset_light=is_asset_light,
                roe_fallback=safe_num(m_ttm.get("roe", 0.0)),
                roa_fallback=safe_num(m_ttm.get("roa", 0.0)),
                roic_fallback=safe_num(m_ttm.get("roic", 0.0)),
            )

            mg_op = res_rent["mg_op"]
            roe = res_rent["roe"]
            val_roe = res_rent["val_roe"]
            col_roe = res_rent["col_roe"]
            msg_roe = res_rent["msg_roe"]

            roa = res_rent["roa"]
            col_roa = res_rent["col_roa"]
            msg_roa = res_rent["msg_roa"]

            roic = res_rent["roic"]
            col_roic = res_rent["col_roic"]
            msg_roic = res_rent["msg_roic"]
            nopat = res_rent["nopat"]
            invested_capital_op = res_rent["invested_capital"]

            divergencia_roa = "⚠️ Alerta de Apalancamiento: Brecha extrema entre ROE y ROA." if (roa > 0 and (roe / roa) > 4.0) else ""
            
            fcf_conv = (fcf_ttm / net_income) if net_income != 0 else 0.0
            if fcf_conv > 1.0: col_fcfc, msg_fcfc = "🟢", "Genera más dinero líquido que ganancia contable."
            elif fcf_conv >= 0.8: col_fcfc, msg_fcfc = "🟢", "Flujo de Caja Sano."
            else: col_fcfc, msg_fcfc = "🔴", "Baja conversión contable a flujo de caja."
            
            alerta_fcf_calidad = f"⚠️ Alerta de Calidad de Flujo: Convierte solo el {fcf_conv*100:.1f}% de utilidades en FCF." if (fcf_conv < 0.8 and net_income > 0) else ""
            earnings_growth_val = m_ttm.get("earnings_growth", 0.0)
            if earnings_growth_val is not None and not pd.isna(earnings_growth_val) and earnings_growth_val != 0.0:
                eps_growth = earnings_growth_val * 100.0 if abs(earnings_growth_val) < 1.0 else earnings_growth_val
                eps_growth_str = f"{eps_growth:.1f}%"
                ni_growth = safe_get(info, ["netIncomeGrowth"], eps_growth / 100.0) * 100.0
                if eps_growth > 10 and ni_growth < 3:
                    col_eps, msg_eps = "🟡", "Alerta: EPS impulsado por recompra de acciones."
                elif eps_growth > 10:
                    col_eps, msg_eps = "🟢", "Crecimiento robusto de beneficios."
                elif eps_growth >= 6:
                    col_eps, msg_eps = "🟡", "Crecimiento moderado."
                elif eps_growth >= 0:
                    col_eps, msg_eps = "🔴", "Crecimiento lento o estancado."
                else:
                    col_eps, msg_eps = "🔴", "Contracción en beneficios por acción (YoY)."
            else:
                eps_growth = 0.0
                eps_growth_str = "N/D"
                col_eps, msg_eps = "⚪", "Crecimiento de EPS no disponible o no reportado."
            # BUYBACK YIELD (Variación interanual de acciones en circulación o recompras netas)
            res_by = calcular_buyback_yield(inc, bs, cf, shares_current=shares_current, mcap=mcap)
            buyback_yield = res_by["buyback_yield"]
            buyback_yield_str = res_by["buyback_yield_str"]
            col_by = res_by["col_by"]
            msg_by = res_by["msg_by"]
            div_rate, div_yield, next_div_date = obtener_datos_dividendos(ticker_input, info, finnhub_key, precio_actual)
            val_div_metric = f"${div_rate:.2f} ({div_yield:.2f}%)" if div_rate > 0 else "N/A"
            msg_div_tooltip = f"Rendimiento anualizado por dividendo: {div_yield:.2f}%\nPróxima fecha ex-dividendo estimada: {next_div_date}"
            fcf_yield = (fcf_ttm / mcap * 100) if mcap > 0 else 0.0
            if fcf_yield >= (tasa_libre_riesgo + 3.0): col_fcfy, msg_fcfy = "🟢", f"Excelente Rendimiento FCF ({fcf_yield:.2f}% vs FRED {tasa_libre_riesgo:.2f}%)."
            elif fcf_yield >= tasa_libre_riesgo: col_fcfy, msg_fcfy = "🟡", f"Rendimiento de Caja Aceptable ({fcf_yield:.2f}% vs FRED {tasa_libre_riesgo:.2f}%)."
            else: col_fcfy, msg_fcfy = "🔴", f"Rendimiento Exigente ({fcf_yield:.2f}% vs FRED {tasa_libre_riesgo:.2f}%)."
            
            # --- MÓDULO 3: VALUACIÓN, FCFF Y DDM (40%) ---
            beta = safe_num(m_ttm.get("beta", 1.0), 1.0)
            rf_tnx = safe_num(obtener_rf_tnx(fallback_fred=tasa_libre_riesgo), 4.20)
            erp_mercado = safe_num(obtener_erp_mercado(fred_key, rf_tnx), 5.0)

            # Extraer partidas FCFF con dual-path (EBIT primario + OCF fallback)
            comp_fcff = extraer_fcff_desapalancado(cf, inc, bs, info)

            # Tasa de crecimiento prospectiva para la Fase 1
            growth_exp = 0.0
            if eps_growth > 0:
                growth_exp = min(eps_growth / 100.0, 0.38) if eps_growth > 38 else (eps_growth / 100.0)
            elif m_ttm.get("revenue_growth", 0.0) > 0:
                growth_exp = m_ttm.get("revenue_growth", 0.0)

            # ── MOTOR FCFF INSTITUCIONAL (Fuente central de verdad) ──
            res_fcff = calcular_fcff_valuation(
                ocf_hist              = comp_fcff["ocf_hist"],
                capex_hist            = comp_fcff["capex_hist"],
                interest_hist         = comp_fcff["interest_hist"],
                pretax_hist           = comp_fcff["pretax_hist"],
                taxprov_hist          = comp_fcff["taxprov_hist"],
                total_debt            = comp_fcff["total_debt"] or total_debt,
                total_cash            = comp_fcff["total_cash"] or total_cash,
                shares_diluted        = comp_fcff["shares_diluted"] or shares_current,
                mcap                  = mcap,
                beta                  = beta,
                rf                    = rf_tnx,
                precio_actual         = precio_actual,
                erp                   = erp_mercado,
                growth_rate_exp       = growth_exp if growth_exp > 0 else None,
                cagr_revenue_hist     = m_ttm.get("cagr_revenue_3_5y", 0.0),
                revenue_growth_api    = m_ttm.get("revenue_growth", 0.0),
                revenue_ttm           = m_ttm.get("revenue_ttm", rev_ttm),
                operating_margin_hist = m_ttm.get("op_margin_hist", mg_op / 100.0 if mg_op > 0 else 0.12),
                finnhub_key           = finnhub_key,
                fred_key              = fred_key,
                ticker                = ticker_input,
                # ── Nuevos parámetros del modelo ──────────────────────────
                buyback_rate          = buyback_rate_ui,
                fade_years            = fade_years_ui,
                g_term_override       = g_term_ui,
                ebit_hist             = comp_fcff.get("ebit_hist", []),
                da_hist               = comp_fcff.get("da_hist", []),
                delta_nwc_hist        = comp_fcff.get("delta_nwc_hist", []),
            )

            # Extraer métricas unificadas del motor FCFF
            wacc        = res_fcff["wacc"]
            ke          = res_fcff["ke"]
            kd          = res_fcff["kd"]
            we          = res_fcff["we"]
            wd          = res_fcff["wd"]
            g_term      = res_fcff["g_term"]
            v_intr_fcff = res_fcff["valor_intrinseco"]
            v_intr_dcf  = v_intr_fcff
            v_intr      = v_intr_dcf
            margen_seguridad = res_fcff["margen_seguridad"]

            # ── Tabla de flujos descontados año a año (fcff_pv_detalle) ──
            pv_detalle = res_fcff.get("fcff_pv_detalle", [])
            if pv_detalle:
                metodo_fcff = res_fcff.get("fcff_method", "ocf")
                metodo_label = {"ebit": "EBIT Primario (desapalancado)", "ocf": "OCF + Escudo Fiscal", "normalizado": "FCFF Normalizado", "sin_datos": "Sin datos"}.get(metodo_fcff, metodo_fcff)
                st.info(f"**Método FCFF:** {metodo_label} | **Acciones post-recompras:** {res_fcff.get('shares_efectivas', 0):,.0f}")
                with st.expander("📋 Detalle Flujos Descontados (Mid-Year Convention)", expanded=False):
                    import pandas as _pd
                    df_pv = _pd.DataFrame(pv_detalle)
                    st.dataframe(df_pv, use_container_width=True, hide_index=True)
                    st.caption(f"Horizonte: {res_fcff.get('n_total', 8)} años · WACC: {res_fcff.get('wacc', 0):.2f}% · g_term: {res_fcff.get('g_term', 0)*100:.2f}% · Buyback: {res_fcff.get('buyback_rate', 0)*100:.1f}%/año")

            # ── Factory de Sensibilidad DCF ──
            fcff_base = res_fcff.get("fcff_base", 0.0)
            flujo_por_accion = (fcff_base / shares_current) if (shares_current > 0 and fcff_base > 0) else (
                max((fcf_ttm / shares_current) if shares_current > 0 else 0.0, eps_ttm * 0.85)
            )
            g_1_5 = res_fcff.get("g_fase1", 0.08)
            calcular_dcf_fn = crear_calculador_dcf(flujo_por_accion, g_1_5, precio_actual, eps_ttm, total_cash, total_debt, shares_current)

            g_div = min(max(g_1_5 * 0.5, 0.015), 0.04)
            res_ddm = calcular_ddm(div_rate, ke, g_div, precio_actual)
            v_intr_ddm = res_ddm["valor_intrinseco_ddm"]
            val_ddm_str = res_ddm["val_ddm_str"]
            col_ddm = res_ddm["status"]
            mostrar_ddm = (div_rate > 0 and (div_yield >= 1.5 or is_fibra_util))
                
            # ── Múltiplos de Valuación Estandarizados (Finviz / Yahoo Finance) ──
            res_mult = calcular_multiplos_valuacion(
                precio_actual=precio_actual,
                mcap=mcap,
                eps_ttm=eps_ttm,
                forward_eps=m_ttm["forward_eps"],
                fcf_ttm=fcf_ttm,
                ebitda_ttm=ebitda,
                total_debt=total_debt,
                total_cash=total_cash,
                revenue_ttm=rev_ttm,
                total_equity=total_eq,
                peg_info=m_ttm["peg_ratio_info"],
                earnings_growth=m_ttm["earnings_growth"],
                buyback_yield=buyback_yield,
            )

            pe = res_mult["pe"]
            pe_str = res_mult["pe_str"]
            col_pe = res_mult["col_pe"]
            msg_pe = res_mult["msg_pe"]

            p_fcf = res_mult["p_fcf"]
            p_fcf_str = res_mult["p_fcf_str"]
            col_pfcf = res_mult["col_pfcf"]
            msg_pfcf = res_mult["msg_pfcf"]

            ev_ebitda = res_mult["ev_ebitda"]
            ev_ebitda_str = res_mult["ev_ebitda_str"]
            col_ev = res_mult["col_ev"]
            msg_ev = res_mult["msg_ev"]

            peg = res_mult["peg"]
            peg_str = res_mult["peg_str"]
            col_peg = res_mult["col_peg"]
            msg_peg = res_mult["msg_peg"]

            p_s = res_mult["p_s"]
            # --- CONSENSO DE WALL STREET (Finnhub Oficial + Fallback FMP) ---
            target = safe_num(
                info.get("targetMeanPrice")
                or info.get("targetMean")
                or info.get("targetMedianPrice")
                or info.get("targetPrice")
                or m_ttm.get("target_mean_price", 0.0),
                0.0
            )
            if target <= 0.0:
                try:
                    consenso_res = obtener_consenso_wall_street(
                        ticker_input,
                        finnhub_api_key=finnhub_key,
                        finnhub_client=finnhub_client,
                        fmp_api_key=fmp_key
                    )
                    target = safe_num(consenso_res.target_mean, 0.0)
                except Exception as e_ws:
                    logger.debug("Error extrayendo consenso para %s: %s", ticker_input, e_ws)
                    target = 0.0

            if target > 0 and precio_actual > 0:
                upside = ((target - precio_actual) / precio_actual) * 100.0
                col_upside = "🟢" if upside >= 5.0 else ("🟡" if upside >= -5.0 else "🔴")
                val_target_str = f"${target:,.2f}"
                delta_target_str = f"{upside:+.1f}% vs Mercado"
            else:
                upside = 0.0
                col_upside = "⚪"
                val_target_str = "N/D"
                delta_target_str = None
            col_vintr = "🟢" if v_intr_dcf >= precio_actual else "🔴"
            
            # --- MÓDULO 4: RIESGOS Y SALUD CONTABLE (15%) ---
            # Altman Z-Score: FMP /financial-score con fallback resiliente al cálculo manual
            altman_z_fmp = safe_num(info.get("altmanZScore"), 0.0)
            if altman_z_fmp > 0.0:
                z_score = altman_z_fmp
                if z_score > 2.99:
                    col_z, msg_z = "🟢", "Riesgo de bancarrota casi nulo."
                elif z_score >= 1.81:
                    col_z, msg_z = "🟡", "Precaución: Zona Gris."
                else:
                    col_z, msg_z = "🔴", "Alto riesgo de insolvencia."
            else:
                res_z = calcular_altman_zscore(debt_eq, roa)
                z_score, col_z, msg_z = res_z["z_score"], res_z["status"], res_z["msg_z"]

            short_int = m_ttm["short_percent_of_float"] * 100.0 if m_ttm["short_percent_of_float"] < 1.0 else m_ttm["short_percent_of_float"]
            
            col_b, msg_b = ("🟢", "Volatilidad baja (Defensiva).") if beta < 0.8 else (("🟡", "Volatilidad moderada.") if beta <= 1.4 else ("🔴", "Alta volatilidad sistémica."))
            col_s, msg_s = ("🟢", "Bajo interés en corto.") if short_int < 5 else (("🟡", "Posicionamiento en corto moderado.") if short_int <= 10 else ("🔴", "Fuerte pesimismo en corto."))
            
            # Piotroski F-Score: FMP con fallback resiliente al cálculo manual
            piotroski_fmp = info.get("piotroskiScore")
            if piotroski_fmp is not None and safe_num(piotroski_fmp, -1) >= 0:
                f_score = int(safe_num(piotroski_fmp, 0))
                fscore_str = f"{f_score}/9"
                if f_score >= 7:
                    col_fscore, msg_fscore = "🟢", "Salud contable sólida y de alta calidad."
                elif f_score >= 4:
                    col_fscore, msg_fscore = "🟡", "Salud contable promedio."
                else:
                    col_fscore, msg_fscore = "🔴", "Riesgo de deterioro contable."
            else:
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
            
            doble_filtro = "🟢 Oportunidad de Alta Confianza (Cotiza por debajo de tu Precio Máx de Compra y Wall Street le ve alto potencial)." if (precio_actual <= precio_max_compra and target > 0 and upside > 15) else ("🟡 Valor Oculto" if (v_intr > precio_actual and target > 0 and upside > 0) else "⚪ Valuación Justa o Mixta")

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
            if target > 0:
                h_ws = f"¿Qué es? Precio Objetivo Consenso (12 meses).\n¿Para qué sirve? Indica la valoración promedio proyectada por analistas de Wall Street.\nDiagnóstico: Precio objetivo de {val_target_str} USD con un potencial (upside) esperado del {upside:+.1f}% respecto al precio actual."
            else:
                h_ws = "¿Qué es? Precio Objetivo Consenso (12 meses).\n¿Para qué sirve? Indica la valoración promedio proyectada por analistas de Wall Street.\nDiagnóstico: Sin cobertura de analistas disponible para este activo."
            
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
            
            if buyback_yield > 0:
                diag_by_m3 = f"La empresa reduce sus acciones en circulación al {buyback_yield:.1f}% anual, incrementando el beneficio por acción (EPS)."
            elif buyback_yield < 0:
                diag_by_m3 = f"Existe una emisión neta o dilución de acciones del {abs(buyback_yield):.1f}% anual."
            else:
                diag_by_m3 = "Sin recompras netas significativas registradas en el período (0.0%)."

            h_by_m3 = (
                "¿Qué es? Recompra de Acciones / Buyback Yield.\n"
                "¿Para qué sirve? Mide el retorno de capital al accionista mediante la compra de acciones propias en el mercado, "
                "reduciendo el número de acciones en circulación y aumentando el beneficio por acción (EPS).\n"
                f"Diagnóstico: {diag_by_m3}"
            )
            
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
                c9.metric(f"{col_eps} EPS YoY", eps_growth_str, help=h_eps)
                c9_by.metric(f"{col_by} Buyback Yield", f"{buyback_yield:.1f}%", help=h_by)
                c9_div.metric("💵 Dividendos", val_div_metric, help=msg_div_tooltip)
                
                # --- GRÁFICOS MÓDULO 2 ---
                g_col1, g_col2 = st.columns(2)
                
                with g_col1:
                    st.write("**Comparativa de Márgenes (5 Años)**")
                    posibles_rev = ['Total Revenue', 'Operating Revenue', 'Revenue', 'TotalRevenue']
                    posibles_op = ['Operating Income', 'OperatingIncome', 'Operating Profit', 'EBIT']
                    posibles_ni = ['Net Income', 'NetIncome', 'Net Income Common Stockholders']

                    row_rev = next((r for r in posibles_rev if r in inc.index), None)
                    row_op = next((r for r in posibles_op if r in inc.index), None)
                    row_ni = next((r for r in posibles_ni if r in inc.index), None)

                    if not inc.empty and row_rev and row_op and row_ni:
                        try:
                            # Filtrar columnas anuales numéricas (hasta 5 años)
                            cols_anuales = sorted([c for c in inc.columns if str(c).isdigit() and len(str(c)) == 4])
                            if len(cols_anuales) > 5:
                                cols_anuales = cols_anuales[-5:]
                            if not cols_anuales:
                                cols_anuales = [c for c in inc.columns if c not in ["MRQ", "TTM"]]
                            if not cols_anuales:
                                cols_anuales = list(inc.columns)

                            s_rev = inc.loc[row_rev, cols_anuales]
                            if isinstance(s_rev, pd.DataFrame): s_rev = s_rev.iloc[0]
                            s_op = inc.loc[row_op, cols_anuales]
                            if isinstance(s_op, pd.DataFrame): s_op = s_op.iloc[0]
                            s_ni = inc.loc[row_ni, cols_anuales]
                            if isinstance(s_ni, pd.DataFrame): s_ni = s_ni.iloc[0]

                            rev_seguro = s_rev.replace(0, np.nan)
                            df_margins = pd.DataFrame({
                                "Margen Operativo (%)": (s_op / rev_seguro) * 100,
                                "Margen Neto (%)": (s_ni / rev_seguro) * 100
                            }, index=cols_anuales).dropna()
                            
                            df_margins.index = [str(x) for x in df_margins.index]
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
                    else:
                        st.write("Gráfico de márgenes no disponible para este ticker.")
                with g_col2:
                    st.write("**Comparativa de Flujos de Efectivo (5 Años - Millones USD)**")
                    posibles_ocf = ['Operating Cash Flow', 'OperatingCashFlow', 'Cash Flow From Continuing Operating Activities', 'Total Cash From Operating Activities', 'operatingCashFlow', 'netCashProvidedByOperatingActivities']
                    posibles_fcf = ['Free Cash Flow', 'FreeCashFlow', 'freeCashFlow']
                    posibles_capex = ['Capital Expenditure', 'CapitalExpenditure', 'Capital Expenditures', 'Purchase Of Property Plant And Equipment', 'capitalExpenditure', 'investmentsInPropertyPlantAndEquipment']

                    row_ocf = next((r for r in posibles_ocf if r in cf.index), None)
                    row_fcf = next((r for r in posibles_fcf if r in cf.index), None)
                    row_capex = next((r for r in posibles_capex if r in cf.index), None)

                    if not cf.empty and (row_ocf or row_fcf):
                        try:
                            # Filtrar columnas anuales numéricas (hasta 5 años)
                            cols_anuales_cf = sorted([c for c in cf.columns if str(c).isdigit() and len(str(c)) == 4])
                            if len(cols_anuales_cf) > 5:
                                cols_anuales_cf = cols_anuales_cf[-5:]
                            if not cols_anuales_cf:
                                cols_anuales_cf = [c for c in cf.columns if c not in ["MRQ", "TTM"]]
                            if not cols_anuales_cf:
                                cols_anuales_cf = list(cf.columns)

                            s_ocf = cf.loc[row_ocf, cols_anuales_cf] if row_ocf else pd.Series(0.0, index=cols_anuales_cf)
                            if isinstance(s_ocf, pd.DataFrame): s_ocf = s_ocf.iloc[0]

                            if row_capex:
                                s_capex = cf.loc[row_capex, cols_anuales_cf].abs()
                                if isinstance(s_capex, pd.DataFrame): s_capex = s_capex.iloc[0]
                            else:
                                s_capex = pd.Series(0.0, index=cols_anuales_cf)

                            if row_fcf:
                                s_fcf = cf.loc[row_fcf, cols_anuales_cf]
                                if isinstance(s_fcf, pd.DataFrame): s_fcf = s_fcf.iloc[0]
                                if not row_capex and row_ocf:
                                    s_capex = (s_ocf.fillna(0.0) - s_fcf.fillna(0.0)).abs()
                            elif row_capex and row_ocf:
                                s_fcf = s_ocf.fillna(0.0) - s_capex.fillna(0.0)
                            else:
                                s_fcf = s_ocf

                            df_cf = pd.DataFrame({
                                "Flujo Operativo (OCF)": s_ocf / 1e6,
                                "CapEx (Inversión Capital)": s_capex / 1e6,
                                "Flujo Libre (FCF)": s_fcf / 1e6
                            }, index=cols_anuales_cf).dropna(how='all')
                            
                            df_cf.index = [str(x) for x in df_cf.index]
                            df_cf = df_cf.sort_index()
                            fig_cf = px.bar(df_cf, barmode='group', text_auto=',.0f', color_discrete_sequence=['#0284C7', '#F59E0B', '#059669'])
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
                    else:
                        st.write("Gráfico de flujos no disponible para este ticker.")
                st.markdown("### 🏷️ Módulo 3: Valuación y Valor Intrínseco (40%)")

                upside_vs_precio = ((v_intr_dcf - precio_actual) / precio_actual * 100.0) if precio_actual > 0 else 0.0
                col_vintr = "🟢" if v_intr_dcf >= precio_actual else "🔴"

                if mostrar_ddm:
                    row1_c1, row1_c2, row1_c3, row1_c4 = st.columns(4)
                    row1_c1.metric(f"{col_vintr} V. Intrínseco (FCFF)", f"${v_intr_dcf:,.2f}", f"{upside_vs_precio:+.1f}% vs Mercado", help=h_vint)
                    row1_c2.metric(f"{col_ddm} V. Intrínseco (DDM)", val_ddm_str, help=h_ddm)
                    row1_c3.metric(f"{col_pmax} P. Máx Compra", f"${precio_max_compra:,.2f}", f"-{desc_req*100:.0f}% Descuento", help=h_pmax)
                    row1_c4.metric(f"{col_upside} Consenso W.St", val_target_str, delta_target_str, help=h_ws)
                else:
                    row1_c1, row1_c2, row1_c3 = st.columns(3)
                    row1_c1.metric(f"{col_vintr} V. Intrínseco (FCFF)", f"${v_intr_dcf:,.2f}", f"{upside_vs_precio:+.1f}% vs Mercado", help=h_vint)
                    row1_c2.metric(f"{col_pmax} P. Máx Compra", f"${precio_max_compra:,.2f}", f"-{desc_req*100:.0f}% Descuento", help=h_pmax)
                    row1_c3.metric(f"{col_upside} Consenso W.St", val_target_str, delta_target_str, help=h_ws)

                st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
                row2_c1, row2_c2, row2_c3, row2_c4, row2_c5 = st.columns(5)
                row2_c1.metric(f"{col_pe} PER (P/E)", pe_str, help=h_pe)
                row2_c2.metric(f"{col_pfcf} P/FCF", p_fcf_str, help=h_pfcf)
                row2_c3.metric(f"{col_peg} PEG Forward", peg_str, help=h_peg)
                row2_c4.metric(f"{col_ev} EV / EBITDA", ev_ebitda_str, help=h_ev)
                row2_c5.metric(f"{col_by} Recompra Acciones", buyback_yield_str, help=h_by_m3)

                if mostrar_ddm:
                    explicacion_modelo = (
                        f"💡 **Análisis Comparativo de Valuación (FCFF vs. DDM):**\n\n"
                        f"* **Modelo FCFF:** Valúa el 100% de la caja libre generada por {nombre} disponible para todos los proveedores de capital (deuda + equity), descontada al WACC ({wacc:.2f}%) y reconciliada vía puente EV → Equity.\n"
                        f"* **Modelo DDM (Gordon Growth):** Valúa la corriente directa de dividendos en efectivo distribuidos al accionista (Yield: {div_yield:.2f}%).\n\n"
                        f"**Criterio:** Al ser una empresa madura o de alto dividendo, ambos modelos ofrecen una perspectiva complementaria."
                    )
                else:
                    explicacion_modelo = (
                        fr"💡 **Criterio de Valuación Seleccionado (FCFF):**" + "\n\n"
                        fr"Para **{nombre}**, el **Modelo de Flujo de Caja Libre para la Firma (FCFF)** descuenta los flujos de caja operativos netos de CapEx con convención de medio año al WACC empírico ({wacc:.2f}%), incorporando el valor terminal de Gordon Shapiro ($g_{{term}} = {g_term*100:.1f}\%$) y reconciliando mediante el puente Enterprise Value → Equity Value por acción."
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
                news_data = news_data_cache if news_data_cache else obtener_noticias_financieras(ticker_input, finnhub_key)
                
                if news_data:
                    for n in news_data:
                        link_href = n['link'] if n['link'] and n['link'].startswith('http') else f"https://finnhub.io/"
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
                if val <= 0:
                    val_str = "N/D"
                else:
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
                "ev_ebitda": ev_ebitda,
                "p_s": p_s,
                "p_b": p_b,
                "gross_margin": res_rent.get("gross_margin", 0.0),
                "net_margin": res_rent.get("net_margin", 0.0),
                "texto_ia_final": texto_ia_final,
                "wacc": wacc,
                "g_term": g_term,
                "calcular_dcf_fn": calcular_dcf_fn,
                "inc": inc,
                "cf": cf,
                "perfil_texto": perfil_texto,
                # ── Campos FCFF / Valuación Adaptativa para PDF ──────────
                "fcff_wacc":      res_fcff.get("wacc", wacc),
                "fcff_ke":        res_fcff.get("ke", ke),
                "fcff_kd":        res_fcff.get("kd", kd),
                "fcff_rf":        res_fcff.get("rf", rf_tnx),
                "fcff_erp":       res_fcff.get("erp", erp_mercado),
                "fcff_we":        res_fcff.get("we", 0.90),
                "fcff_wd":        res_fcff.get("wd", 0.10),
                "fcff_tax_rate":  res_fcff.get("tax_rate_real", 0.21),
                "fcff_ev":        res_fcff.get("enterprise_value", 0.0),
                "fcff_equity":    res_fcff.get("equity_value", 0.0),
                "fcff_margen":    res_fcff.get("margen_seguridad", 0.0),
                "fcff_historico": res_fcff.get("fcff_historico", []),
                "fcff_g_term":    res_fcff.get("g_term", 0.025),
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
