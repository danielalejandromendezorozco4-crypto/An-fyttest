import streamlit as st

def inyectar_estilos():
    st.markdown("""
    <style>
    /* Fondo Global de la Aplicación y Barra Lateral */
    .stApp, section[data-testid="stSidebar"] {
        background-color: #F4F1E8 !important;
        color: #0A192F !important;
    }
    
    /* Visibilidad del Botón Retráctil del Sidebar (<< / >>) */
    button[data-testid="stSidebarCollapseButton"] {
        background-color: #0A192F !important;
        border: 1px solid #D4AF37 !important;
        border-radius: 6px !important;
    }
    button[data-testid="stSidebarCollapseButton"] svg {
        fill: #D4AF37 !important;
        color: #D4AF37 !important;
    }
    
    /* Legibilidad del Sidebar: Textos, Labels e Inputs */
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] div {
        color: #0A192F !important;
        font-weight: 700 !important;
    }
    
    /* ESTILO DEL CUADRO DE BÚSQUEDA TICKER */
    section[data-testid="stSidebar"] div[data-baseweb="input"] {
        background-color: #FFFFFF !important;
        border: 1px solid #C5A059 !important;
        border-radius: 6px !important;
    }
    section[data-testid="stSidebar"] input {
        background-color: #FFFFFF !important;
        color: #0A192F !important;
        font-weight: 800 !important;
        font-size: 15px !important;
    }
    section[data-testid="stSidebar"] div[data-baseweb="input"]:focus-within {
        border: 2px solid #D4AF37 !important;
        box-shadow: 0 0 8px rgba(212, 175, 55, 0.3) !important;
    }
    
    /* Textos Generales */
    p, label, div {
        color: #0A192F !important;
    }
    
    /* PREVENCIÓN DE TRUNCAMIENTO DE TEXTO (...) EN MÉTRICAS */
    div[data-testid="stMetricLabel"] {
        font-size: 12.5px !important; 
        white-space: normal !important; 
        overflow-wrap: break-word !important;
        line-height: 1.3 !important; 
        min-height: 2.6em !important; 
        color: #0A192F !important;
        font-weight: 700 !important;
        display: -webkit-box !important;
        -webkit-line-clamp: 2 !important; 
        -webkit-box-orient: vertical !important;
        overflow: hidden !important;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 20px !important;
        color: #0A192F !important;
        font-weight: 800 !important;
    }
    
    /* Tipografía e Identidad de Marca An-FyT */
    .brand-title {
        font-family: 'Trebuchet MS', 'Segoe UI', sans-serif;
        font-size: 28px !important;
        font-weight: 800 !important;
        color: #0A192F !important;
        letter-spacing: 1.5px;
        margin-bottom: 2px;
    }
    .brand-subtitle {
        color: #475569 !important;
        font-size: 11px;
        font-weight: 700 !important;
        letter-spacing: 0.8px;
        margin-bottom: 15px;
    }
    /* Tarjetas de Pilares e Introducción */
    .pillar-card {
        background-color: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-top: 3px solid #C5A059;
        border-radius: 8px;
        padding: 16px;
        height: 100%;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
    }
    
    /* Botones Genéricos */
    div.stButton > button {
        background: linear-gradient(135deg, #0A192F 0%, #1E293B 100%) !important;
        border: 1px solid #D4AF37 !important;
        border-radius: 6px !important;
        padding: 8px 14px !important;
        transition: all 0.3s ease !important;
    }
    div.stButton > button p {
        color: #D4AF37 !important;
        font-weight: 800 !important;
        font-size: 14px !important;
    }
    div.stButton > button:hover {
        box-shadow: 0 4px 12px rgba(10, 25, 47, 0.3) !important;
        transform: translateY(-1px);
    }
    /* ESTILO DEL BOTÓN DE EXPORTAR PDF */
    div.stDownloadButton > button {
        background: linear-gradient(135deg, #0A192F 0%, #1E293B 100%) !important;
        border: 2px solid #D4AF37 !important;
        border-radius: 6px !important;
        padding: 12px 20px !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }
    div.stDownloadButton > button p {
        color: #D4AF37 !important;
        font-weight: 800 !important;
        font-size: 16px !important;
    }
    div.stDownloadButton > button:hover {
        box-shadow: 0 4px 15px rgba(212, 175, 55, 0.4) !important;
        transform: translateY(-1px);
    }
   /* TOOLTIP EN st.metric */
    [data-testid="stTooltipIcon"] svg {
        background-color: #0A192F !important;
        border-radius: 50% !important;
        fill: none !important;
        stroke: #FFFFFF !important;
        color: #FFFFFF !important;
        padding: 1px !important;
        opacity: 0.9 !important;
    }
    [data-testid="stTooltipIcon"]:hover svg {
        opacity: 1 !important;
        transform: scale(1.1);
    }
    /* RECUADRO EMERGENTE DEL TOOLTIP */
    html body div[data-baseweb="tooltip"],
    html body div[data-baseweb="tooltip"] div,
    html body [role="tooltip"],
    html body [role="tooltip"] div {
        background-color: #0A192F !important;
        border: 1px solid #D4AF37 !important;
        border-radius: 6px !important;
    }
    html body div[data-baseweb="tooltip"] *,
    html body [role="tooltip"] * {
        color: #FFFFFF !important;
        font-size: 12.5px !important;
        font-weight: 600 !important;
        line-height: 1.5 !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }
    /* BADGE DE PLUSVALÍA / MINUSVALÍA */
    span.badge-plusvalia.positivo {
        color: #059669 !important;
    }
    span.badge-plusvalia.negativo {
        color: #DC2626 !important;
    }
    
    /* ESTILIZAR LAS PESTAÑAS (TABS) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #E2E8F0;
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
        color: #0A192F !important;
        font-weight: 700 !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0A192F !important;
        color: #FFFFFF !important;
        border-bottom: 3px solid #D4AF37 !important;
    }
    </style>
    """, unsafe_allow_html=True)
