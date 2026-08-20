import streamlit as st

def inyectar_estilos():
    st.markdown("""
    <style>
    /* Fondo Global de la Aplicación y Barra Lateral */
    .stApp, section[data-testid="stSidebar"] {
        background-color: #F4F1E8 !important;
        color: #0F172A !important;
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
        color: #0F172A !important;
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
        color: #0F172A !important;
        font-weight: 800 !important;
        font-size: 15px !important;
    }
    section[data-testid="stSidebar"] div[data-baseweb="input"]:focus-within {
        border: 2px solid #D4AF37 !important;
        box-shadow: 0 0 8px rgba(212, 175, 55, 0.3) !important;
    }
    
    /* PREVENCIÓN DE TRUNCAMIENTO Y ALTO CONTRASTE EN TÍTULOS DE MÉTRICAS */
    div[data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] label,
    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricLabel"] span,
    [data-testid="stMetricLabel"] div,
    [data-testid="stMetricLabel"] [data-testid="stMetricLabelText"] {
        font-size: 13px !important; 
        white-space: normal !important; 
        overflow-wrap: break-word !important;
        line-height: 1.3 !important; 
        color: #0F172A !important;
        -webkit-text-fill-color: #0F172A !important;
        font-weight: 800 !important;
        opacity: 1 !important;
    }
    
    div[data-testid="stMetricLabel"] {
        min-height: 2.6em !important;
        display: flex !important;
        align-items: center !important;
    }
    
    div[data-testid="stMetricValue"],
    [data-testid="stMetricValue"] *,
    [data-testid="stMetricValue"] div {
        font-size: 20px !important;
        color: #0F172A !important;
        -webkit-text-fill-color: #0F172A !important;
        font-weight: 800 !important;
        opacity: 1 !important;
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
    /* CONTROL ESTRICTO DE HITBOX Y TRIGGER DE HOVER EN TOOLTIPS */
    /* 1. Desactivar captura de eventos del mouse en contenedores envolventes y texto del label */
    [data-testid="stTooltipHoverTarget"],
    div[data-testid="stTooltipHoverTarget"],
    span[data-testid="stTooltipHoverTarget"],
    [data-testid="stMetricLabel"] [data-testid="stTooltipHoverTarget"] {
        pointer-events: none !important;
        display: inline-flex !important;
        align-items: center !important;
        vertical-align: middle !important;
        min-height: 0 !important;
        padding: 0 !important;
        border: none !important;
    }

    [data-testid="stTooltipHoverTarget"] > *:not([data-testid="stTooltipIcon"]) {
        pointer-events: none !important;
        color: #0F172A !important;
        -webkit-text-fill-color: #0F172A !important;
        font-weight: 800 !important;
        opacity: 1 !important;
    }

    /* 2. El ícono circular (?) es el ÚNICO nodo interactivo con hover activo */
    [data-testid="stTooltipIcon"],
    div[data-testid="stTooltipIcon"],
    span[data-testid="stTooltipIcon"],
    [data-testid="stTooltipHoverTarget"] [data-testid="stTooltipIcon"] {
        pointer-events: auto !important;
        cursor: help !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 16px !important;
        height: 16px !important;
        min-width: 16px !important;
        min-height: 16px !important;
        max-width: 16px !important;
        max-height: 16px !important;
        padding: 0 !important;
        margin: 0 !important;
        border-radius: 50% !important;
        box-sizing: border-box !important;
        position: relative !important;
        vertical-align: middle !important;
        line-height: 1 !important;
    }

    /* 3. Eliminar pseudoelementos o márgenes que expandan artificialmente el área de activación */
    [data-testid="stTooltipIcon"]::before,
    [data-testid="stTooltipIcon"]::after,
    [data-testid="stTooltipHoverTarget"]::before,
    [data-testid="stTooltipHoverTarget"]::after {
        display: none !important;
        content: none !important;
        pointer-events: none !important;
        width: 0 !important;
        height: 0 !important;
    }

    /* 4. Estilo y dimensiones del SVG circular dentro del icono */
    [data-testid="stTooltipIcon"] svg {
        pointer-events: auto !important;
        width: 14px !important;
        height: 14px !important;
        min-width: 14px !important;
        min-height: 14px !important;
        max-width: 14px !important;
        max-height: 14px !important;
        background-color: #0A192F !important;
        border-radius: 50% !important;
        fill: none !important;
        stroke: #FFFFFF !important;
        color: #FFFFFF !important;
        padding: 1px !important;
        opacity: 0.9 !important;
        display: block !important;
        box-sizing: border-box !important;
        margin: 0 !important;
        transition: transform 0.15s ease, opacity 0.15s ease !important;
    }
    [data-testid="stTooltipIcon"]:hover svg {
        opacity: 1 !important;
        transform: scale(1.15);
    }

    /* 5. Recuadro emergente del Tooltip (Popup flotante) */
    html body div[data-baseweb="tooltip"],
    html body div[data-baseweb="tooltip"] div,
    html body [role="tooltip"],
    html body [role="tooltip"] div {
        background-color: #0A192F !important;
        border: 1px solid #D4AF37 !important;
        border-radius: 6px !important;
        pointer-events: none !important;
    }
    html body div[data-baseweb="tooltip"] *,
    html body [role="tooltip"] * {
        color: #FFFFFF !important;
        font-size: 12.5px !important;
        font-weight: 600 !important;
        line-height: 1.5 !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }
    div[data-baseweb="popover"],
    div[data-baseweb="tooltip"] {
        z-index: 999999 !important;
    }
    /* BADGE DE PLUSVALÍA / MINUSVALÍA */
    span.badge-plusvalia.positivo {
        color: #059669 !important;
    }
    span.badge-plusvalia.negativo {
        color: #DC2626 !important;
    }
    
    /* ESTILIZAR LAS PESTAÑAS (TABS) - VISIBILIDAD Y CONTRASTE */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px !important;
        border-bottom: 2px solid #CBD5E1 !important;
    }
    .stTabs [data-baseweb="tab"] {
        height: auto !important;
        padding: 10px 18px !important;
        background-color: #E2E8F0 !important;
        border-radius: 8px 8px 0px 0px !important;
        color: #0F172A !important;
    }
    .stTabs [data-baseweb="tab"] p, 
    .stTabs [data-baseweb="tab"] span, 
    .stTabs [data-baseweb="tab"] div {
        color: #0F172A !important;
        font-weight: 700 !important;
        font-size: 14px !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0A192F !important;
        border-bottom: 3px solid #D4AF37 !important;
    }
    .stTabs [aria-selected="true"] p, 
    .stTabs [aria-selected="true"] span, 
    .stTabs [aria-selected="true"] div {
        color: #FFFFFF !important;
        font-weight: 800 !important;
        font-size: 14px !important;
    }

    /* CONTENEDORES DE ALERTAS (ST.INFO, ST.WARNING, ST.ERROR) - ALTO CONTRASTE INTERIOR */
    div[data-testid="stNotification"],
    div.stAlert {
        border-radius: 8px !important;
        padding: 12px 16px !important;
        font-size: 13.5px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
    }
    div.stAlert p, 
    div.stAlert div, 
    div.stAlert span, 
    div.stAlert strong, 
    div.stAlert b,
    div[data-testid="stNotification"] p,
    div[data-testid="stNotification"] div,
    div[data-testid="stNotification"] span {
        color: #0F172A !important;
        font-weight: 700 !important;
        line-height: 1.5 !important;
        -webkit-text-fill-color: #0F172A !important;
    }
    </style>
    """, unsafe_allow_html=True)
