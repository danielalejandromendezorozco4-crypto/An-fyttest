import re
import time
import streamlit as st
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

def _ejecutar_gemini_con_timeout(modelo_ia, prompt, timeout_segundos=10):
    """
    Ejecuta la llamada a la API de Gemini dentro de un sub-hilo con un tiempo límite estricto (timeout).
    """
    def _invocar():
        return modelo_ia.generate_content(prompt)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_invocar)
        try:
            return future.result(timeout=timeout_segundos)
        except FutureTimeoutError:
            raise TimeoutError(f"La consulta a Gemini excedió el tiempo límite de {timeout_segundos}s")

@st.cache_data(ttl=86400)
def obtener_perfil_corporativo(ticker, nombre, sector, industria, gemini_key, mcap, mg_op, roic):
    """
    Obtiene el perfil corporativo estructurado mediante Gemini AI.
    Soporta reintentos, ordenamiento de modelos y timeout para prevenir bloqueos en Streamlit.
    """
    fallback_msg = (
        f"⚠️ **Perfil Corporativo no disponible temporalmente:** La consulta a la IA excedió el tiempo o límite de cuota. "
        f"Las métricas cuantitativas de {nombre} ({ticker}) se encuentran cargadas normalmente en el Dashboard."
    )

    if not gemini_key or not isinstance(gemini_key, str) or not gemini_key.strip():
        return fallback_msg

    prompt = f"""Actúa como un Analista de Equity Research Senior. Genera un Perfil Corporativo y Tesis de Negocio exhaustivo para {nombre} (Ticker: {ticker}), operando en el sector {sector} / {industria}.
    
    Métricas de contexto actuales: Market Cap de ${mcap/1e9:.2f}B USD, Margen Operativo de {mg_op:.1f}%, y ROIC de {roic:.1f}%.
    
    Redacta el análisis estructurado exactamente en los siguientes 5 pilares (usa estos mismos títulos en negritas y con sus emojis correspondientes):
    **🎯 1. Perfil General, Ventaja Competitiva (Economic Moat) y Modelo de Negocio**
    Describe a qué se dedica la empresa y define claramente su ventaja competitiva (Economic Moat) que la protege de sus competidores.
    **⚙️ 2. Modelo de Operación, Monetización y Fuentes de Ingreso**
    Explica cómo gana dinero, el desglose o segmentación de sus ingresos principales, y su diversificación geográfica o de clientes.
    **🏗️ 3. Estrategia de CapEx e Inversiones Recientes**
    Detalla en qué está invirtiendo la empresa (R&D, infraestructura, adquisiciones estratégicas) y cómo asigna su capital.
    **🚀 4. Catalizadores Futuros y Visión Estratégica (3-5 Años)**
    Identifica los planes a futuro, nuevos mercados, o innovaciones tecnológicas que impulsarán su crecimiento.
    **🌱 5. Sostenibilidad (ESG) y Riesgos de Disrupción Operativa**
    Menciona los compromisos ambientales, presiones regulatorias, o riesgos tecnológicos que puedan amenazar su modelo de negocio.
    Reglas de estilo: Sé estrictamente financiero, profesional y cuantitativo. Usa viñetas internamente para facilitar la lectura. No uses metáforas."""
    
    try:
        api_key_limpia = gemini_key.strip().replace('"', '').replace("'", "")
        genai.configure(api_key=api_key_limpia)
        
        try:
            modelos_disponibles = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        except Exception:
            modelos_disponibles = ["models/gemini-1.5-flash", "models/gemini-pro"]

        if modelos_disponibles:
            modelos_ordenados = sorted(modelos_disponibles, key=lambda x: (0 if 'flash' in x else 1 if 'pro' in x else 2))
            
            for m_name in modelos_ordenados:
                if 'vision' in m_name: continue
                
                # Sistema de 2 reintentos por modelo con timeout corto
                for intento in range(2):
                    try:
                        modelo_ia = genai.GenerativeModel(m_name)
                        respuesta = _ejecutar_gemini_con_timeout(modelo_ia, prompt, timeout_segundos=10)
                        if respuesta and respuesta.text:
                            return respuesta.text.strip()
                    except (TimeoutError, Exception):
                        time.sleep(0.3)
                        continue
                        
        return fallback_msg
    except Exception:
        return fallback_msg


def obtener_analisis_macro_ia(ticker_input, nombre, sector, gemini_key):
    """
    Obtiene el análisis macroeconómico y geopolítico vía Gemini AI.
    Reintenta en caso de error y garantiza un retorno neutro por defecto si la API falla.
    """
    texto_ia_fallback = (
        "El análisis macroeconómico asistido por IA no se encuentra disponible en este momento debido a saturación o cuota de la API. "
        "Se aplicará una calificación macroeconómica neutral por defecto."
    )
    macro_score_fallback = 2.5

    if not gemini_key or not isinstance(gemini_key, str) or not gemini_key.strip():
        return texto_ia_fallback, macro_score_fallback

    prompt = f"""Actúa como un analista de renta variable corporativa y estratega macroeconómico experto. Elabora un reporte de investigación institucional (Top-Down) sobre la situación actual de {nombre} (Ticker: {ticker_input}) en el sector {sector}.
    Redacta el análisis estructurado exactamente en los siguientes 4 pilares (usa estos mismos títulos en negritas):
    **1. Contexto Macroeconómico y Factores Externos Clave**
    Analiza la sensibilidad de su modelo de negocio a las tasas de interés actuales, inflación, tipo de cambio y precios de materias primas/insumos.
    **2. Entorno de Noticias y Eventos Recientes**
    Mapea noticias recientes clave, reportes trimestrales recientes, desarrollos del sector y la situación económica de su mercado principal.
    **3. Riesgos Geopolíticos y Cadenas de Suministro**
    Evalúa tensiones comerciales, regulaciones gubernamentales, barreras arancelarias o concentraciones geográficas que amenacen su operación o márgenes.
    **4. Impacto en la Cotización y Veredicto Macro**
    Concluye si el balance general de riesgos externos favorece o penaliza la valoración de la acción a mediano plazo.
    Reglas de estilo: Sé estrictamente financiero, profesional, cuantitativo donde sea posible, no uses metáforas ni lenguaje para minoristas.
    
    IMPORTANTE PARA CALIFICAR: Al final de tu respuesta, en una línea nueva y solitaria, escribe exactamente "PUNTUACION_MACRO: X" donde X es 5 (favorable), 2.5 (neutral) o 0 (desfavorable). NO agregues texto adicional en esa última línea."""

    try:
        api_key_limpia = gemini_key.strip().replace('"', '').replace("'", "")
        genai.configure(api_key=api_key_limpia)
        
        try:
            modelos_disponibles = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        except Exception:
            modelos_disponibles = ["models/gemini-1.5-flash", "models/gemini-pro"]

        if modelos_disponibles:
            modelos_ordenados = sorted(modelos_disponibles, key=lambda x: (0 if 'flash' in x else 1 if 'pro' in x else 2))
            
            for m_name in modelos_ordenados:
                if 'vision' in m_name: continue
                
                for intento in range(2):
                    try:
                        modelo_ia = genai.GenerativeModel(m_name)
                        respuesta_ia = _ejecutar_gemini_con_timeout(modelo_ia, prompt, timeout_segundos=10)
                        
                        if respuesta_ia and respuesta_ia.text:
                            texto_limpio = []
                            macro_score = 2.5
                            lineas = respuesta_ia.text.split('\n')
                            for linea in lineas:
                                if "PUNTUACION_MACRO:" in linea:
                                    match = re.search(r'\d+(\.\d+)?', linea.split(":")[1])
                                    if match:
                                        macro_score = float(match.group(0))
                                else:
                                    texto_limpio.append(linea)
                            
                            texto_ia_final = '\n'.join(texto_limpio).strip()
                            if texto_ia_final:
                                return texto_ia_final, macro_score
                    except (TimeoutError, Exception):
                        time.sleep(0.3)
                        continue

        return texto_ia_fallback, macro_score_fallback
    except Exception:
        return texto_ia_fallback, macro_score_fallback
