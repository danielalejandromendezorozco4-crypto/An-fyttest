# Engine package
# Exportaciones explícitas con manejo defensivo para garantizar resolución
# correcta en Streamlit Cloud, pytest y cualquier entorno de importación.

try:
    from engine.valuation import (
        calcular_wacc,
        calcular_dcf_intr_ps,
        crear_calculador_dcf,
        calcular_ddm,
        calcular_fcff_valuation,
    )

    __all__ = [
        "calcular_wacc",
        "calcular_dcf_intr_ps",
        "crear_calculador_dcf",
        "calcular_ddm",
        "calcular_fcff_valuation",
    ]
except ImportError:
    # Fallback silencioso: los consumidores importan directamente desde
    # engine.valuation (e.g., `from engine.valuation import calcular_fcff_valuation`)
    __all__ = []
