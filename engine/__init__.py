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
    from engine.metrics import (
        calcular_multiplos_valuacion,
        calcular_ratios_rentabilidad,
        calcular_ratios_solvencia,
        calcular_piotroski_fscore,
        calcular_altman_zscore,
        calcular_scoring,
        evaluar_veredicto,
    )

    __all__ = [
        "calcular_wacc",
        "calcular_dcf_intr_ps",
        "crear_calculador_dcf",
        "calcular_ddm",
        "calcular_fcff_valuation",
        "calcular_multiplos_valuacion",
        "calcular_ratios_rentabilidad",
        "calcular_ratios_solvencia",
        "calcular_piotroski_fscore",
        "calcular_altman_zscore",
        "calcular_scoring",
        "evaluar_veredicto",
    ]
except ImportError:
    __all__ = []
