from data.financial_fetcher import obtener_kd_fmp_fred
from config.settings import safe_get

def calcular_wacc(tasa_libre_riesgo, beta, mcap, total_debt, int_exp, fmp_key, fred_key, tax_rate, ticker):
    """
    Función pura para calcular el WACC (Weighted Average Cost of Capital).
    Retorna un diccionario estructurado con los componentes de costo de capital.
    """
    erp = 5.5
    total_capital = mcap + total_debt
    we, wd = (mcap / total_capital, total_debt / total_capital) if total_capital > 0 else (1.0, 0.0)
    ke = tasa_libre_riesgo + (beta * erp)
    kd_real = obtener_kd_fmp_fred(ticker, fmp_key, fred_key, int_exp, total_debt)
    kd = min(max(kd_real, tasa_libre_riesgo), 15.0)
    wacc = max(min((we * ke) + (wd * kd * (1 - tax_rate)), 15.0), 7.5)
    
    return {
        "ke": ke,
        "kd": kd,
        "wacc": wacc,
        "we": we,
        "wd": wd
    }

def calcular_dcf_intr_ps(wacc_var, g_term_var, flujo_por_accion, g_1_5, precio_actual, eps_ttm, total_cash, total_debt, shares_current):
    """
    Función pura para calcular el Valor Intrínseco por acción usando Flujos de Caja Descontados (DCF).
    Retorna un diccionario estructurado con métricas y alertas de semáforo.
    """
    pv_f_ps = 0
    f_ps_var = flujo_por_accion
    for i in range(1, 6):
        f_ps_var *= (1 + g_1_5)
        pv_f_ps += f_ps_var / ((1 + (wacc_var / 100)) ** i)
    tv_gordon_var = (f_ps_var * (1 + g_term_var)) / ((wacc_var / 100) - g_term_var) if (wacc_var / 100) > g_term_var else 0
    
    pe_dinamico_actual = (precio_actual / eps_ttm) if eps_ttm > 0 else 15
    terminal_pe_var = max(min(pe_dinamico_actual * 0.75, 20.0), 10.0) 
    
    tv_hibrido_var = (tv_gordon_var + (f_ps_var * terminal_pe_var)) / 2
    pv_terminal_var = tv_hibrido_var / ((1 + (wacc_var / 100)) ** 5)
    v_calc = pv_f_ps + pv_terminal_var + ((total_cash - total_debt) / shares_current if shares_current > 0 else 0)
    v_final = v_calc if v_calc > 0 else (precio_actual * 0.85)
    
    es_atractivo = v_final >= precio_actual
    semaforo = "verde" if es_atractivo else "rojo"
    status = "🟢" if es_atractivo else "🔴"
    upside = (((v_final - precio_actual) / precio_actual) * 100) if precio_actual > 0 else 0.0

    return {
        "valor_intrinseco": v_final,
        "status": status,
        "semaforo": semaforo,
        "upside": upside
    }

def crear_calculador_dcf(flujo_por_accion, g_1_5, precio_actual, eps_ttm, total_cash, total_debt, shares_current):
    """
    Factory que retorna una función que devuelve el valor numérico flotante del DCF
    para simplificar iteraciones en matrices de sensibilidad y PDF.
    """
    def calculador(wacc_var, g_term_var):
        res = calcular_dcf_intr_ps(
            wacc_var, g_term_var, flujo_por_accion, g_1_5, precio_actual, eps_ttm, total_cash, total_debt, shares_current
        )
        return res["valor_intrinseco"]
    return calculador

def calcular_ddm(div_rate, ke, g_div, precio_actual=0.0):
    """
    Función pura para calcular el Modelo Gordon Growth (DDM).
    Retorna un diccionario estructurado con métricas y alertas de semáforo.
    """
    v_intr_ddm = (div_rate * (1 + g_div)) / ((ke / 100) - g_div) if (div_rate > 0 and (ke / 100) > g_div) else 0.0
    val_ddm_str = f"${v_intr_ddm:,.2f}" if v_intr_ddm > 0 else "N/A"
    
    if v_intr_ddm == 0:
        semaforo = "gris"
        status = "⚪"
    elif v_intr_ddm >= precio_actual:
        semaforo = "verde"
        status = "🟢"
    else:
        semaforo = "rojo"
        status = "🔴"

    return {
        "valor_intrinseco_ddm": v_intr_ddm,
        "val_ddm_str": val_ddm_str,
        "status": status,
        "semaforo": semaforo
    }
