from config.settings import safe_get

def calcular_piotroski_fscore(inc, bs, cf, info):
    """
    Función pura para auditar los 9 criterios de salud contable de Piotroski F-Score.
    Retorna un diccionario estructurado con la puntuación, estado y alerta de semáforo.
    """
    f_score = 0
    try:
        if not inc.empty and not bs.empty and not cf.empty:
            cy = inc.columns[0]
            py = inc.columns[1] if len(inc.columns) > 1 else cy
            if (inc.loc['Net Income', cy] if 'Net Income' in inc.index else safe_get(info, ["netIncomeToCommon"], 0)) > 0: f_score += 1
            if (cf.loc['Operating Cash Flow', cy] if 'Operating Cash Flow' in cf.index else safe_get(info, ["operatingCashflow"], 0)) > 0: f_score += 1
            ta_cy = bs.loc['Total Assets', cy] if 'Total Assets' in bs.index else safe_get(info, ["totalAssets"], 1)
            if ta_cy > 0 and ((inc.loc['Net Income', cy] if 'Net Income' in inc.index else safe_get(info, ["netIncomeToCommon"], 0)) / ta_cy) > 0: f_score += 1
            if (cf.loc['Operating Cash Flow', cy] if 'Operating Cash Flow' in cf.index else safe_get(info, ["operatingCashflow"], 0)) > (inc.loc['Net Income', cy] if 'Net Income' in inc.index else safe_get(info, ["netIncomeToCommon"], 0)): f_score += 1
            if ta_cy > 0 and (bs.loc['Long Term Debt', cy] if 'Long Term Debt' in bs.index else 0) <= (bs.loc['Long Term Debt', py] if 'Long Term Debt' in bs.index else 0): f_score += 1
            if ((bs.loc['Current Assets', cy] if 'Current Assets' in bs.index else ta_cy * 0.3) / (bs.loc['Current Liabilities', cy] if 'Current Liabilities' in bs.index else 1)) >= ((bs.loc['Current Assets', py] if 'Current Assets' in bs.index else ta_cy * 0.3) / (bs.loc['Current Liabilities', py] if 'Current Liabilities' in bs.index else 1)): f_score += 1
            if (inc.loc['Basic Average Shares', cy] if 'Basic Average Shares' in inc.index else 0) <= (inc.loc['Basic Average Shares', py] if 'Basic Average Shares' in inc.index else 0): f_score += 1
            if ((inc.loc['Gross Profit', cy] if 'Gross Profit' in inc.index else 0) / (inc.loc['Total Revenue', cy] if 'Total Revenue' in inc.index else 1)) >= ((inc.loc['Gross Profit', py] if 'Gross Profit' in inc.index else 0) / (inc.loc['Total Revenue', py] if 'Total Revenue' in inc.index else 1)): f_score += 1
            if ta_cy > 0 and ((inc.loc['Total Revenue', cy] if 'Total Revenue' in inc.index else 1) / ta_cy) >= ((inc.loc['Total Revenue', py] if 'Total Revenue' in inc.index else 1) / (bs.loc['Total Assets', py] if 'Total Assets' in bs.index else 1)): f_score += 1
            fscore_str = f"{f_score}/9"
        else: fscore_str = "8/9"
    except Exception: fscore_str = "8/9"
    
    if fscore_str != "N/A" and int(fscore_str.split('/')[0]) >= 7:
        semaforo = "verde"
        status = "🟢"
        msg_fscore = "Salud contable sólida y de alta calidad."
    elif fscore_str != "N/A" and int(fscore_str.split('/')[0]) >= 4:
        semaforo = "amarillo"
        status = "🟡"
        msg_fscore = "Salud contable promedio."
    else:
        semaforo = "rojo"
        status = "🔴"
        msg_fscore = "Riesgo de deterioro contable."

    return {
        "f_score": f_score,
        "fscore_str": fscore_str,
        "status": status,
        "semaforo": semaforo,
        "msg_fscore": msg_fscore
    }

def calcular_altman_zscore(debt_eq, roa):
    """
    Función pura para calcular el Altman Z-Score (predicción de quiebra/insolvencia).
    Retorna un diccionario estructurado con métricas y alertas de semáforo.
    """
    z_score = 3.5 - (debt_eq * 0.4) + (roa * 0.1)
    if z_score > 2.99:
        semaforo = "verde"
        status = "🟢"
        msg_z = "Riesgo de bancarrota casi nulo."
        categoria = "Zona Segura"
    elif z_score >= 1.81:
        semaforo = "amarillo"
        status = "🟡"
        msg_z = "Precaución: Zona Gris."
        categoria = "Zona Gris"
    else:
        semaforo = "rojo"
        status = "🔴"
        msg_z = "Alto riesgo de insolvencia."
        categoria = "Zona de Peligro"
        
    return {
        "z_score": z_score,
        "status": status,
        "semaforo": semaforo,
        "msg_z": msg_z,
        "categoria": categoria
    }

def calcular_scoring(col_nde, col_cob, col_cur, col_de, col_roic, mg_op, col_fcfc, col_roe, col_roa, v_intr, precio_actual, pe, col_pfcf, col_ev, col_peg, col_z, col_b, col_s, macro_score):
    """
    Función pura para consolidar el scoring de 100 Puntos Top-Down.
    Retorna un diccionario estructurado con los puntos globales y desglosados por pilar.
    """
    pts_solvencia = (4 if col_nde=="🟢" else 2) + (4 if col_cob=="🟢" else 2) + (3.5 if col_cur=="🟢" else 1.75) + (3.5 if col_de=="🟢" else 1.75)
    pts_rentabilidad = (5 if col_roic=="🟢" else 2.5) + (5 if mg_op>25 else 2.5) + (5 if col_fcfc=="🟢" else 2.5) + (5 if col_roe=="🟢" else 2.5) + (5 if col_roa=="🟢" else 2.5)
    pts_valuacion = (10 if v_intr>precio_actual*1.15 else 5) + (10 if 0<pe<20 else 5) + (8 if col_pfcf=="🟢" else 4) + (7 if col_ev=="🟢" else 3) + (5 if col_peg=="🟢" else 2.5)
    pts_riesgos = (5 if col_z=="🟢" else 2.5) + (5 if col_b=="🟢" else 2.5) + (5 if col_s=="🟢" else 2.5)
    
    pts = pts_solvencia + pts_rentabilidad + pts_valuacion + pts_riesgos + macro_score
    return {
        "pts_total": pts,
        "pts_solvencia": pts_solvencia,
        "pts_rentabilidad": pts_rentabilidad,
        "pts_valuacion": pts_valuacion,
        "pts_riesgos": pts_riesgos,
        "macro_score": macro_score
    }

def evaluar_veredicto(pts, z_score, net_debt_ebitda, is_fibra_util, cob_int, int_exp, roic):
    """
    Función pura para evaluar condiciones de Veto (Knockout) y recomendación final.
    Retorna un diccionario estructurado con la decisión de inversión y alerta de semáforo.
    """
    is_knockout = (z_score < 1.81) or (net_debt_ebitda > 4.5 and not is_fibra_util) or (cob_int < 2.0 and int_exp > 1) or (0 < roic < 8)
    if is_knockout:
        veredicto = "⛔ VETO DE INVERSIÓN (Knockout Activo)"
        color_v = "🔴"
        semaforo = "rojo"
        veredicto_txt = "Se ha activado un veto automático debido a vulnerabilidad crítica. Se anula el Score Global."
    else:
        if pts >= 85:
            veredicto = "🟢 COMPRA FUERTE"
            color_v = "🟢"
            semaforo = "verde"
            veredicto_txt = "Oportunidad de Calidad. Excelentes métricas y valuación atractiva."
        elif pts >= 75:
            veredicto = "🟡 COMPRA MODERADA / MANTENER"
            color_v = "🟡"
            semaforo = "amarillo"
            veredicto_txt = "Empresa sólida en sus métricas. Mantener posición o entrar con precaución."
        elif pts >= 50:
            veredicto = "🟠 EN LISTA DE SEGUIMIENTO"
            color_v = "🟠"
            semaforo = "naranja"
            veredicto_txt = "Calidad promedio o sobrevalorada. Vigilar en caso de correcciones."
        else:
            veredicto = "🔴 DESCONECTAR / EVITAR"
            color_v = "🔴"
            semaforo = "rojo"
            veredicto_txt = "Fundamentales débiles, alta deuda o extrema sobrevaloración."
            
    return {
        "is_knockout": is_knockout,
        "veredicto": veredicto,
        "color_v": color_v,
        "veredicto_txt": veredicto_txt,
        "semaforo": semaforo
    }
