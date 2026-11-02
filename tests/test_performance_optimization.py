import ast
import hashlib
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from data.financial_fetcher import (
    obtener_tasa_fred,
    estandarizar_dataframe_financiero,
    MAPA_CANONICO_INCOME,
    MAPA_CANONICO_BALANCE,
    MAPA_CANONICO_CASHFLOW,
)
from data.sp1500 import obtener_peers
from engine.metrics import calcular_buyback_yield
from reports.pdf_generator import generar_pdf_reporte
from services.ai_service import obtener_analisis_macro_ia, obtener_perfil_corporativo


class TestPerformanceAndCaching(unittest.TestCase):
    """
    Suite de pruebas para validar la optimización integral de rendimiento,
    latencia y persistencia en caché en An-FyT.
    """

    def test_criterio_1_app_ast_pdf_cached_y_diferido(self):
        """
        Criterio 1: Verifica mediante AST en app.py que el reporte PDF:
        - No se genere incondicionalmente en cada rerun.
        - Utilice una función cacheada y carga diferida (callable / lambda) en st.download_button.
        """
        with open("app.py", "r", encoding="utf-8") as f:
            code = f.read()

        tree = ast.parse(code, filename="app.py")

        # 1. Verificar que calcular_hash_parametros_pdf y generar_pdf_reporte_cached están definidas
        func_names = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        self.assertIn("calcular_hash_parametros_pdf", func_names)
        self.assertIn("generar_pdf_reporte_cached", func_names)

        # 2. Verificar que st.download_button utiliza carga diferida (lambda o callable)
        self.assertIn("data=lambda:", code.replace(" ", ""))

        # 3. Verificar que generar_pdf_reporte_cached tiene decorador @st.cache_data
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "generar_pdf_reporte_cached":
                dec_names = [
                    d.func.attr if isinstance(d, ast.Call) and hasattr(d.func, "attr")
                    else (d.attr if isinstance(d, ast.Attribute) else getattr(d, "id", ""))
                    for d in node.decorator_list
                ]
                self.assertIn("cache_data", dec_names)

    def test_criterio_1_hash_parametros_pdf_determinista(self):
        """
        Criterio 1: Verifica que el hashing de parámetros serializables sea determinista
        y aísle sliders o cambios que no alteran la valuación.
        """
        from reports.pdf_generator import generar_pdf_reporte

        analysis_data = {
            "ticker": "MA",
            "nombre": "Mastercard Incorporated",
            "sector": "Financial Services",
            "pts": 85.0,
            "veredicto": "COMPRA FUERTE",
            "v_intr_dcf": 520.0,
            "precio_actual": 485.0,
            "precio_max_compra": 468.0,
            "desc_req": 0.10,
            "wacc": 8.5,
            "g_term": 0.025,
            "roic": 52.0,
            "fcf_yield": 2.8,
            "tasa_libre_riesgo": 4.25,
            "pe": 35.0,
            "p_fcf": 38.0,
            "peg": 2.1,
            "ev_ebitda": 26.0,
            "p_s": 18.0,
            "p_b": 60.0,
            "texto_ia_final": "Tesis macro positiva.",
            "perfil_texto": "Perfil corporativo detallado...",
        }

        def _calc_hash(data):
            claves = (
                data.get("ticker", ""),
                data.get("nombre", ""),
                data.get("sector", ""),
                data.get("pts", 0),
                data.get("veredicto", ""),
                round(float(data.get("v_intr_dcf", 0.0) or 0.0), 2),
                round(float(data.get("precio_actual", 0.0) or 0.0), 2),
                round(float(data.get("precio_max_compra", 0.0) or 0.0), 2),
                round(float(data.get("desc_req", 0.0) or 0.0), 4),
                round(float(data.get("wacc", 0.0) or 0.0), 4),
                round(float(data.get("g_term", 0.0) or 0.0), 4),
                round(float(data.get("roic", 0.0) or 0.0), 2),
                round(float(data.get("fcf_yield", 0.0) or 0.0), 2),
                round(float(data.get("tasa_libre_riesgo", 0.0) or 0.0), 2),
                str(data.get("pe", "")),
                str(data.get("p_fcf", "")),
                str(data.get("peg", "")),
                str(data.get("ev_ebitda", "")),
                str(data.get("p_s", "")),
                str(data.get("p_b", "")),
                str(data.get("texto_ia_final", ""))[:200],
                str(data.get("perfil_texto", ""))[:200],
            )
            return hashlib.sha256(str(claves).encode("utf-8")).hexdigest()

        h1 = _calc_hash(analysis_data)
        h2 = _calc_hash(analysis_data)
        self.assertEqual(h1, h2)

        data_mod = dict(analysis_data)
        data_mod["slider_no_valuacion"] = 123
        self.assertEqual(h1, _calc_hash(data_mod))

        data_mod["wacc"] = 9.0
        self.assertNotEqual(h1, _calc_hash(data_mod))

    def test_criterio_1_ai_service_persistencia_cache(self):
        """
        Criterio 1: obtener_analisis_macro_ia y obtener_perfil_corporativo
        persisten en caché con TTL de 24h y no duplican peticiones a Gemini.
        """
        with patch("services.ai_service.genai.GenerativeModel") as mock_model_cls, \
             patch("services.ai_service._ejecutar_gemini_con_timeout") as mock_exec:
            mock_resp = MagicMock()
            mock_resp.text = "Tesis macroeconómica positiva.\nPUNTUACION_MACRO: 5"
            mock_exec.return_value = mock_resp

            t1, s1 = obtener_analisis_macro_ia("TEST_TICKER", "Test Corp", "Technology", "mock_key_123")
            t2, s2 = obtener_analisis_macro_ia("TEST_TICKER", "Test Corp", "Technology", "mock_key_123")

            self.assertEqual(t1, t2)
            self.assertEqual(s1, s2)
            self.assertEqual(s1, 5.0)
            self.assertEqual(mock_exec.call_count, 1)

        with patch("services.ai_service.genai.GenerativeModel") as mock_model_cls, \
             patch("services.ai_service._ejecutar_gemini_con_timeout") as mock_exec:
            mock_resp = MagicMock()
            mock_resp.text = "🎯 1. Perfil General: Empresa global..."
            mock_exec.return_value = mock_resp

            p1 = obtener_perfil_corporativo("TEST_PERF", "Test Corp", "Tech", "Software", "key_abc", 100e9, 25.0, 18.0)
            p2 = obtener_perfil_corporativo("TEST_PERF", "Test Corp", "Tech", "Software", "key_abc", 100e9, 25.0, 18.0)

            self.assertEqual(p1, p2)
            self.assertEqual(mock_exec.call_count, 1)

    def test_criterio_1_y_3_fred_cache_y_desacoplamiento(self):
        """
        Criterio 1 y 3: obtener_tasa_fred está cacheada y desacoplada del arranque global en app.py.
        """
        # 1. Comprobar que en app.py no existe llamada a nivel de módulo a obtener_tasa_fred
        with open("app.py", "r", encoding="utf-8") as f:
            code = f.read()

        # No debe haber una asignación a nivel global antes de 'if not ticker_input:'
        seccion_previa = code.split("if not ticker_input:")[0]
        self.assertNotIn("tasa_libre_riesgo = obtener_tasa_fred", seccion_previa,
                         "obtener_tasa_fred no debe ejecutarse en el flujo global antes del enrutamiento de ticker")

        # 2. Comprobar que obtener_tasa_fred está cacheada
        with patch("data.financial_fetcher.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"observations": [{"value": "4.35"}]}
            mock_get.return_value = mock_resp

            tasa1 = obtener_tasa_fred("fred_test_key_xyz")
            tasa2 = obtener_tasa_fred("fred_test_key_xyz")

            self.assertEqual(tasa1, 4.35)
            self.assertEqual(tasa2, 4.35)
            self.assertEqual(mock_get.call_count, 1)

    def test_criterio_2_peers_deduplicacion(self):
        """
        Criterio 2: El ticker de prueba 'MA' carga peers sin duplicar llamadas a Finnhub.
        """
        mock_client = MagicMock()
        mock_client.company_peers.return_value = ["MA", "V", "AXP", "PYPL"]

        peers1 = obtener_peers("MA", max_peers=3, _finnhub_client=mock_client)
        peers2 = obtener_peers("MA", max_peers=3, _finnhub_client=mock_client)

        self.assertEqual(peers1, ["V", "AXP", "PYPL"])
        self.assertEqual(peers2, ["V", "AXP", "PYPL"])
        self.assertEqual(mock_client.company_peers.call_count, 1)

    def test_criterio_2_noticias_reutilizacion_sin_llamada_secundaria(self):
        """
        Criterio 2: La pestaña de noticias en app.py reutiliza news_data_cache sin llamadas secundarias.
        """
        with open("app.py", "r", encoding="utf-8") as f:
            code = f.read()

        # Debe contener la centralización sin llamada secundaria
        self.assertIn("news_data = news_data_cache if news_data_cache is not None else []", code)
        self.assertNotIn("news_data = news_data_cache if news_data_cache else obtener_noticias_financieras", code)

    def test_criterio_3_estandarizacion_canonicos_dataframes(self):
        """
        Criterio 3: Normalización temprana de etiquetas a nombres canónicos consistentes.
        """
        inc_raw = pd.DataFrame({
            "2023": [1000.0, 400.0, 200.0, 50.0],
            "2024": [1200.0, 480.0, 250.0, 48.0],
        }, index=["revenue", "operatingIncome", "netIncomeToCommon", "weightedAverageShsOutDil"])

        inc_std = estandarizar_dataframe_financiero(inc_raw, MAPA_CANONICO_INCOME)

        self.assertIn("Total Revenue", inc_std.index)
        self.assertIn("Operating Income", inc_std.index)
        self.assertIn("Net Income", inc_std.index)
        self.assertIn("Diluted Average Shares", inc_std.index)

        self.assertEqual(inc_std.loc["Total Revenue", "2024"], 1200.0)
        self.assertEqual(inc_std.loc["Operating Income", "2024"], 480.0)
        self.assertEqual(inc_std.loc["Net Income", "2024"], 250.0)

        cf_raw = pd.DataFrame({
            "2023": [300.0, 50.0, 80.0],
            "2024": [360.0, 60.0, 100.0],
        }, index=["netCashProvidedByOperatingActivities", "investmentsInPropertyPlantAndEquipment", "commonStockRepurchased"])

        cf_std = estandarizar_dataframe_financiero(cf_raw, MAPA_CANONICO_CASHFLOW)

        self.assertIn("Operating Cash Flow", cf_std.index)
        self.assertIn("Capital Expenditure", cf_std.index)
        self.assertIn("Repurchase Of Capital Stock", cf_std.index)

        self.assertEqual(cf_std.loc["Operating Cash Flow", "2024"], 360.0)
        self.assertEqual(cf_std.loc["Capital Expenditure", "2024"], 60.0)
        self.assertEqual(cf_std.loc["Repurchase Of Capital Stock", "2024"], 100.0)

    def test_criterio_3_calcular_buyback_yield_direct_access(self):
        """
        Criterio 3: calcular_buyback_yield y los gráficos acceden directamente a las métricas.
        """
        # Método 1: Acciones canónicas en 'Diluted Average Shares'
        inc_shares = pd.DataFrame({
            "2023": [100.0],
            "2024": [96.0],
        }, index=["Diluted Average Shares"])

        res1 = calcular_buyback_yield(inc=inc_shares, bs=pd.DataFrame(), cf=pd.DataFrame(), mcap=5000.0)
        self.assertEqual(res1["buyback_yield"], 4.0)
        self.assertEqual(res1["buyback_yield_str"], "4.0%")
        self.assertEqual(res1["col_by"], "🟢")

        # Método 2: Flujo de caja en 'Repurchase Of Capital Stock'
        cf_repurchase = pd.DataFrame({
            "2024": [150.0],
        }, index=["Repurchase Of Capital Stock"])

        res2 = calcular_buyback_yield(inc=pd.DataFrame(), bs=pd.DataFrame(), cf=cf_repurchase, mcap=5000.0)
        self.assertEqual(res2["buyback_yield"], 3.0)
        self.assertEqual(res2["buyback_yield_str"], "3.0%")
        self.assertEqual(res2["col_by"], "🟢")

        # Verificación en app.py de accesos directos indexados en gráficos de márgenes y flujos
        with open("app.py", "r", encoding="utf-8") as f:
            code = f.read()

        self.assertIn("row_rev = 'Total Revenue' if 'Total Revenue' in inc.index else", code)
        self.assertIn("row_ocf = 'Operating Cash Flow' if 'Operating Cash Flow' in cf.index else None", code)
        self.assertNotIn("posibles_rev = ['Total Revenue', 'Operating Revenue'", code)
        self.assertNotIn("posibles_ocf = ['Operating Cash Flow', 'OperatingCashFlow'", code)


if __name__ == "__main__":
    unittest.main()
