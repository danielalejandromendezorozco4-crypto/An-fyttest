"""
tests/test_api_resilience.py — Pruebas de resiliencia y blindaje de conexión (HTTP 429, 403, Timeouts y Fallbacks).

Cubre:
1. Reintentos con retroceso exponencial (_fetch_with_retry) ante HTTP 429 en Finnhub y FMP.
2. Degradación elegante ante agotamiento de cuotas y registro de diagnósticos transparentes.
3. Fallback automático a cálculo contable manual (Altman Z y Piotroski) cuando FMP /financial-score responde HTTP 403.
4. Manejo defensivo de respuestas vacías, None o corruptas en endpoints secundarios.
5. Auditoría y validación robusta de API keys en config/settings.py sin excepciones no controladas.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import requests

from config.settings import SecretsTuple, cargar_secrets, validar_secrets_configurados
from data.financial_fetcher import (
    FinnhubClient,
    _fetch_with_retry,
    _finnhub_get,
    _fmp_get,
    fetch_cotizacion_intradia,
    fetch_datos_concurrente,
    fetch_datos_fundamentales,
    fetch_fmp_balance_sheet,
    fetch_fmp_cash_flow,
    fetch_fmp_company_profile,
    fetch_fmp_enterprise_values,
    fetch_fmp_financial_score,
    fetch_fmp_income_statement,
    fetch_fmp_key_metrics_ttm,
    fetch_fmp_quote,
    fetch_fmp_ratios_ttm,
    limpiar_diagnosticos_api,
    obtener_consenso_wall_street,
    obtener_diagnosticos_api,
    obtener_noticias_financieras,
    registrar_evento_diagnostico,
)


class TestApiResilience(unittest.TestCase):
    def setUp(self):
        limpiar_diagnosticos_api()

    # ─────────────────────────────────────────────────────────────────────────
    # 1. PRUEBAS DE RETROCESO EXPONENCIAL Y REINTENTOS ANTE HTTP 429
    # ─────────────────────────────────────────────────────────────────────────

    def test_fetch_with_retry_exito_al_primer_intento(self):
        """Verifica que si la respuesta es HTTP 200 no se realicen reintentos adicionales."""
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "ok"}

        mock_fn = MagicMock(return_value=mock_resp)
        res = _fetch_with_retry(mock_fn, api_name="TestAPI", endpoint="test", max_retries=3, base_delay=0.001, burst_pause=0)

        self.assertIsNotNone(res)
        self.assertEqual(res.status_code, 200)
        self.assertEqual(mock_fn.call_count, 1)

    def test_fetch_with_retry_429_recuperacion_exitosa(self):
        """Verifica que ante HTTP 429 se reintente y retorne la respuesta si el reintento es exitoso (HTTP 200)."""
        resp_429 = MagicMock(spec=requests.Response)
        resp_429.status_code = 429

        resp_200 = MagicMock(spec=requests.Response)
        resp_200.status_code = 200

        # Falla 2 veces con 429 y a la 3ra responde 200
        mock_fn = MagicMock(side_effect=[resp_429, resp_429, resp_200])

        res = _fetch_with_retry(mock_fn, api_name="Finnhub", endpoint="stock/profile2", max_retries=3, base_delay=0.001, burst_pause=0)

        self.assertIsNotNone(res)
        self.assertEqual(res.status_code, 200)
        self.assertEqual(mock_fn.call_count, 3)

    def test_fetch_with_retry_429_agotamiento_degradacion_elegante(self):
        """Verifica que tras agotar los 3 reintentos ante HTTP 429 retorne None y registre diagnóstico sin elevar excepción."""
        resp_429 = MagicMock(spec=requests.Response)
        resp_429.status_code = 429

        mock_fn = MagicMock(return_value=resp_429)

        res = _fetch_with_retry(mock_fn, api_name="Finnhub", endpoint="stock/price-target", max_retries=3, base_delay=0.001, burst_pause=0)

        self.assertIsNone(res)
        self.assertEqual(mock_fn.call_count, 4)  # Intento inicial + 3 reintentos

        diagnosticos = obtener_diagnosticos_api()
        self.assertTrue(any(d["status_code"] == 429 and "Finnhub" in d["api"] for d in diagnosticos))
        self.assertTrue(any("Rate limit" in d["detalle"] for d in diagnosticos))

    def test_fetch_with_retry_403_sin_reintentos_inutiles(self):
        """Verifica que ante HTTP 403 (Forbidden) no se malgasten reintentos y se active degradación inmediata."""
        resp_403 = MagicMock(spec=requests.Response)
        resp_403.status_code = 403

        mock_fn = MagicMock(return_value=resp_403)

        res = _fetch_with_retry(mock_fn, api_name="FMP", endpoint="financial-score/AAPL", max_retries=3, base_delay=0.001, burst_pause=0)

        self.assertIsNone(res)
        self.assertEqual(mock_fn.call_count, 1)  # Solo 1 intento, 403 no es transitorio
        diagnosticos = obtener_diagnosticos_api()
        self.assertTrue(any(d["status_code"] == 403 for d in diagnosticos))

    def test_fetch_with_retry_network_timeout_resiliente(self):
        """Verifica que ante errores de conexión o Timeout no colapse el motor."""
        mock_fn = MagicMock(side_effect=requests.exceptions.ConnectTimeout("Connection timed out"))

        res = _fetch_with_retry(mock_fn, api_name="FMP", endpoint="income-statement", max_retries=2, base_delay=0.001, burst_pause=0)

        self.assertIsNone(res)
        self.assertEqual(mock_fn.call_count, 3)
        diagnosticos = obtener_diagnosticos_api()
        self.assertTrue(any("timeout" in d["detalle"].lower() or "conexión" in d["detalle"].lower() for d in diagnosticos))

    # ─────────────────────────────────────────────────────────────────────────
    # 2. PRUEBAS DE CLIENTES _finnhub_get Y _fmp_get ANTE ERRORES HTTP
    # ─────────────────────────────────────────────────────────────────────────

    @patch("data.financial_fetcher.requests.Session.get")
    def test_finnhub_get_429_retorna_none(self, mock_get):
        resp_429 = MagicMock(spec=requests.Response)
        resp_429.status_code = 429
        mock_get.return_value = resp_429

        data = _finnhub_get("quote", params={"symbol": "MSFT"}, api_key="dummy_key")
        self.assertIsNone(data)

    @patch("data.financial_fetcher.requests.Session.get")
    def test_fmp_get_429_retorna_none(self, mock_get):
        resp_429 = MagicMock(spec=requests.Response)
        resp_429.status_code = 429
        mock_get.return_value = resp_429

        data = _fmp_get("quote/MSFT", api_key="dummy_key")
        self.assertIsNone(data)

    # ─────────────────────────────────────────────────────────────────────────
    # 3. FALLBACK DE ALTMAN Z Y PIOTROSKI F-SCORE ANTE HTTP 403 EN FMP
    # ─────────────────────────────────────────────────────────────────────────

    @patch("data.financial_fetcher._fmp_get")
    @patch("data.financial_fetcher._finnhub_get")
    def test_fmp_financial_score_403_fallback_manual(self, mock_finnhub_get, mock_fmp_get):
        """
        Verifica que si FMP /financial-score retorna HTTP 403 (o None),
        fetch_datos_fundamentales calcule automáticamente Altman Z y Piotroski F-Score
        usando los balances de FMP sin romper la compilación ni retornar None.
        """
        # Finnhub devuelve perfil y métricas mínimas
        def finnhub_side_effect(endpoint, *args, **kwargs):
            if "profile2" in endpoint:
                return {"name": "Apple Inc", "finnhubIndustry": "Technology", "marketCapitalization": 3000000}
            if "metric" in endpoint:
                return {"metric": {"beta": 1.15, "epsTTM": 6.5}}
            return {}

        # FMP devuelve estados contables pero /financial-score responde None (403 simulado)
        def fmp_side_effect(endpoint, *args, **kwargs):
            if "financial-score" in endpoint:
                return None  # HTTP 403 simulado
            if "income-statement" in endpoint:
                return [{
                    "calendarYear": "2024",
                    "revenue": 391035000000,
                    "grossProfit": 180683000000,
                    "operatingIncome": 123216000000,
                    "netIncome": 93736000000,
                    "interestExpense": 3933000000,
                    "incomeBeforeTax": 119283000000,
                    "incomeTaxExpense": 25547000000,
                    "weightedAverageShsOutDil": 15400000000,
                }]
            if "balance-sheet-statement" in endpoint:
                return [{
                    "calendarYear": "2024",
                    "totalAssets": 364980000000,
                    "totalCurrentAssets": 152988000000,
                    "totalCurrentLiabilities": 176392000000,
                    "totalStockholdersEquity": 66885000000,
                    "totalDebt": 106629000000,
                    "cashAndCashEquivalents": 29942000000,
                }]
            if "cash-flow-statement" in endpoint:
                return [{
                    "calendarYear": "2024",
                    "operatingCashFlow": 118254000000,
                    "capitalExpenditure": -9457000000,
                    "freeCashFlow": 108797000000,
                    "commonStockRepurchased": -95000000000,
                }]
            if "ratios-ttm" in endpoint:
                return [{"currentRatioTTM": 0.87, "returnOnEquityTTM": 1.40}]
            if "key-metrics-ttm" in endpoint:
                return [{"roicTTM": 0.28}]
            if "enterprise-values" in endpoint:
                return [{"numberOfShares": 15400000000, "marketCapitalization": 3000000000000}]
            return {}

        mock_finnhub_get.side_effect = finnhub_side_effect
        mock_fmp_get.side_effect = fmp_side_effect

        info, inc, bs, cf = fetch_datos_fundamentales("AAPL", finnhub_api_key="test", fmp_api_key="test")

        self.assertIsNotNone(info)
        # Altman Z-Score debe haberse calculado manualmente mediante fórmula Altman
        self.assertIn("altmanZScore", info)
        self.assertGreater(info["altmanZScore"], 0.0)

        # Piotroski F-Score debe haberse evaluado mediante auditoría contable (0 a 9)
        self.assertIn("piotroskiScore", info)
        self.assertIsInstance(info["piotroskiScore"], int)
        self.assertGreaterEqual(info["piotroskiScore"], 0)
        self.assertLessEqual(info["piotroskiScore"], 9)

    # ─────────────────────────────────────────────────────────────────────────
    # 4. PRUEBAS DE RESPUESTAS VACÍAS Y NONE EN ENDPOINTS SECUNDARIOS
    # ─────────────────────────────────────────────────────────────────────────

    @patch("data.financial_fetcher._finnhub_get")
    def test_finnhub_endpoints_secundarios_retornan_defaults_defensivos(self, mock_get):
        """Verifica que ante None o [] en Finnhub no ocurran errores de NoneType o KeyError."""
        mock_get.return_value = None
        client = FinnhubClient(api_key="test")

        self.assertEqual(client.company_peers("AAPL"), [])
        self.assertEqual(client.recommendation_trends("AAPL"), [])
        self.assertEqual(client.company_news("AAPL", "2024-01-01", "2024-01-31"), [])

        pt = client.price_target("AAPL")
        self.assertEqual(pt["targetMean"], 0.0)
        self.assertEqual(pt["numberAnalysts"], 0)

        q = client.quote("AAPL")
        self.assertEqual(q, {})

        sc = client.stock_candles("AAPL", "D", 0, 100)
        self.assertEqual(sc, {})

    @patch("data.financial_fetcher._fmp_get")
    def test_fmp_endpoints_secundarios_retornan_defaults_defensivos(self, mock_get):
        """Verifica que ante None o [] en FMP las funciones retornen listas/dicts vacíos sin excepción."""
        mock_get.return_value = None

        self.assertEqual(fetch_fmp_income_statement("AAPL"), [])
        self.assertEqual(fetch_fmp_balance_sheet("AAPL"), [])
        self.assertEqual(fetch_fmp_cash_flow("AAPL"), [])
        self.assertEqual(fetch_fmp_ratios_ttm("AAPL"), {})
        self.assertEqual(fetch_fmp_key_metrics_ttm("AAPL"), {})
        self.assertEqual(fetch_fmp_financial_score("AAPL"), {})
        self.assertEqual(fetch_fmp_enterprise_values("AAPL"), {})
        self.assertEqual(fetch_fmp_company_profile("AAPL"), {})
        self.assertEqual(fetch_fmp_quote("AAPL"), {})

    # ─────────────────────────────────────────────────────────────────────────
    # 5. COTIZACIÓN INTRADÍA: FALLBACK FMP Y BLINDAJE DE HIST
    # ─────────────────────────────────────────────────────────────────────────

    @patch("data.financial_fetcher._finnhub_get")
    @patch("data.financial_fetcher.fetch_fmp_quote")
    def test_fetch_cotizacion_intradia_fallback_fmp_cuando_finnhub_falla(self, mock_fmp_quote, mock_finnhub_get):
        """
        Si Finnhub quote y stock_candle fallan (HTTP 429), verifica que:
        1. Recupere precio_actual y prev_close desde FMP quote.
        2. Genere un DataFrame hist de 1 fila para evitar que hist.empty rompa la UI.
        """
        mock_finnhub_get.return_value = None
        mock_fmp_quote.return_value = {
            "symbol": "NVDA",
            "price": 130.50,
            "previousClose": 128.00,
        }

        precio, prev, hist = fetch_cotizacion_intradia("NVDA", finnhub_api_key="test", fmp_key="test")

        self.assertEqual(precio, 130.50)
        self.assertEqual(prev, 128.00)
        self.assertFalse(hist.empty)
        self.assertEqual(len(hist), 1)
        self.assertEqual(hist["Close"].iloc[0], 130.50)

    # ─────────────────────────────────────────────────────────────────────────
    # 6. ORQUESTADOR CONCURRENTE: DEGRADACIÓN ELEGANTE TOTAL
    # ─────────────────────────────────────────────────────────────────────────

    @patch("data.financial_fetcher.fetch_cotizacion_intradia")
    @patch("data.financial_fetcher.fetch_datos_fundamentales")
    @patch("data.financial_fetcher.obtener_tasa_fred")
    @patch("data.financial_fetcher.obtener_noticias_financieras")
    def test_fetch_datos_concurrente_degradacion_completa(
        self, mock_news, mock_fred, mock_funda, mock_quote
    ):
        """Verifica que incluso si todos los hilos fallan con excepción, el orquestador retorna estructura segura."""
        mock_quote.side_effect = Exception("Finnhub 429 Rate Limit")
        mock_funda.side_effect = Exception("FMP Connection Error")
        mock_fred.side_effect = Exception("FRED Timeout")
        mock_news.side_effect = Exception("News Error")

        registrar_evento_diagnostico("Finnhub", "quote", 429, "Finnhub: Rate limit temporal (429)")

        datos = fetch_datos_concurrente("AAPL", finnhub_key="k", fred_key="k", fmp_key="k")

        self.assertIsInstance(datos, dict)
        self.assertEqual(datos["precio_actual"], 0.0)
        self.assertEqual(datos["prev_close"], 0.0)
        self.assertTrue(datos["hist"].empty)
        self.assertEqual(datos["info"], {})
        self.assertTrue(datos["inc"].empty)
        self.assertEqual(datos["tasa_fred"], 4.20)
        self.assertEqual(datos["news_data"], [])
        self.assertIn("diagnosticos", datos)

    # ─────────────────────────────────────────────────────────────────────────
    # 7. AUDITORÍA Y BLINDAJE DE SECRETS
    # ─────────────────────────────────────────────────────────────────────────

    def test_validar_secrets_configurados_todas_presentes(self):
        secrets = SecretsTuple(
            gemini_key="ai_gemini_key_12345",
            fred_key="fred_api_key_12345",
            finnhub_key="finnhub_api_key_12345",
            fmp_key="fmp_api_key_12345",
        )
        diag = validar_secrets_configurados(secrets)
        self.assertTrue(diag["fmp_valida"])
        self.assertTrue(diag["finnhub_valida"])
        self.assertTrue(diag["todas_criticas_presentes"])
        self.assertEqual(len(diag["errores"]), 0)

    def test_validar_secrets_configurados_faltan_criticas(self):
        secrets = SecretsTuple(
            gemini_key="",
            fred_key="",
            finnhub_key="",
            fmp_key="",
        )
        diag = validar_secrets_configurados(secrets)
        self.assertFalse(diag["fmp_valida"])
        self.assertFalse(diag["finnhub_valida"])
        self.assertFalse(diag["todas_criticas_presentes"])
        self.assertGreater(len(diag["errores"]), 0)
        self.assertIn("FMP_API_KEY", diag["errores"][0])

    def test_validar_secrets_configurados_placeholders(self):
        secrets = SecretsTuple(
            gemini_key="tu_clave_aqui",
            fred_key="xxx",
            finnhub_key="your_api_key_here",
            fmp_key="tu_clave_aqui",
        )
        diag = validar_secrets_configurados(secrets)
        self.assertFalse(diag["fmp_valida"])
        self.assertFalse(diag["finnhub_valida"])
        self.assertFalse(diag["todas_criticas_presentes"])


if __name__ == "__main__":
    unittest.main()
