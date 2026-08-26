"""
tests/test_finnhub_fetcher.py — Pruebas unitarias completas para el módulo Finnhub API.
"""
from __future__ import annotations

import datetime
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest
import requests

from data.financial_fetcher import (
    _finnhub_get,
    _map_gics_sector,
    extraer_fcff_desapalancado,
    extraer_metricas_ttm,
    fetch_cotizacion_intradia,
    fetch_datos_concurrente,
    fetch_datos_fundamentales,
    obtener_capex_historico,
    obtener_datos_dividendos,
    obtener_kd_finnhub_fred,
    obtener_noticias_financieras,
    obtener_session_finnhub,
    safe_num,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. PRUEBAS DE CLIENTE HTTP Y MANEJO DE ERRORES / CUOTAS
# ─────────────────────────────────────────────────────────────────────────────

class TestFinnhubHttpClient:
    def test_obtener_session_finnhub_headers(self):
        session = obtener_session_finnhub(api_key="test_key_123")
        assert session.headers.get("X-Finnhub-Token") == "test_key_123"
        assert "An-FyT" in session.headers.get("User-Agent", "")

    @patch("requests.Session.get")
    def test_finnhub_get_success_200(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"c": 150.0, "pc": 148.0}
        mock_get.return_value = mock_resp

        res = _finnhub_get("quote", params={"symbol": "AAPL"}, api_key="secret_token")
        assert res == {"c": 150.0, "pc": 148.0}
        assert mock_get.called
        args, kwargs = mock_get.call_args
        assert "quote" in args[0]
        assert kwargs["params"]["token"] == "secret_token"

    @patch("requests.Session.get")
    def test_finnhub_get_rate_limit_429(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_get.return_value = mock_resp

        res = _finnhub_get("quote", params={"symbol": "AAPL"}, api_key="secret_token")
        assert res is None

    @patch("requests.Session.get")
    def test_finnhub_get_unauthorized_401(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_get.return_value = mock_resp

        res = _finnhub_get("quote", params={"symbol": "AAPL"}, api_key="invalid_token")
        assert res is None

    @patch("requests.Session.get")
    def test_finnhub_get_network_timeout(self, mock_get):
        mock_get.side_effect = requests.exceptions.Timeout("Connection timed out")

        res = _finnhub_get("quote", params={"symbol": "AAPL"}, api_key="token")
        assert res is None


# ─────────────────────────────────────────────────────────────────────────────
# 2. PRUEBAS DE COTIZACIÓN Y VELAS DIARIAS (OHLCV)
# ─────────────────────────────────────────────────────────────────────────────

class TestFinnhubQuotesAndCandles:
    @patch("data.financial_fetcher._finnhub_get")
    def test_fetch_cotizacion_intradia_ok(self, mock_get):
        def side_effect(endpoint, params=None, api_key="", timeout=8.0):
            if endpoint == "quote":
                return {"c": 175.50, "pc": 174.00, "d": 1.50, "dp": 0.86, "t": 1700000000}
            if endpoint == "stock/candle":
                return {
                    "s": "ok",
                    "t": [1680000000, 1680086400, 1680172800],
                    "o": [170.0, 172.0, 174.0],
                    "h": [173.0, 175.0, 176.0],
                    "l": [169.0, 171.0, 173.5],
                    "c": [172.0, 174.0, 175.5],
                    "v": [50000000, 48000000, 52000000],
                }
            return None

        mock_get.side_effect = side_effect

        precio_actual, prev_close, hist = fetch_cotizacion_intradia("AAPL", "key_test")

        assert precio_actual == 175.50
        assert prev_close == 174.00
        assert isinstance(hist, pd.DataFrame)
        assert not hist.empty
        assert list(hist.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert isinstance(hist.index, pd.DatetimeIndex)
        assert len(hist) == 3

    @patch("data.financial_fetcher._finnhub_get")
    def test_fetch_cotizacion_empty_or_error(self, mock_get):
        mock_get.return_value = None

        precio_actual, prev_close, hist = fetch_cotizacion_intradia("INVALID", "key_test")
        assert precio_actual == 0.0
        assert prev_close == 0.0
        assert hist.empty


# ─────────────────────────────────────────────────────────────────────────────
# 3. PRUEBAS DE DATOS FUNDAMENTALES Y ESTADOS FINANCIEROS
# ─────────────────────────────────────────────────────────────────────────────

class TestFinnhubFundamentals:
    @patch("data.financial_fetcher._finnhub_get")
    def test_fetch_datos_fundamentales_mapping(self, mock_get):
        def side_effect(endpoint, params=None, api_key="", timeout=8.0):
            if endpoint == "stock/profile2":
                return {
                    "name": "Apple Inc",
                    "ticker": "AAPL",
                    "finnhubIndustry": "Technology",
                    "marketCapitalization": 2800000.0,  # 2.8 Trillion (en millones)
                    "shareOutstanding": 15500.0,        # 15.5 Billion (en millones)
                }
            if endpoint == "stock/metric":
                return {
                    "metric": {
                        "beta": 1.15,
                        "epsTTM": 6.42,
                        "peTTM": 28.5,
                        "dividendPerShareAnnual": 0.96,
                        "currentRatioQuarterly": 1.05,
                        "totalDebt/totalEquityQuarterly": 1.45,
                        "roeTTM": 140.5,
                        "roaTTM": 28.2,
                        "roicTTM": 45.0,
                        "operatingMarginTTM": 30.5,
                        "revenueGrowthTTMYoy": 8.5,
                        "epsGrowthTTMYoy": 11.2,
                        "ebitdaAnnual": 130000000000.0,
                        "fcfAnnual": 100000000000.0,
                    }
                }
            if endpoint == "stock/financials-reported":
                return {
                    "data": [
                        {
                            "year": 2023,
                            "report": {
                                "ic": [
                                    {"concept": "RevenueFromContractWithCustomerExcludingAssessedTax", "value": 383285000000},
                                    {"concept": "GrossProfit", "value": 169148000000},
                                    {"concept": "OperatingIncomeLoss", "value": 114301000000},
                                    {"concept": "NetIncomeLoss", "value": 96995000000},
                                    {"concept": "InterestExpense", "value": 3933000000},
                                    {"concept": "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments", "value": 113736000000},
                                    {"concept": "IncomeTaxExpenseBenefit", "value": 16741000000},
                                ],
                                "bs": [
                                    {"concept": "Assets", "value": 352583000000},
                                    {"concept": "AssetsCurrent", "value": 143566000000},
                                    {"concept": "LiabilitiesCurrent", "value": 145308000000},
                                    {"concept": "StockholdersEquity", "value": 62146000000},
                                    {"concept": "CashAndCashEquivalentsAtCarryingValue", "value": 29965000000},
                                    {"concept": "LongTermDebtNoncurrent", "value": 95281000000},
                                    {"concept": "DebtCurrent", "value": 15807000000},
                                ],
                                "cf": [
                                    {"concept": "NetCashProvidedByUsedInOperatingActivities", "value": 110543000000},
                                    {"concept": "PaymentsToAcquirePropertyPlantAndEquipment", "value": 10959000000},
                                    {"concept": "PaymentsForRepurchaseOfCommonStock", "value": 77550000000},
                                ],
                            }
                        }
                    ]
                }
            return None

        mock_get.side_effect = side_effect

        info, inc, bs, cf = fetch_datos_fundamentales("AAPL", "key_test")

        # Verificar multiplicador de millones a valor absoluto
        assert info["marketCap"] == pytest.approx(2.8e12, rel=1e-3)
        assert info["sharesOutstanding"] == pytest.approx(15.5e9, rel=1e-3)
        assert info["sector"] == "Technology"
        assert info["beta"] == 1.15
        assert info["trailingEps"] == 6.42
        assert info["operatingMargins"] == pytest.approx(0.305, rel=1e-3)

        # Verificar DataFrames
        assert not inc.empty
        assert not bs.empty
        assert not cf.empty

        assert "Total Revenue" in inc.index
        assert inc.loc["Total Revenue", "2023"] == 383285000000

        assert "Total Assets" in bs.index
        assert bs.loc["Total Assets", "2023"] == 352583000000

        assert "Operating Cash Flow" in cf.index
        assert cf.loc["Operating Cash Flow", "2023"] == 110543000000
        assert "Capital Expenditure" in cf.index
        assert cf.loc["Capital Expenditure", "2023"] == 10959000000


# ─────────────────────────────────────────────────────────────────────────────
# 4. PRUEBAS DE NOTICIAS Y MACROECONOMÍA
# ─────────────────────────────────────────────────────────────────────────────

class TestFinnhubNewsAndMacro:
    @patch("data.financial_fetcher._finnhub_get")
    def test_obtener_noticias_financieras(self, mock_get):
        mock_get.return_value = [
            {
                "category": "company",
                "datetime": 1700000000,
                "headline": "Apple Unveils Breakthrough AI Features",
                "id": 12345,
                "image": "https://example.com/img.jpg",
                "related": "AAPL",
                "source": "Bloomberg",
                "summary": "Apple announces new generative AI capabilities.",
                "url": "https://bloomberg.com/news/apple-ai",
            },
            {
                "headline": "Invalid Empty URL",
                "url": "#",
                "source": "Unknown",
            }
        ]

        news = obtener_noticias_financieras("AAPL", "key_test")
        assert len(news) == 1
        assert news[0]["title"] == "Apple Unveils Breakthrough AI Features"
        assert news[0]["publisher"] == "Bloomberg"
        assert news[0]["link"] == "https://bloomberg.com/news/apple-ai"

    def test_obtener_kd_finnhub_fred_direct(self):
        kd = obtener_kd_finnhub_fred("AAPL", int_expense=4000.0, total_debt_val=100000.0)
        assert kd == 4.0

    @patch("requests.get")
    def test_obtener_kd_finnhub_fred_fallback(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"observations": [{"value": "5.75"}]}
        mock_get.return_value = mock_resp

        kd = obtener_kd_finnhub_fred("AAPL", fred_api_key="fred_token", int_expense=0.0, total_debt_val=0.0)
        assert kd == 5.75


# ─────────────────────────────────────────────────────────────────────────────
# 5. PRUEBAS DEL ORQUESTADOR CONCURRENTE
# ─────────────────────────────────────────────────────────────────────────────

class TestFinnhubConcurrentOrchestrator:
    @patch("data.financial_fetcher.fetch_cotizacion_intradia")
    @patch("data.financial_fetcher.fetch_datos_fundamentales")
    @patch("data.financial_fetcher.obtener_tasa_fred")
    @patch("data.financial_fetcher.obtener_noticias_financieras")
    def test_fetch_datos_concurrente(self, mock_news, mock_fred, mock_funda, mock_quote):
        mock_quote.return_value = (180.0, 178.0, pd.DataFrame({"Close": [178.0, 180.0]}))
        mock_funda.return_value = (
            {"symbol": "MSFT", "marketCap": 3e12},
            pd.DataFrame({"2023": [1000]}),
            pd.DataFrame({"2023": [2000]}),
            pd.DataFrame({"2023": [500]}),
        )
        mock_fred.return_value = 4.25
        mock_news.return_value = [{"title": "MSFT Cloud Surge", "link": "https://example.com"}]

        res = fetch_datos_concurrente("MSFT", finnhub_key="key", fred_key="fred")

        assert res["precio_actual"] == 180.0
        assert res["prev_close"] == 178.0
        assert not res["hist"].empty
        assert res["info"]["symbol"] == "MSFT"
        assert res["tasa_fred"] == 4.25
        assert len(res["news_data"]) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 6. PRUEBAS DE MAPPING DE SECTORES GICS
# ─────────────────────────────────────────────────────────────────────────────

class TestFinnhubSectorMapping:
    def test_sector_mappings(self):
        assert _map_gics_sector("Technology") == "Technology"
        assert _map_gics_sector("Semiconductors") == "Technology"
        assert _map_gics_sector("Biotechnology") == "Healthcare"
        assert _map_gics_sector("Banking") == "Financial Services"
        assert _map_gics_sector("Automobiles") == "Consumer Cyclical"
        assert _map_gics_sector("Beverages") == "Consumer Defensive"
        assert _map_gics_sector("Aerospace & Defense") == "Industrials"
        assert _map_gics_sector("Oil & Gas") == "Energy"
        assert _map_gics_sector("Electric Utilities") == "Utilities"
        assert _map_gics_sector("Real Estate Investment Trusts") == "Real Estate"
        assert _map_gics_sector("Telecommunications") == "Communication Services"
        assert _map_gics_sector("Chemicals") == "Basic Materials"
