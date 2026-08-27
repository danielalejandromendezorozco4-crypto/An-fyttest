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
    obtener_consenso_wall_street,
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


# ─────────────────────────────────────────────────────────────────────────────
# 7. PRUEBAS DE EXTRACCIÓN Y VALIDACIÓN DEL CONSENSO DE WALL STREET
# ─────────────────────────────────────────────────────────────────────────────

class TestFinnhubConsensusPriceTarget:
    @patch("data.financial_fetcher._finnhub_get")
    def test_fetch_datos_fundamentales_consenso_target_mean(self, mock_get):
        """Verifica que el targetMean de analistas se extrae correctamente."""
        def side_effect(endpoint, params=None, api_key="", timeout=8.0):
            if endpoint == "stock/profile2":
                return {"name": "NVIDIA Corp", "ticker": "NVDA", "finnhubIndustry": "Semiconductors", "marketCapitalization": 3000000.0, "shareOutstanding": 24000.0}
            if endpoint == "stock/metric":
                return {"metric": {"beta": 1.45, "epsTTM": 2.80, "peTTM": 45.0}}
            if endpoint == "stock/price-target":
                return {
                    "symbol": "NVDA",
                    "targetHigh": 200.0,
                    "targetLow": 110.0,
                    "targetMean": 165.50,
                    "targetMedian": 160.0,
                    "numberAnalysts": 45
                }
            return None

        mock_get.side_effect = side_effect
        info, inc, bs, cf = fetch_datos_fundamentales("NVDA", "key_test")

        assert info["targetMeanPrice"] == 165.50
        assert info["targetHighPrice"] == 200.0
        assert info["targetLowPrice"] == 110.0

        m_ttm = extraer_metricas_ttm(info, inc, bs, cf, precio_actual=120.0)
        assert m_ttm["target_mean_price"] == 165.50

    @patch("data.financial_fetcher._finnhub_get")
    def test_fetch_datos_fundamentales_consenso_target_median_fallback(self, mock_get):
        """Verifica que si targetMean no está disponible, se utiliza targetMedian."""
        def side_effect(endpoint, params=None, api_key="", timeout=8.0):
            if endpoint == "stock/profile2":
                return {"name": "Apple Inc", "ticker": "AAPL", "finnhubIndustry": "Technology", "marketCapitalization": 3000000.0, "shareOutstanding": 15000.0}
            if endpoint == "stock/metric":
                return {"metric": {"beta": 1.10}}
            if endpoint == "stock/price-target":
                return {
                    "symbol": "AAPL",
                    "targetHigh": 280.0,
                    "targetLow": 190.0,
                    "targetMedian": 240.0,
                }
            return None

        mock_get.side_effect = side_effect
        info, inc, bs, cf = fetch_datos_fundamentales("AAPL", "key_test")

        assert info["targetMeanPrice"] == 240.0

        m_ttm = extraer_metricas_ttm(info, inc, bs, cf, precio_actual=220.0)
        assert m_ttm["target_mean_price"] == 240.0

    @patch("data.financial_fetcher._finnhub_get")
    def test_fetch_datos_fundamentales_consenso_high_low_fallback(self, mock_get):
        """Verifica fallback al promedio de High y Low cuando mean y median no existen."""
        def side_effect(endpoint, params=None, api_key="", timeout=8.0):
            if endpoint == "stock/profile2":
                return {"name": "Microsoft Corp", "ticker": "MSFT", "finnhubIndustry": "Technology", "marketCapitalization": 3200000.0, "shareOutstanding": 7400.0}
            if endpoint == "stock/metric":
                return {"metric": {}}
            if endpoint == "stock/price-target":
                return {
                    "symbol": "MSFT",
                    "targetHigh": 500.0,
                    "targetLow": 400.0,
                }
            return None

        mock_get.side_effect = side_effect
        info, inc, bs, cf = fetch_datos_fundamentales("MSFT", "key_test")

        assert info["targetMeanPrice"] == 450.0

    @patch("data.financial_fetcher._finnhub_get")
    def test_fetch_datos_fundamentales_consenso_metric_dict_fallback(self, mock_get):
        """Verifica fallback a /stock/metric cuando /stock/price-target retorna vacío."""
        def side_effect(endpoint, params=None, api_key="", timeout=8.0):
            if endpoint == "stock/profile2":
                return {"name": "Tesla Inc", "ticker": "TSLA", "finnhubIndustry": "Consumer Cyclical", "marketCapitalization": 800000.0, "shareOutstanding": 3100.0}
            if endpoint == "stock/metric":
                return {"metric": {"targetPrice": 260.0}}
            if endpoint == "stock/price-target":
                return {}
            return None

        mock_get.side_effect = side_effect
        info, inc, bs, cf = fetch_datos_fundamentales("TSLA", "key_test")

        assert info["targetMeanPrice"] == 260.0

    @patch("data.financial_fetcher._finnhub_get")
    def test_fetch_datos_fundamentales_sin_cobertura_analistas(self, mock_get):
        """Verifica manejo elegante cuando un activo no tiene cobertura de analistas (retorna 0.0 sin error)."""
        def side_effect(endpoint, params=None, api_key="", timeout=8.0):
            if endpoint == "stock/profile2":
                return {"name": "Small Cap ETF", "ticker": "SML", "finnhubIndustry": "Financial Services", "marketCapitalization": 500.0, "shareOutstanding": 10.0}
            if endpoint == "stock/metric":
                return {"metric": {}}
            if endpoint == "stock/price-target":
                return {}
            return None

        mock_get.side_effect = side_effect
        info, inc, bs, cf = fetch_datos_fundamentales("SML", "key_test")

        assert info["targetMeanPrice"] == 0.0

        m_ttm = extraer_metricas_ttm(info, inc, bs, cf, precio_actual=50.0)
        assert m_ttm["target_mean_price"] == 0.0

    @patch("yfinance.Ticker")
    def test_obtener_consenso_wall_street_yfinance_analyst_price_targets(self, mock_yf_ticker):
        """Verifica fallback a yfinance.Ticker.analyst_price_targets para tickers como MA."""
        mock_instance = MagicMock()
        mock_instance.analyst_price_targets = {
            "current": 598.47,
            "high": 735.0,
            "low": 550.0,
            "mean": 667.30,
            "median": 668.0,
        }
        mock_instance.info = {}
        mock_yf_ticker.return_value = mock_instance

        mean, high, low = obtener_consenso_wall_street("MA")
        assert mean == 667.30
        assert high == 735.0
        assert low == 550.0

    @patch("yfinance.Ticker")
    def test_obtener_consenso_wall_street_yfinance_info_fallback(self, mock_yf_ticker):
        """Verifica fallback a yfinance.Ticker.info cuando analyst_price_targets está vacío."""
        mock_instance = MagicMock()
        mock_instance.analyst_price_targets = None
        mock_instance.info = {
            "targetMeanPrice": 305.80,
            "targetHighPrice": 500.0,
            "targetLowPrice": 180.0,
        }
        mock_yf_ticker.return_value = mock_instance

        mean, high, low = obtener_consenso_wall_street("NVDA")
        assert mean == 305.80
        assert high == 500.0
        assert low == 180.0

    @patch("yfinance.Ticker")
    def test_obtener_consenso_wall_street_sin_cobertura_etf(self, mock_yf_ticker):
        """Verifica que para activos como VOO sin analistas, retorne (0.0, 0.0, 0.0)."""
        mock_instance = MagicMock()
        mock_instance.analyst_price_targets = {}
        mock_instance.info = {}
        mock_yf_ticker.return_value = mock_instance

        mean, high, low = obtener_consenso_wall_street("VOO")
        assert mean == 0.0
        assert high == 0.0
        assert low == 0.0

    @patch("yfinance.Ticker")
    @patch("data.financial_fetcher._finnhub_get")
    def test_fetch_datos_fundamentales_yfinance_integration(self, mock_finnhub_get, mock_yf_ticker):
        """Verifica integración completa cuando Finnhub no tiene target y se recurre a yfinance."""
        def finnhub_side_effect(endpoint, params=None, api_key="", timeout=8.0):
            if endpoint == "stock/profile2":
                return {"name": "Mastercard Inc", "ticker": "MA", "finnhubIndustry": "Credit Services", "marketCapitalization": 500000.0, "shareOutstanding": 930.0}
            if endpoint == "stock/metric":
                return {"metric": {}}
            if endpoint == "stock/price-target":
                return {}  # Finnhub sin target
            return None

        mock_finnhub_get.side_effect = finnhub_side_effect

        mock_instance = MagicMock()
        mock_instance.analyst_price_targets = {
            "mean": 667.30,
            "high": 735.0,
            "low": 550.0,
        }
        mock_yf_ticker.return_value = mock_instance

        info, inc, bs, cf = fetch_datos_fundamentales("MA", "key_test")

        assert info["targetMeanPrice"] == 667.30
        assert info["targetHighPrice"] == 735.0
        assert info["targetLowPrice"] == 550.0

        m_ttm = extraer_metricas_ttm(info, inc, bs, cf, precio_actual=600.0)
        assert m_ttm["target_mean_price"] == 667.30


