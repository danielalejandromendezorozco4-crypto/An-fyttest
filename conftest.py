"""
conftest.py — Configuración global de pytest para An-FyT.

Propósito:
    Inserta un mock liviano de `streamlit` en `sys.modules` ANTES de que
    cualquier módulo del proyecto sea importado por pytest. Esto evita el
    error:
        ImportError: cannot import name 'calcular_fcff_valuation' from 'engine.valuation'
    que se produce porque la cadena de imports:
        engine.valuation → [lazy] → data.financial_fetcher → streamlit
        engine.metrics   → config.settings → streamlit
        services.ai_service → streamlit
    arrastra streamlit como dependencia de runtime, y streamlit no puede
    inicializarse en un entorno de testing puro (sin servidor HTTP, sin
    context de Streamlit).

    El mock expone únicamente las primitivas usadas por el código en nivel
    de módulo (secrets, error, stop, etc.). Las llamadas reales a la UI de
    Streamlit que ocurren dentro de funciones se interceptan igualmente.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock


def _create_streamlit_mock() -> types.ModuleType:
    """Crea un módulo mock de streamlit con las primitivas mínimas necesarias."""
    mock = types.ModuleType("streamlit")

    # secrets — devuelve un dict-like que retorna strings vacías para cualquier clave
    class _SecretsProxy(dict):
        def __getitem__(self, key: str) -> str:
            return ""
        def get(self, key: str, default=None):  # noqa: ANN001
            return default or ""

    mock.secrets = _SecretsProxy()

    # Primitivas de UI / control de flujo
    mock.error   = MagicMock()
    mock.warning = MagicMock()
    mock.info    = MagicMock()
    mock.success = MagicMock()
    mock.stop    = MagicMock()
    mock.spinner = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
    mock.cache_data = lambda func=None, **_kw: (func if func else lambda f: f)
    mock.cache_resource = lambda func=None, **_kw: (func if func else lambda f: f)

    # set_page_config y otros que se llaman en nivel de módulo de app.py
    mock.set_page_config = MagicMock()

    return mock


# Registrar el mock ANTES de que cualquier import del proyecto ocurra
if "streamlit" not in sys.modules:
    sys.modules["streamlit"] = _create_streamlit_mock()
else:
    # Si streamlit ya fue parcialmente importado (e.g., por el propio pytest-streamlit),
    # aseguramos que .secrets y .stop existan
    st = sys.modules["streamlit"]
    if not hasattr(st, "secrets"):
        st.secrets = {}
    if not hasattr(st, "stop"):
        st.stop = MagicMock()
