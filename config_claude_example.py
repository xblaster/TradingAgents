"""
Exemple de configuration pour utiliser Claude Code CLI avec TradingAgents.

Ce fichier montre comment configurer TradingAgents pour utiliser Claude Code
installé localement via CLI au lieu de l'API OpenAI.

Usage:
    from config_claude_example import CLAUDE_CONFIG
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    graph = TradingAgentsGraph(config=CLAUDE_CONFIG)
"""

import os

CLAUDE_CONFIG = {
    # Répertoires
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", "./results"),
    "data_dir": "/path/to/your/data",  # Modifier selon votre environnement
    "data_cache_dir": os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
        "tradingagents/dataflows/data_cache",
    ),

    # Configuration LLM - CLAUDE CLI
    "llm_provider": "claude",  # Utiliser Claude CLI au lieu d'OpenAI
    "deep_think_llm": "claude-sonnet-4-5",  # Modèle pour la réflexion profonde
    "quick_think_llm": "claude-sonnet-4-5",  # Modèle pour les tâches rapides
    "backend_url": None,  # Non utilisé pour Claude CLI

    # Configuration spécifique à Claude CLI
    "claude_timeout": 300,  # Timeout de 5 minutes pour les appels CLI

    # Configuration des débats et discussions
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,

    # Configuration des vendors de données
    # Vous pouvez mélanger les sources : utiliser Claude pour l'analyse,
    # mais d'autres vendors pour les données de marché
    "data_vendors": {
        "core_stock_apis": "yfinance",        # Données de prix via yfinance
        "technical_indicators": "yfinance",   # Indicateurs techniques via yfinance
        "fundamental_data": "claude",         # Analyse fondamentale via Claude CLI
        "news_data": "claude",                # Analyse de news via Claude CLI
    },

    # Configuration au niveau des outils (optionnel)
    # Remplace la configuration par catégorie pour des outils spécifiques
    "tool_vendors": {
        # Exemple : forcer certains outils à utiliser Claude
        # "get_fundamentals": "claude",
        # "get_news": "claude",
        # "get_global_news": "claude",
    },
}


# Configuration hybride : Claude pour l'analyse, API pour les données
HYBRID_CONFIG = {
    **CLAUDE_CONFIG,

    # Utiliser Claude CLI pour les LLMs
    "llm_provider": "claude",

    # Mais utiliser des APIs traditionnelles pour les données
    "data_vendors": {
        "core_stock_apis": "yfinance",
        "technical_indicators": "yfinance",
        "fundamental_data": "alpha_vantage",  # API Alpha Vantage pour les données
        "news_data": "claude",                 # Claude CLI pour l'analyse de news
    },
}


# Configuration tout-Claude : maximise l'utilisation de Claude CLI
ALL_CLAUDE_CONFIG = {
    **CLAUDE_CONFIG,

    "llm_provider": "claude",

    "data_vendors": {
        "core_stock_apis": "yfinance",  # yfinance reste recommandé pour les prix
        "technical_indicators": "yfinance",  # yfinance pour les indicateurs
        "fundamental_data": "claude",   # Claude pour l'analyse fondamentale
        "news_data": "claude",          # Claude pour l'analyse de news
    },

    "tool_vendors": {
        "get_fundamentals": "claude",
        "get_news": "claude",
        "get_global_news": "claude",
    },
}
