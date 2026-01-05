#!/usr/bin/env python3
"""
Exemple d'utilisation de TradingAgents avec Claude Code CLI.

Ce script démontre comment utiliser le système de trading avec Claude CLI
au lieu de l'API OpenAI.

Usage:
    python example_claude_usage.py
"""

import os
from datetime import date, timedelta
from tradingagents.graph.trading_graph import TradingAgentsGraph


def example_1_basic_claude():
    """Exemple basique : utiliser Claude CLI pour tous les LLMs."""
    print("=" * 80)
    print("EXEMPLE 1 : Configuration Claude CLI de base")
    print("=" * 80)

    config = {
        # Configuration LLM
        "llm_provider": "claude",
        "deep_think_llm": "claude-sonnet-4-5",
        "quick_think_llm": "claude-sonnet-4-5",
        "claude_timeout": 300,

        # Répertoires
        "project_dir": os.path.abspath("."),
        "results_dir": "./results",
        "data_cache_dir": "./tradingagents/dataflows/data_cache",

        # Vendors de données
        "data_vendors": {
            "core_stock_apis": "yfinance",
            "technical_indicators": "yfinance",
            "fundamental_data": "claude",
            "news_data": "claude",
        },
    }

    # Créer le graph
    graph = TradingAgentsGraph(
        selected_analysts=["market", "news"],
        config=config,
        debug=True
    )

    # Analyser une action
    curr_date = date.today().strftime("%Y-%m-%d")
    print(f"\n📊 Analyse de AAPL pour le {curr_date}\n")

    try:
        result = graph.execute(ticker="AAPL", curr_date=curr_date)
        print("\n✅ Résultat de l'analyse:")
        print(result)
    except Exception as e:
        print(f"\n❌ Erreur: {e}")


def example_2_hybrid():
    """Exemple hybride : Claude pour LLMs, APIs pour données."""
    print("\n" + "=" * 80)
    print("EXEMPLE 2 : Configuration hybride (Claude + APIs)")
    print("=" * 80)

    config = {
        # Claude pour les LLMs
        "llm_provider": "claude",
        "deep_think_llm": "claude-sonnet-4-5",
        "quick_think_llm": "claude-sonnet-4-5",

        # Répertoires
        "project_dir": os.path.abspath("."),
        "results_dir": "./results",
        "data_cache_dir": "./tradingagents/dataflows/data_cache",

        # APIs traditionnelles pour les données
        "data_vendors": {
            "core_stock_apis": "yfinance",
            "technical_indicators": "yfinance",
            "fundamental_data": "alpha_vantage",  # API pour données
            "news_data": "claude",                 # Claude pour analyse
        },
    }

    graph = TradingAgentsGraph(
        selected_analysts=["market", "fundamentals"],
        config=config
    )

    curr_date = date.today().strftime("%Y-%m-%d")
    print(f"\n📊 Analyse hybride de GOOGL pour le {curr_date}\n")

    try:
        result = graph.execute(ticker="GOOGL", curr_date=curr_date)
        print("\n✅ Résultat:")
        print(result)
    except Exception as e:
        print(f"\n❌ Erreur: {e}")


def example_3_direct_claude_service():
    """Exemple d'utilisation directe du service Claude CLI."""
    print("\n" + "=" * 80)
    print("EXEMPLE 3 : Utilisation directe de ClaudeCliService")
    print("=" * 80)

    from tradingagents.dataflows.claude_cli import ClaudeCliService

    service = ClaudeCliService(timeout=180)

    prompts = [
        "Quelle est la tendance actuelle du marché technologique?",
        "Quels sont les principaux risques pour les actions FAANG en 2024?",
        "Analyse rapide : Apple vs Microsoft, qui est le meilleur investissement?"
    ]

    for i, prompt in enumerate(prompts, 1):
        print(f"\n🤔 Question {i}: {prompt}")
        try:
            response = service.execute(
                prompt=prompt,
                system_prompt="Tu es un analyste financier expert. Réponds de manière concise.",
                expect_json=False
            )
            print(f"\n💡 Réponse:\n{response}\n")
        except Exception as e:
            print(f"\n❌ Erreur: {e}\n")


def example_4_langchain_wrapper():
    """Exemple d'utilisation du wrapper LangChain."""
    print("\n" + "=" * 80)
    print("EXEMPLE 4 : Wrapper LangChain pour Claude CLI")
    print("=" * 80)

    from tradingagents.dataflows.langchain_claude_cli import ChatClaudeCLI
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = ChatClaudeCLI(model="claude-sonnet-4-5", timeout=300)

    messages = [
        SystemMessage(content="Tu es un analyste quantitatif spécialisé en trading algorithmique."),
        HumanMessage(content="Explique en 3 points comment construire une stratégie de momentum trading.")
    ]

    print("\n🤖 Appel au LLM via LangChain...\n")

    try:
        response = llm.invoke(messages)
        print("✅ Réponse du LLM:")
        print(response.content)
    except Exception as e:
        print(f"❌ Erreur: {e}")


def example_5_vendor_functions():
    """Exemple d'utilisation des fonctions Claude directement."""
    print("\n" + "=" * 80)
    print("EXEMPLE 5 : Fonctions Claude pour recherche de données")
    print("=" * 80)

    from tradingagents.dataflows.claude import (
        get_stock_news_claude,
        get_global_news_claude,
        get_fundamentals_claude
    )

    curr_date = date.today()
    start_date = (curr_date - timedelta(days=7)).strftime("%Y-%m-%d")
    end_date = curr_date.strftime("%Y-%m-%d")

    print(f"\n📰 Recherche de news TSLA du {start_date} au {end_date}...\n")

    try:
        news = get_stock_news_claude("TSLA", start_date, end_date)
        print("✅ News trouvées:")
        print(news[:500] + "..." if len(news) > 500 else news)
    except Exception as e:
        print(f"❌ Erreur: {e}")

    print(f"\n🌍 Recherche de news économiques globales...\n")

    try:
        global_news = get_global_news_claude(end_date, look_back_days=7, limit=3)
        print("✅ News globales:")
        print(global_news[:500] + "..." if len(global_news) > 500 else global_news)
    except Exception as e:
        print(f"❌ Erreur: {e}")


def main():
    """Fonction principale pour exécuter les exemples."""
    print("\n" + "=" * 80)
    print("🚀 EXEMPLES D'UTILISATION DE TRADINGAGENTS AVEC CLAUDE CLI")
    print("=" * 80)

    print("\n⚠️  PRÉREQUIS:")
    print("   - Claude Code CLI doit être installé (vérifier avec: claude --version)")
    print("   - Variables d'environnement configurées si nécessaire")
    print("   - Dépendances Python installées")

    print("\n📋 Exemples disponibles:")
    print("   1. Configuration Claude CLI de base")
    print("   2. Configuration hybride (Claude + APIs)")
    print("   3. Utilisation directe de ClaudeCliService")
    print("   4. Wrapper LangChain pour Claude CLI")
    print("   5. Fonctions Claude pour recherche de données")

    print("\n" + "=" * 80)

    # Demander quel exemple exécuter
    print("\nQuel exemple voulez-vous exécuter? (1-5, ou 'all' pour tous, 'q' pour quitter)")
    choice = input("Votre choix: ").strip().lower()

    if choice == 'q':
        print("\n👋 Au revoir!")
        return

    examples = {
        '1': example_1_basic_claude,
        '2': example_2_hybrid,
        '3': example_3_direct_claude_service,
        '4': example_4_langchain_wrapper,
        '5': example_5_vendor_functions,
    }

    if choice == 'all':
        for example_func in examples.values():
            try:
                example_func()
            except Exception as e:
                print(f"\n⚠️  Erreur lors de l'exécution: {e}")
            print("\n" + "-" * 80)
    elif choice in examples:
        try:
            examples[choice]()
        except Exception as e:
            print(f"\n⚠️  Erreur lors de l'exécution: {e}")
    else:
        print(f"\n❌ Choix invalide: {choice}")

    print("\n" + "=" * 80)
    print("✅ Exemples terminés!")
    print("=" * 80)


if __name__ == "__main__":
    main()
