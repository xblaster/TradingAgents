"""
Fonctions de recherche et d'analyse utilisant Claude CLI.

Ce module remplace les fonctions openai.py pour utiliser Claude Code
installé localement via CLI au lieu de l'API OpenAI.
"""

from .claude_cli import execute_claude, ClaudeCliError
from .config import get_config


def get_stock_news_claude(query, start_date, end_date):
    """
    Recherche des actualités sur une action via Claude CLI.

    Args:
        query: Symbole de l'action ou terme de recherche
        start_date: Date de début de la recherche
        end_date: Date de fin de la recherche

    Returns:
        Résultats de recherche au format texte

    Raises:
        ClaudeCliError: En cas d'erreur d'exécution
    """
    config = get_config()

    # Construire le prompt pour Claude
    prompt = f"""Can you search Social Media for {query} from {start_date} to {end_date}?
Make sure you only get the data posted during that period.

Please provide the results in a clear, structured format with:
- Source (Twitter, Reddit, etc.)
- Date
- Key points or sentiment
- Relevance to trading decisions"""

    system_prompt = """You are a financial research assistant specialized in gathering
and analyzing social media sentiment for trading purposes. Focus on credible sources
and provide actionable insights."""

    try:
        response = execute_claude(
            prompt=prompt,
            system_prompt=system_prompt,
            expect_json=False
        )
        return response
    except ClaudeCliError as e:
        # En cas d'erreur, retourner un message informatif
        return f"Error retrieving stock news via Claude CLI: {str(e)}"


def get_global_news_claude(curr_date, look_back_days=7, limit=5):
    """
    Recherche des actualités économiques globales via Claude CLI.

    Args:
        curr_date: Date actuelle
        look_back_days: Nombre de jours à remonter (défaut: 7)
        limit: Nombre maximum d'articles (défaut: 5)

    Returns:
        Résultats de recherche au format texte

    Raises:
        ClaudeCliError: En cas d'erreur d'exécution
    """
    config = get_config()

    # Construire le prompt pour Claude
    prompt = f"""Can you search global or macroeconomics news from {look_back_days} days
before {curr_date} to {curr_date} that would be informative for trading purposes?
Make sure you only get the data posted during that period.
Limit the results to {limit} articles.

Please provide:
- Article title and source
- Publication date
- Key economic indicators or events mentioned
- Potential impact on markets
- Trading implications"""

    system_prompt = """You are a macroeconomic analyst providing insights for trading
decisions. Focus on news that impacts markets: central bank decisions, economic data
releases, geopolitical events, and major policy changes."""

    try:
        response = execute_claude(
            prompt=prompt,
            system_prompt=system_prompt,
            expect_json=False
        )
        return response
    except ClaudeCliError as e:
        return f"Error retrieving global news via Claude CLI: {str(e)}"


def get_fundamentals_claude(ticker, curr_date):
    """
    Recherche des données fondamentales d'une action via Claude CLI.

    Args:
        ticker: Symbole de l'action
        curr_date: Date actuelle

    Returns:
        Données fondamentales au format texte/tableau

    Raises:
        ClaudeCliError: En cas d'erreur d'exécution
    """
    config = get_config()

    # Construire le prompt pour Claude
    prompt = f"""Can you search for fundamental data and discussions on {ticker}
from the month before {curr_date} to the month of {curr_date}.
Make sure you only get the data posted during that period.

Please list the results as a table with the following metrics:
- P/E Ratio (Price-to-Earnings)
- P/S Ratio (Price-to-Sales)
- Cash Flow metrics
- Debt-to-Equity ratio
- Revenue growth
- Earnings per Share (EPS)
- Any other relevant fundamental metrics

Also include:
- Recent analyst ratings or price targets
- Key financial events or announcements
- Management commentary if available"""

    system_prompt = """You are a fundamental analysis expert. Provide accurate,
up-to-date fundamental data for stocks, focusing on metrics that matter for
value investing and financial analysis."""

    try:
        response = execute_claude(
            prompt=prompt,
            system_prompt=system_prompt,
            expect_json=False
        )
        return response
    except ClaudeCliError as e:
        return f"Error retrieving fundamentals via Claude CLI: {str(e)}"


# Aliases pour compatibilité avec l'interface existante
# Ces fonctions peuvent être utilisées comme drop-in replacement pour les fonctions OpenAI
get_stock_news_claudecli = get_stock_news_claude
get_global_news_claudecli = get_global_news_claude
get_fundamentals_claudecli = get_fundamentals_claude
