# Trading Desk

Un desk de trading multi-agents qui raisonne avec **Claude Code en ligne de commande** (`claude -p`), sur des données de marché **point-in-time**, et qui **échoue fermé** : en cas de doute ou d'erreur, il ne trade pas.

```
            ┌──────────────── données (Python, déterministe) ────────────────┐
            │  prix ≤ date · indicateurs · statements publiés · SPY · news   │
            └───────────────────────────────┬────────────────────────────────┘
                     ┌──────────────────────┼──────────────────────┐   en parallèle
                 technique             fondamental              news (+WebSearch en live)
                     └──────────────────────┼──────────────────────┘
                                 bull ⇄ bear  (N rounds)
                                            │
                                   risk manager  →  taille max, stop, veto
                                            │
                                 portfolio manager  →  décision typée
                                            │
                              garde-fous déterministes (code, pas LLM)
```

> Réécriture complète d'un fork de [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents). Elle reprend l'idée (une équipe d'agents qui débat) mais aucune ligne de code.

## Principes

| Principe | Concrètement |
|---|---|
| **Les données ne viennent jamais du LLM** | Prix, indicateurs et états financiers sont calculés en Python. Le modèle raisonne sur un snapshot figé, sans tool calling, donc il ne peut pas inventer de chiffres. |
| **Point-in-time** | Les barres sont coupées à la date d'analyse. Un trimestre n'est visible que 45 jours après sa clôture. Le profil live, les objectifs d'analystes et les news ne sont utilisés qu'en mode live. |
| **Sortie structurée validée deux fois** | `--json-schema` côté CLI, puis Pydantic. Une réponse invalide a droit à une tentative de réparation, pas plus. |
| **Échec fermé** | Moins de 2 analystes OK, risk manager absent, PM en erreur : la décision est HOLD, avec la raison enregistrée. Jamais de trade construit sur un message d'erreur. |
| **Le LLM propose, le code dispose** | Plafonds de taille, conviction minimale, veto du risk manager, stop sous le prix, cohérence action/cible. |
| **Auditable et reproductible** | Chaque appel (modèle, coût, durée, sortie) est enregistré dans `run.json`, avec un rapport Markdown. Un cache disque rend les relances gratuites. |

## Appel à Claude

Chaque agent fait un seul appel isolé et non interactif :

```bash
echo "<prompt>" | claude -p --output-format json --model sonnet \
  --system-prompt "<rôle>" --json-schema '<schéma>' \
  --tools "" --setting-sources "" --strict-mcp-config --no-session-persistence
```

- Le prompt passe par **stdin**, donc il n'est pas limité par `ARG_MAX` et n'apparaît pas dans `ps`.
- Les appels sont asynchrones (`asyncio.create_subprocess_exec`), avec un sémaphore de concurrence, un timeout qui tue le process, des retries avec backoff (sauf sur les erreurs d'authentification) et un `--fallback-model`.
- L'appel est isolé de votre configuration : ni settings, ni hooks, ni MCP, ni CLAUDE.md. Il tourne dans un répertoire temporaire vide.
- `structured_output` est lu directement. Si le modèle a répondu en texte, le JSON est extrait du texte.

Pas de SDK Anthropic ni de LangChain : uniquement `claude`, `pydantic`, `pandas` et `yfinance`.

## Installation

```bash
pip install -e ".[dev]"
desk doctor        # vérifie la CLI claude (appel structuré réel) et Yahoo
```

Prérequis : Python ≥ 3.11 et [Claude Code](https://code.claude.com) installé et connecté.

## Utilisation

```bash
# Analyse live (avec recherche web pour l'analyste news)
desk analyze NVDA --lang French

# Position existante de 8 %, à une date passée, sortie JSON
desk analyze AAPL --date 2026-06-12 --position 8 --json

# Modèles moins chers
desk analyze MSFT --fast-model haiku --deep-model sonnet

# Backtest walk-forward : décision tous les 5 jours, exécution à l'open suivant
desk backtest MSFT --start 2026-09-01 --end 2026-10-31 --every 5
```

Les résultats sont écrits dans `results/<TICKER>/<date>/report.md` et `run.json`. Configuration complète : `desk.example.toml`.

En Python :

```python
import asyncio
from datetime import date
from trading_desk import DeskSettings, TradingDesk
from trading_desk.cli import build_backend
from trading_desk.data import YahooProvider

settings = DeskSettings(language="French")
desk = TradingDesk(settings, build_backend(settings), YahooProvider())
run = asyncio.run(desk.run("NVDA", date.today(), current_position_pct=0))
print(run.decision.action, run.decision.target_position_pct, run.cost_usd)
```

`LLMBackend` et `DataProvider` sont des `Protocol` : pour brancher un autre broker ou une autre source, il suffit d'implémenter `snapshot()` et `prices()`.

## Coûts et latence (mesurés)

| | Appels | Durée | Coût API équivalent |
|---|---|---|---|
| `analyze`, sonnet + opus, live | 7 | ~80 s | ~0,55 $ |
| `analyze`, haiku + sonnet, historique | 6 | ~30 s | ~0,10 $ |
| Relance identique (cache) | 0 | ~1 s | 0 $ |

## Limites, honnêtement

- **Biais de look-ahead des LLM.** Des données point-in-time n'empêchent pas un modèle de *se souvenir* de ce qui s'est passé. Un backtest sur une période antérieure à la date de coupure des modèles est un test de plomberie, pas la preuve d'un edge.
- **Le desk est conservateur par construction.** Le PM préfère HOLD quand les signaux divergent, et la conviction minimale est de 0,55. Sur nos essais, il est resté hors marché sur des hausses modestes. C'est un choix de design, réglable via `risk.min_conviction`, qu'il ne faut pas optimiser sur un backtest.
- **Données Yahoo.** Environ 5 trimestres d'états financiers seulement. Les prix sont ajustés des splits et dividendes, ce qui constitue un léger look-ahead. Pas de news historiques.
- **Mono-ticker, long only.** Pas de gestion de portefeuille multi-lignes, pas de vente à découvert.
- **Ce n'est pas un conseil en investissement.** Ne branchez pas ceci sur un compte réel sans une validation sérieuse hors échantillon.

## Développement

```bash
pytest   # 49 tests, ~1 s, sans réseau ni appel LLM (fausse CLI claude + faux provider)
```

```
trading_desk/
  llm/         claude_cli.py (subprocess async), cache.py, base.py (Protocol)
  data/        yahoo.py (point-in-time), indicators.py, models.py
  agents/      schemas.py (Pydantic = JSON schema), prompts.py, runner.py
  pipeline.py  orchestration et échec fermé
  guardrails.py
  backtest.py  walk-forward, exécution à l'open t+1, stops, coûts
  report.py    Markdown
  cli.py       desk analyze | backtest | doctor
```

Licence Apache-2.0.
