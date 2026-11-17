# Trading Desk

Un desk de trading multi-agents qui raisonne avec **Claude Code en ligne de commande** (`claude -p`), sur des données de marché **point-in-time**, et qui **échoue fermé** : en cas de doute ou d'erreur, il ne trade pas.

Il couvre les actions (`AAPL`) et les **paires crypto de [Revolut X](https://developer.revolut.com/docs/api/revolut-x-crypto-exchange)** (`BTC-EUR`, `ETH-EUR`…). Il est livré avec un **backtest** qui compare le desk LLM à des stratégies simples, dans les mêmes conditions.

```
            ┌──────────────── données (Python, déterministe) ────────────────┐
            │  prix ≤ date · indicateurs · statements publiés · SPY · news   │
            └───────────────────────────────┬────────────────────────────────┘
                     ┌──────────────────────┼──────────────────────┐   en parallèle
                 technique      fondamental (actions)       news (+WebSearch en live)
                                ou structure de marché (crypto)
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

# Crypto Revolut X : le carnet d'ordres live est inclus dans l'analyse
desk analyze BTC-EUR --lang French
```

## Backtest

```bash
# Baselines seules : gratuit, quelques secondes, sur des années
desk backtest BTC-EUR --start 2022-01-01 --no-llm --max-position 100 --every 1

# Desk LLM contre baselines, sur une période postérieure à la date de coupure des modèles
desk backtest BTC-EUR --start 2026-07-01 --every 7 --max-position 100 \
  --fast-model sonnet --deep-model sonnet --knowledge-cutoff 2026-06-30
```

Les mêmes règles s'appliquent à toutes les stratégies :

- **Décision à la clôture de t**, avec les seules données disponibles jusqu'à t (testé).
- **Exécution à l'open de t+1.** Le stop-loss est exécuté au prix du stop, ou à l'open en cas de gap.
- **Coûts sur le turnover** : 15 pb par défaut en crypto (frais taker Revolut X + spread), 5 pb pour les actions. Les dérives de moins de 1 % du portefeuille ne sont pas rééquilibrées.
- **Même plafond de position** (`--max-position`) pour toutes les stratégies. Sans ça, « le desk bat le marché » voudrait juste dire « le desk était plus investi ».

| Stratégie | Règle |
|---|---|
| `llm` | Le desk complet, avec sa taille de position et son stop |
| `buy_hold` | Toujours investi au plafond |
| `trend_sma` | Investi si clôture > SMA50 et SMA20 > SMA50 |
| `momentum` | Investi si la performance sur 3 mois est positive |
| `buy_hold_100` | Référence : 100 % investi, sans plafond |

Chaque décision du LLM est sauvegardée (`results/<TICKER>/<date>/`), avec un rapport global, `equity.csv` et `backtest.json`.

**Contamination.** Déclarez `knowledge_cutoff` : le rapport indique alors si la période est vierge, c'est-à-dire postérieure à la coupure, ou contaminée. Seule une période vierge mesure vraiment le desk LLM.

## Stratégie systématique Revolut X

Le desk LLM est lent et coûteux, et son edge n'est pas démontré. En parallèle, le dépôt contient une **stratégie de tendance multi-actifs** sur 10 paires EUR de Revolut X. Elle est entièrement déterministe, et elle est évaluée selon un **protocole pré-enregistré** ([STRATEGY.md](STRATEGY.md), commité avant l'évaluation).

```bash
desk evaluate                       # protocole complet, verdict PASS/FAIL → reports/strategy_evaluation.md
desk portfolio --cash 5000 --holdings BTC-EUR=0.02   # poids cibles du jour + ordres à passer (n'envoie rien)
desk portfolio --from-revolutx      # idem, à partir de vos soldes réels (clé API)
desk paper run                      # paper trading prospectif v1/v2 aux prix bid/ask live de Revolut X (1 fois par jour)
desk paper status                   # performance prospective et verdict P1-P3 après 26 semaines
```

**Résultat historique honnête : NON VIABLE.** Le rapport est dans [`reports/strategy_evaluation.md`](reports/strategy_evaluation.md).
- **Hors échantillon 2023-2026** : CAGR +16,7 % avec un drawdown max de -36 %, contre +26,7 % et -59 % pour l'équipondéré. Le Sharpe reste inférieur (0,64 contre 0,72), et le gain de drawdown manque le seuil exigé de 0,7 point.
- **Robustesse** : seules 5 variantes sur 12 passent.
- **Ce qui fonctionne** : la protection en marché baissier (2025 à -0,7 % contre -16,5 %) et l'in-sample.
- **Ce qui ne fonctionne pas** : battre le marché haussier de 2023-2024.

Les paramètres n'ont **pas** été retouchés pour passer le test. La suite est une validation prospective de 26 semaines, sur des données que personne n'a encore vues.

## Revolut X

`trading_desk/revolutx.py` est un client REST minimal. Il signe chaque requête en Ed25519 (horodatage + méthode + chemin + query + body), conformément à la spécification Revolut X. Il couvre le carnet d'ordres public, les chandeliers, les paires, les tickers et les soldes.

```bash
openssl genpkey -algorithm ed25519 -out ~/.config/revolut-x/private.pem && chmod 600 ~/.config/revolut-x/private.pem
openssl pkey -in ~/.config/revolut-x/private.pem -pubout   # à coller dans Revolut X > Profil > Clés API
export REVOLUTX_API_KEY=...                               # la clé que Revolut X vous rend
desk doctor
```

La clé privée ne quitte jamais votre machine. Le passage d'ordres n'est **pas encore branché** : il arrivera avec une couche broker (paper, puis réel derrière des garde-fous explicites).

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
- **Données Yahoo.** Environ 5 trimestres d'états financiers seulement. Les prix sont ajustés des splits et dividendes, ce qui constitue un léger look-ahead. Pas de news historiques. En crypto, l'historique vient de Yahoo (même marché, autre carnet) : les chandeliers Revolut X demandent une clé API.
- **Mono-ticker, long only.** Pas de gestion de portefeuille multi-lignes, pas de vente à découvert.
- **Ce n'est pas un conseil en investissement.** Ne branchez pas ceci sur un compte réel sans une validation sérieuse hors échantillon.

## Développement

```bash
pytest   # 87 tests, ~1 s, sans réseau ni appel LLM (fausse CLI claude + faux provider)
```

```
trading_desk/
  llm/         claude_cli.py (subprocess async), cache.py, base.py (Protocol)
  data/        yahoo.py (point-in-time), indicators.py, models.py
  agents/      schemas.py (Pydantic = JSON schema), prompts.py, runner.py
  pipeline.py  orchestration et échec fermé
  guardrails.py
  backtest.py  simulateur commun, baselines, exécution à l'open t+1, stops, coûts
  revolutx.py  client Revolut X (signature Ed25519)
  portfolio.py stratégie de tendance multi-actifs + simulateur de portefeuille
  evaluation.py protocole pré-enregistré (STRATEGY.md) → verdict
  paper.py     paper trading prospectif aux prix Revolut X
  report.py    Markdown
  cli.py       desk analyze | backtest | doctor
```

Licence Apache-2.0.
