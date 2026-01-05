# Guide de Migration : OpenAI → Claude Code CLI

Ce document explique comment migrer votre système de trading TradingAgents d'OpenAI vers Claude Code CLI.

## 📋 Prérequis

1. **Claude Code CLI installé** : Assurez-vous que Claude Code est installé sur votre machine
   ```bash
   # Vérifier l'installation
   claude --version
   ```

2. **Dépendances Python** : Les packages LangChain nécessaires sont déjà installés

## 🎯 Architecture de la Migration

### Nouveaux Modules Créés

1. **`tradingagents/dataflows/claude_cli.py`**
   - Service de bas niveau pour exécuter Claude CLI via subprocess
   - Gère les timeouts, retries, et parsing des réponses
   - Support synchrone et asynchrone

2. **`tradingagents/dataflows/langchain_claude_cli.py`**
   - Wrapper compatible LangChain (remplace `ChatOpenAI`)
   - Implémente l'interface `BaseChatModel`
   - Permet une migration transparente du code existant

3. **`tradingagents/dataflows/claude.py`**
   - Fonctions de recherche et d'analyse via Claude CLI
   - Remplace les fonctions d'`openai.py`
   - Compatible avec le système de routing des vendors

### Modules Modifiés

1. **`tradingagents/dataflows/interface.py`**
   - Ajout de "claude" dans `VENDOR_LIST`
   - Enregistrement des fonctions Claude dans `VENDOR_METHODS`

2. **`tradingagents/graph/trading_graph.py`**
   - Support du provider "claude" pour l'initialisation des LLMs
   - Utilise `ChatClaudeCLI` quand `llm_provider="claude"`

3. **`tradingagents/default_config.py`**
   - Documentation des options Claude
   - Ajout du paramètre `claude_timeout`

## 🚀 Utilisation

### Option 1 : Configuration Simple (Tout Claude)

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph

config = {
    "llm_provider": "claude",
    "deep_think_llm": "claude-sonnet-4-5",
    "quick_think_llm": "claude-sonnet-4-5",
    "claude_timeout": 300,
    "data_vendors": {
        "fundamental_data": "claude",
        "news_data": "claude",
    }
}

graph = TradingAgentsGraph(config=config)
result = graph.execute(ticker="AAPL", curr_date="2024-01-15")
```

### Option 2 : Configuration Hybride

Utiliser Claude pour les LLMs mais conserver les APIs pour les données :

```python
config = {
    "llm_provider": "claude",          # LLMs via Claude CLI
    "deep_think_llm": "claude-sonnet-4-5",
    "quick_think_llm": "claude-sonnet-4-5",
    "data_vendors": {
        "fundamental_data": "alpha_vantage",  # Données via API
        "news_data": "alpha_vantage",         # Données via API
    }
}
```

### Option 3 : Utilisation du Fichier de Config

```python
from config_claude_example import CLAUDE_CONFIG, HYBRID_CONFIG, ALL_CLAUDE_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

# Choisir une configuration
graph = TradingAgentsGraph(config=CLAUDE_CONFIG)
```

## 🔧 Paramètres de Configuration

### Paramètres LLM

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `llm_provider` | `"claude"` | Active l'utilisation de Claude CLI |
| `deep_think_llm` | `"claude-sonnet-4-5"` | Modèle pour la réflexion profonde |
| `quick_think_llm` | `"claude-sonnet-4-5"` | Modèle pour les tâches rapides |
| `claude_timeout` | `300` (secondes) | Timeout pour les appels CLI |

### Vendors de Données

Les fonctions suivantes supportent le vendor `"claude"` :

- `get_fundamentals` : Analyse fondamentale d'une action
- `get_news` : Recherche et analyse de news sur une action
- `get_global_news` : Recherche de news macroéconomiques

Configuration dans `data_vendors` :

```python
"data_vendors": {
    "fundamental_data": "claude",  # ou "openai", "alpha_vantage"
    "news_data": "claude",         # ou "openai", "alpha_vantage", "google"
}
```

## 📊 Comparaison OpenAI vs Claude CLI

| Aspect | OpenAI API | Claude CLI |
|--------|------------|------------|
| **Coût** | Par token (variable) | Inclus avec abonnement Claude |
| **Latence** | ~1-3s | ~2-5s (subprocess overhead) |
| **Fiabilité** | Dépend du réseau | Local, pas de dépendance réseau |
| **Rate Limits** | Selon plan API | Selon limites Claude Code |
| **Streaming** | Supporté | Non supporté dans cette impl. |
| **Setup** | Clé API requise | Installation CLI requise |

## ⚙️ Utilisation Avancée

### Appels Directs au Service

Si vous voulez utiliser Claude CLI en dehors du framework :

```python
from tradingagents.dataflows.claude_cli import ClaudeCliService

service = ClaudeCliService(timeout=180)
response = service.execute(
    prompt="Analyse la situation de AAPL",
    system_prompt="Tu es un analyste financier expert",
    expect_json=False
)
print(response)
```

### Appels Asynchrones

```python
import asyncio
from tradingagents.dataflows.claude_cli import ClaudeCliService

async def analyze_multiple_stocks():
    service = ClaudeCliService()

    tasks = [
        service.execute_async("Analyse AAPL"),
        service.execute_async("Analyse GOOGL"),
        service.execute_async("Analyse MSFT"),
    ]

    results = await asyncio.gather(*tasks)
    return results

# Exécuter
results = asyncio.run(analyze_multiple_stocks())
```

### Utilisation avec LangChain

```python
from tradingagents.dataflows.langchain_claude_cli import ChatClaudeCLI
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatClaudeCLI(model="claude-sonnet-4-5", timeout=300)

messages = [
    SystemMessage(content="Tu es un analyste financier expert"),
    HumanMessage(content="Analyse les perspectives pour AAPL en 2024")
]

response = llm.invoke(messages)
print(response.content)
```

## 🐛 Dépannage

### Problème : Timeout lors des appels

**Solution** : Augmenter le timeout dans la config

```python
config = {
    "claude_timeout": 600,  # 10 minutes
}
```

### Problème : Commande `claude` non trouvée

**Solution** : Vérifier l'installation et le PATH

```bash
which claude
# Si non trouvé, installer Claude Code ou ajuster le PATH
```

**Alternative** : Spécifier le chemin complet

```python
from tradingagents.dataflows.claude_cli import ClaudeCliService

service = ClaudeCliService(claude_command="/chemin/vers/claude")
```

### Problème : Réponses mal formatées

**Solution** : Les réponses CLI peuvent contenir du texte de debug. Le wrapper inclut un nettoyage automatique, mais vous pouvez ajuster :

```python
# Dans claude_cli.py, modifier _clean_response() si nécessaire
```

### Problème : Support des outils limité

**Note** : Claude CLI a un support limité pour les outils comparé à l'API. Les agents qui dépendent fortement des outils peuvent nécessiter des ajustements.

## 📝 Limitations Connues

1. **Pas de streaming** : Le CLI ne supporte pas le streaming natif
2. **Overhead du subprocess** : Légère latence supplémentaire vs API directe
3. **Gestion des outils limitée** : Le binding d'outils est minimal
4. **Pas d'embeddings** : Claude CLI ne fournit pas d'API d'embeddings
   - La mémoire (`memory.py`) continue d'utiliser OpenAI pour les embeddings
   - Une migration complète nécessiterait un service d'embeddings alternatif

## 🎓 Exemples Complets

### Exemple 1 : Analyse Simple

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph

config = {
    "llm_provider": "claude",
    "deep_think_llm": "claude-sonnet-4-5",
    "quick_think_llm": "claude-sonnet-4-5",
    "data_vendors": {
        "fundamental_data": "claude",
        "news_data": "claude",
    }
}

graph = TradingAgentsGraph(
    selected_analysts=["market", "news", "fundamentals"],
    config=config
)

result = graph.execute(ticker="AAPL", curr_date="2024-01-15")
print(result)
```

### Exemple 2 : Migration Progressive

```python
# Commencer avec OpenAI pour certaines fonctions
config = {
    "llm_provider": "claude",  # Agents utilisent Claude
    "data_vendors": {
        "fundamental_data": "alpha_vantage",  # Données via API
        "news_data": "alpha_vantage",
    },
    "tool_vendors": {
        # Migrer progressivement outil par outil
        "get_global_news": "claude",  # Cette fonction utilise Claude
    }
}
```

## 🔄 Rollback vers OpenAI

Si vous devez revenir à OpenAI :

```python
config = {
    "llm_provider": "openai",
    "deep_think_llm": "o4-mini",
    "quick_think_llm": "gpt-4o-mini",
    "backend_url": "https://api.openai.com/v1",
    "data_vendors": {
        "fundamental_data": "openai",
        "news_data": "openai",
    }
}
```

Aucune modification de code n'est nécessaire grâce à l'abstraction du système de vendors.

## 📚 Références

- [Claude Code Documentation](https://docs.anthropic.com/claude/docs)
- [LangChain Documentation](https://python.langchain.com/)
- [TradingAgents README](./README.md)

## 🤝 Support

Pour les questions ou problèmes :
1. Vérifier ce guide de migration
2. Consulter les logs de debug
3. Ouvrir une issue sur le repo GitHub
