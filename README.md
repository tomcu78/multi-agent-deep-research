# Multi-Agent Deep Research & Synthesizer

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph%20%2F%20StateGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Moteur de recherche approfondie et de synthèse autonome multi-agents inspiré par **Deep Research**. Développé avec **Python 3.11+**, **LangGraph / StateGraph**, **Pydantic**, **Asyncio**, **Rich CLI** et **FastAPI**.

Interface web épurée inspirée du design de **[cottutom.fr](https://cottutom.fr)** et **Linear** (fond ivoire `#fdfcf7`, typographie soignée, zéro bruit visuel).

---

## Architecture Multi-Agents

Le système orchestre 4 agents spécialisés à travers une machine à états `StateGraph` avec typage strict et boucles de réflexion :

```mermaid
flowchart TD
    UserQuery([Prompt / Sujet de Recherche]) --> PlannerAgent[1. Planner Agent]
    
    subgraph Planning ["1. Cadrage & Stratégie"]
        PlannerAgent -->|Structured Output| ResearchPlan[Plan de Recherche & Sous-Tâches]
    end
    
    subgraph ParallelResearch ["2. Exploration Parallélisée (Fan-Out)"]
        ResearchPlan --> SubAgent1[Researcher 1]
        ResearchPlan --> SubAgent2[Researcher 2]
        ResearchPlan --> SubAgentN[Researcher N]
        
        SubAgent1 --> Tools1[Search & Scrape Tools]
        SubAgent2 --> Tools2[Search & Scrape Tools]
        SubAgentN --> ToolsN[Search & Scrape Tools]
        
        Tools1 --> Findings[Faits Extraits & Sources Dédupliquées]
        Tools2 --> Findings
        ToolsN --> Findings
    end
    
    subgraph Reflection ["3. Fact-Checking & Reflection Loop"]
        Findings --> CriticAgent[3. Critic / Fact-Checker]
        CriticAgent --> QualityCheck{Qualité & Exhaustivité OK ?}
        QualityCheck -->|Lacunes / Besoin de révisions| PlannerAgent
        QualityCheck -->|Validé| WriterAgent[4. Writer / Synthesizer]
    end
    
    subgraph Synthesis ["4. Synthèse Sourcée"]
        WriterAgent --> FinalReport[Rapport Deep Research .md / .json]
    end

    subgraph Interfaces ["5. Interfaces"]
        FinalReport --> RichCLI[Terminal CLI (Rich)]
        FinalReport --> WebUI[Dashboard Web (FastAPI / SSE)]
    end
```

### Principes Clés
1. **Alignement d'Entité Strict & Anti-Hallucination** : Filtrage des faux positifs et de l'autocorrection des moteurs de recherche (rejet des homophones et des dérives phonétiques).
2. **Rapport Négatif Transparent** : Si une entité ou un axe n'a aucune donnée publique vérifiable sur le web ouvert, le système signale honnêtement l'absence de données sans jamais inventer de faits ni de fausses citations.
3. **Parallélisme Asynchrone (Fan-Out)** : Les sous-agents chercheurs explorent et scrapent le web en parallèle.
4. **Boucle d'Auto-Correction (Reflection Loop)** : L'agent Critique évalue la complétude, la fiabilité des sources et ajuste la stratégie avant la rédaction finale.
5. **Support Multi-Fournisseurs** : Fonctionne avec OpenAI (GPT-4o), Anthropic (Claude 3.5 Sonnet), Google Gemini, ou en mode **Web Direct** autonome sans clé d'API.

---

## Installation

### Prérequis
- **Python 3.11+**
- **[uv](https://github.com/astral-sh/uv)** (recommandé pour une installation instantanée) ou `pip`

### Méthode 1 : Avec uv (Recommandé)

```bash
# 1. Cloner le dépôt
git clone https://github.com/tomcu78/multi-agent-deep-research.git
cd multi-agent-deep-research

# 2. Créer l'environnement virtuel et installer les dépendances
uv venv
source .venv/bin/activate  # Sur macOS/Linux

# 3. Installer le projet en mode éditable
uv pip install -e .
```

### Méthode 2 : Avec pip standard

```bash
git clone https://github.com/tomcu78/multi-agent-deep-research.git
cd multi-agent-deep-research

python3 -m venv .venv
source .venv/bin/activate

pip install -e .
```

---

## Configuration (Optionnel)

Copiez le fichier `.env.example` vers `.env` pour renseigner vos clés d'API :

```bash
cp .env.example .env
```

Contenu du `.env` :
```ini
# Fournisseur LLM par défaut (openai, anthropic, google, ou mock)
DEFAULT_LLM_PROVIDER=mock

# Clés API LLM (optionnelles si vous utilisez le mode Web Direct)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=

# Clé API Recherche Web (optionnelle - DuckDuckGo & Wikipedia sont actifs par défaut)
TAVILY_API_KEY=

# Paramètres de recherche
MAX_SEARCH_RESULTS_PER_QUERY=6
MAX_RESEARCH_ITERATIONS=3
MAX_SUBTASKS=3
```

> **Note :** Sans clé API, le système fonctionne à 100% en mode **Web Direct** avec scraping web en direct et extraction factuelle dynamique.

---

## Utilisation

### 1. Dashboard Web (Interface cottutom.fr / Linear)

Démarrez le serveur FastAPI :

```bash
uv run python start_server.py
# ou directement :
uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8001 --reload
```

- **Interface Web** : Rendez-vous sur **`http://localhost:8001/`**
- **Documentation OpenAPI (Swagger)** : Rendez-vous sur **`http://localhost:8001/docs`**

---

### 2. Terminal CLI Interactif (Rich)

Lancez des recherches approfondies directement depuis votre terminal avec barre de progression et télémétrie en temps réel :

```bash
# Recherche directe avec export Markdown
uv run python -m src.ui.cli --query "Architectures Multi-Agents et LangGraph" --output rapport.md

# Mode interactif
uv run python -m src.ui.cli
```

---

## Tests Automatisés

Le projet comprend une suite complète de tests unitaires et d'intégration validant le graphe d'état, les filtres d'entités, le critic loop et l'API :

```bash
uv run pytest -v
```

---

## Structure du Projet

```
multi-agent-deep-research/
├── pyproject.toml              # Configuration et dépendances
├── .env.example                # Modèle de variables d'environnement
├── README.md                   # Documentation complète
├── start_server.py             # Script de démarrage rapide du serveur
├── src/
│   ├── config.py               # Paramètres Pydantic Settings
│   ├── llm.py                  # Factory LLM & Modèle d'extraction dynamique
│   ├── models/                 # Modèles Pydantic & TypedDict
│   │   ├── state.py            # État centralisé ResearchState
│   │   ├── plan.py             # Plans & sous-tâches
│   │   ├── finding.py          # Preuves, faits extraits & citations
│   │   ├── critic.py           # Évaluations & scores du Critic
│   │   └── report.py           # Structure du rapport final Markdown
│   ├── tools/
│   │   ├── search.py           # Moteurs DuckDuckGo, Wikipedia & Tavily
│   │   └── scraper.py          # Scraper asynchrone httpx + Trafilatura
│   ├── agents/
│   │   ├── base.py             # Agent abstrait avec télémétrie
│   │   ├── planner.py          # Décomposition stratégique
│   │   ├── researcher.py       # Exploration et extraction factuelle stricte
│   │   ├── critic.py           # Fact-checking, détection de lacunes & réflexion
│   │   └── writer.py           # Synthèse sourcée et reporting honnête
│   ├── graph/
│   │   ├── state_graph.py      # Définition LangGraph StateGraph
│   │   └── workflow.py         # Exécuteur asynchrone et streaming d'événements
│   ├── ui/
│   │   ├── cli.py              # Interface terminal Rich
│   │   └── static/index.html   # Dashboard Web minimaliste
│   └── api/
│       ├── app.py              # Application FastAPI
│       ├── routes.py           # Endpoints REST & SSE streaming
│       └── schemas.py          # Schémas de validation API
└── tests/                      # Suite de tests Pytest (11 tests unitaires)
```

---

## Licence

Projet sous licence MIT. Libre d'utilisation et d'adaptation.

