# Market Research Agents — Project Guidelines

## What This Is

A multi-sector market research system for deep-dive investment analysis.
The primary interface is the **`/market-research [sector]`** Claude Code skill —
it runs research → expert debate → Obsidian export entirely within Claude.
Python modules in `agents/` and `utils/` are available for programmatic use.

## How to Use

In VS Code or any Claude Code session:

```
/market-research robotics
/market-research biotech
/market-research climate-tech
```

The skill will:
1. Web-search for recent market data on the sector
2. Load expert personas from `sectors/[sector]/personas/`
3. Run a multi-expert debate on the top contested claims
4. Save a full research note to the Obsidian vault

## Repo Structure

```
market-research-agents/
├── sectors/
│   ├── _template/              ← copy this to add a new sector
│   │   ├── personas/
│   │   │   └── example_persona.yaml
│   │   └── debates/
│   │       └── example_debate.yaml
│   └── robotics/               ← fully populated example sector
│       ├── research/           ← market reports (YAML, one per report)
│       ├── personas/           ← one YAML per expert
│       └── debates/            ← one YAML per debate topic
├── agents/
│   ├── expert.py               ← ExpertAgent: voices a persona in debate
│   ├── moderator.py            ← ModeratorAgent: opens, synthesizes, follows up
│   ├── orchestrator.py         ← DebateOrchestrator: runs the debate loop
│   └── researcher.py           ← ResearchAgent: web results → structured brief
└── utils/
    └── obsidian.py             ← save_to_obsidian(): sector-aware note writer
```

## Adding a New Sector

1. `cp -r sectors/_template sectors/<sector-name>`
2. Drop any market reports as YAML files into `sectors/<sector-name>/research/` (see schema below).
3. Add persona YAMLs in `sectors/<sector-name>/personas/` — one file per expert.
4. Add debate YAMLs in `sectors/<sector-name>/debates/` — one file per topic.
5. Run `/market-research <sector-name>` — the Obsidian vault subfolder is created automatically.

### Research Report YAML Schema

Filename convention: `YYYY-MM-DD_short-title.yaml`

```yaml
title: "Report Title (Source)"
key_insights: >
  Synthesized findings in 3-5 sentences — this is the primary field
  the skill reads. Be specific: name companies, numbers, frameworks.
tags:
  - Foundation Models
  - Hardware
  - Investment Frameworks
url: https://original-source-url
date_added: 'YYYY-MM-DD'
attribution: Independent Research   # or: GP Research, LP Shared, etc.
```

### Persona YAML Schema

```yaml
name: Full Name
title: Their Title
affiliation: Their Organization
core_thesis: >
  2-3 sentence thesis — the claim they'd stake their reputation on.
known_positions:
  - Concrete, sourced position with date and context
  - Another position
skeptical_of:
  - Something they consistently push back on
rhetorical_style: >
  How they argue — methodical, polemical, data-first, etc.
seminal_articles:
  - title: "Article Title"
    url: https://link
```

### Debate YAML Schema

```yaml
topic: "Short debate title — the core tension"
opening_question: >
  The exact question the moderator poses. Specific enough to force a position.
follow_up_prompts:
  - "Sharper follow-up for round 2"
  - "Follow-up for round 3"
context: >
  Optional: paste in a research brief excerpt here as grounding context.
```

## Obsidian Vault

**Default:** `~/Documents/Market Research Vault/`
**Override:** set `OBSIDIAN_VAULT_PATH` env var

Notes land at: `<vault>/<Sector>/<YYYY-MM-DD> — <topic>.md`

Each note contains:
- YAML frontmatter (date, sector, experts, tags, sources)
- Research brief
- Full debate transcript with expert citations

## Python Usage (Programmatic)

```python
from agents.orchestrator import DebateOrchestrator
from utils.obsidian import save_to_obsidian
from openai import OpenAI
import yaml

client = OpenAI(api_key="...", base_url="...")
personas = [yaml.safe_load(f.read_text()) for f in Path("sectors/robotics/personas").glob("*.yaml")]
debate = yaml.safe_load(Path("sectors/robotics/debates/humanoid_vs_specialized.yaml").read_text())

orchestrator = DebateOrchestrator(personas, debate, client, model="claude-sonnet-4-6")
exchanges = list(orchestrator.run())

path = save_to_obsidian(
    topic=debate["topic"],
    experts=[p["name"] for p in personas],
    exchanges=exchanges,
    sector="robotics",
)
```

## GitHub Workflow

- One branch per sector addition: `add-sector/biotech`
- Persona and debate YAMLs are the primary review surface — no code change needed for new sectors
- Never commit `.env` or vault note contents
