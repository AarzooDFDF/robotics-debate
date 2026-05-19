# Robotics Expert Debate

A multi-agent debate system where AI personas representing leading robotics researchers,
investors, and industry executives argue the field's most contested questions.

## Personas

| Name | Affiliation | Role in Debates |
|------|-------------|-----------------|
| Pieter Abbeel | UC Berkeley / Amazon AGI Lab | Foundation model optimist |
| Rodney Brooks | MIT / Robust.AI | Systematic contrarian, timeline skeptic |
| Ken Goldberg | UC Berkeley / AUTOLAB | Human-robot complementarity, uncertainty-first |
| Gill Pratt | Toyota Research Institute | Amplification paradigm, demographic lens |
| Fei-Fei Li | Stanford / World Labs | Spatial intelligence prerequisite |
| Andrej Karpathy | Eureka Labs | Software 2.0, candid pessimist |
| Vinod Khosla | Khosla Ventures | Maximalist bull VC |
| Josh Wolfe (Lux Capital) | Lux Capital | Contrarian VC, backs Physical Intelligence |
| General Catalyst (Teresa Carlson) | General Catalyst | Industrial pragmatist, cobot thesis |
| Wang Xingxing | Unitree Robotics | China hardware champion |
| He Xiaopeng | XPENG Inc. | EV-robotics convergence |
| Dario Amodei | Anthropic | Physical-world AI as the hardest frontier |

## Debate Topics

1. **Humanoid vs. Specialized Form Factors** — Is the bipedal form factor the right bet?
2. **Commercial Timeline** — When does mass-market robotics actually ship?
3. **Model vs. Hardware Bottleneck** — What's really constraining progress?
4. **Replacement vs. Amplification** — What should robots actually do?
5. **China vs. US** — Who wins the robotics race?

## Quick Start (No API Key Needed)

The app defaults to **Ollama** — runs 100% locally, no account, no cost.

```bash
# 1. Install Ollama (one time)
# Download from https://ollama.com and install

# 2. Pull a model (one time, ~4GB download)
ollama pull llama3.3

# 3. Install app dependencies
pip install -r requirements.txt

# 4. Run
streamlit run app.py
```

Select "Ollama (Local — no key needed)" in the sidebar (it's the default) and hit Run.

## Team Sharing (Streamlit Cloud — requires a cloud API key)

Ollama runs locally so it can't be used on Streamlit Cloud. For a shared team URL:

1. Get a free Groq key at [console.groq.com](https://console.groq.com) (no credit card)
2. Push this repo to GitHub
3. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
4. In Streamlit Cloud → Settings → Secrets, add:
   ```toml
   GROQ_API_KEY = "your_key_here"
   OBSIDIAN_VAULT_PATH = "/path/to/vault"
   ```
5. Share the generated URL with your team

## Saving to Obsidian

After a debate completes, the app shows a save panel. Select which exchanges to include,
confirm the vault path, and click Save. Notes are written to:

```
<vault>/Robotics Debates/<date> — <topic>.md
```

Each note includes speaker responses with clickable links to their seminal work,
tagged for easy filtering in Obsidian.

## Adding Personas

Create a new YAML file in `personas/` following the existing format:

```yaml
name: Full Name
title: Their Title
affiliation: Their Organization
core_thesis: >
  Their main argument in 2-3 sentences.
known_positions:
  - Position 1
  - Position 2
skeptical_of:
  - Thing they push back on
rhetorical_style: >
  How they argue.
seminal_articles:
  - title: "Article Title"
    url: https://link-to-article
```

## Adding Debate Topics

Create a new YAML file in `debates/` following the existing format.
