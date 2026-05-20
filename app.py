"""Robotics Expert Debate — Streamlit frontend."""

import io
import json
import os
import re
import datetime
from pathlib import Path

import streamlit as st
import yaml
from dotenv import load_dotenv
from openai import OpenAI

try:
    import requests
    from bs4 import BeautifulSoup
    FETCH_AVAILABLE = True
except ImportError:
    FETCH_AVAILABLE = False

try:
    from pypdf import PdfReader as _PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

from agents.orchestrator import DebateOrchestrator
from utils.obsidian import save_to_obsidian

load_dotenv()

# ── Provider config ────────────────────────────────────────────────────────────
PROVIDERS = {
    "Ollama (Local — no key needed)": {
        "base_url": "http://localhost:11434/v1",
        "models": ["llama3.1", "llama3.3", "mistral", "gemma3"],
        "key_env": None,
        "requires_key": False,
        "note": "Runs 100% locally. Install at [ollama.com](https://ollama.com), then run `ollama pull llama3.1` once.",
    },
    "Anthropic (Claude)": {
        "base_url": "https://api.anthropic.com/v1",
        "models": ["claude-sonnet-4-6", "claude-haiku-4-5-20251001", "claude-opus-4-7"],
        "key_env": "ANTHROPIC_API_KEY",
        "requires_key": True,
        "note": "$5 free credit. Sign up at [console.anthropic.com](https://console.anthropic.com).",
    },
    "Groq (Free cloud)": {
        "base_url": "https://api.groq.com/openai/v1",
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        "key_env": "GROQ_API_KEY",
        "requires_key": True,
        "note": "Free, no credit card. Sign up at [console.groq.com](https://console.groq.com).",
    },
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-4o", "gpt-4o-mini"],
        "key_env": "OPENAI_API_KEY",
        "requires_key": True,
        "note": "Requires billing. Sign up at [platform.openai.com](https://platform.openai.com).",
    },
}

# ── Expert tags ────────────────────────────────────────────────────────────────
DEFAULT_TAGS = {
    "Pieter Abbeel": ["Foundation Models", "Imitation Learning", "Manipulation", "LLMs"],
    "Rodney Brooks": ["Embodied Intelligence", "Timeline Skepticism", "Dexterity"],
    "Ken Goldberg": ["Human-Robot Collaboration", "Uncertainty", "Surgical Robotics"],
    "Gill Pratt": ["Amplification", "Safety", "Automotive", "Demographics"],
    "Fei-Fei Li": ["Spatial Intelligence", "Computer Vision", "World Models"],
    "Andrej Karpathy": ["Software 2.0", "LLMs", "Autonomous Vehicles", "Synthetic Data"],
    "Vinod Khosla": ["VC", "Labor Disruption", "AGI", "Healthcare"],
    "Josh Wolfe (Lux Capital)": ["VC", "Deep Tech", "Hard Tech", "Physical Intelligence"],
    "General Catalyst (Teresa Carlson)": ["VC", "Industrial", "Cobots", "Enterprise"],
    "Wang Xingxing": ["Hardware", "Manufacturing", "China", "Humanoids"],
    "He Xiaopeng": ["EV-Robotics", "China", "Humanoids", "Autonomous Vehicles"],
    "Dario Amodei": ["AI Safety", "AGI", "Frontier AI", "Physical AI"],
}


def get_tags(persona: dict) -> list:
    return persona.get("tags") or DEFAULT_TAGS.get(persona["name"], [])


# ── Completeness score ─────────────────────────────────────────────────────────
def completeness_score(persona: dict) -> int:
    """
    Score 0–100 reflecting how much enriched data the persona has.
    Breakdown: identity 20 | thesis 15 | positions 20 | skeptical 10
               style 10 | articles-with-urls 15 | data_sources bonus 10
    """
    score = 0
    if persona.get("title"):
        score += 10
    if persona.get("affiliation"):
        score += 10
    thesis = persona.get("core_thesis", "")
    if len(thesis) > 100:
        score += 15
    elif thesis:
        score += 7
    positions = persona.get("known_positions", [])
    score += min(20, len(positions) * 5)
    skeptical = persona.get("skeptical_of", [])
    score += min(10, len(skeptical) * 5)
    style = persona.get("rhetorical_style", "")
    if len(style) > 50:
        score += 10
    elif style:
        score += 5
    articles_with_urls = [
        a for a in persona.get("seminal_articles", []) if a.get("url")
    ]
    score += min(15, len(articles_with_urls) * 4)
    data_sources = persona.get("data_sources", [])
    score += min(10, len(data_sources) * 3)
    return min(100, score)


def score_label(score: int) -> str:
    if score >= 80:
        return "High"
    if score >= 50:
        return "Medium"
    return "Low"


def render_tag_pills(tags: list) -> str:
    pills = []
    for t in tags:
        pills.append(
            f'<span style="background:#1e3a5f;color:#7dd3fc;padding:2px 9px;'
            f'border-radius:12px;font-size:0.78em;margin:2px 3px 2px 0;'
            f'display:inline-block;white-space:nowrap">{t}</span>'
        )
    return "".join(pills)


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Robotics Expert Debate",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.round-header { text-align: center; color: #888; font-size: 0.85em;
                letter-spacing: 2px; text-transform: uppercase; padding: 4px 0; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data
def load_personas() -> dict:
    return {
        yaml.safe_load(f.read_text())["name"]: yaml.safe_load(f.read_text())
        for f in sorted(Path("personas").glob("*.yaml"))
    }


@st.cache_data
def load_debates() -> dict:
    return {
        yaml.safe_load(f.read_text())["topic"]: yaml.safe_load(f.read_text())
        for f in sorted(Path("debates").glob("*.yaml"))
    }


@st.cache_data
def load_research() -> list:
    research_dir = Path("research")
    if not research_dir.exists():
        return []
    items = []
    for f in sorted(research_dir.glob("*.yaml"), reverse=True):
        try:
            items.append(yaml.safe_load(f.read_text()))
        except Exception:
            pass
    return items


# ── LLM prompts ────────────────────────────────────────────────────────────────
TOPIC_PROMPT = """You are designing a structured debate for robotics researchers, investors, and executives.

The user wants to debate: "{title}"

Return a JSON object with exactly these fields:

{{
  "description": "2-3 sentence explanation of why this is a contested, important question in robotics right now",
  "opening_question": "A sharp, specific opening question (2-4 sentences) that forces participants to take a real position. Name specific companies, technologies, or data points to make it concrete.",
  "follow_up_prompts": [
    "A harder follow-up that pushes participants to address a specific counterargument or name a concrete number/timeline",
    "A second follow-up that introduces a new angle or forces the weakest position to be defended"
  ]
}}

Rules:
- Make the opening question genuinely contestable — not one that everyone agrees on.
- Reference real companies, products, or recent events where possible.
- Return only the JSON object, no other text."""

RESEARCH_PROMPT = """You are a research assistant building a persona profile for a robotics debate simulation.

Research {name} from {affiliation} and return a JSON object with exactly these fields:

{{
  "title": "their current job title",
  "core_thesis": "their main argument about robotics in 2-3 sentences",
  "known_positions": [
    "specific position or claim they have made publicly",
    "another specific position",
    "another specific position",
    "another specific position"
  ],
  "skeptical_of": [
    "something they push back on or are critical of",
    "another thing they are skeptical of"
  ],
  "rhetorical_style": "one sentence describing how they argue — e.g. empiricist, contrarian, visionary",
  "seminal_articles": [
    {{"title": "Title of a real article, paper, or interview they published", "url": "https://real-url-if-known-otherwise-omit"}},
    {{"title": "Another real piece of work", "url": "https://real-url-if-known-otherwise-omit"}}
  ]
}}

Rules:
- Only include positions and articles you are confident are accurate based on their actual public record.
- For URLs: only include if you are highly confident the URL is real and correct. If unsure, use an empty string.
- Keep core_thesis and rhetorical_style concise.
- Return only the JSON object, no other text."""

PROCESS_URL_PROMPT = """You are extracting structured knowledge from a web article for a robotics research knowledge base.

URL: {url}
Content (excerpt):
{content}

{attribution}

Return a JSON object with exactly these fields:
{{
  "title": "A short descriptive title for this piece (under 12 words)",
  "key_insights": "2-4 sentences capturing what matters most here for robotics research or investment thesis building",
  "tags": ["2-5 tags — choose from: Foundation Models, Humanoids, Hardware, Software, China, VC Funding, Commercial Timeline, Dexterity, Autonomous Vehicles, LLMs, AI Safety, Labor, Manufacturing, Policy, Spatial Intelligence, Simulation"]
}}

Return only valid JSON, no markdown, no other text."""

PROCESS_PDF_PROMPT = """You are distilling a research paper or document for a robotics knowledge base.

Filename: {filename}
Content (excerpt):
{content}

{attribution}

Return a JSON object with exactly these fields:
{{
  "title": "A short descriptive title for this document (under 12 words)",
  "key_insights": "3-5 sentences distilling the most important findings, arguments, or data points relevant to robotics research or investment",
  "tags": ["2-5 tags — choose from: Foundation Models, Humanoids, Hardware, Software, China, VC Funding, Commercial Timeline, Dexterity, Autonomous Vehicles, LLMs, AI Safety, Labor, Manufacturing, Policy, Spatial Intelligence, Simulation"]
}}

Return only valid JSON, no markdown, no other text."""


# ── LLM helpers ───────────────────────────────────────────────────────────────
def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def research_topic(title: str, client: OpenAI, model: str) -> dict:
    prompt = TOPIC_PROMPT.format(title=title)
    response = client.chat.completions.create(
        model=model,
        max_tokens=600,
        messages=[
            {"role": "system", "content": "You are a precise research assistant. Return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
    )
    return _parse_json(response.choices[0].message.content)


def research_expert(name: str, affiliation: str, client: OpenAI, model: str) -> dict:
    prompt = RESEARCH_PROMPT.format(name=name, affiliation=affiliation)
    response = client.chat.completions.create(
        model=model,
        max_tokens=800,
        messages=[
            {"role": "system", "content": "You are a precise research assistant. Return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
    )
    return _parse_json(response.choices[0].message.content)


def gdrive_to_direct(url: str) -> str:
    """Convert a Google Drive share link to a direct-download URL."""
    m = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    if m:
        return f"https://drive.google.com/uc?export=download&id={m.group(1)}"
    return url


def fetch_url_text(url: str) -> str:
    if not FETCH_AVAILABLE:
        return ""
    try:
        target = gdrive_to_direct(url)
        resp = requests.get(target, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        # If the response looks like a PDF, extract text from it
        if "application/pdf" in resp.headers.get("Content-Type", ""):
            return _extract_pdf_bytes(resp.content)
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)[:2500]
    except Exception:
        return ""


def _extract_pdf_bytes(data: bytes, max_chars: int = 2000) -> str:
    if not PDF_AVAILABLE:
        return ""
    try:
        reader = _PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)[:max_chars]
    except Exception:
        return ""


def extract_pdf_text(uploaded_file, max_chars: int = 2000) -> str:
    uploaded_file.seek(0)
    return _extract_pdf_bytes(uploaded_file.read(), max_chars)


def _call_llm_with_retry(client: OpenAI, model: str, prompt: str, max_tokens: int) -> str:
    """Call the LLM, retrying once with a truncated prompt on connection errors.

    Ollama's default context window is 2048 tokens. If the combined prompt
    exceeds that, Ollama drops the TCP connection (reported as APIConnectionError).
    On failure we shorten the content block and retry once.
    """
    from openai import APIConnectionError

    def _call(p: str) -> str:
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": "Return only valid JSON."},
                {"role": "user", "content": p},
            ],
        )
        return response.choices[0].message.content

    try:
        return _call(prompt)
    except APIConnectionError:
        # Truncate everything after "Content (excerpt):" or "Content:" to 800 chars
        for marker in ["Content (excerpt):\n", "Content:\n"]:
            if marker in prompt:
                head, rest = prompt.split(marker, 1)
                truncated = head + marker + rest[:800] + "\n[truncated]\n" + rest.split("\n\n")[-1]
                return _call(truncated)
        raise


def process_pdf(uploaded_file, attribution: str, client: OpenAI, model: str) -> dict:
    filename = uploaded_file.name
    content = extract_pdf_text(uploaded_file, max_chars=2000) or "(Could not extract PDF text)"
    attr_ctx = (
        f"This document is being attributed to expert: {attribution}. "
        "Focus your distillation on insights relevant to their known positions."
        if attribution != "Independent Research"
        else "This is independent research not attributed to a specific expert."
    )
    prompt = PROCESS_PDF_PROMPT.format(filename=filename, content=content, attribution=attr_ctx)
    raw = _call_llm_with_retry(client, model, prompt, max_tokens=500)
    result = _parse_json(raw)
    result["filename"] = filename
    result["source_type"] = "pdf"
    result["date_added"] = str(datetime.date.today())
    result["attribution"] = attribution
    return result


def process_url(url: str, attribution: str, client: OpenAI, model: str) -> dict:
    content = fetch_url_text(url) or "(Could not fetch content — analysing URL and context only)"
    attr_ctx = (
        f"This article is being attributed to expert: {attribution}. "
        "Focus your extraction on insights relevant to their known positions."
        if attribution != "Independent Research"
        else "This is independent research not attributed to a specific expert."
    )
    prompt = PROCESS_URL_PROMPT.format(url=url, content=content, attribution=attr_ctx)
    response = client.chat.completions.create(
        model=model,
        max_tokens=400,
        messages=[
            {"role": "system", "content": "Return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
    )
    result = _parse_json(response.choices[0].message.content)
    result["url"] = url
    result["date_added"] = str(datetime.date.today())
    result["attribution"] = attribution
    return result


# ── Persistence helpers ────────────────────────────────────────────────────────
def save_debate_yaml(debate: dict) -> Path:
    safe = debate["topic"].lower().replace(" ", "_").replace("?", "").replace(":", "").replace("/", "_")[:50]
    path = Path("debates") / f"{safe}.yaml"
    path.write_text(yaml.dump(debate, allow_unicode=True, default_flow_style=False, sort_keys=False, width=100))
    load_debates.clear()
    return path


def save_persona_yaml(persona: dict) -> Path:
    safe = persona["name"].lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
    path = Path("personas") / f"{safe}.yaml"
    path.write_text(yaml.dump(persona, allow_unicode=True, default_flow_style=False, sort_keys=False, width=100))
    load_personas.clear()
    return path


def save_source_to_persona(persona_name: str, source: dict, all_personas: dict) -> None:
    persona = dict(all_personas[persona_name])
    sources = list(persona.get("data_sources", []))
    sources.append({
        "url": source["url"],
        "title": source.get("title", source["url"]),
        "date_added": source.get("date_added", str(datetime.date.today())),
        "key_insights": source.get("key_insights", ""),
        "tags": source.get("tags", []),
    })
    persona["data_sources"] = sources
    save_persona_yaml(persona)


def save_independent_research(source: dict) -> Path:
    Path("research").mkdir(exist_ok=True)
    slug = source.get("title", source["url"])[:40].lower().replace(" ", "_").replace("?", "")
    filename = f"{source.get('date_added', str(datetime.date.today()))}_{slug}.yaml"
    path = Path("research") / filename
    path.write_text(yaml.dump(source, allow_unicode=True, default_flow_style=False, sort_keys=False))
    load_research.clear()
    return path


# ── Session state ──────────────────────────────────────────────────────────────
for key, default in [
    ("debate_history", []),
    ("debate_complete", False),
    ("debate_topic", ""),
    ("debate_experts", []),
    ("custom_personas", {}),
    ("custom_debates", {}),
    ("researched_expert", None),
    ("researched_topic", None),
    ("processed_sources", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

personas = {**load_personas(), **st.session_state.custom_personas}
debates = {**load_debates(), **st.session_state.custom_debates}

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configure Debate")

    provider_name = st.selectbox("LLM Provider", options=list(PROVIDERS.keys()))
    provider = PROVIDERS[provider_name]

    st.caption(provider["note"])
    if provider["requires_key"]:
        api_key = st.text_input(
            "API Key",
            value=os.getenv(provider["key_env"], ""),
            type="password",
        )
    else:
        api_key = "ollama"

    st.divider()

    # ── Debate topic ──────────────────────────────────────────────────────────
    selected_topic = st.selectbox("Debate Topic", options=list(debates.keys()))
    if selected_topic:
        st.caption(debates[selected_topic]["description"])

    with st.expander("➕ Suggest a debate topic"):
        ct_name = st.text_input("Topic title", placeholder="e.g. Will dextrous hands commoditise?", key="ct_name")
        ct_save = st.selectbox(
            "Save preference",
            options=["This session only", "Save permanently (adds to repo)"],
            key="ct_save",
        )
        topic_research_btn = st.button(
            "🔍 Research & Preview",
            key="topic_research_btn",
            disabled=not ct_name,
            use_container_width=True,
        )
        if topic_research_btn and ct_name:
            with st.spinner(f'Generating debate for "{ct_name}"…'):
                try:
                    client_r = OpenAI(api_key=api_key, base_url=provider["base_url"])
                    result = research_topic(ct_name, client_r, provider["models"][0])
                    result["topic"] = ct_name
                    st.session_state.researched_topic = result
                except Exception as e:
                    st.error(f"Research failed: {e}")

        if st.session_state.researched_topic:
            t = st.session_state.researched_topic
            st.markdown(f"**{t['topic']}**")
            st.caption(t.get("description", ""))
            st.markdown(f"*Opening:* {t.get('opening_question', '')[:150]}…")
            for i, fu in enumerate(t.get("follow_up_prompts", []), 1):
                st.markdown(f"*Follow-up {i}:* {fu[:100]}…")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("✓ Add", key="add_topic_btn", use_container_width=True, type="primary"):
                    topic = st.session_state.researched_topic
                    st.session_state.custom_debates[topic["topic"]] = topic
                    if ct_save == "Save permanently (adds to repo)":
                        saved_path = save_debate_yaml(topic)
                        st.success(f"Saved to {saved_path.name}")
                    else:
                        st.success("Added for this session.")
                    st.session_state.researched_topic = None
                    st.rerun()
            with col2:
                if st.button("✗ Discard", key="discard_topic_btn", use_container_width=True):
                    st.session_state.researched_topic = None
                    st.rerun()

    with st.expander("🗑️ Delete a topic"):
        all_topic_names = list(debates.keys())
        topic_to_delete = st.selectbox("Select topic to delete", options=all_topic_names, key="topic_to_delete")
        is_permanent = topic_to_delete and (
            Path("debates") / f"{topic_to_delete.lower().replace(' ', '_').replace('?','').replace(':','').replace('/','_')[:50]}.yaml"
        ).exists()
        if is_permanent:
            st.caption("⚠️ This will delete the YAML file permanently.")
        if st.button("Delete topic", key="delete_topic_btn", disabled=not topic_to_delete, use_container_width=True):
            if topic_to_delete in st.session_state.custom_debates:
                del st.session_state.custom_debates[topic_to_delete]
            if is_permanent:
                for f in Path("debates").glob("*.yaml"):
                    if yaml.safe_load(f.read_text()).get("topic") == topic_to_delete:
                        f.unlink()
                        load_debates.clear()
                        break
            st.success(f"Deleted: {topic_to_delete}")
            st.rerun()

    st.divider()

    # ── Expert selector ───────────────────────────────────────────────────────
    selected_names = st.multiselect(
        "Select Experts (2–6)",
        options=list(personas.keys()),
        default=list(personas.keys())[:4],
        max_selections=6,
    )

    with st.expander("➕ Suggest an expert"):
        ce_name = st.text_input("Full name", placeholder="e.g. Marc Raibert", key="ce_name")
        ce_affil = st.text_input("Affiliation", placeholder="e.g. Boston Dynamics", key="ce_affil")
        ce_save = st.selectbox(
            "Save preference",
            options=["This session only", "Save permanently (adds to repo)"],
            key="ce_save",
        )
        research_btn = st.button(
            "🔍 Research & Preview",
            disabled=not (ce_name and api_key),
            use_container_width=True,
        )
        if research_btn and ce_name:
            with st.spinner(f"Researching {ce_name}…"):
                try:
                    client_r = OpenAI(api_key=api_key, base_url=provider["base_url"])
                    result = research_expert(ce_name, ce_affil, client_r, provider["models"][0])
                    result["name"] = ce_name
                    result["affiliation"] = ce_affil or result.get("affiliation", "Independent")
                    st.session_state.researched_expert = result
                except Exception as e:
                    st.error(f"Research failed: {e}")

        if st.session_state.researched_expert:
            r = st.session_state.researched_expert
            st.markdown(f"**{r['name']}** — {r.get('title', '')}")
            st.caption(r.get("affiliation", ""))
            st.markdown(f"_{r.get('core_thesis', '')}_")
            if r.get("known_positions"):
                st.markdown("**Positions:** " + " · ".join(r["known_positions"][:2]))
            if r.get("seminal_articles"):
                for a in r["seminal_articles"][:2]:
                    if a.get("url"):
                        st.markdown(f"[{a['title']}]({a['url']})")
                    else:
                        st.markdown(f"• {a['title']}")
            st.caption("⚠️ Verify facts before use — based on LLM training data.")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("✓ Add", use_container_width=True, type="primary"):
                    persona = st.session_state.researched_expert
                    st.session_state.custom_personas[persona["name"]] = persona
                    if ce_save == "Save permanently (adds to repo)":
                        saved_path = save_persona_yaml(persona)
                        st.success(f"Saved to {saved_path.name}")
                    else:
                        st.success("Added for this session.")
                    st.session_state.researched_expert = None
                    st.rerun()
            with col2:
                if st.button("✗ Discard", use_container_width=True):
                    st.session_state.researched_expert = None
                    st.rerun()

    st.divider()

    num_rounds = st.slider("Rounds", min_value=1, max_value=3, value=2)
    model_choice = st.selectbox("Model", options=provider["models"])

    st.divider()

    run_disabled = len(selected_names) < 2 or (provider["requires_key"] and not api_key)
    run_btn = st.button(
        "▶ Run Debate",
        type="primary",
        disabled=run_disabled,
        use_container_width=True,
    )
    if len(selected_names) < 2:
        st.caption("Select at least 2 experts.")
    if provider["requires_key"] and not api_key:
        st.caption("Add an API key above.")


# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🤖 Robotics Expert Debate")

tab_debate, tab_experts, tab_data = st.tabs(["🎙️ Debate", "👥 Experts & Knowledge", "📥 Data Dump"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DEBATE
# ══════════════════════════════════════════════════════════════════════════════
with tab_debate:
    if selected_topic:
        st.subheader(selected_topic)

    # Expert cards shown before a debate starts
    if not st.session_state.debate_complete and not run_btn and selected_names:
        st.markdown("#### Participating Experts")
        cols = st.columns(min(len(selected_names), 3))
        for i, name in enumerate(selected_names):
            p = personas[name]
            score = completeness_score(p)
            tags = get_tags(p)
            with cols[i % 3]:
                with st.container(border=True):
                    st.markdown(f"**{name}**")
                    st.caption(f"{p.get('title', '')} · {p.get('affiliation', '')}")
                    if tags:
                        st.markdown(render_tag_pills(tags), unsafe_allow_html=True)
                        st.write("")
                    st.markdown(f"_{p['core_thesis'][:120].rstrip()}…_")
                    label = score_label(score)
                    st.caption(f"Data completeness: **{label}** ({score}/100)")
                    st.progress(score / 100)
                    articles = p.get("seminal_articles", [])
                    if articles:
                        with st.expander("Seminal articles"):
                            for a in articles:
                                if a.get("url"):
                                    st.markdown(f"[{a['title']}]({a['url']})")
                                else:
                                    st.markdown(f"• {a['title']}")

    # ── Run debate ────────────────────────────────────────────────────────────
    def make_client(key: str, base_url: str) -> OpenAI:
        return OpenAI(api_key=key, base_url=base_url)

    if run_btn and not run_disabled:
        st.session_state.debate_history = []
        st.session_state.debate_complete = False
        st.session_state.debate_topic = selected_topic
        st.session_state.debate_experts = selected_names

        client = make_client(api_key, provider["base_url"])
        orchestrator = DebateOrchestrator(
            personas=[personas[n] for n in selected_names],
            debate=debates[selected_topic],
            client=client,
            model=model_choice,
            num_rounds=num_rounds,
        )

        for exchange in orchestrator.run():
            role = exchange.get("role", "expert")
            st.session_state.debate_history.append(exchange)

            if role == "round_header":
                st.markdown(
                    f"<div class='round-header'>{exchange['speaker']}</div>",
                    unsafe_allow_html=True,
                )
                continue

            avatar = "🎙️" if "Moderator" in exchange["speaker"] else "💬"
            with st.chat_message(exchange["speaker"], avatar=avatar):
                st.markdown(f"**{exchange['speaker']}**")
                if exchange["content"]:
                    st.write(exchange["content"])
                for a in exchange.get("articles", []):
                    if a.get("url"):
                        st.markdown(f"[{a['title']}]({a['url']})")

        st.session_state.debate_complete = True
        st.success("Debate complete.")

    elif st.session_state.debate_complete and st.session_state.debate_history:
        for exchange in st.session_state.debate_history:
            role = exchange.get("role", "expert")
            if role == "round_header":
                st.markdown(
                    f"<div class='round-header'>{exchange['speaker']}</div>",
                    unsafe_allow_html=True,
                )
                continue
            avatar = "🎙️" if "Moderator" in exchange["speaker"] else "💬"
            with st.chat_message(exchange["speaker"], avatar=avatar):
                st.markdown(f"**{exchange['speaker']}**")
                if exchange["content"]:
                    st.write(exchange["content"])
                for a in exchange.get("articles", []):
                    if a.get("url"):
                        st.markdown(f"[{a['title']}]({a['url']})")

    # ── Save to Obsidian ──────────────────────────────────────────────────────
    if st.session_state.debate_complete and st.session_state.debate_history:
        st.divider()
        st.subheader("💾 Save to Obsidian")

        saveable = [
            e for e in st.session_state.debate_history
            if e.get("role") != "round_header" and e.get("content")
        ]

        with st.form("save_form"):
            st.markdown("Select exchanges to include in the note:")
            selected_indices = []
            for i, exchange in enumerate(saveable):
                preview = exchange["content"][:90].replace("\n", " ")
                if st.checkbox(f"**{exchange['speaker']}** — {preview}…", value=True, key=f"chk_{i}"):
                    selected_indices.append(i)

            vault_path = st.text_input(
                "Obsidian vault path",
                value=os.getenv(
                    "OBSIDIAN_VAULT_PATH",
                    "/Users/aarzoosharma/Documents/Market Research Agents",
                ),
            )
            submitted = st.form_submit_button("Save selected to Obsidian")

        if submitted:
            to_save = [saveable[i] for i in selected_indices]
            if not to_save:
                st.warning("Select at least one exchange to save.")
            elif not vault_path:
                st.warning("Enter the path to your Obsidian vault.")
            else:
                try:
                    path = save_to_obsidian(
                        topic=st.session_state.debate_topic,
                        experts=st.session_state.debate_experts,
                        exchanges=to_save,
                        vault_path=vault_path,
                    )
                    st.success(f"Saved → `{path}`")
                except Exception as e:
                    st.error(f"Save failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EXPERTS & KNOWLEDGE
# ══════════════════════════════════════════════════════════════════════════════
with tab_experts:
    st.subheader("Expert Profiles")

    # Collect all tags in use
    all_used_tags = sorted({t for p in personas.values() for t in get_tags(p)})
    filter_tags = st.multiselect(
        "Filter by expertise area",
        options=all_used_tags,
        placeholder="Show all",
    )

    filtered = {
        name: p for name, p in personas.items()
        if not filter_tags or any(t in get_tags(p) for t in filter_tags)
    }
    st.caption(f"Showing {len(filtered)} of {len(personas)} experts")

    cols = st.columns(3)
    for i, (name, p) in enumerate(filtered.items()):
        score = completeness_score(p)
        label = score_label(score)
        tags = get_tags(p)
        data_sources = p.get("data_sources", [])
        articles = p.get("seminal_articles", [])

        with cols[i % 3]:
            with st.container(border=True):
                st.markdown(f"**{name}**")
                st.caption(f"{p.get('title', '')} · {p.get('affiliation', '')}")

                if tags:
                    st.markdown(render_tag_pills(tags), unsafe_allow_html=True)
                    st.write("")

                # Completeness score
                st.caption(f"Data completeness: **{label}** — {score}/100")
                st.progress(score / 100)

                with st.expander("Core thesis & positions"):
                    st.markdown(p.get("core_thesis", "—"))
                    if p.get("known_positions"):
                        st.markdown("**Known positions:**")
                        for pos in p["known_positions"]:
                            st.markdown(f"• {pos}")
                    if p.get("skeptical_of"):
                        st.markdown("**Skeptical of:**")
                        for s in p["skeptical_of"]:
                            st.markdown(f"• {s}")

                if data_sources:
                    with st.expander(f"📎 Data sources ({len(data_sources)})"):
                        for ds in data_sources:
                            ds_url = ds.get("url", "")
                            ds_title = ds.get("title") or ds.get("filename") or ds_url or "Untitled"
                            if ds_url:
                                st.markdown(f"[{ds_title}]({ds_url})")
                            else:
                                st.markdown(f"**{ds_title}** _(PDF)_")
                            st.caption(ds.get("key_insights", "")[:150])
                            if ds.get("tags"):
                                st.markdown(render_tag_pills(ds["tags"]), unsafe_allow_html=True)
                            st.write("")

                if articles:
                    with st.expander("Seminal articles"):
                        for a in articles:
                            if a.get("url"):
                                st.markdown(f"[{a['title']}]({a['url']})")
                            else:
                                st.markdown(f"• {a['title']}")

    # Independent research section
    research_items = load_research()
    if research_items:
        st.divider()
        st.subheader("Independent Research")
        ri_cols = st.columns(3)
        for j, item in enumerate(research_items):
            with ri_cols[j % 3]:
                with st.container(border=True):
                    title = item.get("title", item.get("url", "Untitled"))
                    if item.get("url"):
                        st.markdown(f"**[{title}]({item['url']})**")
                    else:
                        st.markdown(f"**{title}**")
                    st.caption(f"Added {item.get('date_added', '')}")
                    if item.get("tags"):
                        st.markdown(render_tag_pills(item["tags"]), unsafe_allow_html=True)
                        st.write("")
                    st.markdown(item.get("key_insights", ""))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DATA DUMP
# ══════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.subheader("Data Dump — Enrich the Knowledge Base")
    st.markdown(
        "Feed the knowledge base with articles, interviews, papers, or PDFs. "
        "Insights are distilled by the LLM and stored under a specific expert "
        "or as independent research."
    )

    if not FETCH_AVAILABLE:
        st.info(
            "Install `requests` and `beautifulsoup4` for full URL fetching: "
            "`pip install requests beautifulsoup4`"
        )
    if not PDF_AVAILABLE:
        st.info("Install `pypdf` to enable PDF uploads: `pip install pypdf`")

    # ── Inputs (outside form so file_uploader state persists) ────────────────
    dd_col1, dd_col2 = st.columns([3, 2])
    with dd_col1:
        urls_input = st.text_area(
            "URLs — paste one per line",
            placeholder=(
                "https://techcrunch.com/...\n"
                "https://drive.google.com/file/d/ABC123/view  ← Google Drive links work too"
            ),
            height=120,
            key="dd_urls",
        )
    with dd_col2:
        uploaded_pdfs = st.file_uploader(
            "Upload PDFs",
            type="pdf",
            accept_multiple_files=True,
            key="dd_pdfs",
            help="Drag and drop one or more PDFs. Text is extracted locally and sent to the LLM.",
        )

    attribution = st.selectbox(
        "Classify under",
        options=["Independent Research"] + list(personas.keys()),
        help="Attribute to a specific expert (raises their completeness score) or file as independent research.",
        key="dd_attribution",
    )

    process_btn = st.button(
        "🔍 Fetch & Analyse",
        type="primary",
        use_container_width=True,
        disabled=(provider["requires_key"] and not api_key),
        key="dd_process",
    )

    if process_btn:
        urls = [u.strip() for u in (urls_input or "").splitlines() if u.strip()]
        pdfs = uploaded_pdfs or []
        if not urls and not pdfs:
            st.warning("Add at least one URL or upload a PDF.")
        else:
            client_d = OpenAI(api_key=api_key, base_url=provider["base_url"])
            new_sources = []
            for url in urls:
                label = url[:70] + ("…" if len(url) > 70 else "")
                with st.spinner(f"Fetching {label}"):
                    try:
                        result = process_url(url, attribution, client_d, model_choice)
                        new_sources.append(result)
                    except Exception as e:
                        st.error(f"Failed on {url}: {e}")
            for pdf in pdfs:
                with st.spinner(f"Reading {pdf.name}…"):
                    try:
                        result = process_pdf(pdf, attribution, client_d, model_choice)
                        new_sources.append(result)
                    except Exception as e:
                        st.error(f"Failed on {pdf.name}: {e}")
            st.session_state.processed_sources = new_sources

    # ── Preview ───────────────────────────────────────────────────────────────
    if st.session_state.processed_sources:
        st.markdown("---")
        st.markdown("#### Distilled Insights — Review before saving")

        for src in st.session_state.processed_sources:
            with st.container(border=True):
                attr_display = src.get("attribution", "Independent Research")
                is_pdf = src.get("source_type") == "pdf"
                title = src.get("title") or src.get("filename") or src.get("url", "Untitled")
                url_val = src.get("url", "")

                header = f"**{title}**" + (" _(PDF)_" if is_pdf else "")
                st.markdown(header)
                st.caption(f"→ {attr_display} · {src.get('date_added', '')}")
                if url_val:
                    st.markdown(f"[{url_val}]({url_val})")
                elif src.get("filename"):
                    st.caption(f"File: {src['filename']}")
                st.markdown(src.get("key_insights", ""))
                if src.get("tags"):
                    st.markdown(render_tag_pills(src["tags"]), unsafe_allow_html=True)

        st.write("")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save all to knowledge base", type="primary", use_container_width=True):
                saved = 0
                for src in st.session_state.processed_sources:
                    attr = src.get("attribution", "Independent Research")
                    try:
                        if attr != "Independent Research" and attr in personas:
                            save_source_to_persona(attr, src, personas)
                        else:
                            save_independent_research(src)
                        saved += 1
                    except Exception as e:
                        st.error(f"Could not save {src.get('url') or src.get('filename', '')}: {e}")
                if saved:
                    st.success(f"Saved {saved} source(s). Switch to the Experts tab to see them.")
                st.session_state.processed_sources = []
                st.rerun()
        with col2:
            if st.button("✗ Discard all", use_container_width=True):
                st.session_state.processed_sources = []
                st.rerun()
