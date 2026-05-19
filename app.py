"""Robotics Expert Debate — Streamlit frontend."""

import json
import os
from pathlib import Path

import streamlit as st
import yaml
from dotenv import load_dotenv
from openai import OpenAI

from agents.orchestrator import DebateOrchestrator
from utils.obsidian import save_to_obsidian

load_dotenv()

# ── Provider config ───────────────────────────────────────────────────────────
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

# ── Page config ───────────────────────────────────────────────────────────────
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


# ── Data loading ──────────────────────────────────────────────────────────────
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


# ── Expert research via LLM ───────────────────────────────────────────────────
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
    raw = response.choices[0].message.content.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def save_persona_yaml(persona: dict) -> Path:
    """Save a persona dict as a YAML file in personas/ and clear the cache."""
    safe_name = persona["name"].lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
    path = Path("personas") / f"{safe_name}.yaml"
    path.write_text(yaml.dump(persona, allow_unicode=True, default_flow_style=False, sort_keys=False, width=100))
    load_personas.clear()
    return path


# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("debate_history", []),
    ("debate_complete", False),
    ("debate_topic", ""),
    ("debate_experts", []),
    ("custom_personas", {}),
    ("custom_debates", {}),
    ("researched_expert", None),   # holds LLM research result pending confirmation
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Merge loaded + custom (reload after any permanent saves)
personas = {**load_personas(), **st.session_state.custom_personas}
debates = {**load_debates(), **st.session_state.custom_debates}

# ── Sidebar ───────────────────────────────────────────────────────────────────
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
        with st.form("custom_topic_form", clear_on_submit=True):
            ct_name = st.text_input("Topic title", placeholder="e.g. Will dextrous hands commoditise?")
            ct_desc = st.text_area("Description (1-2 sentences)", height=68)
            ct_question = st.text_area("Opening question", height=90)
            ct_followup = st.text_area("Follow-up question (optional)", height=68)
            if st.form_submit_button("Add topic"):
                if ct_name and ct_question:
                    st.session_state.custom_debates[ct_name] = {
                        "topic": ct_name,
                        "description": ct_desc or ct_name,
                        "opening_question": ct_question,
                        "follow_up_prompts": [ct_followup] if ct_followup else [],
                    }
                    st.success(f"Added: {ct_name}")
                    st.rerun()
                else:
                    st.warning("Title and opening question are required.")

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

        # Show preview + confirm
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
                        st.success(f"Added for this session.")
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
if selected_topic:
    st.subheader(selected_topic)

if not st.session_state.debate_complete and not run_btn and selected_names:
    st.markdown("#### Participating Experts")
    cols = st.columns(min(len(selected_names), 3))
    for i, name in enumerate(selected_names):
        p = personas[name]
        with cols[i % 3]:
            with st.container(border=True):
                st.markdown(f"**{name}**")
                st.caption(f"{p.get('title', '')} · {p.get('affiliation', '')}")
                st.markdown(f"_{p['core_thesis'][:120].rstrip()}…_")
                articles = p.get("seminal_articles", [])
                if articles:
                    with st.expander("Seminal articles"):
                        for a in articles:
                            if a.get("url"):
                                st.markdown(f"[{a['title']}]({a['url']})")
                            else:
                                st.markdown(f"• {a['title']}")

# ── Run debate ────────────────────────────────────────────────────────────────
def make_client(api_key: str, base_url: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url)


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

# ── Save to Obsidian ──────────────────────────────────────────────────────────
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
