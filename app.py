"""Robotics Expert Debate — Streamlit frontend."""

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
        "note": "Runs 100% locally. Install at [ollama.com](https://ollama.com), then run `ollama pull llama3.3` once.",
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


personas = load_personas()
debates = load_debates()

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("debate_history", []),
    ("debate_complete", False),
    ("debate_topic", ""),
    ("debate_experts", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

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
        api_key = "ollama"  # Ollama accepts any non-empty string

    st.divider()

    selected_topic = st.selectbox("Debate Topic", options=list(debates.keys()))
    if selected_topic:
        st.caption(debates[selected_topic]["description"])

    st.divider()

    selected_names = st.multiselect(
        "Select Experts (2–6)",
        options=list(personas.keys()),
        default=list(personas.keys())[:4],
        max_selections=6,
    )

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

# Expert cards (pre-debate)
if not st.session_state.debate_complete and not run_btn and selected_names:
    st.markdown("#### Participating Experts")
    cols = st.columns(min(len(selected_names), 3))
    for i, name in enumerate(selected_names):
        p = personas[name]
        with cols[i % 3]:
            with st.container(border=True):
                st.markdown(f"**{name}**")
                st.caption(f"{p['title']} · {p['affiliation']}")
                st.markdown(f"_{p['core_thesis'][:120].rstrip()}…_")
                with st.expander("Seminal articles"):
                    for a in p.get("seminal_articles", []):
                        st.markdown(f"[{a['title']}]({a['url']})")

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
