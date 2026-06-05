"""Save research briefs and debate exchanges as Obsidian notes. Sector-aware."""

import os
from datetime import datetime
from pathlib import Path


def save_to_obsidian(
    topic: str,
    experts: list[str],
    exchanges: list[dict],
    vault_path: str | None = None,
    sector: str = "General",
    sources: list[str] | None = None,
    research_brief: str = "",
) -> str:
    vault_path = vault_path or os.environ.get(
        "OBSIDIAN_VAULT_PATH",
        str(Path.home() / "Documents" / "Market Research Vault"),
    )
    date = datetime.now().strftime("%Y-%m-%d")
    safe_topic = topic.replace("/", "-").replace(":", "").replace("?", "")
    filename = f"{date} — {safe_topic}.md"

    target_dir = Path(vault_path) / sector.title()
    target_dir.mkdir(parents=True, exist_ok=True)
    filepath = target_dir / filename

    tags = [sector.lower(), "debate", "research"]
    sources = sources or []

    lines = [
        "---",
        f"date: {date}",
        f'topic: "{topic}"',
        f"sector: {sector}",
        f'experts: [{", ".join(experts)}]',
        f"tags: [{', '.join(tags)}]",
    ]
    if sources:
        lines.append("sources:")
        for s in sources:
            lines.append(f"  - {s}")
    lines += [
        "---",
        "",
        f"# {topic}",
        f"**Sector:** {sector}  ",
        f"**Date:** {date}  ",
        f"**Participants:** {', '.join(experts)}",
        "",
    ]

    if research_brief:
        lines += ["## Research Brief", "", research_brief, "", "---", ""]

    lines += ["## Debate Transcript", ""]

    for exchange in exchanges:
        role = exchange.get("role", "expert")
        speaker = exchange["speaker"]
        content = exchange.get("content", "")

        if role == "round_header":
            lines += ["---", f"## {speaker}", ""]
        elif role in ("moderator_open", "moderator"):
            lines += ["### Moderator — Opening", "", content, ""]
        elif role == "synthesis":
            lines += [f"### {speaker}", "", content, ""]
        elif role == "followup":
            lines += [f"### {speaker}", "", f"> {content}", ""]
        elif role == "final_synthesis":
            lines += ["---", f"## {speaker}", "", content, ""]
        else:
            affiliation = exchange.get("affiliation", "")
            header = f"### {speaker}"
            if affiliation:
                header += f" *({affiliation})*"
            lines += [header, "", content, ""]
            articles = exchange.get("articles", [])
            if articles:
                lines.append("**Key works:**")
                for a in articles:
                    lines.append(f"- [{a['title']}]({a['url']})")
                lines.append("")

    filepath.write_text("\n".join(lines), encoding="utf-8")
    return str(filepath)
