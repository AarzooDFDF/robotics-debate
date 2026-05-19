"""Save selected debate exchanges as a markdown note in an Obsidian vault."""

import os
from datetime import datetime
from pathlib import Path


def save_to_obsidian(
    topic: str,
    experts: list[str],
    exchanges: list[dict],
    vault_path: str,
    subfolder: str = "Robotics Debates",
) -> str:
    date = datetime.now().strftime("%Y-%m-%d")
    safe_topic = topic.replace("/", "-").replace(":", "").replace("?", "")
    filename = f"{date} — {safe_topic}.md"

    target_dir = Path(vault_path) / subfolder
    target_dir.mkdir(parents=True, exist_ok=True)
    filepath = target_dir / filename

    lines = [
        "---",
        f"date: {date}",
        f'topic: "{topic}"',
        f'experts: [{", ".join(experts)}]',
        "tags: [robotics, debate, research]",
        "---",
        "",
        f"# Debate: {topic}",
        f"**Date:** {date}  ",
        f"**Participants:** {', '.join(experts)}",
        "",
    ]

    for exchange in exchanges:
        role = exchange.get("role", "expert")
        speaker = exchange["speaker"]
        content = exchange["content"]

        if role == "round_header":
            lines += ["---", f"## {speaker}", ""]
            continue

        if role in ("moderator_open", "moderator"):
            lines += [f"### Moderator — Opening", "", content, ""]
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
