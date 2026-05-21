"""ExpertAgent: one instance per persona, responds in character to debate questions."""

from openai import OpenAI


class ExpertAgent:
    def __init__(self, persona: dict, client: OpenAI, model: str):
        self.persona = persona
        self.client = client
        self.model = model
        self.name = persona["name"]
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        p = self.persona
        # Truncate each field to keep the system prompt under ~500 tokens
        positions = "\n".join(f"- {pos[:120]}" for pos in p.get("known_positions", [])[:4])
        skepticisms = "\n".join(f"- {s[:100]}" for s in p.get("skeptical_of", [])[:2])
        articles = "\n".join(
            f"- {a['title'][:80]}" for a in p.get("seminal_articles", [])[:2]
        )
        thesis = (p.get("core_thesis") or "")[:250]
        style = (p.get("rhetorical_style") or "")[:120]
        return f"""You are {p['name']}, {p.get('title','')} at {p.get('affiliation','')}.

Core thesis: {thesis}

Known positions:
{positions}

Skeptical of:
{skepticisms}

Rhetorical style: {style}

Key work: {articles}

Instructions: Stay in character. Speak in first person. Engage with what others said.
Be specific. Keep responses to 150-200 words. Do not mention you are an AI."""

    def respond(self, debate_history: list[dict], question: str) -> str:
        context = self._format_history(debate_history)
        user_message = f"{context}\n\n---\nModerator: {question}\n\nRespond as {self.name}:"

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=500,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content.strip()

    def _format_history(self, history: list[dict]) -> str:
        if not history:
            return ""
        # Keep only the last 4 substantive exchanges to stay within context limits
        recent = [e for e in history if e.get("content")][-4:]
        lines = ["--- Recent debate ---"]
        for exchange in recent:
            snippet = (exchange["content"] or "")[:280]
            lines.append(f"{exchange['speaker']}: {snippet}")
        return "\n\n".join(lines)
