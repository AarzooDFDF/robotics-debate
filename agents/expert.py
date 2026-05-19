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
        positions = "\n".join(f"- {pos}" for pos in p.get("known_positions", []))
        skepticisms = "\n".join(f"- {s}" for s in p.get("skeptical_of", []))
        articles = "\n".join(
            f"- {a['title']}: {a['url']}" for a in p.get("seminal_articles", [])
        )
        return f"""You are {p['name']}, {p['title']} at {p['affiliation']}.

Your core thesis: {p['core_thesis']}

Your known positions:
{positions}

You are skeptical of:
{skepticisms}

Your rhetorical style: {p['rhetorical_style']}

Your key published work and interviews:
{articles}

Instructions:
- Stay strictly in character as {p['name']} throughout the debate.
- Speak in first person. Reference your actual research, investments, and public statements.
- Engage directly with what others have said — agree where you genuinely would, push back where your positions diverge.
- Be specific: cite your own work, data points, and positions rather than speaking in generalities.
- Keep responses to 150-200 words. This is a debate, not a lecture.
- Do not break character. Do not mention that you are an AI."""

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
        lines = ["--- Debate so far ---"]
        for exchange in history:
            lines.append(f"{exchange['speaker']}: {exchange['content']}")
        return "\n\n".join(lines)
