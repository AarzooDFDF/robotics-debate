"""ModeratorAgent: opens debates, synthesizes tensions, generates follow-ups."""

from openai import OpenAI


MODERATOR_SYSTEM = """You are a sharp, neutral moderator running a high-stakes robotics debate.
Your participants are leading researchers, investors, and industry executives.

Your job:
- Open debates with the provided question clearly and crisply.
- After each round, identify the 2-3 sharpest points of genuine disagreement.
  Name the people on each side. Be specific — quote or paraphrase what was said.
- Generate follow-up questions that force participants to be more specific,
  address a concrete counterargument, or stake out a harder position.
- Final synthesis: summarize where there is genuine agreement, where disagreement
  is irreconcilable, and what the key open question is.
- Keep all outputs concise. Synthesis: max 200 words. Follow-ups: one sharp question."""


class ModeratorAgent:
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def _call(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=400,
            messages=[
                {"role": "system", "content": MODERATOR_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content.strip()

    def open_debate(self, debate: dict, expert_names: list[str]) -> str:
        participants = ", ".join(expert_names)
        prompt = f"""Open this debate for the following participants: {participants}

Topic: {debate['topic']}
Opening question: {debate['opening_question']}

Write a brief (2-3 sentence) framing of the debate, then pose the opening question directly."""
        return self._call(prompt)

    def synthesize_round(self, history: list[dict], round_num: int) -> str:
        # Use only the most recent expert exchanges (current round) to stay within context
        expert_exchanges = [e for e in history if e.get("role") == "expert"][-6:]
        transcript = "\n\n".join(
            f"{e['speaker']}: {(e['content'] or '')[:280]}"
            for e in expert_exchanges
        )
        prompt = f"""Round {round_num + 1} transcript:

{transcript}

Identify the 2-3 sharpest points of genuine disagreement. Name the people on each side.
Paraphrase what was actually said. Keep to 150 words."""
        return self._call(prompt)

    def generate_followup(self, debate: dict, history: list[dict], round_num: int) -> str:
        follow_ups = debate.get("follow_up_prompts", [])
        if round_num < len(follow_ups):
            return follow_ups[round_num]
        transcript = "\n\n".join(
            f"{e['speaker']}: {e['content']}" for e in history[-6:]
        )
        prompt = f"""Based on this debate excerpt, generate one sharp follow-up question that forces
participants to be more specific or address a concrete counterargument:

{transcript}

Write only the question."""
        return self._call(prompt)

    def final_synthesis(self, history: list[dict]) -> str:
        # Sample key exchanges: opening + all syntheses + last 4 expert turns
        opening = [e for e in history if e.get("role") == "moderator_open"]
        syntheses = [e for e in history if e.get("role") == "synthesis"]
        recent_expert = [e for e in history if e.get("role") == "expert"][-4:]
        sampled = opening + syntheses + recent_expert
        transcript = "\n\n".join(
            f"{e['speaker']}: {(e['content'] or '')[:250]}"
            for e in sampled
        )
        prompt = f"""The debate is over. Key exchanges:

{transcript}

Write a final synthesis (max 200 words) covering:
1. Where there is genuine agreement across participants
2. Where disagreement is irreconcilable and why
3. The single most important open question this debate leaves unresolved"""
        return self._call(prompt)
