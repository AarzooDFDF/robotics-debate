"""FactCheckerAgent: validates expert claims against the stored research corpus."""

import json
from openai import OpenAI


FACT_CHECK_PROMPT = """You are a research validator checking a robotics expert's debate statement against a knowledge base.

Expert: {expert_name}
Their statement:
{response}

Knowledge base:
{sources}

Identify up to 3 specific claims in the statement that are directly relevant to the research above.
For each match, state whether the source SUPPORTS, CONTRADICTS, or adds useful CONTEXT.

Return a JSON array (return [] if nothing matches):
[
  {{
    "claim": "short paraphrase of the specific claim being checked (under 20 words)",
    "verdict": "SUPPORTS" | "CONTRADICTS" | "CONTEXT",
    "source_title": "exact title from the knowledge base",
    "source_ref": "url or filename from the knowledge base entry",
    "explanation": "one sentence: how this source relates to the claim"
  }}
]

Rules:
- Only match claims that the source directly addresses — not superficial keyword overlaps
- Return [] if none of the sources are genuinely relevant
- Return only valid JSON, no other text"""


class FactCheckerAgent:
    def __init__(self, corpus: list[dict], client: OpenAI, model: str):
        self.client = client
        self.model = model
        self._formatted = self._format_corpus(corpus)

    @staticmethod
    def _format_corpus(corpus: list[dict]) -> str:
        if not corpus:
            return ""
        lines = []
        for i, item in enumerate(corpus[:15], 1):
            ref = item.get("url") or item.get("filename") or "(no link)"
            title = item.get("title", "Untitled")
            insights = (item.get("key_insights") or "")[:180]
            lines.append(f"[{i}] {title}\n    Source: {ref}\n    Findings: {insights}")
        return "\n\n".join(lines)

    def validate(self, expert_name: str, response: str) -> list[dict]:
        if not self._formatted:
            return []
        prompt = FACT_CHECK_PROMPT.format(
            expert_name=expert_name,
            response=response[:1200],
            sources=self._formatted[:1800],
        )
        try:
            raw = self.client.chat.completions.create(
                model=self.model,
                max_tokens=500,
                messages=[
                    {"role": "system", "content": "Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
            ).choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            result = json.loads(raw.strip())
            return result if isinstance(result, list) else []
        except Exception:
            return []
