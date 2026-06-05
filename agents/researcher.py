"""ResearchAgent: synthesizes web search results into a structured market brief."""

from openai import OpenAI

RESEARCHER_SYSTEM = """You are a rigorous market analyst preparing a structured brief for an expert debate.
Given a sector name and raw search results, synthesize a concise, factual brief.

Structure:
1. Market Overview (size, CAGR, time horizon, geography)
2. Key Players (leaders, funded startups with round/investor/thesis)
3. Technology Bets (proven vs contested vs speculative)
4. Investment Themes — Bulls vs Bears
5. Open Questions (3 contested claims worth debating)

Be specific with numbers and names. Cite sources inline as [Source N]. Under 600 words."""


class ResearchAgent:
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def synthesize(self, sector: str, search_results: list[dict]) -> str:
        sources_block = "\n\n".join(
            f"[Source {i+1}] {r.get('title', '')}\nURL: {r.get('url', '')}\n{r.get('snippet', '')[:400]}"
            for i, r in enumerate(search_results)
        )
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=900,
            messages=[
                {"role": "system", "content": RESEARCHER_SYSTEM},
                {"role": "user", "content": f"Sector: {sector}\n\nSearch results:\n{sources_block}"},
            ],
        )
        return response.choices[0].message.content.strip()
