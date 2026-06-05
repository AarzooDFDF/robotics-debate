"""DebateOrchestrator: runs the full debate loop, yields exchanges."""

from typing import Generator
from openai import OpenAI

from agents.expert import ExpertAgent
from agents.moderator import ModeratorAgent


class DebateOrchestrator:
    def __init__(
        self,
        personas: list[dict],
        debate: dict,
        client: OpenAI,
        model: str,
        num_rounds: int = 2,
        research_context: str = "",
    ):
        self.debate = debate
        self.num_rounds = num_rounds
        self.research_context = research_context
        self.experts = [ExpertAgent(p, client, model) for p in personas]
        self.moderator = ModeratorAgent(client, model)

    def run(self) -> Generator[dict, None, None]:
        history: list[dict] = []
        expert_names = [e.name for e in self.experts]

        opening = self.moderator.open_debate(self.debate, expert_names)
        exchange = {"speaker": "Moderator", "content": opening, "role": "moderator_open"}
        history.append(exchange)
        yield exchange

        current_question = self.debate["opening_question"]

        for round_num in range(self.num_rounds):
            yield {"speaker": f"— Round {round_num + 1} —", "content": "", "role": "round_header"}

            for expert in self.experts:
                response = expert.respond(history, current_question, self.research_context)
                exchange = {
                    "speaker": expert.name,
                    "content": response,
                    "role": "expert",
                    "affiliation": expert.persona.get("affiliation", ""),
                    "articles": expert.persona.get("seminal_articles", [])[:2],
                }
                history.append(exchange)
                yield exchange

            synthesis = self.moderator.synthesize_round(history, round_num)
            exchange = {"speaker": "Moderator — Synthesis", "content": synthesis, "role": "synthesis"}
            history.append(exchange)
            yield exchange

            if round_num < self.num_rounds - 1:
                follow_up = self.moderator.generate_followup(self.debate, history, round_num)
                current_question = follow_up
                exchange = {"speaker": "Moderator — Follow-up", "content": follow_up, "role": "followup"}
                history.append(exchange)
                yield exchange

        final = self.moderator.final_synthesis(history)
        yield {"speaker": "Moderator — Final Synthesis", "content": final, "role": "final_synthesis"}
