import asyncio
import json
import logging
import os
from typing import Any, Callable, Optional

import yaml

from .experiment import experiment_async
from .persona import generate_personas
from .survey import run_survey

log = logging.getLogger("simulated_web_agent.main.run")
logging.basicConfig(level=logging.INFO)


def load_config(config_file: str) -> dict[str, Any]:
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def _safe_ping(cb: Optional[Callable[[dict], None]], evt: dict):
    if not cb:
        return
    try:
        cb(evt)
    except Exception:
        # never let UI progress kill the job
        pass


async def run_async(
    total_personas: int,
    demographics: list[dict[str, Any]],
    general_intent: str,
    start_url: str,
    max_steps: int,
    questionnaire: dict,
    headless: bool = False,
    concurrency: int = 4,
    example_persona: str = None,
    # Pass in a callback for updating progress
    on_progress: Optional[Callable[[dict], None]] = None,
):
    ping = lambda e: _safe_ping(on_progress, e)
    # ---------------- 1) Personas ----------------
    # starts generation
    ping({"phase": "personas", "status": "start", "total": total_personas})
    all_personas_intents = await generate_personas(
        demographics=demographics,
        general_intent=general_intent,
        n=total_personas,
        on_progress=lambda k, n: ping(
            {"phase": "personas", "status": "progress", "current": k, "total": n}
        ),
        example_text=example_persona,
    )
    # generation ends here
    ping(
        {
            "phase": "personas",
            "status": "progress",
            "current": total_personas,
            "total": total_personas,
        }
    )

    with open("personas.json", "w", encoding="utf-8") as f:
        json.dump(all_personas_intents, f, indent=2, ensure_ascii=False)

    # 2) Run ALL experiments (parallel inside, but function returns only when done)
    total_agents = len(all_personas_intents)
    ping({"phase": "agents", "status": "start", "total": total_agents})

    trace_dirs = await experiment_async(
        agents=all_personas_intents,
        start_url=start_url,
        max_steps=max_steps,
        config_name="base",
        config_path=".",
        concurrency=concurrency,  # add more if safe. so far 4 works. can't garentee this wont explode your pc if = 1000
        headless=headless,
        on_progress=lambda k, n: ping(
            {"phase": "agents", "status": "progress", "current": k, "total": n}
        ),
    )
    ping(
        {
            "phase": "agents",
            "status": "progress",
            "current": total_agents,
            "total": total_agents,
        }
    )

    # 3) Only AFTER experiments finish, run ALL surveys
    ping({"phase": "surveys", "status": "start", "total": total_agents})
    await run_survey(
        trace_dirs=trace_dirs,
        questionnaire=questionnaire,
        on_progress=lambda k, n: ping(
            {"phase": "surveys", "status": "progress", "current": k, "total": n}
        ),
    )
    ping(
        {
            "phase": "surveys",
            "status": "progress",
            "current": total_agents,
            "total": total_agents,
        }
    )


def run(*args, **kwargs):
    asyncio.run(run_async(*args, **kwargs))


if __name__ == "__main__":
    import sys

    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "conf/runConfig.yaml")
    asyncio.run(
        run_async(
            total_personas=cfg.get("total_personas", 8),
            demographics=cfg["demographics"],
            general_intent=cfg["general_intent"],
            start_url=cfg["start_url"],
            max_steps=cfg["max_steps"],
            questionnaire=cfg["questionnaire"],
            example_persona=cfg["example_persona"],
            concurrency=cfg["concurrency"],
            headless=True,
        )
    )
