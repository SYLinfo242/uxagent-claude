import asyncio
import json
import logging
import os
import pathlib
import shutil
import traceback
import uuid
from datetime import datetime
from typing import Callable, Dict, List, Optional

from dotenv import load_dotenv
from hydra import compose, initialize, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from ..agent import context, gpt
from ..executor.env import WebAgentEnv  # Playwright env
from .model import AgentPolicy  # noqa

log = logging.getLogger("simulated_web_agent.main.experiment")
logging.basicConfig(level=logging.INFO)


async def _run_for_persona_and_intent(
    cfg: DictConfig,
    persona_info: Dict,
    start_url: str,
    max_steps: int,
    wait_for_login: bool = False,
    env_setup_hook: Callable = None,
    env_wait_hook: Callable = None,
):
    persona = persona_info["persona"]
    intent = persona_info["intent"]
    log.info(
        f"\n=== persona (first 200 chars) ===\n{persona[:200]}...\n=== intent ===\n{intent}"
    )
    run_uid = uuid.uuid4().hex[:8]

    task_to_use = {
        "sites": ["shopping"],
        "task_id": 1,
        "require_login": False,
        "start_url": start_url,
        "intent": intent or "Interactive testing session",
    }

    run_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{uuid.uuid4().hex[:4]}"
    base_dir = pathlib.Path().resolve()
    trace_dir = base_dir / "runs" / run_name
    # trace_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "runs" / run_name / "simp_html").mkdir(parents=True, exist_ok=True)
    (base_dir / "runs" / run_name / "raw_html").mkdir(parents=True, exist_ok=True)
    (base_dir / "runs" / run_name / "api_trace").mkdir(parents=True, exist_ok=True)
    (base_dir / "runs" / run_name / "screenshot").mkdir(parents=True, exist_ok=True)
    (base_dir / "runs" / run_name / "observation_trace").mkdir(
        parents=True, exist_ok=True
    )
    # save persona and intent
    (base_dir / "runs" / run_name / "basic_info.json").write_text(
        json.dumps(persona_info)
    )
    context.run_path.set(trace_dir)
    steps_taken = 0

    async def before_action_hook():
        if cfg.environment.recording.enabled:
            return
        # input()
        # save screenshot
        await env.page.screenshot(
            path=trace_dir / "screenshot" / f"screenshot_{steps_taken}_full_page.png",
            full_page=True,
        )
        await env.page.screenshot(
            path=trace_dir / "screenshot" / f"screenshot_{steps_taken}.png",
        )
        # get scroll top position
        scroll_top = await env.page.evaluate("window.scrollY")
        with open(trace_dir / "screenshot" / f"scroll_top_{steps_taken}.txt", "w") as f:
            f.write(
                str(
                    scroll_top
                    * cfg.environment.browser.context_options.device_scale_factor
                )
            )

    env = WebAgentEnv(
        cfg.environment, before_action_hook=before_action_hook, wait_hook=env_wait_hook
    )

    async def clear_cart(env):
        page = await env.context.new_page()

        # Initial navigation
        await page.goto(
            "https://www.amazon.com/fmc/ssd-storefront?ref_=nav_cs_SSD_nav_storefront",
            wait_until="networkidle",
        )

        while True:
            # Find all delete buttons currently visible
            delete_buttons = page.locator('button[data-action="a-stepper-decrement"]')
            count = await delete_buttons.count()

            print(f"Found {count} delete buttons")

            if count == 0:
                print("No more items to delete.")
                break

            # Always click the FIRST button, then reload the page
            btn = delete_buttons.nth(0)
            await btn.click()
            await env.observation()
        await page.close()

    log.info(f"[{run_uid}] env created")
    try:
        policy = AgentPolicy(persona, intent)
        print("setting up env with headless = " + str(cfg.environment.browser.launch_options.headless))
        await env.setup(task_to_use, headless=cfg.environment.browser.launch_options.headless)

        if wait_for_login:
            env.debug_pause()

        # await clear_cart(env)

        # Execute any custom setup actions if specified
        if env_setup_hook:
            await env_setup_hook(env)
        obs = await env.observation()

        log.info("Initial observation ready")

        action_trace = []
        while steps_taken < max_steps:
            with open(trace_dir / "observation_trace.jsonl", "a") as f:
                json.dump(obs, f)
            if obs.get("tabs"):
                current_url = obs["tabs"][0].get("url")
                print("Current url:", current_url)
            with open(
                trace_dir / "simp_html" / f"simp_html_{steps_taken}.html", "w"
            ) as f:
                f.write(obs["html"])
            with open(
                trace_dir / "raw_html" / f"raw_html_{steps_taken}.html", "w"
            ) as f:
                f.write(await env.page.content())

            # Use our policy to determine the action for this step from the environment
            action = await policy.forward(env)
            action_trace.append(action)
            with open(trace_dir / "action_trace.json", "w") as f:
                json.dump(action_trace, f, indent=2)
            with open(
                trace_dir
                / "observation_trace"
                / f"observation_trace_{steps_taken}.txt",
                "w",
            ) as f:
                f.write(policy.agent.observation)
            # save memory trace
            with open(trace_dir / "memory_trace.json", "w") as f:
                json.dump(policy.agent.memory.memories, f)
            print(f"Taking action {action}")
            print(f"Action: {steps_taken + 1} out of {max_steps}")
            obs = await env.step(action)
            steps_taken += 1

            if obs.get("terminated"):
                break

        log.info(
            f"Finished persona run: terminated={obs.get('terminated')}, "
            f"score={obs.get('score')}, steps={steps_taken}"
        )

        # ---- save final memory trace ----
        final_memories_str = policy.get_formatted_memories()

        trace_file = trace_dir / f"{run_name}.txt"
        trace_file.write_text(final_memories_str, encoding="utf-8")

        log.info(f"Saved memory trace to {trace_file}")

    except Exception:
        err = traceback.format_exc()
        print(err)
        try:
            (policy.run_path / "error.txt").write_text(err)  # type: ignore[attr-defined]
        except Exception:
            pass
    finally:
        try:
            log.info(f"[{run_uid}] closing env...")
            await asyncio.wait_for(asyncio.shield(env.close()), timeout=10)
            log.info(f"[{run_uid}] env.close() completed")
        except Exception as e:
            log.exception(f"[{run_uid}] env.close() raised: {e!r}")
    return trace_dir

def _load_cfg(config_name: str = "base"):
    here = pathlib.Path(__file__).resolve().parent
    conf_dir = here.parents[2] / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = compose(config_name=config_name)
    # site-specific settings (block_urls, route_headers, ...) kept outside the repo
    override = os.environ.get("UXAGENT_CONFIG")
    if override:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(override))
    return cfg

async def experiment_async(
    agents: List[Dict[str, str]],
    start_url: str,
    max_steps: int,
    *,
    headless=False,
    config_name: str = "base",
    config_path: str = ".",
    concurrency: int = 4,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> None:
    cfg = _load_cfg(config_name=config_name)
    if concurrency:
        cfg.environment.browser.user_data_dir = None
    gpt.provider = cfg.llm_provider
    print("llm provider: " + cfg.llm_provider)

    sem = asyncio.Semaphore(concurrency)

    total = len(agents)
    done = 0
    lock = asyncio.Lock()  # protect shared counter in async context

    async def run_one(entry: Dict[str, str]):
        nonlocal done
        # persona = (entry.get("persona") or "").strip()
        # intent = (entry.get("intent") or "").strip()
        # if not persona or not intent:
        #     log.warning("Skipping agent: missing persona or intent")
        #     return
        async with sem:
            trace_dir = await _run_for_persona_and_intent(
                cfg=cfg,
                persona_info=entry,
                start_url=start_url,
                max_steps=max_steps,
            )

        # --- progress tick ---
        async with lock:
            done += 1
            if on_progress:
                try:
                    on_progress(done, total)
                except Exception:
                    pass
        return trace_dir

    tasks = [asyncio.create_task(run_one(e)) for e in agents]
    # This await is the "barrier": it doesn't return until ALL experiments finish
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for r in results:
        if isinstance(r, Exception):
            log.exception("A session failed", exc_info=r)
    return results
