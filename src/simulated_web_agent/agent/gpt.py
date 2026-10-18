import asyncio
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, cast

# anthropic import for claude computer use
import anthropic
import yaml
from anthropic.types.beta import (
    BetaContentBlockParam,
    BetaTextBlock,
    BetaTextBlockParam,
    BetaToolUseBlockParam,
)

# client and model for claude compute use tool
from dotenv import load_dotenv

# from litellm import drop_params, token_counter
from litellm.router import Router

from . import context

provider = "openai"  # "openai" or "aws" or "anthropic" or "claude_cli"

# claude_cli provider: every chat call runs a headless `claude -p` (Claude Code
# subscription, no API key). Models are Claude Code aliases, override via env.
claude_cli_models = {
    "small": os.environ.get("UXAGENT_CLAUDE_SMALL_MODEL", "haiku"),
    "large": os.environ.get("UXAGENT_CLAUDE_LARGE_MODEL", "sonnet"),
}
claude_cli_bin = os.environ.get("UXAGENT_CLAUDE_BIN", "claude")
# empty cwd so no project CLAUDE.md is picked up
claude_cli_cwd = tempfile.mkdtemp(prefix="uxagent-claude-")

prompt_dir = Path(__file__).parent.absolute() / "shop_prompts"

chat_router = Router(
    model_list=[
        {
            "model_name": "openai",
            "litellm_params": {
                "model": "openai/gpt-5-mini",
                "reasoning_effort": "minimal",
            },
        },
        {
            "model_name": "aws",
            "litellm_params": {
                "model": "bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0",
                "thinking": {
                    "type": "disabled",
                },
            },
        },
        {
            "model_name": "anthropic",
            "litellm_params": {
                "model": "anthropic/claude-sonnet-4-20250514",
            },
        },
        {
            "model_name": "openai_thinking",
            "litellm_params": {
                "model": "openai/gpt-5-mini",
                "reasoning_effort": "high",
            },
        },
        {
            "model_name": "aws_thinking",
            "litellm_params": {
                "model": "bedrock/global.anthropic.claude-sonnet-4-5-20250929-v1:0",
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 32000,
                },
            },
        },
        {
            "model_name": "anthropic_thinking",
            "litellm_params": {
                "model": "anthropic/claude-sonnet-4-20250514",
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 32000,
                },
            },
        },
    ]
)

slow_chat_router = Router(
    model_list=[
        {
            "model_name": "openai",
            "litellm_params": {"model": "openai/gpt-5", "reasoning_effort": "minimal"},
        },
        {
            "model_name": "aws",
            "litellm_params": {
                "model": "bedrock/global.anthropic.claude-sonnet-4-5-20250929-v1:0",
                "thinking": {
                    "type": "disabled",
                },
            },
        },
        {
            "model_name": "anthropic",
            "litellm_params": {
                "model": "anthropic/claude-sonnet-4-20250514",
            },
        },
        {
            "model_name": "openai_thinking",
            "litellm_params": {"model": "openai/gpt-5", "reasoning_effort": "high"},
        },
        {
            "model_name": "aws_thinking",
            "litellm_params": {
                "model": "bedrock/global.anthropic.claude-sonnet-4-5-20250929-v1:0",
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 32000,
                },
            },
        },
        {
            "model_name": "anthropic_thinking",
            "litellm_params": {
                "model": "anthropic/claude-sonnet-4-20250514",
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 32000,
                },
            },
        },
    ]
)

embed_router = Router(
    model_list=[
        {
            "model_name": "openai",
            "litellm_params": {"model": "openai/text-embedding-3-small"},
        },
        {
            "model_name": "aws",
            "litellm_params": {
                "model": "bedrock/cohere.embed-english-v3",
                "input_type": "search_document",
                "truncate": "END",
            },
        },
    ]
)


load_dotenv()  # load anthropic api key from .env
anthropic_client = anthropic.Anthropic()
anthropic_model = "claude-sonnet-4-20250514"


def async_retry(times=10):
    def func_wrapper(f):
        async def wrapper(*args, **kwargs):
            wait = 1
            max_wait = 5
            last_exc = None
            for _ in range(times):
                # noinspection PyBroadException
                try:
                    return await f(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    print("got exc", exc)
                    await asyncio.sleep(wait)
                    wait = min(wait * 2, max_wait)
                    pass
            if last_exc:
                raise last_exc

        return wrapper

    return func_wrapper


def retry(times=10):
    def func_wrapper(f):
        def wrapper(*args, **kwargs):
            wait = 1
            max_wait = 5
            last_exc = None
            for _ in range(times):
                # noinspection PyBroadException
                try:
                    return f(*args, **kwargs)
                except Exception as exc:
                    print("got exc", exc)
                    last_exc = exc
                    # await asyncio.sleep(wait)
                    time.sleep(wait)
                    wait = min(wait * 2, max_wait)
                    pass
            if last_exc:
                raise last_exc

        return wrapper

    return func_wrapper


def _claude_cli_args(messages, model: str) -> tuple[list[str], str]:
    """
    Build `claude -p` argv and stdin from OpenAI-style messages.
    System messages become the system prompt; the rest is sent as one transcript.
    """
    system_parts, turns = [], []
    for m in messages:
        content = m["content"]
        if isinstance(content, list):
            content = "\n".join(p.get("text", "") for p in content)
        if m["role"] == "system":
            system_parts.append(content)
        else:
            turns.append((m["role"], content))
    if len(turns) == 1:
        prompt = turns[0][1]
    else:
        prompt = "\n\n".join(f"<{role}>\n{text}\n</{role}>" for role, text in turns)
    args = [
        claude_cli_bin,
        "-p",
        "--model",
        claude_cli_models["small" if model == "small" else "large"],
        "--output-format",
        "json",
        "--tools",
        "",
        "--setting-sources",
        "",
        "--strict-mcp-config",
        "--disable-slash-commands",
        "--no-session-persistence",
        "--system-prompt",
        "\n\n".join(system_parts) or "You are a helpful assistant.",
    ]
    return args, prompt


def _claude_cli_result(stdout: bytes, stderr: bytes, returncode: int) -> str:
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        raise Exception(f"claude -p failed ({returncode}): {stderr.decode()[-500:]}")
    if data.get("is_error"):
        raise Exception(f"claude -p error: {data.get('result')}")
    return data["result"]


_claude_cli_slots = asyncio.Semaphore(
    int(os.environ.get("UXAGENT_CLAUDE_CONCURRENCY", "4"))
)


async def _async_chat_claude_cli(messages, model: str) -> str:
    args, prompt = _claude_cli_args(messages, model)
    async with _claude_cli_slots:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=claude_cli_cwd,
        )
        stdout, stderr = await proc.communicate(prompt.encode())
    return _claude_cli_result(stdout, stderr, proc.returncode)


def _chat_claude_cli(messages, model: str) -> str:
    args, prompt = _claude_cli_args(messages, model)
    proc = subprocess.run(
        args, input=prompt.encode(), capture_output=True, cwd=claude_cli_cwd
    )
    return _claude_cli_result(proc.stdout, proc.stderr, proc.returncode)


def _hash_embed(text: str, dim: int = 512) -> list[float]:
    # ponytail: hashed bag-of-words, no semantic similarity; swap for a real
    # embedding model if memory retrieval quality matters.
    vec = [0.0] * dim
    for token in re.findall(r"\w+", text.lower()):
        h = int.from_bytes(hashlib.md5(token.encode()).digest()[:4], "little")
        vec[h % dim] += 1.0
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


def _extract_json_string(text: str) -> str:
    import regex

    # Improved pattern to match JSON objects. Note: This is still not foolproof for deeply nested or complex JSON.
    json_pattern = r"\{(?:[^{}]*|(?R))*\}"
    matches = regex.findall(json_pattern, text, regex.DOTALL)
    if matches:
        return matches[0]
    else:
        raise Exception("No JSON object found in the response")


@async_retry()
async def async_chat(
    messages,
    model="small",
    json_mode=False,
    log=True,
    max_tokens=64000,
    enable_thinking=None,
    **kwargs,
):
    """
    Async chat completion that returns the string LLM output.

    For model and provider information, check the head of /src/simulated_web_agent/agent/gpt.py.

    To add your own preferred LLM for chat completion, simply add the model into each of the liteLLM routers,
    and change the provider global variable to your custom provider.

    Args:
        model: "small" for lightweight version of the model
        json_mode: whether the result should be in json deserializable
        log: whether to log the output
        max_tokens: the maximum number of tokens
        enable_thinking: whether to enable thinking, if supported (supported by bedrock and anthropic, not supported by openai)

    Returns:
        A single string object outputted by the LLM.
    """

    if context.api_call_manager.get() and log:
        context.api_call_manager.get().request.append(messages)

    if provider == "claude_cli":
        content = await _async_chat_claude_cli(messages, model)
    else:
        router = chat_router if model == "small" else slow_chat_router
        call_kwargs: Dict[str, Any] = dict(**kwargs)
        if enable_thinking:
            router_model = provider + "_thinking"
        else:
            router_model = provider
        if json_mode and provider == "openai":
            call_kwargs["response_format"] = {"type": "json_object"}
        response = await router.acompletion(
            model=router_model,
            messages=messages,
            max_tokens=max_tokens,
            drop_params=True,  # do not forward unused params, such as thinking for openai
            **call_kwargs,
            tools=None,
        )
        content = response.choices[0].message.get("content", "")

        finish_reason = response.choices[0].finish_reason
        if finish_reason != "stop":
            print("finish_reason:", finish_reason)
            print("content:", content)
            print("response:", response)
    # tokens_used = token_counter(model="openai/gpt-5-mini", text=content)
    # print("Output tokens:", tokens_used)

    if context.api_call_manager.get() and log:
        context.api_call_manager.get().response.append(content)

    if json_mode:
        # Extract JSON substring from the content
        try:
            json_str = _extract_json_string(content)
            _ = json.loads(json_str)
            return json_str
        except Exception as e:
            print(e)
            print(content)
            raise Exception("Invalid JSON in response") from e
    return content


@retry()
def chat(
    messages, model="small", enable_thinking=None, json_mode=False, **kwargs
) -> str:
    """
    Returns LLM text completion given list of formatted messages.

    Args:
        model: set to "small" to use the lightweight version of the chat model.
        messages: List of previous messages
        enable_thinking: Whether to enable thinking or not. Pass in an integer for custom thinking budget.
        json_mode: Whether to enable JSON mode or not.

    Returns:
        String output of the LLM model
    """

    if provider == "claude_cli":
        return _chat_claude_cli(messages, model)

    router = chat_router if model == "small" else slow_chat_router
    call_kwargs: Dict[str, Any] = dict(**kwargs)
    if enable_thinking:
        call_kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": enable_thinking
            if isinstance(enable_thinking, int)
            else 1024,
        }
    if json_mode and provider == "openai":
        call_kwargs["response_format"] = {"type": "json_object"}

    try:
        response = router.completion(
            model=provider,
            messages=messages,
            drop_params=True,  # do not forward unused params, such as thinking for openai
            **call_kwargs,
        )
        return response.choices[0].message["content"]
    except Exception as e:
        print(messages)
        print(e)
        raise e


async def embed_text(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts using the provider configured in /src/simulated_web_agent/agent/gpt.py

    Returns:
        List of list[float] representing each of the embedded texts
    """
    if provider == "claude_cli":
        return [_hash_embed(t) for t in texts]
    try:
        response = await embed_router.aembedding(model=provider, input=texts)
        return [e["embedding"] for e in response.data]
    except Exception as e:
        print(texts)
        print(e)
        raise e


def chat_anthropic_computer_use(
    messages,
    system: BetaTextBlockParam,
    model=anthropic_model,
    screen_width: int = 1024,
    screen_height: int = 768,
) -> (list[BetaToolUseBlockParam], list[Dict, Any]):
    """
    Given a system block and JSON messages, return the tool use block generated by the computer use tool
    """
    response = anthropic_client.beta.messages.create(
        model=model,
        max_tokens=1024,
        tools=[
            {
                "type": "computer_20250124",
                "name": "computer",
                "display_width_px": screen_width,
                "display_height_px": screen_height,
                "display_number": 1,
            },
            {
                "name": "web_browser",
                "description": "High-level browser controls",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": [
                                "switch_tab",
                                "forward",
                                "back",
                                "new_tab",
                                "goto_url",
                                "close_tab",
                                "terminate",
                            ],
                        },
                        "tab_index": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "Zero-based index for switch_tab and close_tab",
                        },
                        "url": {
                            "type": "string",
                            "description": "URL input, only required for goto_url and new_tab",
                        },
                    },
                    "required": ["action"],
                },
            },
        ],
        system=[system],
        messages=messages,
        betas=["computer-use-2025-01-24"],
    )

    return response


def load_prompt(prompt_name):
    p = prompt_dir / f"{prompt_name}.txt"
    return open(p, "r", encoding="utf-8").read()
