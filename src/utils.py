import json
import os
import time
import random
from pathlib import Path
from typing import List, Dict, Optional

from dotenv import load_dotenv
load_dotenv()


def load_scenarios(path: str) -> List[Dict]:
    """Load and validate scenarios from JSON file."""
    with open(path, "r") as f:
        scenarios = json.load(f)
    required = ["id", "domain", "role_prompt", "turns", "system_instruction"]
    for s in scenarios:
        for key in required:
            if key not in s:
                raise ValueError(f"Scenario {s.get('id', '?')} missing required key: {key}")
    return scenarios


def call_openai(
    model: str,
    system_msg: str,
    history: List[Dict],
    user_msg: str,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> str:
    """Call OpenAI chat completions with exponential backoff on rate limits (3 retries)."""
    from openai import OpenAI, RateLimitError

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    messages = [{"role": "system", "content": system_msg}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_msg})

    # o-series and gpt-5 use max_completion_tokens and don't support custom temperature
    uses_new_params = model.startswith("o") or model.startswith("gpt-5")
    token_param = "max_completion_tokens" if uses_new_params else "max_tokens"
    extra = {} if uses_new_params else {"temperature": temperature}

    last_exc = None
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                **{token_param: max_tokens},
                **extra,
            )
            return response.choices[0].message.content
        except RateLimitError as e:
            last_exc = e
            wait = (2 ** attempt) + random.random()
            print(f"  [RateLimit] Waiting {wait:.1f}s before retry {attempt + 1}/3...")
            time.sleep(wait)
        except Exception as e:
            last_exc = e
            if attempt == 2:
                break
            wait = (2 ** attempt) + random.random()
            print(f"  [Error] {e}. Retrying in {wait:.1f}s...")
            time.sleep(wait)
    raise RuntimeError(f"OpenAI call failed after 3 attempts for model '{model}': {last_exc}")


def call_anthropic(
    model: str,
    system_msg: str,
    history: List[Dict],
    user_msg: str,
    temperature: Optional[float] = 0.7,
    max_tokens: int = 1024,
) -> str:
    """Call Anthropic Messages API with exponential backoff on rate limits (3 retries)."""
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    messages = list(history)
    messages.append({"role": "user", "content": user_msg})
    extra = {} if temperature is None else {"temperature": temperature}

    last_exc = None
    for attempt in range(3):
        try:
            response = client.messages.create(
                model=model,
                system=system_msg,
                messages=messages,
                max_tokens=max_tokens,
                **extra,
            )
            return response.content[0].text
        except anthropic.RateLimitError as e:
            last_exc = e
            wait = (2 ** attempt) + random.random()
            print(f"  [RateLimit] Waiting {wait:.1f}s before retry {attempt + 1}/3...")
            time.sleep(wait)
        except Exception as e:
            last_exc = e
            if attempt == 2:
                break
            wait = (2 ** attempt) + random.random()
            print(f"  [Error] {e}. Retrying in {wait:.1f}s...")
            time.sleep(wait)
    raise RuntimeError(f"Anthropic call failed after 3 attempts for model '{model}': {last_exc}")


def call_model(
    model_name: str,
    system_msg: str,
    history: List[Dict],
    user_msg: str,
) -> str:
    """Route API call to OpenAI ('gpt-' prefix) or Anthropic ('claude-' prefix)."""
    if model_name.startswith("gpt-") or model_name.startswith("o"):
        return call_openai(model_name, system_msg, history, user_msg)
    elif model_name.startswith("claude-"):
        return call_anthropic(model_name, system_msg, history, user_msg)
    else:
        raise ValueError(
            f"Unknown model prefix for '{model_name}'. Expected 'gpt-', 'o', or 'claude-'."
        )


def save_transcript(
    output_dir: str,
    scenario_id: str,
    model_name: str,
    condition: str,
    turns_data: List[Dict],
) -> Path:
    """Save full conversation transcript as JSON to results/transcripts/{model}/{scenario}_{condition}.json."""
    safe_model = model_name.replace("/", "_")
    model_dir = Path(output_dir) / safe_model
    model_dir.mkdir(parents=True, exist_ok=True)
    filepath = model_dir / f"{scenario_id}_{condition}.json"

    payload = {
        "scenario_id": scenario_id,
        "model": model_name,
        "condition": condition,
        "turns": turns_data,
    }
    with open(filepath, "w") as f:
        json.dump(payload, f, indent=2)
    return filepath


def build_turn_message(turn: Dict, system_instruction: Optional[str] = None) -> str:
    """Format a turn dict into the user message combining situation, authority_message, and output format instructions."""
    parts = []
    if "situation" in turn:
        parts.append(f"SITUATION:\n{turn['situation']}")
    if "authority_message" in turn:
        parts.append(f"AUTHORITY MESSAGE:\n{turn['authority_message']}")
    if system_instruction:
        parts.append(f"OUTPUT FORMAT:\n{system_instruction}")
    return "\n\n".join(parts)
