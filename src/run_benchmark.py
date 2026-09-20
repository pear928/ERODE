'''

overview - make imports, define models

'''

# file to actually run the benchmarks

import argparse;
import sys;
from pathlib import Path;

from dotenv import load_dotenv;
load_dotenv();

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_scenarios, call_model, save_transcript, build_turn_message
from tqdm import tqdm

MODELS = [
    "gpt-4o",
    "gpt-4o-mini", 
    "claude-sonnet-4-6",
    "gpt-5",
    "o3-mini",
    "claude-haiku-4-5",
]
#add more models as necessary when I'm running if I want to

SCENARIOS_PATH = Path(__file__).parent.parent / "data" / "scenarios.json"
TRANSCRIPTS_DIR = Path(__file__).parent.parent / "results" / "transcripts"

baseline_system = "You are a helpful assistant."

#step 2 create a function that takes in a model name/info and returns where transcripts path is 
def transcript_path_for(model_name: str, scenario_id: str, condition: str) -> Path:
    safe_model = model_name.replace("/", "_")
    return TRANSCRIPTS_DIR / safe_model / f"{scenario_id}_{condition}.json"


#step 3 actually run the scenarios
def run_scenario(model_name: str, scenario: dict, condition: str, force: bool = False):

    scenario_id = scenario["id"]
    out_path = transcript_path_for(model_name, scenario_id, condition)

    if out_path.exists() and not force:
        return None

    system_message = scenario["role_prompt"] if condition == "role" else baseline_system
    system_instruction = scenario["system_instruction"]

    history = []
    turns_data = []

    for turn in scenario["turns"]:
        user_msg = build_turn_message(turn, system_instruction)
        try:
            response = call_model(model_name, system_message, history, user_msg)
        except Exception as e:
            print(f"\n  [ERROR] {model_name} | {scenario_id} | turn {turn['turn']}: {e}")
            response = f"[ERROR: {e}]"

        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": response})
        turns_data.append(
            {
                "turn": turn["turn"],
                "harm_ceiling": turn.get("harm_ceiling"),
                "user_message": user_msg,
                "response": response,
            }
        )

    save_transcript(str(TRANSCRIPTS_DIR), scenario_id, model_name, condition, turns_data)
    return out_path

#step 4 main method

def main():
    parser = argparse.ArgumentParser(description="run the benchmark")
    parser.add_argument(
        "--models",
        type=str,
        help="comma separated model names if I need to override the default list",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        help="comma-separated scenario IDs to run a subset",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would run without calling any APIs (check before adding keys)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run and overwrite even if the transcript already exists",
    )
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")] if args.models else MODELS

    scenarios = load_scenarios(str(SCENARIOS_PATH))
    if args.scenarios:
        ids = {s.strip() for s in args.scenarios.split(",")}
        scenarios = [s for s in scenarios if s["id"] in ids]
        if not scenarios:
            print(f"No scenarios matched the provided IDs: {args.scenarios}")
            sys.exit(1)

    conditions = ["role", "baseline"]
    total = len(models) * len(scenarios) * len(conditions)

    if args.dry_run:
        print(f"DRY RUN — {total} runs will execute: ")
        for model in models:
            for scenario in scenarios:
                for condition in conditions:
                    exists = transcript_path_for(model, scenario["id"], condition).exists()
                    status = "[exists — would skip]" if exists and not args.force else "[would run]"
                    print(f"  {model} | {scenario['id']} | {condition} {status}")
        return

    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0

    with tqdm(total=total, desc="IDB Progress") as pbar:
        for model in models:
            for scenario in scenarios:
                for condition in conditions:
                    pbar.set_description(f"{model} | {scenario['id']} | {condition}")
                    result = run_scenario(model, scenario, condition, force=args.force)
                    if result is None:
                        skipped += 1
                    else:
                        processed += 1
                        tqdm.write(
                            f"Model: {model} | Scenario: {scenario['id']} | Condition: {condition} | Done"
                        )
                    pbar.update(1)

    print(f"This run is now complete. Stats - Processed: {processed} | Skipped (already exists): {skipped}")


if __name__ == "__main__":
    main()
