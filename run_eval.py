"""
run_eval.py — Evaluation runner for the Mobile UI Agent environment.

Usage
-----
    # Heuristic baseline (default):
    python run_eval.py

    # Verbose mode (per-task breakdown):
    python run_eval.py --verbose

    # Evaluate training split instead:
    python run_eval.py --split train

    # Use an OpenAI-compatible LLM endpoint:
    python run_eval.py --agent llm --model gpt-4o

    # Custom output file (JSON):
    python run_eval.py --output results.json

The heuristic baseline uses rule-based action sequences keyed on goal type.
It is a sanity-check for the environment, not a trained RL agent.
It should achieve ~80-90 % success rate on well-specified tasks.

LLM agent notes
---------------
Set OPENAI_API_KEY (or OPENAI_BASE_URL for local models) before using
--agent llm.  The agent is given the task instruction and current observation
in a single prompt and asked to return a JSON array of actions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import time
from collections.abc import Callable
from typing import Any

from mobile_ui_env.dataset import Task, build_dataset
from mobile_ui_env.env import MobileUIEnv

# ---------------------------------------------------------------------------
# Heuristic baseline agent
# ---------------------------------------------------------------------------


def _note_actions(title: str) -> list[dict[str, Any]]:
    return [
        {"action": "tap", "target": "notes_button"},
        {"action": "tap", "target": "add_note_button"},
        {"action": "type", "target": "note_input", "text": title},
        {"action": "tap", "target": "save_note_button"},
    ]


def heuristic_agent(obs: dict[str, Any], task: Task) -> list[dict[str, Any]]:
    """
    Rule-based agent that maps goal types to hand-crafted action sequences.

    This agent has full access to the task's goal dict, which a real RL agent
    would not have (it would only see the natural-language instruction).  It
    serves as an upper-bound sanity check and a baseline for comparison.
    """
    goal = task.goal
    goal_type = goal["type"]
    actions: list[dict[str, Any]] = []

    if goal_type == "note_created":
        actions.extend(_note_actions(goal["title"]))

    elif goal_type == "multi_note_created":
        actions.append({"action": "tap", "target": "notes_button"})
        for title in goal["titles"]:
            actions.extend([
                {"action": "tap", "target": "add_note_button"},
                {"action": "type", "target": "note_input", "text": title},
                {"action": "tap", "target": "save_note_button"},
            ])

    elif goal_type in ("setting_enabled", "setting_disabled"):
        setting = goal["setting"]
        toggle_map = {
            "focus_mode": "focus_mode_toggle",
            "notifications": "notifications_toggle",
        }
        toggle = toggle_map.get(setting, f"{setting}_toggle")

        # Determine current state from the observation (if already on settings)
        current_val: bool
        if obs.get("screen") == "settings":
            current_val = obs.get(setting, False)
        else:
            actions.append({"action": "tap", "target": "settings_button"})
            # Default start values
            current_val = setting == "notifications"  # notifications starts True

        desired = goal_type == "setting_enabled"
        if current_val != desired:
            actions.append({"action": "tap", "target": toggle})

    elif goal_type == "info_found":
        field = goal["field"]
        screen_map = {"username": "profile", "email": "profile", "version": "settings"}
        label_map = {"username": "username_label", "email": "email_label"}
        nav_map = {"profile": "profile_button", "settings": "settings_button"}

        screen = screen_map.get(field, "profile")
        actions.append({"action": "tap", "target": nav_map[screen]})
        if field in label_map:
            actions.append({"action": "tap", "target": label_map[field]})

    elif goal_type == "version_reported":
        actions.extend([
            {"action": "tap", "target": "settings_button"},
            {"action": "tap", "target": "version_label"},
        ])

    elif goal_type == "screen_visited":
        screen = goal["screen"]
        nav_map = {"profile": "profile_button", "settings": "settings_button", "notes": "notes_button"}
        actions.append({"action": "tap", "target": nav_map.get(screen, f"{screen}_button")})
        # Deliberately avoid forbidden actions

    elif goal_type == "multi_goal":
        for subgoal in goal.get("subgoals", []):
            sub_task = Task(
                task_id="sub",
                instruction="",
                goal=subgoal,
                max_steps=task.max_steps,
                split=task.split,
            )
            sub_actions = heuristic_agent(obs, sub_task)
            # Remove trailing 'finish' from sub-sequences
            sub_actions = [a for a in sub_actions if a.get("action") != "finish"]
            actions.extend(sub_actions)
            # Return home between subgoals
            actions.append({"action": "back"})

    actions.append({"action": "finish"})
    return actions


# ---------------------------------------------------------------------------
# LLM agent  (optional — requires openai package + API key)
# ---------------------------------------------------------------------------


def _build_llm_prompt(obs: dict[str, Any], task: Task) -> str:
    obs_str = json.dumps(obs, indent=2)
    return textwrap.dedent(f"""
        You are an AI agent controlling a mobile app.

        TASK: {task.instruction}

        CURRENT OBSERVATION:
        {obs_str}

        Available action types: tap, type, back, finish
        - tap:    {{"action": "tap", "target": "<element_name>"}}
        - type:   {{"action": "type", "target": "note_input", "text": "<text>"}}
        - back:   {{"action": "back"}}
        - finish: {{"action": "finish"}}

        Respond ONLY with a valid JSON array of actions. No markdown, no explanation.
        Example: [{{"action": "tap", "target": "notes_button"}}, {{"action": "finish"}}]
    """).strip()


def make_llm_agent(model: str = "gpt-4o-mini") -> Callable:
    """
    Return an agent function backed by an OpenAI-compatible LLM endpoint.
    Requires the ``openai`` package and OPENAI_API_KEY to be set.
    """
    try:
        import openai  # type: ignore
    except ImportError:
        print("ERROR: Install the openai package to use --agent llm.")
        sys.exit(1)

    client = openai.OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )

    def llm_agent(obs: dict[str, Any], task: Task) -> list[dict[str, Any]]:
        prompt = _build_llm_prompt(obs, task)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=512,
            )
            raw = response.choices[0].message.content.strip()
            # Strip markdown fences if present
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return json.loads(raw)
        except Exception as exc:
            print(f"  [LLM ERROR] {exc} — falling back to finish")
            return [{"action": "finish"}]

    return llm_agent


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def _compute_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    if n == 0:
        return {}

    successes = [r["reward_info"]["success_reward"] for r in results]
    rewards = [r["reward_info"]["final_reward"] for r in results]
    steps = [r.get("steps_taken", r["reward_info"].get("steps_taken", 0)) for r in results]
    invalid_rates = [r["reward_info"]["invalid_action_penalty"] for r in results]
    safety_viols = [r["reward_info"]["safety_penalty"] for r in results]

    return {
        "total_tasks": n,
        "success_rate": sum(successes) / n,
        "avg_reward": sum(rewards) / n,
        "avg_steps": sum(steps) / n if any(s > 0 for s in steps) else None,
        "invalid_action_rate": sum(invalid_rates) / n,
        "safety_violations": int(sum(safety_viols)),
        "per_goal_type": _per_goal_breakdown(results),
    }


def _per_goal_breakdown(results: list[dict[str, Any]]) -> dict[str, dict]:
    from collections import defaultdict
    buckets: dict[str, list[float]] = defaultdict(list)
    for r in results:
        gtype = r.get("goal_type", "unknown")
        buckets[gtype].append(r["reward_info"]["success_reward"])
    return {
        gt: {"count": len(scores), "success_rate": sum(scores) / len(scores)}
        for gt, scores in sorted(buckets.items())
    }


# ---------------------------------------------------------------------------
# Printer
# ---------------------------------------------------------------------------


_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _col(text: str, colour: str) -> str:
    return f"{colour}{text}{_RESET}" if sys.stdout.isatty() else text


def print_metrics(metrics: dict[str, Any], agent_name: str) -> None:
    sep = "─" * 52
    print(f"\n{_BOLD}{sep}{_RESET}")
    print(f"  Mobile UI Env — Eval Results  [{agent_name}]")
    print(f"{_BOLD}{sep}{_RESET}")
    print(f"  Total eval tasks   : {metrics['total_tasks']}")

    sr = metrics["success_rate"]
    sr_str = f"{sr*100:.1f}%"
    colour = _GREEN if sr >= 0.7 else (_YELLOW if sr >= 0.4 else _RED)
    print(f"  Success rate       : {_col(sr_str, colour)}")
    print(f"  Average reward     : {metrics['avg_reward']:.4f}")
    if metrics.get("avg_steps") is not None:
        print(f"  Average steps      : {metrics['avg_steps']:.1f}")
    print(f"  Invalid action rate: {metrics['invalid_action_rate']:.4f}")
    print(f"  Safety violations  : {metrics['safety_violations']}")

    if metrics["per_goal_type"]:
        print(f"\n  {'Goal type':<28} {'Count':>5}  {'Success':>8}")
        print(f"  {'─'*28} {'─'*5}  {'─'*8}")
        for gtype, info in metrics["per_goal_type"].items():
            pct = f"{info['success_rate']*100:.0f}%"
            c = _GREEN if info["success_rate"] >= 0.7 else _RED
            print(f"  {gtype:<28} {info['count']:>5}  {_col(pct, c):>8}")

    print(f"{_BOLD}{sep}{_RESET}\n")


def print_verbose(results: list[dict[str, Any]]) -> None:
    print(f"\n{'─'*80}")
    print(f"  {'ID':<12} {'Instruction':<38} {'Succ':>5} {'Reward':>7} {'Saf':>4}")
    print(f"{'─'*80}")
    for r in results:
        ri = r["reward_info"]
        succ = _col("✓", _GREEN) if ri["success_reward"] == 1.0 else _col("✗", _RED)
        saf = _col("!", _RED) if ri["safety_penalty"] > 0 else " "
        instr = r.get("instruction", "")[:37]
        print(
            f"  {r['task_id']:<12} {instr:<38} {succ:>5} "
            f"{ri['final_reward']:>7.4f} {saf:>4}"
        )
    print(f"{'─'*80}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_eval(
    agent_fn: Callable,
    split: str = "eval",
    verbose: bool = False,
    output_path: str | None = None,
    agent_name: str = "heuristic",
) -> dict[str, Any]:
    tasks = build_dataset(split)  # type: ignore[arg-type]
    results: list[dict[str, Any]] = []

    print(f"\nRunning {agent_name} agent on {len(tasks)} {split} tasks…")
    t0 = time.time()

    for task in tasks:
        env = MobileUIEnv(task)
        obs = env.reset()
        actions = agent_fn(obs, task)
        result = env.step(actions)

        # Attach metadata for reporting
        result["task_id"] = task.task_id
        result["instruction"] = task.instruction
        result["goal_type"] = task.goal["type"]
        result["steps_taken"] = env.state.steps_taken
        results.append(result)

    elapsed = time.time() - t0
    print(f"Finished in {elapsed:.2f}s")

    if verbose:
        print_verbose(results)

    metrics = _compute_metrics(results)
    print_metrics(metrics, agent_name)

    if output_path:
        payload = {"agent": agent_name, "split": split, "metrics": metrics, "results": [
            {
                "task_id": r["task_id"],
                "instruction": r["instruction"],
                "goal_type": r["goal_type"],
                "steps_taken": r["steps_taken"],
                "reward_info": r["reward_info"],
                "done": r["done"],
            }
            for r in results
        ]}
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Results saved to {output_path}\n")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the Mobile UI Agent environment."
    )
    parser.add_argument(
        "--agent",
        choices=["heuristic", "llm", "dummy"],
        default="heuristic",
        help="Which agent to use (default: heuristic).",
    )
    parser.add_argument(
        "--split",
        choices=["train", "eval"],
        default="eval",
        help="Which dataset split to evaluate (default: eval).",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="LLM model name (used only with --agent llm).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-task results table.",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Save results as JSON to FILE.",
    )
    args = parser.parse_args()

    if args.agent == "heuristic":
        agent_fn = heuristic_agent
        agent_name = "heuristic-baseline"

    elif args.agent == "dummy":
        def agent_fn(obs, task):  # type: ignore
            return [{"action": "finish"}]
        agent_name = "dummy-finish-only"

    elif args.agent == "llm":
        agent_fn = make_llm_agent(model=args.model)
        agent_name = f"llm:{args.model}"

    else:
        parser.print_help()
        sys.exit(1)

    run_eval(
        agent_fn=agent_fn,
        split=args.split,
        verbose=args.verbose,
        output_path=args.output,
        agent_name=agent_name,
    )


if __name__ == "__main__":
    main()
