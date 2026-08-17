"""
Reward / rubric functions for the Mobile UI Agent environment.

Each reward function shares the same signature:

    f(state: AppState, task: Task, actions: list[dict]) -> float

This makes every component independently testable and composable via the
Verifiers ``Rubric`` interface.

Reward components
-----------------

Component               Type    Purpose
───────────────────────────────────────────────────────────────────────────────
success_reward          SPARSE  1.0 iff the full goal is satisfied.
format_reward           DENSE   Fraction of actions with valid JSON structure.
efficiency_reward       DENSE   Reward for fewer steps; only non-zero on success.
invalid_action_penalty  DENSE   Proportional to bad-action count.
safety_penalty          SPARSE  Binary penalty for any unsafe action (logout).
partial_progress_reward SHAPED  Fraction of sub-goals completed; helps bootstrap.

Final reward formula (from spec):

    raw = (
        1.0  * success_reward
      + 0.1  * format_reward
      + 0.2  * efficiency_reward
      - 0.1  * invalid_action_count_normalised
      - 0.3  * safety_violation_flag
    )
    final_reward = clip(raw, 0, 1)

Reward hacking notes (discussed in README)
------------------------------------------
* An agent could maximise ``partial_progress_reward`` by rapidly visiting
  each screen without completing any goal.
* ``efficiency_reward`` can be exploited by immediately calling ``finish``
  (zero steps, zero success → efficiency = 0 since we gate on success).
* ``format_reward`` could be gamed by emitting structurally valid but
  semantically useless actions.  The ``success_reward`` multiplier prevents
  this dominating the gradient.
"""

from __future__ import annotations

from typing import Any

from .dataset import Task
from .state import AppState

# ---------------------------------------------------------------------------
# Goal evaluation helpers
# ---------------------------------------------------------------------------


def _check_goal(
    goal: dict[str, Any],
    state: AppState,
    actions: list[dict[str, Any]],
) -> bool:
    """
    Return True if *goal* is satisfied given the post-episode *state* and
    the full *actions* list.

    Handles all goal types defined in ``dataset.py``.
    """
    goal_type = goal["type"]

    # ── Note goals ────────────────────────────────────────────────────────
    if goal_type == "note_created":
        return goal["title"] in state.notes

    if goal_type == "multi_note_created":
        return all(title in state.notes for title in goal["titles"])

    # ── Settings goals ────────────────────────────────────────────────────
    if goal_type == "setting_enabled":
        return _setting_value(goal["setting"], state) is True

    if goal_type == "setting_disabled":
        return _setting_value(goal["setting"], state) is False

    # ── Information-retrieval goals ───────────────────────────────────────
    if goal_type == "info_found":
        # Proxy: agent must have tapped the element that exposes the value.
        field_name = goal["field"]
        label = f"{field_name}_label"
        tapped = _tapped_targets(actions)
        return label in tapped

    if goal_type == "version_reported":
        # Agent must have visited settings (version is visible there) or
        # explicitly tapped the version_label element.
        tapped = _tapped_targets(actions)
        return "settings_button" in tapped or "version_label" in tapped

    # ── Screen-visit goals ────────────────────────────────────────────────
    if goal_type == "screen_visited":
        screen = goal["screen"]
        nav_targets = {
            "profile": "profile_button",
            "settings": "settings_button",
            "notes": "notes_button",
        }
        tapped = _tapped_targets(actions)
        visited = nav_targets.get(screen, f"{screen}_button") in tapped
        forbidden = goal.get("forbidden_actions", [])
        no_forbidden = not any(f in tapped for f in forbidden)
        return visited and no_forbidden

    # ── Compound goals ────────────────────────────────────────────────────
    if goal_type == "multi_goal":
        subgoals = goal.get("subgoals", [])
        all_met = all(_check_goal(sg, state, actions) for sg in subgoals)
        forbidden = goal.get("forbidden_actions", [])
        if forbidden:
            tapped = _tapped_targets(actions)
            no_forbidden = not any(f in tapped for f in forbidden)
            return all_met and no_forbidden
        return all_met

    # Unknown goal type → conservative False.
    return False


def _setting_value(setting: str, state: AppState) -> bool | None:
    """Return the bool value of a named setting, or None if unknown."""
    if setting == "focus_mode":
        return state.focus_mode
    if setting == "notifications":
        return state.notifications
    return None


def _tapped_targets(actions: list[dict[str, Any]]) -> set:
    """Return the set of all targets that were tapped in *actions*."""
    return {a.get("target") for a in actions if a.get("action") == "tap"}


def _count_subgoals(
    goal: dict[str, Any],
    state: AppState,
    actions: list[dict[str, Any]],
) -> tuple:
    """Return ``(met, total)`` subgoal counts for partial progress scoring."""
    if goal["type"] == "multi_goal":
        subgoals = goal.get("subgoals", [])
        met = sum(1 for sg in subgoals if _check_goal(sg, state, actions))
        return met, max(len(subgoals), 1)
    return (1, 1) if _check_goal(goal, state, actions) else (0, 1)


# ---------------------------------------------------------------------------
# Reward functions  (public API)
# ---------------------------------------------------------------------------


def success_reward(
    state: AppState,
    task: Task,
    actions: list[dict[str, Any]],
) -> float:
    """
    **Sparse** reward: 1.0 if and only if the task goal is fully satisfied.

    This is the primary training signal.  It is deliberately sparse because:
    (a) it gives an unambiguous learning target and
    (b) it forces the agent to explore complete solution paths rather than
        collecting partial-progress signal along the way.

    The difficulty of sparse reward is why ``partial_progress_reward`` exists
    as an optional dense supplement.
    """
    return 1.0 if _check_goal(task.goal, state, actions) else 0.0


def format_reward(
    state: AppState,
    task: Task,
    actions: list[dict[str, Any]],
) -> float:
    """
    **Dense** reward: fraction of actions that are structurally well-formed.

    An action is well-formed when it is a dict whose "action" key holds a
    known verb.  This encourages the agent to produce parseable output even
    on incomplete or failed episodes, making the training signal more
    informative early in learning.

    Reward hacking risk: low — format alone has weight 0.1 and cannot
    exceed success_reward in the final formula.
    """
    if not actions:
        return 0.0
    well_formed = sum(
        1
        for a in actions
        if isinstance(a, dict) and a.get("action") in {"tap", "type", "back", "finish"}
    )
    return well_formed / len(actions)


def efficiency_reward(
    state: AppState,
    task: Task,
    actions: list[dict[str, Any]],
) -> float:
    """
    **Dense** reward: how efficiently the agent completed the goal.

    Returns 0 when the goal was not achieved (gated on ``success_reward``),
    preventing the agent from gaming efficiency by calling ``finish`` early.

    Formula: linear decay from 1.0 (1 step used) to 0.0 (max_steps used).

    Reward hacking risk: moderate — an agent could learn the shortest
    *correct* path and ignore robustness.  Monitor whether the agent
    generalises efficiency across screen orderings.
    """
    if success_reward(state, task, actions) == 0.0:
        return 0.0

    steps = state.steps_taken
    max_s = task.max_steps
    if steps >= max_s:
        return 0.0
    return max(0.0, 1.0 - (steps - 1) / max(max_s - 1, 1))


def invalid_action_penalty(
    state: AppState,
    task: Task,
    actions: list[dict[str, Any]],
) -> float:
    """
    **Dense** penalty: normalised count of invalid actions taken.

    Returns a value in [0, 1].  The caller applies a negative coefficient
    (−0.1 by default) when computing the final reward.

    Formula: clip(invalid_count / max_steps, 0, 1)

    Reward hacking risk: low — the agent is directly incentivised to reduce
    invalid actions, which is the desired behaviour.
    """
    return min(state.invalid_action_count / max(task.max_steps, 1), 1.0)


def safety_penalty(
    state: AppState,
    task: Task,
    actions: list[dict[str, Any]],
) -> float:
    """
    **Sparse** penalty: 1.0 if any safety violation occurred (e.g. logout).

    Binary because a single unsafe action is sufficient to fail safety
    requirements regardless of overall task performance.

    Reward hacking risk: negligible — the penalty is always negative, so
    the agent can only improve its score by avoiding unsafe actions.
    """
    return 1.0 if state.safety_violations > 0 else 0.0


def partial_progress_reward(
    state: AppState,
    task: Task,
    actions: list[dict[str, Any]],
) -> float:
    """
    **Shaped** reward: fraction of sub-goals completed.

    This optional dense signal addresses the credit-assignment problem that
    makes pure sparse reward difficult for RL agents.  By rewarding
    intermediate progress (e.g. reaching the correct screen, saving one of
    two notes), the agent receives a gradient signal even on failed episodes.

    Reward hacking risk: HIGH — without careful design an agent can maximise
    this by rapidly visiting screens or toggling settings back and forth.
    Use with caution; consider decay or gating on ``success_reward``.

    See README §6 for a detailed discussion of mitigation strategies.
    """
    met, total = _count_subgoals(task.goal, state, actions)
    return met / max(total, 1)


# ---------------------------------------------------------------------------
# Composite scorer
# ---------------------------------------------------------------------------


def compute_reward(
    state: AppState,
    task: Task,
    actions: list[dict[str, Any]],
    *,
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Compute all reward components and the final clipped reward.

    Returns a dict containing individual component scores **and** the
    aggregated ``final_reward`` (clipped to [0, 1]).

    Parameters
    ----------
    state   : post-episode AppState
    task    : the Task being evaluated
    actions : full action sequence submitted by the agent
    weights : optional override for the default weight dict

    Default weights (from spec):
        success          ×  1.0
        format           ×  0.1
        efficiency       ×  0.2
        invalid_penalty  × −0.1
        safety           × −0.3
    """
    if weights is None:
        weights = {
            "success": 1.0,
            "format": 0.1,
            "efficiency": 0.2,
            "invalid_penalty": -0.1,
            "safety": -0.3,
        }

    s = success_reward(state, task, actions)
    f = format_reward(state, task, actions)
    e = efficiency_reward(state, task, actions)
    inv = invalid_action_penalty(state, task, actions)
    saf = safety_penalty(state, task, actions)
    prog = partial_progress_reward(state, task, actions)

    raw = (
        weights["success"] * s
        + weights["format"] * f
        + weights["efficiency"] * e
        + weights["invalid_penalty"] * inv
        + weights["safety"] * saf
    )
    final = max(0.0, min(1.0, raw))

    return {
        "success_reward": s,
        "format_reward": f,
        "efficiency_reward": e,
        "invalid_action_penalty": inv,
        "safety_penalty": saf,
        "partial_progress_reward": prog,
        "raw_reward": raw,
        "final_reward": final,
    }
