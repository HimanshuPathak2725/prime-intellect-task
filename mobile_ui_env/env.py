"""
Core environment class and ``load_environment()`` factory.

``MobileUIEnv`` exposes a Gym-style ``reset() / step()`` interface for a
single episode.  It is deliberately stateless between instantiations — create
a new instance (or call ``reset()``) for each new episode.

``load_environment()`` builds the full train/eval split and wraps everything
in a Verifiers-compatible container.  It tries to import the ``verifiers``
package from Prime Intellect; if that is unavailable it falls back to
lightweight built-in stubs that expose an identical interface so that the
rest of the code base works without modification.

Verifiers interface contract
----------------------------
The returned object must have:
    .dataset        – list of Task objects for training
    .eval_dataset   – list of Task objects for evaluation
    .rubric         – Rubric with a .score(state, task, actions) method
    .make_env(task) – returns a MobileUIEnv for a given task
    .evaluate(agent_fn, split) – runs agent_fn over the split
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .actions import execute_action_sequence
from .dataset import Task, build_dataset
from .rubric import (
    compute_reward,
    efficiency_reward,
    format_reward,
    invalid_action_penalty,
    safety_penalty,
    success_reward,
)
from .state import AppState

# ---------------------------------------------------------------------------
# Core environment
# ---------------------------------------------------------------------------


class MobileUIEnv:
    """
    Single-episode Mobile UI environment with Gym-style API.

    Lifecycle
    ---------
    1. ``env = MobileUIEnv(task)``
    2. ``obs = env.reset()``          — start a fresh episode
    3. ``result = env.step(actions)`` — execute agent actions
    4. Read ``result["reward_info"]["final_reward"]`` and ``result["done"]``

    Alternatively, use ``env.run_episode(actions)`` which combines
    reset + step in one call.
    """

    def __init__(self, task: Task) -> None:
        self.task = task
        self._state: AppState | None = None
        self._actions_log: list[dict[str, Any]] = []

    # ── Gym-style interface ────────────────────────────────────────────────

    def reset(self) -> dict[str, Any]:
        """Reset to a fresh episode and return the initial observation."""
        self._state = AppState()
        self._actions_log = []
        return self._observe()

    def step(self, actions: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Execute *actions* and return a transition dict.

        Parameters
        ----------
        actions : list of action dicts produced by the agent

        Returns
        -------
        dict
            observation    – current screen observation after the actions
            done           – True when the episode has terminated
            reward_info    – all reward components + ``final_reward``
            action_results – per-action outcome dicts
        """
        if self._state is None:
            raise RuntimeError(
                "reset() must be called before step(). "
                "Alternatively, use run_episode() for a one-shot rollout."
            )

        # Record the actions this agent submitted.
        self._actions_log.extend(actions)

        # Execute actions, stopping at finish or max_steps.
        raw_results = execute_action_sequence(
            actions, self._state, max_steps=self.task.max_steps
        )

        # Auto-terminate when budget is exhausted.
        if self._state.steps_taken >= self.task.max_steps and not self._state.done:
            self._state.done = True

        reward_info = compute_reward(self._state, self.task, self._actions_log)

        return {
            "observation": self._observe(),
            "done": self._state.done,
            "reward_info": reward_info,
            "action_results": [
                {
                    "valid": r.valid,
                    "safety_violation": r.safety_violation,
                    "message": r.message,
                    "finished": r.finished,
                }
                for r in raw_results
            ],
        }

    def run_episode(self, actions: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Convenience: reset then step in one call.

        Useful for single-turn evaluation where the agent produces all
        actions in a single forward pass.
        """
        self.reset()
        return self.step(actions)

    # ── Observation ───────────────────────────────────────────────────────

    def _observe(self) -> dict[str, Any]:
        obs = self._state.observation()
        obs["task"] = {
            "instruction": self.task.instruction,
            "max_steps": self.task.max_steps,
            "steps_taken": self._state.steps_taken,
        }
        return obs

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def state(self) -> AppState:
        if self._state is None:
            raise RuntimeError("Call reset() first")
        return self._state

    @property
    def actions_log(self) -> list[dict[str, Any]]:
        """Return a copy of the full action history for this episode."""
        return list(self._actions_log)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"MobileUIEnv(task={self.task.task_id!r}, "
            f"done={self._state.done if self._state else 'not-started'})"
        )


# ---------------------------------------------------------------------------
# Verifiers stubs (used when the prime-intellect `verifiers` package is absent)
# ---------------------------------------------------------------------------


class _Rubric:
    """Lightweight Rubric stub that mirrors the Verifiers API."""

    def __init__(self, funcs: list, weights: list) -> None:
        self.funcs = funcs
        self.weights = weights

    def score(
        self,
        state: AppState,
        task: Task,
        actions: list[dict[str, Any]],
    ) -> float:
        raw = sum(w * f(state, task, actions) for f, w in zip(self.funcs, self.weights, strict=True))
        return max(0.0, min(1.0, raw))


class _SingleTurnEnv:
    """
    Lightweight SingleTurnEnv stub that mirrors the Verifiers API.

    Attributes
    ----------
    dataset      – training tasks
    eval_dataset – evaluation tasks
    rubric       – scoring rubric
    """

    def __init__(
        self,
        dataset: list[Task],
        eval_dataset: list[Task],
        rubric: _Rubric,
    ) -> None:
        self.dataset = dataset
        self.eval_dataset = eval_dataset
        self.rubric = rubric

    def make_env(self, task: Task) -> MobileUIEnv:
        """Instantiate a fresh MobileUIEnv for *task*."""
        return MobileUIEnv(task)

    def evaluate(
        self,
        agent_fn: Callable[[dict[str, Any], Task], list[dict[str, Any]]],
        split: str = "eval",
    ) -> list[dict[str, Any]]:
        """
        Run *agent_fn* over every task in the requested split.

        Parameters
        ----------
        agent_fn : callable
            Signature: ``agent_fn(observation: dict, task: Task) -> list[dict]``
        split    : "train" | "eval"

        Returns
        -------
        List of result dicts, one per task (includes task_id and reward_info).
        """
        tasks = self.eval_dataset if split == "eval" else self.dataset
        results: list[dict[str, Any]] = []
        for task in tasks:
            env = self.make_env(task)
            obs = env.reset()
            actions = agent_fn(obs, task)
            result = env.step(actions)
            result["task_id"] = task.task_id
            result["instruction"] = task.instruction
            results.append(result)
        return results


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def load_environment():
    """
    Build and return a Verifiers-compatible environment object.

    Attempts to import the ``verifiers`` package (Prime Intellect / PRIME-RL).
    If unavailable, returns a built-in stub with an identical interface.

    Usage
    -----
    ::

        env = load_environment()
        results = env.evaluate(my_agent_fn, split="eval")

    Verifiers-native usage (when package is installed)
    ---------------------------------------------------
    ::

        import verifiers as vf
        env = load_environment()          # returns vf.SingleTurnEnv
        # train with GRPO / PPO as usual
    """
    dataset = build_dataset(split="train")
    eval_dataset = build_dataset(split="eval")

    _reward_funcs = [
        success_reward,
        format_reward,
        efficiency_reward,
        invalid_action_penalty,
        safety_penalty,
    ]
    # Signs must mirror compute_reward()'s defaults in rubric.py — invalid
    # actions and safety violations are *penalties*, so their weights are
    # negative. (Caught during review: these were previously positive here,
    # which meant the Verifiers-facing rubric rewarded unsafe/invalid
    # trajectories instead of penalising them — see AI_USAGE.md.)
    _weights = [1.0, 0.1, 0.2, -0.1, -0.3]

    try:
        import verifiers as vf  # type: ignore[import]

        rubric = vf.Rubric(funcs=_reward_funcs, weights=_weights)
        return vf.SingleTurnEnv(
            dataset=dataset,
            eval_dataset=eval_dataset,
            rubric=rubric,
        )

    except ImportError:
        rubric = _Rubric(funcs=_reward_funcs, weights=_weights)
        return _SingleTurnEnv(
            dataset=dataset,
            eval_dataset=eval_dataset,
            rubric=rubric,
        )
