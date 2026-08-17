"""
mobile_ui_env — Mobile UI Agent RL Environment
===============================================

A lightweight reinforcement-learning environment that simulates a four-screen
mobile app.  An AI agent completes natural-language tasks by producing
structured JSON action sequences and receives shaped rewards based on task
success, action validity, efficiency, and safety.

Quick start
-----------
>>> from mobile_ui_env import load_environment
>>> env = load_environment()
>>> # env.dataset  → 20 training tasks
>>> # env.eval_dataset → 10 eval tasks

>>> from mobile_ui_env import MobileUIEnv, build_dataset
>>> task = build_dataset("eval")[0]
>>> episode_env = MobileUIEnv(task)
>>> obs = episode_env.reset()
>>> result = episode_env.step([
...     {"action": "tap",  "target": "notes_button"},
...     {"action": "type", "target": "note_input", "text": "Buy milk"},
...     {"action": "tap",  "target": "save_note_button"},
...     {"action": "finish"},
... ])
>>> result["reward_info"]["final_reward"]
"""

from .dataset import Task, build_dataset, get_task_by_id
from .env import MobileUIEnv, load_environment
from .rubric import (
    compute_reward,
    efficiency_reward,
    format_reward,
    invalid_action_penalty,
    partial_progress_reward,
    safety_penalty,
    success_reward,
)
from .state import APP_STATIC, SCREEN_ELEMENTS, AppState, Screen

__version__ = "0.1.0"
__author__ = "Prime Intellect Take-Home"

__all__ = [
    # Environment
    "MobileUIEnv",
    "load_environment",
    # State
    "AppState",
    "Screen",
    "SCREEN_ELEMENTS",
    "APP_STATIC",
    # Dataset
    "Task",
    "build_dataset",
    "get_task_by_id",
    # Rewards
    "success_reward",
    "format_reward",
    "efficiency_reward",
    "invalid_action_penalty",
    "safety_penalty",
    "partial_progress_reward",
    "compute_reward",
]
