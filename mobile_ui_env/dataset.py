"""
Task dataset for the Mobile UI Agent environment.

Catalogue
---------
20 training tasks  +  10 evaluation tasks  =  30 tasks total.

Every task is a ``Task`` dataclass with:

    task_id     – unique string identifier
    instruction – natural-language command given to the agent
    goal        – structured goal specification evaluated by the rubric
    max_steps   – hard step limit for the episode
    split       – "train" or "eval"
    hints       – optional chain-of-thought hints (for debugging / few-shot)

Goal types
----------
note_created         single note must appear in state.notes
multi_note_created   two-or-more notes must all appear in state.notes
setting_enabled      boolean setting must be True after the episode
setting_disabled     boolean setting must be False after the episode
info_found           agent must tap the element that exposes the value
version_reported     agent must visit settings (version_label readable there)
screen_visited       agent must navigate to a screen without forbidden taps
multi_goal           conjunction of two or more subgoals; optional forbidden list

Design notes
------------
* Train and eval tasks cover the same goal-type distribution so that the
  rubric code path is exercised fully during training.
* Eval tasks use *different* surface-level content (different note titles,
  combined goals) to test generalisation beyond memorisation.
* ``max_steps`` is set to leave a small buffer above the optimal path length
  so the efficiency reward is non-trivially informative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------


@dataclass
class Task:
    task_id: str
    instruction: str
    goal: Dict[str, Any]
    max_steps: int
    split: Literal["train", "eval"]
    hints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict (matches the spec JSON schema)."""
        return {
            "task_id": self.task_id,
            "instruction": self.instruction,
            "goal": self.goal,
            "max_steps": self.max_steps,
            "split": self.split,
        }


# ---------------------------------------------------------------------------
# Task catalogue
# ---------------------------------------------------------------------------

_CATALOGUE: List[Task] = [
    # ════════════════════════════════════════════════════════════
    #  TRAIN  (20 tasks)
    # ════════════════════════════════════════════════════════════

    # ── Single note creation ─────────────────────────────────────
    Task(
        task_id="train_001",
        instruction='Create a note titled "Buy milk"',
        goal={"type": "note_created", "title": "Buy milk"},
        max_steps=8,
        split="train",
        hints=[
            "Navigate to the Notes screen via notes_button",
            "Tap add_note_button to prepare the input",
            "Type the note title into note_input",
            "Tap save_note_button to persist the note",
            "Call finish",
        ],
    ),
    Task(
        task_id="train_002",
        instruction='Create a note titled "Call dentist"',
        goal={"type": "note_created", "title": "Call dentist"},
        max_steps=8,
        split="train",
    ),
    Task(
        task_id="train_003",
        instruction='Add a note that says "Team meeting at 3pm"',
        goal={"type": "note_created", "title": "Team meeting at 3pm"},
        max_steps=8,
        split="train",
    ),
    Task(
        task_id="train_004",
        instruction='Write a note titled "Water the plants"',
        goal={"type": "note_created", "title": "Water the plants"},
        max_steps=8,
        split="train",
    ),
    Task(
        task_id="train_005",
        instruction='Create a note titled "Pick up kids"',
        goal={"type": "note_created", "title": "Pick up kids"},
        max_steps=8,
        split="train",
    ),

    # ── Settings ─────────────────────────────────────────────────
    Task(
        task_id="train_006",
        instruction="Enable focus mode",
        goal={"type": "setting_enabled", "setting": "focus_mode"},
        max_steps=5,
        split="train",
        hints=["Navigate to Settings via settings_button", "Tap focus_mode_toggle"],
    ),
    Task(
        task_id="train_007",
        instruction="Turn off notifications",
        goal={"type": "setting_disabled", "setting": "notifications"},
        max_steps=5,
        split="train",
    ),
    Task(
        task_id="train_008",
        instruction="Make sure notifications are enabled",
        goal={"type": "setting_enabled", "setting": "notifications"},
        max_steps=5,
        split="train",
        hints=["Notifications start ON; verify without toggling, then finish"],
    ),
    Task(
        task_id="train_009",
        instruction="Disable focus mode",
        goal={"type": "setting_disabled", "setting": "focus_mode"},
        max_steps=5,
        split="train",
    ),
    Task(
        task_id="train_010",
        instruction="Open settings and report the app version number",
        goal={"type": "version_reported"},
        max_steps=6,
        split="train",
        hints=["Navigate to Settings", "Tap version_label to read the version", "Call finish"],
    ),

    # ── Profile info ─────────────────────────────────────────────
    Task(
        task_id="train_011",
        instruction="Find the username shown in the profile screen",
        goal={"type": "info_found", "field": "username", "value": "alice"},
        max_steps=5,
        split="train",
        hints=["Navigate to Profile via profile_button", "Tap username_label to read it"],
    ),
    Task(
        task_id="train_012",
        instruction="Find the email address shown on the profile screen",
        goal={"type": "info_found", "field": "email", "value": "alice@example.com"},
        max_steps=5,
        split="train",
    ),

    # ── Multi-note creation ───────────────────────────────────────
    Task(
        task_id="train_013",
        instruction='Create two notes: "Read book" and "Exercise"',
        goal={"type": "multi_note_created", "titles": ["Read book", "Exercise"]},
        max_steps=16,
        split="train",
    ),
    Task(
        task_id="train_014",
        instruction='Create two notes: "Grocery list" and "Pay rent"',
        goal={"type": "multi_note_created", "titles": ["Grocery list", "Pay rent"]},
        max_steps=16,
        split="train",
    ),

    # ── Safe screen visit ─────────────────────────────────────────
    Task(
        task_id="train_015",
        instruction="Visit the profile screen without pressing the logout button",
        goal={
            "type": "screen_visited",
            "screen": "profile",
            "forbidden_actions": ["logout_button"],
        },
        max_steps=5,
        split="train",
    ),

    # ── Combined / multi-goal ─────────────────────────────────────
    Task(
        task_id="train_016",
        instruction='Enable focus mode and create a note titled "Focus session"',
        goal={
            "type": "multi_goal",
            "subgoals": [
                {"type": "setting_enabled", "setting": "focus_mode"},
                {"type": "note_created", "title": "Focus session"},
            ],
        },
        max_steps=14,
        split="train",
    ),
    Task(
        task_id="train_017",
        instruction='Create a note titled "Buy milk" and then disable notifications',
        goal={
            "type": "multi_goal",
            "subgoals": [
                {"type": "note_created", "title": "Buy milk"},
                {"type": "setting_disabled", "setting": "notifications"},
            ],
        },
        max_steps=14,
        split="train",
    ),
    Task(
        task_id="train_018",
        instruction='Save a note called "Standup notes"',
        goal={"type": "note_created", "title": "Standup notes"},
        max_steps=8,
        split="train",
    ),
    Task(
        task_id="train_019",
        instruction="Navigate to Settings and disable focus mode",
        goal={"type": "setting_disabled", "setting": "focus_mode"},
        max_steps=6,
        split="train",
    ),
    Task(
        task_id="train_020",
        instruction='Create three notes: "Alpha", "Beta", and "Gamma"',
        goal={"type": "multi_note_created", "titles": ["Alpha", "Beta", "Gamma"]},
        max_steps=22,
        split="train",
    ),

    # ════════════════════════════════════════════════════════════
    #  EVAL  (10 tasks)
    #  — Different surface-level content to test generalisation.
    # ════════════════════════════════════════════════════════════

    Task(
        task_id="eval_001",
        instruction='Create a note titled "Send invoice"',
        goal={"type": "note_created", "title": "Send invoice"},
        max_steps=8,
        split="eval",
    ),
    Task(
        task_id="eval_002",
        instruction="Turn on focus mode via the settings screen",
        goal={"type": "setting_enabled", "setting": "focus_mode"},
        max_steps=5,
        split="eval",
    ),
    Task(
        task_id="eval_003",
        instruction="Disable notifications in the app settings",
        goal={"type": "setting_disabled", "setting": "notifications"},
        max_steps=5,
        split="eval",
    ),
    Task(
        task_id="eval_004",
        instruction="Look up the username displayed on the profile screen",
        goal={"type": "info_found", "field": "username", "value": "alice"},
        max_steps=5,
        split="eval",
    ),
    Task(
        task_id="eval_005",
        instruction="Find the email address listed in the user profile",
        goal={"type": "info_found", "field": "email", "value": "alice@example.com"},
        max_steps=5,
        split="eval",
    ),
    Task(
        task_id="eval_006",
        instruction="Check what version of the app is installed",
        goal={"type": "version_reported"},
        max_steps=6,
        split="eval",
    ),
    Task(
        task_id="eval_007",
        instruction='Create two notes: "Prepare slides" and "Send report"',
        goal={"type": "multi_note_created", "titles": ["Prepare slides", "Send report"]},
        max_steps=16,
        split="eval",
    ),
    Task(
        task_id="eval_008",
        instruction="Open the profile page without tapping the logout button",
        goal={
            "type": "screen_visited",
            "screen": "profile",
            "forbidden_actions": ["logout_button"],
        },
        max_steps=5,
        split="eval",
    ),
    Task(
        task_id="eval_009",
        instruction='Enable focus mode and then add a note called "Deep work"',
        goal={
            "type": "multi_goal",
            "subgoals": [
                {"type": "setting_enabled", "setting": "focus_mode"},
                {"type": "note_created", "title": "Deep work"},
            ],
        },
        max_steps=14,
        split="eval",
    ),
    Task(
        task_id="eval_010",
        instruction='Create a note called "Review PR" without logging out',
        goal={
            "type": "multi_goal",
            "subgoals": [
                {"type": "note_created", "title": "Review PR"},
            ],
            "forbidden_actions": ["logout_button"],
        },
        max_steps=10,
        split="eval",
    ),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_dataset(split: Literal["train", "eval", "all"] = "train") -> List[Task]:
    """
    Return tasks for the requested split.

    Parameters
    ----------
    split : "train" | "eval" | "all"
        Which tasks to return.  "all" returns the full catalogue.
    """
    if split == "all":
        return list(_CATALOGUE)
    return [t for t in _CATALOGUE if t.split == split]


def get_task_by_id(task_id: str) -> Task:
    """
    Retrieve a single task by its ID.

    Raises ``KeyError`` if the task is not found.
    """
    for task in _CATALOGUE:
        if task.task_id == task_id:
            return task
    raise KeyError(
        f"Task {task_id!r} not found. "
        f"Available IDs: {[t.task_id for t in _CATALOGUE]}"
    )
