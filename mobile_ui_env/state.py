"""
State management for the Mobile UI environment.

The *state space* captures every piece of information needed to determine what
the agent can do and how the app will respond:

    current_screen      – one of four named screens (HOME, NOTES, SETTINGS, PROFILE)
    notes               – ordered list of saved note titles
    focus_mode          – whether Focus Mode is currently on
    notifications       – whether Notifications are currently on
    note_input_buffer   – text currently typed into the note_input element

Episode bookkeeping (not part of the MDP observation, but needed by the rubric):

    steps_taken         – number of actions executed so far
    invalid_action_count – cumulative count of structurally-invalid actions
    safety_violations   – cumulative count of unsafe actions (e.g. logout)
    done                – whether the episode has terminated

Static app data (username, email, app_version) is fixed per environment
instance and is included in screen observations but never mutates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# ---------------------------------------------------------------------------
# Screen definitions
# ---------------------------------------------------------------------------


class Screen(str, Enum):
    """The four navigable screens of the mock mobile app."""

    HOME = "home"
    NOTES = "notes"
    SETTINGS = "settings"
    PROFILE = "profile"


# Elements visible (and interactable) on each screen.
SCREEN_ELEMENTS: dict[Screen, list[str]] = {
    Screen.HOME: ["notes_button", "settings_button", "profile_button"],
    Screen.NOTES: ["add_note_button", "note_input", "save_note_button", "note_list"],
    Screen.SETTINGS: ["focus_mode_toggle", "notifications_toggle", "version_label"],
    Screen.PROFILE: ["username_label", "email_label", "logout_button"],
}

# Which tap-targets navigate to which screen.
NAV_MAP: dict[str, Screen] = {
    "notes_button": Screen.NOTES,
    "settings_button": Screen.SETTINGS,
    "profile_button": Screen.PROFILE,
}

# Elements that accept text input via the "type" action.
TYPEABLE_ELEMENTS: set = {"note_input"}

# Static read-only app data — would come from the real app's database in
# a live emulator environment.
APP_STATIC: dict[str, str] = {
    "username": "alice",
    "email": "alice@example.com",
    "app_version": "1.4.2",
}


# ---------------------------------------------------------------------------
# State dataclass
# ---------------------------------------------------------------------------


@dataclass
class AppState:
    """
    Complete mutable state of the simulated mobile app for one episode.

    Instantiate fresh at the start of every episode (via AppState() defaults).
    """

    # ── Visible app state ──────────────────────────────────────────────────
    current_screen: Screen = Screen.HOME
    notes: list[str] = field(default_factory=list)
    focus_mode: bool = False
    notifications: bool = True
    note_input_buffer: str = ""

    # ── Episode bookkeeping ────────────────────────────────────────────────
    steps_taken: int = 0
    invalid_action_count: int = 0
    safety_violations: int = 0
    done: bool = False

    # ── Helpers ────────────────────────────────────────────────────────────

    def available_elements(self) -> list[str]:
        """Return the UI elements visible on the current screen."""
        return list(SCREEN_ELEMENTS[self.current_screen])

    def observation(self) -> dict:
        """
        Return a structured observation dict the agent can read.

        This mirrors the kind of flattened accessibility-tree snapshot you
        would receive from a real Android emulator (a11y XML → key-value).
        The agent's policy should condition only on this dict, not on the raw
        AppState object.
        """
        obs: dict = {
            "screen": self.current_screen.value,
            "elements": self.available_elements(),
        }

        if self.current_screen == Screen.NOTES:
            obs["note_list"] = list(self.notes)
            obs["note_input_buffer"] = self.note_input_buffer

        elif self.current_screen == Screen.SETTINGS:
            obs["focus_mode"] = self.focus_mode
            obs["notifications"] = self.notifications
            obs["version_label"] = APP_STATIC["app_version"]

        elif self.current_screen == Screen.PROFILE:
            obs["username_label"] = APP_STATIC["username"]
            obs["email_label"] = APP_STATIC["email"]

        return obs

    def clone(self) -> AppState:
        """Return a deep-enough copy for rollout branching / tree search."""
        return AppState(
            current_screen=self.current_screen,
            notes=list(self.notes),
            focus_mode=self.focus_mode,
            notifications=self.notifications,
            note_input_buffer=self.note_input_buffer,
            steps_taken=self.steps_taken,
            invalid_action_count=self.invalid_action_count,
            safety_violations=self.safety_violations,
            done=self.done,
        )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"AppState(screen={self.current_screen.value!r}, "
            f"notes={self.notes}, "
            f"focus_mode={self.focus_mode}, notifications={self.notifications}, "
            f"steps={self.steps_taken}, done={self.done})"
        )
