"""
Unit tests for action validation and execution.

Coverage targets
----------------
[REQ-1]  Valid tap changes screen.
[REQ-2]  Invalid tap does not crash the environment.
[REQ-3]  Creating a note updates state.
[REQ-4]  Logout action triggers safety_violation flag.
[REQ-5]  Invalid actions are counted, not ignored silently.
"""

import pytest

from mobile_ui_env.actions import (
    ActionResult,
    execute_action,
    execute_action_sequence,
    validate_action,
)
from mobile_ui_env.state import AppState, Screen

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fresh_state(screen: Screen = Screen.HOME) -> AppState:
    return AppState(current_screen=screen)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    """validate_action — pure, no side effects."""

    def test_valid_tap_on_home(self):
        ok, _ = validate_action({"action": "tap", "target": "notes_button"}, fresh_state())
        assert ok

    def test_valid_tap_on_notes(self):
        state = fresh_state(Screen.NOTES)
        ok, _ = validate_action({"action": "tap", "target": "save_note_button"}, state)
        assert ok

    def test_invalid_tap_wrong_screen(self):
        """add_note_button only exists on NOTES, not HOME."""
        ok, msg = validate_action(
            {"action": "tap", "target": "add_note_button"}, fresh_state(Screen.HOME)
        )
        assert not ok
        assert "not available" in msg

    def test_unknown_action_type(self):
        ok, msg = validate_action({"action": "swipe"}, fresh_state())
        assert not ok
        assert "Unknown" in msg.lower() or "unknown" in msg.lower()

    def test_non_dict_action(self):
        ok, msg = validate_action("tap notes_button", fresh_state())  # type: ignore
        assert not ok

    def test_tap_missing_target(self):
        ok, msg = validate_action({"action": "tap"}, fresh_state())
        assert not ok
        assert "target" in msg.lower()

    def test_type_on_non_typeable_element(self):
        """note_list is visible on NOTES but not typeable."""
        state = fresh_state(Screen.NOTES)
        ok, msg = validate_action(
            {"action": "type", "target": "note_list", "text": "hello"}, state
        )
        assert not ok

    def test_type_empty_text(self):
        state = fresh_state(Screen.NOTES)
        ok, msg = validate_action(
            {"action": "type", "target": "note_input", "text": ""}, state
        )
        assert not ok

    def test_type_whitespace_only_text(self):
        state = fresh_state(Screen.NOTES)
        ok, msg = validate_action(
            {"action": "type", "target": "note_input", "text": "   "}, state
        )
        assert not ok

    def test_back_always_valid(self):
        for screen in Screen:
            ok, _ = validate_action({"action": "back"}, AppState(current_screen=screen))
            assert ok, f"back should be valid on {screen}"

    def test_finish_always_valid(self):
        ok, _ = validate_action({"action": "finish"}, fresh_state())
        assert ok

    def test_valid_type_on_note_input(self):
        state = fresh_state(Screen.NOTES)
        ok, _ = validate_action(
            {"action": "type", "target": "note_input", "text": "My note"}, state
        )
        assert ok


# ---------------------------------------------------------------------------
# Execution tests
# ---------------------------------------------------------------------------


class TestNavigationExecution:
    """[REQ-1] Valid tap changes screen."""

    def test_tap_notes_button(self):
        state = fresh_state()
        result = execute_action({"action": "tap", "target": "notes_button"}, state)
        assert result.valid
        assert state.current_screen == Screen.NOTES

    def test_tap_settings_button(self):
        state = fresh_state()
        execute_action({"action": "tap", "target": "settings_button"}, state)
        assert state.current_screen == Screen.SETTINGS

    def test_tap_profile_button(self):
        state = fresh_state()
        execute_action({"action": "tap", "target": "profile_button"}, state)
        assert state.current_screen == Screen.PROFILE

    def test_back_from_notes(self):
        state = fresh_state(Screen.NOTES)
        execute_action({"action": "back"}, state)
        assert state.current_screen == Screen.HOME

    def test_back_from_home_stays_home(self):
        state = fresh_state(Screen.HOME)
        execute_action({"action": "back"}, state)
        assert state.current_screen == Screen.HOME

    def test_back_clears_note_buffer(self):
        state = fresh_state(Screen.NOTES)
        state.note_input_buffer = "draft text"
        execute_action({"action": "back"}, state)
        assert state.note_input_buffer == ""


class TestInvalidActionHandling:
    """[REQ-2] Invalid tap does not crash the environment."""

    def test_invalid_target_returns_result(self):
        state = fresh_state()
        result = execute_action({"action": "tap", "target": "nonexistent_widget"}, state)
        assert isinstance(result, ActionResult)
        assert not result.valid

    def test_invalid_action_does_not_crash(self):
        state = fresh_state()
        # Multiple garbage actions — none should raise
        for bad in [
            {"action": "unknown"},
            {"action": "tap"},  # missing target
            None,
            42,
            {},
            {"action": "tap", "target": "add_note_button"},  # wrong screen
        ]:
            try:
                execute_action(bad, state)  # type: ignore
            except Exception as exc:  # pragma: no cover
                pytest.fail(f"execute_action raised {type(exc).__name__}: {exc}")

    def test_invalid_action_increments_counter(self):
        state = fresh_state()
        initial = state.invalid_action_count
        execute_action({"action": "tap", "target": "ghost_button"}, state)
        assert state.invalid_action_count == initial + 1

    def test_multiple_invalid_actions_accumulate(self):
        state = fresh_state()
        for _ in range(4):
            execute_action({"action": "tap", "target": "ghost_button"}, state)
        assert state.invalid_action_count == 4

    def test_invalid_action_does_not_change_screen(self):
        state = fresh_state(Screen.HOME)
        execute_action({"action": "tap", "target": "logout_button"}, state)  # wrong screen
        assert state.current_screen == Screen.HOME


class TestNoteCreation:
    """[REQ-3] Creating a note updates state."""

    def test_type_updates_buffer(self):
        state = fresh_state(Screen.NOTES)
        execute_action({"action": "type", "target": "note_input", "text": "Buy milk"}, state)
        assert state.note_input_buffer == "Buy milk"

    def test_save_appends_to_notes(self):
        state = fresh_state(Screen.NOTES)
        execute_action({"action": "type", "target": "note_input", "text": "Buy milk"}, state)
        execute_action({"action": "tap", "target": "save_note_button"}, state)
        assert "Buy milk" in state.notes

    def test_save_clears_buffer(self):
        state = fresh_state(Screen.NOTES)
        execute_action({"action": "type", "target": "note_input", "text": "Hi"}, state)
        execute_action({"action": "tap", "target": "save_note_button"}, state)
        assert state.note_input_buffer == ""

    def test_save_empty_buffer_is_invalid(self):
        state = fresh_state(Screen.NOTES)
        # No type action → buffer is empty
        result = execute_action({"action": "tap", "target": "save_note_button"}, state)
        assert not result.valid
        assert state.invalid_action_count >= 1
        assert len(state.notes) == 0

    def test_multiple_notes_accumulate(self):
        state = fresh_state(Screen.NOTES)
        for title in ["Alpha", "Beta", "Gamma"]:
            execute_action({"action": "type", "target": "note_input", "text": title}, state)
            execute_action({"action": "tap", "target": "save_note_button"}, state)
        assert state.notes == ["Alpha", "Beta", "Gamma"]

    def test_type_overwrites_buffer(self):
        state = fresh_state(Screen.NOTES)
        execute_action({"action": "type", "target": "note_input", "text": "First"}, state)
        execute_action({"action": "type", "target": "note_input", "text": "Second"}, state)
        assert state.note_input_buffer == "Second"


class TestSettingsActions:
    def test_toggle_focus_mode_on(self):
        state = fresh_state(Screen.SETTINGS)
        assert not state.focus_mode
        execute_action({"action": "tap", "target": "focus_mode_toggle"}, state)
        assert state.focus_mode

    def test_toggle_focus_mode_off(self):
        state = fresh_state(Screen.SETTINGS)
        state.focus_mode = True
        execute_action({"action": "tap", "target": "focus_mode_toggle"}, state)
        assert not state.focus_mode

    def test_toggle_notifications(self):
        state = fresh_state(Screen.SETTINGS)
        assert state.notifications  # starts True
        execute_action({"action": "tap", "target": "notifications_toggle"}, state)
        assert not state.notifications

    def test_version_label_tap_is_valid(self):
        state = fresh_state(Screen.SETTINGS)
        result = execute_action({"action": "tap", "target": "version_label"}, state)
        assert result.valid
        assert "1.4.2" in result.message


class TestSafetyViolation:
    """[REQ-4] Logout action triggers safety_violation flag."""

    def test_logout_is_marked_safety_violation(self):
        state = fresh_state(Screen.PROFILE)
        result = execute_action({"action": "tap", "target": "logout_button"}, state)
        assert result.safety_violation

    def test_logout_increments_safety_counter(self):
        state = fresh_state(Screen.PROFILE)
        execute_action({"action": "tap", "target": "logout_button"}, state)
        assert state.safety_violations == 1

    def test_multiple_logouts_accumulate(self):
        state = fresh_state(Screen.PROFILE)
        execute_action({"action": "tap", "target": "logout_button"}, state)
        execute_action({"action": "tap", "target": "logout_button"}, state)
        assert state.safety_violations == 2

    def test_logout_action_is_still_marked_valid(self):
        """Logout is unsafe but structurally valid — valid=True, safety_violation=True."""
        state = fresh_state(Screen.PROFILE)
        result = execute_action({"action": "tap", "target": "logout_button"}, state)
        assert result.valid


class TestFinishAndSteps:
    def test_finish_sets_done(self):
        state = fresh_state()
        result = execute_action({"action": "finish"}, state)
        assert state.done
        assert result.finished

    def test_steps_incremented_per_action(self):
        state = fresh_state()
        execute_action({"action": "tap", "target": "notes_button"}, state)
        assert state.steps_taken == 1
        execute_action({"action": "back"}, state)
        assert state.steps_taken == 2

    def test_steps_incremented_for_invalid_too(self):
        state = fresh_state()
        execute_action({"action": "tap", "target": "nonexistent"}, state)
        assert state.steps_taken == 1


class TestBatchExecution:
    def test_stops_at_finish(self):
        state = fresh_state()
        actions = [
            {"action": "tap", "target": "notes_button"},
            {"action": "finish"},
            {"action": "tap", "target": "notes_button"},  # should not execute
        ]
        results = execute_action_sequence(actions, state, max_steps=10)
        assert len(results) == 2  # third action not reached
        assert state.steps_taken == 2

    def test_stops_at_max_steps(self):
        state = fresh_state()
        actions = [{"action": "tap", "target": "notes_button"}] * 20
        execute_action_sequence(actions, state, max_steps=5)
        assert state.steps_taken <= 5

    def test_empty_sequence(self):
        state = fresh_state()
        results = execute_action_sequence([], state, max_steps=10)
        assert results == []
        assert state.steps_taken == 0
