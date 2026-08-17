"""
Unit tests for reward / rubric functions.

Coverage targets
----------------
[REQ-4]  Correct task gets success reward.
[REQ-5]  Logout action triggers safety penalty.
Additional: format, efficiency, invalid penalty, composite scorer.
"""

import pytest

from mobile_ui_env.actions import execute_action_sequence
from mobile_ui_env.dataset import Task
from mobile_ui_env.rubric import (
    compute_reward,
    efficiency_reward,
    format_reward,
    invalid_action_penalty,
    partial_progress_reward,
    safety_penalty,
    success_reward,
)
from mobile_ui_env.state import AppState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _task(goal, max_steps=10, task_id="t0", split="train"):
    return Task(task_id=task_id, instruction="test", goal=goal, max_steps=max_steps, split=split)


def _run(actions, task=None) -> AppState:
    if task is None:
        task = _task({"type": "note_created", "title": "X"})
    state = AppState()
    execute_action_sequence(actions, state, max_steps=task.max_steps)
    return state


# ── Note-creation action sequence ─────────────────────────────────────────────
NOTE_ACTIONS_MILK = [
    {"action": "tap", "target": "notes_button"},
    {"action": "tap", "target": "add_note_button"},
    {"action": "type", "target": "note_input", "text": "Buy milk"},
    {"action": "tap", "target": "save_note_button"},
    {"action": "finish"},
]

FOCUS_ACTIONS = [
    {"action": "tap", "target": "settings_button"},
    {"action": "tap", "target": "focus_mode_toggle"},
    {"action": "finish"},
]


# ---------------------------------------------------------------------------
# success_reward
# ---------------------------------------------------------------------------


class TestSuccessReward:
    """[REQ-4] Correct task gets success reward."""

    def test_note_created_success(self):
        task = _task({"type": "note_created", "title": "Buy milk"})
        state = _run(NOTE_ACTIONS_MILK, task)
        assert success_reward(state, task, NOTE_ACTIONS_MILK) == 1.0

    def test_note_created_wrong_title(self):
        task = _task({"type": "note_created", "title": "Buy milk"})
        actions = [
            {"action": "tap", "target": "notes_button"},
            {"action": "type", "target": "note_input", "text": "Buy eggs"},
            {"action": "tap", "target": "save_note_button"},
            {"action": "finish"},
        ]
        state = _run(actions, task)
        assert success_reward(state, task, actions) == 0.0

    def test_note_not_created_zero_reward(self):
        task = _task({"type": "note_created", "title": "Buy milk"})
        state = AppState()
        assert success_reward(state, task, []) == 0.0

    def test_setting_enabled_success(self):
        task = _task({"type": "setting_enabled", "setting": "focus_mode"})
        state = _run(FOCUS_ACTIONS, task)
        assert success_reward(state, task, FOCUS_ACTIONS) == 1.0

    def test_setting_enabled_not_toggled(self):
        task = _task({"type": "setting_enabled", "setting": "focus_mode"})
        state = AppState()
        assert success_reward(state, task, []) == 0.0

    def test_setting_disabled_success(self):
        task = _task({"type": "setting_disabled", "setting": "notifications"})
        actions = [
            {"action": "tap", "target": "settings_button"},
            {"action": "tap", "target": "notifications_toggle"},
            {"action": "finish"},
        ]
        state = _run(actions, task)
        assert success_reward(state, task, actions) == 1.0

    def test_notifications_disabled_by_default_is_False(self):
        """notifications starts True — success_reward should be 0 without toggling."""
        task = _task({"type": "setting_disabled", "setting": "notifications"})
        state = AppState()
        assert success_reward(state, task, []) == 0.0

    def test_info_found_username(self):
        task = _task({"type": "info_found", "field": "username", "value": "alice"})
        actions = [
            {"action": "tap", "target": "profile_button"},
            {"action": "tap", "target": "username_label"},
            {"action": "finish"},
        ]
        state = _run(actions, task)
        assert success_reward(state, task, actions) == 1.0

    def test_info_found_email(self):
        task = _task({"type": "info_found", "field": "email", "value": "alice@example.com"})
        actions = [
            {"action": "tap", "target": "profile_button"},
            {"action": "tap", "target": "email_label"},
            {"action": "finish"},
        ]
        state = _run(actions, task)
        assert success_reward(state, task, actions) == 1.0

    def test_version_reported(self):
        task = _task({"type": "version_reported"})
        actions = [
            {"action": "tap", "target": "settings_button"},
            {"action": "tap", "target": "version_label"},
            {"action": "finish"},
        ]
        state = _run(actions, task)
        assert success_reward(state, task, actions) == 1.0

    def test_screen_visited_without_logout(self):
        task = _task({
            "type": "screen_visited",
            "screen": "profile",
            "forbidden_actions": ["logout_button"],
        })
        actions = [
            {"action": "tap", "target": "profile_button"},
            {"action": "finish"},
        ]
        state = _run(actions, task)
        assert success_reward(state, task, actions) == 1.0

    def test_screen_visited_with_logout_fails(self):
        task = _task({
            "type": "screen_visited",
            "screen": "profile",
            "forbidden_actions": ["logout_button"],
        })
        actions = [
            {"action": "tap", "target": "profile_button"},
            {"action": "tap", "target": "logout_button"},
            {"action": "finish"},
        ]
        state = _run(actions, task)
        assert success_reward(state, task, actions) == 0.0

    def test_multi_note_created_both_present(self):
        task = _task({"type": "multi_note_created", "titles": ["A", "B"]})
        actions = [
            {"action": "tap", "target": "notes_button"},
            {"action": "type", "target": "note_input", "text": "A"},
            {"action": "tap", "target": "save_note_button"},
            {"action": "type", "target": "note_input", "text": "B"},
            {"action": "tap", "target": "save_note_button"},
            {"action": "finish"},
        ]
        state = _run(actions, task)
        assert success_reward(state, task, actions) == 1.0

    def test_multi_note_created_partial_fails(self):
        task = _task({"type": "multi_note_created", "titles": ["A", "B"]})
        actions = [
            {"action": "tap", "target": "notes_button"},
            {"action": "type", "target": "note_input", "text": "A"},
            {"action": "tap", "target": "save_note_button"},
            {"action": "finish"},
        ]
        state = _run(actions, task)
        assert success_reward(state, task, actions) == 0.0

    def test_multi_goal_all_subgoals_met(self):
        task = _task({
            "type": "multi_goal",
            "subgoals": [
                {"type": "setting_enabled", "setting": "focus_mode"},
                {"type": "note_created", "title": "X"},
            ],
        }, max_steps=20)
        actions = [
            {"action": "tap", "target": "settings_button"},
            {"action": "tap", "target": "focus_mode_toggle"},
            {"action": "back"},
            {"action": "tap", "target": "notes_button"},
            {"action": "type", "target": "note_input", "text": "X"},
            {"action": "tap", "target": "save_note_button"},
            {"action": "finish"},
        ]
        state = _run(actions, task)
        assert success_reward(state, task, actions) == 1.0

    def test_multi_goal_partial_subgoals_fails(self):
        task = _task({
            "type": "multi_goal",
            "subgoals": [
                {"type": "setting_enabled", "setting": "focus_mode"},
                {"type": "note_created", "title": "X"},
            ],
        })
        state = AppState()
        state.focus_mode = True  # only one of two subgoals
        assert success_reward(state, task, []) == 0.0


# ---------------------------------------------------------------------------
# format_reward
# ---------------------------------------------------------------------------


class TestFormatReward:
    def test_all_valid_format(self):
        state = AppState()
        task = _task({"type": "note_created", "title": "X"})
        actions = [{"action": "tap", "target": "notes_button"}, {"action": "finish"}]
        assert format_reward(state, task, actions) == 1.0

    def test_all_invalid_format(self):
        state = AppState()
        task = _task({"type": "note_created", "title": "X"})
        actions = [{"action": "INVALID"}, {"action": "bad_verb"}]
        assert format_reward(state, task, actions) == 0.0

    def test_mixed_format(self):
        state = AppState()
        task = _task({"type": "note_created", "title": "X"})
        actions = [{"action": "tap", "target": "notes_button"}, {"action": "INVALID"}]
        score = format_reward(state, task, actions)
        assert score == pytest.approx(0.5)

    def test_empty_actions(self):
        state = AppState()
        task = _task({"type": "note_created", "title": "X"})
        assert format_reward(state, task, []) == 0.0


# ---------------------------------------------------------------------------
# efficiency_reward
# ---------------------------------------------------------------------------


class TestEfficiencyReward:
    def test_zero_without_success(self):
        task = _task({"type": "note_created", "title": "X"}, max_steps=10)
        state = AppState()
        assert efficiency_reward(state, task, []) == 0.0

    def test_positive_on_success(self):
        task = _task({"type": "setting_enabled", "setting": "focus_mode"}, max_steps=10)
        state = _run(FOCUS_ACTIONS, task)
        score = efficiency_reward(state, task, FOCUS_ACTIONS)
        assert score > 0.0

    def test_efficiency_decreases_with_more_steps(self):
        task = _task({"type": "setting_enabled", "setting": "focus_mode"}, max_steps=15)
        state_short = _run(FOCUS_ACTIONS, task)
        e_short = efficiency_reward(state_short, task, FOCUS_ACTIONS)

        # Pad with back/forth before enabling focus
        long_actions = [
            {"action": "tap", "target": "notes_button"},
            {"action": "back"},
            {"action": "tap", "target": "notes_button"},
            {"action": "back"},
            {"action": "tap", "target": "settings_button"},
            {"action": "tap", "target": "focus_mode_toggle"},
            {"action": "finish"},
        ]
        state_long = _run(long_actions, task)
        e_long = efficiency_reward(state_long, task, long_actions)

        assert e_short >= e_long

    def test_max_steps_gives_zero(self):
        task = _task({"type": "setting_enabled", "setting": "focus_mode"}, max_steps=3)
        # Exhaust all steps before completing goal
        actions = [{"action": "back"}] * 3
        state = _run(actions, task)
        assert efficiency_reward(state, task, actions) == 0.0


# ---------------------------------------------------------------------------
# invalid_action_penalty
# ---------------------------------------------------------------------------


class TestInvalidActionPenalty:
    def test_no_invalid_actions(self):
        task = _task({"type": "note_created", "title": "X"}, max_steps=10)
        state = AppState()
        assert invalid_action_penalty(state, task, []) == 0.0

    def test_penalty_proportional_to_count(self):
        task = _task({"type": "note_created", "title": "X"}, max_steps=10)
        state = AppState()
        state.invalid_action_count = 3
        assert invalid_action_penalty(state, task, []) == pytest.approx(0.3)

    def test_penalty_capped_at_one(self):
        task = _task({"type": "note_created", "title": "X"}, max_steps=5)
        state = AppState()
        state.invalid_action_count = 100
        assert invalid_action_penalty(state, task, []) == 1.0

    def test_penalty_increases_via_real_actions(self):
        task = _task({"type": "note_created", "title": "X"}, max_steps=10)
        state = AppState()
        for _ in range(3):
            state.invalid_action_count += 1
        pen = invalid_action_penalty(state, task, [])
        assert pen == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# safety_penalty
# ---------------------------------------------------------------------------


class TestSafetyPenalty:
    """[REQ-5] Logout action triggers safety penalty."""

    def test_no_violation(self):
        task = _task({"type": "note_created", "title": "X"})
        state = AppState()
        assert safety_penalty(state, task, []) == 0.0

    def test_logout_triggers_penalty(self):
        task = _task({
            "type": "screen_visited",
            "screen": "profile",
            "forbidden_actions": ["logout_button"],
        })
        actions = [
            {"action": "tap", "target": "profile_button"},
            {"action": "tap", "target": "logout_button"},
            {"action": "finish"},
        ]
        state = _run(actions, task)
        assert safety_penalty(state, task, actions) == 1.0

    def test_multiple_violations_still_one(self):
        """Penalty is binary — multiple violations don't stack beyond 1.0."""
        task = _task({"type": "note_created", "title": "X"})
        state = AppState()
        state.safety_violations = 5
        assert safety_penalty(state, task, []) == 1.0


# ---------------------------------------------------------------------------
# partial_progress_reward
# ---------------------------------------------------------------------------


class TestPartialProgressReward:
    def test_zero_progress(self):
        task = _task({"type": "note_created", "title": "X"})
        state = AppState()
        assert partial_progress_reward(state, task, []) == 0.0

    def test_full_progress(self):
        task = _task({"type": "note_created", "title": "Buy milk"})
        state = _run(NOTE_ACTIONS_MILK, task)
        assert partial_progress_reward(state, task, NOTE_ACTIONS_MILK) == 1.0

    def test_partial_multi_goal(self):
        task = _task({
            "type": "multi_goal",
            "subgoals": [
                {"type": "setting_enabled", "setting": "focus_mode"},
                {"type": "note_created", "title": "X"},
            ],
        })
        # Only complete focus mode subgoal
        state = AppState()
        state.focus_mode = True
        score = partial_progress_reward(state, task, FOCUS_ACTIONS)
        # 1 of 2 subgoals met
        assert score == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# compute_reward (composite)
# ---------------------------------------------------------------------------


class TestComputeReward:
    def test_keys_present(self):
        task = _task({"type": "note_created", "title": "X"})
        result = compute_reward(AppState(), task, [])
        expected_keys = {
            "success_reward",
            "format_reward",
            "efficiency_reward",
            "invalid_action_penalty",
            "safety_penalty",
            "partial_progress_reward",
            "raw_reward",
            "final_reward",
        }
        assert expected_keys.issubset(result.keys())

    def test_final_reward_clipped_to_zero_minimum(self):
        task = _task({"type": "note_created", "title": "X"}, max_steps=5)
        state = AppState()
        state.safety_violations = 1
        state.invalid_action_count = 10
        result = compute_reward(state, task, [])
        assert result["final_reward"] >= 0.0

    def test_final_reward_clipped_to_one_maximum(self):
        task = _task({"type": "note_created", "title": "X"})
        state = AppState()
        result = compute_reward(state, task, [], weights={"success": 5.0, "format": 0.0, "efficiency": 0.0, "invalid_penalty": 0.0, "safety": 0.0})
        assert result["final_reward"] <= 1.0

    def test_success_gives_high_reward(self):
        task = _task({"type": "setting_enabled", "setting": "focus_mode"}, max_steps=10)
        state = _run(FOCUS_ACTIONS, task)
        result = compute_reward(state, task, FOCUS_ACTIONS)
        assert result["success_reward"] == 1.0
        assert result["final_reward"] > 0.5

    def test_safety_violation_lowers_reward(self):
        task = _task({
            "type": "screen_visited",
            "screen": "profile",
            "forbidden_actions": ["logout_button"],
        }, max_steps=5)
        actions = [
            {"action": "tap", "target": "profile_button"},
            {"action": "tap", "target": "logout_button"},
            {"action": "finish"},
        ]
        state = _run(actions, task)
        result = compute_reward(state, task, actions)
        assert result["safety_penalty"] == 1.0
        assert result["final_reward"] <= 0.3

    def test_custom_weights_applied(self):
        task = _task({"type": "note_created", "title": "X"}, max_steps=10)
        state = AppState()
        weights = {"success": 0.0, "format": 1.0, "efficiency": 0.0, "invalid_penalty": 0.0, "safety": 0.0}
        actions = [{"action": "tap", "target": "notes_button"}, {"action": "finish"}]
        result = compute_reward(state, task, actions, weights=weights)
        # success=0 so only format contributes
        assert result["format_reward"] == 1.0
        assert result["final_reward"] == pytest.approx(result["format_reward"], rel=0.01)
