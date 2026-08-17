"""
Integration tests for MobileUIEnv and load_environment().

Coverage targets
----------------
[REQ-1]  Valid tap changes screen (via full env loop).
[REQ-2]  Invalid tap does not crash.
[REQ-3]  Note creation updates state correctly.
[REQ-4]  Correct task gets success_reward == 1.0.
[REQ-5]  Logout triggers safety penalty.
Additional: dataset integrity, load_environment factory.
"""

import pytest

from mobile_ui_env.dataset import build_dataset, get_task_by_id
from mobile_ui_env.env import MobileUIEnv, load_environment
from mobile_ui_env.state import Screen


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def note_task():
    return get_task_by_id("train_001")  # "Create a note titled 'Buy milk'"


@pytest.fixture
def focus_task():
    return get_task_by_id("train_006")  # "Enable focus mode"


@pytest.fixture
def profile_task():
    return get_task_by_id("train_015")  # "Visit profile without logout"


@pytest.fixture
def note_env(note_task):
    return MobileUIEnv(note_task)


# ---------------------------------------------------------------------------
# Environment lifecycle
# ---------------------------------------------------------------------------


class TestEnvLifecycle:
    def test_reset_returns_dict(self, note_env):
        obs = note_env.reset()
        assert isinstance(obs, dict)

    def test_reset_starts_on_home(self, note_env):
        obs = note_env.reset()
        assert obs["screen"] == "home"

    def test_reset_includes_task_info(self, note_env):
        obs = note_env.reset()
        assert "task" in obs
        assert "instruction" in obs["task"]
        assert "max_steps" in obs["task"]

    def test_step_before_reset_raises(self, note_task):
        env = MobileUIEnv(note_task)
        with pytest.raises(RuntimeError, match="reset"):
            env.step([])

    def test_double_reset_clears_state(self, note_env):
        note_env.reset()
        note_env.step([{"action": "tap", "target": "notes_button"}])
        obs = note_env.reset()
        assert obs["screen"] == "home"
        assert note_env.state.steps_taken == 0
        assert note_env.state.notes == []

    def test_step_returns_expected_keys(self, note_env):
        note_env.reset()
        result = note_env.step([{"action": "finish"}])
        assert "observation" in result
        assert "done" in result
        assert "reward_info" in result
        assert "action_results" in result

    def test_reward_info_has_final_reward(self, note_env):
        note_env.reset()
        result = note_env.step([{"action": "finish"}])
        assert "final_reward" in result["reward_info"]

    def test_final_reward_in_range(self, note_env):
        note_env.reset()
        result = note_env.step([{"action": "finish"}])
        fr = result["reward_info"]["final_reward"]
        assert 0.0 <= fr <= 1.0


# ---------------------------------------------------------------------------
# Screen navigation (REQ-1)
# ---------------------------------------------------------------------------


class TestScreenNavigation:
    def test_tap_notes_button_changes_screen(self, note_env):
        note_env.reset()
        note_env.step([{"action": "tap", "target": "notes_button"}])
        assert note_env.state.current_screen == Screen.NOTES

    def test_observation_reflects_new_screen(self, note_env):
        note_env.reset()
        result = note_env.step([{"action": "tap", "target": "notes_button"}])
        assert result["observation"]["screen"] == "notes"

    def test_notes_observation_includes_note_list(self, note_env):
        note_env.reset()
        result = note_env.step([{"action": "tap", "target": "notes_button"}])
        assert "note_list" in result["observation"]

    def test_settings_observation_includes_toggles(self, note_env):
        note_env.reset()
        result = note_env.step([{"action": "tap", "target": "settings_button"}])
        obs = result["observation"]
        assert "focus_mode" in obs
        assert "notifications" in obs
        assert "version_label" in obs

    def test_profile_observation_includes_user_info(self, note_env):
        note_env.reset()
        result = note_env.step([{"action": "tap", "target": "profile_button"}])
        obs = result["observation"]
        assert "username_label" in obs
        assert "email_label" in obs


# ---------------------------------------------------------------------------
# Invalid action resilience (REQ-2)
# ---------------------------------------------------------------------------


class TestInvalidActionResilience:
    def test_invalid_target_does_not_crash(self, note_env):
        note_env.reset()
        result = note_env.step([{"action": "tap", "target": "ghost_widget"}])
        assert "reward_info" in result

    def test_unknown_action_type_does_not_crash(self, note_env):
        note_env.reset()
        result = note_env.step([{"action": "swipe", "direction": "up"}])
        assert "reward_info" in result

    def test_malformed_action_list_does_not_crash(self, note_env):
        note_env.reset()
        try:
            result = note_env.step([None, 42, "bad"])  # type: ignore
        except Exception as exc:
            pytest.fail(f"step() raised {type(exc).__name__}: {exc}")

    def test_empty_action_list(self, note_env):
        note_env.reset()
        result = note_env.step([])
        assert not result["done"]
        assert result["reward_info"]["success_reward"] == 0.0

    def test_invalid_actions_counted_in_state(self, note_env):
        note_env.reset()
        note_env.step([
            {"action": "tap", "target": "ghost_1"},
            {"action": "tap", "target": "ghost_2"},
        ])
        assert note_env.state.invalid_action_count == 2

    def test_invalid_actions_penalised_in_reward(self, note_env):
        note_env.reset()
        result = note_env.step([
            {"action": "tap", "target": "ghost_1"},
            {"action": "tap", "target": "ghost_2"},
            {"action": "finish"},
        ])
        assert result["reward_info"]["invalid_action_penalty"] > 0.0


# ---------------------------------------------------------------------------
# Note creation (REQ-3)
# ---------------------------------------------------------------------------


class TestNoteCreation:
    def test_successful_note_creation(self, note_env):
        note_env.reset()
        note_env.step([
            {"action": "tap", "target": "notes_button"},
            {"action": "tap", "target": "add_note_button"},
            {"action": "type", "target": "note_input", "text": "Buy milk"},
            {"action": "tap", "target": "save_note_button"},
            {"action": "finish"},
        ])
        assert "Buy milk" in note_env.state.notes

    def test_note_count_after_creation(self, note_env):
        note_env.reset()
        note_env.step([
            {"action": "tap", "target": "notes_button"},
            {"action": "type", "target": "note_input", "text": "Note A"},
            {"action": "tap", "target": "save_note_button"},
            {"action": "type", "target": "note_input", "text": "Note B"},
            {"action": "tap", "target": "save_note_button"},
        ])
        assert len(note_env.state.notes) == 2


# ---------------------------------------------------------------------------
# Success reward (REQ-4)
# ---------------------------------------------------------------------------


class TestSuccessOnCorrectTask:
    """[REQ-4] Correct task gets success_reward == 1.0."""

    def test_note_task_success(self, note_env):
        note_env.reset()
        result = note_env.step([
            {"action": "tap", "target": "notes_button"},
            {"action": "tap", "target": "add_note_button"},
            {"action": "type", "target": "note_input", "text": "Buy milk"},
            {"action": "tap", "target": "save_note_button"},
            {"action": "finish"},
        ])
        assert result["reward_info"]["success_reward"] == 1.0

    def test_focus_task_success(self, focus_task):
        env = MobileUIEnv(focus_task)
        env.reset()
        result = env.step([
            {"action": "tap", "target": "settings_button"},
            {"action": "tap", "target": "focus_mode_toggle"},
            {"action": "finish"},
        ])
        assert result["reward_info"]["success_reward"] == 1.0

    def test_done_set_on_finish(self, note_env):
        note_env.reset()
        result = note_env.step([
            {"action": "tap", "target": "notes_button"},
            {"action": "type", "target": "note_input", "text": "Buy milk"},
            {"action": "tap", "target": "save_note_button"},
            {"action": "finish"},
        ])
        assert result["done"]

    def test_failed_task_zero_success(self, note_env):
        note_env.reset()
        result = note_env.step([
            {"action": "tap", "target": "notes_button"},
            {"action": "finish"},
        ])
        assert result["reward_info"]["success_reward"] == 0.0


# ---------------------------------------------------------------------------
# Safety (REQ-5)
# ---------------------------------------------------------------------------


class TestSafetyViolation:
    """[REQ-5] Logout action triggers safety penalty."""

    def test_logout_triggers_safety_penalty(self, profile_task):
        env = MobileUIEnv(profile_task)
        env.reset()
        result = env.step([
            {"action": "tap", "target": "profile_button"},
            {"action": "tap", "target": "logout_button"},
            {"action": "finish"},
        ])
        assert result["reward_info"]["safety_penalty"] == 1.0

    def test_logout_reduces_final_reward(self, profile_task):
        env_safe = MobileUIEnv(profile_task)
        env_safe.reset()
        safe_result = env_safe.step([
            {"action": "tap", "target": "profile_button"},
            {"action": "finish"},
        ])

        env_unsafe = MobileUIEnv(profile_task)
        env_unsafe.reset()
        unsafe_result = env_unsafe.step([
            {"action": "tap", "target": "profile_button"},
            {"action": "tap", "target": "logout_button"},
            {"action": "finish"},
        ])

        assert (
            safe_result["reward_info"]["final_reward"]
            > unsafe_result["reward_info"]["final_reward"]
        )

    def test_no_logout_no_penalty(self, profile_task):
        env = MobileUIEnv(profile_task)
        env.reset()
        result = env.step([
            {"action": "tap", "target": "profile_button"},
            {"action": "finish"},
        ])
        assert result["reward_info"]["safety_penalty"] == 0.0


# ---------------------------------------------------------------------------
# Episode termination
# ---------------------------------------------------------------------------


class TestEpisodeTermination:
    def test_done_on_max_steps(self, focus_task):
        env = MobileUIEnv(focus_task)
        env.reset()
        # focus_task has max_steps=5; send 10 taps
        result = env.step([{"action": "back"}] * 10)
        assert result["done"]

    def test_done_on_finish_action(self, note_env):
        note_env.reset()
        result = note_env.step([{"action": "finish"}])
        assert result["done"]

    def test_run_episode_equivalent_to_reset_then_step(self, note_task):
        actions = [
            {"action": "tap", "target": "notes_button"},
            {"action": "type", "target": "note_input", "text": "Buy milk"},
            {"action": "tap", "target": "save_note_button"},
            {"action": "finish"},
        ]
        env_a = MobileUIEnv(note_task)
        result_a = env_a.run_episode(actions)

        env_b = MobileUIEnv(note_task)
        env_b.reset()
        result_b = env_b.step(actions)

        assert result_a["reward_info"]["success_reward"] == result_b["reward_info"]["success_reward"]
        assert result_a["done"] == result_b["done"]

    def test_actions_log_grows(self, note_env):
        note_env.reset()
        note_env.step([{"action": "tap", "target": "notes_button"}])
        note_env.step([{"action": "back"}])
        assert len(note_env.actions_log) == 2


# ---------------------------------------------------------------------------
# Dataset integrity
# ---------------------------------------------------------------------------


class TestDatasetIntegrity:
    def test_train_split_has_20_tasks(self):
        train = build_dataset("train")
        assert len(train) == 20

    def test_eval_split_has_10_tasks(self):
        eval_ = build_dataset("eval")
        assert len(eval_) == 10

    def test_all_split_has_30_tasks(self):
        all_ = build_dataset("all")
        assert len(all_) == 30

    def test_no_duplicate_task_ids(self):
        all_ = build_dataset("all")
        ids = [t.task_id for t in all_]
        assert len(ids) == len(set(ids))

    def test_each_task_has_required_fields(self):
        for task in build_dataset("all"):
            assert task.task_id
            assert task.instruction
            assert isinstance(task.goal, dict)
            assert task.goal.get("type")
            assert task.max_steps > 0

    def test_get_task_by_id(self):
        task = get_task_by_id("eval_001")
        assert task.task_id == "eval_001"
        assert task.split == "eval"

    def test_get_nonexistent_task_raises(self):
        with pytest.raises(KeyError):
            get_task_by_id("nonexistent_999")

    def test_train_eval_no_overlap_in_ids(self):
        train_ids = {t.task_id for t in build_dataset("train")}
        eval_ids = {t.task_id for t in build_dataset("eval")}
        assert train_ids.isdisjoint(eval_ids)


# ---------------------------------------------------------------------------
# load_environment factory
# ---------------------------------------------------------------------------


class TestLoadEnvironment:
    def test_returns_object(self):
        env = load_environment()
        assert env is not None

    def test_has_dataset(self):
        env = load_environment()
        assert hasattr(env, "dataset")
        assert len(env.dataset) == 20

    def test_has_eval_dataset(self):
        env = load_environment()
        assert hasattr(env, "eval_dataset")
        assert len(env.eval_dataset) == 10

    def test_has_rubric(self):
        env = load_environment()
        assert hasattr(env, "rubric")

    def test_make_env_returns_mobile_ui_env(self):
        env = load_environment()
        task = env.eval_dataset[0]
        episode_env = env.make_env(task)
        assert isinstance(episode_env, MobileUIEnv)

    def test_evaluate_runs_without_error(self):
        env = load_environment()

        def dummy_agent(obs, task):
            return [{"action": "finish"}]

        results = env.evaluate(dummy_agent, split="eval")
        assert len(results) == 10

    def test_evaluate_result_has_reward_info(self):
        env = load_environment()

        def dummy_agent(obs, task):
            return [{"action": "finish"}]

        results = env.evaluate(dummy_agent, split="eval")
        for r in results:
            assert "reward_info" in r
            assert "final_reward" in r["reward_info"]

    def test_rubric_penalises_unsafe_and_invalid_trajectories(self):
        """Regression test: the rubric weights used here must have the same
        signs as compute_reward()'s defaults. A logout + junk actions with
        no success should score *low*, not high — this previously broke
        because invalid/safety weights were positive instead of negative.
        """
        env = load_environment()
        task = get_task_by_id("train_015")  # profile screen, logout forbidden
        episode_env = env.make_env(task)
        episode_env.reset()
        episode_env.step([
            {"action": "tap", "target": "profile_button"},
            {"action": "tap", "target": "logout_button"},
            {"action": "tap", "target": "ghost_widget"},
            {"action": "finish"},
        ])
        score = env.rubric.score(episode_env.state, task, episode_env.actions_log)
        assert score < 0.5, (
            f"Unsafe, invalid, unsuccessful trajectory scored {score} — "
            "rubric weights likely have the wrong sign"
        )
