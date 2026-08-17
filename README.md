# Mobile UI Agent — RL Environment

A lightweight, self-contained reinforcement-learning environment that simulates
a four-screen mobile application. An AI agent completes natural-language tasks
by producing structured JSON action sequences and receives shaped rewards based
on task success, action validity, efficiency, and safety.

Built as a take-home assignment for Prime Intellect, designed to be compatible
with the [Verifiers](https://github.com/willieneis/verifiers) / PRIME-RL
framework.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Project Structure](#project-structure)
3. [Environment Design Q&A](#environment-design-qa)
   - [State Space](#1-what-is-the-state-space)
   - [Action Space](#2-what-is-the-action-space)
   - [Episode Termination](#3-what-is-the-episode-termination-condition)
   - [Sparse Rewards](#4-which-rewards-are-sparse)
   - [Dense / Shaped Rewards](#5-which-rewards-are-dense-or-shaped)
   - [Reward Hacking](#6-how-can-reward-hacking-happen)
   - [Scaling to Android Emulator](#7-scaling-to-a-real-android-emulator)
   - [Prime Intellect / Verifiers Integration](#8-prime-intellect--verifiers--prime-rl)
   - [Tests Written](#9-what-tests-were-written)
   - [Tradeoffs](#10-tradeoffs-made-due-to-limited-scope)
4. [App Screens & Elements](#app-screens--elements)
5. [Reward Formula](#reward-formula)
6. [Dataset](#dataset)
7. [Running Locally](#running-locally)
8. [Running Tests](#running-tests)
9. [Running Evaluation](#running-evaluation)
10. [Docker](#docker)
11. [RL Fundamentals Notes](#rl-fundamentals-notes)

---

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url>
cd mobile_ui_env
pip install -e ".[dev]"

# 2. Run tests
pytest

# 3. Run heuristic eval
python run_eval.py --verbose

# 4. Use the environment in Python
python - <<'EOF'
from mobile_ui_env import MobileUIEnv, build_dataset

task = build_dataset("eval")[0]
env = MobileUIEnv(task)
obs = env.reset()
print("Observation:", obs)

result = env.step([
    {"action": "tap",    "target": "notes_button"},
    {"action": "type",   "target": "note_input",  "text": "Send invoice"},
    {"action": "tap",    "target": "save_note_button"},
    {"action": "finish"},
])
print("Reward:", result["reward_info"]["final_reward"])
EOF
```

---

## Project Structure

```
mobile_ui_env/
├── mobile_ui_env/          # Core Python package
│   ├── __init__.py         # Public API surface
│   ├── state.py            # AppState, Screen enum, SCREEN_ELEMENTS
│   ├── actions.py          # validate_action(), execute_action(), ActionResult
│   ├── dataset.py          # Task dataclass + 30-task catalogue
│   ├── rubric.py           # All reward functions + compute_reward()
│   └── env.py              # MobileUIEnv class + load_environment() factory
│
├── tests/
│   ├── test_actions.py     # Action validation and execution unit tests
│   ├── test_rewards.py     # Reward function unit tests
│   └── test_env.py         # Full environment + dataset integration tests
│
├── notebooks/
│   └── exploration.ipynb   # Interactive exploration notebook
│
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions: lint → test (3.10/3.11/3.12) → eval smoke
│
├── run_eval.py             # Evaluation runner (heuristic / dummy / LLM agents)
├── pyproject.toml          # Build config, dev/llm/notebook extras
├── Dockerfile              # Multi-stage Docker image
├── README.md               # This file
└── AI_USAGE.md             # AI tool usage log
```

---

## Environment Design Q&A

### 1. What is the state space?

The state space is the Cartesian product of all app variables:

| Variable | Type | Values |
|---|---|---|
| `current_screen` | enum | `home`, `notes`, `settings`, `profile` |
| `notes` | `list[str]` | Any ordered list of saved note titles |
| `focus_mode` | `bool` | `True` / `False` |
| `notifications` | `bool` | `True` / `False` |
| `note_input_buffer` | `str` | Any string (what's typed but not yet saved) |

Static, read-only app data (`username`, `email`, `app_version`) are also visible
in the observation but do not change — they are not part of the *mutable* state.

Episode bookkeeping (`steps_taken`, `invalid_action_count`, `safety_violations`,
`done`) is tracked on `AppState` but exposed only through the rubric, not the
agent's observation, to avoid cheating.

**Observation format** — what the agent actually receives each turn:

```json
{
  "screen": "notes",
  "elements": ["add_note_button", "note_input", "save_note_button", "note_list"],
  "note_list": ["Buy milk"],
  "note_input_buffer": "",
  "task": {
    "instruction": "Create a note titled 'Send invoice'",
    "max_steps": 8,
    "steps_taken": 2
  }
}
```

This mirrors a flattened Android **accessibility tree** (a11y XML) snapshot —
the same kind of observation you would extract from a real emulator.

---

### 2. What is the action space?

The agent produces a **list** of JSON action dicts in a single turn (single-turn,
not multi-turn). Four action types are supported:

| Type | Required keys | Example |
|---|---|---|
| `tap` | `target` | `{"action": "tap", "target": "notes_button"}` |
| `type` | `target`, `text` | `{"action": "type", "target": "note_input", "text": "Buy milk"}` |
| `back` | — | `{"action": "back"}` |
| `finish` | — | `{"action": "finish"}` |

**Validity rules** — an action is *invalid* (but never crashes) when:
- The `action` key holds an unknown verb.
- `tap` or `type` is missing a `target`.
- The `target` is not in `SCREEN_ELEMENTS[current_screen]`.
- `type` targets a non-typeable element.
- `type` provides an empty or whitespace-only `text`.

Invalid actions are counted and penalised through the reward, not rejected with
an exception, so the environment is always safe to call.

---

### 3. What is the episode termination condition?

An episode ends when **any** of the following is true:

| Condition | Notes |
|---|---|
| Agent executes `{"action": "finish"}` | Explicit opt-out |
| `steps_taken >= task.max_steps` | Budget exhausted (auto-done) |

There is no "success" termination — the agent must call `finish` itself. This
means an agent that completes the goal but never calls `finish` will not be
marked done until the budget runs out, reducing its efficiency score.

---

### 4. Which rewards are sparse?

| Reward | Why sparse |
|---|---|
| `success_reward` | 1.0 only when the entire goal is satisfied; 0 otherwise. This is the primary training signal. It is deliberately binary to give an unambiguous target and force exploration of complete solution paths. |
| `safety_penalty` | 1.0 as soon as *any* unsafe action occurs; 0 otherwise. A single logout suffices to trigger the full penalty regardless of how many times it happens. |

**Why sparse reward alone is hard for RL agents:** sparse reward provides zero
gradient signal on most trajectories. An agent that randomly explores a 4-screen
app with an 8-step budget has a tiny probability of accidentally completing any
non-trivial goal. Without a dense signal to climb, the agent's policy gradient
is almost always zero — the classic *credit assignment* problem. PPO/GRPO need
thousands of successful episodes to learn from scratch with purely sparse reward.

---

### 5. Which rewards are dense or shaped?

| Reward | Type | Notes |
|---|---|---|
| `format_reward` | Dense | Fraction of actions with valid structure. Always gives a gradient. |
| `efficiency_reward` | Dense (gated) | Positive only on success; decays linearly with steps used. Prevents gaming. |
| `invalid_action_penalty` | Dense | Proportional to `invalid_count / max_steps`. Penalises junk actions continuously. |
| `partial_progress_reward` | Shaped | Fraction of sub-goals completed. Provides gradient even on failed episodes. **Use with caution — see §6.** |

`partial_progress_reward` is computed but **not included in the default final
reward formula** — it is available as a supplemental signal that you can inject
selectively (e.g., only during early training, decayed as the agent matures).

---

### 6. How can reward hacking happen in this environment?

Reward hacking occurs when an agent discovers a strategy that maximises the
reward function without actually satisfying the intended goal.

**Vectors that look plausible but are already closed by the current design:**

`partial_progress_reward` only credits a subgoal once `_check_goal` says it is
actually satisfied — for `setting_enabled`/`setting_disabled` that means the
setting's *final* value, and for `note_created` it means the title is
actually in `state.notes`. So neither "visit every screen and finish" nor
"toggle a setting on and off" earns any partial credit: both were tested
directly (see the notebook's reward-hacking demo) and score `0.0`. These are
worth knowing about as *design principles* (check end-state, not "was this
screen touched"), not as live exploits — the naive versions don't work
against this rubric as implemented.

**Real vector found during review (fixed):** `load_environment()` built its
Verifiers-facing rubric with `invalid_action_penalty` and `safety_penalty`
weighted *positively* (`+0.2`, `+0.3`) instead of negatively, the opposite of
`compute_reward()`'s own defaults. In practice this meant a policy trained
through `load_environment()` would be rewarded for logging out and for
submitting garbage actions — the exact behaviour the rubric is supposed to
punish. Fixed in `env.py` so both reward paths share the same signed weights;
`tests/test_env.py::test_rubric_penalises_unsafe_and_invalid_trajectories`
guards against this regressing.

**Vectors that remain genuinely open:**

| Attack | Vector | Mitigation |
|---|---|---|
| **Format farming** | Produce many structurally valid but semantically useless actions. | `format_reward` weight is 0.1 vs `success_reward` weight 1.0, so it cannot dominate — but it is real free reward on a failed episode. |
| **`back` spam to dodge the invalid-action penalty** | `back` is never invalid, so an agent that is unsure what to do can spam it instead of attempting (and risking) real actions. | Doesn't increase reward on its own (no format/efficiency bonus for `back` specifically), but it would suppress `invalid_action_penalty` as a training signal without the agent learning anything useful. Worth watching for in training logs. |

**General defence:** monitor trajectory diversity during training. If the agent
converges to a short repeated pattern (e.g., always tapping the same two buttons),
it has likely found a hacking strategy.

---

### 7. Scaling to a real Android emulator

The mock environment abstracts away the real mobile stack. Here is the migration
path to a live Android emulator:

| Mock component | Real equivalent |
|---|---|
| `AppState.observation()` | Accessibility tree dump via `adb shell uiautomator dump` → parse XML → flatten to key-value dict. Or pixel screenshot + a vision model to identify elements. |
| `SCREEN_ELEMENTS` | Dynamic UI hierarchy from `ViewHierarchy` or `AccessibilityNodeInfo` — elements change based on app state, scrolling, dynamic content. |
| `execute_action()` | `adb shell input tap <x> <y>` for tap; `adb shell input text <text>` for type; or use the Android Debug Bridge Python SDK / `uiautomator2`. |
| `AppState` | Derived from the accessibility tree snapshot after each action, not stored in a Python dict. |
| `Screen` enum | Detected from the current `Activity` or `Fragment` via `adb shell dumpsys activity` or the accessibility tree's root node. |
| `safety_penalty` | Extend to detect app crash (`adb shell am crash`), data-destructive actions (delete account, clear data), or network calls to sensitive endpoints. |

**Real-world observation stack (recommended):**
1. **Accessibility tree** (XML): structured, fast, low token cost.
2. **Screenshot** (PNG → base64): for vision-language models.
3. **UI hierarchy text**: flattened text representation of the tree.

**Action executor architecture:**
```
Agent policy → JSON actions → Action Executor → ADB / uiautomator2 → Emulator
                                     ↑
                              Retry on stale element / recheck tree
```

**Emulator state management:** use `adb shell pm clear <package>` or snapshot
restore between episodes to guarantee a clean starting state, equivalent to
`env.reset()` in the mock.

---

### 8. Prime Intellect / Verifiers / PRIME-RL

`load_environment()` is designed for drop-in compatibility:

```python
# With verifiers installed:
import verifiers as vf
from mobile_ui_env import load_environment

env = load_environment()
# env is a vf.SingleTurnEnv with dataset, eval_dataset, and rubric attached.
# Plug directly into GRPO / PPO training loops.
```

The rubric functions all share the `f(state, task, actions) -> float` signature
required by `vf.Rubric`.  When `verifiers` is not installed, built-in stubs with
identical interfaces are used automatically — no conditional imports needed in
your training code.

**PRIME-RL compatibility checklist:**
- [x] `load_environment()` factory function
- [x] `vf.Rubric`-compatible reward functions (separate, testable, composable)
- [x] Train / eval dataset split
- [x] `SingleTurnEnv`-compatible `.evaluate(agent_fn)` method
- [x] JSON-serialisable actions (LLM output format)
- [ ] Multi-turn rollouts (single-turn for now; see §10)

---

### 9. What tests were written?

**`tests/test_actions.py`** — 30+ unit tests for `validate_action` and `execute_action`:
- Valid tap changes screen ✓
- Invalid tap does not crash ✓
- Invalid action increments counter ✓
- Creating a note updates `state.notes` ✓
- Saving empty note is invalid ✓
- Toggle focus mode and notifications ✓
- Logout triggers `safety_violation` flag ✓
- Batch execution stops at `finish` / `max_steps` ✓

**`tests/test_rewards.py`** — 25+ unit tests for every reward function:
- `success_reward` for all goal types ✓
- `format_reward` on valid/mixed/empty action lists ✓
- `efficiency_reward` gated on success, decreasing with steps ✓
- `invalid_action_penalty` proportional and capped ✓
- `safety_penalty` binary, triggered by logout ✓
- `compute_reward` composite output, clipping, custom weights ✓

**`tests/test_env.py`** — 40+ integration tests:
- `reset()` / `step()` / `run_episode()` lifecycle ✓
- Correct observation content per screen ✓
- Invalid actions in `step()` do not crash ✓
- `done` flag set correctly ✓
- Dataset: 20 train + 10 eval, no duplicates, required fields ✓
- `load_environment()` factory produces correct object ✓
- `evaluate()` runs without error ✓

**Run all tests:**
```bash
pytest --cov=mobile_ui_env --cov-report=term-missing
```

---

### 10. Tradeoffs made due to limited scope

| Tradeoff | What was done | What would change at scale |
|---|---|---|
| **Single-turn** | Agent produces all actions in one pass. Simpler to implement and sufficient for GRPO-style training. | Multi-turn: agent observes the screen *after each action* and decides the next one. Better for long-horizon tasks and partial observability. |
| **Mock state** | `AppState` is a Python dataclass. No real app. | Replace with accessibility tree snapshots from a live emulator. |
| **Static app data** | `username`, `email`, `app_version` never change. | In a real app these are dynamic; the agent must read them from the current observation. |
| **Goal in task dict** | The heuristic agent peeks at `task.goal`. A real RL agent only sees the instruction string. | Agent must infer the goal purely from the natural-language instruction — harder, more realistic. |
| **No partial observability** | Full state is visible in every observation. | A real emulator has dynamic loading, scroll state, pop-ups — the agent cannot assume all elements are always visible. |
| **Four screens only** | Deep linking, modals, and dynamic content are excluded. | A production environment would have tens of screens, conditional navigation, and server-driven UI. |
| **Binary safety** | `logout_button` is the only unsafe action. | A real safety layer would include network calls, permission requests, data deletion, and adversarial prompts. |
| **No image observations** | Text/JSON observation only. | Production: screenshot + accessibility tree + text tree (all three) for multimodal models. |

---

## App Screens & Elements

```
HOME
 ├── notes_button      → navigates to NOTES
 ├── settings_button   → navigates to SETTINGS
 └── profile_button    → navigates to PROFILE

NOTES
 ├── add_note_button   → clears note_input_buffer
 ├── note_input        → accepts type action
 ├── save_note_button  → appends note_input_buffer to notes
 └── note_list         → read-only display of saved notes

SETTINGS
 ├── focus_mode_toggle    → toggles focus_mode bool
 ├── notifications_toggle → toggles notifications bool
 └── version_label        → read-only, shows "1.4.2"

PROFILE
 ├── username_label   → read-only, shows "alice"
 ├── email_label      → read-only, shows "alice@example.com"
 └── logout_button    → valid tap, triggers safety_violation
```

---

## Reward Formula

```
final_reward = clip(
    1.0 × success_reward
  + 0.1 × format_reward
  + 0.2 × efficiency_reward
  − 0.1 × invalid_action_penalty
  − 0.3 × safety_penalty,
  0, 1
)
```

| Component | Type | Range | Weight |
|---|---|---|---|
| `success_reward` | Sparse | {0, 1} | +1.0 |
| `format_reward` | Dense | [0, 1] | +0.1 |
| `efficiency_reward` | Dense, gated | [0, 1] | +0.2 |
| `invalid_action_penalty` | Dense | [0, 1] | −0.1 |
| `safety_penalty` | Sparse | {0, 1} | −0.3 |
| `partial_progress_reward` | Shaped | [0, 1] | optional |

---

## Dataset

30 tasks total across 8 goal types:

| Goal type | Train | Eval |
|---|---|---|
| `note_created` | 6 | 1 |
| `multi_note_created` | 3 | 1 |
| `setting_enabled` | 2 | 1 |
| `setting_disabled` | 3 | 1 |
| `info_found` | 2 | 2 |
| `version_reported` | 1 | 1 |
| `screen_visited` | 1 | 1 |
| `multi_goal` | 2 | 2 |

Eval tasks use different note titles and combined goals to test generalisation
beyond training content.

---

## Running Locally

**Prerequisites:** Python ≥ 3.10

```bash
# Install in editable mode with dev tools
pip install -e ".[dev]"

# Optional: LLM agent support
pip install -e ".[dev,llm]"

# Optional: notebook support
pip install -e ".[dev,notebook]"
```

**Verify installation:**
```python
from mobile_ui_env import load_environment

env = load_environment()
print(len(env.dataset), "train tasks")
print(len(env.eval_dataset), "eval tasks")
```

---

## Running Tests

```bash
# All tests
pytest

# With coverage report
pytest --cov=mobile_ui_env --cov-report=term-missing

# Specific file
pytest tests/test_actions.py -v

# Specific test
pytest tests/test_env.py::TestSuccessOnCorrectTask::test_note_task_success -v
```

---

## Running Evaluation

```bash
# Heuristic baseline on eval split (default)
python run_eval.py

# Verbose: per-task breakdown table
python run_eval.py --verbose

# Evaluate on training split
python run_eval.py --split train

# Dummy agent (always finishes immediately — lower bound)
python run_eval.py --agent dummy

# LLM agent (requires OPENAI_API_KEY)
export OPENAI_API_KEY=sk-...
python run_eval.py --agent llm --model gpt-4o-mini --verbose

# Save results as JSON
python run_eval.py --verbose --output results.json
```

The heuristic agent is a rule-based baseline used to check that the environment and reward logic work end to end. It is not a trained RL policy.

**Expected output (heuristic baseline):**

```
Running heuristic-baseline agent on 10 eval tasks…
Finished in 0.01s

────────────────────────────────────────────────────
  Mobile UI Env — Eval Results  [heuristic-baseline]
────────────────────────────────────────────────────
  Total eval tasks   : 10
  Success rate       : 100.0%
  Average reward     : 1.0000
  Average steps      : 4.5
  Invalid action rate: 0.0000
  Safety violations  : 0
  ...
────────────────────────────────────────────────────
```

---

## Docker

```bash
# Build
docker build -t mobile-ui-env .

# Run heuristic eval
docker run --rm mobile-ui-env

# Run tests inside container
docker run --rm mobile-ui-env python -m pytest tests/ -v

# Run with LLM agent
docker run --rm \
  -e OPENAI_API_KEY=sk-... \
  mobile-ui-env \
  python run_eval.py --agent llm --model gpt-4o-mini --verbose
```

---

## RL Fundamentals Notes

**Why is sparse reward difficult?**
A sparse reward signal (e.g., `success_reward ∈ {0, 1}`) gives zero gradient on
the vast majority of trajectories. For a 4-screen app with an 8-step budget,
random exploration rarely stumbles on a complete solution. Policy gradient
methods like PPO need many positive samples to update productively; with sparse
reward, the expected return variance is high and the signal-to-noise ratio is
low. Shaped rewards (dense signals tied to sub-goal completion or correct screen
visits) act as a curriculum that bootstraps the agent's early learning.

**Train / eval split rationale:**
Separate splits prevent the RL agent from overfitting to specific note titles
or instruction phrasings seen during training. The eval split uses different
surface content (e.g., "Send invoice" vs "Buy milk") to test whether the policy
generalised the *abstract* action sequence (navigate → type → save) rather than
memorising specific token sequences.

**Key metrics beyond average reward:**
- **Success rate by goal type**: reveals which task categories the agent
  struggles with (e.g., multi-goal tasks vs simple note creation).
- **Invalid action rate**: a high rate signals the agent is generating
  hallucinated element names — a formatting or grounding failure.
- **Safety violation count**: must be zero in production; track separately
  from reward so it is never traded off.
- **Steps-to-success distribution**: a bimodal distribution (either very short
  or at max_steps) indicates the agent has learned the optimal path but
  fails entirely on out-of-distribution tasks.
