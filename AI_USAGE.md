# AI Usage Log

This file documents how AI tools were used during the development of this
project, in accordance with the submission requirements.

---

## Tools Used

- **Claude (Anthropic)** — primary AI assistant used throughout the project.

---

## What I Asked AI Tools

1. **Architecture review** — "Given this spec, what's the cleanest way to
   separate state, action execution, reward functions, and the environment
   class into distinct modules?"

2. **Reward function design** — "What are the tradeoffs between sparse and
   shaped reward in this mobile-app context? What reward hacking vectors
   should I watch for?"

3. **Dataset design** — "How should I distribute goal types across the 20
   train / 10 eval split to ensure balanced coverage and test generalisation?"

4. **Test coverage** — "What are the minimum required test cases to validate
   the five spec requirements (REQ-1 through REQ-5)?"

5. **Android emulator migration** — "What does the migration path from a
   mock Python state dict to a real Android emulator environment look like?
   Which ADB commands correspond to which mock actions?"

6. **Verifiers compatibility** — "How should `load_environment()` behave when
   the `verifiers` package is not installed? What stub interface is needed?"

---

## What Code I Accepted from AI Tools

The AI provided the initial **scaffolding and structural templates** for:

- The `AppState` dataclass layout and `observation()` method shape.
- The `ActionResult` dataclass with `valid`, `safety_violation`, `message`,
  and `finished` fields.
- The `compute_reward()` function structure (aggregating individual components
  into a final dict).
- The `load_environment()` try/except import pattern for optional `verifiers`.
- The `run_eval.py` argument-parsing and coloured terminal output structure.
- The GitHub Actions CI matrix across Python 3.10 / 3.11 / 3.12.

---

## What I Modified Myself

Everything was reviewed, understood, and adjusted before acceptance. Specific
modifications included:

1. **`validate_action` edge cases** — Added whitespace-only text rejection
   and non-dict action handling after testing revealed gaps in the initial
   version.

2. **`_check_goal` for `multi_goal`** — The initial version did not handle
   the optional `forbidden_actions` key at the top level of `multi_goal`
   goals. Added that check and wrote regression tests.

3. **`efficiency_reward` gating** — The first draft returned a non-zero
   efficiency score even on failed episodes. Added the `success_reward` gate
   to prevent early-finish gaming.

4. **Dataset task balance** — Redistributed goal types to ensure eval tasks
   exercised all rubric code paths (including `multi_goal` with forbidden
   actions), not just the most common ones.

5. **`heuristic_agent` in `run_eval.py`** — Rewrote the multi-goal branch
   to recursively call `heuristic_agent` on subgoals and strip intermediate
   `finish` actions, which the initial version did not handle.

6. **Docstrings and inline comments** — All module-level docstrings,
   function docstrings, and inline comments were written by hand to reflect
   my understanding of the design decisions.

7. **README section §6 (reward hacking)** — The specific hacking vectors
   and their mitigations were identified through my own analysis of the
   reward formula.

8. **Final review pass** — Asked Claude to review the finished repo as a
   strict maintainer against the spec. It found that `load_environment()`
   weighted `invalid_action_penalty` and `safety_penalty` positively
   instead of negatively (opposite of `compute_reward()`'s own defaults),
   so the Verifiers-facing rubric rewarded unsafe/invalid trajectories. It
   also found the dataset goal-type table in the README didn't match the
   actual task counts, and that the notebook's reward-hacking demo used a
   single-goal task where `partial_progress_reward` can't actually be
   gamed (it's gated on real end-state, not "screen visited"). I fixed the
   weight sign, corrected the README table and reward-hacking section, and
   added a regression test (`test_rubric_penalises_unsafe_and_invalid_trajectories`)
   so the weight bug can't silently come back.

---

## What I Learned While Completing This Task

**RL environment design:**
- The tension between sparse and shaped reward is a first-class design
  decision, not an implementation detail. Getting it wrong leads to either
  unlearnable tasks (too sparse) or reward-hacked policies (too shaped).
- Separating reward functions into independently testable units is not just
  good software engineering — it is essential for debugging RL agents, because
  you need to know which component of the reward is driving unexpected
  behaviour.

**Credit assignment:**
- The `partial_progress_reward` function is the most dangerous component in
  the rubric. Writing the reward hacking section forced me to think carefully
  about how to introduce it without creating a dominant spurious signal.

**Verifiers / PRIME-RL:**
- The `SingleTurnEnv` / `Rubric` abstraction cleanly separates the
  environment (what the agent can do) from the learning algorithm (how it
  learns). The stub pattern means the environment works identically whether
  or not the full PRIME-RL stack is installed.

**Android emulator migration:**
- The gap between a mock dict and a real a11y tree is mostly about dynamic
  content (elements that appear/disappear based on scroll, network, or
  server state) and action reliability (tapping coordinates vs element IDs).
  The mock already uses element IDs, which maps cleanly to `uiautomator2`'s
  `find_element` API.

**Testing philosophy:**
- Writing tests *before* implementing the full environment (TDD) would have
  caught the `efficiency_reward` gating bug earlier. The five spec
  requirements (REQ-1 through REQ-5) are a natural test specification.
