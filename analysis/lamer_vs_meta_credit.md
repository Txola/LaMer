# ALFWorld credit assignment: LaMer-form versus complete meta-trajectories

This is a standalone, code-grounded context note for discussing the ALFWorld
experiments in another chat. The central comparison is **the LaMer-form credit
estimator in experiment A** versus **the complete-meta-trajectory credit
estimator in experiment C**. It distinguishes *what score a generation receives*,
*which other generations form its baseline*, and *how its PPO token losses are
weighted*. Those are related but different design choices.

The worked examples below assume valid, non-overlong generations and ignore
response-local penalties until their own section. The numerical examples are
illustrative, not measurements from a saved rollout.

**Short version:** LaMer-form attaches an attempt-outcome score and a
downstream-return score to each generation, centers them in two different
record-occurrence groups, and **adds** the resulting advantages. The new
method computes one direct discounted utility for the entire sibling
meta-trajectory and subtracts **one selected baseline** for each generation.
Its task/state baselines count unique siblings, not repeated record visits.
Experiment C also changes PPO token weighting, which must be discussed
separately from the credit estimator.

## Scope and vocabulary

The hierarchy is **task group → sibling meta-trajectories → attempts → LLM
generation records → generated tokens**. One sibling meta-trajectory repeatedly
attempts the *same task*, with up to three play attempts. A failed attempt can
produce a reflection before the next attempt; a successful one stops early.
Each play action and each reflection is a distinct policy-generation record.
The rollout writes `environment_reward` on play records and zero on reflection
records. It also writes task/trajectory IDs, attempt and turn indices, phase,
raw anchor observation, and validity metadata. The actual rollout loop and
early-success stop are in
[`alfworld_agent_loop.py`](../agent_system/multi_turn_rollout/alfworld_agent_loop.py)
(approximately lines 550–660).

Here “episode” means one ALFWorld *attempt*, whereas “meta-trajectory” means
all attempts and intervening reflections for one sibling. A sibling is one
sampled policy rollout of a task; eight siblings are sampled per task in the
archived A and C configurations.

### Three code names that should not be conflated

| Mode | Meaning in this repository | Why it matters |
|---|---|---|
| `original_lamer` | Preserved older GiGPO path in the modern trainer. | Its invalid-output penalty is inserted into both legacy score branches **before** centering. |
| `lamer_form` | Experiment A's LaMer/GiGPO **base credit formula** through the modern meta-credit preparation path. | It keeps A and C's response-local penalty and mask semantics matched; it is not a byte-for-byte historical run. |
| `meta_trajectory` | Experiment C's one-$U_j$, one-selected-baseline estimator. | It changes both the score and baseline construction. |

The archived A configuration explicitly says `CREDIT_MODE=lamer_form`,
`LOSS_AGG_MODE=token-mean`, `STEP_ADVANTAGE_WEIGHT=1.0`,
`GIGPO_MODE=mean_norm`, and `FUTURE_AWARE_EPISODE_CREDIT=False`. C says
`CREDIT_MODE=meta_trajectory` and `LOSS_AGG_MODE=token-sum`. Both use
`STEP_GAMMA=0.9675`, `TRAJ_GAMMA=0.6`, three attempts, at most 15 play actions
per attempt, and reward/local-shaping settings documented below. These are
facts from the archived
[A parameters](../outputs/qwen3_4b_modern/lora_lamer_6cat_tokenmean_150/training_parameters.txt)
and [C parameters](../outputs/qwen3_4b_modern/lora_meta_6cat_full_150_retry1/training_parameters.txt),
not inferred from today's launcher defaults.

## Data flow in the two implementations

```text
ALFWorld play rewards (reflections have environment_reward = 0)
  ├─ rollout attaches LaMer episode_reward and backward step_return
  │    └─ A: center episode scores by task records;
  │          center step returns by (task, raw observation) records;
  │          add the two advantages
  └─ C: reconstruct complete sibling groups from rollout metadata;
       compute one direct U per sibling;
       select task or attempt-1 state baseline for each record;
       form one advantage per record

Both modern A and C: add response-local adjustments after base credit,
then send advantages and trainable-token masks to PPO.
```

The switch is in
[`alfworld_gigpo_trainer.py`](../agent_system/multi_turn_rollout/alfworld_gigpo_trainer.py)
(around lines 701–710 and 1039–1071). A's base-credit implementation is
`prepare_lamer_form_batch` and C's is `prepare_meta_trajectory_batch`, both in
[`meta_trajectory_production.py`](../agent_system/meta_trajectory_production.py).

## What each estimator compares

### LaMer-form: two score branches, two occurrence-weighted baselines

The rollout computes `episode_reward` and `step_return` before the trainer
centers anything. If the only reward in an attempt is success reward 10 at its
final action, *all* play generations in that successful attempt receive episode
score 10, including actions that came before success. A reflection receives
episode score zero. Experiment A does **not** select the optional future-aware
episode score.

The step score follows a backward recurrence over play records. Within an
attempt it uses `step_gamma`; when crossing to the next attempt it uses
`traj_gamma`. Reflection records inherit the running return at their position
without adding another discount. Therefore a failed early attempt can still
receive positive **step** credit if a later attempt succeeds. See
`_assign_lamer_credit` in
[`alfworld_agent_loop.py`](../agent_system/multi_turn_rollout/alfworld_agent_loop.py)
(approximately lines 275–347).

For each real policy-generation record $i$, LaMer-form computes two scores:

- **Episode (coarse) score $E_i$:** the reward of the record's *current play
  attempt*. Every play record in that attempt receives that attempt reward;
  reflections receive zero episode score.
- **Step (fine) score $F_i$:** a backward return from that record. It includes
  downstream play rewards, discounts between play records by
  $\gamma_{\mathrm{step}}$, and discounts when crossing an attempt boundary by
  $\gamma_{\mathrm{traj}}$. A reflection inherits the future return at its
  position.

LaMer-form centers $E_i$ over **all real record occurrences for the task** and
centers $F_i$ over **all real occurrences with the same task and raw anchor
observation**. It adds the two results:

\[
A_i^{\mathrm{LaMer}}=
\left(E_i-\overline E_{\mathrm{task,records}}\right)
+w\left(F_i-\overline F_{\mathrm{task,raw\ state,records}}\right).
\]

The current A run uses $w=1$ and mean centering without standard-deviation
division. Its task mean includes *every* real play and reflection record, not
one summary per sibling. Its fine grouping key is `(task_uid, anchor_obs)`
across **all attempts and both phases**; attempt number, reflection memory,
action history, and trajectory ID are not in that key. A singleton fine group
self-centers to zero. Repeated state visits contribute repeated votes to the
fine mean; longer trajectories also contribute more records to the coarse
mean. The grouping and addition are in
[`prepare_lamer_form_batch`](../agent_system/meta_trajectory_production.py)
(approximately lines 195–285).

A reward can influence **both** branches: a successful attempt has episode
score 10, while records with downstream access to that reward have positive
step returns. Adding the centered branches can reinforce or oppose one another.
Success early stopping also changes how many records each sibling contributes
to the means, even if sibling sampling initially had equal counts.

### New estimator: one utility, then a selected baseline

The new estimator first calculates **one directly written utility per complete
sibling meta-trajectory**:

\[
U_j=\sum_n\gamma_{\mathrm{traj}}^n
         \sum_t\gamma_{\mathrm{step}}^t r_{j,t}^{(n)}.
\]

Every generation in sibling $j$ then receives **one** base advantage,
$A_{ji}=U_j-b_{ji}$. The default task baseline is the self-inclusive mean of
the sibling utilities, with **one vote per trajectory**, regardless of its
number of records or repeated visits. An attempt-1 play record may instead use
the mean $U$ of unique siblings at the same raw observation, but only when
that state group contains at least three unique trajectories, at least two
distinct valid projected actions, and a utility range greater than
$\tau_U$. Otherwise it falls back to the task baseline. Reflections and
attempt-2/3 play always use the task baseline. **Task and state advantages are
never added.** The implementation records the selected baseline and, when it
falls back, the reason. Response-local adjustments may subsequently change the
trainable advantage, but they do not change $U_j$ or anyone else's baseline.

The state key remains `(task_uid, raw anchor observation)`, not the full prompt
or policy information state. Restricting it to attempt-1 play avoids grouping
later attempts with different reflection memories despite identical raw
ALFWorld observations. It does **not** prove that all attempt-1 records at a raw
state have identical histories. Moreover, the state eligibility gate depends
on sampled actions and realized utilities: it is a pragmatic credit-selection
heuristic, not a strictly action-independent unbiased control variate. See
[`assign_state_or_task_advantages`](../agent_system/meta_trajectory_credit.py)
(approximately lines 233–360) and the [design document](../META_CREDIT_DESIGN.md).

The task baseline is self-inclusive, **not leave-one-out**. There is no
standard-deviation division or automatic normalization of the new base
advantage. At an eligible state, each visiting sibling gets one baseline vote
even if it visited repeatedly; *every real policy generation* still contributes
trainable tokens to PPO. Baseline vote and loss occurrence are different units.

The direct $U_j$ is not generally equal to LaMer's first step return: the
backward recurrence may also discount through remaining actions of an earlier
attempt before it crosses to a later attempt.

### Comparison at a glance

| Question | LaMer-form A | Complete-meta-trajectory C |
|---|---|---|
| Base score | Current-attempt reward $E_i$ **and** downstream return $F_i$. | One direct complete-sibling utility $U_j$ for every record in sibling $j$. |
| Coarse/task baseline | Episode scores from all real task record occurrences. | Utilities from unique sibling trajectories, one vote each. |
| Fine/state baseline | Step returns of all real records at the same task + raw observation, including repeat visits and later attempts. | Unique sibling utilities at an attempt-1 play raw state, only if support, action-diversity, and utility-range gates pass. |
| Composition | Center both and add: $A^{\rm episode}+wA^{\rm step}$. | Select **one** baseline: $U_j-b^{\rm task}$ or $U_j-b^{\rm state}$. |
| Reflection base credit | Episode score 0; fine score is downstream step return, then each is centered. | Same $U_j$ as its sibling, minus task baseline. |
| Later-attempt play base credit | Its own attempt outcome plus raw-state-centered downstream return. | Same $U_j$ as its sibling, minus task baseline. |
| Repeat visit | Another baseline vote and another trained generation. | No additional state-baseline vote, but still another trained generation. |
| Singleton raw-state group | Fine component is zero after self-centering; episode component remains. | Explicit task-baseline fallback. |

## Four-outcome example with early success stopping

Consider four siblings of one task. Each attempt has one play generation;
before attempts 2 and 3 there is one reflection. An attempt rewards **10 on
success or 0 otherwise**, and the trajectory stops immediately after success.
Use $\gamma_{\mathrm{traj}}=0.6$ and $w=1$. The within-attempt discount has no
effect in this one-play-per-attempt example. Suppose all play records have the
same raw-state group S, while all reflection records have a distinct raw-state
group R. This grouping is an explicit toy assumption: real raw observations
determine the actual fine baselines.

For a single success reward 10 on the sole play generation, the new direct
utility is 10 for success in attempt 1, $0.6(10)=6$ for attempt 2, and
$0.6^2(10)=3.6$ for attempt 3. Failure gives 0. This is why early success is
more valuable under the written objective even though every successful
attempt's undiscounted episode outcome is 10.

| Sibling | Outcome | Records | Episode scores $E$ | Step returns $F$ | Direct $U$ |
|---|---|---|---|---|---:|
| A | Success in attempt 1 | P1 | 10 | 10 | 10 |
| B | Success in attempt 2 | P1, R2, P2 | 0, 0, 10 | 6, 10, 10 | 6 |
| C | Success in attempt 3 | P1, R2, P2, R3, P3 | 0, 0, 0, 0, 10 | 3.6, 6, 6, 10, 10 | 3.6 |
| D | Failure | P1, R2, P2, R3, P3 | 0, 0, 0, 0, 0 | 0, 0, 0, 0, 0 | 0 |

Here P1/P2/P3 denote play records and R2/R3 denote reflections. There are
14 real records. LaMer's episode mean is $30/14=2.143$; its fine means are
$45.6/9=5.067$ for S and $26/5=5.2$ for R. The table shows **episode
advantage, step advantage, and their final sum**:

| Outcome | Record | LaMer episode adv. | LaMer step adv. | LaMer final | New base adv. |
|---|---|---:|---:|---:|---:|
| A: success in 1 | P1 | +7.86 | +4.93 | **+12.79** | +5.10 |
| B: success in 2 | P1 | -2.14 | +0.93 | **-1.21** | +1.10 |
|  | R2 | -2.14 | +4.80 | **+2.66** | +1.10 |
|  | P2 | +7.86 | +4.93 | **+12.79** | +1.10 |
| C: success in 3 | P1 | -2.14 | -1.47 | **-3.61** | -1.30 |
|  | R2 | -2.14 | +0.80 | **-1.34** | -1.30 |
|  | P2 | -2.14 | +0.93 | **-1.21** | -1.30 |
|  | R3 | -2.14 | +4.80 | **+2.66** | -1.30 |
|  | P3 | +7.86 | +4.93 | **+12.79** | -1.30 |
| D: failure | Each play | -2.14 | -5.07 | **-7.21** | -4.90 |
|  | Each reflection | -2.14 | -5.20 | **-7.34** | -4.90 |

The new task baseline is $(10+6+3.6+0)/4=4.9$, giving A/B/C/D advantages
$(+5.1,+1.1,-1.3,-4.9)$. If S passes the state-group eligibility gates, its
baseline is also 4.9 in this particular example because **all four** siblings
visit it. Thus the new advantage is constant across the records of each
sibling here, although every actual generation still participates in PPO.

The most revealing contrast is **B's first play**: LaMer-form gives it
$-1.21$, since its zero attempt-1 episode score outweighs its positive
downstream step advantage. The new method gives it $+1.1$, since B's
complete utility of 6 exceeds the sibling mean of 4.9. C's early records are
negative under both methods here, but for different reasons.

With real multi-action attempts, LaMer assigns the same attempt-level episode
score to each play record in an attempt, while step returns vary by position.
The particular numerical baselines above would therefore change with record
counts and raw-state visits. The structural distinction remains: **LaMer adds
two occurrence-centered signals; the new estimator centers one complete-trajectory
utility using one baseline selected for each record.**

## Further examples that expose different mechanisms

### Direct nested utility versus the backward step recurrence

Suppose attempt 1 has two play actions with reward 0, followed by a reflection;
attempt 2 has two play actions with rewards 0 and 10. With
`gamma_step=0.9675` and `gamma_traj=0.6`, the new written utility is

\[
U=0.6(0.9675)(10)=5.805.
\]

The LaMer backward step returns are 10 at the successful last action, 9.675
at the preceding attempt-2 action, 9.675 at the reflection, 5.805 at the last
attempt-1 action, and approximately **5.616** at the *first* attempt-1 action:

\[
F_{\text{first action}}=(0.9675)(0.6)(0.9675)(10)=5.6163375.
\]

The extra initial factor of `gamma_step` comes from traversing the remaining
zero-reward action of attempt 1. This difference is present before either
method subtracts a baseline. The new method does **not** obtain $U$ by taking
the first LaMer step return; it calculates the nested expression directly
from recorded attempt and turn indices. This is checked in
[`direct_meta_trajectory_utility`](../agent_system/meta_trajectory_credit.py)
(approximately lines 134–171).

### State baseline can change the sign without changing the utility

Reuse the four utilities from the main example: $(10,6,3.6,0)$, whose task
baseline is $4.9$. Now suppose only A, B, and C reach some *attempt-1* raw
state S, with at least two distinct valid actions there. Its three unique
trajectory votes are $(10,6,3.6)$, so its eligible state baseline is
$19.6/3\approx6.533$. At S, the new advantages become approximately
$(+3.467,-0.533,-2.933)$ for A/B/C. D, which did not reach S, keeps its task
advantage $-4.9$ on its records. B's complete utility remains 6, but B is
slightly **below the relevant state peers** despite being above the task-wide
mean. This is the intended role of baseline selection; it is not a second
added reward branch.

If only two siblings reached S, if all projected actions were the same, or if
their utilities differed by at most $\tau_U$, the state group would be
ineligible and its records would use task credit. The archived A/C settings
specify $\tau_U=10^{-6}$; the threshold is an **absolute utility tolerance**,
not a percentage.

### Why repeated visits matter

Consider a raw-state group S with one play occurrence from sibling A, three
play occurrences from B, and one from C. Assume at least two valid projected
actions occur. Suppose their relevant step returns are 10 for A and 0 for
B/C, and their complete utilities are also $(10,0,0)$. Other records in the
task are omitted from this **state-baseline-only** calculation.

- LaMer-form averages the **five occurrences** for its fine baseline:
  $10/5=2$. Its fine component is $+8$ for A and $-2$ at each B/C occurrence.
  Its separate episode component cannot be calculated without *all* the task's
  other play and reflection records.
- The new eligible state baseline averages the **three siblings**:
  $10/3\approx3.333$. Its base advantage at S is $+6.667$ for A and $-3.333$
  at every B/C generation. B has only one utility *vote* but still has three
  trainable generations.

The numerical advantages are not supposed to be equal across estimators; the
example isolates the change in state-baseline voting. LaMer additionally adds
its task-centered episode component, whereas the new estimator does not.

## Response-local shaping is separate from base credit

The A ablation deliberately preserves C's modern response handling instead
of copying every historical `original_lamer` behavior. Both `lamer_form` and
`meta_trajectory` first prepare base credit, then apply configured
response-local invalid/overlong adjustments to that record's final advantage.
A malformed **nontruncated** response can receive an invalid adjustment; a
hard-truncated response does not receive the generic invalid penalty. In the
archived default `soft_shaping` mode, the near-cap overlong penalty reaches its
maximum at the cap, but the capped response tokens remain trainable under the
ordinary response mask. These adjustments do not propagate into the sibling
utilities or baselines. Truncated-reflection memory recovery, if applicable,
changes the next attempt's context, not this base-credit formula. See
[`prepare_meta_trajectory_batch`](../agent_system/meta_trajectory_production.py)
(approximately lines 366–609) and
[`soft_overlong_penalty`](../agent_system/meta_trajectory_credit.py).

By contrast, the separate `original_lamer` compatibility path in
[`alfworld_gigpo_trainer.py`](../agent_system/multi_turn_rollout/alfworld_gigpo_trainer.py)
(approximately lines 701–935) subtracts its invalid penalty from the legacy
episode and step *scores before group centering*. That distinction matters if
someone asks whether experiment A is an exact reproduction of the historical
training run. It is not: A preserves the LaMer-form base estimator under the
modern matched protocol and local shaping rules.

## PPO loss reduction: a different axis from credit assignment

After an advantage is assigned, PPO computes per-token clipped policy losses.
Experiment A uses `token-mean`: within an optimizer minibatch, each trainable
token has the same denominator, so a longer meta-trajectory can contribute
more token mass. Experiment C targets a token mean **inside each complete
meta-trajectory**, then a sibling mean inside each task, then a task mean:

\[
L_C=\frac{1}{|\mathcal G|}\sum_{g\in\mathcal G}
\frac{1}{K_g}\sum_{j=1}^{K_g}
\frac{1}{N_{gj}}\sum_{i,l}m_{gjil}\ell^{\rm PPO}_{gjil}.
\]

Here $N_{gj}$ counts final trainable generated tokens of sibling $j$. The
implementation attaches positive weight
$1/(|\mathcal G|K_gN_{gj})$ to each of its trainable tokens before a token-sum
reduction. These are **loss weights**, not changes to $U_j$ or its baseline.
The implementation also accounts for the actor's PPO minibatch partitioning;
one should not describe the current run as one giant optimizer update for the
entire rollout. See
[`finalize_meta_trajectory_batch`](../agent_system/meta_trajectory_production.py)
(approximately lines 611–667), the trainer's optimizer-visible scaling
(approximately lines 1102–1123), and the
[design document](../META_CREDIT_DESIGN.md).

This matters for scientific interpretation:

| Experiment | Base credit | PPO loss reduction |
|---|---|---|
| A | LaMer-form episode + step | Token mean |
| B (configured ablation) | New one-$U$ credit | Token mean |
| C | New one-$U$ credit | Equal complete-meta-trajectory mass |

A versus B is designed to isolate base credit; B versus C is designed to
isolate loss weighting. A versus C changes **both**, so a difference in their
learning curves cannot be attributed to credit assignment alone. The archived
A and C parameters also show different validation frequencies (10 versus 5)
and computational token budgets; those should be labelled when plotting or
timing the runs. They do not change the base-credit equations, but they make
unqualified runtime or checkpoint-count comparisons misleading. B has an
explicit [launcher configuration](../examples/alfworld/train_lora_meta_6cat_tokenmean_150.sh);
this note does not claim B has completed a training run.

## How to read results and what remains a hypothesis

The equations and code paths above are verified implementation facts. The
following are *interpretations to test*, not conclusions from the formulas:

- LaMer-form may give a later-successful trajectory's early decisions negative
  total advantages because their own attempt failed and the episode branch is
  negative. The four-outcome toy shows one possible case, not its frequency in
  the actual training distribution.
- The new estimator makes every record in a sibling share the same underlying
  $U_j$, but different eligible attempt-1 states can select different baselines.
  It does not guarantee that every action within a successful trajectory gets
  positive credit; the sibling may still be below its peers.
- The new state gate mitigates weak comparison groups but does not eliminate
  raw-observation aliasing. The gate is action/outcome dependent, so calling
  the selected state baseline an unbiased control variate without qualification
  would overstate the theory.
- Equal per-trajectory token mass changes the optimization emphasis under early
  stopping. It is a deliberately length-normalized surrogate, not the literal
  token-summed score-function gradient of a complete autoregressive rollout.

For a new discussion, first specify whether the question concerns **base
credit**, **local shaping/masks**, **PPO token weighting**, or their combined
learning effect. Also specify whether “LaMer” means historical
`original_lamer` or experiment A's matched `lamer_form`; otherwise apparently
contradictory answers can both be describing different code paths.

## Implementation pointers

- [LaMer rollout episode scores and step-return recurrence](../agent_system/multi_turn_rollout/alfworld_agent_loop.py)
- [LaMer-form task/state centering and addition](../agent_system/meta_trajectory_production.py)
- [Direct utility, unique-trajectory baselines, and fallback gates](../agent_system/meta_trajectory_credit.py)
- [Full protocol and loss-reduction specification](../META_CREDIT_DESIGN.md)
