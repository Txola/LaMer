# Meta-trajectory credit assignment design

Status at design freeze: checkpoints 1--4 were approved, and checkpoint 4B
restored standard outer PPO record minibatches while retaining exact global
meta-trajectory credit and the within-minibatch FSDP accumulation invariant.
Multi-step RL training was unauthorized at that checkpoint. It was subsequently
authorized and a 150-step run completed; see `docs/SOURCE_FREEZE.md` for the
source-provenance boundary and reconstruction caveat.

This document is the authoritative specification for the proposed
`meta_trajectory` credit mode. It separates frozen mathematical decisions from
open experimental choices and from implementation mechanics. An implementation
must not silently resolve anything marked **OPEN**.

## Audited code and reference points

The design was checked against these concrete paths and revisions:

- LaMer revision `b47b1b6660a1f27eb5d8318ef7e426d73a6945a2`.
- The active ALFWorld rollout path in
  `agent_system/multi_turn_rollout/alfworld_agent_loop.py`.
- The active modern GiGPO trainer in
  `agent_system/multi_turn_rollout/alfworld_gigpo_trainer.py`.
- The shared Qwen launcher in `examples/alfworld/train_alfworld_modern.sh` and
  the frozen Qwen3-4B meta launcher in
  `examples/alfworld/train_meta_trajectory_qwen3_4b.sh`.
- Pinned VERL revision `fbb4b3a8bf636f290c9c59fc346f756849e9c241`,
  especially `verl/workers/utils/losses.py`,
  `verl/trainer/ppo/core_algos.py`,
  `verl/trainer/ppo/v1/trainer_base.py`,
  `verl/workers/engine_workers.py`, and the FSDP engine.
- The local official GiGPO clone at `/home/benat/verl-agent`, revision
  `20bd331bdbc9026a5668e11362178e10ab7400c8`, especially
  `gigpo/core_gigpo.py` and `verl/trainer/ppo/ray_trainer.py`.
- The public reference is the
  [official verl-agent/GiGPO repository](https://github.com/langfengQ/verl-agent).

Verified current behavior:

1. One play or reflection generation becomes one VERL training record.
2. `task_uid`/dataset `uid` identifies a sampled task group.
3. `traj_uid = f"{task_uid}:{session_id}"` identifies one complete sibling
   meta-trajectory. `phase`, zero-based `attempt_index`, zero-based `turn_index`,
   `anchor_obs`, `environment_reward`, and validity metadata are retained on
   every real record.
4. The existing LaMer recurrence discounts by `step_gamma` between adjacent
   play records and by `traj_gamma` when the attempt index changes. Consequently,
   its first `step_return` is not generally the directly written nested return.
5. Current modern LaMer centers record-level episode scores by task, centers
   legacy step returns by `(task, anchor_obs)`, and adds the two advantages.
   Repeated records therefore affect both means.
6. Official GiGPO likewise constructs anchor-state groups from task identity and
   raw anchor observation, occurrence-centers the step rewards, and adds episode
   and step advantages. The new method is an intentional departure rather than
   a claim that the reference code is defective.
7. VERL already supports asymmetric `clip_ratio_low` and `clip_ratio_high` and a
   distributed-safe `token-sum` reduction. It does not provide a native
   complete-meta-trajectory loss reducer.
8. The FSDP engine safely accumulates memory-microbatch gradients before one
   optimizer step for one PPO minibatch. The enclosing worker performs a
   separate optimizer step for every PPO minibatch. Checkpoint 4B uses that
   same outer record-row minibatch mechanism in both modes; meta mode differs
   only by attaching globally computed credit and the derived minibatch scale
   before partitioning.

The principal estimator differences are therefore explicit:

| Property | Current modern LaMer / official GiGPO pattern | New `meta_trajectory` mode |
|---|---|---|
| Coarse score | Attempt/episode score attached to records | One direct nested utility per complete meta-trajectory |
| Coarse baseline votes | Generated record occurrences in the active modern port | One vote per sibling trajectory |
| Fine score | Legacy step return | The same complete utility $U_j$ |
| Fine group | Task plus raw anchor; occurrence-weighted | Attempt-1 play only; one vote per visiting trajectory |
| Unsupported/singleton state | Self-centering makes the state component zero | Explicit task-advantage fallback below $M_{\min}$ |
| Composition | Episode advantage plus weighted step advantage | Exactly one baseline-selected advantage |
| Invalid output | Penalty enters legacy score branches before centering | Local post-baseline adjustment |
| Loss unit | Token or generated record, depending on configured reducer | Complete meta-trajectory token mean |

## Frozen first-experiment protocol

The initial experiment uses the existing successful **Qwen3-4B LoRA** setup,
not Qwen3.5-9B. It uses three attempts, at most 15 actions per attempt, and
prompt history length 15, so reflection sees the complete failed attempt.
Matched 126-task evaluation produced cumulative success counts `62/80/88` for
3x10, `81/97/101` for 3x15, and `88/101/106` for 3x20. The 3x15 protocol keeps
substantial retry headroom at materially lower cost than 3x20.

The optimizer anchor is the successful earlier LoRA run: AdamW learning rate
`2e-5`, betas `0.9/0.999`, epsilon `1e-8`, weight decay `0.01`, constant
schedule, zero warmup, and global gradient clip `1.0`. The frozen launcher sets
`2e-5` explicitly rather than inheriting the generic `1e-6` default.

Native Qwen thinking is disabled. The existing explicit visible-reasoning and
action prompt remains in use. This is a controlled first-experiment choice,
not a general claim about native thinking in the base model.

| Sampler setting | RL rollout | Evaluation |
|---|---:|---:|
| temperature | 1.0 | 0.7 |
| top-p | 1.0 | 0.8 |
| top-k | disabled / -1 | 20 |
| min-p | 0 | 0 |
| presence penalty | 0 | 0 |

Both play and reflection use `max_response_tokens=1024`.
Truncated-reflection recovery is enabled for the proposed meta-trajectory
protocol, while incomplete play actions are never recovered.

Discounting is frozen at

\[
\gamma_{\mathrm{traj}}=0.6,
\qquad
\gamma_{\mathrm{step}}=0.9675.
\]

This preserves the former full-attempt discount: $0.9675^{14}\approx0.629671$
versus $0.95^9\approx0.630249$. With terminal reward 10, the attempt 1--3
first/last-action values are approximately
`10.000, 6.297, 6.000, 3.778, 3.600, 2.267`.

The initial meta-trajectory actor uses `clip_ratio_low=0.20`,
`clip_ratio_high=0.28`, `ppo_epochs=1`, and `entropy_coeff=0`.
Original-LaMer entropy defaults are unchanged.

## DECIDED

### One complete meta-trajectory utility

One sibling rollout is one complete meta-trajectory containing up to three play
attempts and the reflections between failed attempts. For sibling $j$, the
only Meta-RL utility is

\[
U_j = \sum_n \gamma_{\mathrm{traj}}^n
      \sum_t \gamma_{\mathrm{step}}^t r_{j,t}^{(n)}.
\]

Attempt and turn indices are zero-based. The utility is computed directly from
play-record environment rewards. Reflection records do not add reward terms.
For now, no validity or length shaping is part of $U_j$.

The implementation must not derive $U_j$ from legacy `step_return[0]`.

### One advantage per policy generation

Every real policy-generation record $i$ in meta-trajectory $j$ has one base
advantage

\[
A^{\mathrm{base}}_{ji}=U_j-b_{ji}.
\]

There is no `episode advantage + w * step advantage` composition in this mode.
Coarse versus fine credit is determined only by baseline selection.

### Task baseline

For a task group $g$ with $K_g$ unique sibling meta-trajectories,

\[
b_g^{\mathrm{task}}=\frac{1}{K_g}\sum_{k=1}^{K_g}U_{gk},
\qquad
A_{gj}^{\mathrm{task}}=U_{gj}-b_g^{\mathrm{task}}.
\]

The baseline is self-inclusive mean centering. There is no standard-deviation
division and no leave-one-out rescaling. Every sibling contributes exactly one
utility, independent of its record count or token count.

### Attempt-1 raw-state baseline

Fine grouping deliberately retains GiGPO's raw-state approximation. The key is

\[
(\text{task identity},\ \text{raw ALFWorld anchor observation}).
\]

History, actions, prompt tokens, and reflections are not part of the key.

For an attempt-1 play state $s$, let

\[
J_{g,s}=\{j:\text{trajectory }j\text{ visited }s\text{ during attempt 1}\},
\qquad M_{g,s}=|J_{g,s}|.
\]

The set contains unique trajectory identities. If one trajectory visits the
state repeatedly, its utility appears once in

\[
b_{g,s}^{\mathrm{state}}
  =\frac{1}{M_{g,s}}\sum_{k\in J_{g,s}}U_{gk}.
\]

Let $D_{g,s}$ be the set of distinct, successfully projected ALFWorld action
strings represented by the real record occurrences in the group, and define

\[
\Delta U_{g,s}=\max_{k\in J_{g,s}}U_{gk}
                -\min_{k\in J_{g,s}}U_{gk}.
\]

The raw-state group is eligible exactly when

\[
M_{g,s}\ge 3,
\qquad |D_{g,s}|\ge 2,
\qquad \Delta U_{g,s}>\tau_U,
\]

where the frozen tolerance is $\tau_U=10^{-6}$ in absolute utility units. For
an eligible group,

\[
A^{\mathrm{base}}_{gji}=U_{gj}-b_{g,s}^{\mathrm{state}}.
\]

Otherwise every record in that group receives $A_{gj}^{\mathrm{task}}$. Thus a
same-action/different-utility group cannot claim current-action contrast, while a
different-action/identical-utility group does not replace informative task credit
with a zero state-relative signal. The minimum support of three and the
absolute tolerance are frozen.

Every repeated generation remains a real policy decision and contributes its
trainable generated tokens to the loss. Deduplication applies only when
constructing the baseline. Action diversity is evaluated across all real
occurrences: repeated visits cannot add utility votes, but any successfully
projected action actually taken is represented in $D_{g,s}$. An occurrence with
`is_action_valid=false` or no projected ALFWorld action contributes no member to
$D_{g,s}$; its diagnostic fallback string is not an action category.

This eligibility gate intentionally depends on sampled actions and realized
utilities. It is therefore a batch-dependent credit-selection heuristic, not a
strict action-independent control variate. The selected baseline remains a
self-inclusive mean of unique trajectory utilities; this caveat must not be
hidden in claims about estimator unbiasedness.

### Reflections and later attempts

All reflection records and all attempt-2/attempt-3 play records use the task
baseline. Raw-state grouping is never used for them. This prevents raw
observation equality from conflating different cross-attempt reflection-memory
states.

### Prompt conditioning

Prompt tokens remain masked from the policy loss. Generated response tokens are
trained under log probabilities conditioned on the complete prompt. Reflection
generations are themselves policy records and can receive Meta-RL advantage.

### Local invalid-output shaping

For a malformed or invalid response that is not hard-truncated,

\[
A^{\mathrm{final}}_{ji}
 = A^{\mathrm{base}}_{ji}-\lambda_{\mathrm{invalid}}
   +p_{\mathrm{overlong}}(L_{ji}),
\]

with initial

\[
\lambda_{\mathrm{invalid}}=0.5.
\]

The invalid penalty is local to that generated record. It never changes
$U_j$, task baselines, or state baselines.

### Hard truncation

A response is hard-truncated/censored exactly when
`generated_response_length == MAX_RESPONSE_LENGTH`. Hard truncation remains
useful metadata, but under the default `overlong_mode=soft_shaping` it does not
remove response tokens from PPO. The ordinary response mask remains trainable.
The record is exempt from the generic invalid penalty and receives the
soft-overlong penalty at the cap:

\[
p(L_{\max})=-\lambda_{\mathrm{overlong,max}}.
\]

The environment transition still occurred. Its zero/failed action, consumed
turn, subsequent rewards, and eventual task outcome therefore remain in the
direct calculation of $U_j$.

For a reflection generation only, if the raw hard-truncated response contains a
valid opening `<remark>` tag and no closing `</remark>` tag, the nonempty text
after the selected opening tag through the truncation boundary is recovered,
trimmed, passed through the existing 2,000-character reflection-memory cap, and
supplied as reflection memory to the next attempt. The extraction endpoint is
the generation truncation boundary; the storage cap is the same normalization
already applied to a valid closed reflection. The raw response is not rewritten
and the absent closing tag is not synthesized. Ordinary parse validity therefore
remains false. The record remains hard-truncated and trainable under the ordinary
response mask, is exempt from the generic invalid penalty, and receives the
maximum soft-overlong penalty.

The reflection record carries
`reflection_recovered_from_truncation = true` exactly when recovered text is
actually installed as next-attempt memory. A hard-truncated response that
already has a valid closed `<remark>...</remark>` block continues through the
ordinary parser and has this flag set to false. An opening tag with no nonempty
following text cannot install memory and also has the flag false. No analogous
recovery is performed for incomplete play actions.

### Soft-overlong shaping

For a generated response of phase
$q\in\{\text{play},\text{reflection}\}$, including a hard-truncated response at
the cap, let $C_q$ be the phase-specific cache width and $L_{\max}$ the response
limit. The linear form is

\[
p_q(L)=-\lambda_{\mathrm{overlong,max}}
\operatorname{clip}\!\left(
 \frac{L-(L_{\max}-C_q)}{C_q},0,1
\right),
\]

with initial maximum magnitude

\[
\lambda_{\mathrm{overlong,max}}=0.5.
\]

For the initial experiment, $L_{\max}=1024$ and both cache widths are frozen to
$C_{\mathrm{play}}=C_{\mathrm{reflection}}=150$, so shaping begins at token
874. The default is `overlong_mode=soft_shaping`, and the penalty is continuous
through the cap. A
malformed nontruncated response may receive both invalid and soft-overlong local
terms. A hard-truncated response receives the appropriate soft-overlong term but
not the generic invalid term. `overlong_mode=filter`, which removes capped
responses from PPO, is retained only as an explicit future ablation.

### PPO clipping

The frozen settings are

\[
\epsilon_{\mathrm{low}}=0.20,
\qquad
\epsilon_{\mathrm{high}}=0.28.
\]

This is an optimization setting, not part of $U_j$ or the baselines. Required
diagnostics must distinguish lower-bound and upper-bound activity.

### Complete-meta-trajectory loss

Let $m_{gjil}\in\{0,1\}$ be the final trainable-token mask after prompt
masking, synthetic-padding exclusion, the selected overlong mode, and any
approved rollout-rejection masking. Under the default soft-shaping mode, hard
truncation does not change this mask. Define

\[
N_{gj}=\sum_{i,l}m_{gjil}.
\]

For $N_{gj}>0$,

\[
L_{gj}=\frac{1}{N_{gj}}
       \sum_i\sum_l m_{gjil}
       \ell^{\mathrm{PPO}}_{gjil}(A^{\mathrm{final}}_{gji}),
\]

and

\[
L=\frac{1}{|\mathcal G|}\sum_{g\in\mathcal G}
  \frac{1}{K_g}\sum_{j=1}^{K_g}L_{gj}.
\]

Record boundaries have no statistical role. Splitting or merging a response
record without changing its token losses must not change $L_{gj}$.

This is intentionally a trajectory-length-normalized surrogate. The default is
not a literal token-summed score-function estimator and not a Dr.GRPO-style
fixed denominator. The latter remains a possible ablation.

### Preservation of original behavior

The existing LaMer estimator must remain selectable as
`credit_mode=original_lamer`. The new estimator is selected as
`credit_mode=meta_trajectory`. Original-mode numerical behavior must not change.

## OPEN

The following are not frozen:

1. Dynamic filtering of task groups with zero utility variation.
2. The policy for a trajectory with $N_{gj}=0$. The defined loss is undefined
   in this case. Silently dropping the trajectory changes equal outer weighting;
   dividing by an epsilon creates a zero-contribution trajectory; resampling
   changes the data distribution. Until explicitly decided, implementation and
   tests must fail closed if this occurs.

The first-experiment horizon, discounts, thinking mode, entropy coefficient,
length caches, and rollout-importance denominator convention are now frozen.

## REJECTED ALTERNATIVES

For the default `meta_trajectory` mode, the following are rejected:

- Using legacy `step_return[0]` as $U_j$.
- Optimizing a separate attempt-local return as the fine utility.
- Adding task and state advantages that each contain $U_j$.
- Standard-deviation normalization.
- Leave-one-out task or state baselines.
- Full-prompt/exact-context grouping, because empirical support was too sparse.
- Raw-state baselines for reflections or later attempts.
- Counting repeated visits as repeated baseline votes.
- Treating support count alone as sufficient state-baseline eligibility.
- Putting invalid or soft-length penalties inside $U_j$.
- Treating hard truncation as ordinary invalid output.
- Filtering hard-truncated responses from PPO as the proposed default.
- Recovering an incomplete play action speculatively.
- Averaging each generation record independently.
- Global token mean as the default.
- Fixed-denominator token sums as the default.
- Replacing or mutating `original_lamer` behavior.

## MATHEMATICAL SPECIFICATION

### Record and trajectory reconstruction

For every non-padding record retain:

- task identity (g): checked consistently between queued `uid` and
  `extra_fields.task_uid`;
- trajectory identity (j): `extra_fields.traj_uid`;
- `phase` in `{play, reflect}`;
- zero-based `attempt_index` and `turn_index`;
- raw `anchor_obs`;
- scalar `environment_reward`;
- `is_action_valid`;
- projected ALFWorld action and projection-validity flag for play records
  (currently propagated as `extra_fields.parsed_action` and
  `extra_fields.is_action_valid` from the environment manager);
- hard-truncation flag and generated-response length;
- `reflection_recovered_from_truncation`;
- `reflection_memory_override` (null unless recovery installed memory);
- the response mask and any rollout-correction mask/weight.

Group by `(task_uid, traj_uid)`, not by record adjacency. Sorting is unnecessary
for the direct utility, although `(attempt_index, phase, turn_index)` should be
validated for structural consistency and used in diagnostics.

### Direct utility

For each grouped trajectory,

\[
U_{gj}=\sum_{i:\,\mathrm{phase}_i=\mathrm{play}}
\gamma_{\mathrm{traj}}^{n_i}
\gamma_{\mathrm{step}}^{t_i}r_i.
\]

This supports sparse terminal reward and also remains correct if future
environments emit intermediate rewards.

Example exposing the legacy discrepancy: attempt 1 has two zero-reward play
records; attempt 2 succeeds on its third play record with reward 10. Then

\[
U=0.6\cdot0.95^2\cdot10=5.415,
\]

whereas the legacy recurrence gives `step_return[0] = 5.14425` because it adds an
extra `0.95` while moving backward through the earlier attempt.

### Baseline selection algorithm

1. Compute one $U_{gj}$ for every unique sibling trajectory.
2. Compute one self-inclusive task mean for every task.
3. Build attempt-1 play groups by `(task_uid, hashable(anchor_obs))`.
4. Within each state group, map `traj_uid -> U`; duplicate visits overwrite or
   assert the same utility rather than append another vote. Separately collect
   the set of successfully projected actions from all real occurrences.
5. Compute the unique support, projected-action diversity, and utility range.
   Use the state mean for every occurrence only if support is at least 3,
   action diversity is at least 2, and utility range is strictly greater than
   $\tau_U$. Otherwise use the trajectory's task advantage and record the
   failed gate or gates.
6. Use task advantage for every reflection and later-attempt play record.
7. Add local penalties to obtain the record scalar
   $A^{\mathrm{final}}_{gji}$.
8. Broadcast that scalar only to final trainable response-token positions.

Task advantages sum to zero across unique siblings. Supported state advantages
sum to zero across the unique trajectories in that state. Neither property is
required record-wise after repeated visits, nor after local penalties.

### Exact loss weights

At fixed model parameters, define the positive token coefficient

\[
w_{gj}=\frac{1}{|\mathcal G|K_gN_{gj}}.
\]

Then the desired policy-gradient loss is equivalently a masked token sum with
weighted advantages

\[
\widetilde A_{gjil}=m_{gjil}\,w_{gj}A^{\mathrm{final}}_{gji}.
\]

PPO's clipped loss is positively homogeneous in the advantage, including the
existing dual-clip branch. Therefore multiplying the advantage by positive
$w_{gj}$ and using VERL's existing distributed `token-sum` reduction is
mathematically equivalent to applying the outer task/trajectory and inner token
means explicitly.

This offers a minimally invasive actor interface:

- retain unweighted scalar advantage and baseline metadata for diagnostics;
- put the weighted token advantage into the actor's existing `advantages`
  tensor;
- use `actor.loss_agg_mode=token-sum` in `meta_trajectory` mode;
- optionally retain a token-shaped `meta_traj_loss_weight` for assertions and
  audit dumps, without requiring the generic actor to understand trajectory
  IDs.

Rollout-importance weights multiply the clipped token loss after this weighting
in current VERL. Positive trajectory weights commute with that multiplication.
The frozen convention is that $N_{gj}$ counts tokens selected by the final
**binary** training mask. A binary rejection removes a token before $N_{gj}$ is
computed; continuous importance coefficients $\rho_{gjil}$ do not enter the
denominator. The final contribution is therefore
$N_{gj}^{-1}\rho_{gjil}\ell_{gjil}$, without normalization by
$\sum_{i,l}\rho_{gjil}$. Equal trajectory mass is exact before importance
correction and may be perturbed afterward; this is accepted.

## IMPLEMENTATION INVARIANTS

1. Synthetic padding rows never enter utilities, baselines, support counts,
   $N_j$, or diagnostics for real trajectories.
2. Every real task has the expected unique sibling count; duplicate/missing
   trajectory IDs fail loudly.
3. Queued `uid` and `extra_fields.task_uid` agree for every real record.
4. A trajectory utility is computed once from play rewards and is invariant to
   record order, reflection count, response length, and hard-truncation masks.
5. Task and state baselines consume exactly one utility per unique trajectory.
6. State grouping is exact raw-anchor equality within a task and attempt-1 play
   only. Eligibility additionally requires at least 3 unique trajectories, at
   least 2 distinct successfully projected actions, and utility range
   $>\tau_U$.
7. A state record that fails any eligibility gate, and every later play and
   reflection, uses exactly the task advantage.
8. Local shaping changes only the affected record's advantage. Baseline inputs
   remain the unshaped $U_j$.
9. Hard truncation is exactly generated response length equal to the configured
   maximum. Under `overlong_mode=soft_shaping`, a hard-truncated record retains
   its ordinary response mask, contributes its trainable tokens to $N_j$,
   receives no generic invalid penalty, and receives the soft-overlong penalty
   through the cap, including exactly $-\lambda_{\mathrm{overlong,max}}$ at
   $L=L_{\max}$.
10. Rollout-correction masking is applied to the ordinary response mask under
    soft shaping, and $N_j$ is computed from the resulting final mask. If the
    explicit `overlong_mode=filter` ablation is later implemented, its
    hard-truncation mask is applied before rollout correction and excluded from
    $N_j$.
11. Any `response_mask` modification, including the optional filter ablation or
    rollout rejection, is written back to the TransferQueue so the actor's
    derived `loss_mask` agrees with it.
12. Prompt tokens and padding tokens have zero advantage and zero loss weight.
13. For `meta_traj_mean`, the sum of token weights in trajectory $j$ is
    $1/(|\mathcal G|K_g)$, and the sum within task $g$ is
    $1/|\mathcal G|$.
14. $N_j>0$ is required until the zero-token policy is decided.
15. Reflection recovery may alter only the next-attempt context and its explicit
    recovery metadata. It cannot alter the raw generated response, parse-validity
    metadata, utility, masks, or local shaping. Play actions are never recovered.
16. Global utilities, baselines, shaping, masks, $N_j$, token weights,
    rollout-importance coefficients, and old log probabilities are computed and
    frozen before outer PPO record minibatching.
17. A rollout contains multiple PPO optimizer minibatches. Within one such
    minibatch, parameters remain unchanged across every FSDP/memory fragment;
    gradients accumulate before one global clip, one optimizer step, and one
    scheduler step. Different outer PPO minibatches intentionally see updated
    parameters and may contain records from the same trajectory.
18. The default remains `ppo_epochs=1`. PPO ratios in every later minibatch use
    the original stored old-policy log probabilities, so clipping becomes
    active as the current policy changes. Memory partitioning within a PPO
    minibatch cannot change its gradient except for floating-point reduction
    order.
19. `original_lamer` bypasses every new utility, baseline, masking, shaping, and
    weighting behavior unless an existing option already applied in that mode.
    Truncated-reflection recovery defaults off and changes original-mode rollout
    behavior only when explicitly enabled as a separate protocol choice.

## Implementation-impact report

### 1. How complete meta-trajectories are reconstructed

The trainer already sees all generated records before actor minibatch splitting.
After excluding `batch.tags[*].is_padding`, records are grouped by
`(task_uid, traj_uid)`. The current `traj_uid` construction includes task UID and
session ID, but the pair is retained as a defensive boundary. Phase and attempt
metadata provide validation and diagnostics; record contiguity is not assumed.

No rollout-format redesign is required.

### 2. Where direct utility is computed

Compute it in the repository-owned trainer's advantage stage, where the complete
real record batch and all `extra_fields` are available. Implement it as a small,
pure helper that accepts record metadata plus gammas. Do not change the rollout's
legacy `_assign_lamer_credit`, because original mode still needs those fields.

### 3. How task utilities are centered

Build `task_uid -> {traj_uid: U}`. Validate exactly one utility per sibling, take
the ordinary arithmetic mean, and map $U_j-b_g^{task}$ back to all records of
that trajectory. This removes the current record-count weighting.

### 4. How unique-trajectory state baselines are built

Filter to `phase == "play" and attempt_index == 0`, then build
`(task_uid, hashable(anchor_obs)) -> {traj_uid: U}`. A dictionary enforces one
utility vote. In parallel, build the distinct set of projected actions from
records where `extra_fields.is_action_valid` is true, using
`extra_fields.parsed_action` as the action identity. Validate that a
trajectory's repeated visits always map to the same $U_j$, but do not collapse
the action set. Compute utility range from the deduplicated utility map. A second
pass assigns the selected advantage to every occurrence.

### 5. How fallback works

Precompute task advantage for every trajectory. Initialize every record with
that value, then replace only attempt-1 play records whose state group passes all
three gates: unique support at least 3, projected-action diversity at least 2,
and utility range strictly above $\tau_U$. Persist individual fallback reasons
(`support`, `action_diversity`, `utility_range`) rather than a single opaque
boolean. This makes fallback explicit and prevents later phases from accidentally
entering state logic.

### 6. How local penalties remain outside utility

After baseline selection, compute record-local booleans and lengths and transform
the scalar record advantage. Never mutate the utility map or baseline maps.
Store diagnostic components separately: base advantage, invalid adjustment,
overlong adjustment, and final advantage.

### 7. How hard truncation remains trainable while consequences remain

The rollout already records generated length as `response_token_count` and the
configured maximum as `response_token_limit`; these are the concrete meanings of
`generated_response_length` and `MAX_RESPONSE_LENGTH`. Define
`response_token_cap_hit` by equality between them. Because response IDs are
already sliced to the cap, the current `>=` check is observationally equivalent,
but the new tests use the frozen equality definition. In the advantage stage,
keep the ordinary response mask for `overlong_mode=soft_shaping`, skip the
generic invalid penalty, and add the soft-overlong adjustment evaluated at the
observed length. At the cap this adjustment is exactly
`-max_coefficient`. Rollout correction therefore sees the capped tokens as
potentially trainable. Do not delete the record or its `environment_reward`;
therefore its environment consequences remain in the direct utility.

If the explicit future `overlong_mode=filter` ablation is selected, zero the
entire capped response mask before rollout correction and omit its tokens from
$N_j$. This is not the proposed default and must have a separate diagnostic
label.

Reflection recovery belongs in rollout/context construction, before the next
attempt prompt is built, rather than in advantage computation. Reuse the
ordinary reflection parser's last-opening-tag convention. If and only if the
record is hard-truncated, contains a `<remark>` opening tag, contains no closing
`</remark>` tag after that opening, and has nonempty trimmed suffix text, pass
that suffix through the same 2,000-character memory normalization as the ordinary
parser, then pass it as an explicit reflection-memory override to the environment
manager. The clean interface is an optional recovery field/argument, not a
fabricated closing tag: the manager stores the override for the next attempt
while retaining the original projection result and validity metadata. The
generated record keeps the raw response and sets
`reflection_recovered_from_truncation=true`.

This rollout change is controlled by an explicit rollout/environment option that
defaults to false, so historical runs remain reproducible. The proposed
meta-trajectory launcher enables it. Because recovery changes sampled later
contexts rather than credit computation, it may also be enabled deliberately in
another credit mode, but that run is no longer an unchanged original-behavior
comparison. A complete hard-truncated reflection that the ordinary parser can
already extract is not a recovery case. Play generations never enter this
recovery helper.

### 8. How $N_j$ is computed

Compute $N_j$ only after all approved training-mask changes. It is the sum of
the final response mask over all real records in the trajectory. It excludes
prompt tokens, padding, and rollout-rejected tokens. Under soft shaping it
includes trainable tokens from hard-truncated responses; under the explicit
filter ablation it excludes them. It is not the stored raw response-token count.

### 9. Globally normalized loss under outer PPO record minibatches

Before any actor split, attach the global coefficient
$w_{gjt}=1/(|\mathcal G|K_gN_{gj})$ to every trainable token. Let $R$ be the
number of padded record rows in the rollout and $b$ the global PPO record
minibatch size. For a uniformly sampled size-$b$ record subset $B$, use

\[
\widehat L_B(\theta)
=\frac{R}{b}\sum_{r\in B}\sum_{t\in r}
w_{gjt}\,\ell_{gjt}(\theta).
\]

Every row has inclusion probability $b/R$, hence
$\mathbb E_B[\widehat L_B]=L$. Synthetic padding rows belong to the sampling
population but have zero masks, so they change the inclusion probability without
changing $L$. The required factor is therefore the exact upstream sampling
factor $R/b$, not a heuristic token, record, or trajectory normalization.

For a disjoint epoch of $M=R/b$ minibatches, the arithmetic mean of the $M$
scaled minibatch losses equals the complete frozen-rollout objective at fixed
parameters. With optimizer steps between minibatches, the sequence is ordinary
PPO minibatch SGD: later losses are evaluated at updated parameters while their
credit tensors and old log probabilities remain frozen. If `shuffle=false`, as
in the historical successful 4B run, individual fixed blocks are not random
estimators, but their fixed-parameter epoch mean still equals $L$ exactly. The
frozen meta-trajectory experiment uses the existing seeded DataLoader shuffle,
making each block a random estimator without changing its scale.

The VERL/FSDP hierarchy is:

1. Compute global credit, masks, $N_j$, rollout-importance coefficients, and
   $w_{gjt}$ over the complete rollout.
2. Multiply actor-facing weighted advantages by $R/b$ and freeze them.
3. Let the upstream TensorDict DataLoader partition individual record rows into
   global PPO minibatches of size $b$; a trajectory may span minibatches.
4. For each PPO minibatch, let FSDP partition its rank-local rows into arbitrary
   memory microbatches, accumulate every fragment at unchanged parameters,
   synchronize, clip once, step Adam once, and advance the scheduler once.
5. Move to the next outer PPO minibatch, now evaluating its PPO ratio with the
   updated current policy and the unchanged stored old-policy log probabilities.

Only the memory-microbatch partition has the strong partition-invariance
requirement. Reordering outer PPO minibatches can change the SGD path and is an
intentional optimizer property, not an estimator bug.

### 10. Effect of outer PPO optimizer steps

Multiple PPO minibatch steps are intentionally not equivalent to one
whole-rollout update. After the first step:

- later records are evaluated at changed parameters;
- Adam moments change;
- gradient clipping is applied separately;
- the result depends on record order and shuffling;
- the learning rate is applied once per subset.

That procedure is the restored standard PPO minibatch optimizer. The $R/b$
factor makes each random record minibatch correctly scaled for the written
complete-rollout objective; it does not make a sequence of nonlinear Adam steps
equal to one full-batch Adam step. This distinction is deliberate.

Multiple forward/backward calls remain mathematically inert when they are only
memory fragments of one PPO optimizer minibatch. Parameters and optimizer state
remain fixed until its last fragment; gradients add linearly, and clipping is
applied to the accumulated minibatch gradient rather than separately to each
fragment.

### 11. Propagating identities and weights

Keep `task_uid` and `traj_uid` in `extra_fields` through advantage construction
and diagnostics. The generic actor does not need string IDs if the trainer emits:

- final response mask;
- token-shaped weighted `advantages`;
- optional token-shaped `meta_traj_loss_weight` for validation.

This avoids invasive actor grouping. Because the actor already carries arbitrary
TensorDict fields through microbatch slicing, a diagnostic weight tensor is also
split consistently. The mathematical advantage components should remain in
metadata or diagnostic outputs rather than overloading the actor interface.

### 12. Compatibility and invasiveness

No mathematical choice is incompatible with current VERL. The main obstacles
are engineering boundaries:

- Current `compute_gigpo_advantages` must gain a separate meta mode rather than
  be mutated in place.
- Current invalid penalties are inserted into both legacy score branches; the
  new mode needs post-baseline local shaping.
- The current reflection parser returns empty text unless both tags exist, and
  `env_manager.step(..., phase="reflect")` immediately stores that result. Clean
  recovery therefore needs an explicit memory override or dedicated manager
  method; synthesizing a closing tag would corrupt validity diagnostics.
- Current hard-cap records remain trainable, which matches the new default.
  The new mode must apply the cap-inclusive soft-overlong adjustment and leave
  the response mask unchanged unless the explicit filter ablation is selected.
- Current `seq-mean-token-mean` and `token-mean` cannot express trajectory means.
  Weighted advantages plus existing `token-sum` can.
- Current PPO minibatching steps the optimizer per record minibatch. Exact mode
  requires one logical full-rollout optimizer batch with FSDP memory-microbatch
  accumulation. This is the most consequential trainer change, but the inner
  FSDP accumulation path already exists.
- `token-sum` is also used by VERL for entropy and actor KL aggregation. If a
  nonzero entropy or KL regularizer is enabled, its desired trajectory weighting
  needs a separate decision/implementation. The policy-gradient term itself is
  expressible without an upstream VERL patch.
- Upstream PPO reports combined `pg_clipfrac` and a dual-clip lower metric, but
  not asymmetric lower/upper objective-active fractions. Meta mode registers a
  loss-equivalent `meta_vanilla` wrapper that additionally reports positive-
  advantage upper clipping and negative-advantage lower clipping separately.

GPU activation memory need not scale with the full record batch because existing
microbatching performs backward immediately. Host-side batch metadata and the
time between optimizer steps will grow, and final gradient synchronization may
become a throughput bottleneck; these require a small smoke benchmark.

After the exact path is validated, the following throughput changes preserve the
mathematics: length bucketing; dynamic token-count microbatching; balanced
assignment of token load to data-parallel ranks; concurrent per-rank execution;
FSDP `no_sync`/deferred synchronization on nonfinal fragments; activation
checkpointing; padding removal or sequence packing that preserves token masks;
mixed precision under the same reduction policy; fused attention/optimizer
kernels; input prefetch; and communication overlap that does not expose updated
parameters. Computational shuffling is also valid under deterministic or
sample-stable randomness. Gradient clipping per fragment, optimizer stepping per
fragment, stale-parameter pipelines, and asynchronous local optimizer steps are
not equivalent throughput optimizations and require a separately named ablation.

### 13. Proposed configuration surface

New options:

```yaml
algorithm:
  credit_mode: original_lamer  # original_lamer | meta_trajectory
  meta_trajectory:
    state_baseline:
      enabled: true
      min_unique_trajectories: 3
      min_distinct_projected_actions: 2
      utility_range_tolerance: 1.0e-6
    invalid_penalty:
      enabled: true
      coefficient: 0.5
    overlong_penalty:
      enabled: true
      mode: soft_shaping              # soft_shaping | filter
      max_coefficient: 0.5
      play_cache_tokens: 150
      reflection_cache_tokens: 150
    loss_aggregation: meta_traj_mean
    fixed_sum_denominator: null      # only for future meta_traj_fixed_sum
    require_one_step_per_ppo_minibatch: true

trainer:
  meta_credit_diagnostics:
    enabled: true
    dump_frequency: 1
    max_tasks_to_dump: 4

env:
  recover_truncated_reflection_memory: true  # meta launcher; default is false
```

Existing options set, not newly introduced:

```yaml
algorithm:
  step_gamma: 0.9675
  traj_gamma: 0.6

actor_rollout_ref:
  actor:
    clip_ratio_low: 0.20
    clip_ratio_high: 0.28
    entropy_coeff: 0
    loss_agg_mode: token-sum  # required implementation detail in meta mode
    policy_loss:
      loss_mode: meta_vanilla # vanilla math plus asymmetric diagnostics
    ppo_epochs: 1
    shuffle: false
```

`credit_mode` should own estimator semantics. The old
`algorithm.gigpo.use_invalid_action_penalty` remains meaningful only for original
mode; meta mode reads the new local-shaping block. Invalid combinations should
raise configuration errors rather than silently override one another.

The recovery switch lives in `env` because the ALFWorld agent loop already
receives that configuration while it constructs future prompts; it does not
naturally receive the trainer's advantage block. The meta-mode launcher enables
the switch explicitly. Keeping it orthogonal also permits a controlled rollout
ablation with another credit mode without coupling the rollout worker to trainer
internals; such an ablation must not be labeled unchanged `original_lamer`.

The architecture should reserve loss-aggregation values `token_mean` and
`meta_traj_fixed_sum` without implementing them until an ablation is approved.

### 14. Toy/reference tests

The complete executable-test plan is below.

## IMPLEMENTATION PLAN

Implementation should proceed through independently reviewable checkpoints:

1. **Pure estimator helpers.** Add direct utility, task baseline, three-gate
   state eligibility/deduplication, fallback, local shaping, and reference loss
   helpers with CPU-only tests. Do not connect them to training yet.
2. **Rollout metadata and reflection recovery.** Add the explicit manager memory
   override, recovery metadata, equality-based cap diagnostic, and prompt-level
   tests. Confirm raw output/validity isolation and original-mode snapshots
   before proceeding.
3. **Masks and global weights.** Connect the pure estimator after complete-batch
   reconstruction, write back final masks, compute global $N_j$ and token
   coefficients, and compare the actor's `token-sum` loss against the reference
   without taking an optimizer step.
4. **One logical optimizer update.** Route one intended rollout update through
   a single outer actor batch and existing FSDP memory microbatches. Instrument
   zeroing, synchronization, clipping, stepping, scheduler advancement, and
   parameter versions. First validate with CPU reference/integration tests,
   then with the smallest memory-safe actor-only GPU smoke test.
5. **Real-batch audit and throughput smoke test.** Run the required diagnostics,
   partition-invariance checks, memory measurement, and timing. Add length/token
   bucketing only after the unbucketed path matches the reference.

Each checkpoint preserves `original_lamer`; none should combine a failed
mathematical assertion with a throughput optimization. Production training is
blocked until the acceptance conditions below pass and the remaining
loss-affecting OPEN items are resolved or explicitly disabled.

### Checkpoint 3 implementation boundary

Checkpoint 3 is implemented as a two-stage production adapter. The first stage
excludes synthetic padding, reconstructs complete `(task_uid, traj_uid)` groups,
calls the frozen pure utility/baseline helpers, applies phase-aware local
shaping, and applies only an explicitly selected `overlong_mode=filter` mask.
The default `soft_shaping` mode preserves the ordinary response mask, including
at the generation cap. The trainer then applies any existing decoupled rollout
correction. The second adapter stage computes $N_j$ from that final mask and
broadcasts $1/(|\mathcal G|K_gN_j)$ plus the selected record advantage to token
positions.

The TransferQueue receives the final `response_mask`, weighted actor
`advantages`, unweighted token advantages, token weights, and scalar diagnostic
components. The production adapter fails closed on incomplete sibling groups,
inconsistent task IDs/cap metadata, duplicate structural record indices, or
$N_j=0$. `original_lamer` enters the pre-existing code path before any of these
operations and remains the configuration default.

Checkpoint 3 itself was frozen as a no-update checkpoint: selecting
`credit_mode=meta_trajectory` stopped before the worker update. Checkpoint 4
replaces only that historical guard. No estimator, mask, token-weight, utility,
baseline, or shaping semantics from Checkpoint 3 are changed.

### Checkpoint 4B PPO-minibatch update boundary

The old/original audit established that the outer unit is one generated
record/sequence row. `actor.ppo_mini_batch_size=8` is multiplied by
`rollout.n=8`, giving 64 global records per PPO minibatch. The upstream trainer
pads the rollout with masked synthetic rows to a multiple of 64, so an
incomplete minibatch is neither dropped nor retained at a smaller size. The
successful 4B run used `actor.shuffle=false`; its representative step-80 batch
had 916 real records, was padded to 960, and therefore produced 15 optimizer
steps. The TensorDict DataLoader supports seeded random row shuffling, but it was
disabled in that historical run. The frozen meta-trajectory launcher instead
uses `actor.shuffle=true`: this deliberately departs from the historical record
order so each outer PPO minibatch is a seeded random record sample matching the
stochastic-estimator derivation below.

Every worker PPO minibatch enters the existing engine lifecycle:

1. `BaseEngine.train_batch()` calls `optimizer_zero_grad()` once.
2. FSDP `forward_backward_batch()` partitions the batch into configured memory
   microbatches. Each fragment immediately backpropagates; nonfinal fragments
   use the existing deferred-gradient-synchronization context and the final
   fragment synchronizes.
3. `optimizer_step()` measures/clips the global accumulated norm once and calls
   the optimizer once.
4. The audited Qwen LoRA engine advances the scheduler immediately after that
   optimizer step. Pinned upstream VERL historically advanced it only on the
   rollout's final minibatch; Checkpoint 4B intentionally ties it to every Adam
   step. This difference is numerically inert for the frozen constant schedule.

The trainer rejects `ppo_epochs != 1`,
`require_one_step_per_ppo_minibatch != true`,
`parameter_sync_step != 1`, an empty rollout batch, or aggregation other than
`token-sum`. The frozen Qwen3-4B LoRA launcher also requires
`LAMER_META_PPO_MINIBATCH_UPDATE=1`. Under this flag the audited LoRA FSDP2 engine
suppresses VERL's otherwise redundant successful-context-exit `zero_grad`
calls, while retaining exception cleanup, and releases gradient tensors
directly after each PPO-minibatch optimizer step. Thus each PPO minibatch has
one initial `zero_grad()` without retaining gradient storage between updates.
It emits explicit complete-record, outer-minibatch, estimator-scale,
optimizer-step, and scheduler-step diagnostics.
`original_lamer` still calls the unmodified upstream `_update_actor()`.

This boundary does not serialize computation. Records may be reordered or
distributed, and a trajectory may span PPO minibatches, ranks, or memory
microbatches because its global coefficient was attached before partitioning.
Within each PPO minibatch, length bucketing, dynamic token microbatches,
balanced rank assignment, packing, activation checkpointing, communication
overlap, and fused kernels remain valid so long as every memory fragment sees
the same parameters and no clip, optimizer, or scheduler operation occurs
before the final fragment.

Checkpoint-4B validation is limited to one synthetic rollout containing at
least two PPO optimizer minibatches, each with at least two memory fragments
where feasible. It uses Qwen3-4B LoRA, actor-only FSDP, and no rollout engine.
Immediately before launch, current free VRAM must be
checked and the estimated own peak must leave at least 32 GiB globally free.
If that bound cannot be established confidently, the GPU test is skipped rather
than consuming the reserve. Multi-step rollout/training remains out of scope.

## REQUIRED UNIT TESTS

All mathematical helpers should be CPU-only pure functions with hand-computed
expected values.

### Direct utility

1. **Four-sibling ordering.** With terminal reward 10 at turn zero, construct
   attempt-1 success, attempt-2 success, attempt-3 success, and failure. For
   `gamma_traj=0.6`, utilities must be `[10, 6, 3.6, 0]`; mean `4.9`; task
   advantages `[5.1, 1.1, -1.3, -4.9]`.
2. **Within-attempt discounting.** Move otherwise identical reward between turns
   and verify powers of `gamma_step`.
3. **Nested versus legacy recurrence.** Use the two-failed-actions/attempt-2-
   third-turn-success example and assert direct `5.415` versus legacy
   `5.14425`.
4. **Reflection invariance.** Insert, remove, and reorder reflection records;
   direct utility must not change.
5. **Intermediate rewards.** Multiple rewards across attempts must equal the
   explicit double sum, not only a terminal-reward shortcut.
6. **Record-order invariance.** Shuffle records while retaining indices and
   assert unchanged utilities.

### Task centering

7. Assert each trajectory contributes once despite different record/token
   counts.
8. Assert unique-sibling task advantages sum to zero.
9. Assert all-failure and identical-success groups produce zero task advantage.
10. Assert synthetic copied padding rows do not change utilities or baselines.

### State centering and fallback

11. Three trajectories with utilities `[10, 6, 0]` and projected actions
    `['go north', 'open fridge', 'go north']` sharing an attempt-1 state pass all
    gates and receive state advantages `[14/3, 2/3, -16/3]`.
12. With two unique trajectories, differing actions, and differing utilities,
    assert task-advantage fallback with reason `support`.
13. With at least three unique trajectories and differing utilities but only one
    projected action, assert task-advantage fallback with reason
    `action_diversity`.
14. With at least three unique trajectories and at least two actions but equal
    utilities, assert task-advantage fallback with reason `utility_range` and
    verify useful nonzero task advantages are not overwritten in a task where
    nonvisiting siblings make the task mean different.
15. Assert utility range exactly equal to $\tau_U$ falls back and a representable
    value strictly above it is eligible. Cover the approved absolute/relative
    convention once the open tolerance is frozen.
16. Repeat one trajectory's state visit several times; baseline support and
    utility vote stay unchanged, every occurrence receives the selected
    advantage, and all valid projected actions from those occurrences remain
    represented in the action-diversity set.
17. Assert invalid/unprojectable play outputs and their diagnostic fallback
    strings do not count as distinct projected actions and cannot by themselves
    make a state eligible.
18. Put the same raw observation in another task; it must form another group.
19. Assert exact raw-observation equality and no history/prompt keying.
20. Assert attempt-2/3 play and reflection records always use task advantage,
    even if their raw state would pass every eligibility gate.
21. Assert eligible state advantages sum to zero across unique visiting
    trajectories, not necessarily across occurrences. Assert fallback groups
    retain the task-centering property at task level.

### Local shaping and masks

22. A nontruncated invalid record changes only its own advantage by `-0.5`; all
    utilities and baseline values remain bitwise unchanged.
23. Verify the piecewise soft-overlong penalty below, at, within, and at the end
    of the cache interval separately for play and reflection caches, including
    $p(L_{\max})=-\lambda_{\mathrm{overlong,max}}$.
24. Verify additive invalid plus soft-overlong shaping for a completed malformed
    overlong response.
25. Assert hard truncation is true exactly at
    `generated_response_length == MAX_RESPONSE_LENGTH`; under soft shaping such a
    response retains its ordinary mask, receives no generic invalid penalty,
    receives the maximum soft-overlong penalty, and still retains its environment
    reward in utility computation. Separately assert that the explicit `filter`
    ablation zeros its response mask.
26. Assert modified `response_mask`, actor `loss_mask`, and token advantages agree.
27. Assert an all-masked trajectory raises the specified fail-closed error until
    the $N_j=0$ policy is decided.

### Truncated-reflection recovery

28. A hard-truncated reflection containing `<remark>use the key` and no closing
    tag installs `use the key` in the next-attempt memory and prompt, sets
    `reflection_recovered_from_truncation=true`, but preserves the raw response,
    false ordinary parse validity, hard-truncation flag, ordinary trainable PPO
    mask, generic-invalid exemption, and maximum soft-overlong penalty.
29. Assert no special recovery for each boundary case: nontruncated open-only
    reflection, hard-truncated valid closed reflection, open-only reflection with
    empty/whitespace suffix, and every incomplete play action. Closed reflections
    retain existing ordinary-parser behavior.
30. With multiple opening tags, assert recovery follows the ordinary reflection
    parser's last-opening-tag convention and never captures a closing tag that
    precedes the selected opening.
31. With a suffix longer than 2,000 characters, assert extraction reaches the
    generation boundary but installed memory uses the existing 2,000-character
    cap exactly like a normally closed reflection.
32. Assert recovery changes only later context and explicit diagnostics: the
    recovered record's raw tokens, utility inputs, masks, validity, and shaping
    are unchanged. Snapshot `original_lamer` with recovery disabled.

### Exact trajectory-level loss weighting

33. Construct one trajectory with two records of lengths 2 and 3 and another
    with one record of length 1. Verify the reference result is the mean of the
    5-token trajectory mean and the 1-token trajectory mean, not a three-record
    mean or six-token global mean.
34. Split one response into two records without changing token losses; total loss
    must be unchanged.
35. Merge records; total loss must be unchanged.
36. Verify per-trajectory token weights sum to `1/(num_tasks * K_g)` and per-task
    weights sum to `1/num_tasks`, including variable $K_g$ in the pure helper.
37. Compare explicit nested-loop reference loss with weighted-advantage plus
    VERL `token-sum` to tight tolerance.
38. Derive and test the $R/b$ record-minibatch scale. Averaging over every
    possible equal-size random record subset must recover the complete rollout
    objective and gradient at fixed parameters, including unequal trajectory
    lengths.
39. Instrument at least two outer PPO minibatches and assert one `zero_grad`,
    one global gradient clip, one `optimizer.step`, and one scheduler advancement
    per PPO minibatch. Assert no parameter or Adam-state change between memory
    fragments, and a parameter change between outer minibatches.
40. Add a distributed FSDP integration test (at least two ranks) that splits one
    trajectory across ranks and microbatches, then compares accumulated gradients
    and updated parameters with an unsplit/reference update to reduction-order
    tolerance.
41. Run the partition tests with training-time stochastic layers disabled, then
    with sample-stable RNG if such a layer is supported. A deliberately
    order-sensitive RNG fixture must demonstrate why uncontrolled dropout would
    violate the stronger invariant.
42. Verify upstream record-row shuffle/partition preserves attached utilities,
    advantages, token weights, importance coefficients, masks, and stored old
    log probabilities, including one trajectory split across PPO minibatches.
43. Repeat the equivalence test with asymmetric PPO clipping active on both
    signs of advantage.
44. Repeat with fixed rollout-importance weights and verify continuous rho does
    not enter $N_j$ or the $R/b$ factor.

### Mode isolation

45. Snapshot-test `original_lamer` outputs against current behavior.
46. Assert new config fields, including reflection recovery, do not affect
    original mode.
47. Assert meta mode rejects incompatible aggregation, `ppo_epochs != 1`, an
    invalid padded-record/minibatch divisibility relation, missing IDs,
    inconsistent task IDs, duplicate siblings, and unresolved required values.

## REQUIRED REAL-BATCH DIAGNOSTICS

Before any training experiment, run one saved or freshly generated rollout batch
through a no-update audit and report:

### Reconstruction and utility

- real versus synthetic record counts;
- tasks, unique siblings per task, and completeness failures;
- per sibling: task, trajectory ID, play/reflection count, attempt lengths,
  response tokens, success attempt, direct $U_j$, and legacy first
  `step_return` for comparison;
- task utility mean and unique-trajectory advantage sum;
- counts of nonfinite values and structural-index violations.

### State baseline

- support histogram by unique trajectories;
- distinct projected-action-count and utility-range histograms;
- eligible/fallback record counts by task category and by each individual or
  combined failed gate (`support`, `action_diversity`, `utility_range`);
- occurrence count versus unique-vote count;
- repeated-visit count and maximum visits;
- null/unprojectable-action count in candidate groups;
- per selected task/state: visitor utilities, projected actions by occurrence,
  distinct-action set, utility range, state mean, assigned advantages, and
  fallback reasons.

### Sibling rollout diversity

- task-group size after collapsing every record to one unique trajectory vote;
- mixed-outcome, all-failure, and all-success task-group fractions, plus the
  mean and population standard deviation of successful siblings per task;
- mean fraction of unique direct utilities within each task group and fraction
  of task groups whose sibling utilities are all equal;
- mean fraction of unique complete projected-action paths within each task
  group and fraction of groups whose action paths are fully collapsed;
- mean unique fraction for the first projected action and first generated
  response of each sibling meta-trajectory;
- compute response identity from generated token IDs rather than retaining raw
  rollout text, and exclude synthetic padding records from every statistic.

### Local shaping and masks

- invalid nontruncated, hard-truncated, soft-overlong, and overlapping
  invalid/soft counts by phase;
- recovered-reflection count and rate overall, by source attempt, among all
  reflections, and among hard-truncated reflections;
- next-attempt and eventual downstream success counts/rates for trajectories
  with a recovered reflection, reported separately from hard-truncated
  reflections without recovery (descriptive, not a causal estimate);
- examples containing raw truncated reflection, recovered memory, next prompt,
  ordinary validity, recovery flag, and downstream outcome;
- examples showing $U_j$, base advantage, each local adjustment, and final
  advantage;
- raw response tokens versus final trainable tokens, including confirmation that
  capped tokens remain trainable under soft shaping and are removed only under
  the separately labeled filter ablation;
- trajectories with $N_j=0$, which must stop the run;
- proof that hard-truncated environmental rewards remain in $U_j$.

### Loss weights

- $N_j$ distribution overall and by outcome;
- sum, minimum, and maximum token weight per trajectory;
- assertion residual from expected `1/(num_tasks*K_g)` per trajectory;
- assertion residual from expected `1/num_tasks` per task;
- explicit reference loss versus actor-weighted `token-sum` loss;
- the same comparison after random record and microbatch partitioning;
- fragments and tokens processed per rank/microbatch, imbalance, and peak memory;
- counts and positions of `zero_grad`, final gradient synchronization, global
  gradient clipping, `optimizer.step`, and scheduler advancement per PPO
  optimizer minibatch, plus aggregate counts for the complete rollout update;
- gradient/parameter residuals versus the reference after reordering, length
  bucketing, dynamic token microbatching, and distributed partitioning.

### PPO and rollout correction

- ratio mean, standard deviation, min/max, and P01/P05/P50/P95/P99;
- fraction below `1-epsilon_low` and above `1+epsilon_high`;
- objective-active `clipfrac_low` for negative advantages and
  `clipfrac_high` for positive advantages, plus denominators;
- existing dual-clip lower fraction reported separately;
- correction-weight distribution, zero-weight fraction, rejection-mask fraction,
  and effective sample size;
- trajectory-normalization totals before and after correction under each
  candidate convention needed to settle the open interaction.

### Acceptance conditions before a training launch

1. All complete trajectories reconstruct with the expected sibling count.
2. Direct utilities match an independent reference calculation.
3. Unique task/state centering residuals are numerically negligible.
4. Every state-baseline use passes all three eligibility gates, and every failed
   gate maps exactly to task advantage.
5. Under soft shaping, every hard-truncated record retains its ordinary PPO mask,
   receives no generic invalid penalty, and receives the cap-inclusive
   soft-overlong penalty. Each recovered reflection appears exactly in the
   intended next-attempt context without changing raw parse validity or masks,
   and no play action is recovered. Any filter-ablation run is labeled and
   verified separately.
6. Every trajectory has $N_j>0$, unless a later approved policy replaces this
   temporary invariant.
7. The mean $R/b$-scaled PPO-minibatch loss/gradient at fixed parameters matches
   the explicit complete-rollout reference, including shuffled record splits and
   unequal trajectory lengths.
8. Within every PPO minibatch, parameters and Adam state remain fixed across
   memory fragments; clipping, optimizer stepping, and scheduler advancement
   occur exactly once. Parameters intentionally change between PPO minibatches.
9. Original mode remains numerically unchanged.
