# Source freeze for the ALFWorld experiments

This note records the archival boundary created on 2026-10-15 before starting
the AppWorld repository. It preserves the current ALFWorld implementation as
historical source material; it does not designate every file here for reuse.

## Repository implementation

The production implementation is frozen by commit
`d74f55bf43de047065672787577c58eb0a7e3bea` (`add meta trajectory training`).
It contains the complete-meta-trajectory estimator, the matched LaMer-form
path, the modern VERL trainer integration, ALFWorld rollout metadata and
reflection recovery, Qwen LoRA precision handling, performance instrumentation,
dataset construction, and the three scientific training entry points.

Tests, smoke utilities, notebooks, generated artifacts, temporary commands,
and personal notes were deliberately not included in that implementation
commit under the repository policy. They remain in the local working tree.

## External VERL dependency and local patch

The modern runtime uses upstream VERL revision
`fbb4b3a8bf636f290c9c59fc346f756849e9c241`. The checkout used by the latest
performance diagnostics has one local modification that exports vLLM request
queue, prefill, and decode timings. Its exact diff is stored at
`patches/verl-vllm-rollout-timing.patch` with SHA-256:

```text
f0f2b67973172939e48f968f264f8b0cf5aaf4abaa2786c61736d93260cd402b
```

Apply it from a clean checkout of the pinned VERL revision with:

```bash
git apply /path/to/LaMer/patches/verl-vllm-rollout-timing.patch
```

This patch adds diagnostics only. It was added after the two completed
150-step comparison runs below and is not part of their policy computation.

## Completed LaMer-form comparison

Run directory:

```text
outputs/qwen3_4b_modern/lora_lamer_6cat_tokenmean_150
```

The run reached step 150. The checkpoint selector retained best step 130 and
final step 150. The run contains its exact source archive and tracked diff:

| Artifact | SHA-256 |
| --- | --- |
| `source_snapshot.tar.gz` | `e2d87464a9e71e912e8bfb0a787b4d67b8386aee8f1849413061b3580372c827` |
| `working_tree.patch` | `3a304e51721780226bad63474deee68020870797436251ab6c83cdab8aa1a589` |

The archive is authoritative for the exact run source. Its estimator,
production adapter, trainer, LoRA precision hook, prompts, projection,
reflection recovery, dataset builder, and base environment files are
byte-identical to the implementation commit. The current agent loop,
environment manager, and shared launcher additionally contain later optional
performance instrumentation, so the archived copies should be used for exact
reconstruction.

## Completed complete-meta-trajectory comparison

Run directory:

```text
outputs/qwen3_4b_modern/lora_meta_6cat_full_150_retry1
```

The resumed run reached step 150. The checkpoint selector retained best step
140 and final step 150. Recorded tracked-file patches include:

| Artifact | SHA-256 |
| --- | --- |
| initial `working_tree.patch` | `efaf5e2a49b528d3aefe616ab785efca8556b710d31627e382ac1dac89f2bd0a` |
| final resume `working_tree.patch` | `aed683a7cf70259d7fe7b54fb36e6edc3fafaef5734ef5d7809e66c4279710ba` |

This run was launched before the shared launcher created source tarballs.
Consequently, its directory does not independently contain the then-untracked
estimator files. The later LaMer-form source archive contains byte-identical
copies of the meta-trajectory estimator, production adapter, trainer,
precision hook, prompts, projection, reflection recovery, dataset builder,
and environment implementation now frozen in Git. This is strong
cross-artifact evidence, but it is a reconstruction rather than a direct
per-run source snapshot and must be reported with that caveat.

## Runtime pins shared by the comparison runs

Both runs record the following modern stack:

| Component | Version or revision |
| --- | --- |
| Python | `3.12.14` |
| VERL | `fbb4b3a8bf636f290c9c59fc346f756849e9c241` |
| PyTorch | `2.13.0+cu130` |
| Transformers | `5.12.1` |
| vLLM | `0.29.0` |
| Qwen3-4B | `1cfa9a7208912126459214e8b04321603b3df60c` |

The generated run directories and model/checkpoint data remain outside Git.
They must be backed up together with this repository if future reconstruction
is required.
