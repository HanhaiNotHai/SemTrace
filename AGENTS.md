# AGENTS.md

## Purpose

This repository uses a requirements-confirmation workflow.

Before modifying any source code, configuration, tests, scripts, documentation,
dependencies, datasets, checkpoints, or generated artifacts, establish a shared
and explicitly approved understanding of the task.

Correctness and alignment take priority over speed.

---

## Mandatory Skills

Use the installed skills according to the following rules.

### `interview-me`

Use before implementation whenever any material requirement, constraint,
acceptance criterion, interface, expected behavior, or scope boundary is unclear.

Ask exactly one focused question at a time.

Do not send a batch questionnaire.

### `source-driven-development`

Use before relying on:

* Third-party library APIs.
* Framework behavior.
* Command-line options.
* File formats.
* Configuration schemas.
* Version-specific features.

Inspect the repository's installed versions and existing usage before proposing
or writing code.

Prefer primary documentation and source code over memory or assumptions.

### `karpathy-guidelines`

Use throughout implementation.

Requirements include:

* Surface assumptions instead of silently guessing.
* Prefer the smallest implementation that satisfies the approved request.
* Modify only lines that directly support the task.
* Preserve the repository's existing style and architecture.
* Do not introduce speculative abstractions or future-facing features.
* Define concrete and verifiable completion criteria.

### `test-driven-development`

Use for:

* New behavior.
* Bug fixes.
* Behavioral refactors.
* Public-interface changes.
* Data-processing changes.
* Model-component changes that can be tested deterministically.

For a bug fix, reproduce the failure before applying the fix whenever practical.

### `debugging-and-error-recovery`

Use whenever:

* A test fails unexpectedly.
* Runtime behavior differs from expectations.
* A command returns an unexplained error.
* Multiple attempted fixes have failed.
* The root cause is not yet established.

Do not perform random speculative edits.

### `code-review-and-quality`

Use after implementation and before declaring completion.

Review the final diff for:

* Incorrect behavior.
* Missing edge cases.
* Interface regressions.
* Device, dtype, shape, or state errors.
* Unrelated changes.
* Missing tests.
* Unnecessary complexity.
* Documentation or configuration inconsistencies.

---

# Requirements Confirmation Gate

## Rule

Do not begin implementation until the user has explicitly approved a restatement
of the requirements.

This applies to every task that may modify files or observable behavior.

The gate has four phases:

1. Read-only inspection.
2. Requirements interview.
3. Requirements restatement.
4. Explicit user approval.

---

## Phase 1: Read-Only Inspection

Before asking questions, inspect enough repository context to avoid unnecessary
or uninformed questions.

Allowed read-only actions include:

* Reading source files.
* Reading tests.
* Reading configuration files.
* Reading dependency manifests.
* Searching the repository.
* Inspecting Git status and history.
* Identifying existing interfaces and conventions.
* Checking documented commands.
* Inspecting installed dependency versions without changing them.

During this phase, do not:

* Edit, create, rename, move, or delete files.
* Install or upgrade dependencies.
* Run formatting tools that rewrite files.
* Create commits.
* Start training.
* Download datasets or model weights.
* Run migrations.
* Execute destructive commands.
* Change the current environment.

Do not ask the user questions whose answers can be determined safely from the
repository.

---

## Phase 2: Requirements Interview

After inspection, determine whether the request is complete.

A requirement is materially incomplete when the missing information could change:

* What is implemented.
* Which files or modules are modified.
* A public or internal interface.
* Input or output formats.
* Model architecture.
* Dataset behavior.
* Training or evaluation behavior.
* Performance or memory requirements.
* Backward compatibility.
* Test expectations.
* Dependency choices.
* The definition of completion.

When material information is missing, invoke `interview-me`.

Ask exactly one focused question at a time.

Each question must resolve one concrete uncertainty.

Use this format:

```text
Current understanding:
<one concise sentence>

Question:
<one focused question>

Best guess:
<the most likely answer and why>
```

Wait for the user's answer before asking the next question.

Do not ask several unrelated questions in one message.

Do not repeat a question that the user has already answered.

Do not ask abstract questions such as:

* "What do you want?"
* "What should the focus be?"
* "Any other requirements?"
* "How should I implement it?"

Ask concrete questions tied to an implementation decision.

Examples:

* "Should the new detector preserve the existing `forward(x)` return type?"
* "Should the checkpoint remain loadable by the current evaluation script?"
* "Is CPU execution required for the smoke test?"
* "Should this change affect training only, or training and inference?"
* "Is the target input shape fixed or dynamic?"

---

## Complete Initial Requests

When the user's initial request already defines all material requirements:

* Do not invent additional interview questions.
* Do not ask redundant questions merely to satisfy a process.
* Proceed directly to the requirements restatement.
* Still wait for explicit approval before modifying files.

---

## Phase 3: Requirements Restatement

When enough information has been collected, restate the requirements using this
exact structure:

```text
Requirements confirmation

Objective:
- ...

Expected behavior:
- ...

Likely scope:
- ...

Interfaces to preserve:
- ...

Constraints:
- ...

Acceptance criteria:
- ...

Out of scope:
- ...

Assumptions:
- ...

No files have been modified yet.

Reply "确认执行" to approve this specification, or provide corrections.
```

The restatement must be based on the user's words and verified repository facts.

Do not hide assumptions.

If an assumption can materially affect the implementation, convert it into a
question instead of leaving it implicit.

Always include an `Out of scope` section.

If the user corrects the restatement:

1. Incorporate the correction.
2. Present the complete updated restatement.
3. Request approval again.

---

## Phase 4: Explicit Approval

Implementation may start only after an unambiguous approval of the latest
requirements restatement.

Examples of valid approval:

* `确认执行`
* `按这个方案执行`
* `同意以上需求，开始实现`
* `Yes, implement the confirmed specification`
* Another unmistakable approval of the latest restatement

The following do not count as approval:

* Silence.
* `随便`.
* `你看着办`.
* `应该可以`.
* `差不多`.
* `Sounds good`.
* A response that answers only the last question.
* A new request that changes the task.

When approval is ambiguous, ask one short confirmation question and do not edit
files.

---

## Explicit Skip

The confirmation gate may be skipped only when the user explicitly states in the
current request:

* `直接执行，无需确认`
* `跳过需求确认`
* `不要提问，按当前描述实现`
* An equivalent unambiguous instruction

A skip instruction from an earlier unrelated task does not carry over
automatically.

Even when the requirements interview is skipped, do not perform destructive,
irreversible, security-sensitive, or high-cost operations without explicit
approval.

---

## Read-Only Tasks

The confirmation gate is not required for purely read-only tasks, including:

* Explaining existing code.
* Reviewing code without modifying it.
* Locating files or symbols.
* Summarizing repository structure.
* Reporting Git status.
* Reporting existing test results.
* Comparing implementation options.
* Drafting a plan without executing it.

If the task changes from read-only analysis to file modification, activate the
confirmation gate before making the first modification.

---

# Execution After Approval

After explicit approval, follow this order.

## 1. Recheck Scope

Before editing:

* Re-read the approved restatement.
* Check `git status`.
* Identify the minimum files that need modification.
* Confirm that unrelated user changes will not be overwritten.
* Define the smallest relevant verification command.

If unexpected unrelated changes are present in files that must be modified,
stop and ask the user how to proceed.

Do not overwrite or revert user changes.

## 2. Make Surgical Changes

Implement only the approved scope.

Do not:

* Refactor adjacent code without necessity.
* Rename unrelated symbols.
* Reformat unrelated files.
* Change dependency versions without approval.
* Introduce a new framework when existing project infrastructure is sufficient.
* Add optional features that were not approved.
* Add premature extension points or abstractions.
* Replace an existing working subsystem merely because another design is
  cleaner.

Every changed line must be traceable to an approved requirement.

## 3. Verify Incrementally

After each meaningful implementation slice:

* Run the smallest relevant test.
* Inspect errors before making another change.
* Verify expected inputs and outputs.
* Check compatibility with existing callers.
* Confirm no unrelated files were modified.

Do not postpone all verification until the end.

## 4. Debug Systematically

When verification fails:

1. Reproduce the failure.
2. Record the exact command and error.
3. Establish the expected behavior.
4. Localize the failing layer.
5. Form one testable hypothesis.
6. Test that hypothesis.
7. Apply the smallest root-cause fix.
8. Add or update a regression test.
9. Re-run the relevant verification.

Do not stack multiple speculative fixes into one change.

Do not suppress errors merely to make tests pass.

## 5. Final Review

Before declaring completion:

* Review `git diff`.
* Remove unrelated changes.
* Verify new behavior.
* Verify preserved behavior.
* Run the repository's relevant test command.
* Check linting or static analysis when configured.
* Check imports and module loading.
* Confirm documentation and configuration consistency.
* Confirm all approved acceptance criteria.

Do not claim success if required verification failed.

---

# PyTorch and Research-Code Rules

## Repository Compatibility

Before using a PyTorch, torchvision, timm, Transformers, NumPy, CUDA, or other
third-party API:

* Inspect the repository's dependency files.
* Inspect the actual installed version when available.
* Search for existing usage in the repository.
* Confirm the API exists in that version.
* Preserve current configuration and checkpoint interfaces unless a change was
  explicitly approved.

Do not upgrade a dependency to make an implementation easier without approval.

## Tensor Correctness

For modified model or data code, explicitly verify relevant:

* Batch dimensions.
* Channel dimensions.
* Token dimensions.
* Spatial dimensions.
* Broadcasting behavior.
* Device placement.
* Dtype.
* Gradient flow.
* Training versus evaluation behavior.
* Randomness and determinism.
* Empty, singleton, or partial-batch behavior where applicable.

Do not rely on silent broadcasting unless it is intentional and documented.

Do not introduce unnecessary:

* `detach()`.
* `clone()`.
* CPU transfers.
* NumPy conversions.
* Device synchronization.
* In-place operations.
* Mixed-precision casts.

## Training and Evaluation

Preserve existing behavior for:

* `model.train()`.
* `model.eval()`.
* Checkpoint save and resume.
* Optimizer state.
* Scheduler state.
* Random seeds.
* Distributed samplers.
* Metric computation.
* Inference preprocessing.
* Evaluation thresholds.

Do not add or enable the following unless requested or already required by the
repository:

* AMP.
* DistributedDataParallel.
* FullyShardedDataParallel.
* `torch.compile`.
* PyTorch Lightning.
* Hugging Face Trainer.
* A new experiment-management framework.
* A new configuration framework.

## Long-Running and Expensive Operations

Implementation approval does not automatically authorize expensive execution.

Obtain separate explicit approval before:

* Starting long-running training.
* Launching multi-GPU jobs.
* Reserving cluster resources.
* Downloading large datasets or checkpoints.
* Running a full-dataset evaluation.
* Generating a large cache.
* Overwriting experiment outputs.
* Uploading artifacts to external services.

Prefer a minimal CPU or single-batch smoke test first when feasible.

## Minimum Model Verification

For model-component changes, attempt the smallest relevant set of checks:

1. Import the modified module.
2. Instantiate it using an existing or minimal valid configuration.
3. Run one forward pass.
4. Verify output type and shape.
5. Verify device and dtype consistency.
6. Run one backward pass when training behavior changed.
7. Confirm train/eval behavior when relevant.
8. Run the repository's existing focused tests.

Report checks that could not be run and explain the concrete blocker.

---

# Git and File Safety

Do not execute these operations without explicit user approval:

* `git reset --hard`
* `git clean -fd`
* Force push.
* History rewriting.
* Branch deletion.
* Commit amendment.
* Mass file deletion.
* Dataset deletion.
* Checkpoint deletion.
* Destructive database migration.
* Replacing environment or lock files.
* Removing untracked user files.

Do not create a commit unless requested.

Do not push unless requested.

Do not revert modifications that were not made as part of the current approved
task.

When unexpected changes are found in a file that must be edited, stop and ask the
user how to proceed.

---

# Completion Report

The final response must contain:

```text
Implemented:
- ...

Files changed:
- ...

Verification performed:
- `<exact command>` — passed/failed
- ...

Acceptance criteria:
- ... — satisfied/not satisfied

Not verified:
- ...

Remaining risks:
- ...

Unrelated changes:
- None
```

Use exact commands and actual results.

Do not say:

* "Everything should work."
* "This is probably correct."
* "Tests should pass."
* "Done."

unless verification evidence supports the statement.

When a test cannot be run, state that it was not run. Do not describe it as
passing.

---

# Priority

When instructions conflict, follow this order:

1. The user's latest explicit instructions.
2. The latest explicitly approved requirements restatement.
3. The closest applicable `AGENTS.override.md`.
4. The closest applicable `AGENTS.md`.
5. Repository conventions.
6. Installed Skill defaults.

A later user correction invalidates conflicting earlier assumptions.

When a material conflict cannot be resolved safely, stop before modification and
ask one focused question.
