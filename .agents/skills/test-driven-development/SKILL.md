---
name: test-driven-development
description: Use when adding behavior, fixing bugs, changing public interfaces, processing data, or modifying model behavior that can be tested deterministically.
---

# Test-Driven Development

Prove the requested behavior with a focused test before implementing it. Follow
AGENTS.md for scope, approval, required checks, and expensive-operation boundaries.

## Discover and run the existing tests

Inspect pyproject.toml, nearby tests, and documented commands. Use the existing
pytest infrastructure and conventions. Start with the smallest relevant case;
expand to required checks after the implementation is stable. Pure prose or static
content edits need relevant structural checks rather than artificial model tests.
Instruction changes also need representative scenario evaluation; distinguish
simulated responses from real execution and measured user outcomes.

## RED / GREEN / REFACTOR

1. Write a test for the expected observable result. For a bug, reproduce the
   reported failure before fixing it whenever practical. Run the test and verify
   that it fails for the intended reason, not an import or environment accident.
2. Implement the smallest approved change that passes. Run the focused test;
   diagnose an unexpected failure before attempting another change.
3. Refactor only where necessary for the approved change, then rerun affected
   tests. Do not turn a green test into permission for adjacent cleanup.

Use deterministic synthetic inputs by default. Assert outcomes rather than private
call sequences. Prefer real small components; use test doubles at boundaries where
real execution is nondeterministic, costly, or has uncontrolled effects. Isolate
test state and seeds. A regression test should fail if the relevant bug returns.

## SemTrace coverage

For affected model or data behavior, cover relevant shapes, device, dtype,
gradient flow, train/eval state, and empty or partial batches. Preserve real=0,
fake=1, crop behavior, frozen components, stage-artifact requirements, and metrics.
Use the minimum forward/backward smoke checks applicable to the change; do not
start full training, weight downloads, or full-dataset evaluation as a test default.

## Verification evidence

Run the repository's required checks, subject to the user's authorized environment
and resource boundaries. Record exact commands, outcomes, skips, and blockers.
Reuse a passing result only when no relevant code, configuration, dependency,
input, or environment has changed since that run. Do not repeat the same command
on an unchanged state just because another skill asks for verification.

A passing focused case does not imply a passing full suite. Do not disable tests,
suppress failures, or claim an unrun check passed. If reproduction is impractical,
state why and what evidence the alternative verification actually establishes.
