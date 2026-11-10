---
name: code-review-and-quality
description: Use after implementation and before completion or merging, or when reviewing a specified change by an agent or human.
---

# Code Review and Quality

Review the final diff against the approved requirements and actual verification
evidence. Cover the resulting behavior and relevant callers, not just changed lines.

## Review once with shared evidence

Read the approved scope, diff, relevant tests, and verification results. Reuse an
existing relevant review or test result while its inputs remain unchanged; review
subsequent changes rather than restarting all earlier work. Independent reviewers
can use this same checklist; they do not require a second requirements gate.

Check these axes:

- Correctness: required behavior, edge cases, errors, interfaces, and test coverage.
- Readability and scope: clear control flow, existing style, surgical changes,
  no speculative abstraction or unrelated cleanup.
- Architecture: compatible callers, configuration, artifacts, dependency direction,
  state ownership, and checkpoint save/resume behavior where affected.
- Security: no leaked secrets, untrusted inputs treated as data, and validation at
  relevant boundaries. Check dependency changes only when they are in scope.
- Performance: unbounded work, retained tensors, copies, CPU transfers, device
  synchronization, and unnecessary allocation where relevant. Require measurements
  before claiming a speed or memory improvement.

## SemTrace and skill changes

For affected model/data code check shape, broadcasting, device, dtype, gradients,
train/eval behavior, frozen DINOv3 and predictors, and stage order. Check real=0,
fake=1, production classifier trace_evidence-only input, candidate_trace_residual
terminology, crop/no-resize protocol, metrics, and configured paths as applicable.

For skill changes check name/description, entrypoints, real symlink targets, owned
supporting files, references, lock consistency, trigger boundaries, preserved
capabilities, and conflicts with AGENTS.md. Scenario evaluations do not establish
real-world trigger accuracy, token savings, or runtime efficiency.

## Findings and completion

Lead with actionable defects and supporting file/line evidence. Distinguish
Critical/Required fixes from Optional/Nit suggestions; do not block on preference.
Verify exact commands and results instead of assuming tests, builds, or imports ran.
List blockers and unverified criteria explicitly; do not rubber-stamp a green suite.

Remove a proved exclusive orphan only when its deletion is already authorized.
Preserve ambiguous or unrelated files and report them. Do not ask again for deletion
already covered by approval, or silently expand scope to fix neighboring issues.
No automatic commits, bug reports, new artifacts, or extra reviewers follow from
this checklist. Use separate review tooling when required by applicable instructions.
