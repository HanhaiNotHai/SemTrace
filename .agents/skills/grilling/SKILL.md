---
name: grilling
description: Use when a material requirement, constraint, interface, success criterion, or scope boundary remains unclear before implementation, or when the user explicitly requests a plan or decision to be questioned.
metadata:
  upstream: https://github.com/mattpocock/skills/blob/main/skills/productivity/grilling/SKILL.md
  intent-source: https://github.com/addyosmani/agent-skills/blob/main/skills/interview-me/SKILL.md
  adaptation: SemTrace local skill replacing the separate intent interview workflow.
---

# Grilling

Resolve decisions that affect implementation or research validity. Use the
AGENTS.md requirements-confirmation gate as the single approval step.

## Decide whether to interview

Read the relevant repository context first. Verify facts available in code,
configuration, tests, installed versions, and existing artifacts yourself.
Ask the user about choices or information that the environment cannot supply.
A complete request proceeds directly to the AGENTS.md requirements restatement.
An unchanged, already approved specification proceeds to execution without another
interview or approval. Read-only explanations need no implementation gate.

When the desired outcome is unclear, distinguish the proposed solution from the
underlying goal. Ask a concrete question about the missing success measure or
binding constraint; do not reopen goals that are already established.

## Work in dependency order

Identify only unresolved decisions that can change scope, interfaces,
implementation, research validity, compatibility, or acceptance. Resolve upstream
decisions before asking questions that depend on their answers.

Ask up to three independent, focused questions per round. Each question states a
recommended answer and its reason, with uncertainty explicit. Use one question
when only one decision is ready, or its answer determines the next question.
When evidence is insufficient for a recommendation, say so and give concrete
alternatives. Allow meaningful disagreement rather than steering every answer
toward agreement.

Use the Current understanding / Question(s) / Best guess format in AGENTS.md.
Wait for answers before the next round; update remaining decisions from those
answers. Prefer direct inspection for facts. Use a subagent only when a separate
investigation meaningfully helps and delegation is authorized.

## Stop and hand off

Stop when no material uncertainty blocks implementation or acceptance. Do not
exhaust speculative design branches or assign confidence percentages. When answers
leave a decision unresolved, identify the specific blocker instead of repeating
the same question. Explicitly deferred decisions remain unresolved, not approved.

Present the AGENTS.md requirements restatement once and wait for explicit approval.
A material correction requires an updated restatement; unchanged approval carries
forward. Interview answers alone do not authorize implementation. Do not add a
separate intent approval, automatic checkpoint file, or mandatory downstream skill.

Example: checkpoint compatibility and CPU smoke-test requirements can share a
round if neither is already established. An interface choice that depends on
compatibility belongs after that answer.
