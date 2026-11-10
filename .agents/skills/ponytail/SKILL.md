---
name: ponytail
description: Use when the user explicitly requests ponytail, lazy mode, a simplest-solution comparison, or an over-engineering review. Ordinary coding tasks use karpathy-guidelines without activating this optional strategy.
license: MIT
---

# Ponytail

Choose the smallest correct implementation of the approved request. Apply this
optional strategy to the requested task, not every later task in the conversation.

## The ladder

Stop at the first option that satisfies the requirements:

1. Check whether the requested capability already exists and can be reused.
2. Prefer the standard library or native platform feature when it fits.
3. Prefer an already installed dependency over adding one.
4. Otherwise write the minimum clear implementation in the owning layer.

Read the relevant flow and callers before choosing. For bugs, a small root-cause
fix is better than guards scattered across callers. Fewer lines are not a reason
to use a weaker algorithm, hide assumptions, or discard required behavior.

## Intensity

- lite: implement the request and mention a materially simpler alternative.
- full (default): use the ladder to choose the smallest sufficient implementation.
- ultra: challenge speculative additions before approval; execute the approved scope.

A different scope is a proposal, not permission to ship a reduced substitute.
Follow the project's existing confirmation, source-verification, testing, debugging,
and review requirements. Use existing pytest tests rather than adding a parallel
self-check framework. Preserve validation, error handling, research constraints,
and all explicitly requested behavior. Detailed requested reports and the required
completion format take precedence over brevity preferences.

State consequential tradeoffs briefly. Do not create commentary, scaffolding,
configuration knobs, or future extension points without a concrete need.
