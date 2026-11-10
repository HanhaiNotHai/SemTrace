---
name: source-driven-development
description: Use before relying on third-party APIs, framework behavior, command-line options, file formats, configuration schemas, or version-specific features.
---

# Source-Driven Development

Ground version-sensitive decisions in primary evidence for the project's actual
stack. Preserve existing interfaces and conventions within the approved scope.

## Verify what matters

1. Inspect dependency manifests, lock files, installed versions, and existing
   repository usage. Distinguish a version range from the installed version.
   Read relevant installed source or command help when it can establish behavior.
2. Consult the matching version's official API documentation, source code,
   specification, or changelog for the particular behavior in question. Fetch
   relevant pages, not entire manuals. Tutorials and model memory are not primary
   evidence. Repository usage shows convention; it does not alone prove API validity.
3. Record which version and source support the decision. Reuse evidence already
   verified in this task while the version, source, and relevant behavior remain
   unchanged. Refresh evidence when those inputs change or remain uncertain.
4. Implement within the approved scope, then verify using the project's focused
   test or minimal relevant check. Documentation lookup does not replace testing.

Inspect available evidence before asking the user to identify a version. If a
material fact cannot be established, state the unknown and resolve it before
relying on that behavior; do not present an assumption as verified.

## Resolve conflicts without expanding scope

New documentation does not authorize upgrading dependencies, migrating an existing
pattern, or changing an interface. First check whether the documentation matches
the installed version. If valid options remain, prefer the smallest compatible
change. Ask only when a material unresolved choice requires the user's decision.
When primary sources disagree, explain the discrepancy and test the relevant
behavior against the installed version when feasible.

Cite consequential version-specific decisions using a direct official link or an
installed-source path and version. Consolidate repeated citations in the task
report; avoid repeating a source comment at every call site. Explicitly identify
facts that could not be verified. Speed requests do not waive required verification.

Example: a newer framework's recommended API is not a reason to replace a working
API in the project's installed version during a small compatible change.
