# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- **`GLOSSARY.md`** at the repo root. Its intro links the shared Qilimanjaro glossary for terms used across the stack.
- **`docs/adr/`**: read ADRs that touch the area you're about to work in.
- The user documentation under `docs/modules/` (one page per module) and `docs/tutorials/` explains most of the SDK's terms in more depth than the glossary.

If any of these files don't exist or are empty, **proceed silently**. Don't flag their absence. The `/domain-modeling` skill (reached via `/grill-with-docs` and `/improve-codebase-architecture`) creates entries lazily when terms or decisions actually get resolved.

## File structure

Single-context repo:

```
/
├── GLOSSARY.md
├── docs/adr/
└── src/qilisdk/
```

## Use the glossary's vocabulary

When your output names a domain concept (an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in `GLOSSARY.md`. Don't drift to synonyms the glossary explicitly avoids.

If the concept you need isn't in the glossary yet, that's a signal: either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `/domain-modeling`).

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-NNNN (the decision's title), but worth reopening because…_
