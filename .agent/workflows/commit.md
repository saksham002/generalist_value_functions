---
description: Commit all changes as small, meaningful commits
---

# Commit Changes Workflow

When asked to commit changes, follow these steps to create clean, atomic commits:

## 1. Check Status
// turbo
```bash
git status
```

## 2. Run Ruff Checks
// turbo
```bash
uv run ruff check .
```

If ruff reports any errors, fix them before proceeding. Use `ruff check --fix .` for auto-fixable issues.

## 3. Review All Diffs
// turbo
```bash
git diff
```

Review the diffs to understand all changes and group them logically by:
- Feature/functionality (e.g., "add new config option", "refactor class X")
- File relationships (files that are modified together for the same purpose)
- Independence (changes that can stand alone)

## 4. Create Atomic Commits

For each logical group of changes:

1. Stage only the related files:
```bash
git add <file1> <file2> ...
```

2. Commit with a meaningful message following conventional commit style:
```bash
git commit -m "Short summary (50 chars or less)

Optional longer description explaining:
- What changed
- Why it changed
- Any important details"
```

### Commit Message Guidelines:
- First line: concise summary in imperative mood (e.g., "Add X", "Fix Y", "Refactor Z")
- Keep first line under 50 characters if possible
- Add blank line before extended description if needed
- Use bullet points for multiple changes in the description

### Grouping Heuristics:
- **Same feature**: Files modified for the same feature go together
- **Tests with implementation**: Test files commit with their implementation
- **Config changes**: Separate from code changes unless tightly coupled
- **Refactoring**: Separate from new features
- **Infrastructure**: Build/tooling changes separate from application code
- **Small fixes together**: Don't make individual commits for small fixes (lint fixes, docstring updates, etc.) - group them into meaningful commits

## 5. Verify Commits
// turbo
```bash
git log --oneline -n 10
git status
```

Ensure all changes are committed and the log shows meaningful, atomic commits.

## 6. Push Commits

Show the user the commits that will be pushed:
// turbo
```bash
git log --oneline origin/HEAD..HEAD
```

Ask the user for permission to push, displaying the commit messages. Only push after explicit approval:
```bash
git push
```