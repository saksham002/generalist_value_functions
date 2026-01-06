---
description: Remove AI-generated code slop from the current branch
---

// turbo-all

# Remove AI Code Slop

Check the diff against main and remove all AI-generated slop introduced in this branch.

## Steps

1. Get the diff against main branch:
```bash
git diff main --name-only
```

2. For each changed file, view the full diff to understand what was added:
```bash
git diff main -- <file>
```

3. Review each file and remove the following types of AI slop:
   - **Unnecessary comments**: Extra comments that a human wouldn't add or that are inconsistent with the rest of the file's commenting style
    - **Defensive over-engineering**: Extra defensive checks, or abnormal error handling (especially if called by trusted/validated codepaths)
    - **Try/Except Blocks**: Almost all `try/except` blocks are slop, especially `try/except ImportError`. They should only be used if strictly necessary.
    - **Type escape hatches**: Casts to `any` (TypeScript) or `# type: ignore` (Python) added to work around type issues instead of fixing them properly
   - **Style inconsistencies**: Any other code style that is inconsistent with the surrounding file

4. For each issue found:
   - Check how similar code is handled elsewhere in the same file
   - Remove or simplify the slop to match the existing codebase style
   - Make the minimal change needed

5. After all changes, provide a 1-3 sentence summary of what was changed.
