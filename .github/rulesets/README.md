# Branch Rulesets

Versioned source of truth for this repository's GitHub branch rulesets.
Each JSON file is the exact payload of the
[repository rulesets REST API](https://docs.github.com/en/rest/repos/rules),
so files round-trip cleanly between git and GitHub.

| File                  | Purpose                                            | Bypass          |
| --------------------- | -------------------------------------------------- | --------------- |
| `main-immutable.json` | Block force pushes and deletion of `main`          | Nobody          |
| `main-pr-gate.json`   | Require PR, passing CI, linear history on `main`   | Repo admins     |

The split is deliberate: admins bypass the PR/CI gate (release commits are
pushed directly to `main`), but nobody bypasses history protection.

## Apply a new ruleset

```bash
gh api -X POST repos/gao-hongnan/omniadapters/rulesets \
  --input .github/rulesets/main-pr-gate.json
```

## Update an existing ruleset

Look up the ruleset id, then `PUT` the edited file:

```bash
gh api repos/gao-hongnan/omniadapters/rulesets --jq '.[] | {id, name}'
gh api -X PUT repos/gao-hongnan/omniadapters/rulesets/<id> \
  --input .github/rulesets/main-pr-gate.json
```

## Verify what is enforced on `main`

```bash
gh api repos/gao-hongnan/omniadapters/rules/branches/main
```

## Maintenance notes

- The required status check name is matrix-derived
  (`CI Python ${{ matrix.python-version }} on ${{ matrix.os }}`). When the CI
  matrix changes (for example a Python version bump), update the `context` in
  `main-pr-gate.json` and `PUT` it, or PRs will block on a check that never
  runs.
- `integration_id: 15368` pins the required check to the GitHub Actions app so
  another app cannot satisfy it with a same-named check.
- Edits made in the GitHub UI drift from these files. Re-export from
  Settings -> Rules -> Rulesets (or `gh api .../rulesets/<id>`) back into this
  directory, or treat these files as the only place edits happen.
