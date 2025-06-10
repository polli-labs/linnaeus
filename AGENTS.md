# AGENTS.md — Operating Rules for Codex

## 1  Virtual environment

* **Always** activate the project venv before running any tooling:

  ```bash
  source .venv/bin/activate            # or call binaries via .venv/bin/<tool>
  ```

## 2  Testing

* Execute the test‑suite only with:

  ```bash
  .venv/bin/python -m pytest -q
  ```

## 3  Linting & Formatting

Codex must **clean up all auto‑fixable Ruff findings** and ensure the remaining
style issues are advisory only (job should still succeed).

| Purpose             | Mandatory Command                          |
| ------------------- | ------------------------------------------ |
| Format the code     | `.venv/bin/ruff format .`                  |
| Lint **with fixes** | `.venv/bin/ruff check --fix --exit-zero .` |
| Final verification  | `.venv/bin/ruff check --exit-zero .`       |

* The first two commands may modify files in‑place; commit those changes.
* The final verification must run **after** fixes are applied and **before**
  committing, guaranteeing a clean exit status (0).

## 4  Commit policy

1. Activate the venv.
2. Run formatter, linter‑with‑fix, and final verification (§3).
<!-- 3. Run the test‑suite (§2). Commit only if tests pass. --> NOTE: this policy is not yet active as tests remain unstable, use your judgement and only consider relevant tests when determining whether to commit.