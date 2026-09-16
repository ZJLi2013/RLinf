# Review conventions observed on RLinf PRs

**Fork-local notes. Keep this file out of upstream PR branches.**

`AGENTS.md`, `CONTRIBUTING.md` and `docs/STYLE_GUIDE.md` already state the written rules — Google
style, Ruff (line length 88, double quotes, google pydocstyle), Conventional Commits with
`Signed-off-by`, logging over `print`, static YAML. This file records only what reviewers enforced
that those documents do not say, or say differently, with the evidence.

---

## Tests go into the existing thematic files, not new per-module files

`CONTRIBUTING.md` says every user-facing change needs tests, and `AGENTS.md` repeats it. A reviewer
nonetheless asked for **all** unit tests added by [#1518](https://github.com/RLinf/RLinf/pull/1518)
to be deleted — three new files, 665 lines.

The objection was not to the technique and not to testing env code:

- `tests/unit_tests/` holds 21 files for the whole repository, and env coverage is concentrated in
  two large thematic ones: `test_real_env.py` (3410 lines) and `test_robotics.py` (4536 lines).
- Five upstream files already use the same `importlib.util.spec_from_file_location` plus
  `monkeypatch.setitem(sys.modules, ...)` stubbing that the deleted tests used.

So the shape the repository accepts is **a few large files organised by theme**, and a PR that adds
`test_<new_module>.py` per module is reading against that grain. Add cases to the thematic file that
already owns the area instead.

The reviewer gave no reason, so this is inference from the repository's own layout. If tests are
asked for again, propose the target file before writing them.

## No absolute paths anywhere, including comments you did not write

A `# PYTHONPATH="/mnt/project_rlinf/jzn/workspace/..."` line predating the PR was flagged in a file
the PR rewrote. Touching a file makes its existing content yours to clean.

## No `if __name__ == "__main__"` blocks in library modules

Both world-model env modules carried a smoke-test driver under `__main__`. The reviewer's ask:
either extract it to `examples/` **with usage steps in the docs**, or it does not belong. A debug
entry point with no documented path is the case that gets removed.

## One representation per concept in a signature

`__init__(self, cfg, device: torch.device, device_str: str)` was flagged: two spellings of the same
thing. Resolve the platform's device type once and pass a single `torch.device`; derive the string
where a third-party API needs one (`str(self.device)` for diffsynth).

## `LiberoEnv` is the reference for env semantics

Deviating from its ordering draws a question. In `chunk_step`, metrics are recorded **before**
auto-reset, because `reset()` zeroes the per-slot accumulators — reading `elapsed_steps` afterwards
reports a restarted slot's zero as the finished episode's length. `LiberoEnv.step` does
`_record_metrics` then `_handle_auto_reset`; the world-model env had them reversed.

When a new env diverges from LIBERO on ordering, lifecycle, or the `infos` contract
(`final_info` / `final_observation` / `_elapsed_steps`), say why in the PR body before review asks.

## Iterate parallel collections from one length source

Copilot flagged `for frames in condition` next to `env_ids`-keyed session lookups: two collections
assumed to be the same length with nothing enforcing it. Index both from the same source, or assert
the lengths match.

---

## Mechanics that cost a round trip if missed

- **Every commit needs `Signed-off-by`.** `git commit -s`, and the pre-commit `commit-msg` hook
  checks it: `pre-commit install --hook-type commit-msg`.
- **Two approving reviews** are required, and code owners are auto-requested.
- **CI installs `--env dummy`** for the unit-test job (`.github/workflows/unit-tests.yml`) and runs
  every `tests/unit_tests/test_*.py` one file at a time. Anything that imports a heavy env
  dependency at module scope fails there even if it passes in a model venv.
- **On Windows checkouts, never `git add -A` in this repo.** `core.autocrlf=true` with no
  `.gitattributes` re-normalizes the whole tree — one commit picked up 2110 files. List the files.
