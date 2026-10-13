# ecvol

An identity-controlled evaluation protocol for earnings-call volatility. Code, decision log, checksums and reproduction ledger for the paper. This branch is the
review snapshot; it is the working repository with the course-era material, scratch
scripts and cloud runbooks removed and the contact address withheld.

## Where to look

| What the paper cites | Where it is |
|---|---|
| decision log (dated entries, one per deviation, git-timestamped on the full history) | [DECISIONS.md](DECISIONS.md) |
| research design the log deviates from | [DESIGN.md](DESIGN.md) |
| working journal, including negative results | [JOURNAL.md](JOURNAL.md) |
| command behind every table, and the data each needs | [REPRODUCE.md](REPRODUCE.md) |
| reproduction ledger (what ran as released, what had to be repaired) | [docs/faithful_ledger.md](docs/faithful_ledger.md) |
| audit command for a prediction file, with a worked example | [docs/audit_predictions.md](docs/audit_predictions.md) |
| result CSVs and their run manifests | `data/results/`, `artifacts/runs/<run_id>/run.json` |
| SHA-256 manifests of every data file | `data/manifests/` |
| code-availability audit of the surveyed papers | `data/results/code_availability.csv` |
| paper source, tables and figure generators | `paper/` |

## What runs without the data

The corpora are not redistributed (see [LICENSE-DATA.md](LICENSE-DATA.md)), so a fresh
clone cannot re-extract features. These do run on the clone alone:

```sh
uv sync --locked
uv run pytest -q                      # test suite, including the leakage and target checks
uv run ecvol runs verify              # every committed result CSV matches its run manifest
uv run ecvol audit predictions data/audits/demo/scss_tmlp_anchor_heldout_2022.csv
uv run ecvol audit code               # live probe of the surveyed repositories
```

The audit command reproduces the worked example in `docs/audit_predictions.md` from the
committed prediction frame. The release archives the quickstart below needs are not on a
public host yet; they are built by `ecvol release build` from the local data.

## Install (fresh machine)

Requires [uv](https://docs.astral.sh/uv/) (no separate Python install needed —
uv provisions the pinned interpreter from `.python-version`).

```sh
git clone <this repository> ecvol && cd ecvol
uv sync --locked          # creates .venv from the committed lockfile
uv run ecvol --help       # lists all pipeline verbs
uv run pytest -q          # test suite
uv run pre-commit install # enable lint/format hooks for commits
```

`uv sync --locked` fails if `uv.lock` is out of date with `pyproject.toml`,
guaranteeing the environment matches the committed lockfile exactly.

## Quickstart for researchers (benchmark users)

```sh
uv run ecvol release build                                   # or download the archives from the data host
uv run python examples/quickstart.py --archive data/release/ecvol-bench_fincall_<sha>.zip
```

The second command loads the released targets, splits and pooled features straight from
the archive, fits a past-volatility ridge and a ridge with the text embedding added on the
temporal and ticker-disjoint splits, prints R²_OOS against persistence, and writes a
prediction frame per variant that it runs through `ecvol audit predictions` (the identity
checks from the paper). Replace the ridge with your model and keep the rest; see
[REPRODUCE.md](REPRODUCE.md) and [docs/audit_predictions.md](docs/audit_predictions.md).
