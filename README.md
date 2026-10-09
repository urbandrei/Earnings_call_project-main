# ecvol — earnings-call volatility prediction

Research project predicting post-earnings-call stock volatility from
multimodal data (transcript text + call audio). See [DESIGN.md](DESIGN.md)
for the research design and [CLAUDE.md](CLAUDE.md) for the document map.

## Install (fresh machine)

Requires [uv](https://docs.astral.sh/uv/) (no separate Python install needed —
uv provisions the pinned interpreter from `.python-version`).

```sh
git clone <repo-url> ecvol && cd ecvol
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
