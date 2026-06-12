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
