"""T6R.2 — code-availability audit of the ECC volatility literature (the negative space).

Which of the papers in this lineage released code that can actually be run? The
table is the deliverable DECISIONS 2026-08-23 §5 asks for: one row per paper with
the artefact URL as published, a **live probe** (HTTP status, repository size,
number of code files, last push, licence — from the GitHub API at run time, with
the probe date recorded) and a **curated verdict** with the reason established in
the 2026-08-23 audit (JOURNAL) and re-checked here. Verdicts: `runnable`
(reproduced or reproducible with a stated port), `partial` (code present but the
environment or data is broken as shipped), `none` (no code, figures-only, empty
or dead artefact, paywalled).

Output: `results/code_availability.csv`. Papers without a public artefact carry
`url=""` and a probe status of `no_url`. The probe is network-bound; a missing
network yields status `unreachable` rather than a crash, so the curated columns
always regenerate.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

HEADERS = {"User-Agent": "ecvol-research/0.1 (andrei.roman.personal@gmail.com)"}


@dataclass(frozen=True)
class Paper:
    key: str
    paper: str
    venue: str
    repo: str  # "owner/name" on GitHub, or "" when none was published
    verdict: str  # runnable | partial | none
    reason: str
    reproduced_here: str  # ecvol command, or ""


PAPERS: tuple[Paper, ...] = (
    Paper(
        "qin2019",
        "Qin & Yang, What You Say and How You Say It Matters",
        "ACL 2019",
        "GeminiLn/EarningsCall_Dataset",
        "partial",
        "dataset released (Drive, 5-part zip); no model code in the repository",
        "",
    ),
    Paper(
        "html2020",
        "Yang et al., HTML",
        "WWW 2020",
        "YangLinyi/HTML-Hierarchical-Transformer-based-Multi-task-Learning-for-Volatility-Prediction",
        "partial",
        "model classes run verbatim; the released driver imports TensorFlow-1 set_random_seed "
        "and torchtext and splits features and labels with independent unseeded shuffles; "
        "inputs (WWM-BERT .npy, Praat features) behind dead Drive links — rebuilt; run with a "
        "corrected driver (T6R.3)",
        "ecvol reproduce html-faithful; ecvol reproduce html",
    ),
    Paper(
        "sawhney2020",
        "Sawhney et al., multimodal financial forecasting",
        "ACM MM 2020",
        "midas-research/multimodal-financial-forecasting",
        "partial",
        "TF 2.1 / Keras 2.3.1 / tensorflow-addons 0.8.3 pins; unshipped call inventory, price "
        "files and two lexicons; text-feature script returns after one record; no multi-task "
        "loss in the released code; ported line by line to PyTorch (T6R.3)",
        "ecvol reproduce sawhney (port)",
    ),
    Paper(
        "voltage2020",
        "Sawhney et al., VolTAGE",
        "EMNLP 2020",
        "piyushkhanna00705/VolTAGE",
        "partial",
        "ships the EC split/label files (byte-identical to KeFVP's) and GCN code; "
        "embeddings as .pkl; not ported",
        "",
    ),
    Paper(
        "kefvp2023",
        "Niu et al., KeFVP",
        "EMNLP Findings 2023",
        "hankniu01/KeFVP",
        "partial",
        "requirements.txt does not install as written; the script starts only after six "
        "import-level repairs (missing set_seed, package init importing undefined names, a "
        "dropped class header, an unreleased `latent` package, plotting imports); MAEC "
        "embeddings never released (regenerated with the authors' generator); EC KePt pickle "
        "Drive-only (T6R.3)",
        "ecvol reproduce kefvp",
    ),
    Paper(
        "dialoguegat2023",
        "DialogueGAT",
        "Findings of EMNLP 2022",
        "sangyx/DialogueGAT",
        "partial",
        "MIT code, tau<=15 only; training pickle, SeekingAlpha corpus and CRSP labels never "
        "released; DGL has no wheels for torch 2.11; ported (PyG) onto FinCall turns (T6R.3)",
        "ecvol reproduce dialoguegat (port)",
    ),
    Paper(
        "scss2025",
        "Yu, Liu & He, Same Company, Same Signal",
        "Findings of ACL 2025",
        "piqueyd/Same-Company-Same-Signal",
        "runnable",
        "MIT; ships DEC with features and rolling masks; PEV/STPEV to 3 decimals, TSMixer "
        "76 cells to 1e-5, TMLP 228 cells within single-seed noise on the authors' embeddings",
        "ecvol reproduce scss; scss-tsmixer; scss-tmlp",
    ),
    Paper(
        "fintrust2023",
        "Yang et al., FinTrust",
        "ACL 2023 short",
        "yingpengma/FinTrust",
        "partial",
        "perturbation harness + 3/7/15/30 data; complementary control axis, not a volatility model",
        "",
    ),
    Paper(
        "soundofrisk2025",
        "Sound of Risk",
        "2025",
        "soundai2016/sound_risk",
        "none",
        "repository holds figures only, zero source files; proprietary 1,795-call corpus",
        "",
    ),
    Paper(
        "defvp2024",
        "DeFVP",
        "2024",
        "hankniu01/DeFVP",
        "none",
        "repository exists but is empty (size 0 KB, never pushed; API returns 409)",
        "",
    ),
    Paper(
        "echogl2024",
        "ECHO-GL",
        "2024",
        "pupu0302/ECHOGL",
        "none",
        "README states the code 'can not run at present'; target is direction, not volatility",
        "",
    ),
    Paper(
        "eccanalyzer2024",
        "ECC Analyzer",
        "2024",
        "",
        "none",
        "no code released; GPT-4 Turbo + OpenAI embeddings",
        "",
    ),
    Paper("risklabs2024", "RiskLabs", "2024", "", "none", "no code released; GPT-4/3.5", ""),
    Paper(
        "numhtml2022", "NumHTML", "AAAI 2022", "", "none", "paper promises code; no URL given", ""
    ),
    Paper("gnavol", "GNA-Vol", "", "", "none", "artefact URL returns 404 (checked 2026-08-23)", ""),
    Paper("amalstm", "AMA-LSTM", "", "", "none", "10-byte stub repository, unattributed", ""),
    Paper("atfingpt", "AT-FinGPT", "", "", "none", "paywalled; no artefact", ""),
)


def _get(url: str, timeout: int = 30) -> dict:
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=timeout) as r:
        return json.load(r)


def probe(repo: str) -> dict:
    """Live GitHub probe: status, size_kb, files, code_files, pushed_at, license."""
    if not repo:
        return {"status": "no_url"}
    try:
        meta = _get(f"https://api.github.com/repos/{repo}")
        tree = _get(f"https://api.github.com/repos/{repo}/git/trees/HEAD?recursive=1")
        blobs = [e for e in tree.get("tree", []) if e["type"] == "blob"]
        return {
            "status": "200",
            "size_kb": int(meta.get("size", 0)),
            "files": len(blobs),
            "code_files": sum(1 for e in blobs if e["path"].endswith((".py", ".ipynb", ".sh"))),
            "pushed_at": str(meta.get("pushed_at", ""))[:10],
            "license": (meta.get("license") or {}).get("spdx_id") or "",
        }
    except urllib.error.HTTPError as e:
        return {"status": str(e.code)}
    except Exception as e:  # noqa: BLE001 — network failure must not kill the audit
        return {"status": f"unreachable:{type(e).__name__}"}


def run_code_audit(root: Path, *, live: bool = True) -> pd.DataFrame:
    probed_on = datetime.now(UTC).date().isoformat() if live else ""
    rows = []
    for p in PAPERS:
        info = probe(p.repo) if live else {"status": "not_probed"}
        rows.append(
            {
                "key": p.key,
                "paper": p.paper,
                "venue": p.venue,
                "url": f"https://github.com/{p.repo}" if p.repo else "",
                "verdict": p.verdict,
                "reason": p.reason,
                "reproduced_here": p.reproduced_here,
                "probe_status": info.get("status", ""),
                "size_kb": info.get("size_kb", ""),
                "files": info.get("files", ""),
                "code_files": info.get("code_files", ""),
                "pushed_at": info.get("pushed_at", ""),
                "license": info.get("license", ""),
                "probed_on": probed_on if p.repo else "",
            }
        )
    table = pd.DataFrame(rows)
    out = root / "results"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "code_availability.csv", index=False, lineterminator="\n")
    return table
