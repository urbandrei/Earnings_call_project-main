"""T6R.3 H3/W3: Praat voice-report parsing, clip ordering, sentence alignment."""

import numpy as np
import pandas as pd

from ecvol.features.audio import praat as P

REPORT = """-- Voice report for 2 selected objects --
Pitch:
   Median pitch: 120.5 Hz
   Mean pitch: 118.2 Hz
Pulses:
   Number of pulses: 42
   Number of periods: 40
   Mean period: 8.3E-3 seconds
Voicing:
   Fraction of locally unvoiced frames: 12.5%   (3 / 24)
   Number of voice breaks: 1
Jitter:
   Jitter (local): 1.234%
Harmonicity of the voiced parts only:
   Mean harmonics-to-noise ratio: 15.7 dB
"""


def test_report_parser_reads_numbers_percentages_and_exponents():
    assert P._report(REPORT, "Number of pulses") == 42
    assert P._report(REPORT, "Mean period") == 8.3e-3
    assert P._report(REPORT, "Fraction of locally unvoiced frames") == 12.5
    assert P._report(REPORT, "Jitter (local)") == 1.234
    assert P._report(REPORT, "Mean harmonics-to-noise ratio") == 15.7
    assert np.isnan(P._report(REPORT, "Shimmer (apq3)"))


def test_clip_order_is_numeric_by_turn_then_sentence():
    names = ["A B_10_2.mp3", "A B_2_10.mp3", "A B_2_9.mp3", "junk.mp3"]
    assert sorted(names, key=P.clip_order) == [
        "A B_2_9.mp3",
        "A B_2_10.mp3",
        "A B_10_2.mp3",
        "junk.mp3",
    ]


def test_sentence_audio_aligns_by_position(tmp_path):
    root = tmp_path
    (root / "ec" / "cache").mkdir(parents=True)
    rows = [{"call_id": "c", "sent_idx": i, **{f: float(i) for f in P.FEATURES}} for i in range(3)]
    pd.DataFrame(rows).to_parquet(root / "ec" / "cache" / P.CACHE_FILE, index=False)
    out = P.sentence_audio(root, {"c": 4, "other": 2})
    a = out["c"]
    assert a.shape == (4, 27) and a[2, 0] == 2.0 and np.isnan(a[3]).all()
    assert "other" not in out and len(P.FEATURES) == 27
