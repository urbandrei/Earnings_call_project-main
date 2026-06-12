"""Identity reconstruction logic: name matching, scoring, dates, call types (T1.4).

No network and no real PDFs: load_sec_table/_pdf_signals are exercised against
fixtures; the corpus-scale run is validated by the audited identity CSV.
"""

from ecvol.data.fincall_identity import (
    classify_call,
    clean_candidate,
    extract_date,
    norm_name,
    phrase_mentions,
    resolve_company,
)

SEC = {
    "stanley black decker": ("SWK", "93556"),
    "international paper": ("IP", "51434"),
    "citigroup": ("C", "831001"),
    "target": ("TGT", "27419"),
    "vulcan materials": ("VMC", "1396009"),
}


def test_norm_name_strips_legal_suffixes_and_joins_apostrophes():
    assert norm_name("Stanley Black & Decker, Inc.") == "stanley black decker"
    assert norm_name("Kohl's Corporation") == "kohls"  # matches SEC's "KOHLS CORP"
    assert norm_name("The Vulcan Materials Company") == "vulcan materials"


def test_clean_candidate_drops_event_tail():
    assert clean_candidate("Vulcan Materials Company First Quarter Earnings") == (
        "Vulcan Materials Company"
    )


def test_phrase_mentions_finds_multiword_names():
    text = (
        "Operator: Welcome to the Stanley Black & Decker Fourth Quarter Call. "
        "Stanley Black & Decker issued a press release. Thanks for joining "
        "Stanley Black & Decker's call."
    )
    counts = phrase_mentions(text, SEC)
    assert counts["93556"] >= 2  # keyed by CIK


def test_phrase_mentions_discounts_generic_single_words():
    # One mention of a short generic name ("Target") must not count as full evidence.
    counts = phrase_mentions("Our Target audience grew.", SEC)
    assert counts["27419"] < 1


def test_resolve_company_dominant_match():
    text = (
        "Operator: Welcome to the International Paper First Quarter 2020 Earnings Call. "
        "International Paper reported results. International Paper's CEO spoke."
    )
    ticker, cik, score, _, _, flags = resolve_company(text, None, SEC)
    assert (ticker, cik) == ("IP", "51434")
    assert score >= 3


def test_resolve_company_greeting_dominance_breaks_body_ties():
    # Body text mentions a related company more often than the issuer (GE's
    # call discussing Baker Hughes), but only the issuer has greeting evidence.
    text = (
        "Operator: Welcome to the International Paper First Quarter 2020 Earnings Call. "
        + "Vulcan Materials results. " * 14
    )
    ticker, _, _, runner, _, flags = resolve_company(text, None, SEC)
    assert ticker == "IP"
    assert runner == "VMC"
    assert "greeting_dominant" in flags


def test_resolve_company_ambiguous_is_unresolved():
    text = (
        "International Paper and Vulcan Materials announced a venture. International Paper said. "
        "Vulcan Materials said. International Paper. Vulcan Materials."
    )
    ticker, _, _, _, _, flags = resolve_company(text, None, SEC)
    assert ticker == ""
    assert any(f.startswith("unresolved") for f in flags)


def test_resolve_company_pdf_metadata_boost():
    ticker, _, score, _, _, flags = resolve_company(
        "Welcome to the call. International Paper. ", "International Paper Co", SEC
    )
    assert ticker == "IP"
    assert "match:pdf_company" in flags


def test_resolve_company_no_candidates():
    ticker, _, _, _, _, flags = resolve_company("hello world, nothing here.", None, SEC)
    assert ticker == ""
    assert flags == ["unresolved:no_candidates"]


def test_extract_date_prefers_page1_and_flags_disagreement():
    date, source, flags = extract_date("Results Review  January 16, 2019", "2019-01-15")
    assert (date, source, flags) == ("2019-01-16", "pdf_page1", [])
    _, _, flags = extract_date("January 16, 2019", "2018-10-01")
    assert "date_disagreement" in flags


def test_extract_date_fallbacks():
    assert extract_date(None, "2020-02-12") == (
        "2020-02-12",
        "pdf_created",
        ["date_from_creation_stamp"],
    )
    assert extract_date(None, None) == ("", "", ["no_date"])


def test_classify_call_types():
    assert classify_call("Welcome to the Q4 2018 Earnings Conference Call", None) == "earnings"
    assert classify_call("Welcome to the Ford Monthly Sales Conference Call", None) == "sales"
    assert classify_call("call to discuss the combination of A and B", None) == "ma"
    assert classify_call("I cover hardware at JPMorgan, welcome to our Tech Forum", None) == (
        "conference"
    )
    assert classify_call("Good morning operator.", None) == "unknown"
    # Strong conference markers outrank earnings keywords: a fireside chat
    # quoting "Q4 earnings" is still a conference session.
    assert classify_call("a fireside chat about Q4 earnings with the CEO", None) == "conference"
    assert (
        classify_call(
            "My name is Cory Kasimov. I'm the senior large-cap biotech analyst, and "
            "it's my pleasure to introduce our next company. Q4 earnings were strong.",
            None,
        )
        == "conference"
    )
    assert classify_call("Welcome to the Annual Meeting of Shareholders", None) == "meeting"
    assert classify_call("the Kellogg Company's 2021 Annual Shareholders Meeting", None) == (
        "meeting"
    )
    # ...but ordinary earnings-call phrasing must not trip the conference rule.
    assert classify_call("welcome to the Q3 investor conference call", None) == "earnings"
    assert (
        classify_call(
            "welcome to the Q1 Earnings Call. Adrian will be participating in a "
            "fireside chat at Cowen's retail conference next week.",
            None,
        )
        == "earnings"
    )


def test_norm_name_strips_edgar_state_tags():
    assert norm_name("AMERICAN TOWER CORP /MA/") == "american tower"


def test_resolve_company_share_classes_merge():
    # Two share classes of one company (same CIK) must not trip the ambiguity guard.
    sec = {
        "fox a": ("FOXA", "1308161"),
        "fox b": ("FOX", "1308161"),
    }
    text = "Welcome to the Fox A call. Fox A reported. Fox B also listed. Fox A. Fox B."
    ticker, cik, _, _, _, _ = resolve_company(text, None, sec)
    assert cik == "1308161"
    assert ticker == "FOXA"  # largest-cap class wins


def test_lookup_name_possessive_and_fuzzy():
    from ecvol.data.fincall_identity import _lookup_name

    sec = {"keycorp": ("KEY", "91576"), "align technology": ("ALGN", "1097149")}
    assert _lookup_name("KeyCorp's", sec) == ("KEY", "91576")  # true possessive dropped
    assert _lookup_name("Align Technologies", sec) == ("ALGN", "1097149")  # fuzzy 0.9
    assert _lookup_name("Completely Unrelated Name", sec) is None


def test_norm_name_joins_dotted_initials_consistently():
    # Punctuation removal leaves double spaces; the join must still fire so both
    # spellings land on the same key.
    assert norm_name("C. H. ROBINSON WORLDWIDE, INC.") == "ch robinson worldwide"
    assert norm_name("C.H. Robinson") == "ch robinson"
    assert norm_name("The J. M. Smucker Company") == "jm smucker"
    assert norm_name("J M SMUCKER Co") == "jm smucker"


def test_lookup_name_generic_token_gets_no_prefix_or_fuzzy():
    from ecvol.data.fincall_identity import _lookup_name

    sec = {"financial institutions": ("FISI", "862831"), "southern copper": ("SCCO", "1001838")}
    assert _lookup_name("Financial", sec) is None  # would prefix-match FISI
    assert _lookup_name("Southern Company", sec) is None  # would prefix-match SCCO
    # An exact (override-provided) key still wins for the same token.
    assert _lookup_name("Southern Company", {**sec, "southern": ("SO", "92122")}) == ("SO", "92122")


def test_lookup_name_fuzzy_survives_closer_wrong_candidate():
    from ecvol.data.fincall_identity import _lookup_name

    # "palatin technologies" ranks closer to the query than the right key; the
    # first-token guard must reject it and keep scanning, not give up.
    sec = {"palatin technologies": ("PTN", "911216"), "align technology": ("ALGN", "1097149")}
    assert _lookup_name("Align Technologies", sec) == ("ALGN", "1097149")


def test_clean_candidate_strips_filler_head_and_event_words():
    assert clean_candidate("you to the Adobe") == "Adobe"
    assert clean_candidate("today's Bank of America") == "Bank of America"
    assert clean_candidate("this Exxon Mobil Corporation") == "Exxon Mobil Corporation"
    assert clean_candidate("KLA Corporation September 2020") == "KLA Corporation"
    assert clean_candidate("Paychex, Inc. Reports") == "Paychex, Inc"
    assert clean_candidate("Royal Caribbean Group's Business Update") == "Royal Caribbean Group's"


def test_load_sec_table_merges_overrides(tmp_path):
    import json

    from ecvol.data.fincall_identity import load_sec_table

    # Pre-seed the cache so _download skips the network.
    ref = tmp_path / "raw" / "ref"
    ref.mkdir(parents=True)
    (ref / "company_tickers.json").write_text(
        json.dumps({"0": {"ticker": "ITW", "title": "ILLINOIS TOOL WORKS INC", "cik_str": 49826}})
    )
    ident = tmp_path / "identity"
    ident.mkdir()
    (ident / "fincall_name_overrides.csv").write_text(
        "alias,ticker,cik,note\nITW,ITW,49826,acronym\nNordstrom,JWN,72333,delisted\n"
    )
    table, single_ok = load_sec_table(tmp_path)
    assert table["itw"] == ("ITW", "49826")  # acronym alias for a live company
    assert table["nordstrom"] == ("JWN", "72333")  # absent from the live table
    assert table["illinois tool works"] == ("ITW", "49826")
    # Derived brand aliases must NOT be body-countable; override aliases are.
    assert "illinois" not in single_ok
    assert {"itw", "nordstrom"} <= single_ok


def test_load_sec_table_brand_collision_goes_to_larger_company(tmp_path):
    import json

    from ecvol.data.fincall_identity import load_sec_table

    ref = tmp_path / "raw" / "ref"
    ref.mkdir(parents=True)
    # File order = market-cap order: Vertex Pharmaceuticals before Vertex, Inc.
    (ref / "company_tickers.json").write_text(
        json.dumps(
            {
                "0": {"ticker": "VRTX", "title": "VERTEX PHARMACEUTICALS INC", "cik_str": 875320},
                "1": {"ticker": "VERX", "title": "Vertex, Inc.", "cik_str": 1806837},
            }
        )
    )
    table, _ = load_sec_table(tmp_path)
    assert table["vertex"] == ("VRTX", "875320")  # bare brand -> larger company
    assert table["vertex pharmaceuticals"] == ("VRTX", "875320")


def test_clean_candidate_strips_ordinal_event_tail():
    assert clean_candidate("FMC's 91st") == "FMC's"


def test_greeting_names_require_proper_noun_shape():
    from ecvol.data.fincall_identity import _greeting_names

    # re.I patterns capture lowercase prose; the proper-noun filter must drop it.
    junk = "Hello, and welcome to the ladies first quarter 2020 earnings call."
    assert _greeting_names(junk) == []
    kept = "Welcome to the eBay Q1 2020 Earnings Conference Call."
    assert ("eBay", 6, 0) in _greeting_names(kept)


def test_greeting_names_yield_all_matches_per_pattern():
    from ecvol.data.fincall_identity import _greeting_names

    # A generic operator line must not shadow the host's real greeting later on.
    text = (
        "Operator: Welcome to the Second Quarter 2020 Earnings Call. I will now "
        "turn the call over. Executives: Welcome to Ameriprise Financial's "
        "Second Quarter Earnings Call."
    )
    assert ("Ameriprise Financial", 6, 0) in _greeting_names(text)


def test_phrase_mentions_counts_recurring_distinctive_single_tokens():
    sec = {"allstate": ("ALL", "899051"), **SEC}
    ok = frozenset({"allstate"})
    text = "Allstate announced results. Allstate said growth was strong. Allstate Allstate."
    assert phrase_mentions(text, sec, ok)["899051"] == 3  # capped at 3 despite 4 mentions
    assert phrase_mentions("Allstate announced results.", sec, ok)["899051"] == 0  # no recurrence
    assert phrase_mentions(text, sec)["899051"] == 0  # not in single_ok -> never counted
