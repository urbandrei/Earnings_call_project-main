# Venue memo: ACL via ARR, target cycle and submission readiness

**Date:** 2026-10-06 · **Scope:** the two open questions from DECISIONS 2026-08-23 §2 / HANDOFF ("ACL cycle dates", "anonymity vs. preprint"), plus page-limit, checklist and release-norm readiness of the current draft (`paper/main.pdf`, 30 pp).
**Status:** advisory memo. Nothing here changes DESIGN.md. Any change to DESIGN §4 ("arXiv preprint precedes any submission") needs a DECISIONS.md entry first.

All facts below come from official pages I fetched on **2026-10-06**. Where an official page says TBA, this memo says so and does not estimate.

---

## 1. Recommendation

**Target the ARR January 2027 cycle (submission 2027-01-04, AoE), then commit to ACL 2027 (Kyoto, 17–22 Aug 2027).** Papers that are not selected for the ACL 2027 main conference are automatically considered for Findings of ACL. The ACL 2027 commitment deadline is **TBA**.

Why:
- **It matches the decided venue.** DECISIONS 2026-08-23 says "ACL". ACL 2027 is the only ACL-proper meeting that is still reachable.
- **There is enough time to do the work properly.** The draft is roughly 6 ACL pages over the limit (§4), still uses a non-ACL template, and is waiting on advisor/co-author review (HANDOFF T8.2). It also has a stale Limitations paragraph and gaps against the checklist (§5). Three months is enough for all of that. Six days (the October cycle) is not.
- **Findings comes with it.** [R14] (Same Company, Same Signal) appeared at Findings of ACL 2025. Automatic Findings consideration makes that the realistic floor.
- **The ACL 2027 CFP explicitly invites this kind of paper.** It welcomes papers that "announce the creation of a new resource" and that "reproduce, or fail to reproduce, previous results". Submit under the **Resources and Evaluation** area. The ACL 2027 theme track ("Homogenization and knowledge collapse in LLMs") does not fit this paper.

**Fallback (not recommended unless the user wants an earlier decision):** ARR October 2026 cycle. Submission 2026-10-12, meta-reviews 2026-12-17 (ARR) / 2026-12-18 (NAACL page), commitment 2026-12-23. It commits to **NAACL 2027** (San Francisco, 1–5 Jun 2027; notification 2027-02-10) or COLING 2027. Only 6 days remain, and the advisor review has not happened yet.

### Cycle table (verified 2026-10-06)

| ARR cycle | Submission | Meta-reviews | Commitment | Commits to | Usable? |
|---|---|---|---|---|---|
| May 2026 | 2026-05-25 | 2026-07-30 | 2026-08-02 | EMNLP 2026 (24–29 Oct 2026), AACL 2026 | **Closed.** No EMNLP 2026 window remains. |
| Aug 2026 | 2026-08-03 | 2026-10-08 | 2026-10-11 | EACL 2027 (Athens, 9–14 Mar 2027) | **Closed** for new submissions. EACL commitment needs existing ARR reviews, which we do not have. |
| Oct 2026 | **2026-10-12** | 2026-12-17 | **2026-12-23** | NAACL 2027, COLING 2027 | Possible but rushed (6 days). |
| Jan 2027 | **2027-01-04** (ACL 2027 site; the ARR dates page lists only "January, 2027") | TBA | **TBA** | ACL 2027 (+ Findings) | **Recommended.** |

Two caveats on the table:
- **Review carry-over from October to ACL 2027 is not confirmed.** NAACL and EACL both state that reviews from earlier ARR cycles can be committed. The ACL 2027 pages I fetched do not say this yet, and the ACL commitment date is TBA. Do not plan around "submit in October, then commit to ACL" until that is published.
- **Every ARR submission needs a reviewer.** One coauthor must register to review at least 3 papers. That person needs at least 3 *CL papers in the last 5 years, or the paper must qualify for an exception (for example, newcomer authors). Non-compliance means desk rejection. So decide who the reviewer is (likely the advisor) before submitting.

---

## 2. Preprint and anonymity: verdict

**An arXiv preprint before submission or during review is allowed.** ARR has had **no anonymity period** since 2024-02-15: "there is no anonymity period or limitation on posting or discussing non-anonymous preprints while the work is under peer review." NAACL 2027 repeats this ("no anonymity period requirement").

There is a trade-off. ARR "incentivize[s] anonymous submissions through special paper awards and **priority in acceptance decisions for borderline papers**." Authors who tick **"no non-anonymous preprint"** commit to not posting "until the metareviews are released, under the penalty of desk rejection." ARR also notes that wide sharing makes reviewer recruitment harder.

What this means for DESIGN §4's "arXiv preprint precedes any submission":
- The rule **does not conflict with policy**. The conflict flagged in DECISIONS 2026-08-23 can be closed as "no conflict".
- Following the rule costs us the borderline-priority and award incentive. Our framing (prior multimodal gains largely reflect identity, not content) could plausibly land a borderline Findings/main decision, so that cost is real.
- The submission itself must still be anonymized: no author names, no self-identifying citations, and **no links to the named GitHub/Zenodo/HF release**. Code and data go in as anonymized `.zip`/`.tgz` archives or an Anonymous-GitHub-style link. Dropbox-style cloud links are prohibited.

**Decision left to the user (D2 below):** keep "preprint first", or tick "no non-anonymous preprint" and post to arXiv once meta-reviews are out (for the January cycle, that is spring 2027, exact date TBA).

---

## 3. Format requirements (ARR CFP)

- **Long paper:** up to **8 pages of content**; the accepted version gets 9.
- **Not counted toward the limit:** references (unlimited), the **Limitations** section (mandatory; missing it means desk rejection), an optional ethical-considerations section, and appendices.
- **Limitations placement and content:** it goes after the conclusion and before the references. It "should not introduce new methods, analysis, or results."
- **Appendices** come after the references, in double-column format, and must be fully anonymized. **"Reviewers are not required to consider material in appendices."** Anything the claims depend on must be in the 8 pages.
- **Responsible NLP checklist** is mandatory. "Incorrect, incomplete or misleading information in the checklist can result in desk rejection."
- **No dual submission:** ARR will not consider a paper that is under review at a journal or another conference.
- **Resubmissions** must acknowledge the earlier version.

---

## 4. Page-length gap

How I measured the current build (`main.pdf`): it uses 10pt two-column, A4, 2.5 cm margins, and a home-made layout rather than the ACL style. The main text runs from p.1 to about mid-p.13, where Limitations starts. References and the appendix fill pp.13–30.

| | Current draft | ARR limit |
|---|---|---|
| Main-text pages | ~12.3 at 10pt (~13.5–14 once switched to the ACL 11pt template, my estimate) | 8 |
| Main-text words (sections 00–08, LaTeX source) | ~8,250 (Results alone: ~2,950; Data: ~1,530) | ~5,000–5,500 with this table load (rough) |
| Table floats in main text | 12 (substrate, baselines, text, controls, audio, grid, convention, repro, faithful, repro_controls, earnings25_audio, lookahead + lookahead_measured) + 1 figure | ~4–5 tables + 1 figure is realistic |

**The gap is roughly 5.5–6 ACL pages, about 40% of the main text.** Switch to `acl.sty` first (main.tex lines 11–14 describe how), then measure again.

**Suggested split.** This is a judgement call for the authors; the main-text set below is my suggestion.

- **Keep in the main text** (the benchmark-first spine):
  - Introduction and contributions (trimmed)
  - Substrate audit (Table 5R / `substrate`)
  - Benchmark construction, condensed to about ¾ column: corpora, targets, splits, release
  - Protocol, condensed
  - Baselines plus the consolidated grid (`grid`, or `baselines`, whichever carries RQ1)
  - Identity controls (`controls`), because this is the central claim
  - Reproduction summary (`repro`, with `repro_controls` folded in or reduced to one row per paper)
  - One lookahead table
  - Discussion and conclusion, merged and shortened
- **Move to the appendix** (≈6 pages):
  - The text and audio stage tables, as separate tables (`text`, `audio`). Summarize each in one sentence and point to the grid.
  - Horizon-convention analysis (`convention`)
  - Earnings25 bitrate ladder (`earnings25_audio`)
  - Faithful-reproduction details (`faithful`) and the per-paper reproduction narratives in §6 "Reproductions with the authors' own code"
  - `lookahead_measured` / anchor-sensitivity detail
  - Identity-reconstruction and call-timing details from §3 (keep one sentence and one number each)
  - The pre-registered framing-gate narrative from the Discussion
  - Most of Related Work beyond one compact paragraph (or compress it to about ½ column)

---

## 5. Checklist, Limitations and Ethics vs. the draft

| Requirement | Draft status (`09-limitations.tex`, `10-ethics.tex`, body) | Action |
|---|---|---|
| Limitations section, titled exactly "Limitations" (A1) | Present and substantive. **Stale:** the "Timestamps" item still says re-baselining on the measured anchor "is deferred to the release decision", but option B was adopted (commit 4bf4f66, HANDOFF resolved 2026-10-06). | Rewrite that item to match the re-baselined tables. Check that no item adds new results. |
| Risks (A2) | Ethics covers intended use, gender confound and licensing. | Add one line on misuse (automated trading) and dual use. This is mostly there already. |
| Artifact citations and licenses (B1–B3) | Ethics names only FinCall-Surprise (Apache-2.0) and MAEC (CC-BY-SA-4.0). The body cites Earnings25 as CC-BY-4.0. **EC/EarningsCall (license not stated), the KeFVP/VolTAGE/HTML repos (no license file) and SCSS DEC are not covered in the Ethics statement.** `LICENSE-DATA.md` has the full table and flags confirming EC's terms as an open item. | Add a license sentence or appendix table that mirrors `LICENSE-DATA.md`, and state the scripts-not-data handling for EC. |
| PII and offensive content (B4) | Not addressed. Calls name real executives, and audio is personal data. | Add a sentence: public corporate disclosures, no speaker-level release, pooled features only. |
| Documentation and data statistics (B5–B6) | Split sizes are in §3 and the appendix. No datasheet. | Point to the per-split counts table. A short datasheet in the appendix strengthens a Resources paper. |
| Compute budget and parameters (C1) | Only "one consumer GPU" and "<5M parameters". | Add GPU model and approximate GPU-hours per stage (feature extraction, heads, reproductions). These are in the journal and memory notes. |
| Hyper-parameter search and selection (C2) | Partly covered (grid selected on validation). | Make it explicit per model family, in the appendix. |
| Error bars and seeds (C3) | 5 seeds, cluster bootstrap, DM/Holm. | Fine. State mean vs. single-run per table. |
| Package versions (C4) | Pinned lockfile, but versions are not in the paper. | Add an appendix line with key package versions (uv lock). |
| Human annotators (D1–D5) | The Phase-6 human labels were excised (DECISIONS 2026-08-26). Identity audits were done by the authors. | Answer "No" with that justification. |
| AI assistants (E1) | **Not disclosed anywhere.** The pipeline and draft were built with substantial agentic AI assistance. | Add a disclosure, either in the checklist answer or in an acknowledgements/ethics line, consistent with the ACL publication ethics policy. **The user must decide the wording.** |
| Anonymized supplementary material | The title block reads "Anonymous Author(s)", and §3 "Release" has no URLs. | Before submitting, grep for repo names, `urbandrei`, Zenodo DOIs and acknowledgements. Prepare an anonymized code `.zip` (the `ecvol` CLI plus configs) and a data `.zip` (derived features, splits, manifests) within upload limits. |

---

## 6. Decisions left to the user

1. **Cycle.** January 2027 → ACL 2027 (recommended), or October 12, 2026 → NAACL/COLING 2027.
2. **Preprint.** Post to arXiv before submission, as DESIGN §4 says (allowed, but gives up the anonymous-submission priority for borderline papers). Or tick "no non-anonymous preprint" and post after meta-reviews. Either way, log it in DECISIONS.md, together with closing the 2026-08-23 "conflict" flag (there is no policy conflict).
3. **Reviewer-of-record.** Which coauthor registers as ARR reviewer (≥3 papers, needs ≥3 *CL papers in 5 years), or which exception applies.
4. **What stays in the 8 pages.** Approve or adjust the keep/move split in §4. In particular, decide whether the reproduction study or the benchmark construction gets the space.
5. **AI-assistance disclosure** wording (checklist E1).
6. **EC dataset terms.** Confirm with the authors before release (`LICENSE-DATA.md` open item 1). This affects what the anonymized data `.zip` may contain.
7. **Advisor/co-author review timing** (HANDOFF T8.2). It must finish before cutting to 8 pages, so leave margin before 2027-01-04.

---

## Sources (all read 2026-10-06)

- ARR dates: https://aclrollingreview.org/dates
- ARR call for papers (page limits, Limitations, appendix, supplementary material, anonymity incentives, dual submission, topic areas incl. Resources and Evaluation): https://aclrollingreview.org/cfp
- ARR anonymity policy (no anonymity period since 2024-02-15): https://aclrollingreview.org/anonymity
- ARR Responsible NLP checklist (A–E): https://aclrollingreview.org/responsibleNLPresearch/
- ARR author-reviewer requirement: https://aclrollingreview.org/reviewing-workload-requirement/
- ACL 2027 home (Kyoto, 17–22 Aug 2027; ARR deadline 2027-01-04; commitment TBA): https://2027.aclweb.org/
- ACL 2027 main-conference CFP (paper types, theme track, automatic Findings consideration, page limits): https://2027.aclweb.org/calls/main/
- NAACL 2027 main-conference CFP (dates, no anonymity period, earlier-cycle reviews eligible): https://2027.naacl.org/calls/main_conference_papers/
- EACL 2027 CFP and home (ARR Aug cycle, commitment 2026-10-11, Athens 9–14 Mar 2027): https://2027.eacl.org/calls/papers/ , https://2027.eacl.org/
- EMNLP 2026 CFP (commitment 2026-08-02, passed): https://2026.emnlp.org/calls/main_conference_papers/
