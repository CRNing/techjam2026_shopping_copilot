# Submission Report — Shopping Copilot

This report satisfies the submission requirements for "a short report describing
method, model choice, and limitations" and "a disclosure of latency, token usage,
and estimated model cost."

## 1. Method

The agent (`agent.py`, class `Agent`) is a deterministic, rule- and
statistics-based conversational retrieval system with no learned model and no
LLM calls. It has four components, mirroring the challenge's four pillars:

**a) Dual-track intent routing.** Each turn, the session's accumulated query
terms are counted; ≤3 terms routes to a `browse` track (wide, diverse recall),
>3 terms routes to a `focus` track (tight, high-precision recall). The route is
re-evaluated every turn as the conversation accumulates more information.

**b) Hybrid retrieval, fused with RRF.** Up to five independent retrieval
signals are combined depending on the active track:
- Keyword — BM25 over an in-memory SQLite FTS5 index (`title`, `categories`,
  `features`, `details`, `store`, `description`).
- Category — exact FTS match against the `categories` field.
- Semantic — synonym-expanded FTS recall (hand-authored `SYNTH_MAP`), reranked
  by Jaccard similarity to the original query terms.
- Profile — FTS match against the long-term user profile text.
- Popularity — catalog items ranked by `average_rating * rating_number`.

  Each active signal returns a ranked list of `parent_asin`s; all lists are
  merged with Reciprocal Rank Fusion (`RRF_K = 60`), then reranked by exact
  term-coverage, coverage ratio, and semantic similarity against the
  accumulated query.

**c) Multi-turn dialog state machine.** Session state tracks accumulated query
terms, asked attributes, and seen items across turns (incremental slot
filling). Two conversational signals are detected via regex:
- *Intent override* (e.g. "actually…", "ignore that…") — retains everything
  learned so far, reopens the most recently asked attribute for
  re-clarification, and clears the "seen items" set.
- *Boundary dodge* (e.g. "I don't have a preference for X; use your
  judgment") — stops probing that attribute rather than re-asking it.

**d) Dynamic, entropy-driven clarification questions.** When the candidate
pool is still large (`> CANDIDATE_THRESHOLD = 15`) and turns remain, the agent
scores each not-yet-asked attribute (material, color, style, use_case) against
the *current* candidate pool:

```
score(attribute) = normalized_entropy(attribute) * coverage(attribute)
```

`normalized_entropy` measures how evenly the attribute's detected values are
spread across candidates (0 = all one value → useless to ask, 1 = maximally
split → highly informative); `coverage` measures what fraction of sampled
candidates even expose a detectable value for that attribute. The
highest-scoring attribute is asked about first. The question funnel escalates:
material/color (turn 1) → an open-ended "feature" question (turn 2, unscored)
→ one further attribute only if its score clears a higher bar
(`LATE_ATTRIBUTE_SCORE = 0.3`, turn 3) → a generic "other" catch-all once no
remaining attribute is informative enough. If the pool is already small or
turns are exhausted, the agent returns its current best-ranked results instead
of asking anything.

## 2. Model Choice

**No LLM or trained model is used anywhere in this submission.** All ranking,
routing, and question-selection logic is implemented with classical
information-retrieval and information-theory techniques: BM25 (via SQLite
FTS5), Reciprocal Rank Fusion, Jaccard similarity over a hand-authored synonym
map, and Shannon entropy over attribute value distributions.

This was a deliberate choice, not a fallback:
- **Cost & reproducibility.** The competition does not require or provide paid
  LLM access; a zero-cost, zero-network pipeline removes any dependency on
  external credentials, rate limits, or provider availability, and guarantees
  bit-for-bit reproducible runs.
- **Latency.** In-memory SQLite FTS5 lookups and small in-process Python
  computations are on the order of single-digit milliseconds per call (see
  §4), well under what a network LLM round-trip would cost, which is
  advantageous given the competition's per-turn efficiency (MTTC) objective.
- **Fit for the sub-task.** Attribute selection and slot-tracking are
  low-dimensional, well-structured decisions (a handful of candidate
  attributes, a bounded vocabulary) where a closed-form information-gain
  score is both interpretable and sufficient, without the latency or cost of
  an LLM call per turn.

The trade-off (discussed further in Limitations) is reduced robustness to
language not anticipated by the regex/vocabulary rules, and no deep semantic
understanding beyond synonym expansion + Jaccard similarity.

## 3. Limitations

- **Rule-based intent/attribute detection is brittle.** Override and
  boundary-dodge detection rely on regex patterns and a fixed keyword
  vocabulary (`ATTRIBUTE_VOCAB`); this performs well against the evaluator's
  scripted phrasing but would generalize poorly to noisier, unscripted user
  language.
- **"Semantic" retrieval is shallow.** It is FTS keyword expansion via a
  hand-written synonym map plus Jaccard similarity, not a real embedding-based
  dense retriever — a likely source of the gap between Hit Rate@10 and MRR on
  the `buying` scenario (the correct item is usually retrieved, but not always
  ranked first).
- **`boundary` and `intent_override` scenarios have the highest MTTC** of all
  scenario types, suggesting the clarification loop occasionally re-asks a
  question that has, in effect, already been answered around these two state
  transitions.
- **Attribute vocabulary is manually curated and apparel-specific**; it would
  need to be extended (or learned from catalog term frequency) to generalize
  to other product categories.
- **No within-session personalization beyond a static profile snapshot** —
  `profile_terms` is captured once at session start and does not update from
  in-session engagement signals.

## 4. Disclosure: Latency, Token Usage, and Estimated Model Cost

**Network access:** Not required. The agent runs entirely offline — the
product catalog is loaded once into an in-memory SQLite database at process
start, and every subsequent `respond()` call only reads from that in-memory
index and in-process Python session state. There is no offline/online
fallback distinction because there is no online mode to begin with.

**Environment variables:** None required.

**Token usage:** Always `0`. No LLM is called, so `usage.prompt_tokens` and
`usage.completion_tokens` are reported as `0` in every `respond()` call by
construction (see `agent.py`, end of `respond()`).

**Estimated model cost:** `$0` per session and in aggregate. There is no
metered API in the pipeline; the only cost is local CPU time.

**Latency:** Per-turn latency is dominated by SQLite FTS5 queries (a handful
of `MATCH` calls against a 50,000-row in-memory table) plus O(pool size)
Python-level scoring over at most `RECALL_POOL_SIZE = 500` candidates. No
network I/O occurs on the request path. Index build time (one-time, at
process/session-pool startup, not per turn) scales with catalog size.

Measured on this team's development machine, running:

```bash
time python3 -m evaluator.local_evaluator \
  --catalog data/catalog.jsonl \
  --dataset data/public_set.jsonl \
  --output results.json
```

| Measurement | Value |
|---|---|
| Full run (200 public sessions, incl. one-time catalog index build) | **15.809s** total wall-clock |
| CPU utilization during the run | 99% (`user 15.56s`, `sys 0.16s`) — no meaningful idle/I/O-wait time, consistent with an all-local, no-network pipeline |
| Average wall-clock time per session (200 sessions, includes all turns per session) | ≈ 79 ms/session |

The near-100% CPU utilization and absence of any idle time is itself evidence
that no network round-trips occur on the request path — a pipeline making
external API calls would show substantial wall-clock time attributable to
I/O wait rather than CPU. Per-`respond()`-call percentile timings (mean /
p50 / p95) were not separately profiled for this submission; the aggregate
run-time above is considered sufficient to characterize the system as
low-latency and CPU-bound rather than network-bound.

## 5. Reproducibility

- **Python version:** >= 3.10 (uses `X | Y` union type syntax).
- **Install:** no third-party dependencies; see `requirements.txt`.
- **Run:**
  ```bash
  python3 evaluator.py --catalog data/catalog.jsonl --dataset data/public_set.jsonl --output results.json
  ```
- **Non-obvious environment variables:** none.
