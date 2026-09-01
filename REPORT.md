# Submission Report —Dual-Layer Dynamic Shopping Copilot

This report satisfies the submission requirements for "a short report describing
method, model choice, and limitations" and "a disclosure of latency, token usage,
and estimated model cost."

## 1. Method

The agent (`agent.py`, class `Agent`) is a deterministic, rule- and
statistics-based conversational retrieval system with no learned model and no
LLM calls. Its design can be summarized as a **Dual-Layer Dynamic Shopping
Copilot**: an outer focus/browse retrieval layer and an inner dynamic
attribute-clarification layer. It has four components, mirroring the
challenge's four pillars:

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
- **Latency and deployment stability.** We intentionally do not use a real
  embedding retriever because loading an embedding model and running it on
  every turn would add model-load and inference latency, memory consumption,
  and deployment dependencies. The resulting offline, zero-token pipeline
  provides stable and predictable response times, which is advantageous given
  the competition's per-turn efficiency (MTTC) objective.
- **Fit for the sub-task.** Attribute selection and slot-tracking are
  low-dimensional, well-structured decisions (a handful of candidate
  attributes, a bounded vocabulary) where a closed-form information-gain
  score is both interpretable and sufficient, without the latency or cost of
  an LLM call per turn.

The trade-off (discussed further in Limitations) is reduced robustness to
language not anticipated by the regex/vocabulary rules, and no deep semantic
understanding beyond synonym expansion + Jaccard similarity.

## 3. Limitations

- **Attribute-aware override decay was explored but not shipped.** During development we prototyped a mechanism where, on intent override, the system would classify which attribute the new preference belongs to (e.g. material) and down-weight — rather than fully discard — the previously stated value for that same attribute in reranking (all other attributes, including the open-ended "feature" bucket, were left untouched). The motivation was that a superseded preference still carries some signal about user intent and shouldn't be treated as equally irrelevant to a term the user never mentioned. We ultimately did not include this in the final submission: the current override handling simply retains all previously accumulated terms and reopens the most recently asked attribute for re-clarification, without attribute-specific weighting. With more time, we would revisit this weighted-decay approach and tune it against the `intent_override` scenario specifically, since it currently has one of the highest MTTC values of any scenario type.
- **No real embedding-based semantic retrieval.** The semantic route uses FTS keyword expansion, a hand-written synonym map, and Jaccard-style similarity. We intentionally do not use dense embeddings in this submission. This is a deliberate system-design trade-off rather than a claim that embeddings are not useful: loading and running even a small local embedding model would add model-loading overhead, per-turn query-encoding latency, memory usage, and deployment dependencies. Because the task is an interactive 10-turn conversation, we prioritize predictable response time, fully offline execution, zero-token usage, and a lightweight reproducible deployment. A future version could use sentence embeddings, kept in memory as required by the no-heavy-vector-database constraint, to improve paraphrase handling and ranking quality. This may meaningfully improve MRR, especially in the `buying` scenario where the correct product is often retrieved but not always ranked first.
- **Boundary and Intent-Override scenarios have the highest MTTC** (4.6 and 4.0 turns respectively) of all scenario types, indicating the clarification-question loop occasionally re-asks a question that's already effectively been answered (e.g. right after an override or a dodge). Tightening the `asked_attributes` bookkeeping around these two transitions is the single highest-leverage remaining fix.
- **Attribute vocabulary coverage is manual and English/apparel-specific.** Extending`ATTRIBUTE_VOCAB` (or deriving it automatically from catalog term frequency) would make the clarification strategy transfer to other product categories without hand-curation.
- **No personalization beyond a static profile-term bag.** `profile_terms` is a one-shot snapshot from `user_profile` at session start; it doesn't update from in-session behavior (e.g. which recommended items the user engaged with vs. ignored), which the Runtime Adaptation pillar of the problem statement calls for more fully.

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
| Full run (200 public sessions, incl. one-time catalog index build) | **14.61s** total wall-clock |
| CPU utilization during the run | **99.5%** (`user 14.33s`, `sys 0.20s`) — no meaningful idle/I/O-wait time, consistent with an all-local, no-network pipeline |
| Average wall-clock time per session (200 sessions, includes all turns per session) | **≈ 73.1 ms/session** |
| Respond calls executed by the evaluator | **599** (early-stop after a valid hit) |

The near-100% CPU utilization and absence of any idle time is itself evidence
that no network round-trips occur on the request path — a pipeline making
external API calls would show substantial wall-clock time attributable to
I/O wait rather than CPU. The measured aggregate run corresponds to an
overall average of approximately 24.4 ms per executed `respond()` call when
the one-time index-build cost is included. Per-call mean/p50/p95 timings were
not separately profiled.

## 5. Reproducibility

- **Python version:** >= 3.10 (uses `X | Y` union type syntax).
- **Install:** no third-party dependencies; see `requirements.txt`.
- **Run:**
  ```bash
  python3 -m evaluator.local_evaluator \
    --catalog data/catalog.jsonl \
    --dataset data/public_set.jsonl \
    --output results.json
  ```
- **Non-obvious environment variables:** none.

## 6. Demonstrated Multi-turn Session

The following is a representative evaluator-compatible interaction. The
structured `ask_attribute` field controls the simulator; recommendation IDs
are returned in ranked order and only the first ten valid catalog items are
scored.

```text
Turn 1 — User: I'm looking for a jacket, but I'm still exploring.
Agent:  Here are the closest matches I found. Do you prefer a material such as
        cotton, leather, or polyester?
        ask_attribute = "material"

Turn 2 — User: For that, what matters is: waterproof and lightweight.
Agent:  Is there a specific feature or requirement that matters most to you?
        ask_attribute = "feature"

Turn 3 — User: Actually, ignore my earlier preference. What I need is: black.
Agent:  Is there a specific feature or requirement that matters most to you?
        ask_attribute = "feature"

Turn 4 — Agent: Re-ranks the refreshed candidates and returns the current top-10
        catalog parent_asin values when no further clarification is needed.
```
