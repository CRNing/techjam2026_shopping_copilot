# Shopping Copilot — Conversational Search & Recommendation Agent

A multi-turn conversational shopping agent for the TechJam "Shopping Copilot" challenge.
Given free-text customer messages across a session (up to 10 turns), the agent retrieves
and ranks products from a frozen Amazon `Clothing_Shoes_and_Jewelry` catalog, and decides
whether to return results or ask a targeted clarification question to narrow down the
candidate pool faster.

## Project Overview

The agent is built around four ideas from the problem statement:

1. **Dual-track hybrid retrieval.** Every turn is routed to either a `focus` track
   (query has accumulated ≥4 meaningful terms → tight, high-precision filtering via
   BM25 keyword + category match) or a `browse` track (fewer terms → wider, diverse
   recall combining keyword, lightweight semantic expansion, category, long-term user
   profile, and catalog popularity). All routes are fused with **Reciprocal Rank
   Fusion (RRF)**, then reranked by term-coverage and semantic similarity.

2. **Stateful multi-turn dialog.** Session state accumulates terms turn over turn
   (incremental slot filling). Two conversational signals are detected explicitly:
   - **Intent override** ("actually…", "ignore that…", "changed my mind…") — the
     session keeps everything already learned but reopens the previously-asked
     attribute for re-clarification and clears "seen" items so results can refresh.
   - **Boundary dodge** ("I don't have a preference for X; use your judgment") — the
     agent stops probing that attribute and moves on rather than looping on it.

3. **Dynamic, information-theoretic clarification questions.** Instead of a fixed
   question script, each not-yet-asked attribute (material, color, style, use case)
   is scored against the *current* candidate pool as
   `score = normalized_entropy(attribute) * coverage(attribute)` — i.e. how evenly
   its values are spread across candidates, weighted by how often the attribute is
   even detectable in the catalog text. The agent asks about whichever attribute
   would most reduce uncertainty right now, with an escalating question funnel:
   material/color → open-ended "feature" → one more attribute *only if it clearly
   still helps* → a generic "other" catch-all once structured attributes stop being
   informative. This makes the questioning adaptive per category (e.g. jackets vs.
   shoes surface different useful attributes) instead of a hardcoded order.

4. **Efficiency-aware stopping.** The agent only asks a question when the candidate
   pool is still large (`> CANDIDATE_THRESHOLD`) and turns remain; otherwise it
   returns its best-ranked results immediately, directly optimizing for the
   competition's Mean-Turns-to-Conversion (MTTC) metric.

### Development tools
- VS Code
- Python 3.10+

### APIs used
- None. No hosted/paid LLM or third-party API is used — all retrieval, ranking, and
  clarification logic is deterministic, rule- and statistics-based (BM25, RRF,
  Jaccard similarity, Shannon entropy).

### Libraries and frameworks
- Python standard library only: `sqlite3` (in-memory FTS5 for BM25 keyword search),
  `json`, `re`, `math`, `collections`, `pathlib`, `argparse`, `statistics`, `random`,
  `uuid`.

### Datasets and assets
- Frozen 50,000-product catalog from the **Amazon Reviews 2023** dataset
  (`Clothing_Shoes_and_Jewelry` category), provided by the organizer.
- 200 labeled public development sessions (`data/public_set.jsonl`) for local
  evaluation; 800 private sessions are held by the organizer for final scoring.

## Repository Structure

```
.
├── starter/
│   └── agent.py           # Agent implementation (retrieval, ranking, dialog state)
├── evaluator.py            # Official local evaluator (Hit Rate@10, MRR, MTTC, TechnicalScore)
├── data/
│   ├── catalog.jsonl       # Frozen product catalog (50,000 items)
│   └── public_set.jsonl    # 200 public dev sessions
├── results.json             # Evaluator output (generated)
└── README.md
```

## Setup and Installation

**Requirements:** Python 3.10+ (uses `X | Y` union type syntax). No third-party
packages are required — everything runs on the standard library.

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd <your-repo-name>

# 2. (Optional) create a virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Verify the catalog checksum (as provided by the organizer)
sha256sum -c data/catalog.jsonl.sha256

# No `pip install` step is required — no external dependencies.
```

## Steps to Reproduce Results

Run the official local evaluator against the public development set:

```bash
python3 evaluator.py \
  --catalog data/catalog.jsonl \
  --dataset data/public_set.jsonl \
  --output results.json
```

This prints a summary to stdout and writes full per-session results to
`results.json`. Current results on the 200-sample public set:

| Metric | Overall |
|---|---|
| Hit Rate@10 | 0.990 |
| MRR | 0.701 |
| MTTC | 3.005 |
| Efficiency | 0.800 |
| **TechnicalScore** | **0.865** |

| Scenario | Samples | Hit Rate@10 | MRR | MTTC |
|---|---|---|---|---|
| Boundary | 10 | 0.900 | 0.693 | 4.600 |
| Browsing | 80 | 1.000 | 0.696 | 2.938 |
| Buying | 80 | 0.988 | 0.658 | 2.488 |
| Intent Override | 30 | 1.000 | 0.830 | 4.033 |

To evaluate a modified agent, edit `starter/agent.py` and rerun the same command —
the evaluator interface (`Agent.reset` / `Agent.respond`) is unchanged.

## Limitations & What We'd Improve With More Time

- **Rule-based attribute/intent detection is brittle.** Override, boundary-dodge,
  and attribute classification currently rely on regex and small keyword vocabularies
  (`ATTRIBUTE_VOCAB`). This works well against the evaluator's scripted phrasing but
  would generalize poorly to noisier, real-world user language. A learned intent/slot
  classifier (even a small fine-tuned model) would be more robust.
- **No real embedding-based semantic retrieval.** "Semantic" recall today is FTS
  keyword expansion via a hand-written synonym map (`SYNTH_MAP`) plus Jaccard
  similarity — cheap and dependency-free, but shallow compared to a real dense vector
  retriever. Swapping in sentence embeddings (even a small local model, kept in-memory
  per the "no heavy vector DB" constraint) should meaningfully lift MRR, especially on
  the `buying` scenario where MRR currently lags Hit Rate@10 (0.658 vs 0.988 — the
  correct item is usually *found* but not always ranked first).
- **Boundary and Intent-Override scenarios have the highest MTTC** (4.6 and 4.0 turns
  respectively) of all scenario types, indicating the clarification-question loop
  occasionally re-asks a question that's already effectively been answered (e.g. right
  after an override or a dodge). Tightening the `asked_attributes` bookkeeping around
  these two transitions is the single highest-leverage remaining fix.
- **Attribute vocabulary coverage is manual and English/apparel-specific.** Extending
  `ATTRIBUTE_VOCAB` (or deriving it automatically from catalog term frequency) would
  make the clarification strategy transfer to other product categories without
  hand-curation.
- **No personalization beyond a static profile-term bag.** `profile_terms` is a
  one-shot snapshot from `user_profile` at session start; it doesn't update from
  in-session behavior (e.g. which recommended items the user engaged with vs. ignored),
  which the Runtime Adaptation pillar of the problem statement calls for more fully.

## Team Member Contributions

| Member | Contributions |
|---|---|
| JUNREN&nbsp;YIN | Hybrid retrieval pipeline — BM25/FTS5 indexing, category retrieval, semantic expansion retrieval, and RRF fusion + reranking (coverage/profile rerank) |
| RUINING&nbsp;CAO | Dialog state machine & dynamic clarification controller — dual-track (focus/browse) routing, session state management, intent-override / boundary-dodge handling, entropy-driven attribute question selection |
| SUMMER&nbsp;H | Evaluation, tuning & performance — local evaluator runs, per-scenario metric analysis, parameter tuning (thresholds, RRF_K), latency / token / cost benchmarking |
| LISA&nbsp;LIU | Documentation & submission packaging — README, REPORT (method / model choice / limitations disclosure), DATA_ATTRIBUTION, requirements.txt, repository structure and GitHub submission |
| JUNXIAO&nbsp;CHEN | Demo & presentation — demo video, Devpost project description, final Q&A prep, end-to-end and edge-case testing of the full pipeline |
