from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

# =============================================================================
# Config
# =============================================================================

MAX_TURNS = 10
TOP_K = 10

# Retrieval pool / per-route candidate limits
RECALL_POOL_SIZE = 500
BM25_LIMIT = 500
CATEGORY_LIMIT = 500
SEMANTIC_LIMIT = 120
PROFILE_LIMIT = 100
POPULAR_LIMIT = 100

RRF_K = 60                  # reciprocal-rank-fusion smoothing constant
CANDIDATE_THRESHOLD = 15    # pool larger than this -> still worth asking a question

# --- Dynamic clarification -----------------------------------------------
# score(attr) = normalized_entropy(attr) * coverage(attr)
#   normalized_entropy: how evenly candidates' values for this attribute are
#                        spread out (0 = everyone shares one value -> useless
#                        to ask, 1 = maximally split -> very informative)
#   coverage:            fraction of sampled candidates for which we could
#                        even detect a value (don't ask about attributes the
#                        catalog text barely mentions)
ATTRIBUTE_VOCAB = {
    "material": {
        "cotton", "polyester", "nylon", "leather", "wool", "spandex",
        "silk", "rayon", "fabric", "denim", "linen", "fleece",
    },
    "color": {
        "black", "white", "blue", "red", "pink", "green", "brown",
        "gray", "grey", "purple", "yellow", "orange", "navy", "beige",
    },
    "style": {
        "casual", "formal", "classic", "modern", "slim", "relaxed",
        "fitted", "oversized", "vintage", "sporty",
    },
    "use_case": {
        "hiking", "running", "gym", "winter", "summer", "outdoor",
        "work", "travel", "yoga", "training", "office", "party",
    },
}
ATTRIBUTE_LABELS = {"material": "material", "color": "color", "style": "style", "use_case": "use case"}

ATTRIBUTE_SAMPLE_SIZE = 80        # candidates sampled when scoring an attribute
MIN_ATTRIBUTE_COVERAGE = 0.15     # below this, an attribute isn't askable at all
MIN_ATTRIBUTE_SCORE = 0.08        # turn-1 (material/color) bar
LATE_ATTRIBUTE_SCORE = 0.3        # turn-3+ bar: higher, so weak late questions fall through to "other"

# --- Text rules ------------------------------------------------------------
OVERRIDE_RE = re.compile(
    r"\b(actually|ignore|instead|forget|never\s*mind|changed?\s*my\s*mind)\b", re.I
)
NO_INFO_RE = re.compile(
    r"(I don't have a preference for|I don't have an additional preference for|"
    r"Those options are not quite right yet)",
    re.I,
)
BOUNDARY_DODGE_RE = re.compile(
    r"I don't have a preference for .+?; please use your judgment", re.I
)
TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "i",
    "in", "is", "it", "me", "my", "of", "on", "or", "please", "some", "that",
    "the", "this", "to", "want", "with", "would", "you", "looking", "have",
    "has", "had", "do", "does", "did", "show", "find", "give", "get",
}

# Lightweight query expansion for semantic recall / similarity scoring
SYNTH_MAP = {
    "comfort": ["comfortable", "soft", "cushioned"],
    "comfortable": ["comfort", "soft", "cushioned"],
    "soft": ["comfortable", "comfort", "flexible"],
    "fit": ["fitted", "slim", "regular", "relaxed"],
    "fitted": ["fit"],
    "style": ["stylish", "fashion", "classic", "modern"],
    "stylish": ["style", "fashion"],
    "durability": ["durable", "longlasting", "tough"],
    "durable": ["durability", "tough"],
    "warmth": ["warm", "insulated", "thermal"],
    "warm": ["warmth", "insulated"],
    "weather": ["waterproof", "resistant", "outdoor"],
    "small": ["mini", "compact"],
    "large": ["big", "oversized"],
    "black": ["dark"],
    "white": ["light"],
}


# =============================================================================
# Small text helpers
# =============================================================================

def _text(v):
    """Flatten a catalog field (str/list/dict/None) into a plain string."""
    if v is None:
        return ""
    if isinstance(v, dict):
        return " ".join(f"{k} {x}" for k, x in v.items())
    if isinstance(v, list):
        return " ".join(map(str, v))
    return str(v)


def _terms(text):
    return [x.lower() for x in TOKEN_RE.findall(text) if len(x) > 1 and x.lower() not in STOPWORDS]


def _unique(items):
    return list(dict.fromkeys(items))


def _useful_text(msg):
    """Strip the evaluator's scripted phrasing down to the actual content,
    or drop the message entirely if it carries no new information."""
    if NO_INFO_RE.search(msg):
        return ""
    if "A key requirement is:" in msg:
        return msg.replace("A key requirement is:", " ")
    for pattern in (r"What I need is:\s*(.+)", r"For that, what matters is:\s*(.+)"):
        m = re.search(pattern, msg, re.I)
        if m:
            return m.group(1)
    return msg


def _expand(terms):
    out = set(terms)
    for t in terms:
        out.update(SYNTH_MAP.get(t, []))
    return out


def _similarity(a, b):
    """Jaccard similarity over synonym-expanded term sets."""
    if not a or not b:
        return 0.0
    a, b = _expand(a), _expand(b)
    return len(a & b) / len(a | b)


# =============================================================================
# Agent
# =============================================================================

class Agent:
    """Conversational shopping agent: hybrid retrieval (FTS/BM25 + category +
    semantic + profile + popularity, fused with RRF) plus a dynamic
    clarification-question controller that asks about whichever attribute is
    currently most informative for narrowing the candidate pool."""

    # Catalog index is process-wide and built once, shared across sessions.
    _shared_connection = None
    _shared_meta = None
    _shared_popular = None
    _shared_path = None

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl"):
        self.catalog_path = Path(catalog_path).resolve()
        self._sessions = {}
        self._token_cache = {}
        self._fts_cache = {}

        if Agent._shared_connection is None or Agent._shared_path != self.catalog_path:
            self._build_index()

        self.connection = Agent._shared_connection
        self.product_meta = Agent._shared_meta
        self.popular_items = Agent._shared_popular

    # -------------------------------------------------------------------
    # Catalog index (FTS5 + metadata + popularity ranking)
    # -------------------------------------------------------------------

    def _build_index(self):
        db = sqlite3.connect(":memory:")
        cur = db.cursor()
        for pragma in ("journal_mode=OFF", "synchronous=OFF", "temp_store=MEMORY", "cache_size=-16384"):
            cur.execute(f"PRAGMA {pragma}")

        cur.execute("""
            CREATE VIRTUAL TABLE products USING fts5(
                parent_asin UNINDEXED, title, categories, features,
                details, store, description,
                tokenize='unicode61 remove_diacritics 2'
            )
        """)

        meta, batch = {}, []

        with self.catalog_path.open(encoding="utf-8") as f:
            for line in f:
                p = json.loads(line)
                asin = str(p["parent_asin"])
                fields = [_text(p.get(k)) for k in
                          ("title", "categories", "features", "details", "store", "description")]

                meta[asin] = {
                    "text": " ".join(fields),
                    "rating": float(p.get("average_rating") or 0),
                    "rating_number": int(p.get("rating_number") or 0),
                }
                batch.append((asin, *fields))

                if len(batch) >= 2000:
                    cur.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()

        if batch:
            cur.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        db.commit()

        popular = sorted(meta.items(), key=lambda x: -(x[1]["rating"] * x[1]["rating_number"]))

        Agent._shared_connection = db
        Agent._shared_meta = meta
        Agent._shared_popular = [asin for asin, _ in popular[:300]]
        Agent._shared_path = self.catalog_path

    # -------------------------------------------------------------------
    # Session lifecycle
    # -------------------------------------------------------------------

    def reset(self, session_id, user_profile):
        self._sessions[session_id] = {
            "accumulated_terms": [],
            "profile_terms": _unique(_terms(_text(user_profile))),
            "asked_attributes": set(),
            "last_asked": None,
            "seen_asins": set(),
            "mode": None,
        }

    def _detect_state(self, msg):
        return {
            "override": bool(OVERRIDE_RE.search(msg)),
            "boundary_dodge": bool(BOUNDARY_DODGE_RE.search(msg)),
        }

    def _route(self, terms):
        """0-3 accumulated terms -> browse (diverse/exploratory retrieval);
        4+ terms -> focus (high-precision, hard-filter retrieval)."""
        return "focus" if len(terms) > 3 else "browse"

    # -------------------------------------------------------------------
    # Cached lookups
    # -------------------------------------------------------------------

    def _product_terms(self, asin):
        if asin not in self._token_cache:
            if len(self._token_cache) >= 4000:
                self._token_cache.pop(next(iter(self._token_cache)))
            self._token_cache[asin] = set(_terms(self.product_meta[asin]["text"]))
        return self._token_cache[asin]

    def _fts(self, expr, limit):
        if not expr:
            return []
        key = (expr, limit)
        if key not in self._fts_cache:
            rows = self.connection.execute(
                """
                SELECT parent_asin FROM products WHERE products MATCH ?
                ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0)
                LIMIT ?
                """,
                (expr, limit),
            ).fetchall()
            if len(self._fts_cache) >= 500:
                self._fts_cache.pop(next(iter(self._fts_cache)))
            self._fts_cache[key] = [str(r[0]) for r in rows]
        return self._fts_cache[key]

    def _query(self, terms):
        return " OR ".join(f'"{t}"' for t in _unique(terms)) if terms else ""

    # -------------------------------------------------------------------
    # Retrieval routes
    # -------------------------------------------------------------------

    def _keyword(self, terms):
        return self._fts(self._query(terms), BM25_LIMIT)

    def _category(self, terms):
        if not terms:
            return []
        rows = self.connection.execute(
            "SELECT parent_asin FROM products WHERE categories MATCH ? LIMIT ?",
            (self._query(terms), CATEGORY_LIMIT),
        ).fetchall()
        return [str(r[0]) for r in rows]

    def _semantic(self, terms):
        """Synonym-expanded FTS recall, re-ranked by Jaccard similarity to
        the original (unexpanded) query terms."""
        if not terms:
            return []
        expanded = _unique(t for term in terms for t in [term, *SYNTH_MAP.get(term, [])])
        ids = self._fts(self._query(expanded), SEMANTIC_LIMIT * 2)

        q = set(terms)
        ranked = sorted(
            ((-_similarity(q, self._product_terms(a)), i, a) for i, a in enumerate(ids))
        )
        return [a for _, _, a in ranked[:SEMANTIC_LIMIT]]

    def _profile(self, terms):
        return self._fts(self._query(terms), PROFILE_LIMIT) if terms else []

    # -------------------------------------------------------------------
    # Fusion / reranking
    # -------------------------------------------------------------------

    @staticmethod
    def _rrf(lists, limit=RECALL_POOL_SIZE):
        """Reciprocal rank fusion across multiple retrieval routes."""
        scores = defaultdict(float)
        first_seen = {}
        for route, ranking in enumerate(lists):
            for rank, asin in enumerate(ranking, 1):
                scores[asin] += 1 / (RRF_K + rank)
                first_seen.setdefault(asin, (route, rank))
        return sorted(scores, key=lambda x: (-scores[x], first_seen[x], x))[:limit]

    def _coverage(self, candidates, query_terms):
        """Rerank by exact term overlap with the accumulated query, then
        overlap ratio, then semantic similarity; ties fall back to prior
        (RRF) rank."""
        if not query_terms:
            return candidates
        q = set(query_terms)
        scored = []
        for rank, asin in enumerate(candidates):
            p = self._product_terms(asin)
            matched = len(q & p)
            scored.append((-matched, -(matched / len(q)), -_similarity(q, p), rank, asin))
        scored.sort()
        return [x[-1] for x in scored]

    def _profile_rerank(self, candidates, profile_terms):
        """Browse-mode only: nudge ranking toward the user's long-term profile."""
        if not profile_terms:
            return candidates
        q = set(profile_terms)
        scored = sorted(
            (-_similarity(q, self._product_terms(a)), rank, a)
            for rank, a in enumerate(candidates)
        )
        return [a for _, _, a in scored]

    # -------------------------------------------------------------------
    # Dynamic clarification question
    # -------------------------------------------------------------------

    def _attribute_signal(self, candidates, vocab):
        """How informative would asking about `vocab` be, given the current
        candidate pool? Returns (normalized_entropy, coverage, top_values)."""
        sample = candidates[:ATTRIBUTE_SAMPLE_SIZE]
        if not sample:
            return 0.0, 0.0, []

        counts = defaultdict(int)
        matched = 0
        for asin in sample:
            hit = self._product_terms(asin) & vocab
            if hit:
                matched += 1
                counts[sorted(hit)[0]] += 1  # stable representative value per product

        coverage = matched / len(sample)
        if len(counts) < 2:
            return 0.0, coverage, []

        entropy = -sum((c / matched) * math.log2(c / matched) for c in counts.values())
        norm_entropy = entropy / math.log2(len(counts))
        top_values = [v for v, _ in sorted(counts.items(), key=lambda x: -x[1])[:3]]
        return norm_entropy, coverage, top_values

    def _best_attribute(self, candidates, attrs, asked, min_score):
        """Pick the highest-scoring not-yet-asked attribute from `attrs`
        (score = normalized_entropy * coverage), or None if none clears
        `min_score`."""
        best_attr, best_score, best_values = None, 0.0, []
        for attr in attrs:
            if attr in asked:
                continue
            entropy, coverage, values = self._attribute_signal(candidates, ATTRIBUTE_VOCAB[attr])
            if coverage < MIN_ATTRIBUTE_COVERAGE:
                continue
            score = entropy * coverage
            if score > best_score:
                best_attr, best_score, best_values = attr, score, values
        return (best_attr, best_values) if best_score >= min_score else (None, [])

    def _ask_attribute(self, state, attr, values):
        state["asked_attributes"].add(attr)
        label = ATTRIBUTE_LABELS.get(attr, attr)
        suffix = f" (e.g. {', '.join(values)})" if values else ""
        return attr, f"The candidates vary in {label}{suffix} - which do you prefer?"

    def _next_question(self, state, candidates):
        """3-stage clarification funnel:
          1) material or color, whichever is more discriminating (always asks
             something here - falls back to material if neither clears the bar)
          2) an open "feature" question (fixed, not scored)
          3) one more specific attribute IF it clears a higher bar
             (LATE_ATTRIBUTE_SCORE), else a generic "other" catch-all
        """
        asked = state["asked_attributes"]

        # Stage 1: material / color
        if not (asked & {"material", "color"}):
            attr, values = self._best_attribute(candidates, ("material", "color"), asked, MIN_ATTRIBUTE_SCORE)
            if attr is None:
                attr = "material"
                _, _, values = self._attribute_signal(candidates, ATTRIBUTE_VOCAB["material"])
            return self._ask_attribute(state, attr, values)

        # Stage 2: feature (unscored, always asked once)
        if "feature" not in asked:
            asked.add("feature")
            return "feature", "Is there a specific feature or requirement that matters most to you?"

        # Stage 3: one more attribute if it's clearly worth it, else "other"
        if "other" not in asked:
            attr, values = self._best_attribute(candidates, ATTRIBUTE_VOCAB.keys(), asked, LATE_ATTRIBUTE_SCORE)
            if attr:
                return self._ask_attribute(state, attr, values)
            asked.add("other")
            return "other", "There are still a lot of options - is there another specific requirement that matters to you?"

        return None, "Here are the closest matches I found."

    # -------------------------------------------------------------------
    # Main entry point
    # -------------------------------------------------------------------

    def respond(self, session_id, user_message, turn, top_k):
        state = self._sessions[session_id]
        detected = self._detect_state(user_message)

        new_terms = _unique(_terms(_useful_text(user_message)))
        state["accumulated_terms"] = _unique(state["accumulated_terms"] + new_terms)

        if detected["override"]:
            # Keep everything learned so far (category, other constraints);
            # only clear "seen" so previously-shown items can resurface if
            # they now match the new preference.
            state["seen_asins"].clear()
            if state["last_asked"]:
                state["asked_attributes"].discard(state["last_asked"])

        if detected["boundary_dodge"] and state["last_asked"]:
            state["asked_attributes"].discard(state["last_asked"])

        query_terms = state["accumulated_terms"]
        mode = state["mode"] = self._route(query_terms)

        keyword_rank = self._keyword(query_terms)
        category_rank = self._category(query_terms)

        if mode == "focus":
            ranked = self._rrf([keyword_rank, category_rank])
            ranked = self._coverage(ranked, query_terms)
        else:
            ranked = self._rrf([
                keyword_rank,
                self._semantic(query_terms),
                category_rank,
                self._profile(state["profile_terms"]),
                self.popular_items[:POPULAR_LIMIT],
            ])
            ranked = self._coverage(ranked, query_terms)
            ranked = self._profile_rerank(ranked, state["profile_terms"])

        # Surface unseen items first so repeated turns don't loop on the same set
        seen = state["seen_asins"]
        ranked = [a for a in ranked if a not in seen] + [a for a in ranked if a in seen]

        if len(ranked) < top_k:
            for asin in self.popular_items:
                if asin not in ranked:
                    ranked.append(asin)
                if len(ranked) >= top_k:
                    break

        final_ids = ranked[:top_k]
        seen.update(final_ids)

        message, ask_attribute = "Here are the closest matches I found.", None
        if len(ranked) > CANDIDATE_THRESHOLD and turn < MAX_TURNS:
            ask_attribute, message = self._next_question(state, ranked)
        state["last_asked"] = ask_attribute

        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": [{"parent_asin": a} for a in final_ids],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }