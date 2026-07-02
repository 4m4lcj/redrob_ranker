from __future__ import annotations
import json, os, sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from keywords import RELEVANT_SKILLS
from helpers import _parse_date, _days_since, structure_candidate, build_jd_text

_MODEL_NAME        = "sentence-transformers/all-MiniLM-L6-v2"
_TOP_N             = 100

# ── Embedding functions ────────────────────────────────────────────────────

def embed_candidates(filtered_jsonl_path: str, model_name: str, batch_size: int = 64, 
    _model: SentenceTransformer | None = None) -> tuple[list[dict], np.ndarray]:
    
    """
    Stream filtered_candidates.jsonl, structure each candidate, encode in batches.

    Returns:
        candidate_list     — raw candidate dicts (for metadata access)
        embeddings_matrix  — shape (N, dim), L2-normalised

    Pass _model to reuse an already-loaded SentenceTransformer and avoid a
    second network round-trip when calling embed_jd with the same model.
    """

    candidates: list[dict] = []
    texts:      list[str]  = []

    print(f"Reading candidates from {filtered_jsonl_path} ...")

    with open(filtered_jsonl_path, encoding="utf-8") as f:
        for line in tqdm(f, desc="  structuring", unit="cand"):
            line = line.strip()

            if not line:
                continue

            c = json.loads(line)
            candidates.append(c)
            
            texts.append(structure_candidate(c)["combined_text"])

    print(f"  {len(candidates):,} candidates structured  "
          f"(avg text len {sum(len(t) for t in texts)//len(texts)} chars)")

    if _model is None:
        print(f"Loading model: {model_name}")

        _model = SentenceTransformer(model_name)

    print(f"Encoding with batch_size={batch_size} ...")
    
    embeddings = _model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    return candidates, np.array(embeddings)


def embed_jd(jd_parsed_path: str, model_name: str, _model: SentenceTransformer | None = None) -> np.ndarray:
    """
    Load jd_parsed.json and embed its embedding_text field.

    Returns shape (dim,), L2-normalised.
    Pass _model to reuse an already-loaded SentenceTransformer.
    """
    with open(jd_parsed_path, encoding="utf-8") as f:
        jd = json.load(f)

    jd_text = build_jd_text(jd)
    
    if _model is None:
        _model = SentenceTransformer(model_name)
    
    emb = _model.encode([jd_text], normalize_embeddings=True)
    return np.array(emb[0])


# ── Reasoning draft generator ──────────────────────────────────────────────

def generate_reasoning_draft(candidate: dict, score: float, rank: int) -> str:
    """
    Template-based 1-2 sentence reasoning string. No LLM call.
    Capped at 200 chars.
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    skills  = candidate.get("skills", [])

    yoe     = profile.get("years_of_experience") or 0
    title   = profile.get("current_title", "")
    company = profile.get("current_company", "")
    country = profile.get("country", "")

    # Top relevant skills by endorsement count
    rel = sorted([s for s in skills if s.get("name", "").lower() in RELEVANT_SKILLS], key=lambda s: s.get("endorsements") or 0, reverse=True)[:3]
    skill_str = (", ".join(f"{s['name']} ({s.get('endorsements', 0)} end.)" for s in rel) if rel else "some relevant skills")

    # YoE vs JD range (5-9)
    if yoe < 5:
        yoe_tag = f"{yoe:.0f}yr (below range)"

    elif yoe > 9:
        yoe_tag = f"{yoe:.0f}yr (above range)"

    else:
        yoe_tag = f"{yoe:.0f}yr"

    # Positives
    flags: list[str] = []
    if signals.get("open_to_work_flag"):
        flags.append("open to work")

    if signals.get("willing_to_relocate"):
        flags.append("relocatable")

    notice = signals.get("notice_period_days") or 0
    if notice <= 30:
        flags.append(f"{notice}d notice")

    # Concerns
    concerns: list[str] = []
    if notice > 90:
        concerns.append(f"{notice}d notice")

    if country != "India":
        concerns.append(f"based in {country}")

    days_inactive = _days_since(_parse_date(signals.get("last_active_date")))
    if days_inactive > 90:
        concerns.append(f"inactive {days_inactive}d")

    draft = f"{yoe_tag} {title} at {company}; skills: {skill_str}"
    if flags:
        draft += "; " + ", ".join(flags)

    if concerns:
        draft += " | concerns: " + ", ".join(concerns)

    return draft[:200]


# ── Cosine ranking with behavioral multiplier ──────────────────────────────

def cosine_rank(candidate_list: list[dict], embeddings: np.ndarray,jd_embedding: np.ndarray) -> list[dict]:
    """
    Score = cosine_similarity * behavioral_multiplier.
    Returns top-100 ranked dicts with full metadata.
    """
    cosine_sims = embeddings @ jd_embedding  # dot product on normalised vecs

    results: list[dict] = []
    for idx, (c, cos) in enumerate(zip(candidate_list, cosine_sims)):
        sig = c.get("redrob_signals", {})
        profile = c.get("profile", {})

        # ── Behavioral multiplier ──────────────────────────────────────────
        mult = 1.0

        days_inactive = _days_since(_parse_date(sig.get("last_active_date")))
        if days_inactive > 180:
            mult *= 0.60

        elif days_inactive > 90:
            mult *= 0.80

        elif days_inactive < 30:
            mult *= 1.05

        rr = sig.get("recruiter_response_rate") or 0
        if rr < 0.2:
            mult *= 0.75

        icr = sig.get("interview_completion_rate") or 0
        if 0 < icr < 0.5:   # guard 0 = no history
            mult *= 0.85

        if sig.get("open_to_work_flag"):
            mult *= 1.10

        notice = sig.get("notice_period_days") or 0
        if notice > 90:
            mult *= 0.90

        if sig.get("willing_to_relocate"):
            mult *= 1.05

        final_score = float(cos) * mult

        results.append({
            "candidate_id":          c.get("candidate_id"),
            "score":                 round(final_score, 6),
            "cosine_sim":            round(float(cos), 6),
            "behavioral_multiplier": round(mult, 4),
            "title":                 profile.get("current_title", ""),
            "company":               profile.get("current_company", ""),
            "yoe":                   profile.get("years_of_experience"),
            "location":              f"{profile.get('location', '')} ({profile.get('country', '')})",
            "rank":                  0,    # filled after sort
            "reasoning_draft":       "",   # filled after sort
        })

    results.sort(key=lambda x: x["score"], reverse=True)

    top = results[:_TOP_N]
    for rank, r in enumerate(top, 1):
        r["rank"] = rank
        cand = candidate_list[next(i for i, c in enumerate(candidate_list) if c.get("candidate_id") == r["candidate_id"])]

        r["reasoning_draft"] = generate_reasoning_draft(cand, r["score"], rank)

    return top

# ── Embedder class (kept for ranker.py compatibility) ──────────────────────

class Embedder:
    def __init__(self, model_name: str = _MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    def cosine_scores(self, candidate_embs: np.ndarray, jd_emb: np.ndarray) -> np.ndarray:
        return candidate_embs @ jd_emb

# ── Main: full pipeline ────────────────────────────────────────────────────
"""
if __name__ == "__main__":
    MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # benchmark winner

    _here = Path(__file__).parent
    _candidates_path = str(_here / ".." / "data" / "filtered_candidates.jsonl")
    _jd_path         = str(_here / ".." / "jd_parsed.json")
    _output_path     = _here / ".." / "output" / "ranked.csv"

    # Load model once — reused for both candidate and JD embedding
    print(f"Loading model: {MODEL}")
    _model = SentenceTransformer(MODEL)

    candidates, embeddings = embed_candidates(_candidates_path, MODEL, _model=_model)
    jd_vec  = embed_jd(_jd_path, MODEL, _model=_model)
    ranked  = cosine_rank(candidates, embeddings, jd_vec)

    print(f"\nTop 10 (of {len(ranked)} ranked):")
    for r in ranked[:10]:
        print(f"  #{r['rank']:>3}  {r['candidate_id']}  score={r['score']:.4f}"
              f"  (cos={r['cosine_sim']:.4f} x {r['behavioral_multiplier']:.3f})")
        print(f"         {r['title']} @ {r['company']}  |  {r['yoe']} YoE  |  {r['location']}")
        print(f"         {r['reasoning_draft']}")
        print()

    # Save CSV
    import csv
    _output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(_output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "rank", "candidate_id", "score", "cosine_sim", "behavioral_multiplier",
            "title", "company", "yoe", "location", "reasoning_draft",
        ])
        writer.writeheader()
        writer.writerows(ranked)
    print(f"Saved {len(ranked)} ranked candidates to {_output_path}")

"""