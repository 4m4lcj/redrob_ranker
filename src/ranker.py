from __future__ import annotations
import argparse, csv, json, os, sys, time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from filters import apply_filters
from helpers import structure_candidate, build_jd_text
from embedder import cosine_rank

_HERE       = Path(__file__).parent
_MODELS_DIR = _HERE / ".." / "models"

MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_WARN_SEC = 200
TOP_N          = 100

# ── Model resolution ───────────────────────────────────────────────────────

def resolve_model(shortname: str) -> str:
    """Return local path if pre-downloaded, else HuggingFace model name."""

    local = _MODELS_DIR / shortname

    if local.exists() and any(local.iterdir()):
        return str(local)

    return shortname

# ── Pipeline phases ────────────────────────────────────────────────────────

def filter(candidates_path: str, limit: int | None) -> tuple[list[dict], float]:
    label = f"{limit:,}" if limit else "all"
    print(f"\nFiltering {label} candidates...")

    t0 = time.perf_counter()

    filtered: list[dict] = []
    total_seen = 0
    reject_counts: dict[str, int] = {}

    with open(candidates_path, encoding="utf-8") as f:
        for line in f:
            if limit and total_seen >= limit:
                break

            line = line.strip()
            if not line:
                continue

            c = json.loads(line)
            total_seen += 1

            ok, reason = apply_filters(c)
            if ok:
                filtered.append(c)

            else:
                tag = reason.split(":")[0]
                reject_counts[tag] = reject_counts.get(tag, 0) + 1

    elapsed = time.perf_counter() - t0

    print(f"  Scanned  : {total_seen:,}")
    print(f"  Passed   : {len(filtered):,}  ({100*len(filtered)/max(total_seen,1):.1f}%)")
    print(f"  Rejected : {total_seen - len(filtered):,}")

    top_reasons = sorted(reject_counts.items(), key=lambda x: -x[1])[:5]

    for tag, cnt in top_reasons:
        print(f"    {tag:<35s} {cnt:>5,}")
    print(f"  Filter time: {elapsed:.1f}s")

    return filtered, elapsed


def embed(filtered: list[dict], model_shortname: str, jd_path: str, batch_size: int = 64) -> tuple[np.ndarray, np.ndarray, float]:
    """Returns (candidate_embeddings, jd_embedding, elapsed_seconds)."""

    model_path = resolve_model(model_shortname)
    source = "local cache" if Path(model_path).exists() else "HuggingFace"

    print(f"\nEmbedding {len(filtered):,} candidates with model={model_shortname} ({source})...")

    t0    = time.perf_counter()
    model = SentenceTransformer(model_path, device = "cpu")

    texts = [structure_candidate(c)["combined_text"] for c in tqdm(filtered, desc="  structuring", unit="cand", leave=False)]

    embeddings = model.encode(texts, batch_size = batch_size, show_progress_bar = True, normalize_embeddings = True)

    with open(jd_path, encoding="utf-8") as f:
        jd = json.load(f)

    jd_emb = model.encode([build_jd_text(jd)], normalize_embeddings=True)[0]

    elapsed = time.perf_counter() - t0

    print(f"  Embedded in {elapsed:.1f}s  "
          f"(shape {embeddings.shape}, dim={embeddings.shape[1]})")

    if elapsed > EMBED_WARN_SEC:

        print(f"  WARNING: embedding took {elapsed:.1f}s > {EMBED_WARN_SEC}s budget. "
              f"Consider switching to all-MiniLM or reducing the filtered pool.")

    return np.array(embeddings), np.array(jd_emb), elapsed


def rank(filtered: list[dict], embeddings: np.ndarray, jd_emb: np.ndarray) -> tuple[list[dict], float]:
    print("\nRanking...")

    t0     = time.perf_counter()
    ranked = cosine_rank(filtered, embeddings, jd_emb)
    elapsed = time.perf_counter() - t0
    
    print(f"  Top {len(ranked)} selected in {elapsed:.3f}s")
    return ranked, elapsed


def output(ranked: list[dict], out_path: Path) -> None:
    print(f"\nWriting {out_path} ...")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "candidate_id": r["candidate_id"],
            "rank":         r["rank"],
            "score":        r["score"],
            "reasoning":    r["reasoning_draft"],
        }
        for r in ranked
    ]

    # Validate before writing
    expected = len(rows)

    assert sorted(r["rank"] for r in rows) == list(range(1, expected + 1)), \
        "Ranks are not unique and sequential"
    
    scores = [r["score"] for r in rows]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)), \
        "Scores are not non-increasing"

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Wrote {out_path}")
    print(
        f"  Validation: {len(rows)} rows, "
        f"ranks 1-{len(rows)} unique, scores non-increasing — all OK"
    )

# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description =" Redrob candidate ranking pipeline")
    parser.add_argument("--candidates", required = True, help = "Path to candidates.jsonl (raw, unfiltered)")
    parser.add_argument("--jd", required = True, help = "Path to parsed JD.json")
    parser.add_argument("--out", required = True, help = "Output CSV path")
    parser.add_argument("--limit", type = int, default = None, help = "Process only first N candidates (for speed testing)")
    args = parser.parse_args()

    wall_start = time.perf_counter()

    print("=" * 60)
    print("  Redrob Ranking Pipeline")
    print(f"  candidates : {args.candidates}")
    print(f"  JD         : {args.jd}")
    print(f"  output     : {args.out}")
    print(f"  model      : {MODEL}")
    if args.limit:
        print(f"  limit      : {args.limit:,}  (test mode)")
    print("=" * 60)

    # Phase 1
    filtered, t_filter = filter(args.candidates, args.limit)

    if len(filtered) < TOP_N:
        print(
            f"\nOnly {len(filtered)} candidates passed filters "
            f"(need {TOP_N}). Using all candidates instead."
        )

        filtered = []

        with open(args.candidates, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if args.limit and i >= args.limit:
                    break
                
                line = line.strip()
                if line:
                    filtered.append(json.loads(line))

    embeddings, jd_emb, t_embed = embed(filtered, MODEL, args.jd)

    ranked, t_rank = rank(filtered, embeddings, jd_emb)
    
    output(ranked, Path(args.out))

    wall_time = time.perf_counter() - wall_start

    print()
    print("=" * 60)
    print(f"  DONE  —  total wall-clock time: {wall_time:.1f}s  ({wall_time/60:.1f} min)")
    print(f"  Filter : {t_filter:.1f}s")
    print(f"  Embed  : {t_embed:.1f}s")
    print(f"  Rank   : {t_rank:.3f}s")
    print("=" * 60)

    # Print top 10 to terminal
    """
    print("\nTop 10:")
    for r in ranked[:10]:
        print(f"  #{r['rank']:>3}  {r['candidate_id']}  "
              f"score={r['score']:.4f}  "
              f"(cos={r['cosine_sim']:.4f} x {r['behavioral_multiplier']:.3f})")
        print(f"         {r['title']} @ {r['company']}  |  {r['yoe']} YoE")
        print(f"         {r['reasoning_draft']}")
    """

if __name__ == "__main__":
    main()
