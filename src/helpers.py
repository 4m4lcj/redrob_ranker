from datetime import date, datetime
from .keywords import PURE_SERVICES

_TODAY = date(2026, 6, 18)

_MAX_COMBINED_CHARS = 1900   # ~512 tokens proxy for all-MiniLM wordpiece

def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except Exception:
        return None
    
def _days_since(d: date | None) -> int:
    if d is None:
        return 0
    return (_TODAY - d).days

def _contains_keyword(text: str, keywords: set[str]) -> bool:
    low = text.lower()
    return any(kw in low for kw in keywords)

def structure_candidate(candidate: dict) -> dict:
    """
    Convert a raw candidate dict into structured text blocks for embedding.

    Returns a dict with keys:
      block_summary   — headline + summary (<=300 chars)
      block_skills    — filtered, formatted skill list
      block_work      — last 3 non-services roles with truncated descriptions
      block_meta      — key hiring signals as a compact string
      combined_text   — the string that actually gets embedded (<=512 tokens)
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    skills  = candidate.get("skills", [])
    career  = candidate.get("career_history", [])

    # ── block_summary ──────────────────────────────────────────────────────
    headline = (profile.get("headline") or "").strip()
    summary  = (profile.get("summary")  or "").strip()
    raw = f"{headline}. {summary}" if headline and summary else (headline or summary)
    block_summary = raw[:300]

    # ── block_skills ───────────────────────────────────────────────────────
    def _skill_passes(s: dict) -> bool:
        prof = s.get("proficiency", "")
        end  = s.get("endorsements") or 0
        dur  = s.get("duration_months") or 0
        if prof == "beginner" and dur < 6:
            return False
        return end > 0 or dur > 6

    kept = sorted(
        [s for s in skills if _skill_passes(s)],
        key=lambda s: (s.get("endorsements") or 0),
        reverse=True,
    )
    parts = [
        f"{s.get('name', '')} ({s.get('proficiency', '')}, {s.get('duration_months', 0)}mo)"
        for s in kept
    ]
    block_skills = ("Skills: " + ", ".join(parts)) if parts else "Skills: none"

    # ── block_work ─────────────────────────────────────────────────────────
    product_roles = [
        h for h in career
        if not any(svc in h.get("company", "").lower() for svc in PURE_SERVICES)
    ]
    services_only = len(product_roles) == 0
    roles_to_use  = career if services_only else product_roles

    role_parts = []
    for h in roles_to_use[:3]:
        title   = h.get("title", "")
        company = h.get("company", "")
        desc    = (h.get("description") or "").strip()[:150]
        role_parts.append(f"{title} at {company}: {desc}")

    block_work = " | ".join(role_parts)
    if services_only:
        block_work = "[services-only] " + block_work

    # ── block_meta ─────────────────────────────────────────────────────────
    block_meta = (
        f"YoE: {profile.get('years_of_experience', '')} | "
        f"Country: {profile.get('country', '')} | "
        f"Relocate: {signals.get('willing_to_relocate', '')} | "
        f"Notice: {signals.get('notice_period_days', '')}d | "
        f"Mode: {signals.get('preferred_work_mode', '')}"
    )

    # ── combined_text ──────────────────────────────────────────────────────
    prefix   = f"{block_summary} | {block_skills} | "
    combined = prefix + block_work
    if len(combined) > _MAX_COMBINED_CHARS:
        remaining = max(0, _MAX_COMBINED_CHARS - len(prefix))
        combined  = prefix + block_work[:remaining]

    return {
        "block_summary": block_summary,
        "block_skills":  block_skills,
        "block_work":    block_work,
        "block_meta":    block_meta,
        "combined_text": combined,
    }

def build_jd_text(jd: dict) -> str:
    if jd.get("embedding_text"):
        return jd["embedding_text"]
    skills = " ".join(jd.get("must_have_skills", []) + jd.get("nice_to_have_skills", []))
    return f"{jd.get('role_summary', '')} {skills}"
