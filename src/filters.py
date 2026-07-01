from __future__ import annotations

from helpers import _parse_date, _days_since, _contains_keyword
from keywords import PURE_SERVICES, CORE_AI_SKILLS, RELEVANT_SKILLS, NON_TECH_TITLE_KEYWORDS, RELEVANT_ASSESSED

# ── Check 1: title / headline domain mismatch ──────────────────────────────

def check_title_description_mismatch(candidate: dict) -> bool:
    """Reject if current title or headline is clearly non-technical."""

    profile = candidate.get("profile", {})
    title = (profile.get("current_title") or "").lower()
    headline = (profile.get("headline") or "").lower()

    if _contains_keyword(title, NON_TECH_TITLE_KEYWORDS):
        return False
    
    if _contains_keyword(headline, NON_TECH_TITLE_KEYWORDS):
        return False
    
    return True


# ── Check 2: skills relevance ──────────────────────────────────────────────

# Changed Overlap criteria

def check_skills_relevance(candidate: dict) -> tuple[bool, int]:
    """
    Hard-reject if candidate has zero JD-relevant skills.
    Returns (passed, overlap_count) — overlap_count is useful for pre-scoring.
    """
    skills = candidate.get("skills", [])
    cand_skill_names = {s.get("name", "").lower() for s in skills}

    overlap = 0
    for skill in cand_skill_names:
        # exact match or substring match against multi-word tokens
        if any(rel in skill for rel in RELEVANT_SKILLS):
            overlap += 1

    return overlap >= 5, overlap


# ── Check 3: suspicious endorsement/duration pattern ──────────────────────

#Changed conditions

def check_low_endorsements_low_duration(candidate: dict) -> bool:
    """
    Reject candidates with an implausibly high proportion of skills that
    claim advanced proficiency despite very low endorsements and little
    experience.

    Rule - Reject if >40% of skills are: advanced proficiency, fewer than 2 endorsements, 
           less than 6 months of experience
    """
    skills = candidate.get("skills", [])
    if not skills:
        return True

    flagged = sum(
        1
        for s in skills
        if (
            s.get("proficiency") == "advanced"
            and (s.get("endorsements") or 0) < 2
            and (s.get("duration_months") or 0) < 6
        )
    )

    return flagged / len(skills) <= 0.40

# ── Check 4: all-services background ──────────────────────────────────────

def check_companies(candidate: dict) -> bool:
    """
    Reject pure IT-services backgrounds with no product-company experience.
    If current company is services but they have prior product experience: keep.
    """
    career = candidate.get("career_history", [])
    if not career:
        return True 

    companies = [h.get("company", "").lower().strip() for h in career]
    
    all_services = all(any(svc in co for svc in PURE_SERVICES) for co in companies if co)

    return not all_services

# ── Check 5: open-to-work / salary / notice red flags ─────────────────────

def check_open_to_work_salary_honeypot(candidate: dict) -> bool:
    """
    Reject implausible or unhireable candidates based on signals metadata.

    Fails if:
    - open_to_work=True but salary max < 5 LPA (implausible for Senior AI Eng)
    - open_to_work=False AND last_active > 180 days (ghost candidate)
    - notice_period_days > 120 (>4 months makes hiring very difficult)
    """
    signals = candidate.get("redrob_signals", {})

    otw     = signals.get("open_to_work_flag", False)
    sal     = signals.get("expected_salary_range_inr_lpa") or {}
    sal_max = sal.get("max")
    notice  = signals.get("notice_period_days") or 0
    last_active = _parse_date(signals.get("last_active_date"))
    days_inactive = _days_since(last_active)

    if otw and sal_max is not None and sal_max < 5.0:
        return False

    if not otw and days_inactive is not None and days_inactive > 180:
        return False

    if notice > 120:
        return False

    return True


# ── Check 6: location / relocation ────────────────────────────────────────

def check_location(candidate: dict) -> bool:
    """Reject non-India candidates who are explicitly not willing to relocate."""

    profile  = candidate.get("profile", {})
    signals  = candidate.get("redrob_signals", {})
    country  = (profile.get("country") or "").strip()
    relocate = signals.get("willing_to_relocate")  # bool or None

    if country != "India" and relocate is False:
        return False
    
    return True


# ── Check 7: redrob_signals hard gates ────────────────────────────────────

def check_signals_hard(candidate: dict) -> tuple[bool, str]:
    """
    Hard-reject on conditions derivable purely from redrob_signals that indicate
    the candidate is unhireable, fraudulent, or completely disengaged.
    """
    signals = candidate.get("redrob_signals", {})

    # 1. Dual-unverified identity — likely bot or test account
    if not signals.get("verified_email") and not signals.get("verified_phone"):
        return False, "unverified_identity"

    # 2. Platform assessment contradicts claimed advanced proficiency
    assessed = {k.lower(): v for k, v in (signals.get("skill_assessment_scores") or {}).items()}
    for skill in candidate.get("skills", []):
        if skill.get("proficiency") == "advanced":
            name = skill.get("name", "").lower()
            if name in assessed and assessed[name] < 35:
                return False, "assessment_contradicts_proficiency"

    # 3. Fully disengaged: unavailable + dormant + unresponsive to recruiters
    last_active = _parse_date(signals.get("last_active_date"))
    days_inactive = _days_since(last_active) or 0
    if (not signals.get("open_to_work_flag", True)
            and days_inactive > 90
            and (signals.get("recruiter_response_rate") or 0) < 0.15):
        return False, "fully_disengaged"

    # 4. Interview ghost — only reject if icr > 0 (has interview history) but still low.
    #    icr = 0.0 is the default when no interviews have been scheduled; excluding that
    #    avoids false-positives on candidates new to the platform.
    icr = signals.get("interview_completion_rate") or 0
    if 0 < icr < 0.30:
        return False, "interview_ghost"

    # 5. Extremely sparse profile — likely bot or throwaway account
    completeness = signals.get("profile_completeness_score") or 0
    if completeness < 20:
        return False, "incomplete_profile"

    return True, "passed"


# ── Check 8: honeypot profile integrity ───────────────────────────────────

def check_honeypot_profile(candidate: dict) -> tuple[bool, str]:
    """
    Reject profiles with impossible or highly suspicious timelines.

    Returns (passed, reason_tag).

    Fails if:
    - sum(career duration_months) / 12 > years_of_experience + 2
      (more accumulated career time than stated experience — fabricated timeline)
    - years_of_experience > 18 (implausibly high in this dataset)

    Note: "expert" proficiency is valid in the dataset (contrary to schema docs),
    so it is NOT used as a rejection signal.
    """
    profile = candidate.get("profile", {})
    career  = candidate.get("career_history", [])

    yoe = profile.get("years_of_experience") or 0

    total_months = sum(h.get("duration_months") or 0 for h in career)
    if total_months / 12 > yoe + 2:
        return False, f"timeline_impossible: career={total_months/12:.1f} yrs but yoe={yoe:.1f}"

    if yoe > 18:
        return False, f"experience_inflation: yoe={yoe}"

    return True, ""


# ── Orchestrator ───────────────────────────────────────────────────────────

def apply_filters(candidate: dict) -> tuple[bool, str]:
    """
    Run all hard filters in priority order.
    Returns (True, "passed") or (False, "<reason>").
    Fast path: returns on first failure so we don't waste time on clear rejects.
    """
    profile = candidate.get("profile", {})
    yoe     = profile.get("years_of_experience") or 0

    # Top-level YoE hard gate (before any per-check logic)
    if yoe < 4:
        return False, f"yoe_too_low: {yoe:.1f} < 4"
    if yoe > 20:
        return False, f"yoe_too_high: {yoe:.1f} > 20"

    if not check_title_description_mismatch(candidate):
        title = profile.get("current_title", "")
        return False, f"non_tech_title: '{title}'"

    passed_skills, overlap = check_skills_relevance(candidate)
    if not passed_skills:
        return False, "zero_relevant_skills"

    cand_skill_names = {s.get("name", "").lower() for s in candidate.get("skills", [])}
    has_core = any(
        core in name or name in core
        for name in cand_skill_names
        for core in CORE_AI_SKILLS
    )
    if not has_core:
        return False, "no_core_ai_skill"

    if not check_low_endorsements_low_duration(candidate):
        return False, "honeypot_endorsement_pattern"

    sig_ok, sig_reason = check_signals_hard(candidate)
    if not sig_ok:
        return False, sig_reason

    if not check_companies(candidate):
        return False, "pure_services_background"

    if not check_location(candidate):
        return False, "non_india_non_relocatable"

    if not check_open_to_work_salary_honeypot(candidate):
        signals = candidate.get("redrob_signals", {})
        notice  = signals.get("notice_period_days", 0)
        sal     = (signals.get("expected_salary_range_inr_lpa") or {}).get("max")
        otw     = signals.get("open_to_work_flag", False)
        last_active = _parse_date(signals.get("last_active_date"))
        days_inactive = _days_since(last_active)
        if notice > 120:
            return False, f"notice_too_long: {notice} days"
        if otw and sal is not None and sal < 5.0:
            return False, f"salary_honeypot: open_to_work=True but max={sal} LPA"
        return False, f"ghost_candidate: inactive {days_inactive} days, open_to_work=False"

    hp_ok, hp_reason = check_honeypot_profile(candidate)
    if not hp_ok:
        return False, hp_reason

    return True, "passed"


# ── Signals score multiplier (used by ranker.py, NOT apply_filters) ───────




def signals_score(candidate: dict) -> float:
    """
    Return a float multiplier in [0.50, 1.40] derived from redrob_signals.
    Higher = more hireable / more active / stronger verified signal.
    Does NOT affect hard filtering — used by ranker.py to adjust cosine scores.
    """
    signals = candidate.get("redrob_signals", {})
    multiplier = 1.0

    # ── Activity / availability ────────────────────────────────────────────
    last_active = _parse_date(signals.get("last_active_date"))
    days_inactive = _days_since(last_active) or 0
    if days_inactive > 180:
        multiplier *= 0.60
    elif days_inactive > 90:
        multiplier *= 0.80
    elif days_inactive < 30:
        multiplier *= 1.05

    if signals.get("open_to_work_flag"):
        multiplier *= 1.10
    if (signals.get("applications_submitted_30d") or 0) >= 3:
        multiplier *= 1.05

    avg_response = signals.get("avg_response_time_hours") or 0
    if avg_response > 168:
        multiplier *= 0.85
    elif avg_response < 24:
        multiplier *= 1.05

    # ── Notice period ──────────────────────────────────────────────────────
    notice = signals.get("notice_period_days") or 0
    if notice <= 30:
        multiplier *= 1.10
    elif notice <= 60:
        multiplier *= 1.00
    elif notice <= 90:
        multiplier *= 0.95
    else:
        multiplier *= 0.85

    # ── Location / work-mode fit ───────────────────────────────────────────
    if signals.get("willing_to_relocate"):
        multiplier *= 1.05
    work_mode = signals.get("preferred_work_mode") or ""
    if work_mode in ("hybrid", "flexible"):
        multiplier *= 1.05
    elif work_mode == "remote":
        multiplier *= 0.95

    # ── GitHub activity (engineering proof) ───────────────────────────────
    gh = signals.get("github_activity_score")
    if gh is None or gh == -1:
        pass  # no GitHub linked — neutral
    elif gh >= 70:
        multiplier *= 1.20
    elif gh >= 40:
        multiplier *= 1.10
    elif gh >= 15:
        multiplier *= 1.02
    else:
        multiplier *= 0.90  # linked but inactive

    # ── Platform skill assessment scores ──────────────────────────────────
    assessed = {k.lower(): v for k, v in (signals.get("skill_assessment_scores") or {}).items()}
    relevant_scores = [v for k, v in assessed.items() if k in RELEVANT_ASSESSED]
    if relevant_scores:
        avg_score = sum(relevant_scores) / len(relevant_scores)
        if avg_score >= 75:
            multiplier *= 1.15
        elif avg_score >= 55:
            multiplier *= 1.05
        elif avg_score < 40:
            multiplier *= 0.85

    # ── Recruiter behaviour signals ────────────────────────────────────────
    rr = signals.get("recruiter_response_rate") or 0
    if rr >= 0.7:
        multiplier *= 1.08
    elif rr >= 0.4:
        multiplier *= 1.02
    elif rr < 0.2:
        multiplier *= 0.80

    icr = signals.get("interview_completion_rate") or 0
    if icr >= 0.85:
        multiplier *= 1.05
    elif icr < 0.50 and icr > 0:  # exclude 0 = no history
        multiplier *= 0.88

    if (signals.get("saved_by_recruiters_30d") or 0) >= 5:
        multiplier *= 1.08
    elif (signals.get("saved_by_recruiters_30d") or 0) >= 2:
        multiplier *= 1.03

    # ── Profile quality ────────────────────────────────────────────────────
    completeness = signals.get("profile_completeness_score") or 0
    if completeness >= 85:
        multiplier *= 1.03
    elif completeness < 50:
        multiplier *= 0.92

    # ── Offer history (bonus) ──────────────────────────────────────────────
    oar = signals.get("offer_acceptance_rate")
    if oar is not None and oar >= 0:  # -1 means no offer history
        if oar >= 0.70:
            multiplier *= 1.05
        elif oar < 0.25:
            multiplier *= 0.90

    # ── Identity / social proof (bonus) ───────────────────────────────────
    if signals.get("linkedin_connected"):
        multiplier *= 1.03

    endorsements = signals.get("endorsements_received") or 0
    if endorsements >= 50:
        multiplier *= 1.03
    elif endorsements < 5:
        multiplier *= 0.95

    return round(min(max(multiplier, 0.50), 1.40), 4)


# ── Quick test harness ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import os
    from collections import Counter

    # Resolve candidates.jsonl relative to this script's location.
    # Tries redrob_ranker/data/ first, then the hackathon root two levels up.
    _here = os.path.dirname(os.path.abspath(__file__))
    _data = os.path.join(_here, "..", "data", "candidates.jsonl")
    if not os.path.exists(_data):
        _data = os.path.join(_here, "..", "..", "candidates.jsonl")

    LIMIT = 1000
    total = 0
    passed = 0
    reject_reasons: Counter = Counter()

    print(f"Running filters on first {LIMIT} candidates from:\n  {_data}\n")

    with open(_data, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= LIMIT:
                break
            c = json.loads(line)
            ok, reason = apply_filters(c)
            total += 1
            if ok:
                passed += 1
            else:
                # Bucket reason to the first colon-delimited tag
                tag = reason.split(":")[0]
                reject_reasons[tag] += 1
                print(f"  {c['candidate_id']}  REJECTED  {reason}")

    print("\n" + "=" * 60)
    print(f"  FILTER STATISTICS  ({total} candidates)")
    print("=" * 60)
    print(f"  Passed  : {passed:>5d}  ({100*passed/total:.1f}%)")
    print(f"  Rejected: {total - passed:>5d}  ({100*(total-passed)/total:.1f}%)")
    print("\n  Rejection breakdown:")
    for reason, cnt in reject_reasons.most_common():
        bar = "#" * min(40, int(40 * cnt / max(reject_reasons.values())))
        print(f"    {reason:<35s} {cnt:>4d}  {bar}")
