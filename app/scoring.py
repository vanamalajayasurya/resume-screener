"""Reproducible suitability score. The LLM never produces the number."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

WEIGHTS = {"required": 0.60, "nice": 0.15, "experience": 0.15, "semantic": 0.10}


def semantic_similarity(resume: str, jd: str) -> float:
    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform([resume, jd])
        return float(cosine_similarity(vec[0], vec[1])[0][0])
    except ValueError:  # empty vocabulary
        return 0.0


def _ratio(matched: list, total: list) -> float:
    return len(matched) / len(total) if total else 1.0


def score_candidate(
    resume_text: str,
    resume_skills: list[str],
    resume_years: float | None,
    jd_text: str,
    required: list[str],
    nice: list[str],
    min_years: float | None,
) -> dict:
    rs = set(resume_skills)
    matched_req = [s for s in required if s in rs]
    missing_req = [s for s in required if s not in rs]
    matched_nice = [s for s in nice if s in rs]
    missing_nice = [s for s in nice if s not in rs]

    exp = 1.0 if not min_years else min((resume_years or 0) / min_years, 1.0)
    sim = min(semantic_similarity(resume_text, jd_text) * 2, 1.0)  # TF-IDF cosine rarely exceeds ~0.5

    parts = {
        "required": _ratio(matched_req, required),
        "nice": _ratio(matched_nice, nice),
        "experience": exp,
        "semantic": sim,
    }
    # Drop components that don't apply to this JD, then renormalise weights.
    active = {k: w for k, w in WEIGHTS.items()
              if not (k == "required" and not required) and not (k == "nice" and not nice)
              and not (k == "experience" and not min_years)}
    total_w = sum(active.values())
    score = 100 * sum(parts[k] * w for k, w in active.items()) / total_w

    return {
        "score": round(score, 1),
        "label": label_for(score),
        "breakdown": {
            "required_skills": {"matched": len(matched_req), "total": len(required), "points": round(parts["required"] * 100)},
            "nice_to_have": {"matched": len(matched_nice), "total": len(nice), "points": round(parts["nice"] * 100)},
            "experience": {"candidate_years": resume_years, "required_years": min_years, "points": round(exp * 100)},
            "semantic_match": {"points": round(sim * 100)},
            "weights": {k: round(w / total_w, 2) for k, w in active.items()},
        },
        "matched": matched_req + matched_nice,
        "missing": missing_req,
        "missing_nice": missing_nice,
        "extra": sorted(rs - set(required) - set(nice)),
    }


def label_for(score: float) -> str:
    if score >= 80:
        return "Strong match"
    if score >= 60:
        return "Good match"
    if score >= 40:
        return "Partial match"
    return "Weak match"
