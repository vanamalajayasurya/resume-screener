"""Optional LLM layer. Everything degrades gracefully to heuristics when no API key is set."""
import json
import os
import re

MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5-5")


def enabled() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def _client():
    import anthropic
    return anthropic.Anthropic()


def _json_from(text: str):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    return json.loads(text)


def extract_extra_skills(text: str, kind: str) -> list[str]:
    """Ask the LLM for technical skills the taxonomy might miss. Returns [] on any failure."""
    if not enabled():
        return []
    try:
        msg = _client().messages.create(
            model=MODEL, max_tokens=600,
            system="Extract concrete technical skills/tools/technologies. Reply with ONLY a JSON array of short lowercase strings.",
            messages=[{"role": "user", "content": f"{kind}:\n\n{text[:6000]}"}],
        )
        data = _json_from(msg.content[0].text)
        return [str(s).lower().strip() for s in data if isinstance(s, str)][:40]
    except Exception:
        return []


def build_report(ctx: dict) -> dict:
    """Return {'summary': str, 'questions': [..]} grounded in the computed score context."""
    if enabled():
        try:
            msg = _client().messages.create(
                model=MODEL, max_tokens=1200,
                system=(
                    "You are a recruiting assistant. The numeric score is already computed; do not change it. "
                    "Write decision support, never an auto-reject. Reply with ONLY JSON: "
                    '{"summary": "2-3 sentences", "questions": ["8 interview questions mixing '
                    'verification of matched skills and probing of gaps"]}'
                ),
                messages=[{"role": "user", "content": json.dumps(ctx)}],
            )
            data = _json_from(msg.content[0].text)
            if data.get("summary") and data.get("questions"):
                return {"summary": data["summary"], "questions": data["questions"][:10]}
        except Exception:
            pass
    return fallback_report(ctx)


def fallback_report(ctx: dict) -> dict:
    m, miss, role = ctx["matched"], ctx["missing"], ctx["job_title"]
    summary = (
        f"{ctx['candidate']} scores {ctx['score']}/100 ({ctx['label']}) for {role}. "
        f"Matched {len(m)} skills"
        + (f"; missing required: {', '.join(miss)}." if miss else "; no required skills missing.")
        + " Treat this as decision support, not a final decision."
    )
    qs = [f"Tell me about a project where you used {s}. What was your specific contribution?" for s in m[:3]]
    qs += [f"You don't list {s} on your resume. How have you worked with it, or how would you ramp up?" for s in miss[:3]]
    qs += [
        f"Walk me through the most challenging problem you solved that is relevant to {role}.",
        "Describe a time you had to learn a new technology quickly under a deadline.",
        "How do you test and validate your work before shipping?",
    ]
    return {"summary": summary, "questions": qs[:8]}
