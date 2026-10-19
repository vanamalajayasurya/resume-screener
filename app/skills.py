"""Deterministic skill extraction from a taxonomy, plus experience-year helpers."""
import re
from datetime import date

# canonical name -> aliases (all lowercase)
TAXONOMY: dict[str, list[str]] = {
    "python": ["python"], "java": ["java"], "javascript": ["javascript", "js", "es6"],
    "typescript": ["typescript"], "c++": ["c++"], "c#": ["c#"], "go": ["golang"],
    "sql": ["sql", "mysql", "postgresql", "postgres", "sqlite"], "nosql": ["nosql", "mongodb"],
    "html": ["html", "html5"], "css": ["css", "css3"], "react": ["react", "react.js", "reactjs"],
    "node.js": ["node.js", "nodejs", "node"], "flask": ["flask"], "fastapi": ["fastapi"],
    "django": ["django"], "rest apis": ["rest", "rest api", "rest apis", "restful"],
    "git": ["git", "github", "gitlab"], "docker": ["docker"], "kubernetes": ["kubernetes", "k8s"],
    "aws": ["aws", "amazon web services"], "azure": ["azure"], "gcp": ["gcp", "google cloud"],
    "linux": ["linux"], "ci/cd": ["ci/cd", "cicd", "github actions", "jenkins"],
    "pandas": ["pandas"], "numpy": ["numpy"], "scikit-learn": ["scikit-learn", "sklearn"],
    "machine learning": ["machine learning", "ml"], "deep learning": ["deep learning"],
    "nlp": ["nlp", "natural language processing"], "llm": ["llm", "llms", "large language model", "openai api", "gemini api", "rag", "prompt engineering"],
    "tensorflow": ["tensorflow"], "pytorch": ["pytorch"], "data analysis": ["data analysis", "data analytics"],
    "power bi": ["power bi", "powerbi"], "tableau": ["tableau"], "excel": ["excel"],
    "statistics": ["statistics", "statistical"], "data visualization": ["data visualization", "data visualisation"],
    "figma": ["figma"], "ui/ux": ["ui/ux", "ux", "ui design"], "agile": ["agile", "scrum"],
    "testing": ["pytest", "unit testing", "jest", "unit tests"], "tailwind": ["tailwind", "tailwindcss"],
    "redux": ["redux"], "graphql": ["graphql"], "spark": ["spark", "pyspark"], "etl": ["etl"],
}

_PATTERNS = {
    canon: re.compile(
        r"(?<![\w+#.])(" + "|".join(re.escape(a) for a in sorted(aliases, key=len, reverse=True)) + r")(?![\w+#])",
        re.I,
    )
    for canon, aliases in TAXONOMY.items()
}
_ALIAS_TO_CANON = {a: c for c, al in TAXONOMY.items() for a in al}

_NICE_MARKER = re.compile(
    r"(nice[- ]to[- ]have|preferred|good to have|bonus|a plus|is a plus|desirable|optional)", re.I
)


def canonicalize(skill: str) -> str:
    s = skill.strip().lower()
    return _ALIAS_TO_CANON.get(s, s)


def extract_skills(text: str) -> list[str]:
    return sorted(c for c, p in _PATTERNS.items() if p.search(text))


def split_jd_skills(jd: str) -> tuple[list[str], list[str]]:
    """Skills before a 'nice to have / preferred' marker are required; after it, nice-to-have."""
    m = _NICE_MARKER.search(jd)
    if not m:
        return extract_skills(jd), []
    required = extract_skills(jd[: m.start()])
    nice = [s for s in extract_skills(jd[m.start():]) if s not in required]
    return required, nice


def jd_min_years(jd: str) -> float | None:
    vals = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)", jd, re.I)]
    return min(vals) if vals else None


_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}
_HEADING = re.compile(r"^[A-Z][A-Z &/,\-]{2,40}$")
_EXPERIENCE_HEAD = re.compile(r"experience|employment|internship|work history", re.I)
_SKIP_HEAD = re.compile(r"education|academic|certification|course|award|achievement", re.I)
_RANGE = re.compile(
    r"(?:(?P<sm>[A-Za-z]{3})[a-z]*\.?\s+)?(?P<sy>(?:19|20)\d{2})\s*(?:-|\u2013|\u2014|to)\s*"
    r"(?:(?P<em>[A-Za-z]{3})[a-z]*\.?\s+)?(?P<ey>(?:19|20)\d{2}|present|current)",
    re.I,
)


def _sections(text: str) -> list[tuple[str, str]]:
    """Split a resume into (HEADING, body) using ALL-CAPS heading lines."""
    parts, head, buf = [], "", []
    for line in text.splitlines():
        if _HEADING.match(line.strip()):
            parts.append((head, "\n".join(buf)))
            head, buf = line.strip(), []
        else:
            buf.append(line)
    parts.append((head, "\n".join(buf)))
    return parts


def _work_text(text: str) -> str:
    """Prefer experience sections; otherwise drop education/certification sections."""
    secs = _sections(text)
    exp = [body for head, body in secs if _EXPERIENCE_HEAD.search(head)]
    if exp:
        return "\n".join(exp)
    return "\n".join(body for head, body in secs if not _SKIP_HEAD.search(head))


def _month(token: str | None, default: int) -> int:
    return _MONTHS.get((token or "").lower()[:3], default)


def resume_years(text: str) -> float | None:
    """Explicit 'N years of experience' wins; otherwise sum work date ranges (month-aware,
    overlaps merged, education/certification sections ignored)."""
    stated = [float(x) for x in re.findall(
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)\s+(?:of\s+)?(?:experience|exp)", text, re.I)]
    if stated:
        return max(stated)

    today = date.today()
    spans = []
    for m in _RANGE.finditer(_work_text(text)):
        sy = int(m["sy"])
        sm = _month(m["sm"], 1)
        if m["ey"].lower() in ("present", "current"):
            ey, em = today.year, today.month
        else:
            ey = int(m["ey"])
            em = _month(m["em"], 12)
        start, end = sy * 12 + sm, ey * 12 + em  # inclusive months
        if 0 <= end - start <= 15 * 12:
            spans.append((start, end))

    spans.sort()
    total, cur_s, cur_e = 0, None, None
    for a, b in spans:
        if cur_e is None or a > cur_e + 1:
            if cur_e is not None:
                total += cur_e - cur_s + 1
            cur_s, cur_e = a, b
        else:
            cur_e = max(cur_e, b)
    if cur_e is not None:
        total += cur_e - cur_s + 1
    return round(total / 12, 1) if total else None