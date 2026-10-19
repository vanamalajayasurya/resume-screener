# AI Resume Screening Assistant

Upload a resume PDF, compare it to a job description, and get an explainable suitability score, a skill gap analysis and tailored interview questions.

**Stack:** Python, FastAPI, SQLite (SQLAlchemy), pdfplumber, scikit-learn, optional LLM (Anthropic API), vanilla JS frontend.

## Why the score is trustworthy
The score is computed in code, never by the LLM, so the same inputs always give the same result:

| Component | Weight |
|---|---|
| Required skill overlap | 60% |
| Nice-to-have skill overlap | 15% |
| Experience vs. requirement | 15% |
| TF-IDF semantic similarity | 10% |

Components that don't apply to a JD (e.g. no stated years) are dropped and the weights renormalised. The LLM only extracts extra skills, writes the summary and generates interview questions from the computed gaps. Without an API key the app falls back to a built-in skill taxonomy and templated questions.

## Run it
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # optional: add ANTHROPIC_API_KEY
export $(grep -v '^#' .env | xargs)   # or set the variables yourself
uvicorn app.main:app --reload
```
Open http://localhost:8000 (UI) or http://localhost:8000/docs (API).

## API
| Method | Path | Purpose |
|---|---|---|
| POST | `/analyze` | multipart: `file` (PDF) + `job_id` **or** `job_title` + `job_description` |
| POST | `/jobs` | save a job (form: `title`, `description`) |
| GET | `/jobs` | list jobs |
| GET | `/analyses?job_id=1` | history; with `job_id` returns a ranked candidate list |
| GET | `/analyses/{id}` | one analysis |
| GET | `/health` | status and whether the LLM is enabled |

## Tests
```bash
pip install reportlab
pytest -q
```

## Known limitations
- Scanned/image PDFs are detected and rejected with a clear message (add OCR to support them).
- Two-column layouts can scramble text order; skill matching is order-insensitive so impact is small.
- The taxonomy in `app/skills.py` is easy to extend; the LLM covers skills outside it when a key is set.
- **Decision support only.** Scores should never be used as an automatic rejection. Review resumes manually and be aware of bias in any automated screening.
