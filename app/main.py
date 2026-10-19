import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from sqlalchemy.orm import Session

from . import llm, parser, scoring, skills
from .db import Analysis, Candidate, Job, get_db, init_db

MAX_BYTES = 5 * 1024 * 1024
STATIC = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title="AI Resume Screening Assistant", version="1.0.0", lifespan=lifespan)


def _demo() -> bool:
    """Public-demo mode: nothing is stored and history/ranking endpoints are disabled."""
    return os.getenv("DEMO_MODE") == "1"


def _block_in_demo():
    if _demo():
        raise HTTPException(403, "Disabled in public demo mode (no data is stored).")


def _jd_profile(description: str) -> tuple[list[str], list[str], float | None]:
    required, nice = skills.split_jd_skills(description)
    extra = [skills.canonicalize(s) for s in llm.extract_extra_skills(description, "Job description")]
    required = sorted(set(required) | {s for s in extra if s not in nice})
    return required, nice, skills.jd_min_years(description)


def _job_for(db: Session, title: str, description: str, demo: bool = False) -> Job:
    """Reuse a saved job with the same description, so repeated checks of one job
    end up in the same ranking instead of creating a new job every time."""
    if not demo:
        existing = db.scalar(select(Job).where(Job.description == description))
        if existing:
            return existing
    req, nice, yrs = _jd_profile(description)
    job = Job(title=(title or "").strip() or "Untitled role", description=description,
              required_skills=req, nice_skills=nice, min_years=yrs)
    if not demo:
        db.add(job)
        db.flush()
    return job


def _job_out(j: Job) -> dict:
    return {"id": j.id, "title": j.title, "required_skills": j.required_skills,
            "nice_skills": j.nice_skills, "min_years": j.min_years, "created_at": j.created_at}


def _analysis_out(a: Analysis) -> dict:
    return {
        "id": a.id, "job": {"id": a.job.id, "title": a.job.title},
        "candidate": {"id": a.candidate.id, "name": a.candidate.name, "filename": a.candidate.filename},
        "score": a.score, "label": a.breakdown.get("label"), "breakdown": a.breakdown,
        "matched": a.matched, "missing": a.missing,
        "missing_nice": a.breakdown.get("missing_nice", []), "summary": a.summary,
        "questions": a.questions, "created_at": a.created_at,
        "disclaimer": "Decision support only. Not an automated hiring decision.",
    }


@app.get("/health")
def health():
    return {"status": "ok", "llm_enabled": llm.enabled(), "demo_mode": _demo()}


@app.post("/jobs", status_code=201)
def create_job(title: str = Form(...), description: str = Form(...), db: Session = Depends(get_db)):
    _block_in_demo()
    if len(description.strip()) < 30:
        raise HTTPException(422, "Job description is too short.")
    job = _job_for(db, title, description)
    db.commit()
    return _job_out(job)


@app.get("/jobs")
def list_jobs(db: Session = Depends(get_db)):
    _block_in_demo()
    return [_job_out(j) for j in db.scalars(select(Job).order_by(Job.id.desc()))]


@app.post("/analyze", status_code=201)
async def analyze(
    file: UploadFile = File(...),
    job_id: int | None = Form(None),
    job_title: str = Form("Untitled role"),
    job_description: str | None = Form(None),
    db: Session = Depends(get_db),
):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(415, "Only PDF resumes are supported.")
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "File too large (max 5 MB).")

    demo = _demo()
    if job_id is not None:
        _block_in_demo()
        job = db.get(Job, job_id)
        if not job:
            raise HTTPException(404, "Job not found.")
    elif job_description and len(job_description.strip()) >= 30:
        job = _job_for(db, job_title, job_description, demo)
    else:
        raise HTTPException(422, "Provide job_id or a job_description (30+ characters).")

    try:
        text = parser.extract_text(data)
    except parser.ResumeParseError as e:
        raise HTTPException(422, str(e))

    r_skills = set(skills.extract_skills(text))
    r_skills |= {skills.canonicalize(s) for s in llm.extract_extra_skills(text, "Resume")}
    years = skills.resume_years(text)
    name = parser.guess_name(text, file.filename or "")

    result = scoring.score_candidate(text, sorted(r_skills), years, job.description,
                                     job.required_skills, job.nice_skills, job.min_years)
    report = llm.build_report({
        "candidate": name, "job_title": job.title, "score": result["score"], "label": result["label"],
        "matched": result["matched"], "missing": result["missing"],
        "missing_nice": result["missing_nice"], "extra_skills": result["extra"][:15],
        "candidate_years": years, "required_years": job.min_years,
    })

    cand = Candidate(name=name, filename=file.filename, resume_text="" if demo else text,
                     skills=sorted(r_skills), years_experience=years)
    fields = dict(score=result["score"],
                  breakdown={**result["breakdown"], "label": result["label"],
                             "missing_nice": result["missing_nice"]},
                  matched=result["matched"], missing=result["missing"],
                  summary=report["summary"], questions=report["questions"])
    if demo:  # never touch the database: nothing about the resume is stored
        return _analysis_out(Analysis(job=job, candidate=cand, **fields))
    db.add(cand)
    db.flush()
    analysis = Analysis(job_id=job.id, candidate_id=cand.id, **fields)
    db.add(analysis)
    db.commit()
    return _analysis_out(analysis)


@app.get("/analyses")
def list_analyses(job_id: int | None = None, db: Session = Depends(get_db)):
    _block_in_demo()
    q = select(Analysis).order_by(Analysis.score.desc() if job_id else Analysis.id.desc())
    if job_id:
        q = q.where(Analysis.job_id == job_id)
    return [_analysis_out(a) for a in db.scalars(q)]


@app.get("/analyses/{analysis_id}")
def get_analysis(analysis_id: int, db: Session = Depends(get_db)):
    _block_in_demo()
    a = db.get(Analysis, analysis_id)
    if not a:
        raise HTTPException(404, "Analysis not found.")
    return _analysis_out(a)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")