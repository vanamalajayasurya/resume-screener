import io
import os

os.environ["DATABASE_URL"] = "sqlite:///./test_screener.db"
os.environ.pop("ANTHROPIC_API_KEY", None)

import pytest
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

from app import scoring, skills
from app.main import app

JD = """Junior Python Developer
We need 1+ years of experience. Required: Python, FastAPI, SQL, REST APIs, Git.
Nice to have: Docker, React."""


def make_pdf(lines):
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    y = 800
    for line in lines:
        c.drawString(50, y, line)
        y -= 18
    c.save()
    return buf.getvalue()


def test_skill_extraction_and_split():
    req, nice = skills.split_jd_skills(JD)
    assert {"python", "fastapi", "sql", "git", "rest apis"} <= set(req)
    assert set(nice) == {"docker", "react"}
    assert skills.jd_min_years(JD) == 1.0


def test_c_plus_plus_not_confused_with_c():
    assert "c++" in skills.extract_skills("Experienced in C++ and Python")


def test_score_is_deterministic_and_ordered():
    req, nice = skills.split_jd_skills(JD)
    strong = scoring.score_candidate("python fastapi sql git rest docker react", ["python", "fastapi", "sql", "git", "rest apis", "docker", "react"], 2, JD, req, nice, 1)
    weak = scoring.score_candidate("html css", ["html", "css"], 0, JD, req, nice, 1)
    again = scoring.score_candidate("python fastapi sql git rest docker react", ["python", "fastapi", "sql", "git", "rest apis", "docker", "react"], 2, JD, req, nice, 1)
    assert strong["score"] == again["score"]
    assert strong["score"] > 80 > 40 > weak["score"]
    assert "python" in weak["missing"]


@pytest.fixture(scope="module")
def client():
    from app.db import engine
    with TestClient(app) as c:
        yield c
    engine.dispose()
    if os.path.exists("test_screener.db"):
        os.remove("test_screener.db")


def test_analyze_end_to_end(client):
    pdf = make_pdf([
        "Jane Doe", "Junior developer with 2 years of experience.",
        "Skills: Python, FastAPI, SQLite, Git, REST APIs, Docker",
        "Built an internal API used by 3 teams. Experience 2022 - 2024 at Acme Corp.",
    ])
    r = client.post("/analyze", files={"file": ("jane.pdf", pdf, "application/pdf")},
                    data={"job_title": "Junior Python Dev", "job_description": JD})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["candidate"]["name"] == "Jane Doe"
    assert body["score"] > 70
    assert "react" not in body["matched"]
    assert set(body["missing_nice"]) == {"react"}  # docker is on the resume
    assert len(body["questions"]) >= 5
    assert client.get(f"/analyses/{body['id']}").json()["score"] == body["score"]
    assert client.get("/analyses").status_code == 200


def test_rejects_non_pdf_and_scanned(client):
    r = client.post("/analyze", files={"file": ("x.txt", b"hi", "text/plain")}, data={"job_description": JD})
    assert r.status_code == 415
    empty = make_pdf([])
    r = client.post("/analyze", files={"file": ("scan.pdf", empty, "application/pdf")}, data={"job_description": JD})
    assert r.status_code == 422 and "scanned" in r.json()["detail"].lower()


RESUME = """EDUCATION
Some College 2022 - 2026
PROFESSIONAL EXPERIENCE
Acme Apr 2026 - Jun 2026
Data Intern
Other Corp Apr 2025 - May 2025
Web Intern
CERTIFICATIONS
Cert Program 2025-2026
"""


def test_experience_ignores_education_and_certifications():
    assert skills.resume_years(RESUME) == 0.4  # 3 + 2 months = 5 months


def test_experience_merges_overlaps_and_present():
    txt = "WORK EXPERIENCE\nA Jan 2020 - Dec 2020\nB Jun 2020 - Mar 2021\n"
    assert skills.resume_years(txt) == 1.2  # Jan 2020 .. Mar 2021 = 15 months


def test_llm_aliases():
    got = skills.extract_skills("Used OpenAI API, Gemini API, RAG and prompt engineering")
    assert "llm" in got
    assert "llm" not in skills.extract_skills("I like to drag and drop")


def test_demo_mode_stores_nothing_and_blocks_history(client, monkeypatch):
    from sqlalchemy import func, select
    from app.db import Analysis, Candidate, SessionLocal
    with SessionLocal() as db:
        before = (db.scalar(select(func.count()).select_from(Candidate)),
                  db.scalar(select(func.count()).select_from(Analysis)))
    monkeypatch.setenv("DEMO_MODE", "1")
    pdf = make_pdf(["Demo Dan", "Skills: Python, FastAPI, SQL, Git, REST APIs", "Some project work done during college studies here."])
    r = client.post("/analyze", files={"file": ("d.pdf", pdf, "application/pdf")},
                    data={"job_title": "Dev", "job_description": JD})
    assert r.status_code == 201 and r.json()["score"] > 0
    assert client.get("/analyses").status_code == 403
    assert client.get("/jobs").status_code == 403
    assert client.get("/health").json()["demo_mode"] is True
    with SessionLocal() as db:
        after = (db.scalar(select(func.count()).select_from(Candidate)),
                 db.scalar(select(func.count()).select_from(Analysis)))
    assert before == after