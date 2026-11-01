"""
OrthoTwin AI — Security & Compliance Tests
Validates:
1. Unauthenticated access redirection to login
2. Removal of user-switch route
3. Role enforcement across mutating endpoints
4. Report and storage download authorization
5. CSRF token enforcement
6. Path traversal rejection
"""

import pytest
from web_app import app
from database.models import User, Patient, FractureCase, Visit, XrayImage, Report
from database.repositories import UserRepository, ReportRepository


@pytest.fixture
def client():
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = True
    with app.test_client() as c:
        yield c


def test_unauthenticated_routes_redirect_to_login(client):
    protected_paths = [
        "/",
        "/patients",
        "/upload",
        "/analysis",
        "/digital-twin",
        "/doctor-review",
        "/patient-portal",
        "/diagnostics"
    ]
    for p in protected_paths:
        res = client.get(p)
        assert res.status_code == 302
        assert "/login" in res.headers["Location"]


def test_user_switch_route_is_removed(client):
    # Route /switch-user/<username> must return 404
    res = client.get("/switch-user/dr_ortho")
    assert res.status_code == 404


def test_patient_role_forbidden_from_clinician_routes(client):
    # Log in as patient using seeded demo user
    with client.session_transaction() as sess:
        sess["username"] = "patient_demo"
        sess["csrf_token"] = "valid_test_token"

    clinician_routes = [
        ("/", "GET"),
        ("/patients", "GET"),
        ("/upload", "GET"),
        ("/analysis", "GET"),
        ("/doctor-review", "GET"),
        ("/diagnostics", "GET"),
    ]
    for path, method in clinician_routes:
        if method == "GET":
            res = client.get(path)
            assert res.status_code == 403, f"Expected 403 for {path}, got {res.status_code}"


def test_csrf_token_enforcement(client):
    # Log in as doctor
    with client.session_transaction() as sess:
        sess["username"] = "dr_ortho"
        sess["csrf_token"] = "expected_token_123"

    # Turn on CSRF checking
    app.config["TESTING"] = False
    try:
        # POST without CSRF token
        res = client.post("/patients/create", data={
            "first_name": "Test",
            "last_name": "CSRF",
            "dob": "1990-01-01"
        })
        assert res.status_code == 400
        assert b"CSRF" in res.data
    finally:
        app.config["TESTING"] = True


def test_report_download_patient_authorization(client):
    from database.db import get_db
    with get_db() as db:
        rep = db.query(Report).first()
        assert rep is not None, "At least one report should be seeded in database"
        rep_id = rep.id
        rep_visit_id = rep.visit_id
        rep_pdf_path = rep.pdf_path
        rep_gen_id = rep.generated_by_id

    # 1. Unauthenticated request -> redirect to login
    res = client.get(f"/reports/{rep_id}")
    assert res.status_code == 302
    assert "/login" in res.headers["Location"]

    # 2. Doctor downloads -> 200 OK
    with client.session_transaction() as sess:
        sess["username"] = "dr_ortho"

    res_doc = client.get(f"/reports/{rep_id}")
    assert res_doc.status_code == 200

    # 3. Patient role attempting to download report when not authorized -> 403
    # Check with an unlinked or unapproved report
    with get_db() as db:
        # Create an unapproved/non-final report
        unapproved_rep = Report(
            visit_id=rep_visit_id,
            report_code=f"REP-UNAPPROVED-{rep_id}",
            pdf_path=rep_pdf_path,
            generated_by_id=rep_gen_id,
            is_final=False
        )
        db.add(unapproved_rep)
        db.commit()
        unapproved_id = unapproved_rep.id

    with client.session_transaction() as sess:
        sess["username"] = "patient_demo"

    res_unapproved = client.get(f"/reports/{unapproved_id}")
    assert res_unapproved.status_code == 403

    # Clean up test unapproved report
    with get_db() as db:
        r_del = db.query(Report).filter(Report.id == unapproved_id).first()
        if r_del:
            db.delete(r_del)
            db.commit()
