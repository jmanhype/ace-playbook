from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_app_has_required_landing_copy_and_human_gate():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert (
        "You care for the family. We organise the approved paperwork after the arrangement."
        in source
    )
    assert '"Approved by (full name)"' in source
    assert '"Approve & Export"' in source
    assert "disabled=approval_disabled" in source


def test_exports_are_only_rendered_from_approved_pack():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "if not isinstance(pack, ApprovedPack):" in source
    assert source.count("st.download_button(") == 2  # one PDF plus a loop for five text files
    assert "build_text_exports(pack)" in source
    assert "build_pdf_bytes(pack)" in source


def test_no_case_persistence_or_sending_integrations_are_imported():
    source = (ROOT / "app.py").read_text(encoding="utf-8").lower()
    forbidden_imports = (
        "import sqlite3",
        "import sqlalchemy",
        "import requests",
        "import smtplib",
        "from twilio",
        "import boto3",
        "import logging",
    )
    assert not any(item in source for item in forbidden_imports)
