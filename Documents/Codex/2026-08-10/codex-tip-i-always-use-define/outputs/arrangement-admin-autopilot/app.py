"""Streamlit UI for Arrangement Admin Autopilot."""

from __future__ import annotations

import os
from typing import Optional

import streamlit as st
from anthropic import Anthropic

from core import (
    ARTIFACTS,
    ARTIFACT_KEYS,
    DEFAULT_MODEL,
    ApprovedPack,
    ValidationError,
    build_pdf_bytes,
    build_text_exports,
    can_approve,
    contains_unknown,
    create_approved_pack,
    generate_drafts,
    highlight_unknown_html,
    normalise_approver_name,
    prepare_notes,
    verify_password,
)


def get_secret(name: str) -> Optional[str]:
    """Read a secret without displaying or logging it."""

    value = None
    try:
        value = st.secrets.get(name)
    except (FileNotFoundError, KeyError):
        pass
    if value is None:
        value = os.getenv(name)
    if value is None:
        return None
    clean = str(value).strip()
    return clean or None


def clear_approval() -> None:
    """Remove every approved export whenever review state changes."""

    st.session_state.pop("approved_pack", None)


def require_pilot_password(expected_password: Optional[str]) -> None:
    """Stop unauthenticated sessions before any case notes can be entered."""

    if not expected_password:
        st.error(
            "This pilot is locked because APP_PASSWORD has not been configured. "
            "Ask the app administrator to set it."
        )
        st.stop()

    if st.session_state.get("authenticated") is True:
        return

    st.subheader("Pilot access")
    with st.form("pilot_login", clear_on_submit=True):
        candidate = st.text_input(
            "Pilot password",
            type="password",
            autocomplete="current-password",
        )
        submitted = st.form_submit_button("Unlock")
    if submitted:
        if verify_password(candidate, expected_password):
            st.session_state["authenticated"] = True
            st.rerun()
        st.error("That password was not accepted.")
    st.stop()


def initialise_state() -> None:
    """Initialise session-only workflow state."""

    st.session_state.setdefault("drafts", None)
    st.session_state.setdefault("approver_name", "")
    st.session_state.setdefault("unknown_acknowledged", False)


def install_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {max-width: 1120px; padding-top: 2rem;}
        .safety-note {padding: .85rem 1rem; border-left: 4px solid #6b7280;
          background: #f7f7f5; border-radius: .25rem; margin: 1rem 0;}
        .draft-preview {border: 1px solid #d8d6d1; background: #fcfcfa;
          border-radius: .4rem; padding: .9rem; margin: .4rem 0 .8rem;
          white-space: normal; line-height: 1.5;}
        .unknown-token {background: #ffe08a; color: #6d2500; padding: 0 .18rem;
          border: 1px solid #e0a000; border-radius: .18rem; font-weight: 700;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_intake() -> None:
    st.header("Arrangement notes")
    st.write("Paste the meeting notes, upload a text file, or use both.")
    typed_notes = st.text_area(
        "Typed arrangement notes",
        height=240,
        placeholder="Enter the notes exactly as recorded. Missing details can remain missing.",
    )
    uploaded = st.file_uploader(
        "Optional notes file (.txt or .md)",
        type=["txt", "md"],
        accept_multiple_files=False,
    )

    if st.button("Generate Drafts", type="primary", width="stretch"):
        uploaded_name = uploaded.name if uploaded is not None else None
        uploaded_bytes = uploaded.getvalue() if uploaded is not None else None
        try:
            notes = prepare_notes(typed_notes, uploaded_name, uploaded_bytes)
        except ValidationError as exc:
            st.error(str(exc))
            return

        api_key = get_secret("ANTHROPIC_API_KEY")
        if not api_key:
            st.error(
                "Drafts cannot be generated because the Anthropic API key has not been "
                "configured. Ask the app administrator to set it."
            )
            return

        model = get_secret("ANTHROPIC_MODEL") or DEFAULT_MODEL
        try:
            with st.spinner("Preparing the five drafts for review…"):
                client = Anthropic(api_key=api_key, timeout=90.0, max_retries=1)
                drafts = generate_drafts(client, notes, model=model)
        except ValidationError:
            st.error(
                "Claude did not return a complete, safe set of five drafts. No drafts "
                "were saved. Please try again or ask the administrator for help."
            )
            return
        except Exception:
            # Avoid showing provider response bodies or credentials in a sensitive pilot.
            st.error(
                "The drafting service could not complete this request. No drafts were "
                "saved. Please try again later."
            )
            return

        st.session_state["drafts"] = drafts
        for key in ARTIFACT_KEYS:
            st.session_state[f"edit_{key}"] = drafts[key]
        st.session_state["approver_name"] = ""
        st.session_state["unknown_acknowledged"] = False
        clear_approval()
        st.rerun()


def render_review() -> None:
    stored = st.session_state.get("drafts")
    if not stored:
        return

    st.divider()
    st.header("Review all five drafts")
    st.write(
        "Correct every draft before approval. Highlighted UNKNOWN items need human "
        "attention. Nothing is sent automatically."
    )

    tabs = st.tabs([label for _, label, _ in ARTIFACTS])
    current: dict[str, str] = {}
    for tab, (key, label, _) in zip(tabs, ARTIFACTS):
        with tab:
            editor_key = f"edit_{key}"
            value_before_widget = st.session_state.get(editor_key, stored[key])
            st.markdown("**Preview — UNKNOWN items are highlighted**")
            st.markdown(
                highlight_unknown_html(value_before_widget),
                unsafe_allow_html=True,
            )
            current[key] = st.text_area(
                f"Edit {label.lower()}",
                value=value_before_widget,
                height=320,
                key=editor_key,
            )

    st.session_state["drafts"] = current
    approved_pack = st.session_state.get("approved_pack")
    approver_name = st.text_input(
        "Approved by (full name)",
        key="approver_name",
        placeholder="For example: Jordan Smith",
    )

    # Exports are withdrawn immediately if any approved content or approver changes.
    if isinstance(approved_pack, ApprovedPack) and (
        dict(approved_pack.artifacts) != current
        or approved_pack.approved_by != normalise_approver_name(approver_name)
    ):
        clear_approval()
        approved_pack = None
        st.warning("The approval was cleared because the reviewed content changed.")

    valid_name = normalise_approver_name(approver_name) is not None
    if approver_name and not valid_name:
        st.caption("Enter the approver's first and last name.")

    has_unknown = contains_unknown(current)
    if has_unknown:
        unknown_acknowledged = st.checkbox(
            "I have reviewed every remaining UNKNOWN item and accept that the approved "
            "pack will retain these unresolved details.",
            key="unknown_acknowledged",
        )
    else:
        st.session_state["unknown_acknowledged"] = False
        unknown_acknowledged = False

    approval_disabled = not can_approve(
        current, approver_name, unknown_acknowledged
    )
    if st.button(
        "Approve & Export",
        type="primary",
        disabled=approval_disabled,
        width="stretch",
    ):
        try:
            st.session_state["approved_pack"] = create_approved_pack(
                current,
                approver_name,
                unknown_acknowledged,
            )
        except ValidationError as exc:
            st.error(str(exc))
        else:
            st.success("Approved snapshot created. The export buttons are now available.")
            st.rerun()

    render_exports(st.session_state.get("approved_pack"))


def render_exports(pack: object) -> None:
    """Show downloads only for an immutable ApprovedPack."""

    if not isinstance(pack, ApprovedPack):
        return
    st.divider()
    st.header("Approved exports")
    st.success(f"Approved by {pack.approved_by} at {pack.approved_at_utc}.")
    pdf_bytes = build_pdf_bytes(pack)
    st.download_button(
        "Download approved PDF pack",
        data=pdf_bytes,
        file_name="arrangement-admin-approved-pack.pdf",
        mime="application/pdf",
        on_click="ignore",
        width="stretch",
    )
    st.subheader("Plain-text drafts")
    text_files = build_text_exports(pack)
    columns = st.columns(2)
    for index, (_, label, filename) in enumerate(ARTIFACTS):
        with columns[index % 2]:
            st.download_button(
                f"Download {label}",
                data=text_files[filename],
                file_name=filename,
                mime="text/plain; charset=utf-8",
                key=f"download_{filename}",
                on_click="ignore",
                width="stretch",
            )


def main() -> None:
    st.set_page_config(
        page_title="Arrangement Admin Autopilot",
        page_icon="🕊️",
        layout="wide",
    )
    install_styles()
    st.title("Arrangement Admin Autopilot")
    st.subheader(
        "You care for the family. We organise the approved paperwork after the arrangement."
    )
    st.markdown(
        '<div class="safety-note">Drafts stay in this temporary session, are never sent '
        "automatically, and cannot be exported until a named person approves them.</div>",
        unsafe_allow_html=True,
    )
    require_pilot_password(get_secret("APP_PASSWORD"))
    initialise_state()
    render_intake()
    render_review()


if __name__ == "__main__":
    main()
