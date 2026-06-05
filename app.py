"""
Xylo AI Studios — Bookkeeping Triage Control Center

Operational dashboard for reviewing inbound client emails, reconciling CRM
records, running AI triage, and approving drafted responses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st

from ai_engine import TriageOutput, analyze_inbound_email
from data_manager import (
    TASK_NEW_CLIENT_PROFILE_QUALIFICATION,
    classify_crm_task,
    get_crm_record,
    load_crm_data,
    load_raw_emails,
    update_crm_record,
)

APP_ROOT = Path(__file__).resolve().parent
EMAIL_FOLDER = APP_ROOT / "emails"
CRM_PATH = APP_ROOT / "sample-data" / "crm_export.csv"

STATUS_PENDING = "pending"
STATUS_TRIAGED = "triaged"
STATUS_APPROVED = "approved"

STATUS_ICONS = {
    STATUS_PENDING: "📥",
    STATUS_TRIAGED: "🔍",
    STATUS_APPROVED: "✅",
}

URGENCY_STYLES = {
    "CRITICAL": {"bg": "#fde8e8", "border": "#e03131", "text": "#c92a2a"},
    "HIGH": {"bg": "#fff4e6", "border": "#f08c00", "text": "#e67700"},
    "MEDIUM": {"bg": "#e7f5ff", "border": "#1c7ed6", "text": "#1864ab"},
    "LOW": {"bg": "#ebfbee", "border": "#2f9e44", "text": "#2b8a3e"},
}


def _init_session_state() -> None:
    defaults: dict[str, Any] = {
        "email_status": {},
        "triage_results": {},
        "draft_replies": {},
        "sync_messages": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _format_crm_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    text = str(value).strip()
    return text if text and text.lower() != "nan" else "—"


def _parse_email_headers(body: str) -> dict[str, str]:
    headers = {"from": "—", "subject": "—"}
    for line in body.splitlines():
        lower = line.lower()
        if lower.startswith("from:"):
            headers["from"] = line.split(":", 1)[1].strip()
        elif lower.startswith("subject:"):
            headers["subject"] = line.split(":", 1)[1].strip()
    return headers


def _get_email_status(filename: str) -> str:
    return st.session_state.email_status.get(filename, STATUS_PENDING)


def _set_email_status(filename: str, status: str) -> None:
    st.session_state.email_status[filename] = status


def _format_sidebar_label(filename: str) -> str:
    icon = STATUS_ICONS.get(_get_email_status(filename), "📥")
    return f"{icon}  {filename}"


@st.cache_data(show_spinner=False)
def _load_inbox() -> list[dict[str, str | None]]:
    load_crm_data(CRM_PATH)
    return load_raw_emails(EMAIL_FOLDER)


def _count_active_client_matches(emails: list[dict]) -> int:
    count = 0
    for email in emails:
        sender = email.get("extracted_sender")
        if sender and get_crm_record(sender) is not None:
            count += 1
    return count


def _count_high_critical_urgency() -> int:
    count = 0
    for result in st.session_state.triage_results.values():
        urgency = result.get("urgency_score", "")
        if urgency in {"HIGH", "CRITICAL"}:
            count += 1
    return count


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
            .block-container { padding-top: 1.5rem; max-width: 1400px; }
            .corp-header {
                background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
                color: #ffffff;
                padding: 1.25rem 1.5rem;
                border-radius: 10px;
                margin-bottom: 1rem;
            }
            .corp-header h1 {
                margin: 0;
                font-size: 1.6rem;
                font-weight: 600;
                color: #ffffff !important;
            }
            .corp-header p {
                margin: 0.35rem 0 0 0;
                opacity: 0.9;
                font-size: 0.95rem;
            }
            .panel-card {
                background: #ffffff;
                border: 1px solid #e9ecef;
                border-radius: 10px;
                padding: 1rem 1.1rem;
                margin-bottom: 1rem;
                box-shadow: 0 1px 3px rgba(0,0,0,0.04);
            }
            .panel-title {
                font-size: 0.82rem;
                font-weight: 700;
                letter-spacing: 0.06em;
                text-transform: uppercase;
                color: #495057;
                margin-bottom: 0.75rem;
            }
            .unknown-banner {
                background: #f1f3f5;
                border: 1px solid #ced4da;
                border-left: 4px solid #868e96;
                color: #495057;
                padding: 0.9rem 1rem;
                border-radius: 8px;
                font-weight: 600;
                font-size: 0.92rem;
            }
            .urgency-pill {
                padding: 0.65rem 0.9rem;
                border-radius: 8px;
                border: 1px solid;
                font-weight: 700;
                font-size: 0.95rem;
                margin-bottom: 0.75rem;
            }
            .crm-field-label {
                font-size: 0.75rem;
                color: #868e96;
                text-transform: uppercase;
                letter-spacing: 0.04em;
                margin-bottom: 0.15rem;
            }
            .crm-field-value {
                font-size: 0.98rem;
                color: #212529;
                margin-bottom: 0.65rem;
            }
            div[data-testid="stSidebar"] {
                background-color: #f8f9fa;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_urgency_badge(urgency: str) -> None:
    style = URGENCY_STYLES.get(urgency, URGENCY_STYLES["MEDIUM"])
    st.markdown(
        f"""
        <div class="urgency-pill" style="
            background:{style['bg']};
            border-color:{style['border']};
            color:{style['text']};
        ">
            Urgency: {urgency}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_crm_profile(crm_record: Optional[dict], sender_email: Optional[str]) -> None:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">CRM Match Profile</div>', unsafe_allow_html=True)

    if crm_record is None:
        st.markdown(
            """
            <div class="unknown-banner">
                UNKNOWN SENDER: PROSPECT LEAD GENERATION PROTOCOL REQUIRED
            </div>
            """,
            unsafe_allow_html=True,
        )
        if sender_email:
            st.caption(f"Extracted sender: `{sender_email}` — no matching CRM record on file.")
    else:
        task_type = classify_crm_task(crm_record)
        if task_type == TASK_NEW_CLIENT_PROFILE_QUALIFICATION:
            st.info("Stub CRM lead detected — new client profile qualification required.")

        fields = [
            ("Client ID", "client_id"),
            ("Name", "name"),
            ("Company", "company"),
            ("Current Status", "status"),
            ("Existing Notes", "notes"),
        ]
        for label, key in fields:
            st.markdown(f'<div class="crm-field-label">{label}</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="crm-field-value">{_format_crm_value(crm_record.get(key))}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)


def _render_triage_insights(
    filename: str,
    email_body: str,
    crm_record: Optional[dict],
) -> None:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-title">AI Insights &amp; Decision Hub</div>',
        unsafe_allow_html=True,
    )

    triage_data = st.session_state.triage_results.get(filename)

    if triage_data is None:
        # ── Cache miss: show primary action button ──────────────────────────
        if st.button(
            "Run AI Triage & Verification Agent",
            type="primary",
            use_container_width=True,
            key=f"run_triage_{filename}",
        ):
            with st.spinner("Running triage cascade (primary → fallback tier)..."):
                result = analyze_inbound_email(email_body, crm_record)
                st.session_state.triage_results[filename] = result.model_dump()
                st.session_state.draft_replies[filename] = result.drafted_reply_body
                _set_email_status(filename, STATUS_TRIAGED)
            triage_data = st.session_state.triage_results.get(filename)

        if triage_data is None:
            st.caption(
                "Run the triage agent to generate urgency scoring, CRM notes, and a draft reply."
            )
            st.markdown("</div>", unsafe_allow_html=True)
            return
    else:
        # ── Cache hit: render instantly; expose Force Refresh ───────────────
        action_col, refresh_col = st.columns([3, 1])
        with action_col:
            st.caption("⚡ Loaded from session cache — no API call made.")
        with refresh_col:
            if st.button(
                "🔄 Force Refresh",
                key=f"force_refresh_{filename}",
                help="Clear the cached result and re-run the Gemini triage API",
                use_container_width=True,
            ):
                with st.spinner("Re-running triage cascade (primary → fallback tier)..."):
                    result = analyze_inbound_email(email_body, crm_record)
                    st.session_state.triage_results[filename] = result.model_dump()
                    st.session_state.draft_replies[filename] = result.drafted_reply_body
                    _set_email_status(filename, STATUS_TRIAGED)
                st.rerun()

    # Re-read after any in-frame button execution above to get the freshest value.
    triage_data = st.session_state.triage_results.get(filename)
    triage = TriageOutput.model_validate(triage_data)

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Primary Intent", triage.primary_intent.replace("_", " ").title())
    with col_b:
        st.metric("Known Client", "Yes" if triage.is_known_client else "No")

    _render_urgency_badge(triage.urgency_score)

    if triage.extracted_entities:
        st.markdown("**Extracted Entities**")
        st.write(", ".join(triage.extracted_entities))

    # ── Fixed Section: Render CRM Updates via read-only text container ──
    st.markdown("**Proposed CRM Summary Update**")
    summary_text = triage.crm_reconciliation_notes
    if triage.proposed_crm_status_update:
        summary_text += f"\n\nProposed status change: {triage.proposed_crm_status_update}"
    
    st.text_area(
        label="CRM Update Details",
        value=summary_text,
        height=140,
        label_visibility="collapsed",
        disabled=True,
        key=f"crm_notes_display_{filename}"
    )
    # ────────────────────────────────────────────────────────────────────

    st.markdown("**Drafted Response Email**")
    draft_key = f"draft_{filename}"
    edited_reply = st.text_area(
        "Edit the draft before approval",
        value=st.session_state.draft_replies.get(filename, triage.drafted_reply_body),
        height=220,
        label_visibility="collapsed",
        key=draft_key,
    )
    st.session_state.draft_replies[filename] = edited_reply

    if st.button(
        "Approve Reply & Sync to CRM",
        type="primary",
        use_container_width=True,
        key=f"approve_{filename}",
    ):
        if triage.client_id is None:
            st.warning(
                "CRM sync skipped — no client ID on file. "
                "Create a CRM record for this sender before syncing."
            )
        else:
            update_crm_record(
                client_id=triage.client_id,
                new_notes=triage.crm_reconciliation_notes,
                new_status=triage.proposed_crm_status_update or "active",
            )
            _set_email_status(filename, STATUS_APPROVED)
            st.session_state.sync_messages[filename] = (
                f"Reply approved and CRM record {triage.client_id} updated successfully."
            )

    sync_message = st.session_state.sync_messages.get(filename)
    if sync_message:
        st.success(sync_message)

    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(
        page_title="Bookkeeping Triage Control",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _init_session_state()
    _inject_styles()

    emails = _load_inbox()
    filenames = [email["filename"] for email in emails]

    st.markdown(
        """
        <div class="corp-header">
            <h1>Xylo AI Studios - Bookkeeping Triage Control</h1>
            <p>Inbound reconciliation, urgency routing, and CRM decision workflow</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Total Emails", len(emails))
    metric_col2.metric("Active Client Matches", _count_active_client_matches(emails))
    metric_col3.metric("High / Critical Urgency Items", _count_high_critical_urgency())

    st.sidebar.markdown("### Inbox Navigation")
    st.sidebar.caption("Select an inbound message to review and triage.")

    selected_filename = st.sidebar.selectbox(
        "Message queue",
        options=filenames,
        format_func=_format_sidebar_label,
        label_visibility="collapsed",
    )

    selected_email = next(e for e in emails if e["filename"] == selected_filename)
    email_body = selected_email["body"]
    sender_email = selected_email.get("extracted_sender")
    headers = _parse_email_headers(email_body)
    crm_record = get_crm_record(sender_email) if sender_email else None

    st.sidebar.divider()
    st.sidebar.markdown("**Queue legend**")
    st.sidebar.markdown("📥 Pending  \n🔍 Triaged  \n✅ Approved & synced")

    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Inbound Message Feed</div>', unsafe_allow_html=True)
        st.markdown(f"**From:** {headers['from']}")
        st.markdown(f"**Subject:** {headers['subject']}")
        st.markdown(f"**Source file:** `{selected_filename}`")
        st.code(email_body, language=None)
        st.markdown("</div>", unsafe_allow_html=True)

        _render_crm_profile(crm_record, sender_email)

    with right_col:
        _render_triage_insights(selected_filename, email_body, crm_record)


if __name__ == "__main__":
    main()
