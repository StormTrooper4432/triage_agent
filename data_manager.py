"""
Data ingestion and storage layer for the CRM-Reconciled Triage Agent.

Handles loading and normalizing CRM exports, looking up client records by email,
and parsing raw inbound email text files. No LLM logic lives here.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

# Default path relative to this module; override via load_crm_data(crm_path=...).
DEFAULT_CRM_PATH = Path(__file__).resolve().parent / "sample-data" / "crm_export.csv"

_crm_df: pd.DataFrame | None = None
_crm_path: Path | None = None

_EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w.-]+\.\w+", re.IGNORECASE)

# Triage task labels used when reconciling inbound email senders to CRM records.
TASK_EXISTING_CLIENT = "existing_client"
TASK_NEW_CLIENT_PROFILE_QUALIFICATION = "new_client_profile_qualification"
TASK_UNMAPPED_PROSPECT = "unmapped_prospect"


def _normalize_email(email: str | None) -> str | None:
    """Strip whitespace and lowercase an email address for consistent lookups."""
    if email is None or (isinstance(email, float) and pd.isna(email)):
        return None
    normalized = str(email).strip().lower()
    return normalized or None


def _clean_crm_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a raw CRM export for reliable lookups.

    - Fills missing scalar values with empty strings (keeps client_id numeric).
    - Strips leading/trailing whitespace on string columns.
    - Lowercases email addresses and status values for case-insensitive matching.
    """
    cleaned = df.copy()

    for column in cleaned.columns:
        if column == "client_id":
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
            continue
        if cleaned[column].dtype == object:
            cleaned[column] = (
                cleaned[column]
                .fillna("")
                .astype(str)
                .str.strip()
            )

    if "email" in cleaned.columns:
        cleaned["email"] = cleaned["email"].apply(_normalize_email)

    if "status" in cleaned.columns:
        cleaned["status"] = cleaned["status"].str.lower()

    return cleaned


def load_crm_data(crm_path: str | Path | None = None) -> pd.DataFrame:
    """
    Load ``crm_export.csv`` into a cleaned Pandas DataFrame.

    The result is cached in memory so subsequent lookups do not re-read disk.
    Pass ``crm_path`` to load from a different file location.

    Args:
        crm_path: Optional path to the CRM CSV. Defaults to ``sample-data/crm_export.csv``.

    Returns:
        A normalized DataFrame ready for record lookups.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """
    global _crm_df, _crm_path

    resolved_path = Path(crm_path) if crm_path is not None else DEFAULT_CRM_PATH

    if _crm_df is not None and _crm_path == resolved_path:
        return _crm_df

    if not resolved_path.is_file():
        raise FileNotFoundError(f"CRM export not found: {resolved_path}")

    raw_df = pd.read_csv(
        resolved_path,
        dtype=str,
        keep_default_na=True,
        na_values=["", " ", "NA", "N/A", "null", "None"],
    )

    _crm_df = _clean_crm_dataframe(raw_df)
    _crm_path = resolved_path
    return _crm_df


def _is_blank_field(value: Any) -> bool:
    """Return True when a CRM field is missing or only whitespace."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return True
    return str(value).strip() == ""


def classify_crm_task(record: dict[str, Any] | None) -> str:
    """
    Determine the triage task for an inbound sender based on CRM reconciliation.

    Rules:
      - No CRM row → ``unmapped_prospect`` (email not in CRM at all).
      - CRM row with blank ``name`` and status ``lead`` →
        ``new_client_profile_qualification`` (stub lead needing intake).
      - All other matches → ``existing_client``.

    Args:
        record: CRM row from :func:`get_crm_record`, or ``None``.

    Returns:
        One of the ``TASK_*`` constants defined in this module.
    """
    if record is None:
        return TASK_UNMAPPED_PROSPECT

    status = str(record.get("status", "")).strip().lower()
    if _is_blank_field(record.get("name")) and status == "lead":
        return TASK_NEW_CLIENT_PROFILE_QUALIFICATION

    return TASK_EXISTING_CLIENT


def get_crm_record(email_address: str) -> dict[str, Any] | None:
    """
    Look up a CRM record by email address.

    Args:
        email_address: Sender or client email to search for.

    Returns:
        The matching row as a dictionary, or ``None`` if no CRM row exists.

    Note:
        A returned record may still represent a qualification task (blank name +
        ``lead`` status). Use :func:`classify_crm_task` or :func:`reconcile_sender`
        to distinguish stub leads from fully onboarded clients.
    """
    normalized = _normalize_email(email_address)
    if not normalized:
        return None

    df = load_crm_data()
    if "email" not in df.columns:
        return None

    matches = df[df["email"] == normalized]
    if matches.empty:
        return None

    record = matches.iloc[0].to_dict()

    # Restore native types for downstream use.
    if pd.notna(record.get("client_id")):
        record["client_id"] = int(record["client_id"])

    return record


def reconcile_sender(email_address: str) -> dict[str, Any]:
    """
    Reconcile an inbound sender email to a CRM record and triage task.

    Args:
        email_address: Sender email extracted from an inbound message.

    Returns:
        A dictionary with:
          - ``email``: Normalized sender address.
          - ``crm_record``: Matching CRM row, or ``None``.
          - ``task_type``: Triage label from :func:`classify_crm_task`.
          - ``is_profile_qualification``: ``True`` when the sender is a stub
            lead that needs new-client profile qualification.
    """
    normalized = _normalize_email(email_address)
    record = get_crm_record(email_address) if normalized else None
    task_type = classify_crm_task(record)

    return {
        "email": normalized,
        "crm_record": record,
        "task_type": task_type,
        "is_profile_qualification": task_type == TASK_NEW_CLIENT_PROFILE_QUALIFICATION,
    }


def update_crm_record(client_id: int, new_notes: str, new_status: str) -> None:
    """
    Mock CRM write-back: update notes and status for an existing client.

    In production this would persist changes to the CRM or CSV export.
    For now it updates the in-memory DataFrame and prints a success message.

    Args:
        client_id: Primary key of the client record to update.
        new_notes: Replacement notes text.
        new_status: Replacement status value (stored lowercased).
    """
    df = load_crm_data()
    mask = df["client_id"] == client_id

    if not mask.any():
        print(f"[CRM] No record found for client_id={client_id}. Update skipped.")
        return

    df.loc[mask, "notes"] = new_notes.strip()
    df.loc[mask, "status"] = new_status.strip().lower()

    print(
        f"[CRM] Successfully updated client_id={client_id}: "
        f"status='{new_status.strip().lower()}', notes saved ({len(new_notes)} chars)."
    )


def _extract_sender_from_body(body: str) -> str | None:
    """
    Parse the sender email from a raw email's ``From:`` header line.

    Supports common formats:
      - ``Name <email@example.com>``
      - ``"Last, First" <email@example.com>``
      - ``email@example.com``
    """
    for line in body.splitlines():
        if not line.lower().startswith("from:"):
            continue

        from_value = line.split(":", 1)[1].strip()

        bracket_match = re.search(r"<([^>]+)>", from_value)
        if bracket_match:
            return _normalize_email(bracket_match.group(1))

        plain_match = _EMAIL_PATTERN.search(from_value)
        if plain_match:
            return _normalize_email(plain_match.group(0))

    return None


def load_raw_emails(folder_path: str | Path) -> list[dict[str, str | None]]:
    """
    Read raw text email files from a directory.

    Each file is expected to be a plain-text message (e.g. ``email_01.txt``) with
    a ``From:`` line near the top.

    Args:
        folder_path: Directory containing ``.txt`` email files.

    Returns:
        A list of dictionaries with keys:
          - ``filename``: Base name of the source file.
          - ``body``: Full file contents.
          - ``extracted_sender``: Email parsed from the ``From:`` line, or ``None``.

    Raises:
        FileNotFoundError: If ``folder_path`` does not exist.
        NotADirectoryError: If ``folder_path`` is not a directory.
    """
    directory = Path(folder_path)

    if not directory.exists():
        raise FileNotFoundError(f"Email folder not found: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Expected a directory, got: {directory}")

    emails: list[dict[str, str | None]] = []

    for file_path in sorted(directory.glob("*.txt")):
        body = file_path.read_text(encoding="utf-8")
        emails.append(
            {
                "filename": file_path.name,
                "body": body,
                "extracted_sender": _extract_sender_from_body(body),
            }
        )

    return emails
