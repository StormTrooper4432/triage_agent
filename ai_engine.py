"""
AI orchestration engine for inbound email triage and CRM reconciliation.

Uses the Google GenAI SDK with Gemini structured outputs. No Streamlit or
UI logic lives here.
"""

from __future__ import annotations

import json
import logging
from typing import Any, List, Optional

import pydantic
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from data_manager import (
    TASK_NEW_CLIENT_PROFILE_QUALIFICATION,
    _extract_sender_from_body,
    classify_crm_task,
)

logger = logging.getLogger(__name__)

PRIMARY_MODEL = "gemini-3.5-flash"
FALLBACK_MODEL = "gemini-3.1-flash-lite"

ALLOWED_URGENCY_SCORES = frozenset({"CRITICAL", "HIGH", "MEDIUM", "LOW"})

SYSTEM_INSTRUCTION = (
    "You are an expert AI Operations Assistant deployed at an independent "
    "bookkeeping firm. Triage incoming emails and reconcile them against the "
    "provided CRM data. Personalize responses using context. If a client record "
    "exists but the name is blank and status is 'lead', treat them as a new "
    "prospect qualification task. Flag billing disputes as CRITICAL and new "
    "revenue opportunities as HIGH urgency."
)


class TriageOutput(pydantic.BaseModel):
    sender_email: str
    is_known_client: bool
    client_id: Optional[int]
    urgency_score: str  # Enforce restriction to: CRITICAL, HIGH, MEDIUM, LOW
    primary_intent: str
    extracted_entities: List[str]
    crm_reconciliation_notes: str
    proposed_crm_status_update: Optional[str]
    drafted_reply_body: str

    @pydantic.field_validator("urgency_score")
    @classmethod
    def validate_urgency_score(cls, value: str) -> str:
        normalized = value.strip().upper()
        if normalized not in ALLOWED_URGENCY_SCORES:
            raise ValueError(
                f"urgency_score must be one of {sorted(ALLOWED_URGENCY_SCORES)}"
            )
        return normalized


def _build_user_prompt(email_body: str, crm_context: Optional[dict]) -> str:
    """Format the inbound email and CRM context for the model."""
    if crm_context:
        context_block = json.dumps(crm_context, indent=2, default=str)
    else:
        context_block = "null"

    return (
        "Analyze the inbound email below and reconcile it against the CRM context.\n\n"
        f"INBOUND EMAIL:\n{email_body}\n\n"
        f"CRM CONTEXT:\n{context_block}"
    )


def _extract_sender_email(email_body: str) -> str:
    """Best-effort sender extraction for fallback responses."""
    sender = _extract_sender_from_body(email_body)
    return sender or "unknown@unknown"


def _build_generation_config() -> types.GenerateContentConfig:
    """Shared structured-output config used by every model tier in the cascade."""
    return types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=TriageOutput,
        system_instruction=SYSTEM_INSTRUCTION,
    )


def _parse_model_response(response: Any, model_name: str) -> TriageOutput:
    """
    Parse a Gemini structured-output response into a validated ``TriageOutput``.

    Raises:
        ValueError: When the model returns empty text.
        pydantic.ValidationError: When the JSON does not match ``TriageOutput``.
        json.JSONDecodeError: When the response is not valid JSON.
    """
    response_text = response.text
    if not response_text or not response_text.strip():
        raise ValueError(f"{model_name} returned an empty response.")

    logger.info(
        "Received structured response from %s (%d chars).",
        model_name,
        len(response_text),
    )
    return TriageOutput.model_validate_json(response_text)


def _call_model_triage(
    client: genai.Client,
    model_name: str,
    contents: str,
    config: types.GenerateContentConfig,
) -> TriageOutput:
    """
    Execute a single structured triage request against a Gemini model.

    Raises:
        genai_errors.APIError: On upstream API failures (e.g. 503, 429).
        ValueError, pydantic.ValidationError, json.JSONDecodeError: On bad output.
    """
    logger.info("Invoking model tier: %s", model_name)
    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=config,
    )
    return _parse_model_response(response, model_name)


def _fallback_triage_output(
    email_body: str,
    crm_context: Optional[dict],
    error_message: str,
) -> TriageOutput:
    """
    Return a safe default ``TriageOutput`` when the Gemini call or parsing fails.

    Preserves any known CRM identifiers from ``crm_context`` so downstream
    steps can still route the message for manual review.
    """
    sender_email = _extract_sender_email(email_body)
    is_known_client = crm_context is not None

    client_id: Optional[int] = None
    if crm_context and crm_context.get("client_id") is not None:
        try:
            client_id = int(crm_context["client_id"])
        except (TypeError, ValueError):
            client_id = None

    task_type = classify_crm_task(crm_context)

    if task_type == TASK_NEW_CLIENT_PROFILE_QUALIFICATION:
        reconciliation_notes = (
            "Automated triage unavailable. Stub CRM lead (blank name, lead status) "
            "requires new-client profile qualification."
        )
        proposed_status = "prospect"
    elif is_known_client:
        reconciliation_notes = (
            "Automated triage unavailable. Known CRM record matched; manual review required."
        )
        proposed_status = crm_context.get("status") if crm_context else None
    else:
        reconciliation_notes = (
            "Automated triage unavailable. No CRM record matched; treat as unmapped prospect."
        )
        proposed_status = "lead"

    logger.warning("Falling back to default TriageOutput: %s", error_message)

    return TriageOutput(
        sender_email=sender_email,
        is_known_client=is_known_client,
        client_id=client_id,
        urgency_score="MEDIUM",
        primary_intent="manual_review_required",
        extracted_entities=[],
        crm_reconciliation_notes=f"{reconciliation_notes} Error: {error_message}",
        proposed_crm_status_update=proposed_status,
        drafted_reply_body=(
            "Thank you for your email. A member of our team has received your message "
            "and will follow up with you shortly."
        ),
    )


def analyze_inbound_email(
    email_body: str,
    crm_context: Optional[dict],
) -> TriageOutput:
    """
    Triage an inbound email against optional CRM context using Gemini.

    Model cascade:
      1. ``gemini-3.5-flash`` (primary)
      2. ``gemini-3.1-flash-lite`` (fallback on primary API errors)
      3. Local ``TriageOutput`` backup (if both model tiers fail)

    Args:
        email_body: Raw text of the inbound email (including headers).
        crm_context: Matching CRM row as a dictionary, or ``None`` for new leads.

    Returns:
        A validated ``TriageOutput`` instance. Never raises to callers.
    """
    load_dotenv()

    contents = _build_user_prompt(email_body, crm_context)
    config = _build_generation_config()
    client = genai.Client()

    logger.info("Starting inbound email triage cascade.")

    try:
        return _call_model_triage(client, PRIMARY_MODEL, contents, config)

    except genai_errors.APIError as primary_exc:
        logger.warning(
            "⚠️ Primary model gemini-3.5-flash busy. Cascading to fallback tier..."
        )
        logger.warning(
            "Primary tier %s failed (code=%s): %s",
            PRIMARY_MODEL,
            primary_exc.code,
            primary_exc.message,
        )

        try:
            return _call_model_triage(client, FALLBACK_MODEL, contents, config)

        except genai_errors.APIError as fallback_exc:
            return _fallback_triage_output(
                email_body,
                crm_context,
                (
                    f"All model tiers unavailable. "
                    f"{FALLBACK_MODEL} API error ({fallback_exc.code}): {fallback_exc.message}"
                ),
            )
        except (pydantic.ValidationError, json.JSONDecodeError, ValueError) as exc:
            return _fallback_triage_output(
                email_body,
                crm_context,
                f"Fallback tier {FALLBACK_MODEL} response formatting error: {exc}",
            )
        except Exception as exc:
            return _fallback_triage_output(
                email_body,
                crm_context,
                f"Unexpected error on fallback tier {FALLBACK_MODEL}: {exc}",
            )

    except (pydantic.ValidationError, json.JSONDecodeError, ValueError) as exc:
        return _fallback_triage_output(
            email_body,
            crm_context,
            f"Primary tier {PRIMARY_MODEL} response formatting error: {exc}",
        )
    except Exception as exc:
        return _fallback_triage_output(
            email_body,
            crm_context,
            f"Unexpected error on primary tier {PRIMARY_MODEL}: {exc}",
        )
