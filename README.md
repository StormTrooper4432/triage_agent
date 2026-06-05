# Bookkeeping Triage Control Center

**CRM-Reconciled AI Triage Agent for independent bookkeeping practices.**

---

## Project Overview

Inbound client emails at a bookkeeping firm are high-stakes and unstructured — billing disputes buried in pleasantries, urgent deadlines hidden in forwarded threads, and new revenue leads dressed up as casual questions. Triaging them manually is slow, error-prone, and doesn't scale.

This dashboard changes that. It is an operational AI agent that reads every inbound email, cross-references the sender against your live CRM export, and produces a structured triage packet in seconds: urgency score, primary intent classification, extracted entities, a proposed CRM status update, and a personalized draft reply — ready for one-click approval and CRM sync.

Built specifically for **independent bookkeeping practices**, the system handles the full lifecycle from raw inbox to approved response without leaving the browser.

**Key capabilities:**

- Automatic sender-to-CRM reconciliation — flags unknown prospects, detects stub leads, and surfaces known client history before you read a single line
- Four-tier urgency routing (`CRITICAL` → `HIGH` → `MEDIUM` → `LOW`) with color-coded visual indicators
- AI-drafted, editable reply bodies personalized against each client's CRM record
- One-click CRM sync that writes reconciliation notes and proposed status updates back to the live data layer
- Session-state caching: triage results are stored per-email for the duration of the session — navigating back to a previously triaged message is instant and costs zero API tokens
- Force Refresh override to manually re-run a fresh Gemini call when needed

---

## Setup & Installation

**Prerequisites:** Python 3.10 or newer, a Google AI Studio account, and a `GEMINI_API_KEY`.

```bash
# 1. Enter the project root
cd triage_agent

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your API key
cp .env.template .env
```

Open `.env` in any editor and set your key:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

```bash
# 5. Launch the dashboard
streamlit run app.py
```

The app will open at `http://localhost:8501`. Load the sample data from `sample-data/crm_export.csv` and the 14 inbound test emails in `emails/` to explore the full workflow without connecting live client data.

> **Note:** Never commit your `.env` file. It is excluded from version control by default via `.env.template`.

---

## AI-Native Engineering Methodology

### Multi-Tier Resilience Fallback Cascade

The triage engine (`ai_engine.py`) is designed around the assumption that LLM infrastructure is not perfectly reliable. During periods of high server demand, even well-resourced API providers return `503` or `429` errors. A naive single-model implementation would surface these upstream failures directly to the operator as broken UI states.

Instead, `analyze_inbound_email()` implements a **three-tier cascade**:

**Tier 1 — Primary:** `gemini-2.5-flash`
The flagship model tier. Called first on every request. Delivers the highest quality structured output for intent classification, entity extraction, and drafted reply personalization.

**Tier 2 — Fallback:** `gemini-2.0-flash-lite`
Triggered automatically on any `APIError` from the primary tier (HTTP 429, 503, or upstream quota exhaustion). Lighter-weight and more available under demand surges. Uses the identical structured output configuration — the caller cannot distinguish a Tier 1 from a Tier 2 response.

**Tier 3 — Local Deterministic Fallback:**
If both Gemini tiers are unavailable, `_fallback_triage_output()` constructs a safe `TriageOutput` object locally from whatever CRM context is available. It preserves the client ID, sets urgency to `MEDIUM`, and fills `primary_intent` with `manual_review_required` so the operator knows to handle the email themselves. The system never raises to the UI layer — there is always a response.

```
Request
  └─► gemini-2.5-flash          (Tier 1 — primary)
        │  APIError?
        └─► gemini-2.0-flash-lite  (Tier 2 — fallback)
              │  APIError / parse error?
              └─► _fallback_triage_output()  (Tier 3 — local)
```

The model names are defined as module-level constants (`PRIMARY_MODEL`, `FALLBACK_MODEL`) so swapping tiers as Google releases new Gemini versions requires a one-line change.

### Pydantic Structured Output Validation

All model responses are parsed and validated through a `TriageOutput` Pydantic model before they are ever handed to the UI layer. This eliminates an entire class of runtime errors caused by schema drift, hallucinated fields, or malformed JSON.

```python
class TriageOutput(pydantic.BaseModel):
    sender_email: str
    is_known_client: bool
    client_id: Optional[int]
    urgency_score: str        # validated enum: CRITICAL | HIGH | MEDIUM | LOW
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
            raise ValueError(...)
        return normalized
```

The Gemini SDK is configured with `response_mime_type="application/json"` and `response_schema=TriageOutput`, instructing the model to produce structured JSON conforming to the schema at the generation level. Pydantic validation is then applied as a second gate on the parsed output — specifically to enforce the `urgency_score` enum constraint, which the model schema alone cannot guarantee.

A `ValidationError` or `JSONDecodeError` at either gate is caught by the cascade and routes to the next tier rather than propagating to the caller, ensuring the operator always receives a valid, usable `TriageOutput` regardless of model behavior.

### Session-State Caching

Triage results are stored in `st.session_state.triage_results` keyed by filename the moment they are generated. On every subsequent navigation to that email, the cached `TriageOutput` is deserialized and rendered instantly with zero API calls. A **🔄 Force Refresh** button is exposed when a cache entry exists, allowing the operator to override the cache and re-run the full model cascade when a fresh analysis is needed.

---

## Project Structure

```
triage_agent/
├── app.py              # Streamlit operational dashboard
├── ai_engine.py        # Gemini triage cascade, TriageOutput schema, fallback logic
├── data_manager.py     # CRM loading, email parsing, sender extraction, record updates
├── requirements.txt    # Python dependencies
├── .env.template       # API key configuration template
├── emails/             # 14 sample inbound client emails (plain text)
└── sample-data/
    └── crm_export.csv  # Intentionally messy sample CRM export for testing
```
