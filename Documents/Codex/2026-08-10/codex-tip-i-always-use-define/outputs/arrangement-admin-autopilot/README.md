# Arrangement Admin Autopilot

A small, password-protected Streamlit pilot for independent UK funeral directors. It turns arrangement-meeting notes into **only** these five human-reviewable drafts:

1. Internal case summary
2. Missing-information checklist
3. Family confirmation draft
4. Internal task list
5. Supplier-message drafts

Nothing is sent automatically. A named human must review the drafts and approve an immutable snapshot before any PDF or text export is available.

## Requirements

- Python 3.10 or newer
- An Anthropic API key with access to the configured Claude model

## Install and run

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Choose **one** of the following secret setups.

### Option A: Streamlit secrets

```bash
mkdir -p .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml`:

```toml
APP_PASSWORD = "a-strong-password-for-the-pilot"
ANTHROPIC_API_KEY = "your-api-key"
# Optional; current default:
ANTHROPIC_MODEL = "claude-sonnet-5"
```

The real `secrets.toml` is ignored by Git. Do not commit or share it.

### Option B: environment variables

```bash
export APP_PASSWORD='a-strong-password-for-the-pilot'
export ANTHROPIC_API_KEY='your-api-key'
# Optional:
export ANTHROPIC_MODEL='claude-sonnet-5'
```

Then start the pilot:

```bash
streamlit run app.py
```

The app fails closed if `APP_PASSWORD` is missing. The drafting button gives a non-sensitive configuration message if `ANTHROPIC_API_KEY` is missing. `ANTHROPIC_MODEL` is optional; the default is `claude-sonnet-5`. If an Anthropic account specifically requires the model requested in the original pilot brief, set `ANTHROPIC_MODEL="claude-sonnet-4-20250514"` (subject to that model being available to the account).

## How approval works

- Claude is constrained to a strict JSON schema with exactly five string fields.
- The app validates the response again and rejects missing, extra, empty, malformed, non-text, refused, or truncated output.
- Missing or unclear details must remain the exact token `UNKNOWN`, highlighted during review.
- A plausible full name is mandatory in **Approved by (full name)**.
- If any `UNKNOWN` remains, the approver must explicitly acknowledge it.
- Approval copies the five edited drafts, approver, and UTC time into an immutable in-memory snapshot.
- Editing or regenerating after approval removes the export controls until the material is approved again.
- The PDF and five text downloads are created only from that approved snapshot.

## Temporary processing and privacy

The app has no case database, CRM, analytics, automated messaging, or case-file write path. Notes, drafts, approval state, PDFs, and text exports live only in the active Streamlit session and application memory. Restarting the process or losing the session clears them.

**Important:** generating drafts sends the submitted notes to Anthropic's API for processing. “Temporary” in this MVP means the app does not persist a case database or case files; it does not override Anthropic account, API, retention, regional-processing, or contractual settings. Before using real personal data, the funeral director remains responsible for approving the pilot's privacy, information-governance, access-control, retention, and supplier arrangements. Use synthetic notes during initial testing.

The password gate is appropriate only for a small controlled pilot. Deploy behind HTTPS and suitable organisational access controls. Rotate both secrets if exposure is suspected.

## Tests

Tests are deterministic and do not call Anthropic:

```bash
python -m compileall app.py core.py tests
pytest -q
```

For a bounded local smoke check:

```bash
APP_PASSWORD=test ANTHROPIC_API_KEY=test \
  timeout 10s streamlit run app.py --server.headless true --server.port 8765
```

On macOS, where the `timeout` command may be absent, start Streamlit, confirm the health endpoint responds, and stop it manually.

## PDF character handling

The PDF uses fpdf2's lightweight built-in Helvetica font. Common curly quotes, dashes, ellipses, bullets, and accented Latin characters are normalised for reliable export. Characters the built-in font cannot represent are replaced with `?`; the five UTF-8 text downloads retain the original reviewed characters.
