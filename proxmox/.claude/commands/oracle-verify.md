# Oracle External LLM Verification

> Consult Oracle (GPT-5.2 Pro via browser) to verify code completeness and quality.
> Uses browser mode with manual login for the most capable external verification.

## Usage

```
/project:oracle-verify <verification prompt and files>
```

**Examples:**
```
/project:oracle-verify Verify Phase 6 is complete for BLACKICE specs/007-blackice-3/tasks.md blackice/schemas/*.py
/project:oracle-verify Review the implementation against spec specs/feature/spec.md src/feature/*.ts
/project:oracle-verify Check if all requirements are met spec.md implementation.py tests.py
```

## Prerequisites

1. **Chrome with Remote Debugging**: Launch Chrome with debugging enabled:
   ```bash
   /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222 &
   ```

2. **Manual Login Ready**: You'll need to manually log in to ChatGPT when prompted.

## Command Prompt

You are verifying code using Oracle (external LLM verification tool).

**User Request:** $ARGUMENTS

Follow these steps:

### 1. Parse the Request

Extract from the user's input:
- **PROMPT**: The verification question/task (what to verify)
- **FILES**: The files to include for context (space-separated paths)
- **SLUG**: Generate a short slug from the prompt (kebab-case, max 5 words)

### 2. Validate Files Exist

For each file path:
- Use `Glob` if pattern contains wildcards
- Use `Read` to verify files exist
- Collect full paths of existing files

### 3. Ensure Chrome is Running with Debugging

**CRITICAL**: Before launching Oracle, check if Chrome is running with remote debugging on port 9222.

```bash
# Check if port 9222 is in use by Chrome
curl -s http://localhost:9222/json/version | head -1
```

**If Chrome is NOT running with debugging**, start it:
```bash
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222 &
sleep 5
```

**IMPORTANT**:
- Use the **regular Chrome app** (`/Applications/Google Chrome.app`), NOT "Chrome for Testing"
- The user should already be logged into ChatGPT in their regular Chrome
- If Chrome is already running without debugging, the user must restart it with `--remote-debugging-port=9222`

### 4. Launch Oracle

Run Oracle with browser mode using npx:

```bash
npx -y @steipete/oracle@latest \
  --engine browser \
  --browser-manual-login \
  --browser-port 9222 \
  --force \
  --slug "{SLUG}" \
  --file "{FILE1}" --file "{FILE2}" ... \
  --prompt "{PROMPT}"
```

**Important flags:**
- `--engine browser`: Uses browser-based GPT-5.2 Pro (most capable)
- `--browser-manual-login`: Waits for you to log in manually
- `--browser-port 9222`: Chrome debugging port
- `--force`: Overwrite existing session with same slug
- `--file`: Attach files for context (repeat for each file)
- `--prompt`: The verification question

### 5. Run the Command

Execute using Bash tool (NOT background mode - wait for completion):

```bash
npx -y @steipete/oracle@latest --engine browser --browser-manual-login --browser-port 9222 --force --slug "your-slug" --file "file1.py" --file "file2.py" --prompt "Your verification question"
```

### 6. Report Results

When Oracle completes:
- Parse the output for findings
- Summarize any issues identified
- List any missing files or requirements
- Provide recommendations for next steps

## Notes

- Oracle runs GPT-5.2 Pro through browser automation
- Sessions can take 10-20 minutes depending on context size
- The tool saves session history for later reference
- Use `--browser-manual-login` if not already authenticated
- Chrome must be running with remote debugging enabled

## Troubleshooting

**"No inspectable targets"**: Chrome isn't running with `--remote-debugging-port=9222`

**"Connection refused"**: Wrong port or Chrome not running

**"Timeout"**: Increase timeout or reduce context size

**Session already exists**: Use `--force` to overwrite
