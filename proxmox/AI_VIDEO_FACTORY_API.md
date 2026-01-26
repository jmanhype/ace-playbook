# AI Video Factory API Documentation

**For Integration with External AI Systems**

## Overview

This API provides programmatic access to a multi-agent AI video production system running on a 3090 GPU PC. The system uses a 3-agent crew (Director → Writer → Cameraman) coordinated through Letta framework to generate high-quality videos using the LTX-2 model.

## Architecture

```
External AI System
       ↓
    Director Agent (Coordinator)
       ↓
    Writer Agent (Prompts) → Cameraman Agent (ComfyUI/LTX-2)
       ↓
   Generated Video
```

## Authentication

**API Token**: `e9acq0WgooNgt5ncWWSJEQ`

All requests must include this token in the Authorization header.

## Base URL

```
http://192.168.1.143:8283/v1
```

## Primary Endpoint

### Send Request to Director

**Endpoint**: `POST /agents/{director_id}/messages`

**Director Agent ID**: `agent-22069f59-7a79-4890-bf4f-1f2a69696267`

**Full URL**: `http://192.168.1.143:8283/v1/agents/agent-22069f59-7a79-4890-bf4f-1f2a69696267/messages`

## Request Format

### Headers
```
Content-Type: application/json
Authorization: Bearer e9acq0WgooNgt5ncWWSJEQ
```

### Body
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Your video description here"
    }
  ]
}
```

## Example Integrations

### Python

```python
import requests
import json

API_TOKEN = "e9acq0WgooNgt5ncWWSJEQ"
DIRECTOR_ID = "agent-22069f59-7a79-4890-bf4f-1f2a69696267"
BASE_URL = "http://192.168.1.143:8283/v1"

def generate_video(prompt: str) -> dict:
    """
    Send a video generation request to the Director agent.

    Args:
        prompt: Natural language description of the video to generate

    Returns:
        Response from Director agent
    """
    url = f"{BASE_URL}/agents/{DIRECTOR_ID}/messages"

    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()

# Example usage
result = generate_video("A majestic dragon flying through storm clouds at sunset")
print(result)
```

### cURL

```bash
curl -X POST \
  "http://192.168.1.143:8283/v1/agents/agent-22069f59-7a79-4890-bf4f-1f2a69696267/messages" \
  -H "Authorization: Bearer e9acq0WgooNgt5ncWWSJEQ" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{
      "role": "user",
      "content": "A majestic dragon flying through storm clouds at sunset"
    }]
  }'
```

### Node.js

```javascript
const axios = require('axios');

const API_TOKEN = 'e9acq0WgooNgt5ncWWSJEQ';
const DIRECTOR_ID = 'agent-22069f59-7a79-4890-bf4f-1f2a69696267';
const BASE_URL = 'http://192.168.1.143:8283/v1';

async function generateVideo(prompt) {
  const url = `${BASE_URL}/agents/${DIRECTOR_ID}/messages`;

  const response = await axios.post(url, {
    messages: [
      {
        role: 'user',
        content: prompt
      }
    ]
  }, {
    headers: {
      'Authorization': `Bearer ${API_TOKEN}`,
      'Content-Type': 'application/json'
    }
  });

  return response.data;
}

// Example usage
generateVideo('A majestic dragon flying through storm clouds at sunset')
  .then(result => console.log(result))
  .catch(error => console.error(error));
```

## Workflow Explanation

When you send a request to the Director:

1. **Director receives your prompt** - The coordinator agent understands what you want
2. **Director consults Writer** - Writer agent crafts an optimized prompt for the video model
3. **Director reviews prompt** - Ensures quality and adherence to production standards
4. **Director instructs Cameraman** - Cameraman executes the ComfyUI workflow with LTX-2 model
5. **Video is generated** - Output saved to ComfyUI output directory

### Internal Agent IDs

For reference:
- **Writer**: `agent-e565b3e8-4a59-440a-89ab-6c279d61cfb0`
- **Cameraman**: `agent-f939736a-46fc-4115-a584-0a8cf896212a`
- **Director**: `agent-22069f59-7a79-4890-bf4f-1f2a69696267`

**Note**: You only need to communicate with Director. It handles the internal coordination.

## Response Format

The Director responds with natural language describing:

- What action is being taken
- Coordination with other agents
- Expected output
- Any issues or confirmations

Example response:
```
I'll coordinate with the Writer to develop an excellent prompt for this concept,
then have the Cameraman execute the video generation with proper settings.
```

## Retrieving Generated Videos

Videos are saved to the ComfyUI output directory on the 3090 PC:

**Path**: `/home/straughter/ComfyUI/output/video/`

**Filename pattern**: `LTX-2_XXXXX_.mp4`

### Download via SCP

```bash
scp straughter@192.168.1.143:/home/straughter/ComfyUI/output/video/LTX-2_*.mp4 ./local_video.mp4
```

### Check Latest Videos via SSH

```bash
ssh straughter@192.168.1.143 "ls -lt /home/straughter/ComfyUI/output/video/LTX-2_*.mp4 | head -5"
```

### Check ComfyUI Queue Status

```bash
curl "http://192.168.1.143:8188/queue"
```

## Best Practices

### Prompt Engineering

- **Be specific**: Describe motion, mood, style, subjects
- **Keep it focused**: Single subject works best
- **Mention atmosphere**: Lighting, color palette, emotional tone
- **Short duration**: System optimized for ultra-short clips (3-5 seconds)

### Good Prompts

```
✓ "A ethereal forest spirit materializing from morning mist, iridescent wings spreading,
   soft purple and blue palette, cinematic lighting"

✓ "An astronaut discovering an alien garden, bioluminescent flowers blooming in real-time,
   wonder and awe, 4K quality"

✓ "A samurai warrior drawing their katana in slow motion, cherry blossoms falling around them,
   dramatic lighting, feudal Japan aesthetic"
```

### Weak Prompts

```
✗ "a video" (too vague)
✗ "make something cool" (no direction)
✗ "lots of things happening" (too complex for current format)
```

## System Capabilities

### Video Model
- **Model**: LTX-2 (ltx-2-19b-distilled-fp8.safetensors)
- **Type**: Text-to-video generation
- **Duration**: 3-5 seconds (optimized)
- **Resolution**: High definition

### Production Standards
The Director agent maintains these quality standards:
- Single subject focus
- Ultra-short format (3-5 seconds)
- Dark, moody aesthetics preferred
- Blue/purple color palette default
- Cinematic lighting

### Generation Settings
- **CFG Scale**: Adjustable (typically 1.0-7.0)
- **Steps**: ~30 sampling steps
- **Negative Prompts**: Applied automatically (blurry, low quality, distorted)

## Error Handling

### Common HTTP Status Codes

- **200**: Success
- **400**: Bad request (check JSON format)
- **401**: Unauthorized (check API token)
- **500**: Internal server error (check Letta logs)

### Typical Errors

**Agent not found**:
```
Check Director ID is correct: agent-22069f59-7a79-4890-bf4f-1f2a69696267
```

**Invalid message format**:
```
Ensure messages array contains at least one message with role and content
```

**Timeout**:
```
Video generation takes time. Check ComfyUI queue to see if job is pending.
```

## Monitoring

### Check Agent Status

```bash
curl "http://192.168.1.143:8283/v1/agents/" \
  -H "Authorization: Bearer e9acq0WgooNgt5ncWWSJEQ"
```

### Check ComfyUI Queue

```bash
curl "http://192.168.1.143:8188/queue"
```

### View Recent Videos

```bash
ssh straughter@192.168.1.143 "ls -lth /home/straughter/ComfyUI/output/video/ | head -10"
```

## Advanced Usage

### Batch Requests

Send multiple requests in sequence:

```python
prompts = [
    "Dragon in flight",
    "Underwater city",
    "Robot dance battle"
]

for prompt in prompts:
    result = generate_video(prompt)
    print(f"Submitted: {prompt}")
    time.sleep(30)  # Wait between submissions
```

### Custom Instructions to Director

```python
custom_prompt = """
Please generate a video with these specifications:
- Subject: A cyberpunk hacker in a neon-lit alley
- Style: Blade Runner aesthetic
- Colors: Cyan and magenta highlights
- Mood: Mysterious and tense
- Duration: 4 seconds
"""

generate_video(custom_prompt)
```

## System Requirements for Integration

- **Network access** to `192.168.1.143` on ports 8283 (Letta) and 8188 (ComfyUI)
- **HTTP client** for making REST API calls
- **Ability to handle async operations** (video generation takes time)
- **Storage** for downloaded video files

## Support Information

- **Letta Server**: Port 8283
- **ComfyUI**: Port 8188
- **OS**: Linux (3090 PC)
- **User**: straughter
- **Framework**: Letta multi-agent system
- **Video Backend**: ComfyUI with LTX-2 model

## Quick Start Checklist

1. ✓ Verify network connectivity to 192.168.1.143
2. ✓ Set API token: `e9acq0WgooNgt5ncWWSJEQ`
3. ✓ Use Director ID: `agent-22069f59-7a79-4890-bf4f-1f2a69696267`
4. ✓ Send POST request to `/agents/{director_id}/messages`
5. ✓ Wait for generation (typically 30-60 seconds)
6. ✓ Retrieve video from `/home/straughter/ComfyUI/output/video/`

## Example: Complete Workflow

```python
import requests
import time
import paramiko
import scp

class AIVideoFactory:
    def __init__(self):
        self.api_token = "e9acq0WgooNgt5ncWWSJEQ"
        self.director_id = "agent-22069f59-7a79-4890-bf4f-1f2a69696267"
        self.base_url = "http://192.168.1.143:8283/v1"

    def generate(self, prompt: str, wait_time: int = 45) -> str:
        """Generate video and return local path"""

        # Send request to Director
        url = f"{self.base_url}/agents/{self.director_id}/messages"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        payload = {"messages": [{"role": "user", "content": prompt}]}

        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()

        print(f"Request submitted: {prompt}")
        print(f"Director response: {response.json()}")

        # Wait for generation
        print(f"Waiting {wait_time} seconds for generation...")
        time.sleep(wait_time)

        # Download latest video
        return self._download_latest_video()

    def _download_latest_video(self) -> str:
        """Download the most recently generated video"""
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect('192.168.1.143', username='straughter')

        # Get latest video
        stdin, stdout, stderr = ssh.exec_command(
            "ls -t /home/straughter/ComfyUI/output/video/LTX-2_*.mp4 | head -1"
        )
        remote_path = stdout.read().decode().strip()

        # Download
        local_path = f"./{remote_path.split('/')[-1]}"
        with scp.SCPClient(ssh.get_transport()) as scp_client:
            scp_client.get(remote_path, local_path)

        ssh.close()
        print(f"Video downloaded to: {local_path}")
        return local_path

# Usage
factory = AIVideoFactory()
video_path = factory.generate("A phoenix rising from ashes in slow motion")
print(f"Success! Video saved to: {video_path}")
```

---

**Document Version**: 1.0
**Last Updated**: 2026-01-26
**System Status**: Operational
