# Inbound Test Agent

Automated test agent that connects to your **outbound ElevenLabs Conversational AI agent** over WebSocket (no PSTN/SIP) and plays configurable customer personas. Records full conversations and evaluates outbound agent performance.

Supports two conversation modes:
- **Text mode** (default): sends `user_message` text directly — fastest, no TTS/STT needed
- **Audio mode**: generates persona TTS audio via ElevenLabs REST API, sends as `user_audio_chunk` — tests the full audio pipeline

## Architecture

```
[Test Agent / Persona Engine]
   │
   ├─ Connect → wss://api.elevenlabs.io/v1/convai/conversation?agent_id=...
   ├─ Send conversation_initiation_client_data (+ dynamic_variables)
   │
   ├─ TEXT MODE:
   │    Persona LLM → user_message → agent_response / agent_response_complete
   │
   ├─ AUDIO MODE:
   │    Persona LLM → ElevenLabs TTS REST (pcm_16000) → user_audio_chunk
   │    ← audio (base64 PCM) + agent_response (text transcript)
   │
   └─ Log raw WS events + transcript → SQLite → rule + LLM evaluation
```

**Dependencies:** `websockets`, `openai`, `httpx`, `pyyaml`, `click`, `pydantic`  
**No LiveKit, no Pipecat, no Deepgram, no telephony.**

## Setup

```powershell
cd "inbound tester agent"
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
copy .env.example .env
# Edit .env with your keys and agent_id
```

### Required environment variables

| Variable | Description |
|---|---|
| `ELEVENLABS_API_KEY` | API key with ConvAI WebSocket + TTS access |
| `ELEVENLABS_AGENT_ID` | Production outbound agent ID |
| `OPENAI_API_KEY` | Persona engine + LLM evaluator |

Optional: `DATABASE_URL` (default SQLite at `./data/test_runs.db`), `MAX_TURNS`, `RESPONSE_TIMEOUT_SEC`, `CONVERSATION_MODE`, `PERSONA_VOICE_ID`, `PERSONA_TTS_MODEL`.

If your outbound agent requires **dynamic variables** at initiation, add them under `dynamic_variables` in each scenario YAML (see `scenarios/cooperative.yaml`).

## Usage

### 1. Spike — verify WebSocket connectivity

```powershell
# Text mode (default)
inbound-tester spike --message "Hi"

# Audio mode — synthesizes TTS and sends as user_audio_chunk
inbound-tester spike --message "Hi" --mode audio
```

Logs `conversation_initiation_metadata` and the first `agent_response`.

### 2. Run one scenario

```powershell
# Text mode (default)
inbound-tester run scenarios/cooperative.yaml

# Audio mode
inbound-tester run scenarios/cooperative.yaml --mode audio
```

### 3. Run all scenarios

```powershell
inbound-tester run-all
inbound-tester run-all --parallel   # concurrent runs
inbound-tester run-all --mode audio # audio mode for all
```

### 4. View results

```powershell
inbound-tester list-runs
inbound-tester show-run <run-id-prefix>
```

## Scenario format

Each YAML file in `scenarios/` defines:

- `persona_prompt` — who the test customer is
- `goals` — scripted behavior objectives
- `opening_behavior` — `wait_for_agent` | `silent` | `say:Hello?`
- `dynamic_variables` — passed in `conversation_initiation_client_data`
- `rule_checks` — keyword/regex checks on agent transcript
- `llm_evaluation_prompt` — qualitative scoring criteria

Built-in scenarios:

| File | Persona |
|---|---|
| `cooperative.yaml` | Confirms address willingly |
| `wrong_address.yaml` | Gives wrong address, then corrects |
| `silent.yaml` | Minimal / silent responses |
| `interrupt.yaml` | Interrupts mid-sentence |
| `ask_repeat.yaml` | Asks agent to repeat |

## Project layout

```
src/inbound_tester/
  elevenlabs_ws.py      # WebSocket client (ping/pong, user_message/audio_chunk, events)
  tts_client.py         # ElevenLabs TTS REST client (pcm_16000 output)
  audio_bridge.py       # Bridges TTS audio ↔ WS user_audio_chunk
  persona_engine.py     # OpenAI persona replies
  conversation_runner.py # Text-mode conversation orchestration
  voice_runner.py       # Audio-mode conversation orchestration
  evaluator.py          # Rule checks + LLM judge
  logger.py             # SQLite persistence
  scenarios.py          # YAML schema
  config.py             # Settings / env
  cli.py                # CLI entry point
scenarios/              # Persona/scenario configs
```

## Audio mode details

In audio mode, the persona's text replies are converted to speech via the ElevenLabs TTS REST API (`output_format=pcm_16000`) and sent over the WebSocket as `user_audio_chunk` messages. The outbound agent's responses arrive as:
- `audio` events (base64 PCM) — the agent's spoken audio
- `agent_response` text — used directly as the transcript (no STT needed)
- `user_transcript` — what the agent "heard" from our audio (sanity check)

Turn-taking is sequential: wait for `agent_response_complete`, then generate and send the next persona reply.

## What this validates vs. what it doesn't

| Validates | Does not validate |
|---|---|
| Outbound agent conversation logic | Vonage SIP/PSTN path |
| ASR/LLM/TTS over ConvAI WebSocket | Real phone audio timing/quality |
| Persona handling & address flow | Production telephony latency |
| Audio pipeline (TTS → agent ASR) | Interruption/overlap handling |

## References

- [ElevenLabs Agent WebSocket API](https://elevenlabs.io/docs/eleven-agents/api-reference/eleven-agents/websocket)
- [Realtime monitoring / WS events](https://elevenlabs.io/docs/eleven-agents/guides/realtime-monitoring)
