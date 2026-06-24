"""Quick script to find Hindi/Indian voices on ElevenLabs."""
import httpx
import os

API_KEY = os.getenv("ELEVENLABS_API_KEY")
resp = httpx.get("https://api.elevenlabs.io/v1/voices", headers={"xi-api-key": API_KEY})
voices = resp.json().get("voices", [])

print(f"Total voices: {len(voices)}\n")

# Find Hindi/Indian voices
hindi_voices = []
for v in voices:
    labels = v.get("labels", {})
    lang = labels.get("language", "").lower()
    accent = labels.get("accent", "").lower()
    name = v.get("name", "")
    vid = v.get("voice_id", "")
    
    if "hindi" in lang or "hindi" in accent or "indian" in accent or "indian" in lang:
        hindi_voices.append(v)
        print(f"  {name:25s} | {vid} | lang={lang} accent={accent}")

if not hindi_voices:
    print("No Hindi/Indian voices found. Showing first 20 voices:")
    for v in voices[:20]:
        labels = v.get("labels", {})
        name = v.get("name", "")
        vid = v.get("voice_id", "")
        lang = labels.get("language", "?")
        accent = labels.get("accent", "?")
        print(f"  {name:25s} | {vid} | lang={lang} accent={accent}")
