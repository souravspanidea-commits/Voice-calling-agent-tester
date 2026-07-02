"""OpenAI-powered persona engine for text-only test conversations."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import AsyncGenerator

from openai import AsyncOpenAI

from inbound_tester.scenarios import ScenarioConfig

logger = logging.getLogger(__name__)

PERSONA_SYSTEM_TEMPLATE = """## YOUR IDENTITY — READ THIS FIRST

You are a HOSPITAL EMPLOYEE (receptionist, nurse, or admin staff) sitting at your desk.
Your phone is RINGING. You are about to ANSWER it.

You did NOT make this call. You are RECEIVING this call.
The person on the other end is an OUTBOUND SALES/VERIFICATION AGENT calling YOUR hospital.

Their job: to call hospitals, verify data, and collect information.
Your job: to answer the phone and respond naturally — nothing more.

---

## YOUR CHARACTER
{persona_prompt}

## SCENARIO GOALS
{goals}

## LANGUAGE
{language_instructions}

---

## RULES

- Reply with ONLY what you would say aloud — no stage directions, no labels, no meta-commentary.
- Stay in character at all times.
- Keep replies concise and natural for a phone call (1-3 sentences usually).
- Speak slowly and clearly. Use frequent commas, periods, and ellipses (...) to pace your speech naturally.
- Avoid long, rushed run-on sentences. Break your thoughts into short, distinct phrases.
- To disconnect the call (only when your scenario instructions allow it, or if the other agent remains silent for 15+ seconds), reply with exactly: {end_signal}
- To remain silent on your turn, reply with exactly: [SILENT]
- Never reveal you are an AI or a test agent.

---

## ⚠️ ROLE LOCK — YOU ARE THE DATA SOURCE, NOT THE CALLER

YOU are sitting at the hospital. They are calling YOU.
YOU answer questions. They ask questions.

FORBIDDEN — You may NEVER say these things:
- "Can you provide your phone number?"
- "What address do you have on file?"
- "Can you confirm...?" (about any data field)
- "What information do you have for us?"
- Anything that asks the caller to give YOU data

PERMITTED — You may only ask:
- "Who is calling?" / "What company are you from?"
- "What is this regarding?" / "Why do you need that?"
- "Sorry, could you repeat that?"

---

## ⚠️ GOODBYE LOCK — ONCE YOU SAY GOODBYE, YOU ARE DONE

Once you say a closing phrase ("goodbye", "have a good day", "dhanyavaad", "take care", "bye"):
- Do NOT speak again under any circumstance.
- Do NOT volunteer new information, names, or data after goodbye.
- Stay silent and wait for the caller to disconnect.
- If caller remains silent for 15+ seconds, reply with exactly: {end_signal}
"""



DEFAULT_LANGUAGE_INSTRUCTIONS = """Respond in a natural mix of Hindi and English (Hinglish), like a typical Indian phone conversation.
- Use Hindi as the base language with English words mixed in naturally (e.g., "Haan ji, procurement department mein Dr. Priya Sharma handle karti hain").
- Use English for technical terms, names, numbers, email addresses, and business jargon.
- Keep the tone conversational and natural — the way an Indian professional would actually speak on the phone.
- Write Hindi in Roman script (transliteration), NOT Devanagari. For example: "Haan bilkul, address sahi hai" not "हाँ बिलकुल".
- SPELLING OUT FOR PRONUNCIATION: Write out numbers word-by-word (e.g., 'nine eight seven' instead of '987'). Add spaces between acronym letters (e.g., 'H D F C' instead of 'HDFC').
- PACE CONTROL: Use frequent commas ( , ) and ellipses ( ... ) to force the TTS engine to speak slowly and naturally.
- HUMAN-LIKENESS: Frequently start sentences or insert natural conversational fillers (e.g., "Umm...", "Hmm...", "Achha...", "Haan...", "Dekhiye..."). This makes the AI voice sound like a real person thinking.
"""
 

@dataclass
class PersonaDecision:
    text: str
    should_end: bool
    is_silent: bool
    raw_response: str = ""


@dataclass
class PersonaDecisionChunk:
    text: str
    is_last: bool
    should_end: bool
    is_silent: bool


class PersonaEngine:
    def __init__(self, api_key: str, model: str) -> None:
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def decide_next_reply(
        self,
        scenario: ScenarioConfig,
        transcript: list[dict[str, str]],
        *,
        agent_last_message: str | None = None,
    ) -> PersonaDecision:
        end_signals = scenario.end_signals
        primary_end = end_signals[0] if end_signals else "[END]"

        goals_text = "\n".join(f"- {g}" for g in scenario.goals) or "- Respond naturally in character."
        language_instructions = getattr(scenario, "language_instructions", None) or DEFAULT_LANGUAGE_INSTRUCTIONS
        system = PERSONA_SYSTEM_TEMPLATE.format(
            persona_prompt=scenario.persona_prompt.strip(),
            goals=goals_text,
            language_instructions=language_instructions,
            end_signal=primary_end,
        )

        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        if getattr(scenario, "active_configuration", ""):
            messages.append({"role": "system", "content": scenario.active_configuration})

        for entry in transcript:
            role = entry["role"]
            text = entry["text"]
            if role in ("user", "user_heard"):
                # Our persona's previous replies → assistant role for the LLM
                messages.append({"role": "assistant", "content": text})
            elif role == "agent":
                # The outbound agent's messages → user role (what we respond to)
                messages.append({"role": "user", "content": text})

        if agent_last_message and (
            not transcript or transcript[-1].get("text") != agent_last_message
        ):
            messages.append({"role": "user", "content": agent_last_message})

        messages.append(
            {
                "role": "user",
                "content": "What do you say next as the hospital employee? Reply with spoken text only. Remember to use commas and ellipses (...) frequently to force a slow, natural speaking pace.",
            }
        )

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_completion_tokens=300,
        )
        raw = (response.choices[0].message.content or "").strip()
        logger.debug("Persona raw reply: %s", raw)

        upper = raw.upper()
        should_end = any(signal.upper() in upper for signal in end_signals)
        is_silent = "[SILENT]" in upper

        clean = raw
        for signal in end_signals + ["[SILENT]"]:
            clean = clean.replace(signal, "").strip()

        if should_end and not clean:
            clean = primary_end

        return PersonaDecision(
            text=clean or raw,
            should_end=should_end,
            is_silent=is_silent,
            raw_response=raw,
        )

    async def decide_next_reply_stream(
        self,
        scenario: ScenarioConfig,
        transcript: list[dict[str, str]],
        *,
        agent_last_message: str | None = None,
    ) -> AsyncGenerator[PersonaDecisionChunk, None]:
        end_signals = scenario.end_signals
        primary_end = end_signals[0] if end_signals else "[END]"

        goals_text = "\n".join(f"- {g}" for g in scenario.goals) or "- Respond naturally in character."
        language_instructions = getattr(scenario, "language_instructions", None) or DEFAULT_LANGUAGE_INSTRUCTIONS
        system = PERSONA_SYSTEM_TEMPLATE.format(
            persona_prompt=scenario.persona_prompt.strip(),
            goals=goals_text,
            language_instructions=language_instructions,
            end_signal=primary_end,
        )

        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        if getattr(scenario, "active_configuration", ""):
            messages.append({"role": "system", "content": scenario.active_configuration})

        for entry in transcript:
            role = entry["role"]
            text = entry["text"]
            if role in ("user", "user_heard"):
                messages.append({"role": "assistant", "content": text})
            elif role == "agent":
                messages.append({"role": "user", "content": text})

        if agent_last_message and (
            not transcript or transcript[-1].get("text") != agent_last_message
        ):
            messages.append({"role": "user", "content": agent_last_message})

        messages.append(
            {
                "role": "user",
                "content": "What do you say next as the hospital employee? Reply with spoken text only. Remember to use commas and ellipses (...) frequently to force a slow, natural speaking pace.",
            }
        )

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_completion_tokens=300,
            stream=True,
        )

        raw = ""
        buffer = ""
        should_end = False
        is_silent = False
        upper_end_signals = [s.upper() for s in end_signals]
        if not upper_end_signals:
            upper_end_signals = ["[END]"]

        async for chunk in response:
            content = chunk.choices[0].delta.content or ""
            if not content:
                continue

            raw += content
            buffer += content

            if "[" in buffer:
                idx = buffer.find("[")
                if "]" not in buffer[idx:]:
                    if len(buffer) - idx < 15:
                        continue

            upper_buf = buffer.upper()
            if "[SILENT]" in upper_buf:
                is_silent = True
                buffer = re.sub(r'\[SILENT\]', '', buffer, flags=re.IGNORECASE)
                upper_buf = buffer.upper()

            for sig in upper_end_signals:
                if sig in upper_buf:
                    should_end = True
                    buffer = re.sub(re.escape(sig), '', buffer, flags=re.IGNORECASE)
                    upper_buf = buffer.upper()

            if buffer:
                match = re.search(r'([\s\.,!\?]+)(?!.*[\s\.,!\?])', buffer)
                if match:
                    split_idx = match.end()
                    to_yield = buffer[:split_idx]
                    buffer = buffer[split_idx:]
                    if to_yield.strip():
                        yield PersonaDecisionChunk(
                            text=to_yield,
                            is_last=False,
                            should_end=False,
                            is_silent=False
                        )

        if buffer.strip():
            yield PersonaDecisionChunk(
                text=buffer,
                is_last=False,
                should_end=False,
                is_silent=False
            )

        yield PersonaDecisionChunk(
            text="",
            is_last=True,
            should_end=should_end,
            is_silent=is_silent
        )

    def generate_opening(self, scenario: ScenarioConfig) -> PersonaDecision | None:
        if scenario.opening_behavior == "wait_for_agent":
            return None
        if scenario.opening_behavior == "silent":
            return PersonaDecision(text="", should_end=False, is_silent=True)
        if scenario.opening_behavior.startswith("say:"):
            text = scenario.opening_behavior.removeprefix("say:").strip()
            return PersonaDecision(text=text, should_end=False, is_silent=False)
        return None
