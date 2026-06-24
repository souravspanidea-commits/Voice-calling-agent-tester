# Medtronic / Monika — Verification Agent Test Suite
# README — Scenario Index and Usage Guide
# Version 2.0

## Overview

This suite tests "Monika", Medtronic's outbound verification AI agent, across
12 scenarios covering the full call flow and every real-world edge case.
Each scenario is a self-contained YAML file with its own persona, rule checks,
and LLM evaluation rubric.

All scenarios inherit shared base config from _base_shared.yaml.

---

## Scenario Index

| File | ID | What It Tests | Priority |
|------|----|---------------|----------|
| scenario_01_happy_path.yaml | S01 | Full 6-step flow, mild friction, baseline control | 🔴 CRITICAL |
| scenario_02_ivr_then_human.yaml | S02 | IVR detection, silence during IVR, hold, time-pressured caller | 🔴 CRITICAL |
| scenario_03_wrong_person_transfer.yaml | S03 | Wrong person picks up, transfer accepted, data from 2nd contact | 🔴 CRITICAL |
| scenario_04_address_changed.yaml | S04 | Address correction, SSML on corrected address, correction readback | 🔴 CRITICAL |
| scenario_05_email_refused.yaml | S05 | Hard refusal on email, no re-ask, clean close with partial data | 🔴 CRITICAL |
| scenario_06_partial_phone_number.yaml | S06 | 9-digit number, single re-ask rule, digit-by-digit readback | 🟡 HIGH |
| scenario_07_hindi_only_caller.yaml | S07 | Sustained Hindi mirroring, universal word handling, SSML compliance | 🟡 HIGH |
| scenario_08_voicemail.yaml | S08 | Voicemail detection, silence, correct tool sequence | 🟡 HIGH |
| scenario_09_hostile_caller.yaml | S09 | Hostility detection, polite close, no escalation, no flow attempt | 🟡 HIGH |
| scenario_10_callback_request.yaml | S10 | Busy caller, callback time collection and confirmation, early exit | 🟡 HIGH |
| scenario_11_caller_is_procurement_lead.yaml | S11 | Caller self-identifies as lead, name from same person, flow continues | 🟢 MEDIUM |
| scenario_12_out_of_scope.yaml | S12 | Three OOS interruptions, deflect + re-ask in same turn, no drift | 🟢 MEDIUM |

---

## What Each File Contains

Every scenario YAML has these sections:

```
id:                      Unique identifier for CI/test runner
name:                    Human-readable label
description:             What the scenario tests and why

inherits:                Points to _base_shared.yaml (shared variables, max_turns)

persona_prompt:          The tester AI's character, behavior rules, and
                         step-by-step response instructions

opening_behavior:        What the tester says when the call is first answered

rule_checks:             Fast keyword checks (before LLM eval)
                         - type: keyword_any  → PASS if any pattern found
                         - type: keyword_none → PASS if no pattern found
                         - role: agent        → checks Monika's turns only
                         - role: tool_use     → checks tool invocations
                         - required: true     → FAIL if check fails

llm_evaluation_prompt:   Scoring rubric (1–5 per criterion) + PASS/FAIL gates
```

---

## Running Order Recommendation

Run in this order for fastest failure discovery:

1. S01 (happy path) — if this fails, all others will too
2. S08 (voicemail) — binary test, fast
3. S09 (hostile) — binary test, fast
4. S02 (IVR) — foundational infra test
5. S05 (email refused) — tests a hard rule
6. S10 (callback) — tests early exit
7. S04 (address changed) — tests SSML on dynamic content
8. S03 (transfer) — multi-character, complex
9. S06 (partial phone) — validation logic
10. S07 (Hindi only) — language engine test
11. S11 (caller is lead) — edge case branch
12. S12 (OOS questions) — deflection stress test

---

## Shared Pass/Fail Logic

Every scenario applies the UNIVERSAL PASS CRITERIA from _base_shared.yaml:

  PASS if ALL of:
  1. Identity Gate criterion >= 4  (waived in S09 and S10)
  2. Clean Close criterion >= 3
  3. Overall average >= 3.2

  Plus scenario-specific gates defined in each file.

---

## Scoring Summary — What to Watch

| Behavior | Tests in | Common Failure Mode |
|----------|----------|---------------------|
| Hindi SSML on names | S01, S04, S07, S11 | Name spoken without <lang> tag |
| Address SSML (text Hindi, numbers plain) | S01, S04, S07 | Numbers also wrapped in SSML |
| Digit-by-digit phone readback | S01, S03, S06, S11 | Full number read at once |
| Email spell-back (char by char) | S01, S05, S06, S11, S12 | Spell-back skipped |
| IVR silence | S02 | Greeting spoken before human picks up |
| Voicemail silence + tool sequence | S08 | Message left, or wrong tool order |
| Language mirroring | S01, S07, S12 | Staying Hindi after English response |
| Universal word no-switch | S07 | Switching on "Haan" or "Theek hai" |
| Hard refusal acceptance | S05 | Asking for email a second time |
| Transfer acceptance | S03 | Pushing data questions on wrong person |
| Deflect + re-ask same turn | S12 | Two-turn gap between deflect and re-ask |
| Hostility close | S09 | Flow steps attempted after hostility |
| Callback exit | S10 | Verification steps attempted on busy caller |

---

## Notes for Prompt Engineers

1. SSML appears in the raw transcript — the LLM evaluator sees the tags
   directly. Instruct your evaluator to check for <lang xml:lang="hi-IN">
   around every name and address locality.

2. Tool invocations (end_call, voicemail_detection, play_keypad_touch_tone)
   appear as tool_use blocks in the transcript. Your rule_checks use
   role: tool_use to target these.

3. The persona_prompt uses [TRANSFER] and [END] as structured signals.
   Your test runner should handle these as conversation control events,
   not as spoken text.

4. S03 uses a two-character persona (Suresh → Kavita). Your framework
   must support mid-call persona switching triggered by [TRANSFER].

5. S06 has a Variant B mode (second invalid number). Activate via
   variant: B in the test runner config to run the harder path.
