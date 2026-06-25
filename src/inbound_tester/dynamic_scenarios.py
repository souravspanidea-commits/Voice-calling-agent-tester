import json
import logging
import random
from pathlib import Path
from typing import Any

from inbound_tester.scenarios import RuleCheck, ScenarioConfig

logger = logging.getLogger(__name__)

class ScenarioGenerator:
    def __init__(self, data_dir: Path, scenarios_dir: Path):
        self.data_dir = data_dir
        self.scenarios_dir = scenarios_dir
        
        self.config_path = self.data_dir / "test_configurations.json"
        self.master_prompt_path = self.scenarios_dir / "master_prompt.yaml"
        
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
            
        with open(self.master_prompt_path, "r", encoding="utf-8") as f:
            self.master_prompt_template = f.read()
            
        self.dimensions = self.data.get("dimensions", {})
        self.coverage_tracker = {k: set() for k in self.dimensions.keys()}

    def _render_scenario(self, combo: dict[str, dict], override_id: str = None) -> ScenarioConfig:
        """Render a single ScenarioConfig from a combination of dimension choices."""
        combo = self._resolve_conflicts(combo)
        dim1 = combo.get("dimension_1_personas", {})
        dim2 = combo.get("dimension_2_scenarios", {})
        dim3 = combo.get("dimension_3_languages", {})
        dim4 = combo.get("dimension_4_speech_styles", {})
        dim5 = combo.get("dimension_5_audio_environments", {})
        dim6 = combo.get("dimension_6_interruptions", {})
        dim7 = combo.get("dimension_7_questions", {})
        dim8 = combo.get("dimension_8_compliance", {})
        dim9 = combo.get("dimension_9_telephony", {})
        dim10 = combo.get("dimension_10_data_availability", {})
        dim11 = combo.get("dimension_11_emotional_states", {})
        dim12 = combo.get("dimension_12_stt_difficulty", {})
        dim13 = combo.get("dimension_13_conversation_states", {})
        dim14 = combo.get("dimension_14_hospital_information", {})

        # Extract specific variables needed for template
        prompt = self.master_prompt_template
        
        # Outbound Agent Context
        facility_name = "Apollo Hospital"
        address = "42-B, Sector 14, Gurgaon"
        contact_number = "9876543210"
        
        s_config = dim2.get("config", {})
        expected_behavior = dim9.get("config", {}).get("expected_agent_action", "Act appropriately")
        
        active_config_lines = [
            "ACTIVE TEST CONFIGURATION FOR THIS CALL:",
            "---",
            f"FACILITY_NAME: {facility_name}",
            f"ADDRESS: {address}",
            f"HOSPITAL_PHONE: {contact_number}",
            "---",
            f"scenario_name: {dim2.get('name', 'Unknown')}",
            f"persona: {dim1.get('name', 'Unknown')}",
            f"language_mode: {dim3.get('name', 'Unknown')}",
            f"speech_style: {dim4.get('name', 'Unknown')}",
            f"audio_environment: {dim5.get('name', 'Unknown')}",
            f"interruption_behavior: {dim6.get('name', 'Unknown')}",
            f"question_behavior: {dim7.get('name', 'Unknown')}",
            f"compliance_behavior: {dim8.get('name', 'Unknown')}",
            f"telephony_behavior: {dim9.get('name', 'Unknown')}",
            f"data_availability: {dim10.get('name', 'Unknown')}",
            f"emotional_state: {dim11.get('name', 'Unknown')}",
            f"stt_difficulty: {dim12.get('name', 'Unknown')}",
            f"hospital_information: {dim14.get('name', 'Unknown')}",
            f"conversation_state: {dim13.get('name', 'Unknown')}",
        ]
        active_config_text = "\n".join(active_config_lines)
        
        # Build metadata
        selected_dimensions = {k: v.get("name") for k, v in combo.items()}
        
        # Calculate a pseudo difficulty (summing challenge scores or similar, simplistic for now)
        difficulty = "medium"
        if dim11.get("name") in ["angry", "frustrated"] or dim6.get("name") in ["hostile_interruptions"]:
            difficulty = "hard"

        scenario_id = override_id or f"DYN_{random.randint(1000, 9999)}"
        
        metadata = {
            "scenario_id": scenario_id,
            "difficulty": difficulty,
            "expected_agent_behavior": [expected_behavior],
            "selected_dimensions": selected_dimensions,
            "coverage_tags": list(selected_dimensions.values())
        }

        # Pass dynamic variables to the ScenarioConfig
        dynamic_vars = {
            "hospital_name": facility_name,
            "address": "42-B, Sector 14, Gurgaon",
            "contact_number": "9876543210"
        }

        eval_prompt = f"""
Goal: {s_config.get('goal', 'Unknown')}
Expected Outcome: {s_config.get('expected_outcome', 'Unknown')}
Success Criteria: {', '.join(s_config.get('success_criteria', []))}
Failure Modes: {', '.join(s_config.get('failure_modes', []))}
Required Data: {', '.join(s_config.get('required_data_points', []))}
Expected Agent Behavior: {expected_behavior}

Check if the agent successfully handled the scenario according to the criteria above.
Score 1.0 for perfect handling, 0.5 for partial success, 0.0 for failure.
"""

        rule_checks = [
            RuleCheck(
                id="no_ai_disclosure",
                description="Agent must not reveal it is an AI",
                type="keyword_none",
                role="agent",
                patterns=["language model", "openai", "artificial intelligence"]
            ),
            RuleCheck(
                id="turn_limit",
                description="Conversation must complete within 30 turns",
                type="max_agent_turns",
                role="agent",
                patterns=["30"]
            )
        ]

        # The ScenarioConfig requires id, name, persona_prompt. 
        return ScenarioConfig(
            id=scenario_id,
            name=f"Dynamic Scenario: {dim2.get('name')} + {dim1.get('name')}",
            description=f"Generated combination: {difficulty}",
            persona_prompt=prompt,
            active_configuration=active_config_text,
            llm_evaluation_prompt=eval_prompt,
            rule_checks=rule_checks,
            dynamic_variables=dynamic_vars,
            metadata=metadata
        )

    def generate_regression_suite(self, n: int | None = None) -> list[ScenarioConfig]:
        """Generate deterministic curated scenarios that must run every time.
        
        Uses a seeded generator for unspecified dimensions so that the
        exact same scenario combination (and thus hash) is produced on
        every run, enabling reliable deduplication.
        """
        scenarios = []
        # Create dummy combinations matching the requested ones
        regression_keys = [
            ("LANG_001", "Universal Word Test"),
            ("LANG_002", "Hindi -> English Switch"),
            ("FLOW_001", "Happy Path"),
            ("PHONE_001", "Existing Number Confirmation"),
            ("HOSTILE_001", "Hostile Customer"),
            ("BUSY_001", "Busy Customer Callback")
        ]
        
        if n is not None:
            regression_keys = regression_keys[:n]
            
        for key, name in regression_keys:
            # Deterministic base combination
            combo = self._get_deterministic_combo(seed=key)
            
            # Override specific dimensions to match the test case
            if "HOSTILE" in key:
                combo["dimension_1_personas"] = self._find_by_name("dimension_1_personas", "Angry")
            elif "BUSY" in key:
                combo["dimension_1_personas"] = self._find_by_name("dimension_1_personas", "Busy")
            elif "LANG_002" in key:
                combo["dimension_3_languages"] = self._find_by_name("dimension_3_languages", "Hindi to English Switch")
            elif "FLOW_001" in key:
                combo["dimension_1_personas"] = self._find_by_name("dimension_1_personas", "Cooperative")

            cfg = self._render_scenario(combo, override_id=key)
            scenarios.append(cfg)
        return scenarios

    def generate_random_sampling(self, n: int) -> list[ScenarioConfig]:
        scenarios = []
        for _ in range(n):
            combo = self._get_random_combo()
            scenarios.append(self._render_scenario(combo))
        return scenarios

    def generate_coverage_driven(self, n: int) -> list[ScenarioConfig]:
        scenarios = []
        for _ in range(n):
            combo = {}
            for dim_key, dim_list in self.dimensions.items():
                # Find an item we haven't covered
                uncovered = [item for item in dim_list if item["name"] not in self.coverage_tracker[dim_key]]
                if uncovered:
                    choice = random.choice(uncovered)
                else:
                    choice = random.choice(dim_list)
                self.coverage_tracker[dim_key].add(choice["name"])
                combo[dim_key] = choice
            scenarios.append(self._render_scenario(combo))
        return scenarios

    def generate_stress_testing(self, n: int) -> list[ScenarioConfig]:
        scenarios = []
        for _ in range(n):
            combo = self._get_random_combo()
            # Force difficult traits
            combo["dimension_1_personas"] = self._find_by_name("dimension_1_personas", "angry")
            combo["dimension_4_speech_styles"] = self._find_by_name("dimension_4_speech_styles", "heavy_accent_speaker")
            combo["dimension_6_interruptions"] = self._find_by_name("dimension_6_interruptions", "hostile_interruptions")
            scenarios.append(self._render_scenario(combo))
        return scenarios

    def _get_random_combo(self) -> dict[str, dict]:
        combo = {}
        for dim_key, dim_list in self.dimensions.items():
            if dim_list:
                combo[dim_key] = random.choice(dim_list)
            else:
                combo[dim_key] = {}
        return combo

    def _get_deterministic_combo(self, seed: str) -> dict[str, dict]:
        """Return the same random combination for a given seed."""
        rng = random.Random(seed)
        combo = {}
        # Sort keys to ensure consistent iteration order
        for dim_key in sorted(self.dimensions.keys()):
            dim_list = self.dimensions[dim_key]
            if dim_list:
                combo[dim_key] = rng.choice(dim_list)
            else:
                combo[dim_key] = {}
        return combo

    def _find_by_name(self, dim_key: str, name: str) -> dict:
        """Case-insensitive search for a dimension value by name."""
        target = name.lower()
        for item in self.dimensions.get(dim_key, []):
            if item.get("name", "").lower() == target:
                return item
        return {}

    def _resolve_conflicts(self, combo: dict) -> dict:
        """Resolve logically conflicting dimensions to ensure valid test cases."""
        combo = combo.copy()

        # ── fresh read helper (prevents stale local variable bugs) ────────────
        def _s(dim):
            return combo.get(dim, {}).get("name", "")
        
        # helper to find dimension objects by name, with safe dynamic fallback
        def find(dim, val_name):
            res = self._find_by_name(dim, val_name)
            if not res:
                return {"name": val_name, "config": {}}
            return res

        # ═════════════════════════════════════════════════════════════════
        # STEP 1 — Telephony
        # Must run first. Locks Scenario, State, Persona, Emotion for
        # automated lines. Early-return skips Steps 2–5 for Voicemail/IVR.
        # ═════════════════════════════════════════════════════════════════
        telephony = _s("dimension_9_telephony")

        if telephony in ("Voicemail", "IVR Menu"):
            scenario_name = "Voicemail" if telephony == "Voicemail" else "IVR"
            combo["dimension_2_scenarios"]            = find("dimension_2_scenarios", scenario_name)
            combo["dimension_13_conversation_states"] = find("dimension_13_conversation_states", "initial_greeting")
            combo["dimension_1_personas"]             = find("dimension_1_personas", "Automated System")
            combo["dimension_11_emotional_states"]    = find("dimension_11_emotional_states", "Neutral")
            return combo  # ← early return; all other steps are no-ops

        elif telephony == "Call Transfer":
            combo["dimension_2_scenarios"] = find("dimension_2_scenarios", "Call Transfer")

        # ═════════════════════════════════════════════════════════════════
        # STEP 2 — Conversation State
        # Mid-call state takes priority over hospital info.
        # May rewrite both Scenario AND Hospital Info.
        # Step 3 reads post-Step-2 values only.
        # ═════════════════════════════════════════════════════════════════
        STATE_TO_SCENARIO = {
            "address_verification":        "Address Verification",
            "procurement_name_collection": "Procurement Name Collection",
            "phone_collection":            "Phone Collection",
            "email_collection":            "Email Collection",
        }
        conv_state = _s("dimension_13_conversation_states")

        if conv_state == "wrap_up":
            combo["dimension_13_conversation_states"] = find("dimension_13_conversation_states", "initial_greeting")

        elif conv_state in STATE_TO_SCENARIO:
            # mid-call state takes priority over hospital_info; Step 2 clears hospital_info and sets Scenario to prevent Step 3 from undoing the state-driven scenario. Step 3 operates on the post-Step-2 values only.
            combo["dimension_2_scenarios"]              = find("dimension_2_scenarios", STATE_TO_SCENARIO[conv_state])
            combo["dimension_14_hospital_information"]  = find("dimension_14_hospital_information", "correct_hospital")

        # ═════════════════════════════════════════════════════════════════
        # STEP 3 — Hospital Info ↔ Scenario (Bidirectional)
        # Skip entirely if Step 2 set hospital_info to correct_hospital is handled
        # naturally by conditional checks since mid-call scenarios won't match.
        # ═════════════════════════════════════════════════════════════════
        hospital_info = _s("dimension_14_hospital_information")
        scenario      = _s("dimension_2_scenarios")

        if hospital_info == "wrong_hospital" or scenario == "Wrong Hospital":
            combo["dimension_2_scenarios"]            = find("dimension_2_scenarios", "Wrong Hospital")
            combo["dimension_14_hospital_information"] = find("dimension_14_hospital_information", "wrong_hospital")
            combo["dimension_13_conversation_states"] = find("dimension_13_conversation_states", "initial_greeting")

        elif hospital_info == "address_changed" or scenario == "Address Change":
            combo["dimension_2_scenarios"]            = find("dimension_2_scenarios", "Address Change")
            combo["dimension_14_hospital_information"] = find("dimension_14_hospital_information", "address_changed")

        elif hospital_info == "procurement_unknown" or scenario == "Procurement Unknown":
            combo["dimension_2_scenarios"]            = find("dimension_2_scenarios", "Procurement Unknown")
            combo["dimension_14_hospital_information"] = find("dimension_14_hospital_information", "procurement_unknown")

        # ═════════════════════════════════════════════════════════════════
        # STEP 4 — Data Availability
        # Read scenario after Steps 1–3 have fully settled it.
        # ═════════════════════════════════════════════════════════════════
        scenario   = _s("dimension_2_scenarios")
        data_avail = _s("dimension_10_data_availability")

        SCENARIO_DATA_RULES = {
            "Address Verification":        ["Knows Everything", "Knows Address Only"],
            "Address Change":              ["Knows Everything"],
            "SSML Compliance Test":        ["Knows Everything", "Knows Address Only"],
            "Email Collection":            ["Knows Everything", "Knows Email Only"],
            "Phone Collection":            ["Knows Everything", "Knows Phone Only"],
            "Procurement Name Collection": ["Knows Everything", "Knows Procurement Only"],
            "Wrong Hospital":              ["Knows Everything"],
            "Data Conflict":               ["Knows Everything"],
            "Procurement Unknown":         ["Knows Nothing"],
        }

        if scenario in SCENARIO_DATA_RULES:
            allowed = SCENARIO_DATA_RULES[scenario]
            if data_avail not in allowed:
                combo["dimension_10_data_availability"] = find("dimension_10_data_availability", allowed[0])

        # ═════════════════════════════════════════════════════════════════
        # STEP 5 — Persona → Emotion
        # Read persona after Step 1 may have set it to Automated System.
        # ═════════════════════════════════════════════════════════════════
        persona = _s("dimension_1_personas").lower()
        emotion = _s("dimension_11_emotional_states")

        PERSONA_EMOTION_RULES = {
            "automated system": (["Neutral"],                    "exact"),
            "silent":           (["Neutral", "Calm"],            "exact"),
            "angry":            (["Angry", "Frustrated", "Stressed"], "random"),
            "hostile":          (["Angry", "Frustrated", "Stressed"], "random"),
            "impatient":        (["Angry", "Frustrated", "Stressed"], "random"),
            "cooperative":      (["Calm", "Neutral", "Happy"],   "random"),
            "busy":             (["Stressed", "Frustrated", "Neutral"], "exact_first"),
        }

        if persona in PERSONA_EMOTION_RULES:
            allowed_emotions, mode = PERSONA_EMOTION_RULES[persona]
            if emotion not in allowed_emotions:
                if mode == "random":
                    import random
                    combo["dimension_11_emotional_states"] = find("dimension_11_emotional_states", random.choice(allowed_emotions))
                else:  # exact or exact_first
                    combo["dimension_11_emotional_states"] = find("dimension_11_emotional_states", allowed_emotions[0])

        return combo


