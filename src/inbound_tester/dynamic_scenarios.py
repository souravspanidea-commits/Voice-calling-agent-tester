import json
import logging
import random
from pathlib import Path
from typing import Any

from inbound_tester.scenarios import ScenarioConfig

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
            f"PHONE_DIALED: {contact_number}",
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
            "---",
            f"goal: {s_config.get('goal', 'Unknown')}",
            f"expected_outcome: {s_config.get('expected_outcome', 'Unknown')}",
            f"difficulty: {s_config.get('difficulty', 'Unknown')}",
            f"success_criteria: {', '.join(s_config.get('success_criteria', []))}",
            f"failure_modes: {', '.join(s_config.get('failure_modes', []))}",
            f"required_data_points: {', '.join(s_config.get('required_data_points', []))}",
            f"expected_agent_behavior: {expected_behavior}"
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

        # The ScenarioConfig requires id, name, persona_prompt. 
        return ScenarioConfig(
            id=scenario_id,
            name=f"Dynamic Scenario: {dim2.get('name')} + {dim1.get('name')}",
            description=f"Generated combination: {difficulty}",
            persona_prompt=prompt,
            active_configuration=active_config_text,
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
