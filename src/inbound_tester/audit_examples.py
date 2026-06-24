# #!/usr/bin/env python3
# """Usage examples and quick-start walkthrough for the Audit & Deduplication System.

# Run this script to see the complete flow:
#   1. Database initialization
#   2. Scenario registration (new + duplicate detection)
#   3. Execution logging
#   4. Coverage report generation
#   5. Audit trail retrieval

# Usage:
#     python -m inbound_tester.audit_examples
# """

# from __future__ import annotations

# import json
# import sys
# from pathlib import Path

# # Add project root to path for standalone execution
# project_root = Path(__file__).resolve().parent.parent.parent
# sys.path.insert(0, str(project_root))

# from inbound_tester.audit_system import AuditSystem


# def print_section(title: str) -> None:
#     """Print a formatted section header."""
#     print(f"\n{'='*72}")
#     print(f"  {title}")
#     print(f"{'='*72}")


# def print_json(data: dict, indent: int = 2) -> None:
#     """Pretty-print a JSON-serializable dictionary."""
#     print(json.dumps(data, indent=indent, default=str))


# def main() -> None:
#     # ──────────────────────────────────────────────────────────────────
#     # STEP 1: Initialize the Audit System
#     # ──────────────────────────────────────────────────────────────────
#     # ------------------------------------------------------------------
#     print_section("STEP 1: Initialize the Audit System")

#     db_path = project_root / "data" / "audit_demo.db"
#     print(f"Database path: {db_path}")

#     audit = AuditSystem(db_path)
#     print("[OK] Audit system initialized successfully")

#     # ------------------------------------------------------------------
#     # STEP 2: Register Test Case 1 -- New Scenario (Hospital Procurement)
#     # ------------------------------------------------------------------
#     print_section("STEP 2: Register Test Case 1 (New Scenario)")

#     # Simulate a real scenario combination from test_configurations.json
#     test_case_1_combo = {
#         "dimension_1_personas": {
#             "id": "persona_2",
#             "name": "Angry",
#             "description": "Hostile, aggressive, interrupts frequently",
#             "config": {"cooperation_level": 2, "patience_level": 1},
#         },
#         "dimension_2_scenarios": {
#             "id": "scenario_1",
#             "name": "Identity Verification",
#             "description": "Verify hospital identity and decision-maker",
#             "config": {"scenario_name": "identity_verification", "difficulty": "easy"},
#         },
#         "dimension_3_languages": {
#             "id": "lang_1",
#             "name": "English Only",
#             "config": {"primary_language": "en"},
#         },
#         "dimension_4_speech_styles": {
#             "id": "speech_1",
#             "name": "Normal Speaker",
#             "config": {"speed": "normal"},
#         },
#         "dimension_5_audio_environments": {
#             "id": "env_1",
#             "name": "Clean Audio",
#             "config": {"noise_level": "none"},
#         },
#         "dimension_6_interruptions": {
#             "id": "int_3",
#             "name": "Hostile Interruptions",
#             "config": {"frequency": "high"},
#         },
#         "dimension_7_questions": {
#             "id": "q_1",
#             "name": "Standard Questions",
#             "config": {"question_type": "direct"},
#         },
#         "dimension_8_compliance": {
#             "id": "comp_1",
#             "name": "Full Compliance",
#             "config": {"compliance_level": "full"},
#         },
#         "dimension_9_telephony": {
#             "id": "tel_1",
#             "name": "Normal Connection",
#             "config": {"signal_quality": "good"},
#         },
#         "dimension_10_data_availability": {
#             "id": "data_1",
#             "name": "All Data Available",
#             "config": {"data_completeness": "full"},
#         },
#         "dimension_11_emotional_states": {
#             "id": "emo_2",
#             "name": "Frustrated",
#             "config": {"intensity": "high"},
#         },
#         "dimension_12_stt_difficulty": {
#             "id": "stt_1",
#             "name": "Easy STT",
#             "config": {"difficulty": "easy"},
#         },
#         "dimension_13_conversation_states": {
#             "id": "conv_1",
#             "name": "Fresh Call",
#             "config": {"state": "new"},
#         },
#         "dimension_14_hospital_information": {
#             "id": "hosp_1",
#             "name": "Apollo Hospital Delhi",
#             "config": {"hospital_type": "multi_specialty"},
#         },
#     }

#     # Generate hash first to demonstrate the function
#     hash_1 = AuditSystem.generate_scenario_hash(test_case_1_combo)
#     print(f"Generated hash: {hash_1}")

#     # Check for duplicates
#     dup_check_1 = audit.check_duplicate(test_case_1_combo)
#     print(f"\nDuplicate check result:")
#     print_json(dup_check_1)

#     # Register the scenario
#     reg_result_1 = audit.register_scenario(
#         scenario_name="Angry Caller + Identity Verification (Apollo Hospital)",
#         dimension_combo=test_case_1_combo,
#         description="Hostile caller during identity verification at Apollo Hospital",
#         difficulty="hard",
#         created_by="demo_script",
#     )
#     print(f"\nRegistration result:")
#     print_json(reg_result_1)

#     # ------------------------------------------------------------------
#     # STEP 3: Log Execution of Test Case 1
#     # ------------------------------------------------------------------
#     print_section("STEP 3: Log Execution of Test Case 1")

#     exec_result_1 = audit.log_test_execution(
#         scenario_id=reg_result_1["scenario_id"],
#         status="passed",
#         test_run_id="test-run-001",
#         conversation_id="conv-el-001",
#         duration_sec=45.3,
#         turn_count=8,
#         avg_latency_ms=1250.5,
#         transcript=[
#             {"role": "agent", "text": "Hello, this is MedSupply calling for Apollo Hospital. Am I speaking with Dr. Sharma?"},
#             {"role": "user", "text": "Kaun ho tum? Kyun call kar rahe ho? I'm busy!"},
#             {"role": "agent", "text": "I understand you're busy, Dr. Sharma. This is regarding your recent equipment procurement inquiry."},
#             {"role": "user", "text": "I didn't make any inquiry! Stop bothering me!"},
#             {"role": "agent", "text": "I apologize for the inconvenience. Let me verify - is this Apollo Hospital, Sector 14?"},
#             {"role": "user", "text": "Yes it is, but I'm the wrong person. Talk to procurement."},
#             {"role": "agent", "text": "Thank you for clarifying. Could you help me with the procurement department's contact?"},
#             {"role": "user", "text": "Extension 245. Now let me go."},
#         ],
#         evaluation={
#             "passed": True,
#             "llm_score": 0.85,
#             "llm_summary": "Agent handled hostile caller well, de-escalated and obtained procurement contact.",
#             "rule_results": [
#                 {"id": "identity_confirmed", "passed": True, "detail": "Hospital confirmed"},
#                 {"id": "calm_response", "passed": True, "detail": "Agent remained professional"},
#             ],
#         },
#         triggered_by="demo_script",
#         execution_mode="regression",
#     )
#     print(f"Execution logged:")
#     print_json(exec_result_1)

#     # ------------------------------------------------------------------
#     # STEP 4: Test Case 2 Arrives -- DUPLICATE DETECTED
#     # ------------------------------------------------------------------
#     print_section("STEP 4: Test Case 2 (DUPLICATE) -- Same Configuration")

#     # Same combo as test_case_1 -- this is the deduplication test
#     test_case_2_combo = test_case_1_combo.copy()  # Identical configuration

#     dup_check_2 = audit.check_duplicate(test_case_2_combo)
#     print(f"Duplicate check result:")
#     print_json(dup_check_2)

#     if dup_check_2["is_duplicate"]:
#         print(f"\n[WARNING] DUPLICATE DETECTED!")
#         print(f"  Original scenario: {dup_check_2['existing_scenario']['id'][:8]}...")
#         print(f"  Recommendation: {dup_check_2['recommendation']}")

#         # Try to register -- will get back the existing scenario
#         reg_result_2 = audit.register_scenario(
#             scenario_name="Angry Caller + Identity Verification (Apollo Hospital)",
#             dimension_combo=test_case_2_combo,
#             description="This is a duplicate attempt",
#             created_by="demo_script",
#         )
#         print(f"\nRegistration result (duplicate handling):")
#         print_json(reg_result_2)

#         # Log as skipped
#         audit.log_test_execution(
#             scenario_id=reg_result_2["scenario_id"],
#             status="skipped_duplicate",
#             was_deduplicated=True,
#             dedup_record_id=reg_result_2.get("dedup_record_id"),
#             triggered_by="demo_script",
#             execution_mode="regression",
#         )
#         print("\n[OK] Duplicate logged as skipped -- saved ~2.5 minutes of test time")

#     # ------------------------------------------------------------------
#     # STEP 5: Register a DIFFERENT scenario (Cooperative + Address Change)
#     # ------------------------------------------------------------------
#     print_section("STEP 5: Register Test Case 3 (Different Scenario)")

#     test_case_3_combo = {
#         "dimension_1_personas":             {"name": "Cooperative"},
#         "dimension_2_scenarios":            {"name": "Address Change"},
#         "dimension_3_languages":            {"name": "Hindi to English Switch"},
#         "dimension_4_speech_styles":        {"name": "Slow Speaker"},
#         "dimension_5_audio_environments":   {"name": "Noisy Hospital Ward"},
#         "dimension_6_interruptions":        {"name": "No Interruptions"},
#         "dimension_7_questions":            {"name": "Clarification Questions"},
#         "dimension_8_compliance":           {"name": "Full Compliance"},
#         "dimension_9_telephony":            {"name": "Normal Connection"},
#         "dimension_10_data_availability":   {"name": "Missing Address"},
#         "dimension_11_emotional_states":    {"name": "Neutral"},
#         "dimension_12_stt_difficulty":      {"name": "Medium STT"},
#         "dimension_13_conversation_states": {"name": "Fresh Call"},
#         "dimension_14_hospital_information":{"name": "Fortis Hospital Mumbai"},
#     }

#     reg_result_3 = audit.register_scenario(
#         scenario_name="Cooperative + Address Change (Fortis Mumbai)",
#         dimension_combo=test_case_3_combo,
#         difficulty="medium",
#         created_by="demo_script",
#     )
#     print(f"Registration result:")
#     print_json(reg_result_3)

#     # Log a successful execution
#     audit.log_test_execution(
#         scenario_id=reg_result_3["scenario_id"],
#         status="passed",
#         duration_sec=62.1,
#         turn_count=12,
#         avg_latency_ms=980.0,
#         evaluation={"passed": True, "llm_score": 0.92},
#         triggered_by="demo_script",
#         execution_mode="coverage_driven",
#     )
#     print("[OK] Test Case 3 executed and logged")

#     # ------------------------------------------------------------------
#     # STEP 6: Generate Coverage Report (powered by test_configurations.json)
#     # ------------------------------------------------------------------
#     print_section("STEP 6: Coverage Report")

#     # Show what was loaded from test_configurations.json
#     config_summary = audit.get_test_configurations_summary()
#     print(f"Config file: {config_summary['config_path']}")
#     print(f"Dimensions loaded: {config_summary['total_dimensions']}")
#     print(f"Total possible combinations: {config_summary['total_possible_combinations']:,}")
#     print(f"\nDimension sizes (from test_configurations.json):")
#     for dim_key, count in config_summary["dimension_counts"].items():
#         print(f"  {dim_key}: {count} values")

#     # No need to pass dimension_totals -- auto-loaded from test_configurations.json!
#     coverage = audit.get_coverage_report()
#     print(f"\nOverall Coverage: {coverage['overall_coverage_pct']}%")
#     print(f"Total Scenarios Registered: {coverage['total_scenarios_registered']}")
#     print(f"Total Executions: {coverage['total_executions']}")
#     print(f"\nDeduplication Stats:")
#     print_json(coverage["dedup_stats"])
#     print(f"\nPer-Dimension Coverage:")
#     for dim_key, dim_data in coverage["dimensions"].items():
#         filled = int(dim_data["coverage_pct"] / 5)
#         bar = "#" * filled + "." * (20 - filled)
#         print(f"  {dim_key:45s} [{bar}] {dim_data['coverage_pct']:5.1f}%  ({dim_data['covered_values']}/{dim_data['total_values']})")
#         # Show gaps (untested values from test_configurations.json)
#         if dim_data.get("gap_list"):
#             gap_preview = dim_data["gap_list"][:5]
#             remaining = len(dim_data["gap_list"]) - 5
#             gap_str = ", ".join(gap_preview)
#             if remaining > 0:
#                 gap_str += f" (+{remaining} more)"
#             print(f"    Gaps: {gap_str}")

#     # ------------------------------------------------------------------
#     # STEP 7: Audit Trail for Test Case 1
#     # ------------------------------------------------------------------
#     print_section("STEP 7: Audit Trail")

#     trail = audit.get_audit_trail(scenario_id=reg_result_1["scenario_id"])
#     print(f"Scenario: {trail['scenario']['scenario_name']}")
#     print(f"Hash: {trail['scenario']['scenario_hash'][:16]}...")
#     print(f"\nSummary:")
#     print_json(trail["summary"])

#     print(f"\nExecution History ({len(trail['executions'])} records):")
#     for exec_rec in trail["executions"]:
#         print(f"  [{exec_rec['status']:20s}] {exec_rec['started_at']}  turns={exec_rec.get('turn_count', '-')}")

#     print(f"\nDeduplication Events ({len(trail['deduplication_events'])} records):")
#     for dedup in trail["deduplication_events"]:
#         print(f"  [{dedup['action_taken']:15s}] {dedup['detected_at']}  reason: {dedup['reason']}")

#     # ------------------------------------------------------------------
#     # STEP 8: Dashboard & Efficiency Stats
#     # ------------------------------------------------------------------
#     print_section("STEP 8: Dashboard Stats")

#     dashboard = audit.get_dashboard_stats()
#     print_json(dashboard)

#     efficiency = audit.get_efficiency_report()
#     print(f"\nEfficiency Report:")
#     print(f"  Total active scenarios: {efficiency['total_active_scenarios']}")
#     print(f"  Total executions: {efficiency['total_executions']}")
#     print(f"  Deduplicated runs: {efficiency['deduplicated_runs']}")
#     print(f"  Dedup rate: {efficiency['dedup_rate_pct']}%")
#     print(f"  Estimated time saved: {efficiency['estimated_time_saved_min']} minutes")

#     # ------------------------------------------------------------------
#     # STEP 9: Stale Scenarios
#     # ------------------------------------------------------------------
#     print_section("STEP 9: Stale Scenarios (not run in 7 days)")

#     stale = audit.get_stale_scenarios(days_threshold=7)
#     if stale:
#         for s in stale:
#             print(f"  {s['scenario_name']} -- last run: {s.get('last_executed_at', 'NEVER')}")
#     else:
#         print("  No stale scenarios found (all recently tested)")

#     # ------------------------------------------------------------------
#     # CLEANUP
#     # ------------------------------------------------------------------
#     print_section("DEMO COMPLETE")
#     print(f"\nDatabase saved at: {db_path}")
#     print("You can inspect it with any SQLite browser (DB Browser for SQLite, etc.)")
#     print("\nTo clean up the demo database:")
#     print(f"  del \"{db_path}\"")


# if __name__ == "__main__":
#     main()
