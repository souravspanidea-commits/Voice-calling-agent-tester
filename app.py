import asyncio
from pathlib import Path
import sys

# Ensure src/ is in the python path
sys.path.append(str(Path(__file__).parent / "src"))

import pandas as pd
import streamlit as st

from inbound_tester.config import get_settings
from inbound_tester.conversation_runner import ConversationRunner
from inbound_tester.voice_runner import VoiceRunner
from inbound_tester.logger import ConversationLogger
from inbound_tester.scenarios import load_scenarios

# Page config
st.set_page_config(page_title="Inbound Tester Agent", page_icon="🤖", layout="wide")

st.title("Inbound Tester Agent Dashboard")

tab1, tab2, tab3, tab4 = st.tabs(["Run Scenario", "Dynamic Suite", "Past Runs", "Evaluation Dashboard"])

with tab1:
    st.header("Run a New Test")
    
    scenarios_dir = Path("scenarios")
    try:
        scenarios = load_scenarios(scenarios_dir)
        scenario_map = {s.id: s for s in scenarios}
    except Exception as e:
        st.error(f"Could not load scenarios: {e}")
        scenario_map = {}
        
    if scenario_map:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Configuration")
            selected_scenario_id = st.selectbox("Select Scenario", list(scenario_map.keys()))
            mode = st.radio("Mode", ["text", "audio"], index=0)
            agent_id = st.text_input("Agent ID (Override)", value="", help="Leave blank to use .env value")
            skip_eval = st.checkbox("Skip Evaluation", value=False)
            
            run_button = st.button("Run Test", type="primary", use_container_width=True)
            
        with col2:
            st.subheader("Execution View")
            if run_button:
                with st.spinner(f"Running scenario: {selected_scenario_id}..."):
                    cfg = scenario_map[selected_scenario_id]
                    
                    # Setup settings
                    test_settings = get_settings()
                    if agent_id:
                        test_settings.elevenlabs_agent_id = agent_id
                    test_settings.conversation_mode = mode
                    
                    if test_settings.conversation_mode == "audio":
                        runner = VoiceRunner(test_settings)
                    else:
                        runner = ConversationRunner(test_settings)
                    
                    try:
                        result = asyncio.run(runner.run_scenario(cfg, skip_evaluation=skip_eval))
                        res_dict = result.to_dict()
                        
                        passed_eval = res_dict.get("evaluation", {}).get("passed")
                        if passed_eval is True:
                            st.success("Test Complete! Result: PASS ✅")
                        elif passed_eval is False:
                            st.error("Test Complete! Result: FAIL ❌")
                        else:
                            st.info(f"Test Complete! Status: {res_dict.get('status')}")
                            
                        # Metrics
                        m1, m2, m3, m4, m5 = st.columns(5)
                        m1.metric("Status", res_dict.get('status'))
                        
                        duration = res_dict.get('duration_sec')
                        m2.metric("Duration", f"{duration:.2f}s" if duration else "N/A")
                        
                        tester_lat = res_dict.get('avg_tester_latency_ms')
                        m3.metric("🧑 Tester Latency", f"{tester_lat:.0f}ms" if tester_lat else "N/A")
                        
                        outbound_lat = res_dict.get('avg_outbound_latency_ms')
                        m4.metric("🤖 Outbound Latency", f"{outbound_lat:.0f}ms" if outbound_lat else "N/A")
                        
                        latency = res_dict.get('avg_latency_ms')
                        m5.metric("⏱️ Total Turn Avg", f"{latency:.0f}ms" if latency else "N/A")
                        
                        # Tabs for Output
                        out_tab1, out_tab2 = st.tabs(["Transcript", "Evaluation"])
                        
                        with out_tab1:
                            for msg in res_dict.get("transcript", []):
                                role = msg.get('role', '')
                                text = msg.get('text', '')
                                timing = ""
                                if role.lower() == 'user':
                                    t_ms = msg.get('tester_response_ms')
                                    if t_ms is not None:
                                        llm = msg.get('llm_ms')
                                        tts = msg.get('tts_ms')
                                        if llm is not None and tts is not None and tts > 0:
                                            timing = f" `{t_ms:.0f}ms` *(TTFT: {llm:.0f}ms, TTFA: {tts:.0f}ms)*"
                                        elif t_ms is not None:
                                            timing = f" `{t_ms:.0f}ms`"
                                    st.markdown(f"🧑 **Persona:** {text}{timing}")
                                elif role.lower() == 'agent':
                                    o_ms = msg.get('outbound_response_ms')
                                    if o_ms is not None:
                                        timing = f" `{o_ms:.0f}ms`"
                                    st.markdown(f"🤖 **Agent:** {text}{timing}")
                                else:
                                    st.markdown(f"**{role}:** {text}")
                                    
                        with out_tab2:
                            eval_data = res_dict.get("evaluation")
                            if eval_data:
                                st.write(f"**LLM Score:** {eval_data.get('llm_score', 'N/A')}")
                                st.write(f"**Summary:** {eval_data.get('llm_summary', 'N/A')}")
                                
                                st.write("**Rule Results:**")
                                for rule in eval_data.get("rule_results", []):
                                    mark = "✅" if rule.get("passed") else "❌"
                                    st.write(f"- {mark} **{rule.get('id')}**: {rule.get('detail')}")
                            else:
                                st.write("No evaluation data available.")
                                
                    except Exception as e:
                        st.error(f"Error running scenario: {e}")

with tab2:
    st.header("Dynamic Test Suite")
    from inbound_tester.dynamic_scenarios import ScenarioGenerator
    
    data_dir = Path("data")
    scenarios_dir = Path("scenarios")
    
    try:
        generator = ScenarioGenerator(data_dir, scenarios_dir)
        import math
        total_combos = math.prod(len(v) for v in generator.dimensions.values() if v)
        st.info(f"Total possible theoretical scenarios: {total_combos}")
    except Exception as e:
        generator = None
        st.error(f"Could not load generator: {e}")
        
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Suite Configuration")
        gen_mode = st.radio("Generation Mode", ["Regression Suite", "Random Sampling", "Coverage Driven", "Stress Testing"])
        
        c1, c2 = st.columns(2)
        with c1:
            start_idx = st.number_input("Start Index", min_value=1, max_value=10000, value=1)
        with c2:
            end_idx = st.number_input("End Index", min_value=1, max_value=10000, value=50)
        
        num_scenarios = max(1, end_idx)
        
        mode = st.radio("Execution Mode", ["text", "audio"], index=0, key="dyn_mode")
        agent_id = st.text_input("Agent ID (Override)", value="", key="dyn_agent")
        skip_eval = st.checkbox("Skip Evaluation", value=False, key="dyn_eval")
        
        gen_btn = st.button("Generate & Run Suite", type="primary", use_container_width=True)
        
        import os
        if os.path.exists("test_reports.csv"):
            with open("test_reports.csv", "rb") as f:
                st.download_button("Download Test Reports CSV", f, file_name="test_reports.csv", mime="text/csv", use_container_width=True)
        
    with col2:
        st.subheader("Execution View")
        if gen_btn and generator:
            try:
                st.info("Generating scenarios...")
                
                if gen_mode == "Regression Suite":
                    dyn_scenarios = generator.generate_regression_suite(num_scenarios)
                elif gen_mode == "Random Sampling":
                    dyn_scenarios = generator.generate_random_sampling(num_scenarios)
                elif gen_mode == "Coverage Driven":
                    dyn_scenarios = generator.generate_coverage_driven(num_scenarios)
                else:
                    dyn_scenarios = generator.generate_stress_testing(num_scenarios)
                
                # Apply range filter
                if start_idx <= len(dyn_scenarios):
                    dyn_scenarios = dyn_scenarios[start_idx - 1 : end_idx]
                else:
                    dyn_scenarios = []
                
                # Show metadata summary
                import pandas as pd
                meta_df = pd.DataFrame([s.metadata for s in dyn_scenarios])
                if not meta_df.empty:
                    st.dataframe(meta_df, use_container_width=True)
                
                st.info(f"Generated {len(dyn_scenarios)} scenarios in range {start_idx}-{end_idx}. Starting execution...")
                
                # Setup CSV for live reporting
                import csv
                csv_file_path = "test_reports.csv"
                csv_headers = ["Scenario ID", "Status", "Duration (s)", "Turns", "LLM Score", "Avg Tester Latency (ms)", "Avg Outbound Latency (ms)", "Avg Latency (ms)"]
                
                if not os.path.exists(csv_file_path):
                    with open(csv_file_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(csv_headers)
                
                # Setup settings
                test_settings = get_settings()
                if agent_id:
                    test_settings.elevenlabs_agent_id = agent_id
                test_settings.conversation_mode = mode
                
                if test_settings.conversation_mode == "audio":
                    runner = VoiceRunner(test_settings)
                else:
                    runner = ConversationRunner(test_settings)
                    
                from inbound_tester.audit_system import AuditSystem
                audit = AuditSystem(test_settings.sqlite_path)
                    
                for idx, s in enumerate(dyn_scenarios):
                    st.write(f"### Running Scenario {idx+1}/{len(dyn_scenarios)}: {s.id}")
                    
                    # Deduplication check
                    combo = {k: {"name": v} for k, v in s.metadata.get("selected_dimensions", {}).items()}
                    dup_check = audit.check_duplicate(combo)
                    
                    if dup_check["recommendation"] == "skip":
                        st.warning(f"Skipped duplicate scenario {s.id}. Hash: {dup_check['scenario_hash'][:8]}...")
                        # Register and log as skipped
                        reg = audit.register_scenario(
                            scenario_name=s.name,
                            dimension_combo=combo,
                            difficulty=s.metadata.get("difficulty", "medium"),
                            created_by="dynamic_suite_ui"
                        )
                        audit.log_test_execution(
                            scenario_id=reg["scenario_id"],
                            status="skipped_duplicate",
                            was_deduplicated=True,
                            dedup_record_id=reg.get("dedup_record_id"),
                            triggered_by="ui",
                            execution_mode=gen_mode
                        )
                        continue
                        
                    # Register new scenario
                    reg = audit.register_scenario(
                        scenario_name=s.name,
                        dimension_combo=combo,
                        difficulty=s.metadata.get("difficulty", "medium"),
                        created_by="dynamic_suite_ui"
                    )

                    result = asyncio.run(runner.run_scenario(s, skip_evaluation=skip_eval))
                    res_dict = result.to_dict()
                    
                    passed_eval = res_dict.get("evaluation", {}).get("passed")
                    if passed_eval is True:
                        st.success(f"{s.id} Complete! Result: PASS ✅")
                        status_val = "passed"
                    elif passed_eval is False:
                        st.error(f"{s.id} Complete! Result: FAIL ❌")
                        status_val = "failed"
                    else:
                        st.info(f"{s.id} Complete! Status: {res_dict.get('status')}")
                        status_val = res_dict.get('status')
                        
                    # Log to CSV
                    with open(csv_file_path, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        eval_data = res_dict.get("evaluation", {})
                        writer.writerow([
                            s.id,
                            status_val,
                            res_dict.get("duration_sec", ""),
                            res_dict.get("turn_count", 0),
                            eval_data.get('llm_score', 'N/A') if isinstance(eval_data, dict) else 'N/A',
                            res_dict.get("avg_tester_latency_ms", ""),
                            res_dict.get("avg_outbound_latency_ms", ""),
                            res_dict.get("avg_latency_ms", "")
                        ])
                        
                    # Log execution
                    audit.log_test_execution(
                        scenario_id=reg["scenario_id"],
                        status=status_val,
                        test_run_id=res_dict.get("test_run_id"),
                        conversation_id=res_dict.get("conversation_id"),
                        duration_sec=res_dict.get("duration_sec"),
                        turn_count=res_dict.get("turn_count", 0),
                        avg_latency_ms=res_dict.get("avg_latency_ms"),
                        avg_tester_latency_ms=res_dict.get("avg_tester_latency_ms"),
                        avg_outbound_latency_ms=res_dict.get("avg_outbound_latency_ms"),
                        transcript=res_dict.get("transcript"),
                        evaluation=res_dict.get("evaluation"),
                        triggered_by="ui",
                        execution_mode=gen_mode
                    )

            except Exception as e:
                st.error(f"Error during dynamic suite execution: {e}")

with tab3:
    st.header("Past Runs")
    db_path = get_settings().sqlite_path
    
    if not db_path:
        st.warning("SQLite Database not configured.")
    else:
        logger = ConversationLogger(db_path)
        try:
            runs = logger.list_runs(limit=100)
            
            if not runs:
                st.info("No past runs found in the database.")
            else:
                df_data = []
                for r in runs:
                    eval_data = r.get("evaluation") or {}
                    passed = eval_data.get("passed")
                    status_icon = "✅ PASS" if passed else ("❌ FAIL" if passed is False else "➖ N/A")
                    
                    df_data.append({
                        "Run ID": r["id"],
                        "Scenario": r["scenario_id"],
                        "Passed": status_icon,
                        "Status": r["status"],
                        "Date (UTC)": r["started_at"],
                        "Turns": r["turn_count"],
                        "🧑 Tester (ms)": round(r.get("avg_tester_latency_ms") or 0) if r.get("avg_tester_latency_ms") else "N/A",
                        "🤖 Outbound (ms)": round(r.get("avg_outbound_latency_ms") or 0) if r.get("avg_outbound_latency_ms") else "N/A",
                        "⏱️ Total (ms)": round(r.get("avg_latency_ms") or 0) if r.get("avg_latency_ms") else "N/A",
                    })
                    
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                st.subheader("Transcript Inspector")
                t_selected_run_id = st.selectbox("Select a Run ID to view transcript", [r["id"] for r in runs], key="tab3_run_sel")
                if t_selected_run_id:
                    run_details = logger.get_run(t_selected_run_id)
                    if run_details:
                        transcript = run_details.get("transcript", [])
                        if transcript:
                            st.markdown(f"### Transcript for `{t_selected_run_id}`")
                            for msg in transcript:
                                role = msg.get('role', '')
                                text = msg.get('text', '')
                                timing = ""
                                if role.lower() == 'user':
                                    t_ms = msg.get('tester_response_ms')
                                    if t_ms is not None:
                                        llm = msg.get('llm_ms')
                                        tts = msg.get('tts_ms')
                                        if llm is not None and tts is not None and tts > 0:
                                            timing = f" `[{t_ms:.0f}ms] (TTFT: {llm:.0f}ms, TTFA: {tts:.0f}ms)`"
                                        else:
                                            timing = f" `[{t_ms:.0f}ms]`"
                                    st.markdown(f"?? **Persona:** {text}{timing}")
                                elif role.lower() == 'agent':
                                    o_ms = msg.get('outbound_response_ms')
                                    if o_ms is not None:
                                        timing = f" `[{o_ms:.0f}ms]`"
                                    st.markdown(f"?? **Agent:** {text}{timing}")
                                else:
                                    st.markdown(f"**{role.title()}:** {text}")
                        else:
                            st.write("No transcript available.")

                
        except Exception as e:
            st.error(f"Error loading past runs: {e}")

with tab4:
    st.header("? Evaluation Dashboard")
    st.markdown("A comprehensive, manager-ready view of the outbound agent's performance.")
    
    db_path = get_settings().sqlite_path
    if not db_path:
        st.warning("SQLite Database not configured.")
    else:
        logger = ConversationLogger(db_path)
        try:
            runs = logger.list_runs(limit=100)
            if not runs:
                st.info("No past runs found in the database.")
            else:
                run_options = {r["id"]: f"{r['scenario_id']} ({r['started_at']}) - {r['id']}" for r in runs}
                selected_run_id = st.selectbox(
                    "Select a Run to Evaluate:",
                    options=list(run_options.keys()),
                    format_func=lambda x: run_options[x]
                )
                
                if selected_run_id:
                    run_details = logger.get_run(selected_run_id)
                    if run_details:
                        eval_data = run_details.get("evaluation", {})
                        is_pass = eval_data.get("passed")
                        
                        # --- 1. Top Level Status ---
                        status_color = "green" if is_pass else "red"
                        status_text = "? PASS" if is_pass else "? FAIL"
                        
                        st.markdown(f"""
                        <div style="padding: 20px; border-radius: 10px; background-color: rgba(255,255,255,0.05); text-align: center; border: 1px solid {status_color}; margin-bottom: 30px;">
                            <h2 style="margin:0; color: {status_color};">{status_text}</h2>
                            <p style="margin:5px 0 0 0; opacity:0.8;">Run ID: <code>{selected_run_id}</code></p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # --- 2. High Level Scores ---
                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            with st.container(border=True):
                                st.metric("LLM Score", f"{eval_data.get('llm_score', 'N/A')}")
                        with c2:
                            with st.container(border=True):
                                st.metric("Audio Score", f"{eval_data.get('audio_score', 'N/A')}")
                        with c3:
                            with st.container(border=True):
                                st.metric("Turns", run_details.get('turn_count', '0'))
                        with c4:
                            with st.container(border=True):
                                dur = run_details.get('duration_sec')
                                st.metric("Duration", f"{dur:.1f}s" if dur else "N/A")
                                
                        st.divider()
                        
                        # --- 3. Parameter Boxes (Ticks and Crosses) ---
                        st.markdown("### ?? Semantic Evaluation Parameters")
                        llm_details = eval_data.get('llm_details', {})
                        s_params = eval_data.get('semantic_scores', {})
                        if s_params:
                            cols = st.columns(4)
                            for i, (k, v) in enumerate(s_params.items()):
                                passed = float(v) >= 0.6
                                icon = "?" if passed else "?"
                                with cols[i % 4]:
                                    with st.container(border=True):
                                        st.metric(label=k.replace('_', ' ').title(), value=f"{v} {icon}")
                        else:
                            st.info("No semantic parameters available.")
                            
                        st.markdown("### ??? Acoustic Evaluation Parameters")
                        audio_details = eval_data.get('audio_details', {})
                        a_params = eval_data.get('acoustic_scores', {})
                        if a_params:
                            cols = st.columns(4)
                            for i, (k, v) in enumerate(a_params.items()):
                                passed = float(v) >= 0.6
                                icon = "?" if passed else "?"
                                with cols[i % 4]:
                                    with st.container(border=True):
                                        st.metric(label=k.replace('_', ' ').title(), value=f"{v} {icon}")
                        else:
                            st.info("No acoustic parameters available.")
                            
                        st.divider()
                        
                        # --- 4. Strengths & Issues ---
                        sc1, sc2 = st.columns(2)
                        with sc1:
                            st.markdown("#### ? Top Strengths")
                            s_list = llm_details.get('strengths', []) + audio_details.get('strengths', [])
                            for s in s_list:
                                st.success(s)
                        with sc2:
                            st.markdown("#### ?? Key Issues")
                            i_list = llm_details.get('issues', []) + audio_details.get('issues', [])
                            for i in i_list:
                                st.error(i)
                                
                        st.divider()
                        
                        # --- 5. Rule Checks ---
                        st.markdown("### ?? Objective Rule Gates")
                        rules = eval_data.get("rule_results", [])
                        for rule in rules:
                            if rule.get("passed"):
                                st.info(f"? **{rule.get('id')}**: {rule.get('detail')}")
                            else:
                                st.error(f"? **{rule.get('id')}**: {rule.get('detail')}")

                        st.divider()

                        # --- 6. The Original Markdown Report ---
                        from inbound_tester.evaluator import generate_markdown_report
                        md = generate_markdown_report(run_details)
                        with st.expander("?? View Auto-Generated Markdown Report", expanded=False):
                            st.markdown(md)
                            
        except Exception as e:
             st.error(f"Error loading evaluations: {e}")
