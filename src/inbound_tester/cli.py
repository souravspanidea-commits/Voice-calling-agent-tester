"""CLI for running inbound test agent scenarios."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

import click

from inbound_tester.config import Settings, get_settings
from inbound_tester.conversation_runner import ConversationRunner
from inbound_tester.voice_runner import VoiceRunner
from inbound_tester.elevenlabs_ws import ElevenLabsConvAIClient
from inbound_tester.logger import ConversationLogger
from inbound_tester.scenarios import load_scenario, load_scenarios


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging")
def cli(verbose: bool) -> None:
    _setup_logging(verbose)


@cli.command("spike")
@click.option("--message", default="Hi", help="First user message to send")
@click.option("--timeout", default=45.0, help="Seconds to wait for agent response")
@click.option("--agent-id", default=None, help="Override ELEVENLABS_AGENT_ID from .env")
@click.option("--mode", type=click.Choice(["text", "audio"]), default=None, help="Conversation mode")
def spike(message: str, timeout: float, agent_id: str | None, mode: str | None) -> None:
    """Phase 1 spike: connect, send one message, log raw events."""
    asyncio.run(_run_spike(message, timeout, agent_id=agent_id, mode=mode))


async def _run_spike(message: str, timeout: float, *, agent_id: str | None = None, mode: str | None = None) -> None:
    settings = get_settings()
    if mode:
        settings.conversation_mode = mode
    if agent_id:
        settings.elevenlabs_agent_id = agent_id
    if not settings.elevenlabs_api_key or not settings.elevenlabs_agent_id:
        click.echo("Error: Set ELEVENLABS_API_KEY and ELEVENLABS_AGENT_ID in .env (or use --agent-id)", err=True)
        sys.exit(1)

    client = ElevenLabsConvAIClient(settings.ws_url, settings.elevenlabs_api_key)

    try:
        conversation_id = await client.connect()
        click.echo(f"Connected (Mode: {settings.conversation_mode}) — conversation_id={conversation_id}")

        if settings.conversation_mode == "audio":
            from inbound_tester.tts_client import ElevenLabsTTSClient

            # Detect the agent's expected input audio format
            user_input_fmt = "ulaw_8000"
            for ev in client.session.raw_events:
                meta = ev.get("conversation_initiation_metadata_event", {})
                if meta.get("user_input_audio_format"):
                    user_input_fmt = meta["user_input_audio_format"]
                    break
            click.echo(f"Agent expects input format: {user_input_fmt}")

            tts = ElevenLabsTTSClient(
                api_key=settings.elevenlabs_api_key,
                voice_id=settings.persona_voice_id,
                model_id=settings.persona_tts_model,
                base_url=settings.elevenlabs_tts_base,
                output_format=user_input_fmt,
            )
            try:
                click.echo(f"Synthesizing audio for: {message!r}")
                b64_chunks = await tts.synthesize_base64_chunks(message)
                click.echo(f"TTS returned {len(b64_chunks)} chunks ({user_input_fmt})")
                for chunk in b64_chunks:
                    await client.send_audio_chunk(chunk)
                # Send silence to trigger end-of-utterance in the agent's VAD
                from inbound_tester.audio_bridge import AudioBridge
                bridge = AudioBridge(client)
                await bridge.send_silence(duration_ms=1500, audio_format=user_input_fmt)
                bridge.close()
                click.echo("Sent user_audio_chunk + silence")
            finally:
                await tts.close()
        else:
            await client.send_user_message(message)
            click.echo(f"Sent user_message: {message!r}")

        agent_turn = await client.wait_for_agent_turn(timeout)
        if agent_turn:
            print(f"Agent response: {agent_turn.text}")
        else:
            click.echo("No agent response within timeout", err=True)

        print("\n--- Raw events ---")
        for event in client.session.raw_events:
            print(json.dumps(event, indent=2, ensure_ascii=False))

    except Exception as exc:
        click.echo(f"Spike failed: {exc}", err=True)
        sys.exit(1)
    finally:
        await client.close()


@cli.command("run")
@click.argument("scenario", type=click.Path(exists=True, path_type=Path))
@click.option("--no-eval", is_flag=True, help="Skip post-conversation evaluation")
@click.option("--agent-id", default=None, help="Override ELEVENLABS_AGENT_ID from .env")
@click.option("--mode", type=click.Choice(["text", "audio"]), default=None, help="Conversation mode")
def run_one(scenario: Path, no_eval: bool, agent_id: str | None, mode: str | None) -> None:
    """Run a single scenario YAML file against the outbound agent."""
    cfg = load_scenario(scenario)
    settings = get_settings()
    if agent_id:
        settings.elevenlabs_agent_id = agent_id
    if mode:
        settings.conversation_mode = mode
        
    if settings.conversation_mode == "audio":
        runner = VoiceRunner(settings)
    else:
        runner = ConversationRunner(settings)
        
    result = asyncio.run(runner.run_scenario(cfg, skip_evaluation=no_eval))
    _save_single_call_report(result.to_dict())
    _print_result(result.to_dict())
    
    _save_markdown_report([result.to_dict()])


@cli.command("run-all")
@click.option(
    "--scenarios-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("scenarios"),
    help="Directory containing scenario YAML files",
)
@click.option("--parallel", is_flag=True, help="Run scenarios in parallel")
@click.option("--no-eval", is_flag=True, help="Skip post-conversation evaluation")
@click.option("--agent-id", default=None, help="Override ELEVENLABS_AGENT_ID from .env")
@click.option("--mode", type=click.Choice(["text", "audio"]), default=None, help="Conversation mode")
def run_all(scenarios_dir: Path, parallel: bool, no_eval: bool, agent_id: str | None, mode: str | None) -> None:
    """Run all scenarios in a directory."""
    scenarios = load_scenarios(scenarios_dir)
    if not scenarios:
        click.echo(f"No scenarios found in {scenarios_dir}", err=True)
        sys.exit(1)

    settings = get_settings()
    if agent_id:
        settings.elevenlabs_agent_id = agent_id
    if mode:
        settings.conversation_mode = mode
        
    if settings.conversation_mode == "audio":
        runner = VoiceRunner(settings)
    else:
        runner = ConversationRunner(settings)

    async def _run_all() -> list[dict]:
        if parallel:
            tasks = [runner.run_scenario(s, skip_evaluation=no_eval) for s in scenarios]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            out = []
            for scenario, result in zip(scenarios, results):
                if isinstance(result, Exception):
                    out.append({"scenario_id": scenario.id, "status": "error", "error": str(result)})
                else:
                    res_dict = result.to_dict()
                    _save_single_call_report(res_dict)
                    out.append(res_dict)
            return out

        out = []
        for scenario in scenarios:
            try:
                result = await runner.run_scenario(scenario, skip_evaluation=no_eval)
                res_dict = result.to_dict()
                _save_single_call_report(res_dict)
                out.append(res_dict)
            except Exception as exc:
                out.append({"scenario_id": scenario.id, "status": "error", "error": str(exc)})
        return out

    results = asyncio.run(_run_all())
    passed = sum(1 for r in results if r.get("evaluation", {}) and r["evaluation"].get("passed"))
    click.echo(f"\n=== Summary: {passed}/{len(results)} passed ===")
    for result in results:
        _print_result(result, compact=True)

    if settings.sqlite_path:
        from inbound_tester.audit_system import AuditSystem
        audit = AuditSystem(settings.sqlite_path)
        coverage = audit.get_coverage_report()
        click.echo("\n=== Persona Scenario Coverage Matrix ===")
        click.echo(f"Overall Coverage: {coverage['overall_coverage_pct']}%")
        for dim, data in coverage.get("dimensions", {}).items():
            if data.get("total_values", 0) > 0:
                click.echo(f"  {dim}: {data['coverage_pct']}% ({data['covered_values']}/{data['total_values']})")

    _save_markdown_report(results, coverage=coverage if settings.sqlite_path else None)


@cli.command("list-runs")
@click.option("--limit", default=20, help="Max runs to show")
def list_runs(limit: int) -> None:
    """List recent test runs from the database."""
    settings = get_settings()
    db_path = settings.sqlite_path
    if not db_path:
        click.echo("list-runs only supports SQLite DATABASE_URL", err=True)
        sys.exit(1)

    db_logger = ConversationLogger(db_path)
    runs = db_logger.list_runs(limit=limit)
    for run in runs:
        eval_data = run.get("evaluation") or {}
        passed = eval_data.get("passed")
        status_icon = "PASS" if passed else ("FAIL" if passed is False else "-")
        latency = run.get("avg_latency_ms")
        lat_str = f"  lat={latency:.0f}ms" if latency else ""
        click.echo(
            f"{status_icon}  {run['id'][:8]}  {run['scenario_id']:20}  "
            f"{run['status']:22}  turns={run.get('turn_count', '?')}{lat_str}"
        )


@cli.command("show-run")
@click.argument("run_id")
def show_run(run_id: str) -> None:
    """Show details for a test run (partial ID match supported)."""
    settings = get_settings()
    db_path = settings.sqlite_path
    if not db_path:
        click.echo("show-run only supports SQLite DATABASE_URL", err=True)
        sys.exit(1)

    db_logger = ConversationLogger(db_path)
    full = db_logger.get_run_by_prefix(run_id)
    if not full:
        click.echo(f"No run matching {run_id!r}", err=True)
        sys.exit(1)

    click.echo(json.dumps(full, indent=2, default=str))

def _save_markdown_report(results: list[dict], coverage: dict | None = None) -> None:
    from datetime import datetime
    
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = reports_dir / f"test_report_{ts}.md"
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("# Inbound Tester Evaluation Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        passed = sum(1 for r in results if r.get("evaluation", {}) and r.get("evaluation", {}).get("passed"))
        f.write(f"## Summary\n")
        f.write(f"- **Total Scenarios Run:** {len(results)}\n")
        f.write(f"- **Passed:** {passed}\n")
        f.write(f"- **Failed/Error:** {len(results) - passed}\n\n")
        
        if coverage:
            f.write("## Persona Scenario Coverage\n")
            f.write(f"**Overall Coverage:** {coverage.get('overall_coverage_pct', 0)}%\n\n")
            for dim, data in coverage.get("dimensions", {}).items():
                if data.get("total_values", 0) > 0:
                    f.write(f"- **{dim}**: {data.get('coverage_pct', 0)}% ({data.get('covered_values', 0)}/{data.get('total_values', 0)})\n")
            f.write("\n")
            
        f.write("## Detailed Results\n\n")
        for res in results:
            eval_data = res.get("evaluation") or {}
            is_pass = eval_data.get("passed")
            label = "PASS" if is_pass else ("FAIL" if is_pass is False else "N/A")
            
            f.write(f"### [{label}] {res.get('scenario_id', 'Unknown')}\n")
            f.write(f"- **Status:** {res.get('status')}\n")
            f.write(f"- **Turns:** {res.get('turn_count')}\n")
            if res.get("duration_sec"):
                f.write(f"- **Duration:** {res['duration_sec']:.1f}s\n")
            if res.get("avg_latency_ms"):
                f.write(f"- **Avg Latency:** {res['avg_latency_ms']:.0f}ms\n")
                
            if eval_data:
                f.write("\n#### Evaluation\n")
                if eval_data.get("llm_score") is not None:
                    f.write(f"- **Semantic Score:** {eval_data['llm_score']}\n")
                if eval_data.get("llm_summary"):
                    f.write(f"- **Semantic Summary:** {eval_data['llm_summary']}\n")
                if eval_data.get("audio_score") is not None:
                    f.write(f"- **Audio Score:** {eval_data['audio_score']}\n")
                if eval_data.get("audio_summary"):
                    f.write(f"- **Audio Summary:** {eval_data['audio_summary']}\n")
                    
                rules = eval_data.get("rule_results", [])
                if rules:
                    f.write("\n**Rule Checks:**\n")
                    for rule in rules:
                        rmark = "✅" if rule.get("passed") else "❌"
                        f.write(f"- {rmark} **{rule.get('id')}**: {rule.get('detail')}\n")
                        
            f.write("\n---\n\n")
            
    click.echo(f"\n📝 Saved detailed markdown report to: {filepath}")

def _print_result(result: dict, compact: bool = False) -> None:
    evaluation = result.get("evaluation") or {}
    passed = evaluation.get("passed")
    label = "PASS" if passed else ("FAIL" if passed is False else "N/A")

    if compact:
        latency = result.get("avg_latency_ms")
        lat_str = f" lat={latency:.0f}ms" if latency else ""
        click.echo(f"[{label}] {result.get('scenario_id')} — {result.get('status')} — turns={result.get('turn_count')}{lat_str}")
        return

    click.echo(f"\n=== {result.get('scenario_id')} [{label}] ===")
    click.echo(f"Run ID: {result.get('test_run_id')}")
    click.echo(f"Status: {result.get('status')} | Turns: {result.get('turn_count')}")
    if result.get("duration_sec"):
        click.echo(f"Duration: {result['duration_sec']}s")
    if result.get("avg_latency_ms"):
        click.echo(f"Avg latency: {result['avg_latency_ms']}ms")

    click.echo("\nTranscript:")
    for entry in result.get("transcript") or []:
        text = entry.get('text', '')
        # Handle Windows CMD charmap errors by replacing unsupported characters
        safe_text = text.encode("cp1252", errors="replace").decode("cp1252")
        click.echo(f"  {entry.get('role')}: {safe_text}")

    if evaluation:
        click.echo(f"\nEvaluation passed: {evaluation.get('passed')}")
        if evaluation.get("llm_score") is not None:
            click.echo(f"LLM score: {evaluation.get('llm_score')}")
        if evaluation.get("llm_summary"):
            click.echo(f"Summary: {evaluation.get('llm_summary')}")
        for rule in evaluation.get("rule_results") or []:
            mark = "OK" if rule.get("passed") else "FAIL"
            click.echo(f"  [{mark}] {rule.get('id')}: {rule.get('detail')}")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
