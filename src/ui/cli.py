"""Rich Terminal CLI: Live multi-agent deep research console dashboard."""
import asyncio
import argparse
import sys
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.layout import Layout
from rich.live import Live
from rich import box

from src.graph.workflow import ResearchWorkflowRunner
from src.llm import get_llm
from src.config import settings

console = Console()

AGENT_COLORS = {
    "Planner": "bold magenta",
    "Researcher": "bold cyan",
    "Critic": "bold yellow",
    "Writer": "bold green",
    "System": "bold blue"
}


def print_banner():
    console.print(Panel.fit(
        "[bold cyan]🔍 Multi-Agent Deep Research & Synthesizer[/bold cyan]\n"
        "[italic white]Autonomous LangGraph Pipeline • Planner • Researchers • Critic • Writer[/italic white]\n"
        f"[dim]Provider: [bold yellow]{settings.resolve_provider()}[/bold yellow] | Model: [bold yellow]{settings.DEFAULT_MODEL_NAME}[/bold yellow][/dim]",
        border_style="cyan",
        box=box.DOUBLE
    ))


def format_agent_name(name: str) -> str:
    for prefix, color in AGENT_COLORS.items():
        if prefix in name:
            return f"[{color}]{name}[/{color}]"
    return f"[white]{name}[/white]"


async def run_cli(query: str, clarifications: str = None, output_file: str = None):
    print_banner()

    events_table = Table(title="[bold]Autonomous Multi-Agent Activity Stream[/bold]", box=box.ROUNDED, expand=True)
    events_table.add_column("Time", justify="center", style="dim", width=10)
    events_table.add_column("Agent", justify="left", width=18)
    events_table.add_column("Action / Event", justify="left", style="white")

    def on_event(event: dict):
        time_str = datetime.fromisoformat(event["timestamp"]).strftime("%H:%M:%S")
        agent_formatted = format_agent_name(event.get("agent", "System"))
        msg = event.get("message", "")
        events_table.add_row(time_str, agent_formatted, msg)

    runner = ResearchWorkflowRunner(event_callback=on_event)

    with Progress(
        SpinnerColumn("dots", style="cyan"),
        TextColumn("[bold cyan]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=False
    ) as progress:
        task = progress.add_task("[bold cyan]Executing Multi-Agent Research...", total=None)

        result = await runner.run(query=query, clarifications=clarifications)
        progress.update(task, completed=True, description="[bold green]✓ Research & Synthesis Complete!")

    # Display activity stream table
    console.print("\n")
    console.print(events_table)
    console.print("\n")

    # Display research plan summary
    plan = result.get("plan")
    if plan:
        plan_table = Table(title="[bold green]📋 Structured Research Plan[/bold green]", box=box.ROUNDED, expand=True)
        plan_table.add_column("ID", style="cyan", width=12)
        plan_table.add_column("Subtask Title", style="bold white")
        plan_table.add_column("Target Queries", style="dim")
        for st in plan.subtasks:
            plan_table.add_row(st.id, st.title, ", ".join(st.target_search_queries[:2]))
        console.print(plan_table)
        console.print("\n")

    # Display Critic Evaluation
    critic_reviews = result.get("critic_reviews", [])
    if critic_reviews:
        latest = critic_reviews[-1]
        verdict_color = "green" if latest.verdict == "APPROVE" else "yellow"
        console.print(Panel(
            f"[bold {verdict_color}]Verdict: {latest.verdict}[/bold {verdict_color}] | "
            f"Completeness Score: [bold cyan]{latest.completeness_score}%[/bold cyan] | "
            f"Hallucination Risk: [bold]{latest.hallucination_risk.upper()}[/bold]\n\n"
            f"[italic]{latest.reasoning}[/italic]",
            title="[bold yellow]🧐 Fact-Checking & Critic Review[/bold yellow]",
            border_style="yellow"
        ))
        console.print("\n")

    # Display Final Report in Markdown
    report = result.get("final_report")
    if report:
        console.print(Panel(
            Markdown(report.assemble_markdown()),
            title=f"[bold green]📄 Generated Report: {report.title}[/bold green]",
            border_style="green",
            box=box.HEAVY
        ))

        # Save to file
        default_filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        save_path = output_file or default_filename
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report.full_markdown)
        console.print(f"\n[bold green]✓ Report exported successfully to:[/bold green] [underline cyan]{save_path}[/underline cyan]\n")


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Deep Research & Synthesizer CLI")
    parser.add_argument("--query", "-q", type=str, help="Research topic or question to investigate")
    parser.add_argument("--clarifications", "-c", type=str, default=None, help="Additional context or constraints")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output markdown file path")
    parser.add_argument("--provider", type=str, default=None, help="LLM Provider: openai, anthropic, google, mock")
    parser.add_argument("--model", type=str, default=None, help="Model name (e.g. gpt-4o, claude-3-5-sonnet)")

    args = parser.parse_args()

    if args.provider:
        settings.DEFAULT_LLM_PROVIDER = args.provider
    if args.model:
        settings.DEFAULT_MODEL_NAME = args.model

    query = args.query
    if not query:
        print_banner()
        query = console.input("[bold cyan]Enter research query or topic: [/bold cyan]")
        if not query.strip():
            console.print("[red]Query cannot be empty. Exiting.[/red]")
            sys.exit(1)

    asyncio.run(run_cli(query=query, clarifications=args.clarifications, output_file=args.output))


if __name__ == "__main__":
    main()
