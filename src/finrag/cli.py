"""Command-line interface: finrag ingest / search / ask."""

from __future__ import annotations

import sys

import anthropic
import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import config
from .chunking import build_nodes
from .indexing import build_index, load_index
from .models import ElementType
from .parsing import parse_pdf
from .retrieval import retrieve

console = Console()


def _handle_api_errors(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except anthropic.AuthenticationError:
        console.print("[red]Authentication failed.[/red] Set ANTHROPIC_API_KEY or run `ant auth login`.")
        sys.exit(2)
    except anthropic.RateLimitError as exc:
        retry_after = exc.response.headers.get("retry-after", "unknown")
        console.print(f"[red]Rate limited[/red] (retry-after: {retry_after}s).")
        sys.exit(2)
    except anthropic.APIStatusError as exc:
        console.print(f"[red]API error {exc.status_code}:[/red] {exc.message}")
        sys.exit(2)
    except anthropic.APIConnectionError:
        console.print("[red]Network error[/red] — could not reach the Claude API.")
        sys.exit(2)


@click.group()
def main() -> None:
    """Multi-modal RAG over financial reports."""


@main.command()
@click.argument("pdfs", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--no-vision", is_flag=True, help="Skip Claude vision chart descriptions (index text+tables only).")
@click.option("--visual", is_flag=True, help="Also build the ColPali page-image index (needs the [colpali] extra).")
def ingest(pdfs, no_vision, visual) -> None:
    """Parse and index one or more financial report PDFs."""
    all_elements = []
    for pdf in pdfs:
        console.print(f"Parsing [bold]{pdf}[/bold]…")
        elements = parse_pdf(pdf)
        counts = {t.value: sum(1 for e in elements if e.type is t) for t in ElementType}
        console.print(f"  {counts['text']} text blocks, {counts['table']} tables, {counts['chart']} figures")
        all_elements.extend(elements)

    charts = [e for e in all_elements if e.type is ElementType.CHART]
    if charts and not no_vision:
        from .charts import describe_all

        console.print(f"Describing {len(charts)} figure(s) with Claude vision…")
        _handle_api_errors(
            describe_all,
            all_elements,
            progress=lambda el: console.print(f"  p.{el.page}: {el.text.splitlines()[0]}"),
        )
    elif charts:
        console.print(f"[yellow]Skipping vision:[/yellow] {len(charts)} figure(s) will not be indexed.")
        all_elements = [e for e in all_elements if e.type is not ElementType.CHART]

    nodes = build_nodes(all_elements)
    console.print(f"Indexing [bold]{len(nodes)}[/bold] chunks (dense + sparse) into Qdrant…")
    build_index(nodes)
    console.print(f"[green]Done.[/green] Index stored in {config.STORAGE_DIR}/qdrant")

    if visual:
        from .colpali_store import index_pages
        from .indexing import _client

        for pdf in pdfs:
            n = index_pages(pdf, _client())
            console.print(f"ColPali: indexed {n} page image(s) from {pdf}")


@main.command()
@click.argument("query")
@click.option("--top-k", default=None, type=int, help=f"Results to return (default {config.TOP_K}).")
def search(query, top_k) -> None:
    """Hybrid search (dense + sparse) without answer synthesis."""
    chunks = retrieve(load_index(), query, top_k)
    table = Table(title=f"Hybrid results for: {query}")
    table.add_column("score", justify="right")
    table.add_column("type")
    table.add_column("where")
    table.add_column("text")
    for c in chunks:
        table.add_row(f"{c.score:.3f}", c.element_type, f"{c.source} p.{c.page}", c.text[:120].replace("\n", " "))
    console.print(table)


@main.command()
@click.argument("question")
@click.option("--top-k", default=None, type=int)
@click.option("--show-context", is_flag=True, help="Print the retrieved context blocks.")
def ask(question, top_k, show_context) -> None:
    """Ask a question; answers are grounded in retrieved context with citations."""
    from .answer import answer_question

    chunks = retrieve(load_index(), question, top_k)
    if not chunks:
        console.print("[yellow]Nothing retrieved — did you run `finrag ingest` first?[/yellow]")
        sys.exit(1)
    if show_context:
        for i, c in enumerate(chunks, 1):
            console.print(Panel(c.text[:600], title=f"[{i}] {c.element_type} · {c.source} p.{c.page}"))

    result, cited = _handle_api_errors(answer_question, question, chunks)
    style = "yellow" if result.insufficient_context else "green"
    console.print(Panel(result.answer, title="Answer", border_style=style))
    if cited:
        console.print("Sources: " + ", ".join(f"[{i}] {c.source} p.{c.page} ({c.element_type})"
                                              for i, c in zip(result.cited_blocks, cited)))


@main.command("search-visual")
@click.argument("query")
@click.option("--top-k", default=3, type=int)
def search_visual(query, top_k) -> None:
    """ColPali visual page search (needs the [colpali] extra and a --visual ingest)."""
    from .colpali_store import search_pages
    from .indexing import _client

    for hit in search_pages(query, _client(), top_k):
        console.print(f"{hit['score']:.3f}  {hit['source']} p.{hit['page']}")


if __name__ == "__main__":
    main()
