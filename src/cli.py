"""
Command-line interface for the NPDE TA RAG system.

Provides commands for indexing documents and querying the knowledge base.
"""

import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from typing import Optional, Annotated
from pathlib import Path

from . import __version__

# Initialize typer app and rich console
app = typer.Typer(
    name="npde-rag",
    help="NPDE TA RAG System - A retrieval-augmented assistant for Numerical Methods for PDEs",
    add_completion=False,
)
console = Console()


def print_header():
    """Print the application header."""
    console.print(Panel.fit(
        "[bold blue]NPDE TA RAG System[/bold blue]\n"
        "[dim]Retrieval-Augmented Generation for Numerical Methods for PDEs[/dim]",
        border_style="blue",
    ))


@app.command()
def index(
    force: Annotated[bool, typer.Option("--force", "-f", help="Force re-indexing even if documents exist")] = False,
):
    """
    Index all course documents into the vector store.
    
    This command parses PDFs from data/lectures, data/homework, and data/exams,
    chunks them, generates embeddings, and stores them in ChromaDB.
    """
    print_header()
    console.print("\n[bold]Starting document indexing...[/bold]\n")
    
    from .pdf_parser import parse_all_documents
    from .chunker import chunk_documents
    from .vectorstore import get_vector_store, index_documents
    
    # Check existing index
    store = get_vector_store()
    existing_count = store.count
    
    if existing_count > 0 and not force:
        console.print(f"[yellow]Vector store already contains {existing_count} chunks.[/yellow]")
        console.print("Use --force to re-index.\n")
        
        stats = store.get_stats()
        _print_stats(stats)
        return
    
    if force and existing_count > 0:
        console.print(f"[yellow]Clearing existing {existing_count} chunks...[/yellow]")
        store.delete_all()
    
    # Parse documents
    console.print("[bold]Step 1: Parsing PDF documents[/bold]")
    with console.status("[bold green]Parsing PDFs..."):
        documents = parse_all_documents(show_progress=True)
    console.print(f"[green]✓ Parsed {len(documents)} documents[/green]\n")
    
    # Chunk documents
    console.print("[bold]Step 2: Chunking documents[/bold]")
    with console.status("[bold green]Chunking..."):
        chunks = chunk_documents(documents)
    console.print(f"[green]✓ Created {len(chunks)} chunks[/green]\n")
    
    # Index chunks
    console.print("[bold]Step 3: Generating embeddings and indexing[/bold]")
    index_documents(chunks, show_progress=True)
    console.print(f"[green]✓ Indexed {len(chunks)} chunks[/green]\n")
    
    # Print stats
    stats = store.get_stats()
    _print_stats(stats)


def _print_stats(stats: dict):
    """Print vector store statistics."""
    table = Table(title="Vector Store Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Chunks", str(stats["total_chunks"]))
    table.add_row("Document Types", ", ".join(stats["doc_types"]))
    if stats["years"]:
        table.add_row("Exam Years", ", ".join(map(str, stats["years"])))
    
    console.print(table)


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Question to ask")],
    doc_type: Annotated[Optional[str], typer.Option("--doc-type", help="Filter by document type (lecture/homework/exam)")] = None,
    year: Annotated[Optional[int], typer.Option("--year", help="Filter by year (for exams)")] = None,
    top_k: Annotated[int, typer.Option("--top-k", help="Number of context chunks to retrieve")] = 5,
    hide_sources: Annotated[bool, typer.Option("--hide-sources", help="Hide source documents")] = False,
):
    """
    Ask a question about the course materials.
    
    The system retrieves relevant context and generates an answer using GPT-4o.
    """
    print_header()
    
    from .vectorstore import get_vector_store
    from .llm import ask as llm_ask
    
    # Check if index exists
    store = get_vector_store()
    if store.count == 0:
        console.print("[red]Error: No documents indexed. Run 'index' first.[/red]")
        raise typer.Exit(1)
    
    console.print(f"\n[bold]Question:[/bold] {question}\n")
    
    # Get answer
    with console.status("[bold green]Thinking..."):
        response, chunks = llm_ask(
            query=question,
            top_k=top_k,
            doc_type=doc_type,
            year=year,
        )
    
    # Display answer
    console.print(Panel(
        Markdown(response),
        title="[bold green]Answer[/bold green]",
        border_style="green",
    ))
    
    # Display sources
    if not hide_sources and chunks:
        console.print("\n[bold]Sources:[/bold]")
        for i, chunk in enumerate(chunks, 1):
            relevance = f"{chunk.relevance_score:.2%}"
            console.print(f"  {i}. {chunk.format_source()} [dim](relevance: {relevance})[/dim]")


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="Search query")],
    doc_type: Annotated[Optional[str], typer.Option("--doc-type", help="Filter by document type (lecture/homework/exam)")] = None,
    year: Annotated[Optional[int], typer.Option("--year", help="Filter by year (for exams)")] = None,
    top_k: Annotated[int, typer.Option("--top-k", help="Number of results to show")] = 5,
    full: Annotated[bool, typer.Option("--full", help="Show full chunk content")] = False,
):
    """
    Search for relevant content without generating an answer.
    
    Useful for finding similar problems or specific content.
    """
    print_header()
    
    from .retriever import retrieve
    from .vectorstore import get_vector_store
    
    # Check if index exists
    store = get_vector_store()
    if store.count == 0:
        console.print("[red]Error: No documents indexed. Run 'index' first.[/red]")
        raise typer.Exit(1)
    
    console.print(f"\n[bold]Search:[/bold] {query}\n")
    
    # Search
    with console.status("[bold green]Searching..."):
        chunks = retrieve(
            query=query,
            top_k=top_k,
            doc_type=doc_type,
            year=year,
        )
    
    if not chunks:
        console.print("[yellow]No results found.[/yellow]")
        return
    
    # Display results
    for i, chunk in enumerate(chunks, 1):
        relevance = f"{chunk.relevance_score:.2%}"
        
        # Build metadata string
        meta_parts = [chunk.doc_type]
        if "year" in chunk.metadata:
            meta_parts.append(f"Year: {chunk.metadata['year']}")
        if "exam_type" in chunk.metadata:
            meta_parts.append(chunk.metadata["exam_type"])
        
        title = f"[bold]{i}. {chunk.format_source()}[/bold] [dim]({relevance})[/dim]"
        subtitle = " | ".join(meta_parts)
        
        if full:
            content = chunk.text
        else:
            # Show preview (first 300 chars)
            content = chunk.text[:300]
            if len(chunk.text) > 300:
                content += "..."
        
        console.print(Panel(
            content,
            title=title,
            subtitle=f"[dim]{subtitle}[/dim]",
            border_style="blue",
        ))
        console.print()


@app.command()
def chat():
    """
    Start an interactive chat session.
    
    Allows multi-turn conversation with context from course materials.
    """
    print_header()
    
    from .vectorstore import get_vector_store
    from .llm import get_llm_client
    from .retriever import get_retriever
    
    # Check if index exists
    store = get_vector_store()
    if store.count == 0:
        console.print("[red]Error: No documents indexed. Run 'index' first.[/red]")
        raise typer.Exit(1)
    
    console.print("\n[bold green]Interactive Chat Mode[/bold green]")
    console.print("[dim]Type 'quit' or 'exit' to end the session.[/dim]")
    console.print("[dim]Type 'clear' to start a new conversation.[/dim]")
    console.print("[dim]Commands: /type <lecture|homework|exam>, /year <YYYY>[/dim]\n")
    
    llm = get_llm_client()
    retriever = get_retriever()
    
    messages = []
    doc_type_filter = None
    year_filter = None
    
    while True:
        try:
            user_input = Prompt.ask("[bold blue]You[/bold blue]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break
        
        if not user_input.strip():
            continue
        
        # Handle commands
        if user_input.lower() in ("quit", "exit"):
            console.print("[dim]Goodbye![/dim]")
            break
        
        if user_input.lower() == "clear":
            messages = []
            console.print("[yellow]Conversation cleared.[/yellow]\n")
            continue
        
        if user_input.startswith("/type "):
            doc_type_filter = user_input[6:].strip() or None
            console.print(f"[yellow]Filter set: doc_type = {doc_type_filter}[/yellow]\n")
            continue
        
        if user_input.startswith("/year "):
            try:
                year_filter = int(user_input[6:].strip())
                console.print(f"[yellow]Filter set: year = {year_filter}[/yellow]\n")
            except ValueError:
                year_filter = None
                console.print(f"[yellow]Filter cleared: year[/yellow]\n")
            continue
        
        # Retrieve context for the current message
        chunks = retriever.retrieve(
            query=user_input,
            top_k=3,
            doc_type=doc_type_filter,
            year=year_filter,
        )
        context = retriever.format_context(chunks)
        
        # Add user message
        messages.append({"role": "user", "content": user_input})
        
        # Generate response
        with console.status("[bold green]Thinking..."):
            response = llm.chat(messages, context=context)
        
        # Add assistant message
        messages.append({"role": "assistant", "content": response})
        
        # Display response
        console.print()
        console.print(Panel(
            Markdown(response),
            title="[bold green]Assistant[/bold green]",
            border_style="green",
        ))
        
        # Show sources briefly
        if chunks:
            sources = ", ".join(c.source_name for c in chunks[:3])
            console.print(f"[dim]Sources: {sources}[/dim]\n")


@app.command()
def stats():
    """
    Show statistics about the indexed documents.
    """
    print_header()
    
    from .vectorstore import get_vector_store
    
    store = get_vector_store()
    
    if store.count == 0:
        console.print("[yellow]No documents indexed yet. Run 'index' first.[/yellow]")
        return
    
    stats = store.get_stats()
    _print_stats(stats)


@app.command()
def version():
    """Show version information."""
    console.print(f"NPDE TA RAG System v{__version__}")


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
