"""Stub implementation of a paper ingestion and summarisation tool.

In a production setting this module would download PDFs, extract text using
libraries such as `pdfminer` or `PyPDF2`, and then summarise the content
using NLP techniques. Here we provide a simplified interface that accepts
a file path or URL and returns a dummy summary.
"""

from typing import Dict


class PaperTool:
    """A stub tool for working with scientific papers and documents."""

    def ingest(self, source: str) -> str:
        """Ingest a document from a local file or URL."""
        return f"[Text extracted from {source}]"

    def summarise(self, text: str) -> str:
        """Produce a summary of the given document text."""
        return f"Summary of the document: {text[:100]}..."
