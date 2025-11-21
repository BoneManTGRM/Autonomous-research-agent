"""Stub implementation of a local file reading tool.

This module provides minimal functionality to read files from the local
filesystem and produce summaries or extract key points. It can be extended
to support various file types (CSV, JSON, etc.).
"""

import os
from typing import Dict, List


class FileTool:
    """A simple tool for reading and summarising local files."""

    def read_file(self, filepath: str) -> str:
        """Read a text file from disk."""
        if not os.path.exists(filepath):
            return f"File {filepath} does not exist."
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def summarise(self, content: str) -> str:
        """Produce a naive summary of file content."""
        return content[:100] + "..." if len(content) > 100 else content
