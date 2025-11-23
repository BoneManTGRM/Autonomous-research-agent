import io
import json
from typing import Any, Dict, Optional, Union

import pandas as pd

try:
    import PyPDF2  # type: ignore[import]
except Exception:  # pragma: no cover
    PyPDF2 = None


class DataConnectors:
    """
    Unified interface for loading external data files.
    Used by CoreAgent + Streamlit for ingestion.

    Backwards compatible:
        - load_pdf(pdf_bytes) -> {"ok": bool, "text" or "error": str}
        - load_table(file_bytes, filename) -> DataFrame OR dict with "error"

    Extended capabilities:
        - Multi-page PDF extraction with basic metadata
        - CSV / TSV / XLSX table loading
        - JSON structured loading (dict/list)
        - TXT / MD / HTML lightweight text loading
        - Parquet and Feather support when dependencies are present
        - Generic load_any entry point that returns type and content
    """

    # ------------------------------------------------------------------
    # PDF
    # ------------------------------------------------------------------
    @staticmethod
    def load_pdf(pdf_bytes: bytes) -> Dict[str, Any]:
        """
        Load a PDF into plain text.

        Returns:
            {
                "ok": bool,
                "text": str,          # merged text from all pages
                "pages": int,         # number of pages (if ok)
                "meta": {...},        # basic pdf metadata (if available)
                "error": str,         # on failure
            }

        Backwards compatibility:
            Existing callers expecting {"ok": bool, "text": "..."} still work.
        """
        if not PyPDF2:
            return {"ok": False, "error": "PyPDF2 not installed"}

        try:
            reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
            texts = []
            for page in reader.pages:
                try:
                    page_text = page.extract_text() or ""
                except Exception:
                    page_text = ""
                texts.append(page_text)

            merged = "\n\n".join(texts).strip()
            meta_raw = getattr(reader, "metadata", None) or {}

            # Convert raw metadata to simple dict of str -> str
            meta: Dict[str, Any] = {}
            if isinstance(meta_raw, dict):
                for k, v in meta_raw.items():
                    sk = str(k)
                    try:
                        sv = str(v)
                    except Exception:
                        sv = repr(v)
                    meta[sk] = sv

            return {
                "ok": True,
                "text": merged,
                "pages": len(reader.pages),
                "meta": meta,
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ------------------------------------------------------------------
    # Tabular / structured data
    # ------------------------------------------------------------------
    @staticmethod
    def _load_json(file_bytes: bytes) -> Union[Dict[str, Any], Any]:
        """
        Load JSON content from bytes.

        Returns dict/list or raises.
        """
        # Try text first, fall back to raw bytes decode
        try:
            text = file_bytes.decode("utf-8")
        except Exception:
            text = file_bytes.decode("utf-8", errors="ignore")
        return json.loads(text)

    @staticmethod
    def _load_parquet(file_bytes: bytes):
        """
        Load Parquet file into a pandas DataFrame.

        Uses pandas.read_parquet with in-memory buffer.
        May require pyarrow or fastparquet to be installed.
        """
        try:
            buf = io.BytesIO(file_bytes)
            return pd.read_parquet(buf)
        except Exception as e:
            return {"error": f"Parquet load failed: {e}"}

    @staticmethod
    def _load_feather(file_bytes: bytes):
        """
        Load Feather/Arrow file into a pandas DataFrame.
        """
        try:
            buf = io.BytesIO(file_bytes)
            return pd.read_feather(buf)
        except Exception as e:
            return {"error": f"Feather load failed: {e}"}

    @staticmethod
    def load_table(file_bytes: bytes, filename: str):
        """
        Load a tabular or structured file into a DataFrame or Python object.

        Possible inputs:
            - .csv   -> pandas.DataFrame
            - .tsv   -> pandas.DataFrame
            - .xlsx  -> pandas.DataFrame
            - .json  -> dict/list (JSON)
            - .parquet -> pandas.DataFrame (if supported)
            - .feather / .ft   -> pandas.DataFrame (if supported)

        Returns:
            - On success:
                * DataFrame for most table formats
                * dict/list for JSON
            - On failure:
                {"error": "message"}

        Backwards compatible with existing usage:
            Existing code that expects DataFrame for CSV/TSV/XLSX will still work.
            Existing code that expects {"error": "..."} still works.
        """
        name = filename.lower().strip()

        try:
            if name.endswith(".csv"):
                return pd.read_csv(io.BytesIO(file_bytes))

            if name.endswith(".tsv"):
                return pd.read_csv(io.BytesIO(file_bytes), sep="\t")

            if name.endswith(".xlsx") or name.endswith(".xls"):
                return pd.read_excel(io.BytesIO(file_bytes))

            if name.endswith(".json"):
                return DataConnectors._load_json(file_bytes)

            if name.endswith(".parquet"):
                return DataConnectors._load_parquet(file_bytes)

            if name.endswith(".feather") or name.endswith(".ft"):
                return DataConnectors._load_feather(file_bytes)

        except Exception as e:
            return {"error": str(e)}

        return {"error": f"Unsupported file type for table load: {filename}"}

    # ------------------------------------------------------------------
    # Text / generic loaders
    # ------------------------------------------------------------------
    @staticmethod
    def load_text(file_bytes: bytes, filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Load a simple text-like file: .txt, .md, .html, etc.

        Returns:
            {
                "ok": bool,
                "text": str,
                "filename": str or None,
                "error": str (if any),
            }
        """
        try:
            text = file_bytes.decode("utf-8")
        except Exception:
            try:
                text = file_bytes.decode("latin-1")
            except Exception as e:
                return {"ok": False, "error": f"text decode failed: {e}", "filename": filename}

        return {"ok": True, "text": text, "filename": filename}

    @staticmethod
    def load_any(file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """
        Generic entry point that tries to infer how to load a file
        based on its extension.

        Returns:
            {
                "ok": bool,
                "kind": "pdf" | "table" | "json" | "text" | "unknown",
                "data": ...         # DataFrame / dict / list / text
                "meta": {...},      # optional metadata
                "error": str,       # on failure
                "filename": str,
            }
        """
        name = filename.lower().strip()

        # PDF
        if name.endswith(".pdf"):
            pdf_res = DataConnectors.load_pdf(file_bytes)
            if pdf_res.get("ok"):
                return {
                    "ok": True,
                    "kind": "pdf",
                    "data": pdf_res.get("text", ""),
                    "meta": {
                        "pages": pdf_res.get("pages"),
                        "meta": pdf_res.get("meta", {}),
                    },
                    "filename": filename,
                }
            return {
                "ok": False,
                "kind": "pdf",
                "error": pdf_res.get("error", "unknown pdf error"),
                "filename": filename,
            }

        # Table / structured
        if any(
            name.endswith(ext)
            for ext in (".csv", ".tsv", ".xlsx", ".xls", ".json", ".parquet", ".feather", ".ft")
        ):
            table = DataConnectors.load_table(file_bytes, filename)
            if isinstance(table, dict) and "error" in table:
                return {
                    "ok": False,
                    "kind": "table",
                    "error": table["error"],
                    "filename": filename,
                }
            return {
                "ok": True,
                "kind": "table" if not isinstance(table, dict) else "json",
                "data": table,
                "filename": filename,
                "meta": {},
            }

        # Text / markdown / html
        if any(name.endswith(ext) for ext in (".txt", ".md", ".markdown", ".html", ".htm")):
            txt_res = DataConnectors.load_text(file_bytes, filename)
            if txt_res.get("ok"):
                return {
                    "ok": True,
                    "kind": "text",
                    "data": txt_res["text"],
                    "filename": filename,
                    "meta": {},
                }
            return {
                "ok": False,
                "kind": "text",
                "error": txt_res.get("error", "unknown text load error"),
                "filename": filename,
            }

        # Unknown type
        return {
            "ok": False,
            "kind": "unknown",
            "error": f"Unsupported or unknown file type: {filename}",
            "filename": filename,
        }
