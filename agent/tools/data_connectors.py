import io
import json
from typing import Any, Dict, Optional, Union, List

import pandas as pd

try:
    import PyPDF2  # type: ignore
except Exception:
    PyPDF2 = None

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:
    BeautifulSoup = None

try:
    import pyarrow  # noqa: F401
except Exception:
    pyarrow = None


class DataConnectors:
    """
    Unified, Option-C hardened data ingestion layer.

    Capabilities:
        • Multi-page PDF text extraction
        • CSV / TSV / XLSX / Parquet / Feather loading
        • JSON / NDJSON structured ingestion
        • HTML table extraction (best effort)
        • TXT/MD/HTML text loading
        • Unified load_any() returning {ok, kind, data, meta, error}

    Fully backwards compatible:
        • load_pdf()
        • load_table()
        • load_text()
        • load_any()
    """

    # ------------------------------------------------------------------
    # PDF
    # ------------------------------------------------------------------
    @staticmethod
    def load_pdf(pdf_bytes: bytes) -> Dict[str, Any]:
        if not PyPDF2:
            return {"ok": False, "error": "PyPDF2 not installed"}

        try:
            reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
            pages = []
            for page in reader.pages:
                try:
                    ptxt = page.extract_text() or ""
                except Exception:
                    ptxt = ""
                pages.append(ptxt)

            merged = "\n\n".join(pages).strip()

            meta_raw = getattr(reader, "metadata", None) or {}
            meta: Dict[str, str] = {}

            if isinstance(meta_raw, dict):
                for k, v in meta_raw.items():
                    try:
                        meta[str(k)] = str(v)
                    except Exception:
                        meta[str(k)] = repr(v)

            return {
                "ok": True,
                "text": merged,
                "pages": len(pages),
                "meta": meta,
            }
        except Exception as e:
            return {"ok": False, "error": f"PDF parse failed: {e}"}

    # ------------------------------------------------------------------
    # JSON + NDJSON
    # ------------------------------------------------------------------
    @staticmethod
    def _load_json(file_bytes: bytes) -> Union[Dict[str, Any], Any]:
        try:
            text = file_bytes.decode("utf-8")
        except Exception:
            text = file_bytes.decode("utf-8", errors="ignore")
        return json.loads(text)

    @staticmethod
    def _load_ndjson(file_bytes: bytes) -> List[Any]:
        try:
            text = file_bytes.decode("utf-8", errors="ignore")
            items = []
            for line in text.splitlines():
                ln = line.strip()
                if not ln:
                    continue
                try:
                    items.append(json.loads(ln))
                except Exception:
                    continue
            return items
        except Exception as e:
            return [{"error": str(e)}]

    # ------------------------------------------------------------------
    # Parquet / Feather / DataFrame loaders
    # ------------------------------------------------------------------
    @staticmethod
    def _load_parquet(file_bytes: bytes):
        try:
            return pd.read_parquet(io.BytesIO(file_bytes))
        except Exception as e:
            return {"error": f"Parquet load failed: {e}"}

    @staticmethod
    def _load_feather(file_bytes: bytes):
        try:
            return pd.read_feather(io.BytesIO(file_bytes))
        except Exception as e:
            return {"error": f"Feather load failed: {e}"}

    # ------------------------------------------------------------------
    # HTML table extraction (best effort)
    # ------------------------------------------------------------------
    @staticmethod
    def _load_html_table(file_bytes: bytes):
        if BeautifulSoup is None:
            return {"error": "bs4 not installed for HTML table extraction"}

        try:
            text = file_bytes.decode("utf-8", errors="ignore")
            soup = BeautifulSoup(text, "html.parser")
            tables = soup.find_all("table")
            if not tables:
                return {"error": "No HTML <table> found"}

            # Convert first table to DataFrame
            rows = []
            header = []

            # Extract header
            thead = tables[0].find("thead")
            if thead:
                ths = thead.find_all("th")
                header = [th.get_text(strip=True) for th in ths]

            # Extract body rows
            trs = tables[0].find_all("tr")
            for tr in trs:
                cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                if cells:
                    rows.append(cells)

            if not header and rows:
                header = [f"col_{i}" for i in range(len(rows[0]))]

            import pandas as _pd
            df = _pd.DataFrame(rows, columns=header)
            return df

        except Exception as e:
            return {"error": f"HTML table parse error: {e}"}

    # ------------------------------------------------------------------
    # Tabular unified loader
    # ------------------------------------------------------------------
    @staticmethod
    def load_table(file_bytes: bytes, filename: str):
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

            if name.endswith(".ndjson"):
                return DataConnectors._load_ndjson(file_bytes)

            if name.endswith(".parquet"):
                return DataConnectors._load_parquet(file_bytes)

            if name.endswith(".feather") or name.endswith(".ft"):
                return DataConnectors._load_feather(file_bytes)

            if name.endswith(".html") or name.endswith(".htm"):
                return DataConnectors._load_html_table(file_bytes)

        except Exception as e:
            return {"error": f"Table load failed: {e}"}

        return {"error": f"Unsupported file type for table load: {filename}"}

    # ------------------------------------------------------------------
    # Text
    # ------------------------------------------------------------------
    @staticmethod
    def load_text(file_bytes: bytes, filename: Optional[str] = None) -> Dict[str, Any]:
        try:
            text = file_bytes.decode("utf-8")
        except Exception:
            try:
                text = file_bytes.decode("latin-1")
            except Exception as e:
                return {"ok": False, "error": f"text decode failed: {e}", "filename": filename}

        return {"ok": True, "text": text, "filename": filename}

    # ------------------------------------------------------------------
    # UNIVERSAL LOADER — Option C upgrade
    # ------------------------------------------------------------------
    @staticmethod
    def load_any(file_bytes: bytes, filename: str) -> Dict[str, Any]:
        name = filename.lower().strip()

        # 1. PDF
        if name.endswith(".pdf"):
            pdf = DataConnectors.load_pdf(file_bytes)
            if pdf.get("ok"):
                return {
                    "ok": True,
                    "kind": "pdf",
                    "data": pdf.get("text", ""),
                    "meta": {
                        "pages": pdf.get("pages"),
                        "metadata": pdf.get("meta", {}),
                    },
                    "filename": filename,
                }
            return {
                "ok": False,
                "kind": "pdf",
                "error": pdf.get("error", "pdf error"),
                "filename": filename,
            }

        # 2. Structured / table
        if any(
            name.endswith(ext)
            for ext in [
                ".csv", ".tsv", ".xlsx", ".xls",
                ".json", ".ndjson",
                ".parquet", ".feather", ".ft",
                ".html", ".htm",
            ]
        ):
            table = DataConnectors.load_table(file_bytes, filename)

            if isinstance(table, dict) and "error" in table:
                return {
                    "ok": False,
                    "kind": "table",
                    "error": table["error"],
                    "filename": filename,
                }

            kind = "json" if isinstance(table, dict) else "table"

            return {
                "ok": True,
                "kind": kind,
                "data": table,
                "filename": filename,
                "meta": {},
            }

        # 3. Text files
        if any(name.endswith(ext) for ext in [".txt", ".md", ".markdown"]):
            txt = DataConnectors.load_text(file_bytes, filename)
            if txt.get("ok"):
                return {
                    "ok": True,
                    "kind": "text",
                    "data": txt["text"],
                    "filename": filename,
                    "meta": {},
                }
            return {
                "ok": False,
                "kind": "text",
                "error": txt.get("error"),
                "filename": filename,
            }

        # 4. Unknown
        return {
            "ok": False,
            "kind": "unknown",
            "error": f"Unsupported file type: {filename}",
            "filename": filename,
        }
