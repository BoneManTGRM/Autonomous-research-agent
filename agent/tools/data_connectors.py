import io
import json
import pandas as pd
from typing import Dict, Any, Optional

try:
    import PyPDF2
except Exception:
    PyPDF2 = None

class DataConnectors:
    """
    Unified interface for loading external data files.
    Used by CoreAgent + Streamlit for ingestion.
    """

    @staticmethod
    def load_pdf(pdf_bytes: bytes) -> Dict[str, Any]:
        if not PyPDF2:
            return {"ok": False, "error": "PyPDF2 not installed"}

        try:
            reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            return {"ok": True, "text": text}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @staticmethod
    def load_table(file_bytes: bytes, filename: str):
        name = filename.lower()

        try:
            if name.endswith(".csv"):
                return pd.read_csv(io.BytesIO(file_bytes))
            if name.endswith(".tsv"):
                return pd.read_csv(io.BytesIO(file_bytes), sep="\t")
            if name.endswith(".json"):
                return json.load(io.BytesIO(file_bytes))
            if name.endswith(".xlsx"):
                return pd.read_excel(io.BytesIO(file_bytes))
        except Exception as e:
            return {"error": str(e)}

        return {"error": "Unsupported file type"}
