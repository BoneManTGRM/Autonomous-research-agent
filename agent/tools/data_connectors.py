import io
import json
from typing import Any, Dict, Optional, Union

import pandas as pd

# ----------------------------------------------------------------------
# Optional PDF engines
# ----------------------------------------------------------------------
try:
    import PyPDF2  # type: ignore[import]
except Exception:  # pragma: no cover
    PyPDF2 = None  # type: ignore[assignment]

# Layout-aware PDF engine
try:
    import fitz  # PyMuPDF  # type: ignore[import]
except Exception:  # pragma: no cover
    fitz = None  # type: ignore[assignment]

# OCR fallback stack
try:
    import pytesseract  # type: ignore[import]
    from PIL import Image  # type: ignore[import]
except Exception:  # pragma: no cover
    pytesseract = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment]


class DataConnectors:
    """
    Unified interface for loading external data files.
    Used by CoreAgent + Streamlit for ingestion.

    Backwards compatible:
        - load_pdf(pdf_bytes) -> {"ok": bool, "text" or "error": str}
        - load_table(file_bytes, filename) -> DataFrame OR dict with "error"

    Extended capabilities (Hybrid PDF Engine, Option C):
        - PyPDF2 baseline text extraction + metadata
        - PyMuPDF layout-aware text extraction (if installed)
        - OCR fallback for image-only PDFs (if pytesseract + PIL available)
        - CSV / TSV / XLSX / Parquet / Feather / JSON table loading
        - TXT / MD / HTML lightweight text loading
        - Generic load_any entry point that returns type and content

    The PDF pipeline never raises and always degrades gracefully:
        PyPDF2 -> PyMuPDF -> OCR (when available).
    """

    # ------------------------------------------------------------------
    # Internal helpers for PDF quality estimation
    # ------------------------------------------------------------------
    @staticmethod
    def _estimate_text_density(text: str, pages: int) -> float:
        """
        Rough heuristic for text density:
            chars per page (lower values often indicate image-only scans).
        """
        if pages <= 0:
            pages = 1
        return len(text) / float(pages)

    @staticmethod
    def _is_likely_image_only(text: str, pages: int) -> bool:
        """
        Decide if a PDF is likely image-only based on low text density.
        Threshold is intentionally conservative.
        """
        density = DataConnectors._estimate_text_density(text, pages)
        # e.g. < 80 chars/page is suspiciously low for typical papers
        return density < 80.0

    # ------------------------------------------------------------------
    # PDF (Hybrid Engine)
    # ------------------------------------------------------------------
    @staticmethod
    def _load_pdf_pypdf2(pdf_bytes: bytes) -> Dict[str, Any]:
        """Baseline PDF extraction via PyPDF2."""
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
                "engine": "pypdf2",
            }
        except Exception as e:
            return {"ok": False, "error": str(e), "engine": "pypdf2"}

    @staticmethod
    def _load_pdf_pymupdf(pdf_bytes: bytes) -> Dict[str, Any]:
        """
        Layout-aware extraction via PyMuPDF (fitz).

        Produces generally better text than PyPDF2 for complex scientific PDFs.
        """
        if not fitz:
            return {"ok": False, "error": "PyMuPDF not installed", "engine": "pymupdf"}

        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")  # type: ignore[arg-type]
            page_texts = []
            for page_index in range(doc.page_count):
                try:
                    page = doc.load_page(page_index)
                    # "text" mode balances structure vs. raw content
                    text = page.get_text("text") or ""
                except Exception:
                    text = ""
                page_texts.append(text)

            merged = "\n\n".join(page_texts).strip()

            meta_raw = doc.metadata or {}
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
                "pages": doc.page_count,
                "meta": meta,
                "engine": "pymupdf",
            }
        except Exception as e:
            return {"ok": False, "error": str(e), "engine": "pymupdf"}

    @staticmethod
    def _load_pdf_ocr(pdf_bytes: bytes, max_ocr_pages: int = 10) -> Dict[str, Any]:
        """
        OCR fallback using PyMuPDF to render pages and pytesseract to read text.

        Only used when:
            - PDF appears image-only, OR
            - other extractors fail to produce meaningful text.

        max_ocr_pages caps runtime and is safe for 90-day runs.
        """
        if not (fitz and pytesseract and Image):
            return {
                "ok": False,
                "error": "OCR stack not available (requires PyMuPDF, pytesseract, and PIL).",
                "engine": "ocr",
            }

        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")  # type: ignore[arg-type]
            ocr_texts = []
            pages_used = min(doc.page_count, max_ocr_pages)

            for page_index in range(pages_used):
                try:
                    page = doc.load_page(page_index)
                    pix = page.get_pixmap()
                    img_mode = "RGB" if pix.alpha == 0 else "RGBA"
                    img = Image.frombytes(img_mode, (pix.width, pix.height), pix.samples)  # type: ignore[arg-type]
                    text = pytesseract.image_to_string(img) or ""
                except Exception:
                    text = ""
                ocr_texts.append(text)

            merged = "\n\n".join(ocr_texts).strip()

            return {
                "ok": True,
                "text": merged,
                "pages": doc.page_count,
                "meta": {
                    "ocr_pages_processed": pages_used,
                    "ocr_engine": "pytesseract",
                },
                "engine": "ocr",
            }
        except Exception as e:
            return {"ok": False, "error": str(e), "engine": "ocr"}

    @staticmethod
    def load_pdf(pdf_bytes: bytes) -> Dict[str, Any]:
        """
        Load a PDF into plain text using a hybrid extraction strategy.

        Order:
            1. PyPDF2 baseline
            2. PyMuPDF layout-aware (if installed)
            3. OCR fallback for image-only PDFs (if stack available)

        Returns:
            {
                "ok": bool,
                "text": str,          # merged text from all pages
                "pages": int,         # number of pages (if ok)
                "meta": {...},        # includes engine + extras
                "error": str,         # on failure
            }

        Backwards compatibility:
            Existing callers expecting {"ok": bool, "text": "..."} still work.
        """
        # If nothing is installed, fail fast but safely.
        if not (PyPDF2 or fitz or (pytesseract and Image)):
            return {
                "ok": False,
                "error": "No PDF engine available (need PyPDF2 or PyMuPDF or OCR stack).",
            }

        # ---- 1) PyPDF2 baseline (if available) ----
        best_res: Optional[Dict[str, Any]] = None
        pypdf2_res: Optional[Dict[str, Any]] = None

        if PyPDF2:
            pypdf2_res = DataConnectors._load_pdf_pypdf2(pdf_bytes)
            if pypdf2_res.get("ok"):
                best_res = pypdf2_res

        # ---- 2) PyMuPDF layout-aware (if available) ----
        pymu_res: Optional[Dict[str, Any]] = None
        if fitz:
            pymu_res = DataConnectors._load_pdf_pymupdf(pdf_bytes)
            if pymu_res.get("ok"):
                # Heuristic: prefer PyMuPDF when it clearly yields more text
                if best_res is None:
                    best_res = pymu_res
                else:
                    base_text = best_res.get("text", "") or ""
                    adv_text = pymu_res.get("text", "") or ""
                    if len(adv_text) > len(base_text) * 1.2:
                        best_res = pymu_res

        # Decide if we need OCR
        need_ocr = False
        base_for_density = best_res or pypdf2_res or pymu_res
        if base_for_density and base_for_density.get("ok"):
            text = base_for_density.get("text", "") or ""
            pages = int(base_for_density.get("pages") or 0)
            if DataConnectors._is_likely_image_only(text, pages):
                need_ocr = True
        else:
            # If nothing worked well, OCR is our last hope (if available)
            need_ocr = True

        # ---- 3) OCR fallback (if stack available & needed) ----
        ocr_res: Optional[Dict[str, Any]] = None
        if need_ocr and pytesseract and Image and fitz:
            ocr_res = DataConnectors._load_pdf_ocr(pdf_bytes)
            if ocr_res.get("ok"):
                # Prefer OCR result if it adds significant text
                if best_res is None:
                    best_res = ocr_res
                else:
                    base_text = best_res.get("text", "") or ""
                    ocr_text = ocr_res.get("text", "") or ""
                    if len(ocr_text) > len(base_text) * 1.1:
                        best_res = ocr_res

        # If we have a best result, normalize and return
        if best_res and best_res.get("ok"):
            meta = best_res.get("meta", {}) or {}
            engine = best_res.get("engine")
            if engine:
                meta["engine"] = engine

            return {
                "ok": True,
                "text": best_res.get("text", "") or "",
                "pages": int(best_res.get("pages") or 0),
                "meta": meta,
            }

        # Otherwise, build a combined error summary
        errors = []
        for res in (pypdf2_res, pymu_res, ocr_res):
            if res and not res.get("ok") and res.get("error"):
                engine = res.get("engine") or "unknown"
                errors.append(f"{engine}: {res['error']}")

        return {
            "ok": False,
            "error": "; ".join(errors) if errors else "Unknown PDF parse error",
        }

    # ------------------------------------------------------------------
    # Tabular / structured data
    # ------------------------------------------------------------------
    @staticmethod
    def _load_json(file_bytes: bytes) -> Union[Dict[str, Any], Any]:
        """
        Load JSON content from bytes.

        Returns dict/list or raises.
        """
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
