from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import pymupdf as fitz  # type: ignore
except Exception:  # pragma: no cover
    try:
        import fitz  # type: ignore
    except Exception:  # pragma: no cover
        fitz = None

try:
    import pytesseract  # type: ignore
except Exception:  # pragma: no cover
    pytesseract = None

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None

try:
    from src.document_readability import analyze_page, compute_cer
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from src.document_readability import analyze_page, compute_cer


DEFAULT_TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def configure_tesseract(tesseract_cmd: Optional[str] = None) -> None:
    if pytesseract is None:
        raise RuntimeError("pytesseract is required for PDF OCR")

    pytesseract.pytesseract.tesseract_cmd = (
        tesseract_cmd
        or os.environ.get("TESSERACT_CMD")
        or DEFAULT_TESSERACT_CMD
    )


def _ensure_pdf_dependencies() -> None:
    missing = []
    if fitz is None:
        missing.append("PyMuPDF")
    if Image is None:
        missing.append("Pillow")
    if missing:
        raise RuntimeError(
            "PDF OCR requires the following packages: " + ", ".join(missing)
        )


def ocr_pdf_document(
    pdf_path: Union[str, os.PathLike],
    tesseract_cmd: Optional[str] = None,
    dpi: int = 300,
    render_dir: Optional[Union[str, os.PathLike]] = "review_outputs/rendered_pages",
) -> Dict[str, Any]:
    _ensure_pdf_dependencies()
    configure_tesseract(tesseract_cmd)

    document = fitz.open(os.fspath(pdf_path))
    page_texts: List[str] = []
    page_char_counts: List[int] = []
    rendered_paths: List[str] = []

    if render_dir is not None:
        render_path = Path(render_dir)
        render_path.mkdir(parents=True, exist_ok=True)
    else:
        render_path = None

    for page_number, page in enumerate(document, start=1):
        pixmap = page.get_pixmap(dpi=dpi)
        if render_path is not None:
            page_path = render_path / f"page_{page_number:04d}.png"
            pixmap.save(os.fspath(page_path))
            rendered_paths.append(str(page_path))

        mode = "RGBA" if pixmap.alpha else "RGB"
        image = Image.frombytes(mode, [pixmap.width, pixmap.height], pixmap.samples)
        text = pytesseract.image_to_string(image)
        cleaned_text = text.strip()
        page_texts.append(cleaned_text)
        page_char_counts.append(len(cleaned_text))
        print(
            f"Processed PDF page {page_number}/{len(document)} "
            f"({len(cleaned_text)} OCR characters).",
            flush=True,
        )

    full_text = "\n\n".join(page_texts).strip()
    document.close()
    return {
        "pdf_path": os.fspath(pdf_path),
        "page_count": len(page_texts),
        "page_texts": page_texts,
        "page_char_counts": page_char_counts,
        "total_char_count": sum(page_char_counts),
        "render_dir": str(render_path) if render_path is not None else None,
        "rendered_paths": rendered_paths,
        "ocr_text": full_text,
    }


def compare_image_with_pdf(
    image_path: Union[str, os.PathLike],
    pdf_path: Union[str, os.PathLike],
    ground_truth_text: Optional[str] = None,
    output_dir: Optional[str] = "preprocessed_outputs",
    tesseract_cmd: Optional[str] = None,
    render_dir: Optional[Union[str, os.PathLike]] = "review_outputs/rendered_pages",
) -> Dict[str, Any]:
    page_result = analyze_page(
        image_path,
        ground_truth_text=None,
        output_dir=output_dir,
    )

    pdf_result = ocr_pdf_document(
        pdf_path=pdf_path,
        tesseract_cmd=tesseract_cmd,
        render_dir=render_dir,
    )

    result: Dict[str, Any] = {
        "image_path": os.fspath(image_path),
        "pdf_path": os.fspath(pdf_path),
        **page_result,
        "pdf_page_count": pdf_result["page_count"],
        "pdf_ocr_text": pdf_result["ocr_text"],
        "pdf_page_texts": pdf_result["page_texts"],
        "pdf_page_char_counts": pdf_result["page_char_counts"],
        "pdf_total_char_count": pdf_result["total_char_count"],
        "pdf_render_dir": pdf_result["render_dir"],
        "pdf_rendered_paths": pdf_result["rendered_paths"],
    }

    if ground_truth_text is not None:
        result["pdf_cer"] = compute_cer(str(ground_truth_text), pdf_result["ocr_text"])

    return result


def build_pdf_review_report(
    image_path: Union[str, os.PathLike],
    pdf_path: Union[str, os.PathLike],
    output_dir: Optional[str] = "preprocessed_outputs",
    review_dir: Optional[str] = "review_outputs",
    tesseract_cmd: Optional[str] = None,
    render_dir: Optional[Union[str, os.PathLike]] = "review_outputs/rendered_pages",
) -> Dict[str, Any]:
    print("Analyzing representative PNG readability metrics...", flush=True)
    page_result = analyze_page(
        image_path,
        ground_truth_text=None,
        output_dir=output_dir,
    )

    print("OCR-ing notebook PDF...", flush=True)
    pdf_result = ocr_pdf_document(
        pdf_path=pdf_path,
        tesseract_cmd=tesseract_cmd,
        render_dir=render_dir,
    )

    result: Dict[str, Any] = {
        "image_path": os.fspath(image_path),
        "pdf_path": os.fspath(pdf_path),
        **page_result,
        "pdf_page_count": pdf_result["page_count"],
        "pdf_ocr_text": pdf_result["ocr_text"],
        "pdf_page_texts": pdf_result["page_texts"],
        "pdf_page_char_counts": pdf_result["page_char_counts"],
        "pdf_total_char_count": pdf_result["total_char_count"],
        "pdf_render_dir": pdf_result["render_dir"],
        "pdf_rendered_paths": pdf_result["rendered_paths"],
    }

    if review_dir is not None:
        review_path = Path(review_dir)
        review_path.mkdir(parents=True, exist_ok=True)

        stem = Path(os.fspath(image_path)).stem
        (review_path / f"{stem}_pdf_ocr.txt").write_text(
            pdf_result["ocr_text"],
            encoding="utf8",
        )
        (review_path / f"{stem}_pdf_pages.json").write_text(
            json.dumps(pdf_result["page_texts"], indent=2),
            encoding="utf8",
        )
        (review_path / f"{stem}_pdf_page_char_counts.json").write_text(
            json.dumps(pdf_result["page_char_counts"], indent=2),
            encoding="utf8",
        )
        (review_path / f"{stem}_pdf_char_count.txt").write_text(
            str(pdf_result["total_char_count"]),
            encoding="utf8",
        )
        result["review_dir"] = str(review_path)

    print(
        f"Finished. Processed {result['pdf_page_count']} PDF pages and "
        f"{result['pdf_total_char_count']} OCR characters.",
        flush=True,
    )
    return result
