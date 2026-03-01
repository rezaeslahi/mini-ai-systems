#!/usr/bin/env python3
"""
Download a curated set of RAG / vector-search / LLM system PDFs into a folder
at the root of your project.

Usage:
  python scripts/download_rag_docs.py
  python scripts/download_rag_docs.py --out rag_docs
  python scripts/download_rag_docs.py --force
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Dict, Tuple

import requests


DOCS: Dict[str, str] = {
    # Core RAG / Retrieval
    "01_rag_retrieval_augmented_generation_lewis_2020.pdf": "https://arxiv.org/pdf/2005.11401.pdf",
    "02_dpr_dense_passage_retrieval_karpukhin_2020.pdf": "https://arxiv.org/pdf/2004.04906.pdf",
    "03_faiss_similarity_search_johnson_2017.pdf": "https://arxiv.org/pdf/1702.08734.pdf",

    # Embeddings / Bi-encoders / Re-ranking
    "04_sentence_bert_reimers_2019.pdf": "https://arxiv.org/pdf/1908.10084.pdf",
    "05_colbert_late_interaction_khattab_2020.pdf": "https://arxiv.org/pdf/2004.12832.pdf",

    # LLM basics / context scaling (useful for RAG prompting + context)
    "06_attention_is_all_you_need_vaswani_2017.pdf": "https://arxiv.org/pdf/1706.03762.pdf",
    "07_bert_devlin_2018.pdf": "https://arxiv.org/pdf/1810.04805.pdf",
    "08_gpt3_brown_2020.pdf": "https://arxiv.org/pdf/2005.14165.pdf",
    "09_llama_touvron_2023.pdf": "https://arxiv.org/pdf/2302.13971.pdf",

    # Practical/production-ish: "Lost in the Middle" (RAG + ordering effects)
    "10_lost_in_the_middle_liu_2023.pdf": "https://arxiv.org/pdf/2307.03172.pdf",
}


def download_one(
    url: str,
    out_path: Path,
    *,
    timeout_s: int = 30,
    retries: int = 3,
    backoff_s: float = 1.5,
    chunk_size: int = 1024 * 256,
    force: bool = False,
) -> Tuple[bool, str]:
    """
    Returns (success, message).
    """
    if out_path.exists() and out_path.stat().st_size > 0 and not force:
        return True, f"SKIP (exists): {out_path.name}"

    tmp_path = out_path.with_suffix(out_path.suffix + ".part")

    last_err: str = ""
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout_s, allow_redirects=True) as r:
                r.raise_for_status()

                ctype = (r.headers.get("Content-Type") or "").lower()
                # arXiv typically returns application/pdf; be tolerant.
                if "pdf" not in ctype and not url.lower().endswith(".pdf"):
                    return False, f"NOT A PDF? content-type={ctype} url={url}"

                total = int(r.headers.get("Content-Length") or 0)

                out_path.parent.mkdir(parents=True, exist_ok=True)
                bytes_written = 0

                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        f.write(chunk)
                        bytes_written += len(chunk)

                # Basic sanity check: non-empty and, if length known, not wildly off.
                if bytes_written == 0:
                    raise RuntimeError("Downloaded 0 bytes")
                if total and bytes_written < min(total, 50_000):
                    # If total known and very small, likely an error page.
                    raise RuntimeError(f"Downloaded too small ({bytes_written} bytes, expected {total})")

                os.replace(tmp_path, out_path)
                return True, f"OK: {out_path.name} ({bytes_written:,} bytes)"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            # cleanup .part file
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass

            if attempt < retries:
                sleep_s = backoff_s ** (attempt - 1)
                time.sleep(sleep_s)
            else:
                break

    return False, f"FAIL: {out_path.name} -> {last_err}"


def main() -> int:
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="rag_docs", help="Output folder relative to project root")
    ap.add_argument("--force", action="store_true", help="Re-download even if file already exists")
    ap.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds")
    ap.add_argument("--retries", type=int, default=3, help="Number of retries per file")
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {len(DOCS)} PDFs into: {out_dir}")

    ok = 0
    for filename, url in DOCS.items():
        success, msg = download_one(
            url,
            out_dir / filename,
            timeout_s=args.timeout,
            retries=args.retries,
            force=args.force,
        )
        print(msg)
        ok += int(success)

    print(f"\nDone. Success: {ok}/{len(DOCS)}")
    return 0 if ok == len(DOCS) else 2


if __name__ == "__main__":
    raise SystemExit(main())
