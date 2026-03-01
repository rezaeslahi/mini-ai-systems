from config import AppConfig
from typing import List
from pathlib import Path



def load_doc_paths(config:AppConfig)->List[str]:
    allowed_suffix = {".pdf"}
    doc_folder_path = Path(config.documents_folder)

    # paths = [str(p) for p in doc_folder_path.iterdir() if p.is_file()]

    paths = [str(p) for p in doc_folder_path.rglob("*") if p.suffix.lower() in allowed_suffix]
    return paths[:config.max_doc_num]
