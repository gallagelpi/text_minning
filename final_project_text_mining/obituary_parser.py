import re
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

# --------- Patterns para limpieza y abreviaturas ---------
_DOT = "<DOT>"
ABBREV_PATTERNS = [
    (r"\bp\.m\.(?=\W|$)", f"p{_DOT}m{_DOT}"),
    (r"\ba\.m\.(?=\W|$)", f"a{_DOT}m{_DOT}"),
    (r"\bmr\.(?=\W|$)", f"mr{_DOT}"),
    (r"\bmrs\.(?=\W|$)", f"mrs{_DOT}"),
    (r"\bms\.(?=\W|$)", f"ms{_DOT}"),
    (r"\bdr\.(?=\W|$)", f"dr{_DOT}"),
    (r"\bjr\.(?=\W|$)", f"jr{_DOT}"),
    (r"\bsr\.(?=\W|$)", f"sr{_DOT}"),
    (r"\b([A-Z])\.(?=\W|$)", rf"\1{_DOT}"),  # iniciales
    (r"\b(ST|RD|AVE|BLVD|LN|DR|PL|TER|CIR|PKWY|HWY|TRL|SQ)\.(?=\W|$)", rf"\1{_DOT}")
]

ABBREV_PATTERNS_RE = [(re.compile(p, flags=re.I), repl) for p, repl in ABBREV_PATTERNS]

# --------- Global Cues ---------
BIO_CUES: List[str] = []
FAMILY_CUES: List[str] = []
MEMORIAL_CUES: List[str] = []
BIO_CUES_L: tuple[str, ...] = ()
FAMILY_CUES_L: tuple[str, ...] = ()
MEMORIAL_CUES_L: tuple[str, ...] = ()

def load_cues_file(candidates: List[str]) -> Dict[str, List[str]]:
    sections = {"BIO_CUES": [], "FAMILY_CUES": [], "MEMORIAL_CUES": []}
    cue_file = None
    for candidate in candidates:
        p = Path(candidate)
        if p.exists():
            cue_file = p
            break

    if cue_file is None:
        raise FileNotFoundError("Could not find cues.txt.")

    current = None
    for raw in cue_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ";")): continue
        if line.startswith("[") and line.endswith("]"):
            name = line[1:-1].strip().upper()
            current = name if name in sections else None
            continue
        if current: sections[current].append(line)
    return sections

def configure_cues(cue_candidates: Optional[List[str]] = None) -> None:
    global BIO_CUES, FAMILY_CUES, MEMORIAL_CUES
    global BIO_CUES_L, FAMILY_CUES_L, MEMORIAL_CUES_L

    module_dir = Path(__file__).resolve().parent
    candidates = cue_candidates or [str(module_dir / "cues.txt"), "cues.txt"]
    
    loaded_cues = load_cues_file(candidates)
    BIO_CUES = loaded_cues["BIO_CUES"]
    FAMILY_CUES = loaded_cues["FAMILY_CUES"]
    MEMORIAL_CUES = loaded_cues["MEMORIAL_CUES"]

    BIO_CUES_L = tuple(c.lower() for c in BIO_CUES)
    FAMILY_CUES_L = tuple(c.lower() for c in FAMILY_CUES)
    MEMORIAL_CUES_L = tuple(c.lower() for c in MEMORIAL_CUES)

# --------- Sentence Splitting ---------
def protect_abbrev_dots(text: str) -> str:
    out = text
    for pattern, repl in ABBREV_PATTERNS_RE:
        out = pattern.sub(repl, out)
    return out

def restore_abbrev_dots(text: str) -> str:
    return text.replace(_DOT, ".")

def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    protected = protect_abbrev_dots(text)
    protected = re.sub(r"([.!?])(?=[A-Z])", r"\1 ", protected)
    parts = re.split(r"(?<=[.!?])\s+", protected)
    return [restore_abbrev_dots(p).strip() for p in parts if p.strip()]

# --------- Section Logic ---------
def _has_cue(sentence: str, cues: tuple[str, ...]) -> bool:
    s = sentence.lower()
    return any(c in s for c in cues)

def split_into_sections(text: str) -> Dict[str, str]:
    sentences = split_sentences(text)
    bio, life, family, memorial = [], [], [], []

    for sent in sentences:
        if _has_cue(sent, FAMILY_CUES_L):
            family.append(sent)
        elif _has_cue(sent, BIO_CUES_L):
            bio.append(sent)
        elif _has_cue(sent, MEMORIAL_CUES_L):
            memorial.append(sent)
        else:
            life.append(sent)

    return {
        "bio": " ".join(bio).strip(),
        "life": " ".join(life).strip(),
        "family": " ".join(family).strip(),
        "memorial": " ".join(memorial).strip(),
    }

# --------- Main Parsing ---------
def parse_obituary(text: str) -> Dict:
    text = str(text) if text is not None else ""
    sections = split_into_sections(text)
    return {"sections": sections}

def _safe_parse_text_obj(text_obj: object) -> Dict:
    # Maneja diccionarios o tuplas/listas extrayendo solo el texto
    obituary_text = ""
    if isinstance(text_obj, (tuple, list)) and len(text_obj) >= 2:
        obituary_text = text_obj[1]
    elif isinstance(text_obj, dict):
        obituary_text = text_obj.get("biografia", text_obj.get("text", ""))
    else:
        obituary_text = str(text_obj)
    
    return parse_obituary(obituary_text)

def _parse_pairs_chunk(chunk: List[tuple[object, object]]) -> List[tuple[object, Dict]]:
    configure_cues()
    return [(idx, _safe_parse_text_obj(text)) for idx, text in chunk]

def parse_pairs_nonbatch(pairs: List[tuple[object, object]]) -> Dict[object, Dict]:
    """
    Procesa los obituarios uno a uno de forma secuencial. 
    Útil para datasets pequeños o para debugging.
    """
    configure_cues()
    parsed: Dict[object, Dict] = {}
    for idx, text_obj in pairs:
        parsed[idx] = _safe_parse_text_obj(text_obj)
    return parsed

def parse_pairs_parallel(
    pairs: List[tuple[object, object]], 
    chunk_size: int = 1000,
    max_workers: Optional[int] = None,
) -> Dict[object, Dict]:
    configure_cues()
    if not pairs: return {}

    workers = max_workers or max(1, (os.cpu_count() or 1) - 1)
    chunks = [pairs[i : i + chunk_size] for i in range(0, len(pairs), chunk_size)]

    parsed: Dict[object, Dict] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for chunk_results in tqdm(executor.map(_parse_pairs_chunk, chunks), total=len(chunks), desc="Processing Sections"):
            for idx, result in chunk_results:
                parsed[idx] = result
    return parsed

__all__ = [
    "parse_obituary",
    "split_into_sections",
    "parse_pairs_parallel",
    "configure_cues",
]