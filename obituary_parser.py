import re
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from datetime import datetime

from dateparser import parse


# --------- Patterns ---------
MONTH = r"(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
DAY_NAME = r"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
US_STATES = r"(AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY|DC)"
ADDRESS_ABBREVS = r"(ST|RD|AVE|BLVD|LN|DR|PL|TER|CIR|PKWY|HWY|TRL|SQ)"


DATE_PATTERNS = [
    rf"{MONTH}\s+\d{{1,2}},\s*\d{{4}}",  # March 12, 2012 / March 12,2012
    rf"\d{{1,2}}\s+{MONTH}\s+\d{{4}}",  # 12 March 2012
    rf"{MONTH}\s+\d{{4}}",  # March 2012
    r"\d{1,2}/\d{1,2}/\d{2,4}",  # 03/12/2012
    r"\d{4}-\d{1,2}-\d{1,2}",  # 2012-03-12
    rf"{DAY_NAME},\s+{MONTH}\s+\d{{1,2}},\s*\d{{4}}",
    rf"{DAY_NAME}\s+{MONTH}\s+\d{{1,2}},\s*\d{{4}}",
    # Patterns without explicit year (e.g., 'March 12', '12 March', '03/12')
    rf"{MONTH}\s+\d{{1,2}}",  # March 12 (no year)
    rf"\d{{1,2}}\s+{MONTH}",  # 12 March (no year)
    r"\d{1,2}/\d{1,2}",  # 03/12 (month/day without year)
]
ANY_DATE = rf"(?:{'|'.join(DATE_PATTERNS)})"
ANY_DATE_RE = re.compile(ANY_DATE, flags=re.I)


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
    (r"\b([A-Z])\.(?=\W|$)", rf"\1{_DOT}"),  # initials
    (rf"\b({US_STATES})\.(?=\s+(?:\d|(?-i:[a-z])))", rf"\1{_DOT}"),
    (rf"\b({ADDRESS_ABBREVS})\.(?=\W|$)", rf"\1{_DOT}")
]

ABBREV_PATTERNS_RE = [(re.compile(p, flags=re.I), repl) for p, repl in ABBREV_PATTERNS]


name_token = r"[A-Z][A-Za-z'`.-]+"
paren_token = r"\([A-Za-z'`.-]+\)"

HEADER_PATTERN_STRS = [
    rf"^\s*(?:[A-Z][A-Za-z .'-]+,\s*[A-Z]{{2}}\s*---\s*)?"
    rf"(?:Mr|Mrs|Ms|Miss|Dr)\.?\s+{name_token}(?:\s+{name_token}|\s+{paren_token}){{0,6}}\s*,\s*(\d{{1,3}})\s*,",
    rf"^\s*(?:[A-Z][A-Za-z .'-]+,\s*[A-Z]{{2}}\s*---\s*)?"
    rf"{name_token}\s*,\s*{name_token}(?:\s+{name_token}|\s+{paren_token}){{0,6}}\s*,\s*(\d{{1,3}})\s*,",
    rf"^\s*(?:[A-Z][A-Za-z .'-]+,\s*[A-Z]{{2}}\s*---\s*)?"
    rf"{name_token}(?:\s+{name_token}|\s+{paren_token}){{0,8}}\s*,\s*(\d{{1,3}})\s*,",
    rf"^\s*(?:[A-Z][A-Za-z .'-]+,\s*[A-Z]{{2}}\s*---\s*)?"
    rf"{name_token}(?:\s+{name_token}|\s+{paren_token}){{0,8}}\s*\((\d{{1,3}})\)",
]
HEADER_PATTERNS_RE = [re.compile(p, flags=re.M) for p in HEADER_PATTERN_STRS]

FIRST_SENTENCE_PATTERN_STRS = [
    r",\s*age\s*(\d{1,3})\s*,",  # Name, age 84,
    r"\bage\s*[:\-]?\s*(\d{1,3})\b",  # age 84 / age: 84
    r"\baged?\s*(\d{1,3})\b",  # age 84 / aged 84
    r"\b(\d{1,3})\s*(?:years? old|yrs? old|yo)\b",  # 84 years old / 84 yo
    r"\((\d{1,3})\)",  # (84)
    r",\s*(\d{1,3})\s*,",  # Name, 84,
    r"\b[A-Z][A-Za-z'`.-]+(?:\s+[A-Z][A-Za-z'`.-]+){0,5}\s+(\d{1,3})\b",  # Name 84
    r",\s*(\d{1,3})\b",  # Name, 62 ...
]
FIRST_SENTENCE_PATTERNS_RE = [re.compile(p, flags=re.I) for p in FIRST_SENTENCE_PATTERN_STRS]


@dataclass
class ExtractedFields:
    birth_date_raw: Optional[str] = None
    death_date_raw: Optional[str] = None
    obituary_date_raw: Optional[str] = None
    birth_date: Optional[datetime] = None
    death_date: Optional[datetime] = None
    obituary_date: Optional[datetime] = None
    age: Optional[int] = None
    gender: Optional[str] = None


BIO_CUES: List[str] = []
FAMILY_CUES: List[str] = []
MEMORIAL_CUES: List[str] = []
DEATH_CUES: List[str] = []
BIO_CUES_L: tuple[str, ...] = ()
FAMILY_CUES_L: tuple[str, ...] = ()
MEMORIAL_CUES_L: tuple[str, ...] = ()


def load_cues_file(candidates: List[str]) -> Dict[str, List[str]]:
    sections = {
        "BIO_CUES": [],
        "FAMILY_CUES": [],
        "MEMORIAL_CUES": [],
        "DEATH_CUES": [],
    }

    cue_file = None
    for candidate in candidates:
        p = Path(candidate)
        if p.exists():
            cue_file = p
            break

    if cue_file is None:
        raise FileNotFoundError(
            "Could not find cues.txt. Expected one of: " + ", ".join(candidates)
        )

    current = None
    for raw in cue_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue

        if line.startswith("[") and line.endswith("]"):
            name = line[1:-1].strip().upper()
            current = name if name in sections else None
            continue

        if current:
            sections[current].append(line)

    return sections


def _default_cue_candidates() -> List[str]:
    module_dir = Path(__file__).resolve().parent
    # Primary source: cues.txt beside this module (scratch/cues.txt).
    primary = module_dir / "cues.txt"
    return [
        str(primary),
        "cues.txt",
        "scratch/cues.txt",
        "../scratch/cues.txt",
    ]


def configure_cues(cue_candidates: Optional[List[str]] = None) -> None:
    global BIO_CUES, FAMILY_CUES, MEMORIAL_CUES, DEATH_CUES
    global BIO_CUES_L, FAMILY_CUES_L, MEMORIAL_CUES_L

    candidates = cue_candidates or _default_cue_candidates()
    loaded_cues = load_cues_file(candidates)

    BIO_CUES = loaded_cues["BIO_CUES"]
    FAMILY_CUES = loaded_cues["FAMILY_CUES"]
    MEMORIAL_CUES = loaded_cues["MEMORIAL_CUES"]
    DEATH_CUES = loaded_cues["DEATH_CUES"]

    BIO_CUES_L = tuple(c.lower() for c in BIO_CUES)
    FAMILY_CUES_L = tuple(c.lower() for c in FAMILY_CUES)
    MEMORIAL_CUES_L = tuple(c.lower() for c in MEMORIAL_CUES)


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


def period_sentences(text: str) -> List[str]:
    protected = protect_abbrev_dots(text)
    chunks = [m.group(0).strip() for m in re.finditer(r"[^.]*\.", protected, flags=re.S)]
    return [restore_abbrev_dots(c) for c in chunks]


def _has_cue(sentence: str, cues: tuple[str, ...]) -> bool:
    s = sentence.lower()
    return any(c in s for c in cues)


def is_bio_sentence(sentence: str) -> bool:
    return _has_cue(sentence, BIO_CUES_L)


def is_family_sentence(sentence: str) -> bool:
    return _has_cue(sentence, FAMILY_CUES_L)


def is_memorial_sentence(sentence: str) -> bool:
    return _has_cue(sentence, MEMORIAL_CUES_L)


def split_into_sections(text: str) -> Dict[str, str]:
    sentences = split_sentences(text)

    bio: List[str] = []
    life: List[str] = []
    family: List[str] = []
    memorial: List[str] = []

    for sent in sentences:
        if is_family_sentence(sent):
            family.append(sent)
        elif is_bio_sentence(sent):
            bio.append(sent)
        elif is_memorial_sentence(sent):
            memorial.append(sent)


        else:
            life.append(sent)

    return {
        "bio": " ".join(bio).strip(),
        "life": " ".join(life).strip(),
        "family": " ".join(family).strip(),
        "memorial": " ".join(memorial).strip(),
    }


def find_date_in_text(s: str) -> Optional[str]:
    m = ANY_DATE_RE.search(s)
    if m:
        return m.group(0).strip()
    return None


def extract_birth_date(text: str) -> Optional[str]:
    birth_cues = [r"\bborn\b", r"\bbirth(?:day| date)?\b"]
    for sent in period_sentences(text):
        if any(re.search(c, sent, flags=re.I) for c in birth_cues):
            d = find_date_in_text(sent)
            if d:
                return d
    return None


def extract_death_date(text: str) -> Optional[str]:
    for sent in period_sentences(text):
        if any(re.search(rf"\b{re.escape(c.lower())}\b", sent.lower()) for c in DEATH_CUES):
            d = find_date_in_text(sent)
            if d:
                return d
    return None


def extract_age(text: str) -> Optional[int]:
    for pattern in HEADER_PATTERNS_RE:
        m = pattern.search(text)
        if not m or m.lastindex is None:
            continue
        try:
            age = int(m.group(1))
        except Exception:
            continue
        if 0 < age < 125:
            return age

    sents = split_sentences(text)
    if sents:
        first = sents[0]
        for pattern in FIRST_SENTENCE_PATTERNS_RE:
            m = pattern.search(first)
            if not m:
                continue
            age = int(m.group(1))
            if 0 < age < 125:
                return age
    return None


def infer_gender(text: str) -> Optional[str]:
    s = text.lower()
    male_score = len(re.findall(r"\b(he|his|him|husband|father|son)\b", s))
    female_score = len(re.findall(r"\b(she|her|hers|wife|mother|daughter)\b", s))
    if male_score > female_score and male_score > 0:
        return "male"
    if female_score > male_score and female_score > 0:
        return "female"
    return None

def parse_date_base(
    date_str: Optional[str], reference_dt: Optional[datetime]
) -> Optional[datetime]:
    if not date_str:
        return None

    base = reference_dt or datetime(1900, 1, 1)
    parsed = parse(date_str, settings={"RELATIVE_BASE": base})
    parsed = pd.to_datetime(parsed)
    return parsed


def parse_obituary(text: str, obituary_date: Optional[object] = None) -> Dict:
    text = str(text) if text is not None else ""
    sections = split_into_sections(text)
    full_text = f" {text.strip()} "

    parsed_obit_date = pd.to_datetime(obituary_date, errors="coerce")
    reference_dt = (
        parsed_obit_date.to_pydatetime()
        if not pd.isna(parsed_obit_date)
        else datetime(1900, 1, 1)
    )

    birth_date_raw = extract_birth_date(full_text)
    death_date_raw = extract_death_date(full_text)

    fields = ExtractedFields(
        birth_date_raw=birth_date_raw,
        death_date_raw=death_date_raw,
        obituary_date_raw=obituary_date,
        birth_date=parse_date_base(birth_date_raw, reference_dt),
        death_date=parse_date_base(death_date_raw, reference_dt),
        obituary_date=parsed_obit_date,
        age=extract_age(full_text),
        gender=infer_gender(full_text),
    )

    return {
        "sections": sections,
        "extracted": asdict(fields),
    }


def _safe_parse_text_obj(text: object) -> Dict:
    obituary_text: object = text
    obituary_date: Optional[object] = None

    if isinstance(text, (tuple, list)) and len(text) >= 2:
        obituary_date, obituary_text = text[0], text[1]
    elif isinstance(text, dict):
        obituary_date = text.get("obituary_date", text.get("data"))
        obituary_text = text.get("biografia", text.get("text", ""))

    return parse_obituary(obituary_text, obituary_date)


def parse_pairs_nonbatch(pairs: List[tuple[object, object]]) -> Dict[object, Dict]:
    parsed: Dict[object, Dict] = {}
    for idx, text in pairs:
        parsed[idx] = _safe_parse_text_obj(text)
    return parsed


def _parse_pairs_chunk(chunk: List[tuple[object, object]]) -> List[tuple[object, Dict]]:
    return [(idx, _safe_parse_text_obj(text)) for idx, text in chunk]


def parse_pairs_parallel(
    pairs: List[tuple[object, object]],
    chunk_size: int = 1000,
    max_workers: Optional[int] = None,
) -> Dict[object, Dict]:
    if not pairs:
        return {}

    workers = max_workers or max(1, (os.cpu_count() or 1) - 1)
    chunks: List[List[tuple[object, object]]] = [
        pairs[i : i + chunk_size] for i in range(0, len(pairs), chunk_size)
    ]

    parsed: Dict[object, Dict] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for chunk_results in executor.map(_parse_pairs_chunk, chunks):
            for idx, result in chunk_results:
                parsed[idx] = result
    return parsed


# Load cues at import time so parse_obituary is ready immediately.
configure_cues()


__all__ = [
    "parse_obituary",
    "split_into_sections",
    "extract_birth_date",
    "extract_death_date",
    "extract_age",
    "infer_gender",
    "parse_pairs_nonbatch",
    "parse_pairs_parallel",
    "configure_cues",
    "load_cues_file",
]
