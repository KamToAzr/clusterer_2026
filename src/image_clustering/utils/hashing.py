from __future__ import annotations
import hashlib
import json
from typing import Any, Dict


def stable_hash_dict(d: Dict[str, Any], n_chars: int = 10) -> str:
    payload = json.dumps(d, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:n_chars]


def stable_hash_str(s: str, n_chars: int = 16) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:n_chars]