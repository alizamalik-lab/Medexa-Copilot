"""In-memory indexes for deterministic billing JSON lookups."""

from __future__ import annotations

import json
import threading
from pathlib import Path


class BillingDataStore:
    """Lazy-loaded CPT indexes backed by local JSON knowledge files."""

    def __init__(self, json_dir: Path):
        self.json_dir = json_dir
        self._lock = threading.Lock()
        self._loaded = False
        self._mue: dict[str, dict] = {}
        self._ptp: dict[str, dict] = {}
        self._icd10: dict[str, dict] = {}
        self._aoc: dict[str, dict] = {}
        self._general: dict[str, dict] = {}
        self._knowledge: dict[str, dict] = {}

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            self._mue = self._index_file("cpt_mue_info.json")
            self._ptp = self._index_file("cpt_ptp_info.json")
            self._icd10 = self._index_file("cpt_icd10_info.json")
            self._aoc = self._index_file("cpt_aoc_info.json")
            self._general = self._index_file("cpt_general_info.json")
            self._knowledge = self._index_lookup_file("medexa_cpt_lookup.json")
            self._loaded = True

    def _index_file(self, filename: str) -> dict[str, dict]:
        path = self.json_dir / filename
        if not path.exists():
            return {}
        with path.open(encoding="utf-8") as handle:
            records = json.load(handle)
        indexed: dict[str, dict] = {}
        for record in records:
            code = str(record.get("cpt_code", "")).strip()
            if code:
                indexed[code] = record
        return indexed

    def _index_lookup_file(self, filename: str) -> dict[str, dict]:
        path = self.json_dir / filename
        if not path.exists():
            return {}
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            return {}
        indexed: dict[str, dict] = {}
        for key, record in data.items():
            if key.startswith("_") or not isinstance(record, dict):
                continue
            code = str(record.get("code") or key).strip()
            if code:
                indexed[code] = record
        return indexed

    def get_mue(self, cpt_code: str) -> dict | None:
        self.ensure_loaded()
        return self._mue.get(cpt_code)

    def get_ptp(self, cpt_code: str) -> dict | None:
        self.ensure_loaded()
        return self._ptp.get(cpt_code)

    def get_icd10(self, cpt_code: str) -> dict | None:
        self.ensure_loaded()
        return self._icd10.get(cpt_code)

    def get_aoc(self, cpt_code: str) -> dict | None:
        self.ensure_loaded()
        return self._aoc.get(cpt_code)

    def get_general(self, cpt_code: str) -> dict | None:
        self.ensure_loaded()
        return self._general.get(cpt_code)

    def get_knowledge(self, cpt_code: str) -> dict | None:
        self.ensure_loaded()
        return self._knowledge.get(cpt_code)
