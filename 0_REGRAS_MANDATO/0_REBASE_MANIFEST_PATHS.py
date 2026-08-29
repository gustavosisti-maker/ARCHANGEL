# -*- coding: utf-8 -*-
"""Rebaseia caminhos absolutos dos manifests ARCHANGEL após mover o projeto."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_JSON_DIR = PROJECT_ROOT / "0_REGRAS_MANDATO" / "BASE_JSON"
LEGACY_ROOT = r"C:\1_ARCHANGEL"

MANIFEST_NAMES = (
    "ARCHANGEL_MACHINE_PROFILE.json",
    "BASE_ARQUIVOS.json",
    "MAPA_ATIVOS.json",
    "3_JSON_FEATURES.json",
    "4_JSON_LABELS.json",
)


def rebase_paths(value: Any) -> tuple[Any, int]:
    if isinstance(value, str):
        replacements = value.count(LEGACY_ROOT)
        return value.replace(LEGACY_ROOT, str(PROJECT_ROOT)), replacements

    if isinstance(value, list):
        rebuilt: list[Any] = []
        total = 0
        for item in value:
            item_rebased, item_count = rebase_paths(item)
            rebuilt.append(item_rebased)
            total += item_count
        return rebuilt, total

    if isinstance(value, dict):
        rebuilt: dict[str, Any] = {}
        total = 0
        for key, item in value.items():
            item_rebased, item_count = rebase_paths(item)
            rebuilt[key] = item_rebased
            total += item_count
        return rebuilt, total

    return value, 0


def write_json_atomic(path: Path, payload: Any) -> None:
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
    os.replace(temp_path, path)


def main() -> None:
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"LEGACY_ROOT:  {LEGACY_ROOT}")

    for name in MANIFEST_NAMES:
        path = BASE_JSON_DIR / name
        if not path.is_file():
            print(f"[SKIP] {name}: não encontrado")
            continue

        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        payload, replacements = rebase_paths(payload)
        if replacements:
            write_json_atomic(path, payload)
            print(f"[UPDATED] {name}: {replacements} caminho(s) migrado(s)")
        else:
            print(f"[OK] {name}: nenhum caminho legado")


if __name__ == "__main__":
    main()
