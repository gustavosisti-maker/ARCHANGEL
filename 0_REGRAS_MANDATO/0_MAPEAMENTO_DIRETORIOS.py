# -*- coding: utf-8 -*-
"""
ARCHANGEL v1 - MAPEAMENTO DE DIRETÓRIOS E ARQUIVOS

Arquivo:
    <PROJECT_ROOT>\\0_REGRAS_MANDATO\\0_MAPEAMENTO_DIRETORIOS.py

Saída:
    <PROJECT_ROOT>\\0_REGRAS_MANDATO\\BASE_JSON\\BASE_ARQUIVOS.json

Objetivo:
    Criar uma base JSON completa com:
        - Estrutura de diretórios dentro de <PROJECT_ROOT>
        - Lista de arquivos encontrados
        - Metadados básicos dos arquivos
        - Resumo por extensão
        - Resumo por diretório principal
        - Caminhos relevantes do sistema ARCHANGEL

Uso:
    python <PROJECT_ROOT>\\0_REGRAS_MANDATO\\0_MAPEAMENTO_DIRETORIOS.py

Observação:
    Este script é pensado para retroalimentação AI e para permitir que
    os módulos MAIN saibam onde estão arquivos, bases, indicadores e regras.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List


# =============================================================================
# 1. CONFIGURAÇÕES FIXAS DO SISTEMA
# =============================================================================

# A raiz é inferida do local deste script, sem depender da letra da unidade.
ROOT_DIR = Path(__file__).resolve().parent.parent

RULES_DIR = ROOT_DIR / "0_REGRAS_MANDATO"
BASE_JSON_DIR = RULES_DIR / "BASE_JSON"

OUTPUT_JSON_NAME = "BASE_ARQUIVOS.json"
OUTPUT_JSON_PATH = BASE_JSON_DIR / OUTPUT_JSON_NAME

SYSTEM_NAME = "ARCHANGEL"
SYSTEM_VERSION = "v1"
SCHEMA_VERSION = "ARCHANGEL_DIRECTORY_FILE_MAP_1.0"


# =============================================================================
# 2. CONFIGURAÇÕES DE VARREDURA
# =============================================================================

# Diretórios operacionais grandes ficam fora do inventário AI para evitar
# BASE_ARQUIVOS.json gigante e pouco útil para o Codex.
EXCLUDED_DIR_NAMES = {
    "__pycache__",
    "_CACHE",
    "_PYTHON",
    ".git",
    ".idea",
    ".vscode",
    ".pytest_cache",
    ".mypy_cache",
    ".ipynb_checkpoints",
}

# Arquivos temporários geralmente inúteis para inventário AI.
EXCLUDED_FILE_SUFFIXES = {
    ".tmp",
    ".temp",
    ".bak",
    ".old",
    ".lock",
}

# Tamanho máximo para tentar capturar preview textual.
# Neste primeiro mapa, deixamos sem preview de conteúdo para evitar JSON gigante.
ENABLE_TEXT_PREVIEW = False
MAX_TEXT_PREVIEW_CHARS = 1000

TEXT_EXTENSIONS = {
    ".py",
    ".json",
    ".txt",
    ".md",
    ".csv",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
    ".log",
}


# =============================================================================
# 3. FUNÇÕES UTILITÁRIAS
# =============================================================================

def now_iso() -> str:
    """Retorna timestamp local em ISO format."""
    return datetime.now().isoformat(timespec="seconds")


def ensure_dir(path: Path) -> Path:
    """Cria diretório se não existir."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def path_to_str(path: Path) -> str:
    """Converte Path para string Windows."""
    return str(path)


def safe_relative_path(path: Path, root: Path) -> str:
    """Retorna caminho relativo com tolerância a erro."""
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def get_file_extension(path: Path) -> str:
    """Retorna extensão em lowercase ou string vazia."""
    return path.suffix.lower()


def get_top_level_folder(path: Path, root: Path) -> str:
    """
    Retorna a primeira pasta abaixo do ROOT_DIR.

    Exemplo:
        <PROJECT_ROOT>\\4_LEADING_INDICATORS\\2_MACD_CROSS.py
        -> 4_LEADING_INDICATORS
    """
    try:
        rel = path.relative_to(root)
        parts = rel.parts
        if len(parts) > 0:
            return parts[0]
        return "."
    except Exception:
        return "UNKNOWN"


def file_size_mb(size_bytes: int) -> float:
    """Converte bytes para MB."""
    return round(size_bytes / (1024 * 1024), 6)


def should_exclude_dir(dir_path: Path) -> bool:
    """Define se um diretório deve ser ignorado."""
    return dir_path.name in EXCLUDED_DIR_NAMES


def should_exclude_file(file_path: Path) -> bool:
    """Define se um arquivo deve ser ignorado."""
    if file_path.name.startswith("~$"):
        return True

    if file_path.suffix.lower() in EXCLUDED_FILE_SUFFIXES:
        return True

    return False


def read_text_preview(file_path: Path) -> str | None:
    """
    Lê pequeno preview textual do arquivo, se habilitado.

    Por padrão fica desligado para evitar JSON grande.
    """
    if not ENABLE_TEXT_PREVIEW:
        return None

    if file_path.suffix.lower() not in TEXT_EXTENSIONS:
        return None

    try:
        with file_path.open("r", encoding="utf-8", errors="ignore") as f:
            return f.read(MAX_TEXT_PREVIEW_CHARS)
    except Exception:
        return None


# =============================================================================
# 4. MAPEAMENTO DE DIRETÓRIOS
# =============================================================================

def collect_directory_record(dir_path: Path, root: Path) -> Dict[str, Any]:
    """
    Cria registro de diretório.
    """
    try:
        stat = dir_path.stat()
        modified_at = datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
        created_at = datetime.fromtimestamp(stat.st_ctime).isoformat(timespec="seconds")
    except Exception:
        modified_at = None
        created_at = None

    relative_path = safe_relative_path(dir_path, root)
    top_level_folder = get_top_level_folder(dir_path, root)

    return {
        "name": dir_path.name,
        "absolute_path": path_to_str(dir_path),
        "relative_path": relative_path,
        "top_level_folder": top_level_folder,
        "created_at": created_at,
        "modified_at": modified_at,
    }


def build_directory_tree(root: Path) -> Dict[str, Any]:
    """
    Cria árvore hierárquica simplificada de diretórios.

    Estrutura:
        {
            "name": "1_ARCHANGEL",
            "path": "<PROJECT_ROOT>",
            "children": [...]
        }
    """

    def build_node(current: Path) -> Dict[str, Any]:
        node = {
            "name": current.name,
            "absolute_path": path_to_str(current),
            "relative_path": safe_relative_path(current, root),
            "children": [],
        }

        try:
            children_dirs = [
                p for p in current.iterdir()
                if p.is_dir() and not should_exclude_dir(p)
            ]
            children_dirs = sorted(children_dirs, key=lambda p: p.name.lower())

            for child in children_dirs:
                node["children"].append(build_node(child))

        except PermissionError:
            node["permission_error"] = True
        except Exception as exc:
            node["error"] = str(exc)

        return node

    return build_node(root)


# =============================================================================
# 5. MAPEAMENTO DE ARQUIVOS
# =============================================================================

def collect_file_record(file_path: Path, root: Path) -> Dict[str, Any]:
    """
    Cria registro de arquivo com metadados básicos.
    """
    try:
        stat = file_path.stat()
        size_bytes = stat.st_size
        modified_at = datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
        created_at = datetime.fromtimestamp(stat.st_ctime).isoformat(timespec="seconds")
    except Exception:
        size_bytes = None
        modified_at = None
        created_at = None

    extension = get_file_extension(file_path)
    relative_path = safe_relative_path(file_path, root)
    parent_dir = file_path.parent
    top_level_folder = get_top_level_folder(file_path, root)

    record = {
        "name": file_path.name,
        "stem": file_path.stem,
        "extension": extension,
        "absolute_path": path_to_str(file_path),
        "relative_path": relative_path,
        "parent_dir": path_to_str(parent_dir),
        "top_level_folder": top_level_folder,
        "size_bytes": size_bytes,
        "size_mb": None if size_bytes is None else file_size_mb(size_bytes),
        "created_at": created_at,
        "modified_at": modified_at,
        "is_text_candidate": extension in TEXT_EXTENSIONS,
    }

    preview = read_text_preview(file_path)
    if preview is not None:
        record["text_preview"] = preview

    return record


def scan_archangel(root: Path) -> Dict[str, Any]:
    """
    Varre todos os diretórios e arquivos dentro do ROOT_DIR.
    """
    directories: List[Dict[str, Any]] = []
    files: List[Dict[str, Any]] = []

    errors: List[Dict[str, str]] = []

    for current_dir, dir_names, file_names in os.walk(root):
        current_path = Path(current_dir)

        # Filtra diretórios excluídos in-place para o os.walk não entrar neles.
        dir_names[:] = [
            d for d in dir_names
            if not should_exclude_dir(current_path / d)
        ]

        try:
            directories.append(collect_directory_record(current_path, root))
        except Exception as exc:
            errors.append({
                "type": "directory_record_error",
                "path": path_to_str(current_path),
                "error": str(exc),
            })

        for file_name in file_names:
            file_path = current_path / file_name

            if should_exclude_file(file_path):
                continue

            try:
                files.append(collect_file_record(file_path, root))
            except Exception as exc:
                errors.append({
                    "type": "file_record_error",
                    "path": path_to_str(file_path),
                    "error": str(exc),
                })

    return {
        "directories": directories,
        "files": files,
        "errors": errors,
    }


# =============================================================================
# 6. RESUMOS E ÍNDICES
# =============================================================================

def build_extension_summary(files: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Cria resumo por extensão.
    """
    summary: Dict[str, Dict[str, Any]] = {}

    for item in files:
        ext = item.get("extension") or "[no_extension]"
        size_bytes = item.get("size_bytes") or 0

        if ext not in summary:
            summary[ext] = {
                "count": 0,
                "total_size_bytes": 0,
                "total_size_mb": 0.0,
            }

        summary[ext]["count"] += 1
        summary[ext]["total_size_bytes"] += size_bytes

    for ext, data in summary.items():
        data["total_size_mb"] = file_size_mb(data["total_size_bytes"])

    return dict(sorted(summary.items(), key=lambda kv: kv[0]))


def build_top_folder_summary(
    directories: List[Dict[str, Any]],
    files: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Cria resumo por pasta principal.
    """
    summary: Dict[str, Dict[str, Any]] = {}

    for d in directories:
        top = d.get("top_level_folder", "UNKNOWN")
        if top not in summary:
            summary[top] = {
                "directory_count": 0,
                "file_count": 0,
                "total_file_size_bytes": 0,
                "total_file_size_mb": 0.0,
            }
        summary[top]["directory_count"] += 1

    for f in files:
        top = f.get("top_level_folder", "UNKNOWN")
        size_bytes = f.get("size_bytes") or 0

        if top not in summary:
            summary[top] = {
                "directory_count": 0,
                "file_count": 0,
                "total_file_size_bytes": 0,
                "total_file_size_mb": 0.0,
            }

        summary[top]["file_count"] += 1
        summary[top]["total_file_size_bytes"] += size_bytes

    for top, data in summary.items():
        data["total_file_size_mb"] = file_size_mb(data["total_file_size_bytes"])

    return dict(sorted(summary.items(), key=lambda kv: kv[0]))


def build_known_archangel_paths(root: Path) -> Dict[str, str]:
    """
    Mapa dos caminhos principais do sistema.
    """
    return {
        "root_dir": path_to_str(root),
        "old_dir": path_to_str(root / "_old"),
        "rules_mandate_dir": path_to_str(root / "0_REGRAS_MANDATO"),
        "base_json_dir": path_to_str(root / "0_REGRAS_MANDATO" / "BASE_JSON"),
        "bases_dir": path_to_str(root / "2_BASES"),
        "features_dir": path_to_str(root / "3_FEATURES"),
        "labels_dir": path_to_str(root / "4_LABELS"),
        "datasets_ml_dir": path_to_str(root / "5_DATASETS_ML"),
        "experiments_dir": path_to_str(root / "6_EXPERIMENTS"),
    }


def build_quick_indexes(files: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Cria índices rápidos para uso por outros módulos.

    Importante:
        Esses índices ajudam o MAIN a localizar arquivos sem varrer tudo.
    """

    by_name: Dict[str, List[str]] = {}
    by_extension: Dict[str, List[str]] = {}
    by_top_level_folder: Dict[str, List[str]] = {}

    for f in files:
        name = f.get("name", "")
        ext = f.get("extension") or "[no_extension]"
        top = f.get("top_level_folder", "UNKNOWN")
        abs_path = f.get("absolute_path", "")

        by_name.setdefault(name, []).append(abs_path)
        by_extension.setdefault(ext, []).append(abs_path)
        by_top_level_folder.setdefault(top, []).append(abs_path)

    return {
        "by_name": by_name,
        "by_extension": by_extension,
        "by_top_level_folder": by_top_level_folder,
    }


# =============================================================================
# 7. PAYLOAD FINAL
# =============================================================================

def build_payload(root: Path) -> Dict[str, Any]:
    """
    Constrói JSON final de mapeamento.
    """

    scan = scan_archangel(root)

    directories = scan["directories"]
    files = scan["files"]
    errors = scan["errors"]

    total_size_bytes = sum(
        f.get("size_bytes") or 0
        for f in files
    )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "system": {
            "name": SYSTEM_NAME,
            "version": SYSTEM_VERSION,
            "generated_at": now_iso(),
            "purpose": (
                "Mapa completo de diretórios e arquivos do sistema ARCHANGEL "
                "para uso por módulos Python, backtests, AI agents e auditoria."
            ),
        },
        "root": {
            "absolute_path": path_to_str(root),
            "exists": root.exists(),
        },
        "known_archangel_paths": build_known_archangel_paths(root),
        "summary": {
            "total_directories": len(directories),
            "total_files": len(files),
            "total_size_bytes": total_size_bytes,
            "total_size_mb": file_size_mb(total_size_bytes),
            "total_errors": len(errors),
            "text_preview_enabled": ENABLE_TEXT_PREVIEW,
            "excluded_dir_names": sorted(EXCLUDED_DIR_NAMES),
        },
        "extension_summary": build_extension_summary(files),
        "top_level_folder_summary": build_top_folder_summary(directories, files),
        "directory_tree": build_directory_tree(root),
        "directories": directories,
        "files": files,
        "quick_indexes": build_quick_indexes(files),
        "errors": errors,
        "output": {
            "json_path": path_to_str(OUTPUT_JSON_PATH),
            "json_name": OUTPUT_JSON_NAME,
        },
        "usage_notes": {
            "main_backtest_usage": (
                "Os módulos MAIN podem ler este JSON para localizar indicadores, "
                "bases, regras, relatórios e arquivos auxiliares sem depender "
                "de estrutura colada manualmente no prompt."
            ),
            "recommended_read_code": (
                "import json\\n"
                "from pathlib import Path\\n"
                f"path = Path(r'{path_to_str(OUTPUT_JSON_PATH)}')\\n"
                "data = json.loads(path.read_text(encoding='utf-8'))"
            ),
        },
    }

    return payload


# =============================================================================
# 8. SALVAMENTO
# =============================================================================

def save_payload(payload: Dict[str, Any], output_path: Path) -> None:
    """
    Salva JSON final.
    """
    ensure_dir(output_path.parent)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=4, default=str)


# =============================================================================
# 9. MAIN
# =============================================================================

def main() -> None:
    """
    Executa mapeamento completo do ARCHANGEL.
    """

    print("=" * 80)
    print("ARCHANGEL v1 | MAPEAMENTO DE DIRETÓRIOS E ARQUIVOS")
    print("=" * 80)

    if not ROOT_DIR.exists():
        raise FileNotFoundError(f"ROOT_DIR não encontrado: {ROOT_DIR}")

    ensure_dir(BASE_JSON_DIR)

    print(f"[INFO] ROOT_DIR: {ROOT_DIR}")
    print(f"[INFO] BASE_JSON_DIR: {BASE_JSON_DIR}")
    print(f"[INFO] OUTPUT_JSON_PATH: {OUTPUT_JSON_PATH}")
    print("-" * 80)

    payload = build_payload(ROOT_DIR)
    save_payload(payload, OUTPUT_JSON_PATH)

    print("[DONE] Mapeamento concluído.")
    print(f"[OUTPUT] JSON salvo em: {OUTPUT_JSON_PATH}")
    print("-" * 80)
    print("[SUMMARY]")
    print(json.dumps(payload["summary"], indent=4, ensure_ascii=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
