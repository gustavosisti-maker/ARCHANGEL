#ETAPAS
#0_DIAGNOSTICO_HARDWARE_BARRAMENTO.py
#0_MAPEAMENTO_DIRETORIOS.py
#1_MAPA_ATIVOS.py#7
#2_BUSCA_DADOS.py
#0_AUDITA_QUALIDADE_DADOS.py
#3_GERA_FEATURES.py
#4_GERA_LABELS.py
#4B_GERA_LABELS_PANEL.py
#5_MONTA_DATASETS_ML.py
#6_WALK_FORWARD_TRAINING.py
#7_BACKTEST_PORTFOLIO.py
#8_EXPERIMENT_REGISTRY.py
#9_EXECUTION_MODULE.py

# -*- coding: utf-8 -*-
from __future__ import annotations
import json
import os
import re
import sys
import subprocess
from pathlib import Path
from datetime import datetime


# =============================================================================
# 1. CONFIGURAÇÃO
# =============================================================================

# Resolve a base a partir deste arquivo para que o orquestrador funcione
# independentemente da unidade ou do diretório em que o projeto foi instalado.
BASE_DIR = Path(__file__).resolve().parent
BASE_JSON_DIR = BASE_DIR / "BASE_JSON"
DEFAULT_ARCHANGEL_PYTHON_EXE = BASE_DIR.parent / "_PYTHON" / "Python314" / "python.exe"

STAGES = [
    {
        "script": "0_DIAGNOSTICO_HARDWARE_BARRAMENTO.py",
        "nome": "Diagnostico de hardware/barramento",
    },
    {
        "script": "0_GERA_PYTHON_ENVIRONMENT_JSON.py",
        "nome": "Ambiente Python/CUDA",
    },
    {
        "script": "0_MAPEAMENTO_DIRETORIOS.py",
        "nome": "Mapeamento de diretorios inicial",
    },
    {
        "script": "1_MAPA_ATIVOS.py",
        "nome": "Mapa de ativos inicial",
    },
    {
        "script": "2_BUSCA_DADOS.py",
        "nome": "Busca de dados",
    },
    {
        "script": "1_MAPA_ATIVOS.py",
        "nome": "Mapa de ativos pos-busca",
    },
    {
        "script": "0_AUDITA_QUALIDADE_DADOS.py",
        "nome": "Auditoria de qualidade dos dados",
    },
    {
        "script": "3_GERA_FEATURES.py",
        "nome": "Geracao de features",
    },
    {
        "script": "4_GERA_LABELS.py",
        "nome": "Geracao de labels",
    },
    {
        "script": "5_MONTA_DATASETS_ML.py",
        "nome": "Montagem dos datasets ML",
    },
    {
        "script": "6_WALK_FORWARD_TRAINING.py",
        "nome": "Walk-forward training",
    },
    {
        "script": "7_BACKTEST_PORTFOLIO.py",
        "nome": "Backtest de portfolio",
    },
    {
        "script": "8_EXPERIMENT_REGISTRY.py",
        "nome": "Experiment registry formal",
    },
    {
        "script": "0_MAPEAMENTO_DIRETORIOS.py",
        "nome": "Mapeamento de diretorios final",
    },
]

# Mantido para compatibilidade com qualquer uso externo simples deste arquivo.
SCRIPTS = [BASE_DIR / stage["script"] for stage in STAGES]



# True  = se um script falhar, para a sequência.
# False = continua mesmo que algum script falhe.
PARAR_SE_ERRO = True

# Mostra saída dos scripts em tempo real no terminal.
MOSTRAR_OUTPUT_EM_TEMPO_REAL = True


# =============================================================================
# 2. UTILITÁRIOS
# =============================================================================

def agora() -> str:
    return datetime.now().isoformat(timespec="seconds")


def linha(char: str = "=", tamanho: int = 100) -> None:
    print(char * tamanho)


def substituir_arquivo_windows_safe(tmp_path: Path, final_path: Path) -> None:
    try:
        os.replace(tmp_path, final_path)
    except PermissionError:
        final_path.write_text(tmp_path.read_text(encoding="utf-8"), encoding="utf-8")
        try:
            tmp_path.unlink()
        except OSError:
            pass


def formatar_duracao(inicio: datetime, fim: datetime) -> str:
    duracao = fim - inicio
    total_segundos = int(duracao.total_seconds())

    horas = total_segundos // 3600
    minutos = (total_segundos % 3600) // 60
    segundos = total_segundos % 60

    if horas > 0:
        return f"{horas}h {minutos:02d}m {segundos:02d}s"

    return f"{minutos}m {segundos:02d}s"


def resolver_python_executable() -> str:
    if DEFAULT_ARCHANGEL_PYTHON_EXE.is_file():
        return str(DEFAULT_ARCHANGEL_PYTHON_EXE)

    env_python = os.environ.get("ARCHANGEL_PYTHON_EXE")
    if env_python and Path(env_python).is_file():
        return str(Path(env_python))

    machine_profile_path = BASE_JSON_DIR / "ARCHANGEL_MACHINE_PROFILE.json"
    if machine_profile_path.is_file():
        try:
            with machine_profile_path.open("r", encoding="utf-8") as handle:
                profile = json.load(handle)
            python_exe = (
                profile.get("software_environment", {})
                .get("python_runtime", {})
                .get("python_executable")
            )
            if python_exe and Path(python_exe).is_file():
                return str(Path(python_exe))
        except Exception:
            pass

    return sys.executable


def montar_env_python() -> dict:
    env = os.environ.copy()
    pythonpath_atual = env.get("PYTHONPATH")
    paths = [str(BASE_DIR)]
    if pythonpath_atual:
        paths.append(pythonpath_atual)
    env["ARCHANGEL_PYTHON_EXE"] = resolver_python_executable()
    env.setdefault("CUPY_CACHE_DIR", str(BASE_DIR.parent / "_CACHE" / "cupy_kernel_cache"))
    env.setdefault("NUMBA_CACHE_DIR", str(BASE_DIR.parent / "_CACHE" / "numba_cache"))
    env["PYTHONUTF8"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def salvar_run_all_report(
    inicio: datetime,
    fim: datetime,
    stages_selecionadas: list[dict],
    resultados: list[dict],
    status: str,
) -> None:
    BASE_JSON_DIR.mkdir(parents=True, exist_ok=True)
    run_id = inicio.strftime("%Y%m%d_%H%M%S")
    payload = {
        "schema_version": "ARCHANGEL_RUN_ALL_REPORT_1.0",
        "system": {
            "name": "ARCHANGEL",
            "version": "v1",
            "layer": "00_RUN_ALL",
            "script": "00_RUN_ALL.py",
            "run_id": run_id,
            "generated_at": fim.isoformat(timespec="seconds"),
        },
        "paths": {
            "rules_dir": str(BASE_DIR),
            "base_json_dir": str(BASE_JSON_DIR),
            "run_report_path": str(BASE_JSON_DIR / "00_RUN_ALL_RUN_REPORT_LATEST.json"),
            "run_report_latest_path": str(BASE_JSON_DIR / "00_RUN_ALL_RUN_REPORT_LATEST.json"),
        },
        "config": {
            "parar_se_erro": PARAR_SE_ERRO,
            "mostrar_output_em_tempo_real": MOSTRAR_OUTPUT_EM_TEMPO_REAL,
            "python_executable": resolver_python_executable(),
            "launcher_python_executable": sys.executable,
        },
        "summary": {
            "status": status,
            "stages_selected": len(stages_selecionadas),
            "stages_executed": len(resultados),
            "stages_ok": sum(1 for item in resultados if item.get("returncode") == 0),
            "stages_error": sum(1 for item in resultados if item.get("returncode") != 0),
            "started_at": inicio.isoformat(timespec="seconds"),
            "finished_at": fim.isoformat(timespec="seconds"),
            "duration": formatar_duracao(inicio, fim),
        },
        "selected_stages": stages_selecionadas,
        "results": resultados,
    }
    latest_path = BASE_JSON_DIR / "00_RUN_ALL_RUN_REPORT_LATEST.json"
    tmp_latest_path = latest_path.with_suffix(latest_path.suffix + ".tmp")
    tmp_latest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    substituir_arquivo_windows_safe(tmp_latest_path, latest_path)


def carregar_json_base(nome: str) -> dict:
    path = BASE_JSON_DIR / nome
    if not path.is_file():
        return {"missing": True, "path": str(path)}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            return data
        return {"invalid_type": type(data).__name__, "path": str(path)}
    except Exception as exc:
        return {"error": str(exc), "path": str(path)}


def resumir_json_base(nome: str, papel: str, leitura: str) -> dict:
    path = BASE_JSON_DIR / nome
    data = carregar_json_base(nome)
    system = data.get("system") if isinstance(data.get("system"), dict) else {}
    summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
    return {
        "file": nome,
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "schema_version": data.get("schema_version"),
        "run_id": data.get("run_id") or system.get("run_id"),
        "generated_at": data.get("generated_at")
        or data.get("generated_at_utc")
        or system.get("generated_at")
        or system.get("generated_at_utc"),
        "role": papel,
        "ai_reading_hint": leitura,
        "summary": summary,
    }


def salvar_ai_context_index() -> None:
    BASE_JSON_DIR.mkdir(parents=True, exist_ok=True)
    machine = carregar_json_base("ARCHANGEL_MACHINE_PROFILE.json")
    python_env = carregar_json_base("ARCHANGEL_PYTHON_ENVIRONMENT.json")
    machine_profile = machine.get("ai_compute_profile", {}) if isinstance(machine, dict) else {}
    gpu_cuda = machine_profile.get("gpu_cuda", {}) if isinstance(machine_profile, dict) else {}
    python_stack = machine_profile.get("python_stack", {}) if isinstance(machine_profile, dict) else {}
    python_env_cuda = python_env.get("cuda", {}) if isinstance(python_env, dict) else {}
    python_env_summary = python_env.get("summary", {}) if isinstance(python_env, dict) else {}

    files = [
        resumir_json_base("00_RUN_ALL_RUN_REPORT_LATEST.json", "status geral do pipeline", "Comece por aqui para saber se a execução completa passou."),
        resumir_json_base("RUN_STATE.json", "estado incremental da coleta", "Use para entender run_count, última coleta e política incremental."),
        resumir_json_base("ARCHANGEL_MACHINE_PROFILE.json", "perfil de hardware e software", "Use para decidir CPU, RAM, GPU/CUDA e dependências disponíveis."),
        resumir_json_base("ARCHANGEL_PYTHON_ENVIRONMENT.json", "ambiente Python/CUDA", "Use para saber pacotes instalados, CUDA validado e próximos passos de migração."),
        resumir_json_base("BASE_ARQUIVOS.json", "inventário físico do projeto", "Use para localizar arquivos e diretórios do sistema."),
        resumir_json_base("CATALOGO_ARCHANGEL_SERIES.json", "catálogo da etapa de dados", "Use para entender séries baixadas e integridade inicial."),
        resumir_json_base("MAPA_ATIVOS.json", "mapa de ativos e universo", "Use para entender ativos, fontes, timeframes e arquivos Parquet."),
        resumir_json_base("DATA_QUALITY_REPORT.json", "qualidade dos dados", "Use antes de features/datasets para entender bloqueios e cautelas ML."),
        resumir_json_base("DATA_QUALITY_ROOT_CAUSE_REPORT.json", "triagem de causa-raiz", "Use para priorizar correções de qualidade."),
        resumir_json_base("3_JSON_FEATURES.json", "manifesto de features", "Use para entender feature store, famílias, outputs e governança anti-leakage."),
        resumir_json_base("3_FEATURES_RUN_REPORT_LATEST.json", "performance da geração de features", "Use para gargalos, memória, tempos por família e retry."),
        resumir_json_base("3_FEATURES_CUDA_BENCHMARK_LATEST.json", "benchmark CPU vs CUDA da etapa 3", "Use para decidir se a migração CUDA de features melhora performance e preserva equivalência numérica."),
        resumir_json_base("3_FEATURES_RETRY_PLAN_LATEST.json", "plano de retry de features", "Use para saber se há séries pendentes após a etapa 3."),
        resumir_json_base("4_JSON_LABELS.json", "manifesto de labels", "Use para entender targets, políticas anti-leakage e outputs."),
        resumir_json_base("4_LABELS_RUN_REPORT_LATEST.json", "performance da geração de labels", "Use para auditoria compacta e status por série."),
        resumir_json_base("5_JSON_DATASETS_ML.json", "manifesto dos datasets ML", "Use para entender datasets treináveis e colunas permitidas."),
        resumir_json_base("5_DATASETS_ML_RUN_REPORT_LATEST.json", "performance da montagem de datasets", "Use para status, linhas treináveis e gates ML."),
        resumir_json_base("6_JSON_WALK_FORWARD.json", "resultados walk-forward", "Use para métricas OOS, modelos e experimentos."),
        resumir_json_base("6_WALK_FORWARD_RUN_REPORT_LATEST.json", "performance do treino walk-forward", "Use para avaliar acurácia, erros e caminhos de modelos/predições."),
        resumir_json_base("7_JSON_BACKTEST_PORTFOLIO.json", "backtest de portfolio", "Use para avaliar PnL líquido, custos, stops, take profit, sizing, drawdown e meta anual."),
        resumir_json_base("7_BACKTEST_PORTFOLIO_RUN_REPORT_LATEST.json", "performance do backtest de portfolio", "Use para telemetria, tempos por fase, erros e caminhos de trades/equity."),
        resumir_json_base("7_BACKTEST_PARAM_SEARCH_LATEST.json", "busca de parametros do backtest", "Use para comparar thresholds, stops, take profit, long-only vs long-short e filtros de custo."),
        resumir_json_base("7_BACKTEST_PARAM_SEARCH_RUN_REPORT_LATEST.json", "performance da busca de parametros", "Use para auditar ranking, score e custo computacional da calibração da etapa 7."),
        resumir_json_base("7_BACKTEST_VALIDATION_LATEST.json", "validação automatica do backtest", "Use para verificar schema, trades, equity, custos, fill ratio e consistência temporal."),
        resumir_json_base("7_BACKTEST_STRESS_LATEST.json", "stress test do backtest", "Use para avaliar sensibilidade a custos, slippage, funding e piores regimes aproximados."),
        resumir_json_base("7_EXPERIMENT_REGISTRY_LATEST.json", "registry de experimentos", "Use para versionar dataset, previsões, custos, config, métricas, validação e stress."),
        resumir_json_base("8_JSON_EXPERIMENT_REGISTRY.json", "registry formal de experimentos", "Use como tabela mestre AI-friendly para rastrear dataset, features, labels, modelo, custos, risco, métricas e status."),
        resumir_json_base("8_EXPERIMENT_REGISTRY_LATEST.json", "registry formal latest", "Use como primeira fonte para comparar experimentos e entender linhagem completa."),
        resumir_json_base("8_EXPERIMENT_REGISTRY_RUN_REPORT_LATEST.json", "performance do registry formal", "Use para auditar geração do registry, caminhos persistidos e telemetria."),
        resumir_json_base("8_EXPERIMENT_REGISTRY_VALIDATION_LATEST.json", "validacao do registry formal", "Use para checar unicidade, artefatos, SQLite, Parquet e separacao entre pesquisa e aprovação."),
        resumir_json_base("COST_MODEL.json", "modelo de custos", "Use para interpretar labels/datasets líquidos e premissas de trading."),
    ]

    run_all = carregar_json_base("00_RUN_ALL_RUN_REPORT_LATEST.json")
    q = carregar_json_base("DATA_QUALITY_REPORT.json")
    features = carregar_json_base("3_JSON_FEATURES.json")
    labels = carregar_json_base("4_JSON_LABELS.json")
    datasets = carregar_json_base("5_JSON_DATASETS_ML.json")
    walk = carregar_json_base("6_JSON_WALK_FORWARD.json")
    backtest = carregar_json_base("7_JSON_BACKTEST_PORTFOLIO.json")
    backtest_search = carregar_json_base("7_BACKTEST_PARAM_SEARCH_LATEST.json")
    backtest_validation = carregar_json_base("7_BACKTEST_VALIDATION_LATEST.json")
    backtest_stress = carregar_json_base("7_BACKTEST_STRESS_LATEST.json")
    experiment_registry = carregar_json_base("7_EXPERIMENT_REGISTRY_LATEST.json")
    formal_experiment_registry = carregar_json_base("8_EXPERIMENT_REGISTRY_LATEST.json")
    formal_experiment_registry_validation = carregar_json_base("8_EXPERIMENT_REGISTRY_VALIDATION_LATEST.json")

    payload = {
        "schema_version": "ARCHANGEL_AI_CONTEXT_INDEX_1.0",
        "system": {
            "name": "ARCHANGEL",
            "version": "v1",
            "layer": "AI_CONTEXT",
            "script": "00_RUN_ALL.py",
            "generated_at": agora(),
        },
        "paths": {
            "project_root": str(BASE_DIR.parent),
            "rules_dir": str(BASE_DIR),
            "base_json_dir": str(BASE_JSON_DIR),
            "index_path": str(BASE_JSON_DIR / "ARCHANGEL_AI_CONTEXT_INDEX.json"),
            "python_executable": resolver_python_executable(),
        },
        "ai_reading_order": [
            "ARCHANGEL_AI_CONTEXT_INDEX.json",
            "00_RUN_ALL_RUN_REPORT_LATEST.json",
            "RUN_STATE.json",
            "ARCHANGEL_MACHINE_PROFILE.json",
            "ARCHANGEL_PYTHON_ENVIRONMENT.json",
            "MAPA_ATIVOS.json",
            "DATA_QUALITY_REPORT.json",
            "3_JSON_FEATURES.json",
            "3_FEATURES_RUN_REPORT_LATEST.json",
            "3_FEATURES_CUDA_BENCHMARK_LATEST.json",
            "4_JSON_LABELS.json",
            "5_JSON_DATASETS_ML.json",
            "6_JSON_WALK_FORWARD.json",
            "7_JSON_BACKTEST_PORTFOLIO.json",
            "7_BACKTEST_PARAM_SEARCH_LATEST.json",
            "7_BACKTEST_VALIDATION_LATEST.json",
            "7_BACKTEST_STRESS_LATEST.json",
            "7_EXPERIMENT_REGISTRY_LATEST.json",
            "8_EXPERIMENT_REGISTRY_LATEST.json",
            "8_EXPERIMENT_REGISTRY_VALIDATION_LATEST.json",
        ],
        "summary": {
            "pipeline_status": ((run_all.get("summary") or {}).get("status") if isinstance(run_all, dict) else None),
            "stages_ok": ((run_all.get("summary") or {}).get("stages_ok") if isinstance(run_all, dict) else None),
            "stages_error": ((run_all.get("summary") or {}).get("stages_error") if isinstance(run_all, dict) else None),
            "total_mapped_series": ((q.get("summary") or {}).get("total_series") if isinstance(q, dict) else None),
            "ml_ready_series": ((q.get("summary") or {}).get("ml_ready_series_count") if isinstance(q, dict) else None),
            "ml_caution_series": ((q.get("summary") or {}).get("ml_caution_series_count") if isinstance(q, dict) else None),
            "ml_blocked_series": ((q.get("summary") or {}).get("ml_blocked_series_count") if isinstance(q, dict) else None),
            "features_ok": ((features.get("summary") or {}).get("series_ok") if isinstance(features, dict) else None),
            "labels_ok": ((labels.get("summary") or {}).get("series_ok") if isinstance(labels, dict) else None),
            "datasets_ok": ((datasets.get("summary") or {}).get("datasets_ok") if isinstance(datasets, dict) else None),
            "datasets_trainable_for_broad_ml": ((datasets.get("summary") or {}).get("datasets_trainable_for_broad_ml") if isinstance(datasets, dict) else None),
            "walk_forward_experiments_ok": ((walk.get("summary") or {}).get("experiments_ok") if isinstance(walk, dict) else None),
            "walk_forward_avg_balanced_accuracy": ((walk.get("summary") or {}).get("avg_balanced_accuracy") if isinstance(walk, dict) else None),
            "backtest_portfolio_status": ((backtest.get("summary") or {}).get("portfolio_status") if isinstance(backtest, dict) else None),
            "backtest_portfolio_research_status": ((backtest.get("summary") or {}).get("portfolio_research_status") if isinstance(backtest, dict) else None),
            "backtest_portfolio_total_trades": ((backtest.get("summary") or {}).get("portfolio_total_trades") if isinstance(backtest, dict) else None),
            "backtest_portfolio_total_return": ((backtest.get("summary") or {}).get("portfolio_total_return") if isinstance(backtest, dict) else None),
            "backtest_portfolio_cagr": ((backtest.get("summary") or {}).get("portfolio_cagr") if isinstance(backtest, dict) else None),
            "backtest_portfolio_max_drawdown": ((backtest.get("summary") or {}).get("portfolio_max_drawdown") if isinstance(backtest, dict) else None),
            "backtest_reference_drawdown_limit": ((backtest.get("summary") or {}).get("reference_drawdown_limit") if isinstance(backtest, dict) else None),
            "backtest_approval_status": ((backtest.get("summary") or {}).get("approval_status") if isinstance(backtest, dict) else None),
            "backtest_param_search_candidates": ((backtest_search.get("summary") or {}).get("candidates_evaluated") if isinstance(backtest_search, dict) else None),
            "backtest_param_search_best_score": ((backtest_search.get("summary") or {}).get("best_score") if isinstance(backtest_search, dict) else None),
            "backtest_param_search_best_cagr": ((backtest_search.get("summary") or {}).get("best_cagr") if isinstance(backtest_search, dict) else None),
            "backtest_param_search_best_max_drawdown": ((backtest_search.get("summary") or {}).get("best_max_drawdown") if isinstance(backtest_search, dict) else None),
            "backtest_param_search_passes_research_references": ((backtest_search.get("summary") or {}).get("passes_research_references") if isinstance(backtest_search, dict) else None),
            "backtest_validation_status": ((backtest_validation.get("summary") or {}).get("status") if isinstance(backtest_validation, dict) else None),
            "backtest_stress_worst_scenario": ((backtest_stress.get("summary") or {}).get("worst_scenario") if isinstance(backtest_stress, dict) else None),
            "experiment_registry_latest_config_hash": ((experiment_registry.get("summary") or {}).get("latest_config_hash") if isinstance(experiment_registry, dict) else None),
            "formal_experiment_registry_status": ((formal_experiment_registry.get("summary") or {}).get("status") if isinstance(formal_experiment_registry, dict) else None),
            "formal_experiment_registry_rows": ((formal_experiment_registry.get("summary") or {}).get("registry_rows") if isinstance(formal_experiment_registry, dict) else None),
            "formal_experiment_registry_unique_experiments": ((formal_experiment_registry.get("summary") or {}).get("unique_experiments") if isinstance(formal_experiment_registry, dict) else None),
            "formal_experiment_registry_validation_status": ((formal_experiment_registry_validation.get("summary") or {}).get("status") if isinstance(formal_experiment_registry_validation, dict) else None),
            "formal_experiment_registry_sqlite_path": ((formal_experiment_registry.get("paths") or {}).get("registry_sqlite_path") if isinstance(formal_experiment_registry, dict) else None),
        },
        "compute_readiness": {
            "cpu": machine_profile.get("cpu") if isinstance(machine_profile, dict) else {},
            "memory": machine_profile.get("memory") if isinstance(machine_profile, dict) else {},
            "gpu_cuda": gpu_cuda,
            "python_stack": python_stack,
            "python_environment": python_env_summary,
            "python_cuda": python_env_cuda,
            "cuda_code_ready": bool(
                (
                    isinstance(python_env_cuda, dict)
                    and python_env_cuda.get("ready_for_code_migration")
                )
                or (
                    isinstance(gpu_cuda, dict)
                    and gpu_cuda.get("cuda_likely_available")
                    and any(gpu_cuda.get(name) for name in [
                        "torch_cuda_available",
                        "cupy_cuda_available",
                        "numba_cuda_available",
                        "tensorflow_cuda_available",
                    ])
                )
            ),
            "cuda_note": (
                "CUDA aparece no hardware via NVIDIA/nvidia-smi, mas o código só deve usar GPU "
                "quando torch/cupy/numba/tensorflow com CUDA estiverem instalados e validados."
            ),
        },
        "files": files,
        "recommendations_for_codex": [
            "Use os arquivos *_LATEST.json no BASE_JSON para estado atual; histórico por execução fica nos diretórios _logs.",
            "Use 5_JSON_DATASETS_ML.allowed_feature_columns como fonte autorizada de features para treino.",
            "Nunca use colunas label_, meta_ ou quality_ como features, salvo metadados explicitamente permitidos pelo manifesto.",
            "Trate ML_CAUTION_ACCEPTABLE como treinável com registro explícito e ML_BLOCKED como bloqueio.",
            "Consulte performance_summary antes de mudar paralelismo ou CUDA.",
            "Use backend CUDA por detecção de runtime, com fallback CPU, sem hard-code para a GPU atual.",
            "Antes de execução em testnet, valide 7_JSON_BACKTEST_PORTFOLIO.json para PnL líquido, custos, drawdown e sizing.",
            "Use 7_BACKTEST_PARAM_SEARCH_LATEST.json apenas como diagnóstico de sensibilidade; ele não aprova testnet.",
            "Use 7_BACKTEST_VALIDATION_LATEST.json, 7_BACKTEST_STRESS_LATEST.json e 7_EXPERIMENT_REGISTRY_LATEST.json para rastreabilidade da etapa 7.",
            "Use 8_EXPERIMENT_REGISTRY_LATEST.json como registry formal antes de ablação, comparação de hipóteses, retreino ou preparação de testnet.",
        ],
    }

    index_path = BASE_JSON_DIR / "ARCHANGEL_AI_CONTEXT_INDEX.json"
    tmp_path = index_path.with_suffix(index_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    substituir_arquivo_windows_safe(tmp_path, index_path)


def atualizar_roadmap_status() -> None:
    script = BASE_DIR / "0_ATUALIZA_ROADMAP_STATUS.py"
    if not script.is_file():
        print(f"[ROADMAP] Script nao encontrado: {script}")
        return
    try:
        processo = subprocess.run(
            [resolver_python_executable(), str(script)],
            cwd=str(BASE_DIR),
            env=montar_env_python(),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
        )
        if processo.stdout:
            print(processo.stdout)
        if processo.returncode != 0:
            print(f"[ROADMAP] Falha ao atualizar roadmap/status: returncode={processo.returncode}")
            if processo.stderr:
                print(processo.stderr)
    except Exception as exc:
        print(f"[ROADMAP] Falha ao atualizar roadmap/status: {exc}")


def caminho_stage(stage: dict) -> Path:
    return BASE_DIR / stage["script"]


def limpar_texto(texto: str) -> str:
    texto = texto.strip().lower()
    texto = texto.replace("á", "a").replace("à", "a").replace("ã", "a")
    texto = texto.replace("â", "a").replace("é", "e").replace("ê", "e")
    texto = texto.replace("í", "i").replace("ó", "o").replace("ô", "o")
    texto = texto.replace("õ", "o").replace("ú", "u").replace("ç", "c")
    return texto


def mostrar_menu_etapas() -> None:
    print("\nSELECAO DE ETAPAS")
    linha("-")
    print("Escolha ate onde ou quais etapas executar. A ordem original sempre sera preservada.\n")
    print("Seq | Parte | Script                         | Descricao")
    linha("-")

    for indice, stage in enumerate(STAGES, start=1):
        script = stage["script"]
        parte = script.split("_", 1)[0]
        print(f"{indice:>3} | {parte:<5} | {script:<30} | {stage['nome']}")

    linha("-")
    print("Atalhos:")
    print("  Enter ou tudo   = executa todas as etapas")
    print("  2               = executa em sequencia ate 2_BUSCA_DADOS.py")
    print("  3               = executa em sequencia ate 3_GERA_FEATURES.py")
    print("  #7              = executa em sequencia ate a sequencia 7 do menu")
    print("  lista #4,#7,#8  = executa somente essas sequencias, na ordem original")
    print("  4-8             = executa o intervalo de sequencias do menu")
    linha("-")


def resolver_indice_fim(token: str) -> int:
    """
    Resolve um alvo de parada para indice 1-based dentro de STAGES.
    Numeros simples priorizam a parte do pipeline: 2 -> 2_BUSCA_DADOS, 3 -> 3_GERA_FEATURES.
    Use #N quando quiser apontar diretamente para a sequencia N exibida no menu.
    """
    token = limpar_texto(token)
    token = token.removeprefix("ate ").strip()
    token = token.removeprefix("until ").strip()

    aliases = {
        "dados": "2_BUSCA_DADOS.py",
        "busca": "2_BUSCA_DADOS.py",
        "busca_dados": "2_BUSCA_DADOS.py",
        "features": "3_GERA_FEATURES.py",
        "gera_features": "3_GERA_FEATURES.py",
        "labels": "4_GERA_LABELS.py",
        "gera_labels": "4_GERA_LABELS.py",
        "datasets": "5_MONTA_DATASETS_ML.py",
        "ml": "5_MONTA_DATASETS_ML.py",
        "walk_forward": "6_WALK_FORWARD_TRAINING.py",
        "treino": "6_WALK_FORWARD_TRAINING.py",
        "backtest": "7_BACKTEST_PORTFOLIO.py",
        "portfolio": "7_BACKTEST_PORTFOLIO.py",
        "registry": "8_EXPERIMENT_REGISTRY.py",
        "experimentos": "8_EXPERIMENT_REGISTRY.py",
        "experiment_registry": "8_EXPERIMENT_REGISTRY.py",
    }

    if token.startswith("#") and token[1:].isdigit():
        indice = int(token[1:])
        if 1 <= indice <= len(STAGES):
            return indice
        raise ValueError(f"Sequencia fora do intervalo: {token}")

    alvo_script = aliases.get(token)

    if alvo_script is None and token.isdigit():
        prefixo = f"{token}_"
        candidatos = [
            indice
            for indice, stage in enumerate(STAGES, start=1)
            if stage["script"].startswith(prefixo)
        ]
        if candidatos:
            return candidatos[-1]

    if alvo_script is None:
        alvo_normalizado = token.replace(".py", "")
        candidatos = [
            indice
            for indice, stage in enumerate(STAGES, start=1)
            if alvo_normalizado in limpar_texto(stage["script"].replace(".py", ""))
        ]
        if candidatos:
            return candidatos[-1]

    if alvo_script is not None:
        candidatos = [
            indice
            for indice, stage in enumerate(STAGES, start=1)
            if stage["script"] == alvo_script
        ]
        if candidatos:
            return candidatos[-1]

    raise ValueError(f"Nao entendi a selecao: {token}")


def selecionar_etapas() -> list[dict]:
    if not sys.stdin.isatty():
        print("\n[SELECAO] Terminal nao interativo detectado. Executando todas as etapas.")
        return STAGES

    mostrar_menu_etapas()

    while True:
        try:
            escolha = input("\nDigite sua escolha: ").strip()
        except EOFError:
            print("\n[SELECAO] Entrada indisponivel. Executando todas as etapas.")
            return STAGES

        escolha_limpa = limpar_texto(escolha)

        if escolha_limpa in {"", "tudo", "todos", "all", "a"}:
            return STAGES

        try:
            if escolha_limpa.startswith("lista "):
                tokens = [
                    token.strip()
                    for token in re.split(r"[,; ]+", escolha_limpa.removeprefix("lista ").strip())
                    if token.strip()
                ]
                indices = sorted({resolver_indice_fim(token) for token in tokens})
                return [STAGES[indice - 1] for indice in indices]

            intervalo = re.fullmatch(r"#?(\d+)\s*-\s*#?(\d+)", escolha_limpa)
            if intervalo:
                inicio = int(intervalo.group(1))
                fim = int(intervalo.group(2))
                if inicio > fim:
                    inicio, fim = fim, inicio
                if not (1 <= inicio <= len(STAGES) and 1 <= fim <= len(STAGES)):
                    raise ValueError("Intervalo fora das sequencias exibidas no menu.")
                return STAGES[inicio - 1:fim]

            indice_fim = resolver_indice_fim(escolha_limpa)
            return STAGES[:indice_fim]

        except ValueError as erro:
            print(f"[ERRO] {erro}")
            print("Tente novamente. Exemplos validos: 2, 3, #7, tudo, 4-8, lista #4,#7.")


def validar_scripts(stages: list[dict]) -> bool:
    """
    Verifica se todos os scripts existem antes de iniciar.
    """
    print("\n[VALIDAÇÃO] Verificando existência dos scripts...\n")

    ok = True

    for stage in stages:
        script = caminho_stage(stage)
        if script.exists() and script.is_file():
            print(f"[OK] Encontrado: {script}")
        else:
            print(f"[ERRO] Arquivo não encontrado: {script}")
            ok = False

    return ok


def rodar_script(script_path: Path) -> int:
    """
    Executa um script Python como subprocesso.
    Retorna o código de saída.
    """
    linha()
    print(f"[INÍCIO] {agora()}")
    print(f"[SCRIPT] {script_path.name}")
    print(f"[CAMINHO] {script_path}")
    python_executable = resolver_python_executable()
    print(f"[PYTHON] {python_executable}")
    linha()

    inicio = datetime.now()

    comando = [
        python_executable,
        str(script_path),
    ]

    if MOSTRAR_OUTPUT_EM_TEMPO_REAL:
        processo = subprocess.run(
            comando,
            cwd=str(BASE_DIR),
            env=montar_env_python(),
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    else:
        processo = subprocess.run(
            comando,
            cwd=str(BASE_DIR),
            env=montar_env_python(),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
        )

        if processo.stdout:
            print("\n[STDOUT]")
            print(processo.stdout)

        if processo.stderr:
            print("\n[STDERR]")
            print(processo.stderr)

    fim = datetime.now()

    linha()
    print(f"[FIM] {agora()}")
    print(f"[SCRIPT] {script_path.name}")
    print(f"[CÓDIGO DE SAÍDA] {processo.returncode}")
    print(f"[DURAÇÃO] {formatar_duracao(inicio, fim)}")
    linha()

    return processo.returncode


# =============================================================================
# 3. MAIN
# =============================================================================

def main() -> int:
    inicio_geral = datetime.now()
    stages_selecionadas = selecionar_etapas()

    print("\nARCHANGEL v1 - ORQUESTRADOR GERAL")
    linha()
    print(f"Base: {BASE_DIR}")
    print(f"Python: {resolver_python_executable()}")
    print(f"Início geral: {agora()}")
    print(f"Parar se erro: {PARAR_SE_ERRO}")
    print(f"Etapas selecionadas: {len(stages_selecionadas)} de {len(STAGES)}")
    linha()

    print("\nPLANO DE EXECUCAO")
    linha("-")
    for indice_original, stage in enumerate(STAGES, start=1):
        if stage in stages_selecionadas:
            print(f"{indice_original:02d} | {stage['script']} | {stage['nome']}")
    linha("-")

    if not validar_scripts(stages_selecionadas):
        print("\n[ABORTADO] Um ou mais scripts não foram encontrados.")
        salvar_run_all_report(
            inicio_geral,
            datetime.now(),
            stages_selecionadas,
            [],
            "ABORTED_MISSING_SCRIPT",
        )
        return 1

    resultados = []

    for indice, stage in enumerate(stages_selecionadas, start=1):
        script = caminho_stage(stage)
        ordem_original = STAGES.index(stage) + 1

        print("\n")
        linha("-")
        print(
            f"[ETAPA {indice}/{len(stages_selecionadas)} | "
            f"ORDEM ORIGINAL {ordem_original}/{len(STAGES)}] Executando: {script.name}"
        )
        print(f"[DESCRICAO] {stage['nome']}")
        linha("-")

        codigo_saida = rodar_script(script)

        resultados.append({
            "ordem": indice,
            "ordem_original": ordem_original,
            "script": script.name,
            "nome": stage["nome"],
            "returncode": codigo_saida,
            "status": "OK" if codigo_saida == 0 else "ERRO",
        })

        if codigo_saida != 0 and PARAR_SE_ERRO:
            print(
                f"\n[PARADA] O script falhou e a execução sequencial foi interrompida: "
                f"{script.name}"
            )
            break

    fim_geral = datetime.now()

    print("\nRESUMO FINAL")
    linha()

    for item in resultados:
        print(
            f"{item['ordem']:02d}/{len(stages_selecionadas):02d} "
            f"(origem {item['ordem_original']:02d}) | "
            f"{item['status']:>5} | "
            f"returncode={item['returncode']} | "
            f"{item['script']} | "
            f"{item['nome']}"
        )

    linha()
    print(f"Início geral: {inicio_geral.isoformat(timespec='seconds')}")
    print(f"Fim geral:    {fim_geral.isoformat(timespec='seconds')}")
    print(f"Duração:      {formatar_duracao(inicio_geral, fim_geral)}")

    houve_erro = any(item["returncode"] != 0 for item in resultados)

    if houve_erro:
        print("\n[FINALIZADO COM ERRO] Verifique o script marcado como ERRO acima.")
        salvar_run_all_report(inicio_geral, fim_geral, stages_selecionadas, resultados, "ERROR")
        atualizar_roadmap_status()
        salvar_ai_context_index()
        return 1

    if len(resultados) < len(stages_selecionadas):
        print("\n[FINALIZADO PARCIALMENTE] Nem todos os scripts foram executados.")
        salvar_run_all_report(inicio_geral, fim_geral, stages_selecionadas, resultados, "PARTIAL")
        atualizar_roadmap_status()
        salvar_ai_context_index()
        return 1

    if len(stages_selecionadas) < len(STAGES):
        print("\n[FINALIZADO COM SUCESSO] Etapas selecionadas executadas em sequência.")
        salvar_run_all_report(inicio_geral, fim_geral, stages_selecionadas, resultados, "OK_SELECTED")
        atualizar_roadmap_status()
        salvar_ai_context_index()
        return 0

    print("\n[FINALIZADO COM SUCESSO] Todas as etapas foram executadas em sequência.")
    salvar_run_all_report(inicio_geral, fim_geral, stages_selecionadas, resultados, "OK_ALL")
    atualizar_roadmap_status()
    salvar_ai_context_index()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
