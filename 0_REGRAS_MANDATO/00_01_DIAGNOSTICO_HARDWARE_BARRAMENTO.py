# -*- coding: utf-8 -*-
"""
ARCHANGEL v1 - Machine Profile AI Friendly

Arquivo sugerido:
    <PROJECT_ROOT>\\0_REGRAS_MANDATO\\00_01_DIAGNOSTICO_HARDWARE_BARRAMENTO.py

Única saída:
    <PROJECT_ROOT>\\0_REGRAS_MANDATO\\BASE_JSON\\00_01_MACHINE_PROFILE_LATEST.json

Objetivo:
    Criar um único JSON completo, estruturado e amigável para IA descrevendo
    a máquina atual, com foco em otimização de trading systems, backtests,
    walk-forward analysis, ML/DL, CUDA/GPU e processamento paralelo.

Foco:
    - Perfil computacional resumido para IA
    - Recomendações de uso de CPU/RAM/GPU
    - CPU / threads / cores
    - RAM
    - GPU / CUDA / NVIDIA quando disponível
    - Discos / storage
    - Barramentos PCI/USB
    - Drivers
    - Sistema operacional
    - Pacotes Python relevantes
    - Diagnósticos brutos opcionais dentro do mesmo JSON

Observação:
    Este JSON pode conter dados sensíveis como hostname, usuário, serial number,
    MAC address e IPs. Use localmente no ARCHANGEL ou sanitize antes de compartilhar.
"""

from __future__ import annotations

import os
import sys
import json
import socket
import shutil
import platform
import subprocess
import importlib.util
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

# A raiz é inferida do local deste script, sem depender da letra da unidade.
ROOT_DIR = Path(__file__).resolve().parent.parent
BASE_JSON_DIR = ROOT_DIR / "0_REGRAS_MANDATO" / "BASE_JSON"

OUTPUT_JSON_NAME = "00_01_MACHINE_PROFILE_LATEST.json"
OUTPUT_JSON_PATH = BASE_JSON_DIR / OUTPUT_JSON_NAME

POWERSHELL_TIMEOUT_SECONDS = 45

# Mantém o JSON completo.
# Se no futuro o arquivo ficar grande demais, mude para False.
ENABLE_RAW_TEXT_DIAGNOSTICS = True

# Mantém lista completa de drivers.
# Em algumas máquinas isso pode gerar JSON grande, mas é mais completo para IA.
INCLUDE_FULL_DRIVER_LIST = True

# Coleta dispositivos USB/PCI detalhados.
INCLUDE_BUS_DEVICES = True

# Coleta drivers PnP assinados.
INCLUDE_DRIVERS = True


# =============================================================================
# UTILITÁRIOS GERAIS
# =============================================================================

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def now_local_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        return str(value)
    except Exception:
        return repr(value)


def bytes_to_gb(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return round(float(value) / (1024 ** 3), 2)
    except Exception:
        return None


def kb_to_gb(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return round(float(value) / (1024 ** 2), 2)
    except Exception:
        return None


def safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def run_command(command: List[str], timeout: int = POWERSHELL_TIMEOUT_SECONDS) -> Dict[str, Any]:
    """
    Executa comando no sistema e retorna stdout/stderr/código.
    """
    result = {
        "command": command,
        "ok": False,
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "error": None,
    }

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            shell=False,
        )

        result["returncode"] = completed.returncode
        result["stdout"] = completed.stdout.strip()
        result["stderr"] = completed.stderr.strip()
        result["ok"] = completed.returncode == 0

    except subprocess.TimeoutExpired as e:
        result["error"] = f"Timeout após {timeout}s: {e}"

    except Exception as e:
        result["error"] = str(e)

    return result


def run_powershell_json(script: str, timeout: int = POWERSHELL_TIMEOUT_SECONDS) -> Dict[str, Any]:
    """
    Executa PowerShell e tenta converter a saída JSON para objeto Python.
    """
    command = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        script,
    ]

    raw = run_command(command, timeout=timeout)

    output = {
        "ok": raw["ok"],
        "data": None,
        "raw": raw,
        "parse_error": None,
    }

    if not raw["stdout"]:
        return output

    try:
        output["data"] = json.loads(raw["stdout"])
    except Exception as e:
        output["parse_error"] = str(e)

    return output


def ps_get_cim_json(class_name: str, properties: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Consulta uma classe CIM/WMI e retorna JSON.
    """
    if properties:
        props = ", ".join(properties)
        ps = (
            f"Get-CimInstance -ClassName {class_name} | "
            f"Select-Object {props} | "
            f"ConvertTo-Json -Depth 8 -Compress"
        )
    else:
        ps = (
            f"Get-CimInstance -ClassName {class_name} | "
            f"ConvertTo-Json -Depth 8 -Compress"
        )

    return run_powershell_json(ps)


def normalize_ps_data(data: Any) -> List[Dict[str, Any]]:
    """
    PowerShell pode retornar objeto único ou lista. Aqui padroniza para lista.
    """
    if data is None:
        return []

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        return [data]

    return [{"value": data}]


def compact_raw_status(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Status compacto de execução PowerShell.
    """
    raw = response.get("raw", {})
    return {
        "ok": bool(response.get("ok")) and response.get("parse_error") is None,
        "returncode": raw.get("returncode"),
        "stderr": raw.get("stderr"),
        "error": raw.get("error"),
        "parse_error": response.get("parse_error"),
    }


# =============================================================================
# COLETORES BÁSICOS
# =============================================================================

def collect_python_runtime() -> Dict[str, Any]:
    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "python_version_info": {
            "major": sys.version_info.major,
            "minor": sys.version_info.minor,
            "micro": sys.version_info.micro,
        },
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "architecture": platform.architecture(),
        "sys_path_first_entries": sys.path[:10],
    }


def collect_python_packages() -> Dict[str, Any]:
    """
    Coleta disponibilidade e versão de pacotes relevantes para backtesting,
    otimização, ML/DL e aceleração.
    """
    packages = [
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "numba",
        "torch",
        "cupy",
        "pyarrow",
        "fastparquet",
        "ta",
        "talib",
        "joblib",
        "polars",
        "dask",
        "ray",
        "optuna",
        "skopt",
        "statsmodels",
        "xgboost",
        "lightgbm",
        "catboost",
        "tensorflow",
    ]

    result: Dict[str, Any] = {}

    for pkg in packages:
        item = {
            "installed": False,
            "version": None,
            "cuda_available": None,
            "extra": {},
        }

        try:
            spec = importlib.util.find_spec(pkg)
            item["installed"] = spec is not None

            if spec is not None:
                module = __import__(pkg)
                item["version"] = getattr(module, "__version__", None)

                if pkg == "torch":
                    try:
                        item["cuda_available"] = bool(module.cuda.is_available())
                        item["extra"]["cuda_device_count"] = int(module.cuda.device_count())
                        if module.cuda.is_available():
                            item["extra"]["cuda_device_name_0"] = module.cuda.get_device_name(0)
                            item["extra"]["torch_cuda_version"] = getattr(module.version, "cuda", None)
                            item["extra"]["cudnn_available"] = bool(module.backends.cudnn.is_available())
                            item["extra"]["cudnn_version"] = module.backends.cudnn.version()
                    except Exception as exc:
                        item["extra"]["torch_cuda_error"] = str(exc)

                if pkg == "cupy":
                    try:
                        item["cuda_available"] = True
                        item["extra"]["cupy_cuda_runtime_version"] = str(module.cuda.runtime.runtimeGetVersion())
                        item["extra"]["cupy_device_count"] = int(module.cuda.runtime.getDeviceCount())
                    except Exception as exc:
                        item["cuda_available"] = False
                        item["extra"]["cupy_cuda_error"] = str(exc)

                if pkg == "numba":
                    try:
                        from numba import cuda
                        item["cuda_available"] = bool(cuda.is_available())
                    except Exception as exc:
                        item["extra"]["numba_cuda_error"] = str(exc)

                if pkg == "tensorflow":
                    try:
                        item["extra"]["tensorflow_gpu_devices"] = [
                            str(x) for x in module.config.list_physical_devices("GPU")
                        ]
                        item["cuda_available"] = len(item["extra"]["tensorflow_gpu_devices"]) > 0
                    except Exception as exc:
                        item["extra"]["tensorflow_gpu_error"] = str(exc)

        except Exception as exc:
            item["error"] = str(exc)

        result[pkg] = item

    return result


def collect_basic_system_info() -> Dict[str, Any]:
    info = {
        "hostname": socket.gethostname(),
        "fqdn": socket.getfqdn(),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "edition": None,
        },
        "environment": {
            "username": os.environ.get("USERNAME") or os.environ.get("USER"),
            "userdomain": os.environ.get("USERDOMAIN"),
            "computername": os.environ.get("COMPUTERNAME"),
            "processor_architecture": os.environ.get("PROCESSOR_ARCHITECTURE"),
            "processor_identifier": os.environ.get("PROCESSOR_IDENTIFIER"),
            "number_of_processors_env": os.environ.get("NUMBER_OF_PROCESSORS"),
        },
    }

    ps = run_powershell_json(
        "Get-ComputerInfo | "
        "Select-Object WindowsProductName,WindowsVersion,OsHardwareAbstractionLayer,"
        "OsArchitecture,OsBuildNumber,OsInstallDate,CsDomain,CsWorkgroup,CsPartOfDomain | "
        "ConvertTo-Json -Depth 6 -Compress",
        timeout=45,
    )

    info["windows_computer_info"] = {
        "items": normalize_ps_data(ps.get("data")),
        "raw_status": compact_raw_status(ps),
    }

    return info


# =============================================================================
# CIM / WMI
# =============================================================================

def collect_cim_inventory() -> Dict[str, Any]:
    """
    Coleta classes importantes do Windows via CIM/WMI.
    """
    classes = {
        "computer_system": {
            "class": "Win32_ComputerSystem",
            "properties": [
                "Manufacturer",
                "Model",
                "SystemType",
                "SystemFamily",
                "TotalPhysicalMemory",
                "NumberOfProcessors",
                "NumberOfLogicalProcessors",
                "HypervisorPresent",
                "Domain",
                "Workgroup",
                "PartOfDomain",
            ],
        },
        "bios": {
            "class": "Win32_BIOS",
            "properties": [
                "Manufacturer",
                "Name",
                "SMBIOSBIOSVersion",
                "Version",
                "ReleaseDate",
                "SerialNumber",
            ],
        },
        "baseboard": {
            "class": "Win32_BaseBoard",
            "properties": [
                "Manufacturer",
                "Product",
                "Version",
                "SerialNumber",
            ],
        },
        "processor": {
            "class": "Win32_Processor",
            "properties": [
                "Name",
                "Manufacturer",
                "Description",
                "Architecture",
                "NumberOfCores",
                "NumberOfLogicalProcessors",
                "MaxClockSpeed",
                "CurrentClockSpeed",
                "L2CacheSize",
                "L3CacheSize",
                "SocketDesignation",
                "ProcessorId",
                "VirtualizationFirmwareEnabled",
                "SecondLevelAddressTranslationExtensions",
                "VMMonitorModeExtensions",
            ],
        },
        "physical_memory": {
            "class": "Win32_PhysicalMemory",
            "properties": [
                "BankLabel",
                "DeviceLocator",
                "Manufacturer",
                "PartNumber",
                "SerialNumber",
                "Capacity",
                "Speed",
                "ConfiguredClockSpeed",
                "MemoryType",
                "SMBIOSMemoryType",
                "FormFactor",
                "DataWidth",
                "TotalWidth",
            ],
        },
        "memory_array": {
            "class": "Win32_PhysicalMemoryArray",
            "properties": [
                "MemoryDevices",
                "MaxCapacity",
                "MaxCapacityEx",
            ],
        },
        "disk_drive": {
            "class": "Win32_DiskDrive",
            "properties": [
                "Model",
                "Manufacturer",
                "SerialNumber",
                "InterfaceType",
                "MediaType",
                "Size",
                "Partitions",
                "BytesPerSector",
                "FirmwareRevision",
                "PNPDeviceID",
                "Status",
            ],
        },
        "logical_disk": {
            "class": "Win32_LogicalDisk",
            "properties": [
                "DeviceID",
                "VolumeName",
                "FileSystem",
                "DriveType",
                "Size",
                "FreeSpace",
                "ProviderName",
            ],
        },
        "disk_partition": {
            "class": "Win32_DiskPartition",
            "properties": [
                "DiskIndex",
                "Index",
                "Name",
                "Type",
                "Size",
                "StartingOffset",
                "BootPartition",
                "PrimaryPartition",
            ],
        },
        "video_controller": {
            "class": "Win32_VideoController",
            "properties": [
                "Name",
                "VideoProcessor",
                "AdapterRAM",
                "DriverVersion",
                "DriverDate",
                "PNPDeviceID",
                "CurrentHorizontalResolution",
                "CurrentVerticalResolution",
                "CurrentRefreshRate",
                "Status",
            ],
        },
        "network_adapter": {
            "class": "Win32_NetworkAdapter",
            "properties": [
                "Name",
                "Manufacturer",
                "AdapterType",
                "MACAddress",
                "NetConnectionID",
                "NetEnabled",
                "Speed",
                "PNPDeviceID",
                "PhysicalAdapter",
                "Status",
            ],
        },
        "network_adapter_config": {
            "class": "Win32_NetworkAdapterConfiguration",
            "properties": [
                "Description",
                "MACAddress",
                "DHCPEnabled",
                "IPAddress",
                "IPSubnet",
                "DefaultIPGateway",
                "DNSServerSearchOrder",
            ],
        },
        "operating_system": {
            "class": "Win32_OperatingSystem",
            "properties": [
                "Caption",
                "Version",
                "BuildNumber",
                "OSArchitecture",
                "InstallDate",
                "LastBootUpTime",
                "TotalVisibleMemorySize",
                "FreePhysicalMemory",
                "TotalVirtualMemorySize",
                "FreeVirtualMemory",
                "SerialNumber",
            ],
        },
        "pagefile": {
            "class": "Win32_PageFileUsage",
            "properties": [
                "Name",
                "AllocatedBaseSize",
                "CurrentUsage",
                "PeakUsage",
            ],
        },
    }

    inventory = {}

    for key, spec in classes.items():
        response = ps_get_cim_json(spec["class"], spec["properties"])

        inventory[key] = {
            "source_class": spec["class"],
            "description": f"Coleta CIM/WMI da classe {spec['class']}.",
            "ok": response["ok"] and response["parse_error"] is None,
            "items": normalize_ps_data(response["data"]),
            "raw_status": compact_raw_status(response),
        }

    return inventory


# =============================================================================
# GPU / CUDA / NVIDIA
# =============================================================================

def collect_nvidia_smi() -> Dict[str, Any]:
    """
    Coleta dados de GPU NVIDIA via nvidia-smi, se disponível.
    """
    nvidia_smi_path = shutil.which("nvidia-smi")

    result = {
        "available": nvidia_smi_path is not None,
        "path": nvidia_smi_path,
        "ok": False,
        "items": [],
        "raw": None,
        "notes": [],
    }

    if nvidia_smi_path is None:
        result["notes"].append("nvidia-smi não encontrado no PATH.")
        return result

    query = (
        "name,driver_version,memory.total,memory.free,memory.used,"
        "temperature.gpu,utilization.gpu,utilization.memory,"
        "compute_cap,power.limit,power.draw"
    )

    command = [
        nvidia_smi_path,
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ]

    raw = run_command(command, timeout=20)
    result["raw"] = raw
    result["ok"] = raw["ok"]

    if not raw["ok"] or not raw["stdout"]:
        return result

    fields = [
        "name",
        "driver_version",
        "memory_total_mb",
        "memory_free_mb",
        "memory_used_mb",
        "temperature_gpu_c",
        "utilization_gpu_pct",
        "utilization_memory_pct",
        "compute_capability",
        "power_limit_w",
        "power_draw_w",
    ]

    for line in raw["stdout"].splitlines():
        parts = [x.strip() for x in line.split(",")]
        item = {}

        for idx, field in enumerate(fields):
            value = parts[idx] if idx < len(parts) else None

            if field.endswith("_mb") or field.endswith("_pct") or field.endswith("_c") or field.endswith("_w"):
                item[field] = safe_float(value)
            else:
                item[field] = value

        result["items"].append(item)

    return result


# =============================================================================
# BARRAMENTOS, STORAGE, DRIVERS
# =============================================================================

def collect_pci_devices() -> Dict[str, Any]:
    if not INCLUDE_BUS_DEVICES:
        return {
            "enabled": False,
            "description": "Coleta de dispositivos PCI/PCIe desativada.",
            "items": [],
        }

    ps = r"""
    Get-PnpDevice -PresentOnly |
        Where-Object { $_.InstanceId -like 'PCI*' } |
        Select-Object Class, FriendlyName, InstanceId, Manufacturer, Status, Problem, ConfigManagerErrorCode |
        Sort-Object Class, FriendlyName |
        ConvertTo-Json -Depth 8 -Compress
    """

    response = run_powershell_json(ps, timeout=45)

    return {
        "enabled": True,
        "description": "Dispositivos presentes no barramento PCI/PCIe via Get-PnpDevice.",
        "ok": response["ok"] and response["parse_error"] is None,
        "items": normalize_ps_data(response["data"]),
        "raw_status": compact_raw_status(response),
    }


def collect_usb_devices() -> Dict[str, Any]:
    if not INCLUDE_BUS_DEVICES:
        return {
            "enabled": False,
            "description": "Coleta de dispositivos USB desativada.",
            "items": [],
        }

    ps = r"""
    Get-PnpDevice -PresentOnly |
        Where-Object { $_.InstanceId -like 'USB*' } |
        Select-Object Class, FriendlyName, InstanceId, Manufacturer, Status, Problem, ConfigManagerErrorCode |
        Sort-Object Class, FriendlyName |
        ConvertTo-Json -Depth 8 -Compress
    """

    response = run_powershell_json(ps, timeout=45)

    return {
        "enabled": True,
        "description": "Dispositivos presentes no barramento USB via Get-PnpDevice.",
        "ok": response["ok"] and response["parse_error"] is None,
        "items": normalize_ps_data(response["data"]),
        "raw_status": compact_raw_status(response),
    }


def collect_storage_advanced() -> Dict[str, Any]:
    scripts = {
        "get_disk": r"""
            Get-Disk |
                Select-Object Number,FriendlyName,SerialNumber,HealthStatus,OperationalStatus,PartitionStyle,
                BusType,MediaType,Size,AllocatedSize,LogicalSectorSize,PhysicalSectorSize,FirmwareVersion,
                IsBoot,IsSystem,IsReadOnly,IsOffline |
                ConvertTo-Json -Depth 8 -Compress
        """,
        "get_physical_disk": r"""
            Get-PhysicalDisk |
                Select-Object FriendlyName,SerialNumber,MediaType,BusType,HealthStatus,OperationalStatus,
                Size,AllocatedSize,LogicalSectorSize,PhysicalSectorSize,FirmwareVersion,SpindleSpeed |
                ConvertTo-Json -Depth 8 -Compress
        """,
        "get_volume": r"""
            Get-Volume |
                Select-Object DriveLetter,FileSystemLabel,FileSystem,DriveType,HealthStatus,OperationalStatus,
                Size,SizeRemaining,Path |
                ConvertTo-Json -Depth 8 -Compress
        """,
    }

    output = {}

    for key, ps in scripts.items():
        response = run_powershell_json(ps, timeout=45)
        output[key] = {
            "ok": response["ok"] and response["parse_error"] is None,
            "items": normalize_ps_data(response["data"]),
            "raw_status": compact_raw_status(response),
        }

    return output


def collect_pnp_problem_devices() -> Dict[str, Any]:
    ps = r"""
    Get-PnpDevice |
        Where-Object { $_.Status -ne 'OK' } |
        Select-Object Class, FriendlyName, InstanceId, Manufacturer, Status, Problem, ConfigManagerErrorCode |
        Sort-Object Status, Class, FriendlyName |
        ConvertTo-Json -Depth 8 -Compress
    """

    response = run_powershell_json(ps, timeout=45)

    return {
        "description": "Dispositivos cujo status não está OK no PnP/Gerenciador de Dispositivos.",
        "ok": response["ok"] and response["parse_error"] is None,
        "items": normalize_ps_data(response["data"]),
        "raw_status": compact_raw_status(response),
    }


def collect_drivers() -> Dict[str, Any]:
    if not INCLUDE_DRIVERS:
        return {
            "enabled": False,
            "description": "Coleta de drivers desativada.",
            "total": 0,
            "summary_by_device_class": {},
            "items": [],
        }

    response = ps_get_cim_json(
        "Win32_PnPSignedDriver",
        [
            "DeviceName",
            "Manufacturer",
            "DriverProviderName",
            "DriverVersion",
            "DriverDate",
            "DeviceClass",
            "InfName",
            "IsSigned",
            "Signer",
        ],
    )

    items = normalize_ps_data(response["data"])

    summary_by_class = {}
    for item in items:
        cls = safe_str(item.get("DeviceClass") or "UNKNOWN")
        summary_by_class[cls] = summary_by_class.get(cls, 0) + 1

    payload = {
        "enabled": True,
        "description": "Drivers PnP assinados registrados no Windows.",
        "ok": response["ok"] and response["parse_error"] is None,
        "total": len(items),
        "summary_by_device_class": dict(sorted(summary_by_class.items())),
        "raw_status": compact_raw_status(response),
    }

    payload["items"] = items if INCLUDE_FULL_DRIVER_LIST else []

    return payload


# =============================================================================
# DIAGNÓSTICOS BRUTOS OPCIONAIS
# =============================================================================

def collect_network_ipconfig() -> Dict[str, Any]:
    raw = run_command(["ipconfig", "/all"], timeout=30)

    return {
        "description": "Saída bruta do ipconfig /all.",
        "ok": raw["ok"],
        "stdout": raw["stdout"],
        "stderr": raw["stderr"],
        "returncode": raw["returncode"],
        "error": raw["error"],
    }


def collect_systeminfo_raw() -> Dict[str, Any]:
    raw = run_command(["systeminfo"], timeout=60)

    return {
        "description": "Saída bruta do comando systeminfo.",
        "ok": raw["ok"],
        "stdout": raw["stdout"],
        "stderr": raw["stderr"],
        "returncode": raw["returncode"],
        "error": raw["error"],
    }


def collect_powercfg_battery_report_info() -> Dict[str, Any]:
    raw = run_command(["powercfg", "/a"], timeout=30)

    return {
        "description": "Estados de energia/suspensão suportados pela máquina.",
        "ok": raw["ok"],
        "stdout": raw["stdout"],
        "stderr": raw["stderr"],
        "returncode": raw["returncode"],
        "error": raw["error"],
    }


def collect_raw_text_diagnostics() -> Dict[str, Any]:
    if not ENABLE_RAW_TEXT_DIAGNOSTICS:
        return {
            "enabled": False,
            "notes": "Diagnósticos brutos desativados para reduzir tamanho e tempo de execução.",
        }

    return {
        "enabled": True,
        "description": (
            "Diagnósticos brutos textuais. Úteis para auditoria humana e para IA "
            "quando campos estruturados não forem suficientes."
        ),
        "systeminfo": collect_systeminfo_raw(),
        "ipconfig_all": collect_network_ipconfig(),
        "powercfg_available_sleep_states": collect_powercfg_battery_report_info(),
    }


# =============================================================================
# EXTRAÇÃO DE RESUMOS
# =============================================================================

def get_first_item(block: Dict[str, Any]) -> Dict[str, Any]:
    items = block.get("items", [])
    if isinstance(items, list) and items:
        return items[0]
    return {}


def extract_total_ram_from_cim(cim: Dict[str, Any]) -> Optional[int]:
    computer = get_first_item(cim.get("computer_system", {}))
    value = safe_int(computer.get("TotalPhysicalMemory"))

    if value is not None:
        return value

    total = 0
    for item in cim.get("physical_memory", {}).get("items", []):
        total += safe_int(item.get("Capacity"), 0) or 0

    return total or None


def summarize_storage(storage_advanced: Dict[str, Any], cim: Dict[str, Any]) -> Dict[str, Any]:
    disks = storage_advanced.get("get_disk", {}).get("items", [])
    volumes = storage_advanced.get("get_volume", {}).get("items", [])
    disk_drive = cim.get("disk_drive", {}).get("items", [])

    total_disk_bytes = 0
    total_free_bytes = 0

    disk_bus_types = {}
    disk_media_types = {}

    for d in disks:
        total_disk_bytes += safe_int(d.get("Size"), 0) or 0
        bus = safe_str(d.get("BusType") or "UNKNOWN")
        media = safe_str(d.get("MediaType") or "UNKNOWN")
        disk_bus_types[bus] = disk_bus_types.get(bus, 0) + 1
        disk_media_types[media] = disk_media_types.get(media, 0) + 1

    for v in volumes:
        total_free_bytes += safe_int(v.get("SizeRemaining"), 0) or 0

    return {
        "disk_count_get_disk": len(disks),
        "disk_count_win32_diskdrive": len(disk_drive),
        "volume_count": len(volumes),
        "total_disk_size_bytes_detected": total_disk_bytes or None,
        "total_disk_size_gb_detected": bytes_to_gb(total_disk_bytes) if total_disk_bytes else None,
        "total_volume_free_bytes_detected": total_free_bytes or None,
        "total_volume_free_gb_detected": bytes_to_gb(total_free_bytes) if total_free_bytes else None,
        "disk_bus_types": dict(sorted(disk_bus_types.items())),
        "disk_media_types": dict(sorted(disk_media_types.items())),
    }


def summarize_python_stack(python_packages: Dict[str, Any]) -> Dict[str, Any]:
    installed = [
        name for name, data in python_packages.items()
        if isinstance(data, dict) and data.get("installed")
    ]

    missing = [
        name for name, data in python_packages.items()
        if isinstance(data, dict) and not data.get("installed")
    ]

    cuda_related = {
        "torch_cuda_available": python_packages.get("torch", {}).get("cuda_available"),
        "cupy_cuda_available": python_packages.get("cupy", {}).get("cuda_available"),
        "numba_cuda_available": python_packages.get("numba", {}).get("cuda_available"),
        "tensorflow_cuda_available": python_packages.get("tensorflow", {}).get("cuda_available"),
    }

    return {
        "installed_packages": installed,
        "missing_packages": missing,
        "cuda_related": cuda_related,
        "core_stack_ready": {
            "numpy": python_packages.get("numpy", {}).get("installed"),
            "pandas": python_packages.get("pandas", {}).get("installed"),
            "pyarrow": python_packages.get("pyarrow", {}).get("installed"),
            "scipy": python_packages.get("scipy", {}).get("installed"),
            "sklearn": python_packages.get("sklearn", {}).get("installed"),
            "numba": python_packages.get("numba", {}).get("installed"),
            "optuna": python_packages.get("optuna", {}).get("installed"),
        },
    }


# =============================================================================
# PERFIL PARA IA / OTIMIZAÇÃO
# =============================================================================

def build_ai_compute_profile(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cria resumo compacto para IA decidir como usar a máquina em otimizações.
    """

    cim = data.get("hardware_inventory", {}).get("cim_wmi", {})
    computer_items = cim.get("computer_system", {}).get("items", [])
    processor_items = cim.get("processor", {}).get("items", [])
    memory_items = cim.get("physical_memory", {}).get("items", [])

    total_ram_bytes = extract_total_ram_from_cim(cim)
    logical_processors = None
    physical_cores = None
    cpu_name = None

    if computer_items:
        logical_processors = safe_int(computer_items[0].get("NumberOfLogicalProcessors"))

    if processor_items:
        cpu_name = processor_items[0].get("Name")
        physical_cores = sum(
            safe_int(x.get("NumberOfCores"), 0) or 0
            for x in processor_items
        ) or safe_int(processor_items[0].get("NumberOfCores"))

        if logical_processors is None:
            logical_processors = sum(
                safe_int(x.get("NumberOfLogicalProcessors"), 0) or 0
                for x in processor_items
            ) or None

    ram_modules = len(memory_items)
    total_ram_gb = bytes_to_gb(total_ram_bytes)

    python_packages = data.get("software_environment", {}).get("python_packages", {})
    nvidia = data.get("hardware_inventory", {}).get("gpu_acceleration", {}).get("nvidia_smi", {})

    torch_cuda = python_packages.get("torch", {}).get("cuda_available")
    cupy_cuda = python_packages.get("cupy", {}).get("cuda_available")
    numba_cuda = python_packages.get("numba", {}).get("cuda_available")
    tensorflow_cuda = python_packages.get("tensorflow", {}).get("cuda_available")

    cuda_likely_available = bool(
        nvidia.get("available") and nvidia.get("items")
    ) or bool(torch_cuda) or bool(cupy_cuda) or bool(numba_cuda) or bool(tensorflow_cuda)

    nvidia_gpu_count = len(nvidia.get("items", [])) if isinstance(nvidia.get("items"), list) else 0

    try:
        lp = int(logical_processors or os.cpu_count() or 1)
    except Exception:
        lp = os.cpu_count() or 1

    # Conservador: deixa 1-2 threads livres.
    if lp >= 16:
        recommended_cpu_workers = max(1, lp - 2)
    elif lp >= 8:
        recommended_cpu_workers = max(1, lp - 1)
    else:
        recommended_cpu_workers = max(1, lp)

    if total_ram_gb is not None and total_ram_gb < 16:
        recommended_parallel_jobs = max(1, min(2, recommended_cpu_workers))
    elif total_ram_gb is not None and total_ram_gb < 32:
        recommended_parallel_jobs = max(1, min(4, recommended_cpu_workers))
    elif total_ram_gb is not None and total_ram_gb < 64:
        recommended_parallel_jobs = max(1, min(6, recommended_cpu_workers))
    else:
        recommended_parallel_jobs = max(1, min(8, recommended_cpu_workers))

    storage_summary = data.get("ai_friendly_summary", {}).get("storage_summary", {})
    python_stack_summary = data.get("ai_friendly_summary", {}).get("python_stack_summary", {})

    gpu_memory_total_mb = None
    if nvidia.get("items"):
        try:
            gpu_memory_total_mb = sum(
                safe_float(g.get("memory_total_mb"), 0.0) or 0.0
                for g in nvidia.get("items", [])
            )
        except Exception:
            gpu_memory_total_mb = None

    profile = {
        "purpose": (
            "Resumo para IA e módulos ARCHANGEL decidirem uso de CPU, RAM, GPU/CUDA "
            "e paralelização em backtests, otimizações, walk-forward, ML e DL."
        ),
        "cpu": {
            "name": cpu_name,
            "physical_cores_detected": physical_cores,
            "logical_processors_detected": logical_processors,
            "os_cpu_count": os.cpu_count(),
            "recommended_cpu_workers": recommended_cpu_workers,
        },
        "memory": {
            "total_ram_bytes": total_ram_bytes,
            "total_ram_gb": total_ram_gb,
            "ram_modules_detected": ram_modules,
        },
        "gpu_cuda": {
            "cuda_likely_available": cuda_likely_available,
            "nvidia_smi_available": nvidia.get("available"),
            "nvidia_gpu_count": nvidia_gpu_count,
            "gpu_memory_total_mb_detected": gpu_memory_total_mb,
            "torch_cuda_available": torch_cuda,
            "cupy_cuda_available": cupy_cuda,
            "numba_cuda_available": numba_cuda,
            "tensorflow_cuda_available": tensorflow_cuda,
            "nvidia_gpus": nvidia.get("items", []),
        },
        "storage": storage_summary,
        "python_stack": python_stack_summary,
        "optimization_recommendations": {
            "recommended_parallel_jobs_initial": recommended_parallel_jobs,
            "recommended_cpu_workers_per_job_if_single_job": recommended_cpu_workers,
            "use_vectorized_pandas_numpy": True,
            "use_numba_when_available": bool(python_packages.get("numba", {}).get("installed")),
            "use_gpu_for_deep_learning_if_available": cuda_likely_available,
            "use_gpu_for_backtest_only_if_large_matrix_or_ml": cuda_likely_available,
            "prefer_parquet_pyarrow_for_large_ohlcv": bool(python_packages.get("pyarrow", {}).get("installed")),
            "suggested_backtest_parallelization_axis": [
                "asset",
                "timeframe",
                "parameter_grid_chunk",
                "walk_forward_window",
            ],
            "avoid_excessive_workers_note": (
                "Não usar todos os threads se o processo também faz leitura pesada de Parquet, "
                "otimização com muitos parâmetros ou treino ML. Começar conservador e medir."
            ),
        },
        "trading_system_specific_guidance": {
            "backtests": (
                "Priorizar cálculo vetorial com pandas/numpy. Paralelizar por ativo, timeframe "
                "ou bloco de parâmetros. Evitar paralelizar candles linha a linha."
            ),
            "optimization": (
                "Usar Optuna/Ray/Dask/joblib se disponíveis. Em grids grandes, dividir por chunks "
                "e salvar checkpoints para evitar perda de progresso."
            ),
            "machine_learning": (
                "Usar CPU para feature engineering tabular e GPU para deep learning ou modelos "
                "com matrizes grandes quando CUDA estiver disponível."
            ),
            "data_loading": (
                "Usar Parquet com pyarrow quando disponível. Evitar reler arquivos a cada iteração "
                "do mesmo teste; preferir cache controlado por memória."
            ),
        },
    }

    return profile


# =============================================================================
# COLETA GERAL
# =============================================================================

def collect_all() -> Dict[str, Any]:
    """
    Coleta todos os blocos e monta um único JSON AI friendly.
    """

    python_runtime = collect_python_runtime()
    python_packages = collect_python_packages()
    basic_system_info = collect_basic_system_info()
    cim_wmi = collect_cim_inventory()
    gpu_acceleration = {
        "nvidia_smi": collect_nvidia_smi(),
    }
    storage_advanced = collect_storage_advanced()

    hardware_inventory = {
        "cim_wmi": cim_wmi,
        "gpu_acceleration": gpu_acceleration,
        "storage_advanced": storage_advanced,
        "bus_inventory": {
            "pci_devices": collect_pci_devices(),
            "usb_devices": collect_usb_devices(),
        },
        "pnp_problem_devices": collect_pnp_problem_devices(),
        "drivers": collect_drivers(),
    }

    software_environment = {
        "python_runtime": python_runtime,
        "python_packages": python_packages,
        "basic_system_info": basic_system_info,
    }

    ai_friendly_summary = {
        "python_stack_summary": summarize_python_stack(python_packages),
        "storage_summary": summarize_storage(storage_advanced, cim_wmi),
    }

    data = {
        "schema_version": "ARCHANGEL_MACHINE_PROFILE_AI_FRIENDLY_1.0",
        "generated_at_utc": now_utc_iso(),
        "generated_at_local": now_local_iso(),
        "file_identity": {
            "output_file_name": OUTPUT_JSON_NAME,
            "output_file_path": str(OUTPUT_JSON_PATH),
            "script_expected_location": str(BASE_JSON_DIR / "DIAGNOSTICO_HARDWARE_BARRAMENTO.py"),
            "root_dir": str(ROOT_DIR),
            "base_json_dir": str(BASE_JSON_DIR),
        },
        "purpose": (
            "Único JSON completo e AI friendly com descrição da máquina para orientar "
            "backtests, otimizações, walk-forward analysis, ML/DL, CUDA/GPU, uso de CPU/RAM "
            "e decisões de paralelização do sistema ARCHANGEL."
        ),
        "privacy_and_security": {
            "contains_sensitive_identifiers": True,
            "possible_sensitive_fields": [
                "hostname",
                "fqdn",
                "username",
                "userdomain",
                "computername",
                "serial_numbers",
                "mac_addresses",
                "ip_addresses",
                "disk_serials",
                "bios_serial",
                "baseboard_serial",
            ],
            "recommendation": (
                "Usar localmente no ARCHANGEL. Antes de compartilhar externamente, "
                "remover ou mascarar identificadores sensíveis."
            ),
        },
        "ai_reading_order": [
            "ai_compute_profile",
            "ai_friendly_summary",
            "software_environment.python_packages",
            "hardware_inventory.cim_wmi.processor",
            "hardware_inventory.cim_wmi.physical_memory",
            "hardware_inventory.gpu_acceleration",
            "hardware_inventory.storage_advanced",
            "hardware_inventory.pnp_problem_devices",
            "hardware_inventory.drivers.summary_by_device_class",
            "raw_text_diagnostics",
        ],
        "host": {
            "hostname": socket.gethostname(),
            "fqdn": socket.getfqdn(),
        },
        "software_environment": software_environment,
        "hardware_inventory": hardware_inventory,
        "ai_friendly_summary": ai_friendly_summary,
        "raw_text_diagnostics": collect_raw_text_diagnostics(),
    }

    data["ai_compute_profile"] = build_ai_compute_profile(data)

    # Resumo executivo no topo lógico do JSON.
    # Mantido também por redundância amigável para IA.
    data["executive_summary_for_ai"] = {
        "machine_role_suggestion": (
            "Máquina apta para backtests vetoriais e otimização paralela conforme recursos detectados. "
            "Usar GPU para ML/DL se CUDA estiver disponível."
        ),
        "recommended_first_use": [
            "Rodar backtests vetoriais por ativo/timeframe.",
            "Paralelizar otimização por chunks de parâmetros.",
            "Usar validação walk-forward com checkpoints.",
            "Reservar GPU para ML/DL e experimentos com matrizes grandes.",
        ],
        "most_important_fields": {
            "cpu": data["ai_compute_profile"]["cpu"],
            "memory": data["ai_compute_profile"]["memory"],
            "gpu_cuda": data["ai_compute_profile"]["gpu_cuda"],
            "optimization_recommendations": data["ai_compute_profile"]["optimization_recommendations"],
        },
    }

    return data


# =============================================================================
# SALVAMENTO ÚNICO
# =============================================================================

def save_single_json(data: Dict[str, Any]) -> Dict[str, str]:
    """
    Salva apenas um único JSON AI friendly.
    """
    ensure_dir(BASE_JSON_DIR)

    with OUTPUT_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    return {
        "json_path": str(OUTPUT_JSON_PATH),
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("=" * 90)
    print("ARCHANGEL | GERANDO ÚNICO MACHINE PROFILE JSON AI FRIENDLY")
    print("=" * 90)
    print(f"Python...........: {sys.executable}")
    print(f"Root ARCHANGEL...: {ROOT_DIR}")
    print(f"Saída JSON.......: {OUTPUT_JSON_PATH}")
    print("Coletando informações. Pode levar alguns segundos...")
    print("-" * 90)

    if not ROOT_DIR.exists():
        raise FileNotFoundError(f"Diretório raiz não encontrado: {ROOT_DIR}")

    ensure_dir(BASE_JSON_DIR)

    data = collect_all()
    paths = save_single_json(data)

    print("")
    print("=" * 90)
    print("MACHINE PROFILE JSON CONCLUÍDO")
    print("=" * 90)
    print(f"JSON salvo em: {paths['json_path']}")
    print("")
    print("Este é o único arquivo de saída e deve ser usado para alimentar IA.")
    print("Atenção: contém possíveis dados sensíveis como serial number, MAC, IP e hostname.")
    print("=" * 90)


if __name__ == "__main__":
    main()
