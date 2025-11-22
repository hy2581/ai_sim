#!/usr/bin/env python3
import json
from pathlib import Path
import argparse

def gen_cpu(i):
    arch = "SPARC V8" if i % 2 == 0 else "PowerPC"
    cores = int(1 + (i % 8))
    return {
        "name": f"LEON Variant {i:03d}",
        "type": "rad-hard CPU",
        "category": "cpu",
        "parameters": {
            "architecture": arch,
            "cores": cores,
            "frequency_mhz": int(150 + (i % 200)),
            "radiation_tolerance_krad": int(100 + (i % 50)),
            "rtos_support": True,
            "power_w": float(3 + (i % 12) * 0.3),
            "temperature_range": "-55..125"
        },
        "real_world_equivalent": (
            "Cobham Gaisler GR740 (LEON4/LEON5 class)" if arch == "SPARC V8" and cores >= 4 else
            "Cobham Gaisler GR712RC (LEON3 dual-core)" if arch == "SPARC V8" else
            "BAE Systems RAD5545" if cores >= 4 else
            "BAE Systems RAD750"
        ),
        "interconnect_effects": [
            "AMBA bus" if i % 2 == 0 else "PCIe",
            "shared memory",
            "thermal management"
        ]
    }

def gen_gpu(i):
    fp = int(8 + (i % 40))
    return {
        "name": f"DSP/GPU Core {i:03d}",
        "type": "compute accelerator",
        "category": "gpu",
        "parameters": {
            "fp32_gflops": fp,
            "memory_bandwidth_gbps": int(5 + (i % 12)),
            "rad_hard": True,
            "shared_memory": True,
            "temperature_range": "-55..125"
        },
        "real_world_equivalent": (
            "Xilinx Virtex-5QV (Space-grade FPGA)" if fp >= 40 else
            "Microchip RTG4 (Space-grade FPGA)"
        ),
        "interconnect_effects": [
            "shared memory",
            "heat dissipation",
            "sensor high-throughput"
        ]
    }

def gen_memory_sram(i):
    return {
        "name": f"Rad-Hard SRAM {i:03d}",
        "type": "memory",
        "category": "memory",
        "parameters": {
            "capacity_gb": float(1 + (i % 4) * 0.5),
            "edac": True,
            "seu_protection": True,
            "bandwidth_gbps": float(4 + (i % 6) * 0.5),
            "temperature_range": "-55..125"
        },
        "real_world_equivalent": "Teledyne e2v Rad-Hard SRAM (64–128Mb)",
        "interconnect_effects": [
            "close to CPU",
            "3D stacked",
            "signal integrity"
        ]
    }

def gen_memory_mram(i):
    return {
        "name": f"Rad-Hard MRAM {i:03d}",
        "type": "memory",
        "category": "memory",
        "parameters": {
            "capacity_gb": float(0.5 + (i % 4) * 0.5),
            "non_volatile": True,
            "edac": True,
            "temperature_range": "-55..125"
        },
        "real_world_equivalent": "Avalanche Technology Space MRAM",
        "interconnect_effects": [
            "non-volatile storage",
            "low-latency",
            "EMC considerations"
        ]
    }

def gen_storage_nand(i):
    return {
        "name": f"Rad-Hard NAND Flash {i:03d}",
        "type": "storage",
        "category": "storage",
        "parameters": {
            "capacity_gb": int(16 + (i % 8) * 16),
            "endurance_cycles": int(100000 + (i % 5) * 10000),
            "data_retention_years": int(10 + (i % 5)),
            "temperature_range": "-55..125"
        },
        "real_world_equivalent": "Micron SLC NAND (industrial, radiation-tolerant)",
        "interconnect_effects": [
            "reliable interface",
            "space constraints",
            "power integrity"
        ]
    }

def gen_storage_nor(i):
    return {
        "name": f"Rad-Hard NOR Flash {i:03d}",
        "type": "storage",
        "category": "storage",
        "parameters": {
            "capacity_gb": int(8 + (i % 8) * 4),
            "read_latency_ns": int(90 - (i % 5) * 5),
            "temperature_range": "-55..125"
        },
        "real_world_equivalent": "Infineon/Spansion S29GL NOR (rad-tolerant)",
        "interconnect_effects": [
            "code storage",
            "signal integrity",
            "EMC considerations"
        ]
    }

def gen_sensor_raman(i):
    return {
        "name": f"Micro Laser Raman Probe {i:03d}",
        "type": "spectrometer",
        "category": "sensors",
        "parameters": {
            "resolution_cm_minus1": float(1 - min(0.5, (i % 5) * 0.1)),
            "miniaturized": True,
            "power_w": float(2 + (i % 5) * 0.2)
        },
        "real_world_equivalent": "Ocean Insight Raman Probe (micro)",
        "interconnect_effects": [
            "ADC precision",
            "EMC shielding",
            "low-noise cabling"
        ]
    }

def gen_sensor_hyperspec(i):
    return {
        "name": f"Hyperspectral Imager {i:03d}",
        "type": "imager",
        "category": "sensors",
        "parameters": {
            "bands_count": int(150 + (i % 50)),
            "spatial_resolution_um": float(max(4, 10 - (i % 5))),
            "power_w": float(4 + (i % 5) * 0.5)
        },
        "real_world_equivalent": "Headwall Hyperspec VNIR / Resonon Pika L",
        "interconnect_effects": [
            "high-throughput data",
            "thermal management",
            "synchronization"
        ]
    }

def gen_comm_spacewire(i):
    return {
        "name": f"SpaceWire Controller (Rad-Hard) {i:03d}",
        "type": "communication",
        "category": "communication",
        "parameters": {
            "data_rate_mbps": int(200 + (i % 10) * 50),
            "redundancy": True,
            "emc": True
        },
        "real_world_equivalent": "Cobham Gaisler GRSPW2/GRSPW4",
        "interconnect_effects": [
            "low-latency link",
            "signal integrity",
            "power integrity"
        ]
    }

def gen_comm_spacefibre(i):
    return {
        "name": f"SpaceFibre Controller {i:03d}",
        "type": "communication",
        "category": "communication",
        "parameters": {
            "data_rate_gbps": float(2.5 + (i % 10) * 0.5),
            "redundancy": True,
            "emc": True
        },
        "real_world_equivalent": "STAR-Dundee SpaceFibre Interface",
        "interconnect_effects": [
            "high-bandwidth",
            "signal integrity",
            "power integrity"
        ]
    }

def gen_comm_can(i):
    return {
        "name": f"CAN Bus Controller (Rad-Hard) {i:03d}",
        "type": "communication",
        "category": "communication",
        "parameters": {
            "data_rate_kbps": int(1000 + (i % 10) * 250),
            "redundancy": True,
            "emc": True
        },
        "real_world_equivalent": "Microchip ATA6560 / MCP2515",
        "interconnect_effects": [
            "bus arbitration",
            "EMC",
            "redundant topology"
        ]
    }

def gen_controller(i, kind, base):
    name_map = {
        "piezo": "Piezoelectric Drill Controller",
        "pump": "MEMS Pump/Valve Driver",
        "arm": "Robotic Arm Motion Controller",
        "thermal": "Thermal Control Unit",
        "brush": "Brush Sweeper Controller"
    }
    params = {
        "piezo": {"precision_nm": 100, "response_ms": 1, "drive_voltage_v": 100},
        "pump": {"response_ms": 1, "closed_loop": True, "drive_voltage_v": 12},
        "arm": {"closed_loop": True, "redundancy": True, "response_ms": 1},
        "thermal": {"sensor_channels": 8, "response_ms": 10, "power_w": 5},
        "brush": {"precision_nm": 500, "response_ms": 2, "drive_voltage_v": 24}
    }
    p = params[kind].copy()
    for k in p:
        if isinstance(p[k], (int, float)):
            p[k] = p[k] + (i % base)
    return {
        "name": f"{name_map[kind]} {i:03d}",
        "type": "controller",
        "category": "controllers",
        "parameters": p,
        "real_world_equivalent": (
            "PI E-727 Piezo Controller" if kind == "piezo" else
            "Bartels Mikrotechnik mp6 Driver" if kind == "pump" else
            "Maxon EPOS4 Motion Controller" if kind == "arm" else
            "Lakeshore Model 336" if kind == "thermal" else
            "Maxon EPOS4 / Custom Brush Controller"
        ),
        "interconnect_effects": [
            "real-time control",
            "power integrity",
            "EMC"
        ]
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", type=int, default=120)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    cpus = []
    gpus = []
    memory = []
    storage = []
    sensors = []
    communication = []
    controllers = []

    for i in range(1, args.variants + 1):
        cpus.append(gen_cpu(i))
        gpus.append(gen_gpu(i))
        memory.append(gen_memory_sram(i))
        memory.append(gen_memory_mram(i))
        storage.append(gen_storage_nand(i))
        storage.append(gen_storage_nor(i))
        sensors.append(gen_sensor_raman(i))
        sensors.append(gen_sensor_hyperspec(i))
        communication.append(gen_comm_spacewire(i))
        communication.append(gen_comm_spacefibre(i))
        communication.append(gen_comm_can(i))
        controllers.append(gen_controller(i, "piezo", 5))
        controllers.append(gen_controller(i, "pump", 3))
        controllers.append(gen_controller(i, "arm", 2))
        controllers.append(gen_controller(i, "thermal", 4))
        controllers.append(gen_controller(i, "brush", 6))

    device_library = {
        "device_library": {
            "cpus": cpus,
            "gpus": gpus,
            "memory": memory,
            "storage": storage,
            "sensors": sensors,
            "communication": communication,
            "controllers": controllers
        }
    }

    out_path = Path(args.output) if args.output else (Path(__file__).parent.parent / "device_library.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(device_library, f, ensure_ascii=False, indent=2)
    print(f"Wrote device library to {out_path}")

if __name__ == "__main__":
    main()