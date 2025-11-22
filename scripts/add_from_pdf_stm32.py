#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any
import argparse

# project path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Reuse DeepSeek client
from core.ai_integration.deepseek_api_client import DeepSeekAPIClient

PDF_PATH = Path("/home/hao123/chiplet/benchmark/ai_sim/STM32F103C8T6.pdf")
LIB_PATH = Path(__file__).parent.parent / "device_library.json"

SYSTEM_PROMPT = """
你是一个嵌入式器件解析专家。请从提供的PDF数据手册文本中提取STM32F103C8T6器件的关键信息，并输出JSON用于加入器件库。

输出字段要求：
- name: 器件名称，例如 "STM32F103C8T6"
- type: 器件类型，例如 "microcontroller"
- category: 固定为 "controllers" 或 "cpu" 按应用选择（此器件为微控制器，建议 "controllers"）
- parameters: 包含关键参数键值对，例如：
  - core: "ARM Cortex-M3"
  - frequency_mhz: 主频（如 72）
  - flash_kb: Flash容量（如 64）
  - sram_kb: SRAM容量（如 20）
  - peripherals: 主要外设列表（如 ["ADC", "USART", "SPI", "I2C", "Timers"]）
  - operating_voltage_v: 供电电压范围（如 "2.0–3.6"）
  - temperature_range: 工作温度范围（如 "-40..85"）
  - package: 封装（如 "LQFP48"）
- interconnect_effects: 列出系统集成影响（如总线、EMC、时钟树）
- real_world_equivalent: 若为现实器件，可重复其名称 "STM32F103C8T6"

JSON示例：
{
  "name": "STM32F103C8T6",
  "type": "microcontroller",
  "category": "controllers",
  "parameters": {
    "core": "ARM Cortex-M3",
    "frequency_mhz": 72,
    "flash_kb": 64,
    "sram_kb": 20,
    "peripherals": ["ADC", "USART", "SPI", "I2C", "Timers"],
    "operating_voltage_v": "2.0–3.6",
    "temperature_range": "-40..85",
    "package": "LQFP48"
  },
  "interconnect_effects": ["APB/AHB bus", "EMC", "clock tree", "power integrity"],
  "real_world_equivalent": "STM32F103C8T6"
}
"""

def read_pdf_text(pdf_path: Path) -> str:
    # Minimal PDF text extraction via pdftotext if available; fallback to raw bytes
    txt = ""
    try:
        import subprocess
        tmp_txt = pdf_path.with_suffix('.txt')
        subprocess.run(["pdftotext", str(pdf_path), str(tmp_txt)], check=True)
        if tmp_txt.exists():
            txt = tmp_txt.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        try:
            txt = pdf_path.read_bytes().decode('latin-1', errors='ignore')
        except Exception:
            txt = ""
    # Limit input size to avoid API request errors
    return txt[:12000]

def extract_with_ai(pdf_text: str, api_key: str, model_type: str) -> Dict[str, Any]:
    client = DeepSeekAPIClient(model_type=model_type, api_key=api_key)
    content = pdf_text if pdf_text else "PDF内容无法读取，请依据常识给出STM32F103C8T6典型参数。"
    result_text = client._call_api(SYSTEM_PROMPT, content)
    try:
        import json as _json
        return _json.loads(result_text)
    except Exception:
        import re, json as _json
        m = re.search(r"\{[\s\S]*\}", result_text)
        if m:
            return _json.loads(m.group())
    raise RuntimeError("AI未返回可解析JSON")

def default_stm32_item() -> Dict[str, Any]:
    return {
        "name": "STM32F103C8T6",
        "type": "microcontroller",
        "category": "controllers",
        "parameters": {
            "core": "ARM Cortex-M3",
            "frequency_mhz": 72,
            "flash_kb": 64,
            "sram_kb": 20,
            "peripherals": ["ADC", "USART", "SPI", "I2C", "Timers"],
            "operating_voltage_v": "2.0–3.6",
            "temperature_range": "-40..85",
            "package": "LQFP48"
        },
        "interconnect_effects": ["APB/AHB bus", "EMC", "clock tree", "power integrity"],
        "real_world_equivalent": "STM32F103C8T6"
    }

def merge_into_library(item: Dict[str, Any]):
    lib = {}
    if LIB_PATH.exists():
        lib = json.loads(LIB_PATH.read_text(encoding='utf-8'))
    else:
        lib = {"device_library": {"cpus": [], "gpus": [], "memory": [], "storage": [], "sensors": [], "communication": [], "controllers": []}}

    cat = item.get("category", "controllers")
    if cat not in lib.get("device_library", {}):
        lib["device_library"][cat] = []

    # 去重：按名称
    existing = lib["device_library"][cat]
    for d in existing:
        if d.get("name") == item.get("name"):
            return lib  # 已存在则不重复添加

    existing.append(item)
    lib["device_library"][cat] = existing
    LIB_PATH.write_text(json.dumps(lib, ensure_ascii=False, indent=2), encoding='utf-8')
    return lib

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", type=str, default=os.getenv("DEEPSEEK_API_KEY", ""))
    parser.add_argument("--model", type=str, default="deepseek-api", choices=["deepseek-api", "qwen0.5b", "deepseek-r1"])
    args = parser.parse_args()

    if args.model == "deepseek-api" and not args.api_key:
        raise RuntimeError("使用 deepseek-api 需要提供 --api-key 或设置环境变量 DEEPSEEK_API_KEY")

    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF未找到: {PDF_PATH}")
    pdf_text = read_pdf_text(PDF_PATH)
    try:
        item = extract_with_ai(pdf_text, args.api_key, args.model)
    except Exception as e:
        item = default_stm32_item()
    # sane defaults
    item.setdefault("type", "microcontroller")
    item.setdefault("category", "controllers")
    item.setdefault("real_world_equivalent", "STM32F103C8T6")
    lib = merge_into_library(item)
    print(json.dumps(item, ensure_ascii=False, indent=2))
    print("已加入器件库: ", item.get("name"))

if __name__ == "__main__":
    main()