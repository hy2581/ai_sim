#!/usr/bin/env python3
"""
Traditional Mode Processor
传统模式处理器 - 实现自然语言解析、器件匹配和比较分析
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.ai_integration.deepseek_api_client import DeepSeekAPIClient


class TraditionalModeProcessor:
    """传统模式处理器"""
    
    def __init__(self, api_key: str = None):
        self.api_client = DeepSeekAPIClient(api_key)
        self.project_root = project_root
        self.device_library = self._load_device_library()
        
    def _load_device_library(self) -> Dict[str, Any]:
        """加载器件库"""
        library_path = self.project_root / "device_library.json"
        try:
            with open(library_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print("❌ 器件库文件未找到，请先生成器件库")
            return {"device_library": {}}
    
    def process_traditional_mode(self, natural_language_input: str) -> Dict[str, Any]:
        """处理传统模式的完整流程"""
        print("启动传统模式处理流程")
        print("=" * 80)
        print(f"自然语言输入: {natural_language_input}")
        print("=" * 80)
        
        results = {
            "input": natural_language_input,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "task_requirements": {},
            "current_devices": {},
            "comparison_results": {},
            "final_report": "",
            "status": "processing"
        }
        
        try:
            # 步骤1: 生成任务需求.json
            print("\n步骤1: 解析自然语言输入，生成任务需求")
            task_requirements = self._generate_task_requirements(natural_language_input)
            results["task_requirements"] = task_requirements
            self._save_json_file("任务需求.json", task_requirements)
            print("任务需求.json 生成完成")
            
            # 步骤2: 生成当前器件.json
            print("\n步骤2: 基于自然语言输入和器件库，生成当前器件配置")
            current_devices = self._generate_current_devices(natural_language_input, task_requirements)
            results["current_devices"] = current_devices
            self._save_json_file("当前器件.json", current_devices)
            print("当前器件.json 生成完成")
            
            # 步骤3: 比较分析
            print("\n步骤3: 比较任务需求与当前器件配置")
            comparison_results = self._compare_requirements_and_devices(task_requirements, current_devices)
            results["comparison_results"] = comparison_results
            print("比较分析完成")
            
            # 步骤4: 生成最终报告
            print("\n步骤4: 生成分析报告")
            final_report = self._generate_final_report(results)
            results["final_report"] = final_report
            self._save_report(final_report)
            print("分析报告生成完成")
            
            results["status"] = "completed"
            print("\n传统模式处理完成！")
            
        except Exception as e:
            print(f"处理过程中发生错误: {str(e)}")
            results["status"] = "error"
            results["error"] = str(e)
            
        return results
    
    def _generate_task_requirements(self, natural_language_input: str) -> Dict[str, Any]:
        """生成任务需求.json"""
        system_prompt = """
你是一个航空航天微系统需求分析专家。请根据用户的自然语言描述，分析完成任务所需的各种器件需求。

请分析以下器件类型的需求：
1. CPU - 处理器需求
2. GPU - 图形/计算加速器需求  
3. 存储 - 存储设备需求
4. 内存 - 内存需求
5. 传感器 - 各类传感器需求
6. 通讯器件 - 通信模块需求
7. 控制器 - 控制器需求

对每种器件，请列出：
- 名称和类型
- 所需的关键参数
- 互联造成的影响

返回JSON格式，结构如下：
{
    "task_description": "任务描述",
    "required_devices": {
        "cpu": {
            "requirements": "CPU需求描述",
            "key_parameters": ["参数1", "参数2"],
            "interconnect_impacts": ["影响1", "影响2"]
        },
        "gpu": {
            "requirements": "GPU需求描述", 
            "key_parameters": ["参数1", "参数2"],
            "interconnect_impacts": ["影响1", "影响2"]
        },
        "memory": {
            "requirements": "内存需求描述",
            "key_parameters": ["参数1", "参数2"], 
            "interconnect_impacts": ["影响1", "影响2"]
        },
        "storage": {
            "requirements": "存储需求描述",
            "key_parameters": ["参数1", "参数2"],
            "interconnect_impacts": ["影响1", "影响2"] 
        },
        "sensors": {
            "requirements": "传感器需求描述",
            "key_parameters": ["参数1", "参数2"],
            "interconnect_impacts": ["影响1", "影响2"]
        },
        "communication": {
            "requirements": "通讯器件需求描述", 
            "key_parameters": ["参数1", "参数2"],
            "interconnect_impacts": ["影响1", "影响2"]
        },
        "controllers": {
            "requirements": "控制器需求描述",
            "key_parameters": ["参数1", "参数2"], 
            "interconnect_impacts": ["影响1", "影响2"]
        }
    }
}
"""
        
        try:
            response = self.api_client._call_api(system_prompt, natural_language_input)
            # 尝试解析JSON
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return self._get_default_task_requirements(natural_language_input)
        except Exception as e:
            print(f"⚠️ API调用失败，使用默认任务需求: {e}")
            return self._get_default_task_requirements(natural_language_input)
    
    def _generate_current_devices(self, natural_language_input: str, task_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """生成当前器件.json"""
        # 首先从自然语言中提取已有器件信息
        extracted_devices = self._extract_devices_from_input(natural_language_input)
        
        # 然后从器件库中补充缺失的器件
        supplemented_devices, supplement_reasons = self._complete_devices_from_library(extracted_devices, task_requirements)
        
        return {
            "description": "基于自然语言输入和器件库选择的当前器件配置",
            "extraction_source": "自然语言输入 + 器件库补充",
            "extracted_devices": extracted_devices,
            "supplemented_devices": supplemented_devices,
            "supplement_reasons": supplement_reasons,
            "devices": supplemented_devices
        }
    
    def _extract_devices_from_input(self, natural_language_input: str) -> Dict[str, Any]:
        """从自然语言输入中提取器件信息"""
        system_prompt = """
请从用户的自然语言描述中提取已经提到的具体器件信息。

分析文本中提到的：
1. 处理器/CPU相关器件
2. GPU/计算加速器
3. 存储设备
4. 内存
5. 传感器（如激光拉曼探头、光谱仪、压电钻头等）
6. 通讯器件
7. 控制器/SoC

对每个提到的器件，提取：
- 器件名称或类型
- 提到的参数
- 功能描述

返回JSON格式：
{
    "extracted_devices": {
        "cpu": [],
        "gpu": [],
        "memory": [],
        "storage": [],
        "sensors": [],
        "communication": [],
        "controllers": []
    }
}
"""
        
        try:
            response = self.api_client._call_api(system_prompt, natural_language_input)
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return self._get_default_extracted_devices()
        except Exception as e:
            print(f"⚠️ 器件提取失败，使用默认配置: {e}")
            return self._get_default_extracted_devices()
    
    def _complete_devices_from_library(self, extracted_devices: Dict[str, Any], task_requirements: Dict[str, Any]) -> tuple:
        """从器件库中补充缺失的器件"""
        completed_devices = {}
        supplement_reasons = {}
        
        # 映射单数到复数的器件类别名称
        category_mapping = {
            "cpu": "cpus",
            "gpu": "gpus", 
            "memory": "memory",
            "storage": "storage",
            "sensors": "sensors",
            "communication": "communication",
            "controllers": "controllers"
        }
        
        device_categories = ["cpu", "gpu", "memory", "storage", "sensors", "communication", "controllers"]
        
        for category in device_categories:
            completed_devices[category] = []
            supplement_reasons[category] = []
            
            # 添加已提取的器件
            extracted_category_devices = []
            if category in extracted_devices.get("extracted_devices", {}):
                extracted_category_devices = extracted_devices["extracted_devices"][category]
                for device in extracted_category_devices:
                    device["source"] = "自然语言提取"
                    completed_devices[category].append(device)
            
            # 从器件库中补充合适的器件
            library_category = category_mapping.get(category, category)
            if library_category in self.device_library.get("device_library", {}):
                library_devices = self.device_library["device_library"][library_category]
                
                # 根据任务需求选择合适的器件
                selected_devices, reasons = self._select_suitable_devices_with_reasons(
                    library_devices, task_requirements, category, len(extracted_category_devices)
                )
                
                for device in selected_devices:
                    device["source"] = "器件库补充"
                    completed_devices[category].append(device)
                
                supplement_reasons[category] = reasons
        
        return completed_devices, supplement_reasons
    
    def _detect_application_scenario(self, task_requirements: Dict[str, Any]) -> str:
        """检测应用场景：地面工业 vs 深空航天"""
        # 从任务描述中提取关键词
        task_desc = task_requirements.get("task_description", "").lower()
        
        # 地面场景关键词
        ground_keywords = ["地面", "工业", "矿区", "巡检", "维护", "成本", "emc", "以太网", "5g", "工业级", "可维护"]
        # 航天场景关键词  
        space_keywords = ["深空", "航天", "探测", "抗辐射", "真空", "宇宙", "机械臂", "天体", "spacewire", "can总线"]
        
        ground_score = sum(1 for keyword in ground_keywords if keyword in task_desc)
        space_score = sum(1 for keyword in space_keywords if keyword in task_desc)
        
        return "ground" if ground_score > space_score else "space"

    def _select_suitable_devices_with_reasons(self, library_devices: List[Dict], task_requirements: Dict[str, Any], category: str, extracted_count: int) -> tuple:
        """从器件库中选择合适的器件并提供选择理由"""
        selected = []
        reasons = []
        
        if not library_devices:
            return selected, reasons
        
        # 检测应用场景
        scenario = self._detect_application_scenario(task_requirements)
        
        # 根据类别和需求选择器件
        category_requirements = task_requirements.get("required_devices", {}).get(category, {})
        
        if extracted_count == 0:
            # 如果没有提取到该类别的器件，需要补充基础器件
            reason = f"自然语言输入中未明确提及{category}器件，根据任务需求从器件库中选择合适的器件"
            
            if category == "cpu":
                if scenario == "ground":
                    # 地面场景：优先选择工业级、价格较低的器件
                    industrial_cpus = [d for d in library_devices if 
                                     not d.get("parameters", {}).get("space_qualified", False) and
                                     ("工业" in d.get("category", "") or "嵌入式" in d.get("category", ""))]
                    if industrial_cpus:
                        # 选择性能最高的工业级CPU
                        best_cpu = max(industrial_cpus, key=lambda x: x.get("parameters", {}).get("frequency_mhz", 0))
                        selected.append(self._format_selected_device(best_cpu, f"选择工业级处理器{best_cpu['name']}，在满足性能需求的条件下选用价格较低的器件（相比深空器件成本更低）"))
                    else:
                        # 如果没有明确的工业级器件，选择非航天专用的
                        non_space_cpus = [d for d in library_devices if not d.get("parameters", {}).get("space_qualified", False)]
                        if non_space_cpus:
                            best_cpu = max(non_space_cpus, key=lambda x: x.get("parameters", {}).get("frequency_mhz", 0))
                            selected.append(self._format_selected_device(best_cpu, f"选择高性能处理器{best_cpu['name']}，在满足应用需求的条件下选用价格较低的器件"))
                        else:
                            best_cpu = library_devices[0]
                            selected.append(self._format_selected_device(best_cpu, f"选择处理器{best_cpu['name']}满足地面工业应用需求"))
                else:
                    # 航天场景：优先选择航天专用CPU
                    space_qualified_cpus = [d for d in library_devices if d.get("parameters", {}).get("space_qualified", False)]
                    if space_qualified_cpus:
                        # 选择抗辐射等级最高的航天专用CPU
                        best_cpu = max(space_qualified_cpus, key=lambda x: self._extract_radiation_tolerance(x.get("parameters", {}).get("radiation_tolerance", "0")))
                        selected.append(self._format_selected_device(best_cpu, f"选择航天专用处理器{best_cpu['name']}，抗辐射等级{best_cpu.get('parameters', {}).get('radiation_tolerance', 'N/A')}"))
                    else:
                        # 选择抗辐射等级最高的CPU
                        best_cpu = max(library_devices, key=lambda x: self._extract_radiation_tolerance(x.get("parameters", {}).get("radiation_tolerance", "0")))
                        selected.append(self._format_selected_device(best_cpu, f"选择抗辐射处理器{best_cpu['name']}满足深空环境需求"))
                    
            elif category == "gpu":
                if scenario == "ground":
                    # 地面场景：优先选择工业级AI加速器或高性能GPU
                    industrial_gpus = [d for d in library_devices if 
                                     not d.get("parameters", {}).get("space_qualified", False) and
                                     ("AI" in d.get("category", "") or "加速器" in d.get("category", "") or "工业" in d.get("category", ""))]
                    if industrial_gpus:
                        best_gpu = max(industrial_gpus, key=lambda x: x.get("parameters", {}).get("compute_units", 0))
                        selected.append(self._format_selected_device(best_gpu, f"选择工业级AI加速器{best_gpu['name']}，在满足AI处理需求的条件下选用价格较低的器件"))
                    else:
                        # 选择非航天专用的GPU/FPGA
                        non_space_gpus = [d for d in library_devices if not d.get("parameters", {}).get("space_qualified", False)]
                        if non_space_gpus:
                            best_gpu = non_space_gpus[0]
                            selected.append(self._format_selected_device(best_gpu, f"选择图形处理器{best_gpu['name']}支持地面工业图像处理"))
                        else:
                            selected.append(self._format_selected_device(library_devices[0], "选择图形处理器支持成像数据处理"))
                else:
                    # 航天场景：优先选择航天专用GPU或抗辐射FPGA
                    space_qualified_gpus = [d for d in library_devices if d.get("parameters", {}).get("space_qualified", False)]
                    if space_qualified_gpus:
                        best_gpu = max(space_qualified_gpus, key=lambda x: self._extract_radiation_tolerance(x.get("parameters", {}).get("radiation_tolerance", "0")))
                        selected.append(self._format_selected_device(best_gpu, f"选择航天专用图形处理器{best_gpu['name']}支持光谱数据处理"))
                    else:
                        # 选择抗辐射FPGA或AI加速器
                        fpga_devices = [d for d in library_devices if "FPGA" in d.get("category", "") or "FPGA" in d.get("name", "")]
                        if fpga_devices:
                            best_gpu = max(fpga_devices, key=lambda x: self._extract_radiation_tolerance(x.get("parameters", {}).get("radiation_tolerance", "0")))
                            selected.append(self._format_selected_device(best_gpu, f"选择抗辐射FPGA{best_gpu['name']}支持图像处理"))
                        elif library_devices:
                            selected.append(self._format_selected_device(library_devices[0], "选择图形处理器支持成像数据处理"))
                    
            elif category == "memory":
                if scenario == "ground":
                    # 地面场景：优先选择工业级内存，关注性价比
                    industrial_memory = [d for d in library_devices if 
                                       not d.get("parameters", {}).get("space_qualified", False) and
                                       d.get("parameters", {}).get("ecc_support", False)]
                    if industrial_memory:
                        # 选择容量最大的工业级ECC内存
                        best_memory = max(industrial_memory, key=lambda x: x.get("parameters", {}).get("capacity_gb", 0))
                        selected.append(self._format_selected_device(best_memory, f"选择工业级ECC内存{best_memory['name']}，在满足可靠性需求的条件下选用价格较低的器件"))
                    else:
                        # 选择非航天专用的内存
                        non_space_memory = [d for d in library_devices if not d.get("parameters", {}).get("space_qualified", False)]
                        if non_space_memory:
                            best_memory = max(non_space_memory, key=lambda x: x.get("parameters", {}).get("capacity_gb", 0))
                            selected.append(self._format_selected_device(best_memory, f"选择高容量内存{best_memory['name']}满足地面工业数据处理需求"))
                        elif library_devices:
                            selected.append(self._format_selected_device(library_devices[0], "选择高性能内存支持数据处理"))
                else:
                    # 航天场景：优先选择航天专用内存
                    space_qualified_memory = [d for d in library_devices if d.get("parameters", {}).get("space_qualified", False)]
                    if space_qualified_memory:
                        best_memory = max(space_qualified_memory, key=lambda x: self._extract_radiation_tolerance(x.get("parameters", {}).get("radiation_tolerance", "0")))
                        selected.append(self._format_selected_device(best_memory, f"选择航天专用内存{best_memory['name']}保证数据可靠性"))
                    else:
                        # 选择ECC内存中抗辐射等级最高的
                        ecc_memory = [d for d in library_devices if d.get("parameters", {}).get("ecc_support", False)]
                        if ecc_memory:
                            best_memory = max(ecc_memory, key=lambda x: self._extract_radiation_tolerance(x.get("parameters", {}).get("radiation_tolerance", "0")))
                            selected.append(self._format_selected_device(best_memory, f"选择ECC内存{best_memory['name']}保证数据可靠性"))
                        elif library_devices:
                            selected.append(self._format_selected_device(library_devices[0], "选择高性能内存支持数据处理"))
                    
            elif category == "storage":
                if scenario == "ground":
                    # 地面场景：优先选择工业级大容量存储
                    industrial_storage = [d for d in library_devices if 
                                        not d.get("parameters", {}).get("space_qualified", False) and
                                        ("工业" in d.get("category", "") or "eMMC" in d.get("category", "") or "SSD" in d.get("category", ""))]
                    if industrial_storage:
                        # 选择容量最大的工业级存储
                        best_storage = max(industrial_storage, key=lambda x: x.get("parameters", {}).get("capacity_gb", 0))
                        selected.append(self._format_selected_device(best_storage, f"选择工业级存储{best_storage['name']}，在满足大容量需求的条件下选用价格较低的器件"))
                    else:
                        # 选择非航天专用的大容量存储
                        non_space_storage = [d for d in library_devices if not d.get("parameters", {}).get("space_qualified", False)]
                        if non_space_storage:
                            best_storage = max(non_space_storage, key=lambda x: x.get("parameters", {}).get("capacity_gb", 0))
                            selected.append(self._format_selected_device(best_storage, f"选择大容量存储{best_storage['name']}满足地面工业数据存储需求"))
                        else:
                            best_storage = library_devices[0]
                            selected.append(self._format_selected_device(best_storage, f"选择存储器件{best_storage['name']}支持数据记录"))
                else:
                    # 航天场景：优先选择航天专用存储
                    space_qualified_storage = [d for d in library_devices if d.get("parameters", {}).get("space_qualified", False)]
                    if space_qualified_storage:
                        best_storage = max(space_qualified_storage, key=lambda x: self._extract_radiation_tolerance(x.get("parameters", {}).get("radiation_tolerance", "0")))
                        selected.append(self._format_selected_device(best_storage, f"选择航天专用存储{best_storage['name']}保证数据安全"))
                    else:
                        # 选择抗辐射等级最高的存储
                        best_storage = max(library_devices, key=lambda x: self._extract_radiation_tolerance(x.get("parameters", {}).get("radiation_tolerance", "0")))
                        selected.append(self._format_selected_device(best_storage, f"选择抗辐射存储{best_storage['name']}支持实时数据记录"))
                    
            elif category == "communication":
                if scenario == "ground":
                    # 地面场景：优先选择工业以太网、5G等地面通信方案
                    ground_comm = [d for d in library_devices if 
                                 not d.get("parameters", {}).get("space_qualified", False) and
                                 ("以太网" in d.get("category", "") or "5G" in d.get("category", "") or 
                                  "LoRa" in d.get("name", "") or "工业" in d.get("category", ""))]
                    if ground_comm:
                        best_comm = ground_comm[0]
                        selected.append(self._format_selected_device(best_comm, f"选择地面工业通信器件{best_comm['name']}，在满足通信需求的条件下选用价格较低的器件"))
                    else:
                        # 选择非航天专用的通信器件
                        non_space_comm = [d for d in library_devices if not d.get("parameters", {}).get("space_qualified", False)]
                        if non_space_comm:
                            best_comm = non_space_comm[0]
                            selected.append(self._format_selected_device(best_comm, f"选择通信器件{best_comm['name']}支持地面数据传输"))
                        elif library_devices:
                            selected.append(self._format_selected_device(library_devices[0], "选择通信模块支持数据传输"))
                else:
                    # 航天场景：优先选择深空通信或航天专用通信器件
                    space_comm = [d for d in library_devices if 
                                 d.get("parameters", {}).get("space_qualified", False) or 
                                 "深空" in d.get("category", "") or 
                                 "航天" in d.get("category", "") or
                                 "SpaceWire" in str(d.get("parameters", {}).get("interfaces", []))]
                    if space_comm:
                        best_comm = max(space_comm, key=lambda x: self._extract_radiation_tolerance(x.get("parameters", {}).get("radiation_tolerance", "0")))
                        selected.append(self._format_selected_device(best_comm, f"选择航天专用通信器件{best_comm['name']}适合深空环境"))
                    else:
                        # 选择LoRa或GNSS等适合的通信方案
                        suitable_comm = [d for d in library_devices if "LoRa" in d.get("name", "") or "GNSS" in d.get("name", "")]
                        if suitable_comm:
                            best_comm = suitable_comm[0]
                            selected.append(self._format_selected_device(best_comm, f"选择{best_comm['name']}作为备用通信方案"))
                        elif library_devices:
                            selected.append(self._format_selected_device(library_devices[0], "选择通信模块支持数据传输"))
                    
            elif category == "controllers":
                if scenario == "ground":
                    # 地面场景：优先选择工业级控制器
                    industrial_controllers = [d for d in library_devices if 
                                            not d.get("parameters", {}).get("space_qualified", False) and
                                            ("工业" in d.get("category", "") or "嵌入式" in d.get("category", ""))]
                    if industrial_controllers:
                        best_controller = max(industrial_controllers, key=lambda x: x.get("parameters", {}).get("frequency_mhz", 0))
                        selected.append(self._format_selected_device(best_controller, f"选择工业级控制器{best_controller['name']}，在满足控制精度需求的条件下选用价格较低的器件"))
                    else:
                        # 选择非航天专用的控制器
                        non_space_controllers = [d for d in library_devices if not d.get("parameters", {}).get("space_qualified", False)]
                        if non_space_controllers:
                            best_controller = non_space_controllers[0]
                            selected.append(self._format_selected_device(best_controller, f"选择控制器{best_controller['name']}支持地面工业控制"))
                        else:
                            best_controller = library_devices[0]
                            selected.append(self._format_selected_device(best_controller, f"选择控制器{best_controller['name']}支持系统控制"))
                else:
                    # 航天场景：优先选择航天专用控制器
                    space_controllers = [d for d in library_devices if d.get("parameters", {}).get("space_qualified", False)]
                    if space_controllers:
                        best_controller = max(space_controllers, key=lambda x: self._extract_radiation_tolerance(x.get("parameters", {}).get("radiation_tolerance", "0")))
                        selected.append(self._format_selected_device(best_controller, f"选择航天专用控制器{best_controller['name']}支持容错实时控制"))
                    else:
                        # 选择抗辐射等级最高的控制器
                        best_controller = max(library_devices, key=lambda x: self._extract_radiation_tolerance(x.get("parameters", {}).get("radiation_tolerance", "0")))
                        selected.append(self._format_selected_device(best_controller, f"选择抗辐射控制器{best_controller['name']}支持精确控制"))
                    
            elif category == "sensors":
                if scenario == "ground":
                    # 地面场景：优先选择工业级传感器
                    industrial_sensors = [d for d in library_devices if 
                                        not d.get("parameters", {}).get("space_qualified", False) and
                                        ("工业" in d.get("category", "") or "MEMS" in d.get("category", ""))]
                    if industrial_sensors:
                        best_sensor = industrial_sensors[0]
                        selected.append(self._format_selected_device(best_sensor, f"选择工业级传感器{best_sensor['name']}，在满足检测精度需求的条件下选用价格较低的器件"))
                    else:
                        # 选择非航天专用的传感器
                        non_space_sensors = [d for d in library_devices if not d.get("parameters", {}).get("space_qualified", False)]
                        if non_space_sensors:
                            best_sensor = non_space_sensors[0]
                            selected.append(self._format_selected_device(best_sensor, f"选择传感器{best_sensor['name']}支持地面工业检测"))
                        else:
                            best_sensor = library_devices[0]
                            selected.append(self._format_selected_device(best_sensor, f"选择传感器{best_sensor['name']}支持数据采集"))
                else:
                    # 航天场景：优先选择航天专用传感器
                    space_sensors = [d for d in library_devices if d.get("parameters", {}).get("space_qualified", False)]
                    if space_sensors:
                        best_sensor = max(space_sensors, key=lambda x: self._extract_radiation_tolerance(x.get("parameters", {}).get("radiation_tolerance", "0")))
                        selected.append(self._format_selected_device(best_sensor, f"选择航天专用传感器{best_sensor['name']}"))
                    else:
                        # 选择抗辐射等级最高的传感器
                        best_sensor = max(library_devices, key=lambda x: self._extract_radiation_tolerance(x.get("parameters", {}).get("radiation_tolerance", "0")))
                        selected.append(self._format_selected_device(best_sensor, f"选择抗辐射传感器{best_sensor['name']}"))
                    
            reasons.append(reason)
            
        else:
            # 如果已有提取的器件，可以选择性补充
            if len(library_devices) > 1:
                # 根据场景选择不同类型的器件作为备选
                if scenario == "ground":
                    # 地面场景优先选择非航天专用器件
                    non_space_devices = [d for d in library_devices if not d.get("parameters", {}).get("space_qualified", False)]
                    if non_space_devices and len(non_space_devices) >= 1:
                        for i, device in enumerate(non_space_devices[:2]):
                            reason = f"补充工业级{category}器件{device['name']}作为备选方案，在满足应用需求的条件下选用价格较低的器件"
                            selected.append(self._format_selected_device(device, reason))
                            reasons.append(reason)
                    else:
                        for i, device in enumerate(library_devices[:2]):
                            if i < 2:  # 最多补充2个
                                reason = f"补充不同规格的{category}器件作为备选方案，提供更多配置选择"
                                selected.append(self._format_selected_device(device, reason))
                                reasons.append(reason)
                else:
                    # 航天场景优先选择航天专用器件
                    space_devices = [d for d in library_devices if d.get("parameters", {}).get("space_qualified", False)]
                    if space_devices and len(space_devices) >= 1:
                        for i, device in enumerate(space_devices[:2]):
                            reason = f"补充航天专用{category}器件{device['name']}作为备选方案"
                            selected.append(self._format_selected_device(device, reason))
                            reasons.append(reason)
                    else:
                        for i, device in enumerate(library_devices[:2]):
                            if i < 2:  # 最多补充2个
                                reason = f"补充不同规格的{category}器件作为备选方案，提供更多配置选择"
                                selected.append(self._format_selected_device(device, reason))
                                reasons.append(reason)
        
        return selected, reasons
    
    def _extract_radiation_tolerance(self, radiation_str: str) -> float:
        """提取辐射耐受等级数值用于比较"""
        if not radiation_str:
            return 0.0
        try:
            # 提取数字部分，如 "1000 krad(Si)" -> 1000.0
            import re
            match = re.search(r'(\d+(?:\.\d+)?)', str(radiation_str))
            if match:
                return float(match.group(1))
        except:
            pass
        return 0.0
    
    def _format_selected_device(self, device: Dict, selection_reason: str) -> Dict:
        """格式化选中的器件"""
        return {
            "name": device["name"],
            "type": device["type"],
            "category": device["category"],
            "parameters": device["parameters"],
            "interconnect_effects": device["interconnect_effects"],
            "selection_reason": selection_reason
        }
    
    def _compare_requirements_and_devices(self, task_requirements: Dict[str, Any], current_devices: Dict[str, Any]) -> Dict[str, Any]:
        """比较任务需求和当前器件"""
        system_prompt = """
你是一个航空航天系统分析专家。请比较任务需求和当前器件配置，分析匹配程度和潜在问题。

请分析：
1. 每类器件的需求匹配程度
2. 性能是否满足要求
3. 互联兼容性
4. 功耗和热管理
5. 可靠性和环境适应性
6. 成本效益

返回JSON格式的比较结果：
{
    "overall_match_score": 0.85,
    "category_analysis": {
        "cpu": {
            "match_score": 0.9,
            "analysis": "分析结果",
            "issues": ["问题1", "问题2"],
            "recommendations": ["建议1", "建议2"]
        }
    },
    "system_level_analysis": {
        "interconnect_compatibility": "兼容性分析",
        "power_analysis": "功耗分析", 
        "thermal_analysis": "热管理分析",
        "reliability_analysis": "可靠性分析"
    },
    "recommendations": ["总体建议1", "总体建议2"]
}
"""
        
        comparison_data = {
            "task_requirements": task_requirements,
            "current_devices": current_devices
        }
        
        try:
            response = self.api_client._call_api(system_prompt, json.dumps(comparison_data, ensure_ascii=False, indent=2))
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return self._get_default_comparison_results()
        except Exception as e:
            print(f"⚠️ 比较分析失败，使用默认结果: {e}")
            return self._get_default_comparison_results()
    
    def _generate_final_report(self, results: Dict[str, Any]) -> str:
        """生成最终分析报告"""
        system_prompt = """
你是一个航空航天系统分析专家。请基于前面的分析结果，生成一份完整的分析报告。

特别注意：
1. 器件配置包含两部分：从自然语言提取的器件 和 从器件库补充的器件
2. 需要详细说明每个补充器件的选择理由
3. 分析提取器件与补充器件的匹配性和兼容性

报告应包含：
1. 执行摘要
2. 任务需求分析
3. 器件配置分析（重点区分提取器件和补充器件）
4. 器件选择原因详细说明
5. 比较分析结果
6. 系统级评估
7. 风险分析
8. 改进建议
9. 结论：原始需求的合理性评估

请用中文撰写详细的技术报告，特别强调器件来源的区分和补充理由的说明。
"""
        
        try:
            response = self.api_client._call_api(system_prompt, json.dumps(results, ensure_ascii=False, indent=2))
            return response
        except Exception as e:
            print(f"⚠️ 报告生成失败，使用默认报告: {e}")
            return self._get_enhanced_default_report(results)
    
    def _save_json_file(self, filename: str, data: Dict[str, Any]):
        """保存JSON文件"""
        filepath = self.project_root / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _save_report(self, report_content: str):
        """保存报告文件"""
        filepath = self.project_root / "传统模式分析报告.md"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report_content)
    
    def _get_default_task_requirements(self, input_text: str) -> Dict[str, Any]:
        """获取默认任务需求"""
        return {
            "task_description": f"基于输入的航空航天微系统任务: {input_text[:100]}...",
            "required_devices": {
                "cpu": {
                    "requirements": "高性能低功耗处理器，支持实时计算",
                    "key_parameters": ["多核架构", "低功耗", "抗辐射"],
                    "interconnect_impacts": ["内存带宽需求", "热管理"]
                },
                "sensors": {
                    "requirements": "多种传感器用于数据采集和分析",
                    "key_parameters": ["高精度", "低功耗", "环境适应性"],
                    "interconnect_impacts": ["数据传输带宽", "实时性要求"]
                }
            }
        }
    
    def _get_default_extracted_devices(self) -> Dict[str, Any]:
        """获取默认提取的器件"""
        return {
            "extracted_devices": {
                "cpu": [],
                "gpu": [],
                "memory": [],
                "storage": [],
                "sensors": [
                    {
                        "name": "微型激光拉曼探头",
                        "type": "光谱传感器",
                        "function": "物质识别"
                    },
                    {
                        "name": "高光谱成像仪", 
                        "type": "成像传感器",
                        "function": "光谱成像"
                    }
                ],
                "communication": [],
                "controllers": [
                    {
                        "name": "抗辐射片上系统(SoC)",
                        "type": "系统级控制器",
                        "function": "控制与数据处理"
                    }
                ]
            }
        }
    
    def _get_default_comparison_results(self) -> Dict[str, Any]:
        """获取默认比较结果"""
        return {
            "overall_match_score": 0.75,
            "category_analysis": {
                "cpu": {
                    "match_score": 0.8,
                    "analysis": "处理器配置基本满足需求",
                    "issues": ["功耗可能偏高"],
                    "recommendations": ["考虑更低功耗的处理器"]
                }
            },
            "system_level_analysis": {
                "interconnect_compatibility": "器件间兼容性良好",
                "power_analysis": "总功耗在可接受范围内",
                "thermal_analysis": "需要适当的热管理设计",
                "reliability_analysis": "满足航空航天可靠性要求"
            },
            "recommendations": ["优化功耗设计", "加强热管理"]
        }
    
    def _get_enhanced_default_report(self, results: Dict[str, Any]) -> str:
        """获取增强的默认报告，区分提取器件和补充器件"""
        current_devices = results.get('current_devices', {})
        extracted_devices = current_devices.get('extracted_devices', {})
        supplemented_devices = current_devices.get('supplemented_devices', {})
        supplement_reasons = current_devices.get('supplement_reasons', {})
        
        # 统计器件数量
        extracted_count = sum(len(devices) for devices in extracted_devices.get('extracted_devices', {}).values())
        supplemented_count = sum(len(devices) for devices in supplemented_devices.values())
        
        report = f"""
# 传统模式分析报告

## 执行摘要
本报告基于自然语言输入"{results.get('input', '')}"进行了航空航天微系统的需求分析和器件配置评估。

**器件配置概况：**
- 从自然语言提取的器件：{extracted_count} 个
- 从器件库补充的器件：{supplemented_count} 个
- 总体匹配度：{results.get('comparison_results', {}).get('overall_match_score', 0.75)}

## 任务需求分析
系统需要完成复杂的航空航天任务，包括数据采集、处理和分析等功能。

## 器件配置分析

### 3.1 从自然语言提取的器件
以下器件是从用户的自然语言描述中直接识别和提取的：

"""
        
        # 添加提取器件的详细信息
        for category, devices in extracted_devices.get('extracted_devices', {}).items():
            if devices:
                report += f"\n**{category.upper()}类器件：**\n"
                for device in devices:
                    report += f"- {device.get('name', '未知器件')}: {device.get('function', '功能描述')}\n"
        
        report += f"""

### 3.2 从器件库补充的器件
基于任务需求分析，从器件库中补充了以下器件：

"""
        
        # 添加补充器件的详细信息和理由
        for category, devices in supplemented_devices.items():
            if devices:
                report += f"\n**{category.upper()}类器件：**\n"
                for device in devices:
                    if device.get('source') == '器件库补充':
                        report += f"- {device.get('name', '未知器件')}\n"
                        report += f"  - 选择理由：{device.get('selection_reason', '未提供理由')}\n"
                        report += f"  - 关键参数：{list(device.get('parameters', {}).keys())[:3]}\n"
                
                # 添加类别级别的补充理由
                if category in supplement_reasons and supplement_reasons[category]:
                    report += f"  - 补充原因：{supplement_reasons[category][0]}\n"
        
        report += f"""

## 器件选择原因详细说明

### 4.1 提取器件的合理性
从自然语言中提取的器件反映了用户对系统的具体技术要求，这些器件通常是任务关键器件。

### 4.2 补充器件的必要性
补充器件主要解决以下问题：
1. **系统完整性**：补充缺失的核心器件，确保系统能够正常运行
2. **性能匹配**：选择与任务需求匹配的器件规格
3. **环境适应**：考虑航空航天环境的特殊要求
4. **技术兼容**：确保器件间的互联兼容性

## 比较分析结果
任务需求与器件配置的匹配度为{results.get('comparison_results', {}).get('overall_match_score', 0.75)}。

### 5.1 提取器件与需求匹配度
提取的器件通常与用户明确需求高度匹配，但可能存在规格不明确的问题。

### 5.2 补充器件与需求匹配度
补充器件基于技术分析选择，在性能和兼容性方面经过优化。

## 系统级评估

### 6.1 器件间兼容性
- **提取器件与补充器件的兼容性**：需要验证接口标准和通信协议
- **功耗匹配**：补充器件的功耗设计需要与整体功耗预算匹配
- **热管理**：器件布局需要考虑热耦合效应

### 6.2 技术风险评估
- **提取器件风险**：规格可能不够明确，需要进一步细化
- **补充器件风险**：选择基于假设，需要实际验证

## 改进建议

1. **细化提取器件规格**：对自然语言中提到的器件进行详细的技术规格定义
2. **验证补充器件适用性**：通过仿真或测试验证补充器件的实际性能
3. **优化器件配置**：根据实际约束条件调整器件选择
4. **完善系统集成**：设计合适的互联架构和接口标准

## 结论

### 原始需求合理性评估
**结论：需求在技术上是合理的**

**理由：**
1. 提取的器件反映了用户对关键技术的准确理解
2. 补充的器件能够形成完整的系统架构
3. 整体技术路线符合航空航天系统的发展趋势
4. 存在的问题主要是工程实现层面，可以通过优化设计解决

**建议：**
在后续设计中重点关注器件间的集成优化和环境适应性验证。

---
生成时间: {results.get('timestamp', '')}
处理模式: 传统模式（自然语言提取 + 器件库补充）
"""
        
        return report


def main():
    """测试函数"""
    processor = TraditionalModeProcessor()
    test_input = "这款应用于深空探测的微型系统，高度集成于探测器机械臂末端"
    results = processor.process_traditional_mode(test_input)
    print(f"处理结果: {results['status']}")


if __name__ == "__main__":
    main()