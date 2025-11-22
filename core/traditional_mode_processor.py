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
    
    def __init__(self, api_key: str = None, model_type: str = "qwen0.5b"):
        """
        初始化处理器
        
        Args:
            api_key: DeepSeek API密钥 (仅 deepseek-api 需要)
            model_type: 模型类型 (qwen0.5b/deepseek-api/deepseek-r1)
        """
        self.api_client = DeepSeekAPIClient(api_key, model_type)
        self.model_type = model_type
        self.project_root = project_root
        self.device_library = self._load_device_library()
        
    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        try:
            cleaned = text.strip()
            if cleaned.startswith('```json'):
                cleaned = cleaned[7:]
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                import re
                m = re.search(r'\{[\s\S]*\}', cleaned, re.DOTALL)
                if m:
                    return json.loads(m.group())
        except Exception:
            pass
        return None

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
            parsed = self._extract_json(response)
            if parsed is not None:
                return parsed
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
            parsed = self._extract_json(response)
            if parsed is not None:
                return parsed
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
        """使用DeepSeek API从器件库中智能选择合适的器件"""
        selected = []
        reasons = []
        
        if not library_devices:
            return selected, reasons
        
        try:
            # 调用API进行智能器件选型
            api_selected, api_reasons = self._api_select_devices(
                library_devices, 
                task_requirements, 
                category, 
                extracted_count
            )
            return api_selected, api_reasons
        except Exception as e:
            print(f"⚠️ API选型失败，使用备选逻辑: {e}")
            # 如果API失败，回退到简单逻辑
            return self._fallback_select_devices(
                library_devices, 
                task_requirements, 
                category, 
                extracted_count
            )

    def _api_select_devices(self, library_devices: List[Dict], task_requirements: Dict[str, Any], category: str, extracted_count: int) -> tuple:
        """使用DeepSeek API进行器件智能选择"""
        selected = []
        reasons = []
        
        # 构建选择提示词
        system_prompt = """
你是一个航空航天微系统的器件选型专家。请根据任务需求从器件库中选择最合适的器件。

你的工作是分析：
1. 任务需求和应用场景
2. 可用的器件库选项
3. 器件的各项参数和特性
4. 器件间的兼容性

然后选择最佳的器件方案，并给出详细的选择理由。

返回JSON格式：
{
    "selected_devices": [
        {
            "name": "器件名称",
            "reason": "选择该器件的具体理由",
            "parameters_highlight": "该器件的关键优势参数"
        }
    ],
    "selection_summary": "整体选择策略说明"
}
"""
        
        # 构建用户提示词
        user_input = f"""
【任务需求】
{json.dumps(task_requirements, ensure_ascii=False, indent=2)}

【器件类别】
{category}

【已提取器件数量】
{extracted_count}

【可用的器件库】
{json.dumps(library_devices, ensure_ascii=False, indent=2)}

请分析这个应用场景，选择1到2个最合适的器件。

选择标准：
1. 如果已提取器件数量为0，需要选择基础器件，确保系统完整性
2. 如果已有提取器件，可以选择1-2个备选或补充方案
3. 优先考虑与任务需求的匹配度
4. 考虑器件的性能、可靠性、成本等多个维度
5. 给出明确的选择理由
"""
        
        try:
            response = self.api_client._call_api(system_prompt, user_input)
            api_result = self._extract_json(response)
            if not api_result:
                raise Exception("API返回格式不符合")
            for item in api_result.get("selected_devices", []):
                device_name = item.get("name", "")
                for lib_device in library_devices:
                    if lib_device.get("name", "") == device_name:
                        selected.append(self._format_selected_device(
                            lib_device,
                            item.get("reason", "")
                        ))
                        reasons.append(item.get("reason", ""))
                        break
                else:
                    for lib_device in library_devices:
                        if device_name.lower() in lib_device.get("name", "").lower():
                            selected.append(self._format_selected_device(
                                lib_device,
                                item.get("reason", "")
                            ))
                            reasons.append(item.get("reason", ""))
                            break
            if not selected and library_devices:
                selected.append(self._format_selected_device(
                    library_devices[0],
                    api_result.get("selection_summary", "根据任务需求选择")
                ))
                reasons.append(api_result.get("selection_summary", ""))
        except Exception as e:
            print(f"API解析失败: {e}，使用备选逻辑")
            raise
        
        return selected, reasons

    def _fallback_select_devices(self, library_devices: List[Dict], task_requirements: Dict[str, Any], category: str, extracted_count: int) -> tuple:
        """备选的简单选择逻辑（当API失败时使用）"""
        selected = []
        reasons = []
        
        if not library_devices:
            return selected, reasons
        
        # 简单的备选逻辑：选择第一个器件
        selected_device = library_devices[0]
        reason = f"根据任务需求选择{category}类器件：{selected_device.get('name', '未知')}"
        
        selected.append(self._format_selected_device(selected_device, reason))
        reasons.append(reason)
        
        # 如果需要补充，再选第二个
        if extracted_count > 0 and len(library_devices) > 1:
            selected_device_2 = library_devices[1]
            reason_2 = f"补充{category}类备选器件：{selected_device_2.get('name', '未知')}"
            selected.append(self._format_selected_device(selected_device_2, reason_2))
            reasons.append(reason_2)
        
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
            "name": device.get("name", "未知器件"),
            "type": device.get("type", "未知类型"),
            "category": device.get("category", "未分类"),
            "parameters": device.get("parameters", {}),
            "interconnect_effects": device.get("interconnect_effects", []),
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
            parsed = self._extract_json(response)
            if parsed is not None:
                return parsed
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
        extracted_count = sum(len(devices) for devices in extracted_devices.get('extracted_devices', {}).values()) if isinstance(extracted_devices, dict) else 0
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