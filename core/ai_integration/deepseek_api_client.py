#!/usr/bin/env python3
"""
DeepSeek API Client
用于处理自然语言输入并生成航空航天微系统需求定义
"""

import os
import json
import requests
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class DeepSeekConfig:
    """DeepSeek API配置"""
    api_key: str
    base_url: str = "https://api.deepseek.com/v1/chat/completions"
    model: str = "deepseek-chat"
    max_tokens: int = 4000
    temperature: float = 0.7
    model_type: str = "qwen0.5b"  # 支持: qwen0.5b, deepseek-api, deepseek-r1


class DeepSeekAPIClient:
    """DeepSeek/Ollama API客户端 - 支持多种模型"""
    
    def __init__(self, api_key: str = None, model_type: str = "qwen0.5b"):
        """
        初始化客户端
        
        Args:
            api_key: DeepSeek API密钥 (仅 deepseek-api 需要)
            model_type: 模型类型
                - qwen0.5b: 本地 Ollama 小模型 (默认)
                - deepseek-api: 外部 DeepSeek API
                - deepseek-r1: 本地 Ollama DeepSeek-R1
        """
        self.model_type = model_type
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY', 'dummy_key')
        self.config = DeepSeekConfig(
            api_key=self.api_key,
            model_type=model_type
        )
        self.session = requests.Session()
        
        # 根据模型类型设置不同的配置
        if model_type == "deepseek-api":
            # 外部 DeepSeek API
            self.config.base_url = "https://api.deepseek.com/v1/chat/completions"
            self.config.model = "deepseek-chat"
            self.session.headers.update({
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            })
        else:
            # 本地 Ollama (qwen0.5b 或 deepseek-r1)
            self.config.base_url = "http://localhost:11434/api/chat"
            if model_type == "qwen0.5b":
                self.config.model = "qwen:0.5b"
            elif model_type == "deepseek-r1":
                self.config.model = "deepseek-r1"
            self.session.headers.update({
                'Content-Type': 'application/json'
            })
        
        print(f"🤖 初始化AI客户端: {model_type}")
        print(f"   模型: {self.config.model}")
        print(f"   接口: {'外部API' if model_type == 'deepseek-api' else '本地Ollama'}")
    
    def _truncate(self, text: str, limit: int = 12000) -> str:
        if not text:
            return ""
        t = str(text)
        return t[:limit]

    def _post_with_retries(self, url: str, payload: dict, timeout: int = 120, retries: int = 3) -> requests.Response:
        last_exc = None
        for i in range(retries):
            try:
                resp = self.session.post(url, json=payload, timeout=timeout)
                return resp
            except requests.exceptions.RequestException as e:
                last_exc = e
                time.sleep(1 + i)
        raise last_exc

    def parse_natural_language_requirements(self, user_input: str) -> Dict[str, Any]:
        """解析自然语言需求描述"""
        print(f"正在解析用户需求: {user_input}")
        
        system_prompt = """
你是一个航空航天微系统需求分析专家。请根据用户的自然语言描述，提取并分析航空航天微系统的需求，并以JSON格式返回结构化的需求定义。

请分析以下几个方面：
1. 应用场景和任务类型
2. 性能需求（计算、存储、通信）
3. 实时性要求
4. 可靠性和环境要求
5. 功耗约束
6. 器件类型偏好

返回格式应该是JSON，包含以下字段：
{
    "application_scenario": "应用场景描述",
    "task_types": ["任务类型列表"],
    "performance_requirements": {
        "cpu_performance": "CPU性能要求",
        "gpu_performance": "GPU性能要求", 
        "memory_requirements": "内存要求",
        "storage_requirements": "存储要求"
    },
    "realtime_requirements": {
        "max_latency_ms": "最大延迟要求",
        "min_frequency_hz": "最小频率要求"
    },
    "reliability_requirements": {
        "mtbf_hours": "平均故障间隔时间",
        "radiation_tolerance": "辐射容忍度",
        "temperature_range": "工作温度范围"
    },
    "power_constraints": {
        "max_power_w": "最大功耗",
        "battery_life_hours": "电池寿命要求"
    },
    "preferred_devices": {
        "cpu_architecture": "CPU架构偏好",
        "gpu_architecture": "GPU架构偏好",
        "memory_type": "内存类型偏好"
    }
}
"""
        
        try:
            response = self._call_api(system_prompt, user_input)
            
            # 尝试解析JSON响应
            try:
                requirements = json.loads(response)
                print("需求解析成功")
                return requirements
            except json.JSONDecodeError:
                # 如果不是标准JSON，尝试提取JSON部分
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    requirements = json.loads(json_match.group())
                    print("需求解析成功（从文本中提取）")
                    return requirements
                else:
                    print("无法解析JSON，使用默认需求")
                    return self._get_default_requirements()
                    
        except Exception as e:
            print(f"API调用失败: {e}")
            print("使用默认需求配置")
            return self._get_default_requirements()
    
    def generate_task_analysis(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """基于需求生成任务分析"""
        print("正在生成任务分析...")
        
        system_prompt = """
你是航空航天微系统任务分析专家。基于给定的需求，分析需要执行的具体任务，并为每个任务定义性能基准测试方案。

请分析以下任务类型并给出测试方案：
1. 导航计算任务（如卡尔曼滤波）
2. 控制算法任务（如PID控制）
3. 图像处理任务（如实时图像分析）
4. 通信处理任务
5. 传感器数据处理任务

对每个任务，请定义：
- 算法复杂度
- 计算需求
- 内存访问模式
- 实时性要求
- 测试参数

返回JSON格式的任务分析结果。
"""
        
        try:
            requirements_text = json.dumps(requirements, ensure_ascii=False, indent=2)
            response = self._call_api(system_prompt, f"需求定义：\n{requirements_text}")
            
            try:
                task_analysis = json.loads(response)
                print("✅ 任务分析生成成功")
                return task_analysis
            except json.JSONDecodeError:
                print("⚠️ 任务分析解析失败，使用默认分析")
                return self._get_default_task_analysis()
                
        except Exception as e:
            print(f"❌ 任务分析生成失败: {e}")
            return self._get_default_task_analysis()
    
    def generate_device_mapping_strategy(self, requirements: Dict[str, Any], 
                                       task_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """生成器件映射策略"""
        print("正在生成器件映射策略...")
        
        system_prompt = """
你是航空航天微系统器件选型专家。基于需求和任务分析，制定器件映射策略。

请分析以下器件类型的选型策略：
1. CPU选型（架构、核数、频率）
2. GPU选型（架构、SM数量、内存）
3. 内存选型（类型、容量、带宽）
4. 专用器件选型（传感器、通信、控制器）

对每种器件，请考虑：
- 性能需求匹配
- 功耗约束
- 可靠性要求
- 成本效益
- 航空航天标准合规性

返回JSON格式的器件映射策略。
"""
        
        try:
            context = {
                "requirements": requirements,
                "task_analysis": task_analysis
            }
            context_text = json.dumps(context, ensure_ascii=False, indent=2)
            response = self._call_api(system_prompt, f"分析上下文：\n{context_text}")
            
            try:
                mapping_strategy = json.loads(response)
                print("✅ 器件映射策略生成成功")
                return mapping_strategy
            except json.JSONDecodeError:
                print("⚠️ 器件映射策略解析失败，使用默认策略")
                return self._get_default_mapping_strategy()
                
        except Exception as e:
            print(f"❌ 器件映射策略生成失败: {e}")
            return self._get_default_mapping_strategy()
    
    def generate_verification_plan(self, requirements: Dict[str, Any],
                                 device_config: Dict[str, Any]) -> Dict[str, Any]:
        """生成验证计划"""
        print("正在生成验证计划...")
        
        system_prompt = """
你是航空航天微系统验证专家。基于需求和器件配置，制定详细的验证计划。

请制定以下验证项目：
1. 性能验证（计算、内存、通信）
2. 实时性验证
3. 可靠性验证
4. 功耗验证
5. 环境适应性验证
6. 标准合规性验证

对每个验证项目，请定义：
- 验证目标
- 测试方法
- 通过标准
- 风险评估

返回JSON格式的验证计划。
"""
        
        try:
            context = {
                "requirements": requirements,
                "device_config": device_config
            }
            context_text = json.dumps(context, ensure_ascii=False, indent=2)
            response = self._call_api(system_prompt, f"验证上下文：\n{context_text}")
            
            try:
                verification_plan = json.loads(response)
                print("✅ 验证计划生成成功")
                return verification_plan
            except json.JSONDecodeError:
                print("⚠️ 验证计划解析失败，使用默认计划")
                return self._get_default_verification_plan()
                
        except Exception as e:
            print(f"❌ 验证计划生成失败: {e}")
            return self._get_default_verification_plan()
    
    def generate_final_report(self, all_results: Dict[str, Any]) -> str:
        """生成最终报告"""
        print("正在生成最终报告...")
        
        system_prompt = """
你是航空航天微系统报告撰写专家。基于完整的分析和验证结果，生成专业的航空航天微系统产品需求定义模型仿真报告。

报告应该包含以下章节：
1. 报告概要
2. 执行摘要
3. 阶段1: 执行任务需求定义
4. 阶段2: 器件需求定义
5. 阶段3: 仿真验证
6. 需求映射关系验证
7. 航空航天标准合规性验证
8. 结论与建议

请使用专业的技术语言，包含详细的数据表格和分析结果。
报告格式应该是Markdown格式，适合技术文档。
"""
        
        try:
            results_text = json.dumps(all_results, ensure_ascii=False, indent=2)
            response = self._call_api(system_prompt, f"分析结果：\n{results_text}")
            
            print("✅ 最终报告生成成功")
            return response
                
        except Exception as e:
            print(f"❌ 最终报告生成失败: {e}")
            return self._get_default_report()
    
    def _call_api(self, system_prompt: str, user_message: str) -> str:
        """调用DeepSeek API或本地Ollama"""
        
        if self.model_type == "deepseek-api":
            # 使用外部 DeepSeek API
            return self._call_deepseek_api(system_prompt, user_message)
        else:
            # 使用本地 Ollama
            return self._call_ollama_local(system_prompt, user_message)
    
    def _call_deepseek_api(self, system_prompt: str, user_message: str) -> str:
        """调用外部DeepSeek API"""
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": self._truncate(system_prompt)},
                {"role": "user", "content": self._truncate(user_message)}
            ],
            "max_tokens": min(self.config.max_tokens, 4000),
            "temperature": self.config.temperature
        }
        try:
            response = self._post_with_retries(
                "https://api.deepseek.com/v1/chat/completions",
                payload,
                timeout=120,
                retries=3
            )
            response.raise_for_status()
            result = response.json()
            content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            return content
        except requests.exceptions.HTTPError as e:
            try:
                err = response.json()
                msg = err.get('error', str(e))
                print(f"DeepSeek API错误: {msg}")
            except Exception:
                print(f"DeepSeek API错误: {e}")
            raise
        except requests.exceptions.RequestException as e:
            print(f"DeepSeek API请求失败: {e}")
            raise
        except KeyError as e:
            print(f"DeepSeek API响应格式错误: {e}")
            raise
    
    def _call_ollama_local(self, system_prompt: str, user_message: str) -> str:
        """调用本地Ollama模型"""
        # 组合 system prompt 和 user message
        combined_message = f"{system_prompt}\n\n用户输入:\n{user_message}"
        
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "user", "content": combined_message}
            ],
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens
            }
        }
        
        try:
            response = self.session.post(
                "http://localhost:11434/api/chat",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            content = result.get('message', {}).get('content', '')
            
            # 清理可能的格式问题
            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            return content
            
        except requests.exceptions.ConnectionError:
            error_msg = (
                f"无法连接到Ollama服务 (模型: {self.config.model})\n"
                "请确保:\n"
                "1. Ollama服务正在运行: ollama serve\n"
                "2. 模型已安装: ollama list\n"
                f"3. 如果使用 deepseek-r1，需要足够的内存/显卡"
            )
            print(f"❌ {error_msg}")
            raise Exception(error_msg)
        except requests.exceptions.RequestException as e:
            error_detail = str(e)
            # 检查是否是内存不足错误
            try:
                error_json = response.json()
                error_detail = error_json.get('error', error_detail)
            except:
                pass
            
            if "system memory" in error_detail.lower():
                error_msg = (
                    f"模型 {self.config.model} 需要的内存超过系统可用内存\n"
                    "建议:\n"
                    "- 使用 qwen0.5b 小模型 (需要 ~400MB)\n"
                    "- 或在有更大内存/显卡的机器上使用 deepseek-r1\n"
                    "- 或使用外部 deepseek-api"
                )
                print(f"❌ {error_msg}")
                raise Exception(error_msg)
            
            print(f"Ollama请求失败: {e}")
            raise
        except KeyError as e:
            print(f"Ollama响应格式错误: {e}")
            raise
    
    def _get_default_requirements(self) -> Dict[str, Any]:
        """获取默认需求配置"""
        return {
            "application_scenario": "航空航天微系统应用",
            "task_types": ["导航计算", "姿态控制", "图像处理"],
            "performance_requirements": {
                "cpu_performance": "中等性能多核处理器",
                "gpu_performance": "并行计算能力",
                "memory_requirements": "大容量高带宽内存",
                "storage_requirements": "可靠存储"
            },
            "realtime_requirements": {
                "max_latency_ms": 10,
                "min_frequency_hz": 1000
            },
            "reliability_requirements": {
                "mtbf_hours": 10000,
                "radiation_tolerance": "100 krad",
                "temperature_range": "-55°C to +125°C"
            },
            "power_constraints": {
                "max_power_w": 500,
                "battery_life_hours": 24
            },
            "preferred_devices": {
                "cpu_architecture": "ARM",
                "gpu_architecture": "ampere",
                "memory_type": "DDR5"
            }
        }
    
    def _get_default_task_analysis(self) -> Dict[str, Any]:
        """获取默认任务分析"""
        return {
            "navigation_task": {
                "algorithm": "卡尔曼滤波",
                "complexity": "中等",
                "cpu_requirement": "1.5 GIPS",
                "memory_pattern": "矩阵运算密集"
            },
            "control_task": {
                "algorithm": "PID控制",
                "complexity": "低",
                "realtime_requirement": "10ms响应",
                "frequency": "1kHz"
            },
            "image_processing_task": {
                "algorithm": "实时图像处理",
                "complexity": "高",
                "gpu_requirement": "0.6 GFLOPS",
                "memory_bandwidth": "高带宽需求"
            }
        }
    
    def _get_default_mapping_strategy(self) -> Dict[str, Any]:
        """获取默认映射策略"""
        return {
            "cpu_strategy": {
                "architecture": "ARM",
                "cores": 8,
                "frequency": "2.5GHz",
                "rationale": "平衡性能和功耗"
            },
            "gpu_strategy": {
                "architecture": "ampere",
                "sm_count": 80,
                "memory": "16GB GDDR6",
                "rationale": "满足并行计算需求"
            },
            "memory_strategy": {
                "type": "DDR5",
                "capacity": "64GB",
                "frequency": "4800MHz",
                "rationale": "高带宽低延迟"
            }
        }
    
    def _get_default_verification_plan(self) -> Dict[str, Any]:
        """获取默认验证计划"""
        return {
            "performance_verification": {
                "cpu_test": "基准测试",
                "gpu_test": "并行计算测试",
                "memory_test": "带宽延迟测试"
            },
            "realtime_verification": {
                "latency_test": "响应时间测试",
                "frequency_test": "采样频率测试"
            },
            "reliability_verification": {
                "mtbf_analysis": "可靠性分析",
                "environmental_test": "环境适应性测试"
            }
        }
    
    def _get_default_report(self) -> str:
        """获取默认报告模板"""
        return """
# 航空航天微系统产品需求定义模型仿真报告

## 报告概要

**项目名称**: 微系统产品需求定义模型构建  
**报告类型**: 仿真验证报告  
**生成时间**: 自动生成  
**验证方法**: 两阶段分离式验证

## 1. 执行摘要

本次仿真采用创新的两阶段分离式验证方法，通过DeepSeek AI辅助分析用户需求，结合项目自带的仿真程序，自动生成完整的验证报告。

## 2. 阶段1: 执行任务需求定义

基于AI分析的用户需求，定义了系统执行任务的性能要求。

## 3. 阶段2: 器件需求定义

通过AI辅助的器件映射策略，将任务需求转换为具体的器件参数需求。

## 4. 阶段3: 仿真验证

使用项目自带的仿真程序验证器件配置是否满足任务需求。

## 5. 结论与建议

基于AI分析和仿真验证的结果，系统能够满足用户定义的需求。

---
**报告编制**: AI辅助航空航天微系统仿真平台  
**生成方式**: 自动化生成  
"""


def main():
    """测试函数"""
    client = DeepSeekAPIClient()
    
    # 测试需求解析
    test_input = "我需要一个用于无人机导航的微系统，要求实时性好，功耗低，能在恶劣环境下工作"
    requirements = client.parse_natural_language_requirements(test_input)
    print("需求解析结果:", json.dumps(requirements, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()