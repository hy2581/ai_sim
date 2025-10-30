#!/usr/bin/env python3
"""
Independent Chiplet Performance Testing Parser
Parses natural language descriptions for independent chiplet performance testing
"""

import json
import re
import requests
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict


@dataclass
class ChipletTask:
    task_type: str = "compute_intensive"  # compute_intensive, memory_intensive, mixed
    operations: List[str] = None  # ["add", "mul", "div", "func_call"]
    iterations: int = 1000000
    data_size: int = 1024  # KB
    complexity: str = "medium"  # low, medium, high
    
    def __post_init__(self):
        if self.operations is None:
            self.operations = ["add", "mul", "div"]


@dataclass
class CPUConfig:
    arch: str = "x86"
    cores: int = 8
    freq_ghz: float = 3.5
    cache_l1_kb: int = 32
    cache_l2_kb: int = 256
    cache_l3_mb: int = 8
    task: ChipletTask = None
    
    def __post_init__(self):
        if self.task is None:
            self.task = ChipletTask()


@dataclass
class GPUConfig:
    arch: str = "ampere"
    sm_count: int = 80
    mem_gb: int = 16
    freq_mhz: int = 1400
    mem_bw_gbps: int = 900
    task: ChipletTask = None
    
    def __post_init__(self):
        if self.task is None:
            self.task = ChipletTask(task_type="parallel_compute", 
                                  operations=["matrix_ops", "vector_ops", "parallel_add"])


@dataclass
class MemoryConfig:
    type: str = "DDR4"
    size_gb: int = 32
    freq_mhz: int = 3200
    channels: int = 4


@dataclass
class SystemConfig:
    cpu: CPUConfig
    gpu: GPUConfig
    memory: MemoryConfig
    test_duration_sec: int = 60
    performance_metrics: List[str] = None
    
    def __post_init__(self):
        if self.performance_metrics is None:
            self.performance_metrics = ["throughput", "latency", "power", "utilization"]


class IndependentChipletParser:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        
    def _create_prompt(self, description: str) -> str:
        """Create structured prompt for independent chiplet testing"""
        prompt = """
你是一个计算机系统性能测试专家。请将以下自然语言描述转换为独立芯粒性能测试的JSON配置。

输入描述: """ + description + """

请严格按照以下JSON格式输出，不要添加任何其他文字：

{
  "cpu": {
    "arch": "x86或arm",
    "cores": 核心数量,
    "freq_ghz": 主频GHz,
    "cache_l1_kb": 32,
    "cache_l2_kb": 256,
    "cache_l3_mb": 8,
    "task": {
      "task_type": "compute_intensive或memory_intensive或mixed",
      "operations": ["add", "mul", "div", "func_call", "loop"],
      "iterations": 迭代次数,
      "data_size": 数据大小KB,
      "complexity": "low或medium或high"
    }
  },
  "gpu": {
    "arch": "ampere或turing或pascal",
    "sm_count": SM数量,
    "mem_gb": 显存GB,
    "freq_mhz": 1400,
    "mem_bw_gbps": 900,
    "task": {
      "task_type": "parallel_compute或memory_bound或mixed",
      "operations": ["matrix_ops", "vector_ops", "parallel_add", "parallel_mul"],
      "iterations": 迭代次数,
      "data_size": 数据大小KB,
      "complexity": "low或medium或high"
    }
  },
  "memory": {
    "type": "DDR4或DDR5",
    "size_gb": 内存大小GB,
    "freq_mhz": 3200,
    "channels": 4
  },
  "test_duration_sec": 测试时长秒数,
  "performance_metrics": ["throughput", "latency", "power", "utilization"]
}

解析规则：
- 从描述中提取CPU和GPU的硬件配置
- 根据描述推断每个芯粒应该执行的独立任务类型
- 如果提到"重复的加减乘除"，设置operations为["add", "sub", "mul", "div"]
- 如果提到"函数调用"，添加"func_call"到operations
- 根据任务复杂度设置iterations和complexity
- 如果没有明确指定，使用合理的默认值
"""
        return prompt

    def parse(self, description: str) -> Optional[SystemConfig]:
        """Parse natural language description to SystemConfig for independent testing"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "user",
                        "content": self._create_prompt(description)
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 1500
            }
            
            response = requests.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in response")
                
            config_dict = json.loads(json_match.group())
            
            # Convert to SystemConfig
            cpu_dict = config_dict["cpu"]
            cpu_task = ChipletTask(**cpu_dict["task"]) if "task" in cpu_dict else ChipletTask()
            cpu_config = CPUConfig(
                arch=cpu_dict["arch"],
                cores=cpu_dict["cores"],
                freq_ghz=cpu_dict["freq_ghz"],
                cache_l1_kb=cpu_dict.get("cache_l1_kb", 32),
                cache_l2_kb=cpu_dict.get("cache_l2_kb", 256),
                cache_l3_mb=cpu_dict.get("cache_l3_mb", 8),
                task=cpu_task
            )
            
            gpu_dict = config_dict["gpu"]
            gpu_task = ChipletTask(**gpu_dict["task"]) if "task" in gpu_dict else ChipletTask()
            gpu_config = GPUConfig(
                arch=gpu_dict["arch"],
                sm_count=gpu_dict["sm_count"],
                mem_gb=gpu_dict["mem_gb"],
                freq_mhz=gpu_dict.get("freq_mhz", 1400),
                mem_bw_gbps=gpu_dict.get("mem_bw_gbps", 900),
                task=gpu_task
            )
            
            memory_config = MemoryConfig(**config_dict["memory"])
            
            system_config = SystemConfig(
                cpu=cpu_config,
                gpu=gpu_config,
                memory=memory_config,
                test_duration_sec=config_dict.get("test_duration_sec", 60),
                performance_metrics=config_dict.get("performance_metrics", 
                                                  ["throughput", "latency", "power", "utilization"])
            )
            
            return system_config
            
        except Exception as e:
            print(f"Error parsing description: {e}")
            return None
    
    def config_to_dict(self, config: SystemConfig) -> Dict[str, Any]:
        """Convert SystemConfig to dictionary"""
        return asdict(config)


def main():
    """Test the independent chiplet parser"""
    api_key = "sk-c5446da718c24bfab2536bf0b2d5abb0"
    parser = IndependentChipletParser(api_key)
    
    # Test with independent chiplet description
    description = "测试一个8核x86 CPU和80SM Ampere GPU的独立性能，每个芯粒运行重复的加减乘除和函数调用运算，持续60秒"
    
    config = parser.parse(description)
    if config:
        print("Parsed independent chiplet configuration:")
        print(json.dumps(parser.config_to_dict(config), indent=2, ensure_ascii=False))
    else:
        print("Failed to parse description")


if __name__ == "__main__":
    main()