#!/usr/bin/env python3
"""
Independent Chiplet Parser with Aerospace Enhancements
Parses natural language descriptions into system configurations for aerospace microsystem simulation
"""

import json
import re
import requests
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field


@dataclass
class AerospaceReliabilityConfig:
    """航空航天可靠性配置"""
    radiation_hardening: bool = True
    seu_tolerance: float = 1e-12  # 单粒子翻转容错率
    tid_resistance: float = 100.0  # 总剂量抗性 (krad)
    temperature_range: tuple = (-55, 125)  # 工作温度范围 (°C)
    mtbf_hours: float = 100000  # 平均故障间隔时间
    aging_model: str = "NBTI_PBTI"  # 老化模型
    ecc_enabled: bool = True  # 纠错码
    redundancy_level: int = 2  # 冗余级别


@dataclass
class PowerManagementConfig:
    """功耗管理配置"""
    dvfs_enabled: bool = True  # 动态电压频率调节
    power_gating: bool = True  # 功率门控
    thermal_throttling: bool = True  # 热节流
    low_power_modes: List[str] = field(default_factory=lambda: ["sleep", "deep_sleep", "hibernate"])
    max_power_watts: float = 50.0
    idle_power_watts: float = 5.0


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
    # 航空航天增强特性
    aerospace_reliability: AerospaceReliabilityConfig = field(default_factory=AerospaceReliabilityConfig)
    power_management: PowerManagementConfig = field(default_factory=PowerManagementConfig)
    
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
    # 航空航天增强特性
    aerospace_reliability: AerospaceReliabilityConfig = field(default_factory=AerospaceReliabilityConfig)
    power_management: PowerManagementConfig = field(default_factory=PowerManagementConfig)
    
    def __post_init__(self):
        if self.task is None:
            self.task = ChipletTask(task_type="parallel_compute", 
                                  operations=["matrix_ops", "vector_ops", "parallel_add"])


@dataclass
class SensorConfig:
    """传感器器件配置"""
    sensor_type: str = "IMU"  # IMU, environmental, optical, radiation
    sampling_rate_hz: float = 1000.0
    resolution_bits: int = 16
    accuracy_percent: float = 99.5
    power_consumption_mw: float = 10.0
    # 航空航天特性
    aerospace_reliability: AerospaceReliabilityConfig = field(default_factory=AerospaceReliabilityConfig)
    power_management: PowerManagementConfig = field(default_factory=PowerManagementConfig)


@dataclass
class CommunicationConfig:
    """通信器件配置"""
    comm_type: str = "RF_transceiver"  # RF_transceiver, antenna_controller, signal_processor
    frequency_band: str = "S_band"  # S_band, X_band, Ka_band
    data_rate_mbps: float = 100.0
    transmission_power_w: float = 5.0
    range_km: float = 1000.0
    # 航空航天特性
    aerospace_reliability: AerospaceReliabilityConfig = field(default_factory=AerospaceReliabilityConfig)
    power_management: PowerManagementConfig = field(default_factory=PowerManagementConfig)


@dataclass
class ControllerConfig:
    """控制器件配置"""
    controller_type: str = "attitude_controller"  # attitude_controller, power_manager, thermal_controller
    control_frequency_hz: float = 100.0
    response_time_ms: float = 10.0
    accuracy_degrees: float = 0.1  # 对于姿态控制器
    power_consumption_w: float = 2.0
    # 航空航天特性
    aerospace_reliability: AerospaceReliabilityConfig = field(default_factory=AerospaceReliabilityConfig)
    power_management: PowerManagementConfig = field(default_factory=PowerManagementConfig)


@dataclass
class MemoryConfig:
    type: str = "DDR4"
    size_gb: int = 32
    freq_mhz: int = 3200
    channels: int = 4
    # 航空航天增强特性
    aerospace_reliability: AerospaceReliabilityConfig = field(default_factory=AerospaceReliabilityConfig)
    power_management: PowerManagementConfig = field(default_factory=PowerManagementConfig)


@dataclass
class SystemConfig:
    cpu: CPUConfig
    gpu: GPUConfig
    memory: MemoryConfig
    # 新增航空航天器件
    sensors: List[SensorConfig] = field(default_factory=list)
    communications: List[CommunicationConfig] = field(default_factory=list)
    controllers: List[ControllerConfig] = field(default_factory=list)
    # 测试配置
    test_duration_sec: int = 60
    performance_metrics: List[str] = None
    # 环境配置
    environment_config: Dict[str, Any] = field(default_factory=lambda: {
        "radiation_level": "low",  # low, medium, high
        "temperature_c": 25.0,
        "vibration_level": "low",
        "vacuum_environment": False
    })
    
    def __post_init__(self):
        if self.performance_metrics is None:
            self.performance_metrics = ["throughput", "latency", "power", "utilization", "reliability", "thermal"]
        
        # 默认添加基本的航空航天器件
        if not self.sensors:
            self.sensors = [
                SensorConfig(sensor_type="IMU"),
                SensorConfig(sensor_type="environmental"),
                SensorConfig(sensor_type="radiation")
            ]
        
        if not self.communications:
            self.communications = [
                CommunicationConfig(comm_type="RF_transceiver", frequency_band="S_band")
            ]
            
        if not self.controllers:
            self.controllers = [
                ControllerConfig(controller_type="attitude_controller"),
                ControllerConfig(controller_type="power_manager"),
                ControllerConfig(controller_type="thermal_controller")
            ]


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
            # 尝试使用API解析
            if self.api_key and self.api_key != "dummy_key":
                return self._parse_with_api(description)
            else:
                # 使用本地简单解析
                return self._parse_locally(description)
        except Exception as e:
            print(f"Error parsing description: {e}")
            # 返回默认配置
            return self._get_default_config()
    
    def _parse_with_api(self, description: str) -> Optional[SystemConfig]:
        """使用API解析描述"""
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
    
    def _parse_locally(self, description: str) -> SystemConfig:
        """本地简单解析，基于关键词"""
        # 提取CPU核心数
        cores = 8
        if "16核" in description or "16-core" in description:
            cores = 16
        elif "4核" in description or "4-core" in description:
            cores = 4
        
        # 提取CPU架构
        arch = "x86"
        if "ARM" in description or "arm" in description:
            arch = "ARM"
        
        # 提取GPU SM数量
        sm_count = 80
        if "120SM" in description:
            sm_count = 120
        elif "60SM" in description:
            sm_count = 60
        
        # 创建传感器配置
        sensors = []
        if "IMU" in description or "惯性" in description:
            sensors.append(SensorConfig(sensor_type="IMU"))
        if "环境传感器" in description or "environmental" in description:
            sensors.append(SensorConfig(sensor_type="environmental"))
        
        # 创建通信配置
        communications = []
        if "射频" in description or "RF" in description or "通信" in description:
            communications.append(CommunicationConfig(comm_type="RF_transceiver"))
        
        # 创建控制器配置
        controllers = []
        if "姿态控制" in description or "attitude" in description:
            controllers.append(ControllerConfig(controller_type="attitude_controller"))
        if "电源管理" in description or "power" in description:
            controllers.append(ControllerConfig(controller_type="power_manager"))
        if "热控制" in description or "thermal" in description:
            controllers.append(ControllerConfig(controller_type="thermal_controller"))
        
        return SystemConfig(
            cpu=CPUConfig(arch=arch, cores=cores, freq_ghz=2.5),
            gpu=GPUConfig(arch="ampere", sm_count=sm_count, mem_gb=16),
            memory=MemoryConfig(type="DDR4", size_gb=32),
            sensors=sensors,
            communications=communications,
            controllers=controllers
        )
    
    def _get_default_config(self) -> SystemConfig:
        """获取默认配置"""
        return SystemConfig(
            cpu=CPUConfig(arch="x86", cores=8, freq_ghz=2.5),
            gpu=GPUConfig(arch="ampere", sm_count=80, mem_gb=16),
            memory=MemoryConfig(type="DDR4", size_gb=32),
            sensors=[SensorConfig(sensor_type="IMU")],
            communications=[CommunicationConfig(comm_type="RF_transceiver")],
            controllers=[ControllerConfig(controller_type="attitude_controller")]
        )
    
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