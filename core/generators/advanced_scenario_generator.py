#!/usr/bin/env python3
"""
Advanced Scenario Generator for Aerospace Microsystem Simulation
高级航空航天微系统仿真场景生成器

支持多层次复杂度配置，生成专业级仿真场景
"""

import json
import random
import math
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import yaml


class ComplexityLevel(Enum):
    """仿真复杂度等级"""
    BASIC = "basic"           # 基础级：单一场景，稳定环境
    INTERMEDIATE = "intermediate"  # 中级：多场景，部分动态变化
    ADVANCED = "advanced"     # 高级：复杂场景，动态环境变化
    EXPERT = "expert"         # 专家级：极端场景，多重故障注入
    MISSION_CRITICAL = "mission_critical"  # 任务关键级：真实任务仿真


class MissionType(Enum):
    """任务类型"""
    EARTH_OBSERVATION = "earth_observation"      # 对地观测
    DEEP_SPACE_EXPLORATION = "deep_space"        # 深空探测
    COMMUNICATION_SATELLITE = "communication"    # 通信卫星
    NAVIGATION_SATELLITE = "navigation"          # 导航卫星
    SCIENTIFIC_RESEARCH = "scientific"           # 科学研究
    MILITARY_RECONNAISSANCE = "military"         # 军用侦察


@dataclass
class OrbitParameters:
    """轨道参数"""
    altitude_km: float
    inclination_deg: float
    eccentricity: float
    period_minutes: float
    radiation_level: str  # low, medium, high, extreme
    thermal_cycling: bool
    eclipse_duration_minutes: float


@dataclass
class EnvironmentalConditions:
    """环境条件"""
    temperature_range: Tuple[float, float]  # 温度范围 (°C)
    radiation_dose_rate: float  # 辐射剂量率 (rad/s)
    magnetic_field_strength: float  # 磁场强度 (nT)
    plasma_density: float  # 等离子体密度 (cm^-3)
    micrometeorite_flux: float  # 微流星体通量 (impacts/m²/s)
    solar_activity_level: str  # quiet, moderate, active, storm


@dataclass
class WorkloadProfile:
    """工作负载配置"""
    cpu_utilization_pattern: List[float]  # CPU利用率时间序列
    gpu_compute_intensity: List[float]    # GPU计算强度时间序列
    memory_access_pattern: str            # sequential, random, mixed
    io_burst_frequency: float             # I/O突发频率
    communication_load: List[float]       # 通信负载时间序列
    sensor_sampling_rates: Dict[str, float]  # 传感器采样率


@dataclass
class FaultInjectionProfile:
    """故障注入配置"""
    seu_rate: float                       # 单粒子翻转率
    latchup_probability: float            # 闩锁概率
    thermal_stress_events: List[Dict]     # 热应力事件
    power_fluctuations: List[Dict]        # 电源波动
    communication_dropouts: List[Dict]    # 通信中断
    sensor_degradation: Dict[str, float]  # 传感器性能退化


class AdvancedScenarioGenerator:
    """高级仿真场景生成器"""
    
    def __init__(self):
        self.complexity_configs = self._load_complexity_configs()
        self.mission_templates = self._load_mission_templates()
        
    def _load_complexity_configs(self) -> Dict[str, Dict]:
        """加载复杂度配置"""
        return {
            ComplexityLevel.BASIC.value: {
                "simulation_duration_hours": 1,
                "environment_changes": 0,
                "fault_injection_rate": 0.0,
                "workload_variations": 1,
                "component_count_multiplier": 1.0
            },
            ComplexityLevel.INTERMEDIATE.value: {
                "simulation_duration_hours": 6,
                "environment_changes": 3,
                "fault_injection_rate": 1e-6,
                "workload_variations": 5,
                "component_count_multiplier": 1.5
            },
            ComplexityLevel.ADVANCED.value: {
                "simulation_duration_hours": 24,
                "environment_changes": 10,
                "fault_injection_rate": 1e-5,
                "workload_variations": 15,
                "component_count_multiplier": 2.0
            },
            ComplexityLevel.EXPERT.value: {
                "simulation_duration_hours": 72,
                "environment_changes": 25,
                "fault_injection_rate": 1e-4,
                "workload_variations": 30,
                "component_count_multiplier": 3.0
            },
            ComplexityLevel.MISSION_CRITICAL.value: {
                "simulation_duration_hours": 168,  # 一周
                "environment_changes": 50,
                "fault_injection_rate": 1e-3,
                "workload_variations": 100,
                "component_count_multiplier": 4.0
            }
        }
    
    def _load_mission_templates(self) -> Dict[str, Dict]:
        """加载任务模板"""
        return {
            MissionType.EARTH_OBSERVATION.value: {
                "orbit": OrbitParameters(
                    altitude_km=700,
                    inclination_deg=98.2,
                    eccentricity=0.001,
                    period_minutes=98.8,
                    radiation_level="medium",
                    thermal_cycling=True,
                    eclipse_duration_minutes=35
                ),
                "primary_sensors": ["optical_camera", "infrared_sensor", "radar"],
                "data_processing_load": "high",
                "communication_requirements": "moderate"
            },
            MissionType.DEEP_SPACE_EXPLORATION.value: {
                "orbit": OrbitParameters(
                    altitude_km=150000000,  # 火星轨道
                    inclination_deg=0,
                    eccentricity=0.1,
                    period_minutes=525600,  # 1年
                    radiation_level="extreme",
                    thermal_cycling=False,
                    eclipse_duration_minutes=0
                ),
                "primary_sensors": ["spectrometer", "magnetometer", "particle_detector"],
                "data_processing_load": "extreme",
                "communication_requirements": "critical"
            },
            MissionType.COMMUNICATION_SATELLITE.value: {
                "orbit": OrbitParameters(
                    altitude_km=35786,  # 地球同步轨道
                    inclination_deg=0,
                    eccentricity=0,
                    period_minutes=1436,  # 24小时
                    radiation_level="high",
                    thermal_cycling=True,
                    eclipse_duration_minutes=70
                ),
                "primary_sensors": ["antenna_array", "signal_processor"],
                "data_processing_load": "extreme",
                "communication_requirements": "critical"
            }
        }
    
    def generate_scenario(self, 
                         complexity: ComplexityLevel,
                         mission_type: MissionType,
                         custom_requirements: Optional[Dict] = None) -> Dict[str, Any]:
        """生成仿真场景"""
        
        # 获取基础配置
        complexity_config = self.complexity_configs[complexity.value]
        mission_template = self.mission_templates[mission_type.value]
        
        # 生成轨道参数
        orbit_params = self._generate_orbit_parameters(mission_template["orbit"], complexity)
        
        # 生成环境条件
        env_conditions = self._generate_environmental_conditions(orbit_params, complexity)
        
        # 生成工作负载配置
        workload_profile = self._generate_workload_profile(mission_template, complexity_config)
        
        # 生成故障注入配置
        fault_profile = self._generate_fault_injection_profile(complexity_config, env_conditions)
        
        # 生成系统配置
        system_config = self._generate_system_configuration(mission_template, complexity_config)
        
        # 生成测试序列
        test_sequence = self._generate_test_sequence(complexity_config, workload_profile)
        
        scenario = {
            "metadata": {
                "scenario_id": f"{mission_type.value}_{complexity.value}_{random.randint(1000, 9999)}",
                "complexity_level": complexity.value,
                "mission_type": mission_type.value,
                "generation_timestamp": "2024-01-01T00:00:00Z",
                "estimated_runtime_hours": complexity_config["simulation_duration_hours"]
            },
            "orbit_parameters": asdict(orbit_params),
            "environmental_conditions": asdict(env_conditions),
            "system_configuration": system_config,
            "workload_profile": asdict(workload_profile),
            "fault_injection_profile": asdict(fault_profile),
            "test_sequence": test_sequence,
            "success_criteria": self._generate_success_criteria(complexity, mission_type),
            "performance_targets": self._generate_performance_targets(mission_template, complexity_config)
        }
        
        # 应用自定义需求
        if custom_requirements:
            scenario = self._apply_custom_requirements(scenario, custom_requirements)
        
        return scenario
    
    def _generate_orbit_parameters(self, base_orbit: OrbitParameters, complexity: ComplexityLevel) -> OrbitParameters:
        """生成轨道参数"""
        if complexity in [ComplexityLevel.BASIC, ComplexityLevel.INTERMEDIATE]:
            return base_orbit
        
        # 高级复杂度：添加轨道扰动
        perturbation_factor = 0.1 if complexity == ComplexityLevel.ADVANCED else 0.2
        
        return OrbitParameters(
            altitude_km=base_orbit.altitude_km * (1 + random.uniform(-perturbation_factor, perturbation_factor)),
            inclination_deg=base_orbit.inclination_deg + random.uniform(-5, 5),
            eccentricity=max(0, base_orbit.eccentricity + random.uniform(-0.01, 0.01)),
            period_minutes=base_orbit.period_minutes * (1 + random.uniform(-0.05, 0.05)),
            radiation_level=base_orbit.radiation_level,
            thermal_cycling=base_orbit.thermal_cycling,
            eclipse_duration_minutes=base_orbit.eclipse_duration_minutes * (1 + random.uniform(-0.2, 0.2))
        )
    
    def _generate_environmental_conditions(self, orbit: OrbitParameters, complexity: ComplexityLevel) -> EnvironmentalConditions:
        """生成环境条件"""
        base_temp_ranges = {
            "low": (-100, 50),
            "medium": (-150, 80),
            "high": (-200, 120),
            "extreme": (-250, 150)
        }
        
        temp_range = base_temp_ranges[orbit.radiation_level]
        
        # 根据复杂度调整环境参数
        complexity_multipliers = {
            ComplexityLevel.BASIC: 1.0,
            ComplexityLevel.INTERMEDIATE: 1.2,
            ComplexityLevel.ADVANCED: 1.5,
            ComplexityLevel.EXPERT: 2.0,
            ComplexityLevel.MISSION_CRITICAL: 3.0
        }
        
        multiplier = complexity_multipliers[complexity]
        
        return EnvironmentalConditions(
            temperature_range=temp_range,
            radiation_dose_rate=0.001 * multiplier,
            magnetic_field_strength=50000 / (orbit.altitude_km / 6371) ** 3,
            plasma_density=1e6 / (orbit.altitude_km / 1000) ** 2,
            micrometeorite_flux=1e-12 * multiplier,
            solar_activity_level="moderate" if complexity == ComplexityLevel.BASIC else "active"
        )
    
    def _generate_workload_profile(self, mission_template: Dict, complexity_config: Dict) -> WorkloadProfile:
        """生成工作负载配置"""
        duration_hours = complexity_config["simulation_duration_hours"]
        variations = complexity_config["workload_variations"]
        
        # 生成时间序列数据
        time_points = np.linspace(0, duration_hours, variations)
        
        # CPU利用率模式（基于任务类型）
        base_cpu_load = 0.3 if mission_template["data_processing_load"] == "moderate" else 0.7
        cpu_pattern = [base_cpu_load + 0.3 * math.sin(2 * math.pi * t / 24) + 
                      random.uniform(-0.1, 0.1) for t in time_points]
        
        # GPU计算强度
        base_gpu_load = 0.2 if mission_template["data_processing_load"] == "moderate" else 0.8
        gpu_pattern = [base_gpu_load + 0.2 * math.cos(2 * math.pi * t / 12) + 
                      random.uniform(-0.1, 0.1) for t in time_points]
        
        # 通信负载
        comm_base = 0.1 if mission_template["communication_requirements"] == "moderate" else 0.6
        comm_pattern = [comm_base + 0.4 * random.random() for _ in time_points]
        
        return WorkloadProfile(
            cpu_utilization_pattern=cpu_pattern,
            gpu_compute_intensity=gpu_pattern,
            memory_access_pattern="mixed",
            io_burst_frequency=random.uniform(0.1, 1.0),
            communication_load=comm_pattern,
            sensor_sampling_rates={
                sensor: random.uniform(1.0, 100.0) 
                for sensor in mission_template["primary_sensors"]
            }
        )
    
    def _generate_fault_injection_profile(self, complexity_config: Dict, env_conditions: EnvironmentalConditions) -> FaultInjectionProfile:
        """生成故障注入配置"""
        base_rate = complexity_config["fault_injection_rate"]
        
        # 根据环境条件调整故障率
        radiation_multiplier = {
            "low": 1.0, "medium": 2.0, "high": 5.0, "extreme": 10.0
        }.get(env_conditions.solar_activity_level, 1.0)
        
        return FaultInjectionProfile(
            seu_rate=base_rate * radiation_multiplier,
            latchup_probability=base_rate * 0.1,
            thermal_stress_events=[
                {"time_hours": random.uniform(0, complexity_config["simulation_duration_hours"]),
                 "severity": random.uniform(0.1, 1.0),
                 "duration_minutes": random.uniform(5, 60)}
                for _ in range(complexity_config["environment_changes"] // 3)
            ],
            power_fluctuations=[
                {"time_hours": random.uniform(0, complexity_config["simulation_duration_hours"]),
                 "voltage_drop_percent": random.uniform(5, 20),
                 "duration_seconds": random.uniform(1, 10)}
                for _ in range(complexity_config["environment_changes"] // 2)
            ],
            communication_dropouts=[
                {"time_hours": random.uniform(0, complexity_config["simulation_duration_hours"]),
                 "duration_minutes": random.uniform(1, 30),
                 "signal_loss_percent": random.uniform(50, 100)}
                for _ in range(complexity_config["environment_changes"] // 4)
            ],
            sensor_degradation={
                "optical_camera": random.uniform(0, 0.1),
                "infrared_sensor": random.uniform(0, 0.05),
                "radar": random.uniform(0, 0.02)
            }
        )
    
    def _generate_system_configuration(self, mission_template: Dict, complexity_config: Dict) -> Dict[str, Any]:
        """生成系统配置"""
        multiplier = complexity_config["component_count_multiplier"]
        
        return {
            "cpu": {
                "cores": int(8 * multiplier),
                "freq_ghz": 3.0,
                "cache_hierarchy": ["L1", "L2", "L3"],
                "architecture": "ARM_Cortex_A78" if multiplier > 2 else "x86"
            },
            "gpu": {
                "sm_count": int(80 * multiplier),
                "freq_mhz": 1400,
                "memory_gb": int(16 * multiplier),
                "architecture": "Ampere"
            },
            "memory": {
                "size_gb": int(32 * multiplier),
                "type": "DDR5",
                "ecc_enabled": True
            },
            "sensors": [
                {"type": sensor, "count": int(random.randint(1, 3) * multiplier)}
                for sensor in mission_template["primary_sensors"]
            ],
            "communication": {
                "transceivers": int(2 * multiplier),
                "antennas": int(4 * multiplier),
                "frequency_bands": ["S", "X", "Ka"]
            },
            "power_system": {
                "solar_panels_w": int(1000 * multiplier),
                "battery_capacity_wh": int(500 * multiplier),
                "power_management_units": int(2 * multiplier)
            }
        }
    
    def _generate_test_sequence(self, complexity_config: Dict, workload_profile: WorkloadProfile) -> List[Dict]:
        """生成测试序列"""
        sequence = []
        duration_hours = complexity_config["simulation_duration_hours"]
        
        # 基础功能测试
        sequence.append({
            "phase": "initialization",
            "duration_minutes": 30,
            "tests": ["system_boot", "component_discovery", "health_check"]
        })
        
        # 正常运行测试
        sequence.append({
            "phase": "normal_operation",
            "duration_hours": duration_hours * 0.6,
            "tests": ["nominal_workload", "sensor_data_collection", "communication_test"]
        })
        
        # 压力测试
        if complexity_config["workload_variations"] > 5:
            sequence.append({
                "phase": "stress_testing",
                "duration_hours": duration_hours * 0.2,
                "tests": ["peak_load", "thermal_stress", "memory_stress"]
            })
        
        # 故障恢复测试
        if complexity_config["fault_injection_rate"] > 0:
            sequence.append({
                "phase": "fault_recovery",
                "duration_hours": duration_hours * 0.15,
                "tests": ["fault_injection", "recovery_validation", "redundancy_test"]
            })
        
        # 长期稳定性测试
        if duration_hours > 24:
            sequence.append({
                "phase": "long_term_stability",
                "duration_hours": duration_hours * 0.05,
                "tests": ["aging_simulation", "degradation_analysis", "lifetime_prediction"]
            })
        
        return sequence
    
    def _generate_success_criteria(self, complexity: ComplexityLevel, mission_type: MissionType) -> Dict[str, Any]:
        """生成成功标准"""
        base_criteria = {
            "system_availability_percent": 99.0,
            "data_integrity_percent": 99.9,
            "power_efficiency_target": 0.8,
            "thermal_compliance": True,
            "radiation_tolerance": True
        }
        
        # 根据复杂度调整标准
        complexity_adjustments = {
            ComplexityLevel.BASIC: {"system_availability_percent": 95.0},
            ComplexityLevel.INTERMEDIATE: {"system_availability_percent": 97.0},
            ComplexityLevel.ADVANCED: {"system_availability_percent": 98.5},
            ComplexityLevel.EXPERT: {"system_availability_percent": 99.5},
            ComplexityLevel.MISSION_CRITICAL: {"system_availability_percent": 99.9}
        }
        
        base_criteria.update(complexity_adjustments[complexity])
        
        # 根据任务类型添加特定标准
        mission_specific = {
            MissionType.EARTH_OBSERVATION: {
                "image_quality_score": 0.9,
                "coverage_completeness_percent": 95.0
            },
            MissionType.DEEP_SPACE_EXPLORATION: {
                "communication_reliability_percent": 99.9,
                "scientific_data_quality": 0.95
            },
            MissionType.COMMUNICATION_SATELLITE: {
                "signal_quality_db": -10,
                "throughput_gbps": 10.0
            }
        }
        
        base_criteria.update(mission_specific.get(mission_type, {}))
        
        return base_criteria
    
    def _generate_performance_targets(self, mission_template: Dict, complexity_config: Dict) -> Dict[str, Any]:
        """生成性能目标"""
        return {
            "computational_performance": {
                "cpu_throughput_gops": 100 * complexity_config["component_count_multiplier"],
                "gpu_throughput_tflops": 50 * complexity_config["component_count_multiplier"],
                "memory_bandwidth_gbps": 500 * complexity_config["component_count_multiplier"]
            },
            "power_performance": {
                "total_power_budget_w": 300 * complexity_config["component_count_multiplier"],
                "power_efficiency_gops_per_w": 0.5,
                "battery_life_hours": 72
            },
            "reliability_targets": {
                "mtbf_hours": 100000,
                "seu_tolerance_rate": 1e-12,
                "thermal_cycling_endurance": 10000
            },
            "communication_performance": {
                "data_rate_mbps": 100,
                "latency_ms": 50,
                "error_rate": 1e-9
            }
        }
    
    def _apply_custom_requirements(self, scenario: Dict[str, Any], custom_requirements: Dict) -> Dict[str, Any]:
        """应用自定义需求"""
        # 深度合并自定义需求
        def deep_merge(base_dict, custom_dict):
            for key, value in custom_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_merge(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_merge(scenario, custom_requirements)
        return scenario
    
    def export_scenario(self, scenario: Dict[str, Any], output_path: str, format: str = "yaml") -> None:
        """导出场景配置"""
        if format.lower() == "yaml":
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(scenario, f, default_flow_style=False, allow_unicode=True, indent=2)
        elif format.lower() == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(scenario, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}")


def main():
    """主函数 - 演示场景生成器使用"""
    generator = AdvancedScenarioGenerator()
    
    # 生成不同复杂度的场景
    scenarios = []
    
    for complexity in [ComplexityLevel.INTERMEDIATE, ComplexityLevel.ADVANCED, ComplexityLevel.EXPERT]:
        for mission in [MissionType.EARTH_OBSERVATION, MissionType.DEEP_SPACE_EXPLORATION]:
            scenario = generator.generate_scenario(complexity, mission)
            scenarios.append(scenario)
            
            # 导出场景
            filename = f"scenario_{mission.value}_{complexity.value}.yaml"
            generator.export_scenario(scenario, filename)
            print(f"Generated scenario: {filename}")
    
    print(f"Total scenarios generated: {len(scenarios)}")


if __name__ == "__main__":
    main()