#!/usr/bin/env python3
"""
Aerospace Performance Analyzer with Power Analysis and AI Feasibility Assessment
Enhanced performance analyzer for aerospace microsystem simulation
"""

import json
import math
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import requests


@dataclass
class PowerAnalysis:
    """功耗分析结果"""
    total_power_w: float
    cpu_power_w: float
    gpu_power_w: float
    memory_power_w: float
    sensors_power_w: float
    communications_power_w: float
    controllers_power_w: float
    thermal_power_w: float
    power_efficiency: float
    battery_life_hours: float


@dataclass
class ReliabilityAnalysis:
    """可靠性分析结果"""
    overall_mtbf_hours: float
    radiation_impact_score: float
    thermal_stress_score: float
    aging_degradation_percent: float
    fault_tolerance_level: str
    redundancy_effectiveness: float


@dataclass
class AIFeasibilityAssessment:
    """AI可行性评估"""
    feasibility_score: float  # 0-100
    technical_risks: List[str]
    performance_bottlenecks: List[str]
    optimization_suggestions: List[str]
    implementation_complexity: str  # low, medium, high
    estimated_development_time_months: int
    confidence_level: float


class AerospacePerformanceAnalyzer:
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        
    def analyze_power_consumption(self, config: Dict[str, Any]) -> PowerAnalysis:
        """分析系统功耗"""
        # CPU功耗计算
        cpu_base_power = config['cpu']['cores'] * 15  # 每核心15W基础功耗
        cpu_freq_factor = config['cpu']['freq_ghz'] / 3.0
        cpu_power = cpu_base_power * cpu_freq_factor
        
        # GPU功耗计算
        gpu_base_power = config['gpu']['sm_count'] * 2  # 每SM 2W基础功耗
        gpu_freq_factor = config['gpu']['freq_mhz'] / 1400
        gpu_power = gpu_base_power * gpu_freq_factor
        
        # 内存功耗计算
        memory_power = config['memory']['size_gb'] * 0.5  # 每GB 0.5W
        
        # 传感器功耗计算
        sensors_power = len(config.get('sensors', [])) * 0.01  # 每个传感器10mW
        
        # 通信器件功耗计算
        comm_power = 0
        for comm in config.get('communications', []):
            if comm.get('comm_type') == 'RF_transceiver':
                comm_power += 5.0  # RF收发器5W
            else:
                comm_power += 2.0  # 其他通信器件2W
        
        # 控制器功耗计算
        controllers_power = len(config.get('controllers', [])) * 2.0  # 每个控制器2W
        
        # 热管理功耗（总功耗的10%）
        subtotal = cpu_power + gpu_power + memory_power + sensors_power + comm_power + controllers_power
        thermal_power = subtotal * 0.1
        
        total_power = subtotal + thermal_power
        
        # 功耗效率计算（性能/功耗比）
        performance_score = (config['cpu']['cores'] * config['cpu']['freq_ghz'] + 
                           config['gpu']['sm_count'] * config['gpu']['freq_mhz'] / 1000)
        power_efficiency = performance_score / total_power
        
        # 电池寿命估算（假设100Wh电池）
        battery_capacity_wh = 100
        battery_life_hours = battery_capacity_wh / total_power
        
        return PowerAnalysis(
            total_power_w=total_power,
            cpu_power_w=cpu_power,
            gpu_power_w=gpu_power,
            memory_power_w=memory_power,
            sensors_power_w=sensors_power,
            communications_power_w=comm_power,
            controllers_power_w=controllers_power,
            thermal_power_w=thermal_power,
            power_efficiency=power_efficiency,
            battery_life_hours=battery_life_hours
        )
    
    def analyze_reliability(self, config: Dict[str, Any]) -> ReliabilityAnalysis:
        """分析系统可靠性"""
        # 基础MTBF计算
        cpu_mtbf = 50000 / config['cpu']['cores']  # 核心越多，MTBF越低
        gpu_mtbf = 40000 / (config['gpu']['sm_count'] / 10)  # SM数量影响
        memory_mtbf = 80000
        
        # 航空航天器件MTBF
        sensors_mtbf = 100000
        comm_mtbf = 60000
        controllers_mtbf = 120000
        
        # 系统整体MTBF（串联系统）
        overall_mtbf = 1 / (1/cpu_mtbf + 1/gpu_mtbf + 1/memory_mtbf + 
                           1/sensors_mtbf + 1/comm_mtbf + 1/controllers_mtbf)
        
        # 辐射影响评分
        radiation_level = config.get('environment_config', {}).get('radiation_level', 'low')
        radiation_scores = {'low': 0.1, 'medium': 0.5, 'high': 0.9}
        radiation_impact = radiation_scores.get(radiation_level, 0.1)
        
        # 热应力评分
        temp = config.get('environment_config', {}).get('temperature_c', 25)
        if temp < -40 or temp > 85:
            thermal_stress = 0.8
        elif temp < -20 or temp > 60:
            thermal_stress = 0.5
        else:
            thermal_stress = 0.2
        
        # 老化退化估算（10年使用期）
        aging_degradation = (radiation_impact + thermal_stress) * 10  # 10年退化百分比
        
        # 容错等级评估
        redundancy_count = sum([
            config['cpu'].get('aerospace_reliability', {}).get('redundancy_level', 1),
            config['gpu'].get('aerospace_reliability', {}).get('redundancy_level', 1),
            config['memory'].get('aerospace_reliability', {}).get('redundancy_level', 1)
        ])
        
        if redundancy_count >= 6:
            fault_tolerance = "high"
            redundancy_effectiveness = 0.95
        elif redundancy_count >= 4:
            fault_tolerance = "medium"
            redundancy_effectiveness = 0.80
        else:
            fault_tolerance = "low"
            redundancy_effectiveness = 0.60
        
        return ReliabilityAnalysis(
            overall_mtbf_hours=overall_mtbf,
            radiation_impact_score=radiation_impact,
            thermal_stress_score=thermal_stress,
            aging_degradation_percent=aging_degradation,
            fault_tolerance_level=fault_tolerance,
            redundancy_effectiveness=redundancy_effectiveness
        )
    
    def generate_ai_feasibility_assessment(self, config: Dict[str, Any], 
                                         power_analysis: PowerAnalysis,
                                         reliability_analysis: ReliabilityAnalysis) -> AIFeasibilityAssessment:
        """生成AI可行性评估"""
        
        # 基础可行性评分
        feasibility_score = 70.0  # 基础分数
        
        # 技术风险评估
        technical_risks = []
        performance_bottlenecks = []
        optimization_suggestions = []
        
        # 功耗风险评估
        if power_analysis.total_power_w > 100:
            technical_risks.append("功耗过高，可能影响热管理和电池寿命")
            feasibility_score -= 10
            optimization_suggestions.append("考虑使用低功耗处理器或优化算法")
        
        if power_analysis.battery_life_hours < 24:
            technical_risks.append("电池寿命不足24小时，不满足长期任务需求")
            feasibility_score -= 15
            optimization_suggestions.append("增加电池容量或实施更激进的功耗管理")
        
        # 可靠性风险评估
        if reliability_analysis.overall_mtbf_hours < 10000:
            technical_risks.append("系统MTBF过低，可靠性不足")
            feasibility_score -= 20
            optimization_suggestions.append("增加冗余设计和容错机制")
        
        if reliability_analysis.aging_degradation_percent > 20:
            technical_risks.append("10年老化退化超过20%，长期可靠性堪忧")
            feasibility_score -= 10
            optimization_suggestions.append("采用抗老化设计和定期校准机制")
        
        # 性能瓶颈分析
        cpu_gpu_ratio = (config['cpu']['cores'] * config['cpu']['freq_ghz']) / (config['gpu']['sm_count'] * config['gpu']['freq_mhz'] / 1000)
        if cpu_gpu_ratio > 2:
            performance_bottlenecks.append("GPU性能相对CPU不足，可能限制并行计算能力")
            optimization_suggestions.append("增加GPU SM数量或提高GPU频率")
        elif cpu_gpu_ratio < 0.5:
            performance_bottlenecks.append("CPU性能相对GPU不足，可能限制串行计算能力")
            optimization_suggestions.append("增加CPU核心数或提高CPU频率")
        
        # 内存带宽评估
        memory_bw_per_core = (config['memory']['freq_mhz'] * config['memory']['channels'] * 8) / config['cpu']['cores']
        if memory_bw_per_core < 1000:  # MB/s per core
            performance_bottlenecks.append("内存带宽不足，可能成为性能瓶颈")
            optimization_suggestions.append("增加内存通道数或提高内存频率")
        
        # 实现复杂度评估
        component_count = (len(config.get('sensors', [])) + 
                          len(config.get('communications', [])) + 
                          len(config.get('controllers', [])))
        
        if component_count > 10:
            implementation_complexity = "high"
            estimated_months = 18
            feasibility_score -= 5
        elif component_count > 5:
            implementation_complexity = "medium"
            estimated_months = 12
        else:
            implementation_complexity = "low"
            estimated_months = 8
        
        # 置信度计算
        confidence_factors = [
            1.0 if power_analysis.total_power_w < 80 else 0.7,
            1.0 if reliability_analysis.overall_mtbf_hours > 20000 else 0.8,
            1.0 if len(technical_risks) < 2 else 0.6,
            1.0 if implementation_complexity == "low" else 0.8
        ]
        confidence_level = sum(confidence_factors) / len(confidence_factors)
        
        # 最终可行性评分调整
        feasibility_score = max(0, min(100, feasibility_score))
        
        return AIFeasibilityAssessment(
            feasibility_score=feasibility_score,
            technical_risks=technical_risks,
            performance_bottlenecks=performance_bottlenecks,
            optimization_suggestions=optimization_suggestions,
            implementation_complexity=implementation_complexity,
            estimated_development_time_months=estimated_months,
            confidence_level=confidence_level
        )
    
    def generate_comprehensive_report(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """生成综合分析报告"""
        
        # 执行各项分析
        power_analysis = self.analyze_power_consumption(config)
        reliability_analysis = self.analyze_reliability(config)
        ai_assessment = self.generate_ai_feasibility_assessment(config, power_analysis, reliability_analysis)
        
        # 生成时间戳
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # 构建综合报告
        report = {
            "analysis_timestamp": timestamp,
            "system_overview": {
                "cpu_config": f"{config['cpu']['cores']}-core {config['cpu']['arch']} @ {config['cpu']['freq_ghz']}GHz",
                "gpu_config": f"{config['gpu']['sm_count']}SM {config['gpu']['arch']} @ {config['gpu']['freq_mhz']}MHz",
                "memory_config": f"{config['memory']['size_gb']}GB {config['memory']['type']} @ {config['memory']['freq_mhz']}MHz",
                "aerospace_components": {
                    "sensors": len(config.get('sensors', [])),
                    "communications": len(config.get('communications', [])),
                    "controllers": len(config.get('controllers', []))
                }
            },
            "power_analysis": asdict(power_analysis),
            "reliability_analysis": asdict(reliability_analysis),
            "ai_feasibility_assessment": asdict(ai_assessment),
            "recommendations": {
                "priority_actions": self._generate_priority_actions(power_analysis, reliability_analysis, ai_assessment),
                "design_optimizations": ai_assessment.optimization_suggestions,
                "risk_mitigation": self._generate_risk_mitigation(ai_assessment.technical_risks)
            },
            "aerospace_compliance": {
                "radiation_hardening": "implemented" if config.get('cpu', {}).get('aerospace_reliability', {}).get('radiation_hardening') else "not_implemented",
                "temperature_rating": f"{config.get('cpu', {}).get('aerospace_reliability', {}).get('temperature_range', (-55, 125))}°C",
                "redundancy_level": config.get('cpu', {}).get('aerospace_reliability', {}).get('redundancy_level', 1),
                "mtbf_compliance": "pass" if reliability_analysis.overall_mtbf_hours > 10000 else "fail"
            }
        }
        
        return report
    
    def _generate_priority_actions(self, power: PowerAnalysis, reliability: ReliabilityAnalysis, 
                                 ai_assessment: AIFeasibilityAssessment) -> List[str]:
        """生成优先行动建议"""
        actions = []
        
        if power.total_power_w > 100:
            actions.append("立即优化功耗设计，目标降低至100W以下")
        
        if reliability.overall_mtbf_hours < 10000:
            actions.append("增强系统可靠性设计，提高MTBF至10000小时以上")
        
        if ai_assessment.feasibility_score < 60:
            actions.append("重新评估系统架构，解决关键技术风险")
        
        if power.battery_life_hours < 24:
            actions.append("优化电源管理策略，确保24小时以上续航")
        
        return actions
    
    def _generate_risk_mitigation(self, risks: List[str]) -> List[str]:
        """生成风险缓解措施"""
        mitigation = []
        
        for risk in risks:
            if "功耗" in risk:
                mitigation.append("实施动态电压频率调节(DVFS)和功率门控技术")
            elif "电池" in risk:
                mitigation.append("采用高能量密度电池和太阳能补充充电")
            elif "MTBF" in risk or "可靠性" in risk:
                mitigation.append("实施N+1冗余架构和故障自动切换机制")
            elif "老化" in risk:
                mitigation.append("采用抗老化材料和定期在轨校准程序")
        
        return mitigation


def main():
    """主函数 - 用于测试"""
    # 示例配置
    test_config = {
        "cpu": {
            "arch": "x86",
            "cores": 8,
            "freq_ghz": 3.0,
            "aerospace_reliability": {
                "radiation_hardening": True,
                "redundancy_level": 2,
                "temperature_range": (-55, 125)
            }
        },
        "gpu": {
            "arch": "ampere",
            "sm_count": 80,
            "freq_mhz": 1400,
            "aerospace_reliability": {
                "radiation_hardening": True,
                "redundancy_level": 2
            }
        },
        "memory": {
            "type": "DDR4",
            "size_gb": 32,
            "freq_mhz": 3200,
            "channels": 4
        },
        "sensors": [{"sensor_type": "IMU"}, {"sensor_type": "environmental"}],
        "communications": [{"comm_type": "RF_transceiver"}],
        "controllers": [{"controller_type": "attitude_controller"}],
        "environment_config": {
            "radiation_level": "medium",
            "temperature_c": 25.0
        }
    }
    
    analyzer = AerospacePerformanceAnalyzer()
    report = analyzer.generate_comprehensive_report(test_config)
    
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()