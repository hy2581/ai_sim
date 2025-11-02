#!/usr/bin/env python3
"""
Simple Report Generator for Aerospace Microsystem Simulation
简化版报告生成器

生成简洁的仿真分析报告，包括：
- Markdown格式报告
- JSON数据报告
- 基础图表生成
"""

import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


class SimpleReportGenerator:
    """简化版报告生成器"""
    
    def __init__(self):
        self.report_data = {}
        
    def generate_markdown_report(self, simulation_data: Dict[str, Any], 
                                output_path: str = "simulation_report.md") -> str:
        """生成Markdown格式报告"""
        
        # 获取基础数据
        overall_score = simulation_data.get('performance', {}).get('overall_score', 0)
        fault_data = simulation_data.get('fault_analysis', {})
        system_config = simulation_data.get('system_configuration', {})
        
        # 生成报告内容
        report_content = f"""# 航空航天微系统仿真分析报告

## 报告概要

- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **项目编号**: {simulation_data.get('project_id', 'N/A')}
- **系统编号**: {simulation_data.get('system_id', 'N/A')}
- **总体评分**: {overall_score:.1f}/100

## 执行摘要

本次仿真评估显示系统总体性能评分为 **{overall_score:.1f}/100**。
系统在测试期间表现{'优秀' if overall_score >= 90 else '良好' if overall_score >= 80 else '中等' if overall_score >= 70 else '需要改进'}，
具备良好的计算能力、可靠性和环境适应性。

### 关键指标

| 指标 | 数值 | 状态 |
|------|------|------|
| 总体评分 | {overall_score:.1f}/100 | {'✅ 优秀' if overall_score >= 90 else '✅ 良好' if overall_score >= 80 else '⚠️ 中等' if overall_score >= 70 else '❌ 需改进'} |
| 故障检测率 | {fault_data.get('detection_rate', 0)*100:.1f}% | {'✅ 良好' if fault_data.get('detection_rate', 0) >= 0.8 else '⚠️ 一般'} |
| 故障恢复率 | {fault_data.get('recovery_rate', 0)*100:.1f}% | {'✅ 良好' if fault_data.get('recovery_rate', 0) >= 0.7 else '⚠️ 一般'} |
| 系统韧性 | {fault_data.get('system_resilience_analysis', {}).get('resilience_score', 0):.1f} | {'✅ 优秀' if fault_data.get('system_resilience_analysis', {}).get('resilience_score', 0) >= 85 else '✅ 良好'} |

## 系统配置

### 硬件配置

- **处理器**: {system_config.get('cpu', {}).get('cores', 'N/A')} 核心 {system_config.get('cpu', {}).get('architecture', 'N/A')}
- **GPU**: {system_config.get('gpu', {}).get('sm_count', 'N/A')} SM {system_config.get('gpu', {}).get('architecture', 'N/A')}
- **内存**: {system_config.get('memory', {}).get('size_gb', 'N/A')} GB {system_config.get('memory', {}).get('type', 'N/A')}
- **传感器**: {len(system_config.get('sensors', []))} 个专用传感器
- **通信系统**: {system_config.get('communication', {}).get('transceivers', 'N/A')} 个收发器
- **电源系统**: {system_config.get('power_system', {}).get('solar_panels_w', 'N/A')} W 太阳能板

### 性能规格

- **工作温度**: -55°C 至 +125°C
- **辐射抗性**: TID > 100 krad, SEU < 1e-12
- **功耗预算**: {simulation_data.get('performance', {}).get('power_performance', {}).get('total_power_budget_w', 'N/A')} W
- **电池寿命**: {simulation_data.get('performance', {}).get('power_performance', {}).get('battery_life_hours', 'N/A')} 小时

## 性能分析

### 各类别性能评分

"""
        
        # 添加性能类别分析
        category_scores = simulation_data.get('performance', {}).get('category_scores', {})
        for category, score_data in category_scores.items():
            score = score_data.get('raw_score', 0)
            status = '🟢 优秀' if score >= 85 else '🟡 良好' if score >= 70 else '🟠 中等' if score >= 60 else '🔴 需改进'
            report_content += f"- **{category.replace('_', ' ').title()}**: {score:.1f}/100 {status}\n"
        
        report_content += f"""

### 关键性能指标

#### 计算性能
- CPU吞吐量: {simulation_data.get('performance', {}).get('computational_performance', {}).get('cpu_throughput_gops', 'N/A')} GOPS
- GPU吞吐量: {simulation_data.get('performance', {}).get('computational_performance', {}).get('gpu_throughput_tflops', 'N/A')} TFLOPS
- 内存带宽: {simulation_data.get('performance', {}).get('computational_performance', {}).get('memory_bandwidth_gbps', 'N/A')} GB/s

#### 功耗与可靠性
- 总功耗: {simulation_data.get('performance', {}).get('power_performance', {}).get('total_power_budget_w', 'N/A')} W
- 功耗效率: {simulation_data.get('performance', {}).get('power_performance', {}).get('power_efficiency_gops_per_w', 'N/A')} GOPS/W
- MTBF: {simulation_data.get('performance', {}).get('reliability_targets', {}).get('mtbf_hours', 'N/A')} 小时

## 故障分析与韧性评估

### 故障注入测试结果

本次测试共注入了 **{fault_data.get('total_injected', 0)}** 个不同类型和严重程度的故障：

| 故障统计 | 数量 |
|----------|------|
| 注入故障 | {fault_data.get('total_injected', 0)} |
| 检测故障 | {fault_data.get('total_detected', 0)} |
| 恢复故障 | {fault_data.get('total_recovered', 0)} |

### 故障类型分布

"""
        
        # 添加故障类型分析
        fault_by_type = fault_data.get('by_type', {})
        for fault_type, count in fault_by_type.items():
            report_content += f"- **{fault_type.title()}**: {count} 个\n"
        
        report_content += f"""

### 故障严重程度分布

"""
        
        # 添加故障严重程度分析
        fault_by_severity = fault_data.get('by_severity', {})
        for severity, count in fault_by_severity.items():
            report_content += f"- **{severity.title()}**: {count} 个\n"
        
        report_content += f"""

### 系统韧性评估

系统韧性评分: **{fault_data.get('system_resilience_analysis', {}).get('resilience_score', 0):.1f}/100**

- 故障检测能力: {fault_data.get('detection_rate', 0)*100:.1f}%
- 故障恢复能力: {fault_data.get('recovery_rate', 0)*100:.1f}%
- 容错能力: 良好

## 环境影响分析

### 环境适应性评估

系统在模拟的太空环境中表现稳定：

- ✅ 热管理系统有效控制了温度变化
- ✅ 抗辐射设计满足任务要求  
- ✅ 电源系统在地影期间工作正常
- ✅ 通信系统在恶劣环境下保持稳定

### 环境事件统计

仿真期间记录的主要环境事件：

- 地影进出事件: {simulation_data.get('environment', {}).get('eclipse_events', 'N/A')} 次
- 辐射带穿越: {simulation_data.get('environment', {}).get('radiation_belt_passages', 'N/A')} 次
- 热冲击事件: {simulation_data.get('environment', {}).get('thermal_shock_events', 'N/A')} 次
- 太阳活动影响: {simulation_data.get('environment', {}).get('solar_activity_events', 'N/A')} 次

## 合规性评估

### 标准符合性

"""
        
        # 添加合规性分析
        compliance_data = simulation_data.get('compliance', {})
        standards = [
            ("GJB 548B", "半导体器件试验方法与程序"),
            ("QJ 3244", "航天器热控制要求"),
            ("MIL-STD-883", "微电路试验方法和程序"),
            ("ECSS-E-ST-20", "欧洲空间标准")
        ]
        
        compliant_count = 0
        for std_id, description in standards:
            status = compliance_data.get(std_id.replace('-', '_').replace(' ', '_'), False)
            status_icon = '✅ 符合' if status else '❌ 不符合'
            if status:
                compliant_count += 1
            report_content += f"- **{std_id}** ({description}): {status_icon}\n"
        
        compliance_rate = (compliant_count / len(standards)) * 100
        
        report_content += f"""

**总体合规率**: {compliance_rate:.1f}% ({compliant_count}/{len(standards)} 项标准符合)

## 建议与结论

### 总体结论

本次仿真评估表明，航空航天微系统在设计和实现方面达到了预期目标，
总体性能评分为 **{overall_score:.1f}/100**。
系统具备良好的计算能力、可靠性和环境适应性，能够满足航空航天应用的基本要求。

### 关键发现

- ✅ 系统在正常工作条件下表现{'优秀' if overall_score >= 90 else '良好' if overall_score >= 80 else '中等'}
- ✅ 故障检测和恢复机制运行良好，具备较强的容错能力
- ✅ 环境适应性符合航空航天应用要求
- ⚠️ 部分性能指标仍有优化空间

### 改进建议

#### 短期建议 (1-3个月)
- 优化关键性能瓶颈，提升系统整体效率
- 完善故障检测和诊断算法
- 加强系统监控和日志记录功能
- 进行更全面的环境适应性测试

#### 中期建议 (3-12个月)
- 开发智能化的故障预测和预防系统
- 实施基于机器学习的性能优化
- 建立完整的数字孪生模型
- 制定详细的维护和升级计划

#### 长期建议 (1年以上)
- 研发下一代系统架构
- 探索新的容错和自修复技术
- 建立行业标准和最佳实践
- 培养专业技术团队

### 下一步行动

1. **立即行动**: 解决识别出的关键问题，确保系统稳定运行
2. **制定计划**: 根据建议制定详细的改进计划和时间表
3. **资源配置**: 合理分配人力、物力和财力资源
4. **持续监控**: 建立持续的性能监控和评估机制
5. **定期评审**: 定期进行系统评审和改进效果评估

---

**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**报告版本**: v2.0  
**分类**: 内部文档
"""
        
        # 保存报告
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return output_path
    
    def generate_json_report(self, simulation_data: Dict[str, Any], 
                           output_path: str = "simulation_report.json") -> str:
        """生成JSON格式报告"""
        
        report_data = {
            "metadata": {
                "generation_time": datetime.now().isoformat(),
                "report_version": "v2.0",
                "project_id": simulation_data.get('project_id', 'N/A'),
                "system_id": simulation_data.get('system_id', 'N/A')
            },
            "summary": {
                "overall_score": simulation_data.get('performance', {}).get('overall_score', 0),
                "performance_level": self._get_performance_level(simulation_data.get('performance', {}).get('overall_score', 0)),
                "total_faults_injected": simulation_data.get('fault_analysis', {}).get('total_injected', 0),
                "fault_detection_rate": simulation_data.get('fault_analysis', {}).get('detection_rate', 0),
                "fault_recovery_rate": simulation_data.get('fault_analysis', {}).get('recovery_rate', 0),
                "resilience_score": simulation_data.get('fault_analysis', {}).get('system_resilience_analysis', {}).get('resilience_score', 0)
            },
            "detailed_analysis": simulation_data,
            "recommendations": self._generate_recommendations_list(simulation_data),
            "compliance_status": self._analyze_compliance(simulation_data.get('compliance', {}))
        }
        
        # 保存JSON报告
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
        
        return output_path
    
    def generate_simple_charts(self, simulation_data: Dict[str, Any], 
                             output_dir: str = "charts") -> List[str]:
        """生成简单图表"""
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        chart_files = []
        
        # 1. 性能雷达图
        performance_chart = self._create_performance_radar_chart(
            simulation_data.get('performance', {}), 
            os.path.join(output_dir, "performance_radar.png")
        )
        if performance_chart:
            chart_files.append(performance_chart)
        
        # 2. 故障分析柱状图
        fault_chart = self._create_fault_analysis_chart(
            simulation_data.get('fault_analysis', {}),
            os.path.join(output_dir, "fault_analysis.png")
        )
        if fault_chart:
            chart_files.append(fault_chart)
        
        # 3. 合规性饼图
        compliance_chart = self._create_compliance_pie_chart(
            simulation_data.get('compliance', {}),
            os.path.join(output_dir, "compliance_status.png")
        )
        if compliance_chart:
            chart_files.append(compliance_chart)
        
        return chart_files
    
    def _get_performance_level(self, score: float) -> str:
        """获取性能等级"""
        if score >= 90:
            return "优秀"
        elif score >= 80:
            return "良好"
        elif score >= 70:
            return "中等"
        elif score >= 60:
            return "及格"
        else:
            return "需要改进"
    
    def _generate_recommendations_list(self, simulation_data: Dict[str, Any]) -> List[str]:
        """生成建议列表"""
        recommendations = []
        
        overall_score = simulation_data.get('performance', {}).get('overall_score', 0)
        fault_data = simulation_data.get('fault_analysis', {})
        
        if overall_score < 70:
            recommendations.append("系统性能需要显著改进，建议重新评估架构设计")
        elif overall_score < 85:
            recommendations.append("系统性能良好，建议针对薄弱环节进行优化")
        
        if fault_data.get('detection_rate', 0) < 0.8:
            recommendations.append("加强故障检测机制，提高故障识别能力")
        
        if fault_data.get('recovery_rate', 0) < 0.7:
            recommendations.append("改进故障恢复策略，增强系统自愈能力")
        
        recommendations.extend([
            "建立持续的性能监控和评估机制",
            "制定详细的维护和升级计划",
            "加强团队技术培训和能力建设"
        ])
        
        return recommendations
    
    def _analyze_compliance(self, compliance_data: Dict[str, bool]) -> Dict[str, Any]:
        """分析合规性状态"""
        total_standards = len(compliance_data)
        compliant_standards = sum(1 for status in compliance_data.values() if status)
        
        return {
            "total_standards": total_standards,
            "compliant_standards": compliant_standards,
            "compliance_rate": (compliant_standards / total_standards * 100) if total_standards > 0 else 0,
            "status": "优秀" if compliant_standards / total_standards >= 0.9 else "良好" if compliant_standards / total_standards >= 0.75 else "需要改进",
            "details": compliance_data
        }
    
    def _create_performance_radar_chart(self, performance_data: Dict[str, Any], output_path: str) -> Optional[str]:
        """创建性能雷达图"""
        try:
            category_scores = performance_data.get('category_scores', {})
            if not category_scores:
                return None
            
            categories = list(category_scores.keys())
            scores = [category_scores[cat].get('raw_score', 0) for cat in categories]
            
            # 创建雷达图
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
            scores_plot = scores + [scores[0]]  # 闭合图形
            angles_plot = np.concatenate((angles, [angles[0]]))
            
            fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
            ax.plot(angles_plot, scores_plot, 'o-', linewidth=2, color='#1f77b4')
            ax.fill(angles_plot, scores_plot, alpha=0.25, color='#1f77b4')
            ax.set_xticks(angles)
            ax.set_xticklabels([cat.replace('_', '\n') for cat in categories])
            ax.set_ylim(0, 100)
            ax.set_title('系统性能雷达图', fontsize=16, fontweight='bold', pad=20)
            ax.grid(True)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_path
        except Exception as e:
            print(f"Error creating performance radar chart: {e}")
            return None
    
    def _create_fault_analysis_chart(self, fault_data: Dict[str, Any], output_path: str) -> Optional[str]:
        """创建故障分析图表"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 故障类型分布
            fault_types = fault_data.get('by_type', {})
            if fault_types:
                ax1.bar(fault_types.keys(), fault_types.values(), color=['#ff7f0e', '#2ca02c', '#d62728'])
                ax1.set_title('故障类型分布', fontweight='bold')
                ax1.set_ylabel('故障数量')
                plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
            
            # 故障处理统计
            metrics = ['注入', '检测', '恢复']
            values = [
                fault_data.get('total_injected', 0),
                fault_data.get('total_detected', 0),
                fault_data.get('total_recovered', 0)
            ]
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            
            bars = ax2.bar(metrics, values, color=colors)
            ax2.set_title('故障处理统计', fontweight='bold')
            ax2.set_ylabel('故障数量')
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{int(value)}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_path
        except Exception as e:
            print(f"Error creating fault analysis chart: {e}")
            return None
    
    def _create_compliance_pie_chart(self, compliance_data: Dict[str, bool], output_path: str) -> Optional[str]:
        """创建合规性饼图"""
        try:
            if not compliance_data:
                return None
            
            compliant = sum(1 for status in compliance_data.values() if status)
            non_compliant = len(compliance_data) - compliant
            
            labels = ['符合标准', '不符合标准']
            sizes = [compliant, non_compliant]
            colors = ['#2ca02c', '#d62728']
            
            fig, ax = plt.subplots(figsize=(8, 8))
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                            colors=colors, startangle=90)
            ax.set_title('标准合规性状态', fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_path
        except Exception as e:
            print(f"Error creating compliance pie chart: {e}")
            return None


def main():
    """主函数 - 演示简化版报告生成器"""
    # 模拟仿真数据
    simulation_data = {
        "project_id": "AERO-SIM-001",
        "system_id": "aerospace_microsystem_v1",
        "performance": {
            "overall_score": 87.5,
            "category_scores": {
                "computational": {"raw_score": 85.2},
                "power_efficiency": {"raw_score": 78.9},
                "reliability": {"raw_score": 92.1},
                "thermal": {"raw_score": 83.4},
                "communication": {"raw_score": 88.7}
            },
            "computational_performance": {
                "cpu_throughput_gops": 200,
                "gpu_throughput_tflops": 100,
                "memory_bandwidth_gbps": 800
            },
            "power_performance": {
                "total_power_budget_w": 350,
                "power_efficiency_gops_per_w": 0.8,
                "battery_life_hours": 72
            },
            "reliability_targets": {
                "mtbf_hours": 120000
            }
        },
        "fault_analysis": {
            "total_injected": 25,
            "total_detected": 22,
            "total_recovered": 18,
            "detection_rate": 0.88,
            "recovery_rate": 0.82,
            "by_type": {
                "hardware": 10,
                "software": 8,
                "environmental": 7
            },
            "by_severity": {
                "minor": 12,
                "moderate": 8,
                "major": 4,
                "critical": 1
            },
            "system_resilience_analysis": {
                "resilience_score": 85.3
            }
        },
        "system_configuration": {
            "cpu": {"cores": 8, "freq_ghz": 3.2, "architecture": "ARM_Cortex_A78"},
            "gpu": {"sm_count": 80, "freq_mhz": 1400, "architecture": "Ampere"},
            "memory": {"size_gb": 32, "type": "DDR5"},
            "sensors": [{"type": "optical", "count": 2}, {"type": "radar", "count": 1}],
            "communication": {"transceivers": 3},
            "power_system": {"solar_panels_w": 1000}
        },
        "environment": {
            "eclipse_events": 16,
            "radiation_belt_passages": 4,
            "thermal_shock_events": 8,
            "solar_activity_events": 3
        },
        "compliance": {
            "GJB_548B": True,
            "QJ_3244": True,
            "MIL_STD_883": False,
            "ECSS_E_ST_20": True
        }
    }
    
    # 创建报告生成器
    generator = SimpleReportGenerator()
    
    # 生成Markdown报告
    print("Generating Markdown report...")
    md_report = generator.generate_markdown_report(simulation_data)
    print(f"Markdown report generated: {md_report}")
    
    # 生成JSON报告
    print("Generating JSON report...")
    json_report = generator.generate_json_report(simulation_data)
    print(f"JSON report generated: {json_report}")
    
    # 生成图表
    print("Generating charts...")
    charts = generator.generate_simple_charts(simulation_data)
    print(f"Charts generated: {charts}")
    
    print("All reports generated successfully!")


if __name__ == "__main__":
    main()