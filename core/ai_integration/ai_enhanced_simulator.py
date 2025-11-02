#!/usr/bin/env python3
"""
AI Enhanced Simulator
整合DeepSeek API和现有仿真程序的AI增强仿真器
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import asdict

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from core.ai_integration.deepseek_api_client import DeepSeekAPIClient
from core.requirements.task_requirements_analyzer import TaskRequirementsAnalyzer, TaskRequirements
from core.requirements.device_requirements_mapper import DeviceRequirementsMapper
from core.verification.simulation_verifier import SimulationVerifier


class AIEnhancedSimulator:
    """AI增强的航空航天微系统仿真器"""
    
    def __init__(self, api_key: str = None):
        self.api_client = DeepSeekAPIClient(api_key)
        self.task_analyzer = TaskRequirementsAnalyzer()
        self.device_mapper = DeviceRequirementsMapper()
        self.verifier = SimulationVerifier()
        self.project_root = project_root
        
    def process_natural_language_input(self, user_input: str) -> Dict[str, Any]:
        """处理自然语言输入的完整流程"""
        print("🚀 启动AI增强的航空航天微系统仿真平台")
        print("=" * 80)
        print(f"用户输入: {user_input}")
        print("=" * 80)
        
        results = {
            "user_input": user_input,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "ai_analysis": {},
            "simulation_results": {},
            "verification_results": {},
            "final_report": "",
            "status": "processing"
        }
        
        try:
            # 阶段1: AI需求分析
            print("\n🧠 阶段1: AI需求分析")
            print("-" * 60)
            ai_requirements = self.api_client.parse_natural_language_requirements(user_input)
            task_analysis = self.api_client.generate_task_analysis(ai_requirements)
            
            results["ai_analysis"] = {
                "requirements": ai_requirements,
                "task_analysis": task_analysis
            }
            
            # 阶段2: 执行任务需求定义（结合AI分析和实际仿真）
            print("\n🎯 阶段2: 执行任务需求定义")
            print("-" * 60)
            simulation_results = self._run_task_simulation(ai_requirements, task_analysis)
            task_requirements = self._generate_task_requirements(simulation_results, ai_requirements)
            
            results["simulation_results"]["task_requirements"] = asdict(task_requirements)
            results["simulation_results"]["benchmark_results"] = simulation_results
            
            # 阶段3: AI辅助器件映射
            print("\n🔧 阶段3: AI辅助器件映射")
            print("-" * 60)
            mapping_strategy = self.api_client.generate_device_mapping_strategy(
                ai_requirements, task_analysis
            )
            device_config = self._generate_device_configuration(
                task_requirements, mapping_strategy
            )
            
            results["simulation_results"]["device_config"] = asdict(device_config)
            results["ai_analysis"]["mapping_strategy"] = mapping_strategy
            
            # 阶段4: 仿真验证
            print("\n🔍 阶段4: 仿真验证")
            print("-" * 60)
            verification_plan = self.api_client.generate_verification_plan(
                ai_requirements, asdict(device_config)
            )
            verification_results = self._run_verification(task_requirements, device_config)
            
            results["verification_results"] = asdict(verification_results)
            results["ai_analysis"]["verification_plan"] = verification_plan
            
            # 阶段5: AI生成最终报告
            print("\n📝 阶段5: AI生成最终报告")
            print("-" * 60)
            final_report = self.api_client.generate_final_report(results)
            results["final_report"] = final_report
            
            # 保存报告
            self._save_report(final_report)
            
            results["status"] = "completed"
            print("\n✅ 仿真流程完成！")
            
        except Exception as e:
            print(f"\n❌ 仿真过程中出现错误: {e}")
            results["status"] = "failed"
            results["error"] = str(e)
            
            # 即使出错也生成基础报告
            try:
                fallback_report = self._generate_fallback_report(results, str(e))
                results["final_report"] = fallback_report
                self._save_report(fallback_report)
            except Exception as fallback_error:
                print(f"生成备用报告也失败: {fallback_error}")
        
        return results
    
    def _run_task_simulation(self, ai_requirements: Dict[str, Any], 
                           task_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """运行任务仿真，结合AI分析调整参数"""
        print("执行基于AI分析的任务基准测试...")
        
        # 运行现有的基准测试
        benchmark_results = self.task_analyzer.run_all_benchmarks()
        
        # 基于AI分析调整测试参数
        if "performance_requirements" in ai_requirements:
            perf_req = ai_requirements["performance_requirements"]
            
            # 根据AI分析的性能要求调整基准测试结果
            if "cpu_performance" in perf_req:
                cpu_factor = self._extract_performance_factor(perf_req["cpu_performance"])
                if cpu_factor:
                    benchmark_results["navigation"]["required_cpu_gips"] *= cpu_factor
            
            if "gpu_performance" in perf_req:
                gpu_factor = self._extract_performance_factor(perf_req["gpu_performance"])
                if gpu_factor:
                    benchmark_results["image_processing"]["required_gflops"] *= gpu_factor
        
        return benchmark_results
    
    def _extract_performance_factor(self, performance_desc: str) -> Optional[float]:
        """从AI描述中提取性能因子"""
        performance_desc = performance_desc.lower()
        
        if "高性能" in performance_desc or "high" in performance_desc:
            return 1.5
        elif "低功耗" in performance_desc or "low power" in performance_desc:
            return 0.8
        elif "中等" in performance_desc or "medium" in performance_desc:
            return 1.0
        else:
            return None
    
    def _extract_numeric_value(self, text: str) -> Optional[float]:
        """从文本中提取数值"""
        import re
        
        # 查找数字模式
        patterns = [
            r'(\d+\.?\d*)\s*W',  # 功耗单位
            r'(\d+\.?\d*)\s*瓦',  # 中文功耗单位
            r'(\d+\.?\d*)\s*GIPS',  # 性能单位
            r'(\d+\.?\d*)\s*GFLOPS',  # GPU性能单位
            r'(\d+\.?\d*)\s*GB/s',  # 带宽单位
            r'(\d+\.?\d*)\s*ms',  # 时间单位
            r'(\d+\.?\d*)\s*Hz',  # 频率单位
            r'(\d+\.?\d*)',  # 纯数字
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        
        return None
    
    def _generate_task_requirements(self, simulation_results: Dict[str, Any],
                                  ai_requirements: Dict[str, Any]) -> TaskRequirements:
        """基于仿真结果和AI分析生成任务需求"""
        # 使用现有的需求分析逻辑
        requirements = self.task_analyzer.analyze_requirements(simulation_results)
        
        # 基于AI分析调整需求
        if "realtime_requirements" in ai_requirements:
            rt_req = ai_requirements["realtime_requirements"]
            if "max_latency_ms" in rt_req:
                try:
                    requirements.max_control_response_ms = float(rt_req["max_latency_ms"])
                except (ValueError, TypeError):
                    # 如果无法转换，保持默认值
                    pass
            if "min_frequency_hz" in rt_req:
                try:
                    requirements.min_sensor_sampling_hz = float(rt_req["min_frequency_hz"])
                except (ValueError, TypeError):
                    pass
        
        if "power_constraints" in ai_requirements:
            power_req = ai_requirements["power_constraints"]
            if "max_power_w" in power_req:
                try:
                    # 尝试从描述中提取数值
                    power_value = self._extract_numeric_value(str(power_req["max_power_w"]))
                    if power_value:
                        requirements.max_total_power_w = power_value
                except (ValueError, TypeError):
                    pass
        
        return requirements
    
    def _generate_device_configuration(self, task_requirements: TaskRequirements,
                                     mapping_strategy: Dict[str, Any]):
        """基于任务需求和AI映射策略生成器件配置"""
        # 使用现有的器件映射逻辑
        device_config = self.device_mapper.generate_device_configuration(task_requirements)
        
        # 基于AI策略调整器件配置
        if "cpu_strategy" in mapping_strategy:
            cpu_strategy = mapping_strategy["cpu_strategy"]
            if "architecture" in cpu_strategy:
                device_config.cpu.architecture = str(cpu_strategy["architecture"])
            if "cores" in cpu_strategy:
                try:
                    cores_value = self._extract_numeric_value(str(cpu_strategy["cores"]))
                    if cores_value:
                        device_config.cpu.min_cores = int(cores_value)
                except (ValueError, TypeError):
                    pass
            if "frequency" in cpu_strategy:
                freq_value = self._extract_numeric_value(str(cpu_strategy["frequency"]))
                if freq_value:
                    device_config.cpu.min_frequency_ghz = freq_value
        
        if "gpu_strategy" in mapping_strategy:
            gpu_strategy = mapping_strategy["gpu_strategy"]
            if "architecture" in gpu_strategy:
                device_config.gpu.architecture = str(gpu_strategy["architecture"])
            if "sm_count" in gpu_strategy:
                try:
                    sm_value = self._extract_numeric_value(str(gpu_strategy["sm_count"]))
                    if sm_value:
                        device_config.gpu.min_sm_count = int(sm_value)
                except (ValueError, TypeError):
                    pass
        
        return device_config
    
    def _run_verification(self, task_requirements: TaskRequirements, device_config):
        """运行仿真验证"""
        # 临时保存需求和配置供验证器使用
        temp_req_file = self.project_root / "temp_task_requirements.json"
        temp_config_file = self.project_root / "temp_device_configuration.json"
        
        try:
            # 保存临时文件
            with open(temp_req_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "requirements": asdict(task_requirements)
                }, f, indent=2, ensure_ascii=False)
            
            with open(temp_config_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(device_config), f, indent=2, ensure_ascii=False)
            
            # 运行验证
            verification_results = self.verifier.run_comprehensive_verification()
            
            return verification_results
            
        finally:
            # 清理临时文件
            if temp_req_file.exists():
                temp_req_file.unlink()
            if temp_config_file.exists():
                temp_config_file.unlink()
    
    def _save_report(self, report_content: str):
        """保存报告到指定位置"""
        report_file = self.project_root / "aerospace_simulation_report.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"📁 报告已保存到: {report_file}")
    
    def _generate_fallback_report(self, results: Dict[str, Any], error_msg: str) -> str:
        """生成备用报告"""
        return f"""
# 航空航天微系统产品需求定义模型仿真报告

## 报告概要

**项目名称**: 微系统产品需求定义模型构建  
**报告类型**: 仿真验证报告（备用版本）  
**生成时间**: {results.get('timestamp', 'Unknown')}  
**验证方法**: AI增强两阶段验证  

## 1. 执行摘要

用户输入: {results.get('user_input', 'Unknown')}

本次仿真采用AI增强的两阶段分离式验证方法，通过DeepSeek API分析用户自然语言需求，结合项目自带的仿真程序进行验证。

## 2. 处理状态

**状态**: {results.get('status', 'Unknown')}
**错误信息**: {error_msg}

## 3. AI分析结果

{json.dumps(results.get('ai_analysis', {}), ensure_ascii=False, indent=2)}

## 4. 仿真结果

{json.dumps(results.get('simulation_results', {}), ensure_ascii=False, indent=2)}

## 5. 结论

由于处理过程中出现错误，本报告为备用版本。建议检查API配置和输入格式后重新运行。

---
**报告编制**: AI增强航空航天微系统仿真平台  
**生成方式**: 自动化生成（备用模式）  
"""


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI增强的航空航天微系统仿真平台')
    parser.add_argument('--input', type=str, required=True, help='自然语言需求描述')
    parser.add_argument('--api-key', type=str, help='DeepSeek API密钥')
    
    args = parser.parse_args()
    
    # 创建AI增强仿真器
    simulator = AIEnhancedSimulator(args.api_key)
    
    # 处理用户输入
    results = simulator.process_natural_language_input(args.input)
    
    # 显示结果摘要
    print("\n" + "=" * 80)
    print("🎉 处理完成！")
    print("=" * 80)
    print(f"状态: {results['status']}")
    if results['status'] == 'completed':
        print("✅ 报告已自动生成到 aerospace_simulation_report.md")
    else:
        print(f"❌ 处理失败: {results.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()