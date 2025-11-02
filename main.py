#!/usr/bin/env python3
"""
AI增强的航空航天微系统需求定义与验证平台主入口
AI-Enhanced Main Entry Point for Aerospace Microsystem Requirements Definition and Verification
"""

import sys
import os
import json
import argparse
from pathlib import Path
from dataclasses import asdict

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from core.requirements.task_requirements_analyzer import TaskRequirementsAnalyzer
from core.requirements.device_requirements_mapper import DeviceRequirementsMapper
from core.verification.simulation_verifier import SimulationVerifier
from core.ai_integration.ai_enhanced_simulator import AIEnhancedSimulator


class EnhancedAerospaceSimulationPlatform:
    """增强的航空航天仿真平台"""
    
    def __init__(self):
        self.project_root = project_root
        self.simulator_root = str(project_root.parent.parent.parent)
        
    def run_task_requirements_analysis(self):
        """阶段1: 任务需求定义"""
        print("🎯 阶段1: 任务需求定义")
        print("=" * 60)
        print("通过执行真实的航空航天任务模拟程序来定义系统性能需求")
        print()
        
        # 直接调用任务需求分析器的功能
        analyzer = TaskRequirementsAnalyzer()
        benchmark_results = analyzer.run_all_benchmarks()
        requirements = analyzer.analyze_requirements(benchmark_results)
        
        # 生成报告
        report = analyzer.generate_requirements_report(requirements, benchmark_results)
        output_dir = Path(self.project_root)
        
        # 保存结果
        with open(output_dir / "task_requirements.json", 'w', encoding='utf-8') as f:
            json.dump({
                "requirements": asdict(requirements),
                "benchmark_results": benchmark_results
            }, f, indent=2, ensure_ascii=False)
        
        with open(output_dir / "task_requirements_report.md", 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("\n✅ 任务需求定义完成")
        return requirements, benchmark_results
    
    def run_device_requirements_mapping(self):
        """阶段2: 器件需求定义"""
        print("\n🔧 阶段2: 器件需求定义")
        print("=" * 60)
        print("基于任务需求，通过仿真器辅助定义器件参数需求")
        print()
        
        # 加载任务需求
        task_req_file = Path(self.project_root) / "task_requirements.json"
        with open(task_req_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            from core.requirements.task_requirements_analyzer import TaskRequirements
            task_req = TaskRequirements(**data["requirements"])
        
        # 创建器件需求映射器并生成配置
        mapper = DeviceRequirementsMapper(self.simulator_root)
        device_config = mapper.generate_device_configuration(task_req)
        
        # 生成报告
        report = mapper.generate_mapping_report(task_req, device_config)
        output_dir = Path(self.project_root)
        
        # 保存结果
        with open(output_dir / "device_configuration.json", 'w', encoding='utf-8') as f:
            json.dump(asdict(device_config), f, indent=2, ensure_ascii=False)
        
        with open(output_dir / "device_mapping_report.md", 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("\n✅ 器件需求定义完成")
        return device_config
    
    def run_simulation_verification(self):
        """阶段3: 仿真验证"""
        print("\n🔍 阶段3: 仿真验证")
        print("=" * 60)
        print("验证器件组合能否满足任务执行需求")
        print()
        
        verifier = SimulationVerifier()
        verification_report = verifier.run_comprehensive_verification()
        
        # 生成报告
        report_text = verifier.generate_verification_report(verification_report)
        output_dir = Path(self.project_root)
        
        # 保存验证报告
        with open(output_dir / "verification_report.md", 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print("\n✅ 仿真验证完成")
        return verification_report
    
    def run_complete_workflow(self):
        """运行完整的验证工作流"""
        print("🚀 航空航天微系统需求定义与验证平台")
        print("=" * 80)
        print("新的两阶段验证方法:")
        print("1. 任务需求定义: 通过模拟程序定义执行任务的性能需求")
        print("2. 器件需求定义: 通过仿真器辅助定义器件参数需求")
        print("3. 仿真验证: 验证器件组合能否满足任务需求")
        print("=" * 80)
        print()
        
        try:
            # 阶段1: 任务需求定义
            task_req, benchmark_results = self.run_task_requirements_analysis()
            
            # 阶段2: 器件需求定义
            device_config = self.run_device_requirements_mapping()
            
            # 阶段3: 仿真验证
            verification_report = self.run_simulation_verification()
            
            # 显示最终结果
            self.display_final_results(verification_report)
            
            return {
                "task_requirements": task_req,
                "device_configuration": device_config,
                "verification_report": verification_report,
                "status": "success"
            }
            
        except Exception as e:
            print(f"\n❌ 验证过程中出现错误: {e}")
            return {"status": "failed", "error": str(e)}
    
    def display_final_results(self, verification_report):
        """显示最终结果"""
        print("\n" + "=" * 80)
        print("🎉 验证流程完成！")
        print("=" * 80)
        
        status_icon = "✅" if verification_report.overall_passed else "❌"
        status_text = "通过" if verification_report.overall_passed else "未通过"
        
        print(f"📊 验证结果: {status_icon} {status_text}")
        print(f"📈 通过率: {verification_report.pass_rate:.1f}%")
        print(f"📋 测试项目: {verification_report.passed_tests}/{verification_report.total_tests}")
        
        if verification_report.bottlenecks:
            print(f"\n⚠️  发现的瓶颈:")
            for bottleneck in verification_report.bottlenecks:
                print(f"   • {bottleneck}")
        
        if verification_report.recommendations:
            print(f"\n💡 改进建议:")
            for recommendation in verification_report.recommendations:
                print(f"   • {recommendation}")
        
        print(f"\n📁 详细报告已保存到:")
        print(f"   • 任务需求报告: task_requirements_report.md")
        print(f"   • 器件映射报告: device_mapping_report.md")
        print(f"   • 验证结果报告: verification_report.md")
        
        print("\n🎯 新验证方法的优势:")
        print("   ✓ 基于真实任务执行定义需求，更准确反映实际性能需求")
        print("   ✓ 通过仿真器辅助器件选型，提供科学的配置依据")
        print("   ✓ 分离需求定义和验证过程，避免循环依赖问题")
        print("   ✓ 提供量化的性能余量分析，支持优化决策")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='AI增强的航空航天微系统需求定义与验证平台')
    parser.add_argument('--mode', choices=['ai', 'complete', 'task', 'device', 'verify'], 
                       default='ai', help='运行模式')
    parser.add_argument('--input', type=str, 
                       help='自然语言需求描述（AI模式）')
    parser.add_argument('--api-key', type=str, 
                       help='DeepSeek API密钥')
    parser.add_argument('--description', type=str, 
                       help='系统描述（传统模式）')
    
    args = parser.parse_args()
    
    # 设置环境变量
    os.environ['SIMULATOR_ROOT'] = str(project_root.parent.parent.parent)
    os.environ['BENCHMARK_ROOT'] = str(project_root)
    
    if args.mode == 'ai':
        # AI增强模式 - 新的主要模式
        if not args.input:
            print("❌ AI模式需要提供 --input 参数")
            print("示例: python3 main.py --mode ai --input \"我需要一个用于无人机导航的微系统\"")
            return
        
        print("🤖 启动AI增强模式")
        simulator = AIEnhancedSimulator(args.api_key)
        results = simulator.process_natural_language_input(args.input)
        
        if results['status'] == 'completed':
            print("✅ AI增强仿真完成！报告已自动生成。")
        else:
            print(f"❌ AI增强仿真失败: {results.get('error', 'Unknown error')}")
    
    else:
        # 传统模式
        print("🔧 启动传统模式")
        platform = EnhancedAerospaceSimulationPlatform()
        
        if args.mode == 'complete':
            # 运行完整工作流
            results = platform.run_complete_workflow()
        elif args.mode == 'task':
            # 仅运行任务需求分析
            platform.run_task_requirements_analysis()
        elif args.mode == 'device':
            # 仅运行器件需求映射
            platform.run_device_requirements_mapping()
        elif args.mode == 'verify':
            # 仅运行仿真验证
            platform.run_simulation_verification()
    
    print("\n🎉 程序执行完成！")


if __name__ == "__main__":
    main()