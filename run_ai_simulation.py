#!/usr/bin/env python3
"""
AI增强航空航天微系统仿真运行脚本
简化的入口点，专门用于AI模式
"""

import os
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from core.ai_integration.ai_enhanced_simulator import AIEnhancedSimulator


def main():
    """主函数"""
    print("🚀 AI增强航空航天微系统仿真平台")
    print("=" * 60)
    
    # 获取用户输入
    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
    else:
        print("请输入您的需求描述:")
        user_input = input("> ")
    
    if not user_input.strip():
        print("❌ 请提供有效的需求描述")
        return
    
    # 获取API密钥
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        print("⚠️  未设置DEEPSEEK_API_KEY环境变量，将使用默认配置")
    
    # 设置环境变量
    os.environ['SIMULATOR_ROOT'] = str(project_root.parent.parent.parent)
    os.environ['BENCHMARK_ROOT'] = str(project_root)
    
    # 创建AI增强仿真器
    simulator = AIEnhancedSimulator(api_key)
    
    # 处理用户输入
    results = simulator.process_natural_language_input(user_input)
    
    # 显示结果
    print("\n" + "=" * 60)
    if results['status'] == 'completed':
        print("✅ 仿真完成！")
        print("📁 报告已自动生成到: aerospace_simulation_report.md")
        print("\n📊 处理摘要:")
        print(f"   • 用户输入: {user_input}")
        print(f"   • 处理时间: {results['timestamp']}")
        print(f"   • AI分析: {'成功' if results.get('ai_analysis') else '失败'}")
        print(f"   • 仿真验证: {'成功' if results.get('verification_results') else '失败'}")
    else:
        print("❌ 仿真失败")
        print(f"错误信息: {results.get('error', 'Unknown error')}")
        if results.get('final_report'):
            print("📁 已生成备用报告到: aerospace_simulation_report.md")


if __name__ == "__main__":
    main()