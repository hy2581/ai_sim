#!/usr/bin/env python3
"""
AI增强航空航天微系统仿真平台演示脚本
展示从自然语言输入到自动生成报告的完整流程
"""

import os
import time
from pathlib import Path

def demo_ai_enhanced_simulation():
    """演示AI增强仿真流程"""
    
    print("🚀 AI增强航空航天微系统仿真平台演示")
    print("=" * 80)
    print()
    
    # 演示用例
    demo_cases = [
        {
            "name": "无人机导航系统",
            "input": "我需要一个用于无人机导航的微系统，要求实时性好，功耗低，能在恶劣环境下工作",
            "description": "适用于无人机自主导航，需要处理GPS、IMU等传感器数据"
        },
        {
            "name": "卫星姿态控制",
            "input": "设计一个用于小型卫星姿态控制的微系统，需要高可靠性，抗辐射，支持多传感器融合",
            "description": "用于CubeSat等小型卫星的三轴稳定控制"
        },
        {
            "name": "航空电子系统",
            "input": "开发一个航空电子微系统，用于飞行控制，要求高性能计算，低延迟响应，符合航空标准",
            "description": "商用或军用飞机的飞行管理系统"
        }
    ]
    
    print("📋 可用演示用例:")
    for i, case in enumerate(demo_cases, 1):
        print(f"  {i}. {case['name']}")
        print(f"     {case['description']}")
        print()
    
    # 用户选择
    while True:
        try:
            choice = input("请选择演示用例 (1-3) 或输入 'q' 退出: ").strip()
            if choice.lower() == 'q':
                print("👋 演示结束")
                return
            
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(demo_cases):
                selected_case = demo_cases[choice_idx]
                break
            else:
                print("❌ 无效选择，请输入 1-3")
        except ValueError:
            print("❌ 请输入有效数字")
    
    print(f"\n🎯 选择的用例: {selected_case['name']}")
    print(f"📝 需求描述: {selected_case['input']}")
    print()
    
    # 确认执行
    confirm = input("是否开始仿真? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ 演示取消")
        return
    
    print("\n" + "=" * 80)
    print("🚀 开始AI增强仿真...")
    print("=" * 80)
    
    # 执行仿真
    import subprocess
    import sys
    
    try:
        # 运行AI仿真
        result = subprocess.run([
            sys.executable, 'run_ai_simulation.py', selected_case['input']
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        print("📊 仿真输出:")
        print("-" * 60)
        print(result.stdout)
        
        if result.stderr:
            print("⚠️  警告信息:")
            print(result.stderr)
        
        # 检查生成的文件
        report_file = Path(__file__).parent / "aerospace_simulation_report.md"
        if report_file.exists():
            print(f"\n✅ 报告已生成: {report_file}")
            print(f"📄 文件大小: {report_file.stat().st_size} 字节")
            
            # 显示报告摘要
            with open(report_file, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                print("\n📋 报告摘要 (前10行):")
                print("-" * 40)
                for line in lines[:10]:
                    print(line)
                if len(lines) > 10:
                    print("...")
        else:
            print("❌ 报告文件未生成")
            
    except Exception as e:
        print(f"❌ 仿真执行失败: {e}")
    
    print("\n" + "=" * 80)
    print("🎉 演示完成！")
    print("=" * 80)


def show_system_info():
    """显示系统信息"""
    print("🔧 系统信息:")
    print("-" * 40)
    
    # 检查Python版本
    import sys
    print(f"Python版本: {sys.version}")
    
    # 检查依赖包
    required_packages = ['requests', 'numpy']
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}: 已安装")
        except ImportError:
            print(f"❌ {package}: 未安装")
    
    # 检查API密钥
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if api_key:
        print(f"✅ DEEPSEEK_API_KEY: 已设置 ({api_key[:8]}...)")
    else:
        print("⚠️  DEEPSEEK_API_KEY: 未设置 (将使用默认配置)")
    
    # 检查项目文件
    project_files = [
        'run_ai_simulation.py',
        'core/ai_integration/deepseek_api_client.py',
        'core/ai_integration/ai_enhanced_simulator.py'
    ]
    
    print("\n📁 项目文件:")
    for file_path in project_files:
        full_path = Path(__file__).parent / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}: 缺失")
    
    print()


def main():
    """主函数"""
    print("🎬 AI增强航空航天微系统仿真平台演示")
    print("=" * 60)
    print()
    
    while True:
        print("请选择操作:")
        print("1. 运行仿真演示")
        print("2. 查看系统信息")
        print("3. 退出")
        print()
        
        choice = input("请输入选择 (1-3): ").strip()
        
        if choice == '1':
            demo_ai_enhanced_simulation()
        elif choice == '2':
            show_system_info()
        elif choice == '3':
            print("👋 再见！")
            break
        else:
            print("❌ 无效选择")
        
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()