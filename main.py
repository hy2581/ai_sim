#!/usr/bin/env python3
"""
航空航天微系统需求定义与验证平台 - 传统模式处理
Traditional Mode Processor for Aerospace Microsystem Requirements
"""

import sys
import os
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from core.traditional_mode_processor import TraditionalModeProcessor


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='航空航天微系统需求定义与验证平台 - 传统模式')
    parser.add_argument('--mode', choices=['traditional'], default='traditional', help='运行模式')
    parser.add_argument('--input', type=str, required=True, help='自然语言需求描述')
    parser.add_argument('--api-key', type=str, required=True, help='DeepSeek API密钥')
    
    args = parser.parse_args()
    
    # 设置环境变量
    os.environ['SIMULATOR_ROOT'] = str(project_root.parent.parent.parent)
    os.environ['BENCHMARK_ROOT'] = str(project_root)
    
    if args.mode == 'traditional':
        print("🚀 启动传统模式处理")
        print("=" * 80)
        processor = TraditionalModeProcessor(args.api_key)
        results = processor.process_traditional_mode(args.input)
        
        if results['status'] == 'completed':
            print("\n✅ 传统模式处理完成！所有文件已生成。")
            print("\n📁 生成的文件:")
            print("   - 任务需求.json")
            print("   - 当前器件.json")
            print("   - 传统模式分析报告.md")
        else:
            print(f"\n❌ 传统模式处理失败: {results.get('error', 'Unknown error')}")
            return 1
    
    print("\n✨ 程序执行完成！")
    return 0


if __name__ == "__main__":
    exit(main())
