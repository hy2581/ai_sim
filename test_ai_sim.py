#!/usr/bin/env python3
"""
测试 ai_sim 项目的三种模型调用
"""

import subprocess
import sys
import os

def test_model(model_type, input_text, api_key=None):
    """测试指定模型"""
    print(f"\n{'='*60}")
    print(f"测试模型: {model_type}")
    print(f"输入: {input_text}")
    print(f"{'='*60}\n")
    
    # 构造命令
    cmd = [
        "python3", "main.py",
        "--mode", "traditional",
        "--input", input_text,
        "--model", model_type
    ]
    
    # 只有 deepseek-api 需要 API Key
    if model_type == "deepseek-api":
        if not api_key:
            print("❌ deepseek-api 模型需要提供 API Key")
            print("   用法: python3 test_ai_sim.py deepseek-api <api_key>")
            return False
        cmd.extend(["--api-key", api_key])
    
    try:
        print(f"执行命令: {' '.join(cmd)}\n")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        print("输出:")
        print(result.stdout)
        
        if result.stderr:
            print("\n错误输出:")
            print(result.stderr)
        
        if result.returncode == 0:
            print(f"\n✅ 测试成功 (退出码: {result.returncode})")
            
            # 检查生成的文件
            files = ["任务需求.json", "当前器件.json", "传统模式分析报告.md"]
            print("\n检查生成的文件:")
            for file in files:
                if os.path.exists(file):
                    print(f"  ✅ {file}")
                else:
                    print(f"  ❌ {file} (未找到)")
            
            return True
        else:
            print(f"\n❌ 测试失败 (退出码: {result.returncode})")
            return False
            
    except subprocess.TimeoutExpired:
        print("\n❌ 测试超时 (120秒)")
        return False
    except Exception as e:
        print(f"\n❌ 测试异常: {e}")
        return False


def show_usage():
    """显示使用说明"""
    print("""
AI Sim 模型测试工具

支持的模型:
  1. qwen0.5b      - 本地 Ollama 小模型 (默认)
  2. deepseek-api  - 外部 DeepSeek API (需要 API Key)
  3. deepseek-r1   - 本地 Ollama DeepSeek-R1 (需要大显卡)

用法:
  # 测试 qwen0.5b (默认)
  python3 test_ai_sim.py

  # 测试指定模型
  python3 test_ai_sim.py qwen0.5b
  python3 test_ai_sim.py deepseek-r1

  # 测试 deepseek-api (需要 API Key)
  python3 test_ai_sim.py deepseek-api sk-your-api-key-here

  # 自定义输入文本
  python3 test_ai_sim.py qwen0.5b "你的自定义输入文本"

注意:
  - 确保 Ollama 服务正在运行 (ollama serve)
  - qwen0.5b 需要约 400MB 内存
  - deepseek-r1 需要 28GB 内存 + 48GB 显卡
  - deepseek-api 需要网络连接和 API Key
""")


if __name__ == "__main__":
    # 默认测试文本
    default_input = "设计一个用于火星探测的微型系统，需要图像采集和数据处理功能"
    
    # 解析命令行参数
    if len(sys.argv) == 1:
        # 无参数，使用默认
        model = "qwen0.5b"
        input_text = default_input
        api_key = None
    elif sys.argv[1] in ["-h", "--help", "help"]:
        # 显示帮助
        show_usage()
        sys.exit(0)
    elif len(sys.argv) == 2:
        # 只指定模型
        model = sys.argv[1]
        input_text = default_input
        api_key = None
    elif len(sys.argv) == 3:
        # 模型 + API Key 或 自定义文本
        model = sys.argv[1]
        if model == "deepseek-api":
            # 第二个参数是 API Key
            api_key = sys.argv[2]
            input_text = default_input
        else:
            # 第二个参数是自定义文本
            input_text = sys.argv[2]
            api_key = None
    elif len(sys.argv) == 4:
        # 模型 + API Key + 自定义文本
        model = sys.argv[1]
        api_key = sys.argv[2]
        input_text = sys.argv[3]
    else:
        print("❌ 参数错误")
        show_usage()
        sys.exit(1)
    
    # 验证模型类型
    if model not in ["qwen0.5b", "deepseek-api", "deepseek-r1"]:
        print(f"❌ 不支持的模型类型: {model}")
        show_usage()
        sys.exit(1)
    
    # 切换到 ai_sim 目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("AI Sim 模型测试")
    print("=" * 60)
    print(f"工作目录: {os.getcwd()}")
    
    # 执行测试
    success = test_model(model, input_text, api_key)
    
    sys.exit(0 if success else 1)
