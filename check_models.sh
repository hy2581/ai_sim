#!/bin/bash
# 快速验证脚本 - 检查所有配置

echo "========================================"
echo "AI Sim 多模型配置检查"
echo "========================================"
echo ""

# 检查当前目录
CURRENT_DIR=$(pwd)
echo "当前目录: $CURRENT_DIR"
echo ""

# 检查 Ollama 服务
echo "1. 检查 Ollama 服务..."
if pgrep -x "ollama" > /dev/null; then
    echo "   ✅ Ollama 服务正在运行"
    echo ""
    echo "   已安装的模型:"
    ollama list | grep -E "qwen|deepseek" | head -5
else
    echo "   ❌ Ollama 服务未运行"
    echo "   启动命令: ollama serve"
fi

echo ""
echo "2. 支持的AI模型:"
echo "   ┌─────────────────┬──────────┬──────────────┬─────────┐"
echo "   │ 模型            │ 内存需求 │ 适用场景     │ 费用    │"
echo "   ├─────────────────┼──────────┼──────────────┼─────────┤"
echo "   │ qwen0.5b        │ ~1GB     │ 快速测试     │ 免费    │"
echo "   │ deepseek-api    │ 无       │ 生产环境     │ 付费    │"
echo "   │ deepseek-r1     │ 28GB+48G │ 高性能服务器 │ 免费    │"
echo "   └─────────────────┴──────────┴──────────────┴─────────┘"

echo ""
echo "3. 快速测试命令:"
echo ""
echo "   测试 qwen0.5b (推荐用于当前电脑):"
echo "   $ python3 test_ai_sim.py"
echo ""
echo "   测试 deepseek-api (需要 API Key):"
echo "   $ python3 test_ai_sim.py deepseek-api sk-your-key"
echo ""
echo "   测试 deepseek-r1 (需要大显卡):"
echo "   $ python3 test_ai_sim.py deepseek-r1"
echo ""
echo "   查看详细帮助:"
echo "   $ python3 test_ai_sim.py --help"

echo ""
echo "4. 通过 Web 界面使用:"
echo "   $ cd /home/hao123/chiplet/hello_world"
echo "   $ python3 server.py"
echo "   然后在浏览器访问: http://localhost:8000"

echo ""
echo "5. 查看详细文档:"
echo "   $ cat MODEL_GUIDE.md"

echo ""
echo "========================================"
echo "配置检查完成"
echo "========================================"
