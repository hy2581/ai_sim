# AI Sim 多模型支持说明

## 概述

AI Sim 项目现在支持三种 AI 模型选择,可以根据您的硬件环境和需求灵活切换。

## 支持的模型

### 1. qwen0.5b (默认) - 本地小模型
- **类型**: 本地 Ollama 模型
- **大小**: ~400MB
- **内存需求**: ~1GB
- **优点**: 快速、轻量、免费、数据隐私
- **缺点**: 能力相对有限
- **适用场景**: 快速测试、开发调试、资源受限环境
- **要求**: Ollama 服务运行中

### 2. deepseek-api - 外部 API
- **类型**: 在线 DeepSeek API
- **内存需求**: 无
- **优点**: 能力强、无需本地资源
- **缺点**: 需要网络、需要 API Key、有使用费用
- **适用场景**: 生产环境、需要高质量结果
- **要求**: 网络连接 + DeepSeek API Key

### 3. deepseek-r1 - 本地大模型
- **类型**: 本地 Ollama 大模型
- **大小**: ~28GB
- **内存需求**: 28.2GB 系统内存 + 48GB 显卡
- **优点**: 能力强、数据隐私、无网络依赖、免费
- **缺点**: 需要高性能硬件
- **适用场景**: 有高性能服务器、需要离线使用
- **要求**: Ollama 服务 + 大内存/显卡

## 使用方法

### 命令行直接调用

```bash
cd /home/hao123/chiplet/benchmark/ai_sim

# 1. 使用 qwen0.5b (默认，推荐用于当前电脑)
python3 main.py \
  --mode traditional \
  --input "设计一个用于深空探测的微系统" \
  --model qwen0.5b

# 2. 使用 deepseek-api (需要 API Key)
python3 main.py \
  --mode traditional \
  --input "设计一个用于深空探测的微系统" \
  --model deepseek-api \
  --api-key sk-your-api-key-here

# 3. 使用 deepseek-r1 (需要大显卡，迁移后使用)
python3 main.py \
  --mode traditional \
  --input "设计一个用于深空探测的微系统" \
  --model deepseek-r1
```

### 通过测试脚本

```bash
cd /home/hao123/chiplet/benchmark/ai_sim

# 测试 qwen0.5b
python3 test_ai_sim.py

# 测试其他模型
python3 test_ai_sim.py qwen0.5b
python3 test_ai_sim.py deepseek-api sk-your-api-key
python3 test_ai_sim.py deepseek-r1

# 查看帮助
python3 test_ai_sim.py --help
```

### 通过 Web 界面 (server.py)

```bash
# 1. 启动服务器
cd /home/hao123/chiplet/hello_world
python3 server.py

# 2. 在浏览器访问 http://localhost:8000

# 3. 在 AI 仿真页面选择模型:
#    - qwen0.5b (本地小模型)
#    - deepseek-api (在线 API)
#    - deepseek-r1 (本地大模型)
```

### API 调用示例

```python
import requests

url = "http://localhost:8000/api/ai-sim"
payload = {
    "sessionId": "test-session-123",
    "input": "设计一个用于深空探测的微系统",
    "model": "qwen0.5b",  # 或 "deepseek-api" 或 "deepseek-r1"
    "apiKey": "sk-xxx"    # 仅 deepseek-api 需要
}

response = requests.post(url, json=payload)
print(response.json())
```

## 环境准备

### 1. Ollama 服务 (qwen0.5b 和 deepseek-r1 需要)

```bash
# 检查 Ollama 是否运行
pgrep ollama

# 启动 Ollama 服务
ollama serve

# 查看已安装的模型
ollama list

# 如果模型不存在，拉取模型
ollama pull qwen:0.5b           # 小模型 (~400MB)
# ollama create deepseek-r1     # 大模型已创建
```

### 2. DeepSeek API Key (deepseek-api 需要)

```bash
# 在 .env 文件中配置 (可选)
echo "DEEPSEEK_API_KEY=sk-your-api-key" > .env

# 或在命令行中直接提供
--api-key sk-your-api-key
```

## 生成的文件

成功执行后，会在 ai_sim 目录生成以下文件:

1. **任务需求.json** - AI 解析的任务需求
2. **当前器件.json** - 选择的器件配置
3. **传统模式分析报告.md** - 完整的分析报告

## 性能对比

| 模型 | 响应速度 | 分析质量 | 硬件需求 | 网络需求 | 费用 |
|------|---------|---------|---------|---------|-----|
| qwen0.5b | 快 (5-10s) | 基础 | 低 (~1GB) | 无 | 免费 |
| deepseek-api | 中 (10-30s) | 优秀 | 无 | 需要 | 付费 |
| deepseek-r1 | 快 (5-15s) | 优秀 | 高 (28GB+48GB GPU) | 无 | 免费 |

## 故障排查

### Q: 提示 "无法连接到Ollama服务"
**A:** 
```bash
# 检查 Ollama 是否运行
pgrep ollama

# 启动 Ollama
ollama serve
```

### Q: deepseek-r1 提示 "model requires more system memory"
**A:** 您的当前电脑内存不足。请:
- 使用 qwen0.5b 小模型
- 或使用 deepseek-api 在线模型
- 或等迁移到有 48GB 显卡的电脑后使用

### Q: deepseek-api 提示 "API Key 错误"
**A:** 
```bash
# 检查 API Key 是否正确
echo $DEEPSEEK_API_KEY

# 重新设置 API Key
export DEEPSEEK_API_KEY=sk-your-real-key
```

### Q: qwen0.5b 模型不存在
**A:**
```bash
# 拉取模型
ollama pull qwen:0.5b
```

## 迁移到新电脑

当您迁移到有 48GB 显卡的新电脑后:

1. **导出 WSL** (在旧电脑):
```bash
wsl --export Ubuntu D:\wsl-backup.tar
```

2. **导入 WSL** (在新电脑):
```bash
wsl --import Ubuntu C:\WSL\Ubuntu D:\wsl-backup.tar
```

3. **验证模型**:
```bash
cd /home/hao123/chiplet/benchmark/ai_sim
ollama list  # 查看 deepseek-r1 是否存在
python3 test_ai_sim.py deepseek-r1  # 测试大模型
```

## 推荐使用方式

- **当前电脑**: 使用 `qwen0.5b` (内存受限)
- **迁移后**: 使用 `deepseek-r1` (有大显卡)
- **紧急情况**: 使用 `deepseek-api` (需要最佳质量)

## 技术细节

模型选择通过以下方式传递:

1. **命令行参数**: `--model qwen0.5b`
2. **API 参数**: `{"model": "qwen0.5b"}`
3. **默认值**: 如果不指定，默认使用 `qwen0.5b`

底层实现:
- `qwen0.5b` 和 `deepseek-r1`: 调用本地 Ollama API (http://localhost:11434)
- `deepseek-api`: 调用外部 DeepSeek API (https://api.deepseek.com)
