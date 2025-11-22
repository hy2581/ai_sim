#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.ai_integration.deepseek_api_client import DeepSeekAPIClient

REPORT_MD = project_root / "传统模式分析报告.md"
OUTPUT_HTML = project_root / "传统模式分析可视化.html"

SYSTEM_PROMPT = """
你是一个前端可视化工程专家。请将提供的报告内容转化为一个自包含的HTML文件。
要求：
1. 使用原生HTML+CSS+JS（可引用CDN）
2. 包含以下可视化：
   - 各器件类别匹配度的柱状图（CPU/GPU/内存/存储/传感器/通信/控制器）
   - 系统功耗与热管理的简要图示（文本+进度条）
   - 器件排布二维示意（SVG绘制：传感器、处理器、存储、通信、控制器模块的相对布局）
3. 页面结构：标题、摘要、图表区、二维排布区、数据表格（关键器件与参数）
4. 用中文注释关键区域；JS中内嵌数据对象，数据来源于报告解析的摘要信息。
5. 若报告中缺少某类数据，合理填充默认值并标注“估计”。

输出完整HTML。
"""

def main():
    if not REPORT_MD.exists():
        raise FileNotFoundError(f"报告未找到: {REPORT_MD}")
    report_text = REPORT_MD.read_text(encoding='utf-8')[:24000]

    model = os.getenv("VIS_MODEL", "qwen0.5b")
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    client = DeepSeekAPIClient(model_type=model, api_key=api_key)
    try:
        html = client._call_api(SYSTEM_PROMPT, report_text)
    except Exception:
        # 回退：生成一个基础HTML，嵌入报告摘要与占位图
        html = f"""
<!doctype html>
<html lang=zh>
<meta charset=utf-8>
<title>传统模式分析可视化</title>
<style>body{{font-family:sans-serif;margin:24px}} .chart,.layout{{border:1px solid #ccc;padding:12px;margin:12px 0}} .bar{{background:#4a90e2;height:18px;margin:6px 0}}</style>
<h1>传统模式分析可视化</h1>
<p>该页面为报告的可视化摘要。当外部API不可用时，展示占位图。</p>
<div class=chart>
  <h2>类别匹配度（占位）</h2>
  <div class=bar style="width:82%">总体匹配度 ≈ 0.82</div>
  <div class=bar style="width:90%">CPU 0.90</div>
  <div class=bar style="width:85%">GPU 0.85</div>
  <div class=bar style="width:80%">内存 0.80</div>
  <div class=bar style="width:85%">存储 0.85</div>
  <div class=bar style="width:75%">传感器 0.75</div>
  <div class=bar style="width:70%">通信 0.70</div>
  <div class=bar style="width:78%">控制器 0.78</div>
  <small>数据来源：报告文本推断（占位）。</small>
</div>
<div class=layout>
  <h2>器件二维排布示意（占位）</h2>
  <svg width=640 height=360 style="background:#f9f9f9">
    <rect x=20 y=20 width=180 height=80 fill="#ffd" stroke="#333"></rect>
    <text x=30 y=50>传感器阵列</text>
    <rect x=240 y=20 width=160 height=80 fill="#def" stroke="#333"></rect>
    <text x=250 y=50>CPU/GPU</text>
    <rect x=420 y=20 width=200 height=80 fill="#efe" stroke="#333"></rect>
    <text x=430 y=50>内存/存储</text>
    <rect x=20 y=140 width=220 height=80 fill="#fee" stroke="#333"></rect>
    <text x=30 y=170>控制器</text>
    <rect x=260 y=140 width=200 height=80 fill="#eef" stroke="#333"></rect>
    <text x=270 y=170>通信模块</text>
  </svg>
</div>
<div>
  <h2>报告摘要</h2>
  <pre style="white-space:pre-wrap">{report_text[:1200]}</pre>
</div>
</html>"""

    # 处理可能的代码块包裹
    html = html.strip()
    if html.startswith("```html"):
        html = html[7:]
    if html.endswith("```"):
        html = html[:-3]
    html = html.strip()

    OUTPUT_HTML.write_text(html, encoding='utf-8')
    print(f"已生成可视化HTML: {OUTPUT_HTML}")

if __name__ == "__main__":
    main()