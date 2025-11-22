#!/usr/bin/env python3
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
REPORT = ROOT / "传统模式分析报告.md"
CURRENT = ROOT / "当前器件.json"
HTML = ROOT / "传统模式分析可视化.html"

def parse_compatibility(md: str):
    cats = {
        'CPU': None,
        'GPU': None,
        '内存': None,
        '存储': None,
        '传感器': None,
        '通信': None,
        '控制器': None,
    }
    # Patterns for sections like "CPU（匹配度0.9）" or "内存系统（匹配度0.80）"
    for m in re.finditer(r"([A-Za-z\u4e00-\u9fff]+)（匹配度([0-9.]+)）", md):
        name = m.group(1)
        val = float(m.group(2))
        key = None
        if 'CPU' in name:
            key = 'CPU'
        elif 'GPU' in name:
            key = 'GPU'
        elif '内存' in name:
            key = '内存'
        elif '存储' in name:
            key = '存储'
        elif '传感' in name:
            key = '传感器'
        elif '通信' in name or '接口' in name:
            key = '通信'
        elif '控制' in name:
            key = '控制器'
        if key:
            cats[key] = val
    # Fallback defaults
    defaults = {'CPU':0.9,'GPU':0.85,'内存':0.8,'存储':0.85,'传感器':0.75,'通信':0.85,'控制器':0.78}
    for k,v in defaults.items():
        if cats[k] is None:
            cats[k] = v
    return cats

def parse_devices(cur: dict):
    rows = []
    supp = cur.get('supplemented_devices', {})
    mapping = {
        'cpu':'CPU', 'gpu':'GPU','memory':'内存','storage':'存储','sensors':'传感器','communication':'通信','controllers':'控制器'
    }
    for ckey, label in mapping.items():
        items = supp.get(ckey, [])
        if not items:
            continue
        for d in items[:2]:
            name = d.get('name') or d.get('器件名称或类型') or '未知器件'
            params = d.get('parameters')
            if isinstance(params, dict):
                # pick up to 3 key params
                kvs = []
                for pk in ['architecture','cores','frequency_mhz','capacity_gb','data_rate_mbps','data_rate_gbps','bands_count']:
                    if pk in params:
                        kvs.append(f"{pk}:{params[pk]}")
                if not kvs:
                    kvs = [k for k in list(params.keys())[:3]]
                pstr = ', '.join(kvs)
            elif isinstance(params, list):
                pstr = ', '.join(params[:3])
            else:
                pstr = '—'
            rows.append({'category':label,'model':name,'params':pstr})
    return rows

def replace_report_data(html_text: str, cats: dict, rows: list):
    # Build JS object text
    labels = ['CPU','GPU','内存','存储','传感器','通信','控制器']
    values = [cats[l] for l in labels]
    devices_js = ',\n                '.join([
        f"{{category: '{r['category']}', model: '{r['model']}', params: '{r['params']}', compatibility: {cats.get(r['category'], 0.75)} }}"
        for r in rows[:14]
    ])
    js_block = (
        "const reportData = {\n"
        "            compatibility: {\n"
        f"                labels: {json.dumps(labels, ensure_ascii=False)},\n"
        f"                values: {json.dumps(values)},\n"
        "                colors: [\n"
        "                    'rgba(79, 195, 247, 0.7)',\n"
        "                    'rgba(66, 165, 245, 0.7)',\n"
        "                    'rgba(41, 182, 246, 0.7)',\n"
        "                    'rgba(3, 155, 229, 0.7)',\n"
        "                    'rgba(2, 119, 189, 0.7)',\n"
        "                    'rgba(21, 101, 192, 0.7)',\n"
        "                    'rgba(13, 71, 161, 0.7)'\n"
        "                ]\n"
        "            },\n"
        "            devices: [\n"
        f"                {devices_js}\n"
        "            ],\n"
        "            powerManagement: { totalPower: 28, maxPower: 35, thermalEfficiency: 75, radiationResistance: 92 }\n"
        "        };"
    )
    # Replace existing const reportData block
    new_html = re.sub(r"const reportData = \{[\s\S]*?\};", js_block, html_text)
    # Fix SVG newline label by using tspan
    new_html = new_html.replace("处理器\n(CPU+GPU)", "处理器 (CPU+GPU)")
    return new_html

def main():
    md = REPORT.read_text(encoding='utf-8') if REPORT.exists() else ''
    cur = json.loads(CURRENT.read_text(encoding='utf-8')) if CURRENT.exists() else {}
    cats = parse_compatibility(md)
    rows = parse_devices(cur)
    html_text = HTML.read_text(encoding='utf-8')
    enriched = replace_report_data(html_text, cats, rows)
    HTML.write_text(enriched, encoding='utf-8')
    print("已填充图表与表格数据并修正SVG文本")

if __name__ == '__main__':
    main()