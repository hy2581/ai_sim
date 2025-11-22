#!/usr/bin/env python3
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
REQ = ROOT / "任务需求.json"
CUR = ROOT / "当前器件.json"
REP = ROOT / "传统模式分析报告.md"
SVG = ROOT / "器件排布图.svg"
OUT = ROOT / "传统模式分析可视化.html"

def read_json(p: Path):
    if p.exists():
        return json.loads(p.read_text(encoding='utf-8'))
    return {}

def read_text(p: Path, limit=40000):
    if p.exists():
        return p.read_text(encoding='utf-8')[:limit]
    return ""

def parse_compatibility(md: str):
    cats = {'CPU':0.9,'GPU':0.85,'内存':0.8,'存储':0.85,'传感器':0.75,'通信':0.85,'控制器':0.78}
    for m in re.finditer(r"([A-Za-z\u4e00-\u9fff]+)（匹配度([0-9.]+)）", md):
        name = m.group(1)
        val = float(m.group(2))
        if 'CPU' in name:
            cats['CPU']=val
        elif 'GPU' in name:
            cats['GPU']=val
        elif '内存' in name:
            cats['内存']=val
        elif '存储' in name:
            cats['存储']=val
        elif '传感' in name:
            cats['传感器']=val
        elif '通信' in name or '接口' in name:
            cats['通信']=val
        elif '控制' in name:
            cats['控制器']=val
    return cats

def build_devices(cur: dict, md_cats: dict):
    rows = []
    supp = cur.get('supplemented_devices', {})
    mapping = {'cpu':'CPU','gpu':'GPU','memory':'内存','storage':'存储','sensors':'传感器','communication':'通信','controllers':'控制器'}
    for k,label in mapping.items():
        items = supp.get(k, [])
        for d in items[:6]:
            name = d.get('name') or d.get('器件名称或类型') or '未知器件'
            params = d.get('parameters')
            if isinstance(params, dict):
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
                pstr = ''
            rwe = d.get('real_world_equivalent') or ''
            rows.append({'category':label,'model':name,'params':pstr,'equiv':rwe,'compat':md_cats.get(label,0.75)})
    return rows

def main():
    req = read_json(REQ)
    cur = read_json(CUR)
    md = read_text(REP)
    cats = parse_compatibility(md)
    devs = build_devices(cur, cats)
    svg_content = read_text(SVG, limit=500000)
    labels = ["CPU","GPU","内存","存储","传感器","通信","控制器"]
    values = [cats[l] for l in labels]
    table_rows = "\n".join([
        f"<tr><td>{d['category']}</td><td>{d['model']}</td><td>{d['params']}</td><td>{d['equiv']}</td><td>{int(d['compat']*100)}%</td></tr>" for d in devs
    ])
    html = f"""
<!doctype html>
<html lang=zh>
<meta charset=utf-8>
<title>传统模式分析可视化（数据驱动）</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body{{font-family:sans-serif;margin:0;background:#f5f7fa;color:#222}}
.wrap{{max-width:1200px;margin:20px auto;padding:20px;background:#fff;box-shadow:0 4px 16px rgba(0,0,0,.08);border-radius:12px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
.card{{padding:16px;border:1px solid #e0e0e0;border-radius:10px}}
table{{width:100%;border-collapse:collapse}}
th,td{{border-bottom:1px solid #eee;padding:8px;text-align:left;font-size:13px}}
.svgbox{{border:1px solid #e0e0e0;border-radius:10px;overflow:hidden}}
</style>
<div class=wrap>
  <h1>传统模式分析可视化（数据驱动）</h1>
  <div class=grid>
    <div class=card>
      <h3>类别匹配度</h3>
      <canvas id="compat"></canvas>
    </div>
    <div class=card>
      <h3>关键器件与参数</h3>
      <table>
        <thead><tr><th>类别</th><th>型号</th><th>参数</th><th>现实器件</th><th>匹配度</th></tr></thead>
        <tbody>
          {table_rows}
        </tbody>
      </table>
    </div>
  </div>
  <div class=card>
    <h3>器件排布（PCB风格）</h3>
    <div class=svgbox>
      {svg_content}
    </div>
  </div>
</div>
<script>
const labels = {json.dumps(labels, ensure_ascii=False)};
const values = {json.dumps(values)};
new Chart(document.getElementById('compat').getContext('2d'), {{
  type: 'bar',
  data: {{ labels, datasets: [{{ label: '匹配度', data: values, backgroundColor: 'rgba(66,165,245,0.6)' }}] }},
  options: {{ scales: {{ y: {{ beginAtZero:true, max:1, ticks:{{ callback:(v)=> (v*100)+'%' }} }} }} }}
}});
</script>
</html>
"""
    OUT.write_text(html, encoding='utf-8')
    print(f"生成: {OUT}")

if __name__ == '__main__':
    main()