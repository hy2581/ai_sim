#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
CUR = ROOT / "当前器件.json"
LIB = ROOT / "device_library.json"
OUT = ROOT / "器件排布图.html"

def load_data():
    cur = {}
    lib = {}
    if CUR.exists():
        cur = json.loads(CUR.read_text(encoding='utf-8'))
    if LIB.exists():
        lib = json.loads(LIB.read_text(encoding='utf-8'))
    return cur, lib

def pick_devices(cur, lib):
    supp = cur.get('supplemented_devices', {})
    if not supp:
        supp = {}
    mapping = ['cpu','gpu','memory','storage','sensors','communication','controllers']
    picked = {k: [] for k in mapping}
    for cat in mapping:
        items = supp.get(cat, [])
        if items:
            picked[cat] = items[:4]
        else:
            lib_items = lib.get('device_library', {}).get(cat if cat != 'cpu' else 'cpus', [])
            if cat == 'gpu':
                lib_items = lib.get('device_library', {}).get('gpus', [])
            picked[cat] = lib_items[:2]
    return picked

def make_html(picked):
    nodes = []
    edges = []
    cat_order = ['sensors','controllers','cpu','gpu','memory','storage','communication']
    idx = 0
    id_map = {}
    for cat in cat_order:
        for d in picked.get(cat, []):
            name = d.get('name') or d.get('器件名称或类型') or f"{cat}_{idx}"
            nid = f"{cat}_{idx}"
            id_map.setdefault(cat, []).append(nid)
            nodes.append({"data": {"id": nid, "label": name, "category": cat}})
            idx += 1
    def connect(frm, to):
        for a in id_map.get(frm, []):
            for b in id_map.get(to, []):
                edges.append({"data": {"source": a, "target": b}})
    connect('cpu','gpu')
    connect('cpu','memory')
    connect('cpu','storage')
    connect('cpu','communication')
    connect('sensors','cpu')
    connect('sensors','communication')
    connect('controllers','sensors')
    connect('controllers','cpu')

    html = f"""
<!doctype html>
<html lang=zh>
<meta charset=utf-8>
<title>器件排布二维图</title>
<script src="https://cdn.jsdelivr.net/npm/cytoscape@3.28.0/dist/cytoscape.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/dagre@0.8.5/dist/dagre.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/cytoscape-dagre@2.5.0/cytoscape-dagre.js"></script>
<style>
body{{font-family:sans-serif;margin:0;background:#f5f7fa}}
#cy{{width:100vw;height:90vh;display:block}}
.panel{{padding:10px}}
</style>
<div class=panel>
  <h1>器件排布与互联</h1>
  <p>基于当前器件与器件库生成的二维拓扑，节点按类别着色，边表示互联路径。</p>
</div>
<div id="cy"></div>
<script>
const elements = {json.dumps(nodes + edges, ensure_ascii=False)};
const cy = cytoscape({{
  container: document.getElementById('cy'),
  elements: elements,
  style: [
    {{ selector: 'node', style: {{ 'label': 'data(label)', 'text-valign': 'center', 'text-halign': 'center', 'color': '#222', 'width': 100, 'height': 40, 'shape': 'round-rectangle', 'font-size': 12 }} }},
    {{ selector: 'edge', style: {{ 'width': 2, 'line-color': '#888', 'target-arrow-color': '#888', 'target-arrow-shape': 'triangle', 'curve-style': 'bezier' }} }},
    {{ selector: 'node[category = "cpu"]', style: {{ 'background-color': '#4fc3f7' }} }},
    {{ selector: 'node[category = "gpu"]', style: {{ 'background-color': '#81d4fa' }} }},
    {{ selector: 'node[category = "memory"]', style: {{ 'background-color': '#29b6f6' }} }},
    {{ selector: 'node[category = "storage"]', style: {{ 'background-color': '#0288d1' }} }},
    {{ selector: 'node[category = "sensors"]', style: {{ 'background-color': '#ffb74d' }} }},
    {{ selector: 'node[category = "communication"]', style: {{ 'background-color': '#4caf50' }} }},
    {{ selector: 'node[category = "controllers"]', style: {{ 'background-color': '#ba68c8' }} }},
  ],
  layout: {{ name: 'dagre', rankDir: 'LR', nodeSep: 50, rankSep: 120 }}
}});
</script>
</html>
"""
    return html

def main():
    cur, lib = load_data()
    picked = pick_devices(cur, lib)
    html = make_html(picked)
    OUT.write_text(html, encoding='utf-8')
    print(f"生成: {OUT}")

if __name__ == '__main__':
    main()