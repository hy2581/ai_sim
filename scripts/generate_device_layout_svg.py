#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
CUR = ROOT / "当前器件.json"
LIB = ROOT / "device_library.json"
OUT = ROOT / "器件排布图.svg"

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
    mapping = ['sensors','controllers','cpu','gpu','memory','storage','communication']
    picked = {k: [] for k in mapping}
    for cat in mapping:
        items = supp.get(cat, [])
        if items:
            picked[cat] = items[:4]
        else:
            # fall back to library
            key = cat
            if cat == 'cpu':
                key = 'cpus'
            if cat == 'gpu':
                key = 'gpus'
            picked[cat] = lib.get('device_library', {}).get(key, [])[:2]
    return picked

def layout_positions(picked):
    cols = ['sensors','controllers','cpu','gpu','memory','storage','communication']
    x_step = 160
    positions = {}
    # base rows to create bands; closer spacing
    base_y = {
        'sensors': 80,
        'controllers': 170,
        'cpu': 130,
        'gpu': 80,
        'memory': 180,
        'storage': 250,
        'communication': 130
    }
    # simple comp size to avoid overlaps
    comp_w = 140
    comp_h = 50
    # place per column and compress vertically using minimal gap
    min_gap = 6
    for ci, cat in enumerate(cols):
        x = 60 + ci * x_step
        items = picked.get(cat, [])
        y = base_y.get(cat, 120)
        for i, d in enumerate(items[:3]):
            name = d.get('name') or d.get('器件名称或类型') or f"{cat}_{i}"
            # ensure non-overlap within column
            if positions.get(cat):
                last = positions[cat][-1]
                last_y = last[2]
                y = max(y, last_y + comp_h + min_gap)
            positions.setdefault(cat, []).append((name, x, y))
            y += comp_h + min_gap
    return positions

def make_svg(positions):
    width = 1320
    height = 560
    comp_w = 140
    comp_h = 50
    colors = {
        'sensors': '#ffe082', 'controllers': '#d1c4e9', 'cpu': '#80d8ff', 'gpu': '#b3e5fc', 'memory': '#64b5f6', 'storage': '#1e88e5', 'communication': '#66bb6a'
    }
    reps = {}
    for cat, arr in positions.items():
        if arr:
            reps[cat] = arr[0]
    def comp(name, x, y, cat):
        cx = colors.get(cat, '#ddd')
        rect = f"<rect x='{x}' y='{y}' width='{comp_w}' height='{comp_h}' rx='10' fill='{cx}' stroke='#263238' stroke-width='2'/>"
        text = f"<text x='{x+comp_w/2}' y='{y+comp_h/2+5}' fill='#0d47a1' font-size='12' text-anchor='middle'>{name}</text>"
        pins = []
        for i in range(4):
            pins.append(f"<circle cx='{x+10+i*15}' cy='{y-8}' r='3' fill='#b0bec5' stroke='#455a64' />")
            pins.append(f"<circle cx='{x+comp_w-10-i*15}' cy='{y+comp_h+8}' r='3' fill='#b0bec5' stroke='#455a64' />")
        for i in range(3):
            pins.append(f"<circle cx='{x-8}' cy='{y+10+i*15}' r='3' fill='#b0bec5' stroke='#455a64' />")
            pins.append(f"<circle cx='{x+comp_w+8}' cy='{y+comp_h-10-i*15}' r='3' fill='#b0bec5' stroke='#455a64' />")
        return [rect, text] + pins
    def hp(cat):
        name, x, y = reps[cat]
        return (x, y, x+comp_w, y+comp_h)
    def path_bus(x1, y1, x2, y2, color):
        midx = (x1 + x2) / 2
        return f"<path d='M {x1} {y1} L {midx} {y1} L {midx} {y2} L {x2} {y2}' stroke='{color}' stroke-width='3' fill='none' marker-end='url(#arrow)'/>"
    board = [
        f"<rect x='10' y='10' width='{width-20}' height='{height-20}' rx='20' fill='#2e7d32' stroke='#1b5e20' stroke-width='6'/>"
    ]
    grid = []
    for gx in range(40, width-40, 40):
        grid.append(f"<line x1='{gx}' y1='20' x2='{gx}' y2='{height-20}' stroke='#388e3c' stroke-width='1' opacity='0.2'/>")
    for gy in range(40, height-40, 40):
        grid.append(f"<line x1='20' y1='{gy}' x2='{width-20}' y2='{gy}' stroke='#388e3c' stroke-width='1' opacity='0.2'/>")
    comps = []
    for cat, arr in positions.items():
        if arr:
            name, x, y = arr[0]
            comps += comp(name, x, y, cat)
    traces = []
    if 'cpu' in reps and 'gpu' in reps:
        x1, y1, x1r, y1b = hp('cpu')
        x2, y2, x2r, y2b = hp('gpu')
        traces.append(path_bus(x1r+8, (y1+y1b)/2, x2-8, (y2+y2b)/2, '#ffb300'))
    if 'cpu' in reps and 'memory' in reps:
        x1, y1, x1r, y1b = hp('cpu')
        x2, y2, x2r, y2b = hp('memory')
        traces.append(path_bus(x1r+8, y1b-10, x2-8, y2+10, '#ffa000'))
    if 'cpu' in reps and 'storage' in reps:
        x1, y1, x1r, y1b = hp('cpu')
        x2, y2, x2r, y2b = hp('storage')
        traces.append(path_bus(x1r+8, y1b-5, x2-8, y2+20, '#fb8c00'))
    if 'cpu' in reps and 'communication' in reps:
        x1, y1, x1r, y1b = hp('cpu')
        x2, y2, x2r, y2b = hp('communication')
        traces.append(path_bus(x1r+8, y1+10, x2-8, y2+10, '#1976d2'))
    if 'sensors' in reps and 'cpu' in reps:
        x1, y1, x1r, y1b = hp('sensors')
        x2, y2, x2r, y2b = hp('cpu')
        traces.append(path_bus(x1r+8, y1+10, x2-8, y2+10, '#7cb342'))
    if 'controllers' in reps and 'cpu' in reps:
        x1, y1, x1r, y1b = hp('controllers')
        x2, y2, x2r, y2b = hp('cpu')
        traces.append(path_bus(x1r+8, y1+10, x2-8, y2+10, '#8e24aa'))
    legend_y = height - 90
    legend = [
        f"<rect x='20' y='{legend_y}' width='340' height='70' fill='#1b5e20' stroke='#263238'/>",
        f"<text x='35' y='{legend_y+18}' fill='#c8e6c9' font-size='12'>Legend</text>",
        f"<line x1='35' y1='{legend_y+32}' x2='85' y2='{legend_y+32}' stroke='#ffb300' stroke-width='3'/>",
        f"<text x='95' y='{legend_y+36}' fill='#c8e6c9' font-size='12'>Data bus</text>",
        f"<line x1='35' y1='{legend_y+46}' x2='85' y2='{legend_y+46}' stroke='#1976d2' stroke-width='3'/>",
        f"<text x='95' y='{legend_y+50}' fill='#c8e6c9' font-size='12'>Comm</text>",
        f"<line x1='35' y1='{legend_y+60}' x2='85' y2='{legend_y+60}' stroke='#8e24aa' stroke-width='3'/>",
        f"<text x='95' y='{legend_y+64}' fill='#c8e6c9' font-size='12'>Control</text>",
        f"<text x='20' y='{height-12}' fill='#c8e6c9' font-size='11'>说明：仅显示代表器件及关键总线，实际互联简化</text>"
    ]
    svg = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        "<defs><marker id='arrow' viewBox='0 0 10 10' refX='10' refY='5' markerWidth='6' markerHeight='6' orient='auto-start-reverse'><path d='M 0 0 L 10 5 L 0 10 z' fill='#cfd8dc'/></marker></defs>",
        *board,
        *grid,
        *traces,
        *comps,
        *legend,
        "</svg>"
    ]
    return "\n".join(svg)

def main():
    cur, lib = load_data()
    picked = pick_devices(cur, lib)
    positions = layout_positions(picked)
    svg = make_svg(positions)
    OUT.write_text(svg, encoding='utf-8')
    print(f"生成: {OUT}")

if __name__ == '__main__':
    main()