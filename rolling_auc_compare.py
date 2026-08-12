#!/usr/bin/env python3
"""Generate a self-contained HTML report comparing rolling AUC curves."""

import argparse
import csv
import html
import json
import os
import sys


DEFAULT_BASE_DIR = "/home/luban/rank-ssl/chenpinyuan/tf_rank_BR"
TASKS = ("buy", "cat", "click", "ext")
COLORS = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2")


def resolve_experiment(exp, base_dir):
    return exp if os.path.isabs(exp) else os.path.join(base_dir, exp)


def load_metrics(exp, base_dir):
    exp_dir = resolve_experiment(exp, base_dir)
    path = os.path.join(exp_dir, "model", "rolling_metrics.tsv")
    if not os.path.isfile(path):
        raise FileNotFoundError("rolling metrics not found: %s" % path)
    rows = {}
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            # The multi-day 20260721-20260724 row is the fixed-window check, not a rolling point.
            if row["test_start_day"] != row["test_end_day"]:
                continue
            key = (row["test_end_day"], row["task"])
            rows[key] = {
                "checkpoint_day": row["checkpoint_day"],
                "auc": float(row["auc"]),
                "size": int(row["size"]),
                "pos": int(row["pos"]),
            }
    if not rows:
        raise ValueError("no single-day rolling rows in %s" % path)
    return os.path.basename(exp_dir.rstrip(os.sep)), path, rows


def validate_alignment(data):
    names = list(data)
    base_name = names[0]
    base_keys = set(data[base_name])
    errors = []
    for name in names[1:]:
        keys = set(data[name])
        if keys != base_keys:
            errors.append("%s date/task keys differ from baseline" % name)
            continue
        for key in sorted(base_keys):
            base = data[base_name][key]
            cur = data[name][key]
            if (cur["checkpoint_day"], cur["size"], cur["pos"]) != (
                    base["checkpoint_day"], base["size"], base["pos"]):
                errors.append("%s mismatch at %s/%s" % (name, key[0], key[1]))
    return errors


def build_payload(data):
    names = list(data)
    dates = sorted({day for day, _ in data[names[0]]})
    payload = {"names": names, "dates": dates, "tasks": {}}
    for task in TASKS:
        series = {}
        for name in names:
            series[name] = [data[name].get((day, task), {}).get("auc") for day in dates]
        payload["tasks"][task] = series
    return payload


def render_html(payload, source_paths, alignment_errors):
    safe_payload = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    source_items = "".join("<li><code>%s</code></li>" % html.escape(path) for path in source_paths)
    if alignment_errors:
        alignment = '<div class="status bad">数据对齐失败：%s</div>' % html.escape("; ".join(alignment_errors))
    else:
        alignment = '<div class="status good">数据对齐通过：所有分支的 checkpoint_day、size、pos 完全一致。</div>'
    cards = "".join(
        '<section class="card"><h2>%s AUC</h2><div id="chart-%s" class="chart"></div>'
        '<div id="summary-%s" class="summary"></div><div id="table-%s" class="table-wrap"></div></section>'
        % (task.upper(), task, task, task) for task in TASKS
    )
    template = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rolling AUC Comparison</title>
<style>
body{margin:0;background:#f4f6f8;color:#17202a;font:14px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:28px}.head,.card{background:white;border:1px solid #dfe4ea;border-radius:12px;box-shadow:0 2px 8px #0000000a}.head{padding:22px;margin-bottom:18px}.head h1{margin:0 0 8px}.status{padding:10px 12px;border-radius:8px;margin:14px 0}.good{background:#eaf8ef;color:#176b36}.bad{background:#fdecec;color:#a51d2d}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.card{padding:18px;overflow:hidden}.card h2{margin:0 0 10px}.chart{height:390px}.chart svg{width:100%;height:100%;display:block}.legend{display:flex;flex-wrap:wrap;gap:12px;margin:4px 0 8px}.legend span:before{content:"";display:inline-block;width:18px;height:3px;margin-right:6px;vertical-align:middle;background:var(--c)}.summary{margin:8px 0 12px}.summary table,.table-wrap table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}.summary th,.summary td,.table-wrap th,.table-wrap td{border:1px solid #e2e6ea;padding:6px;text-align:right}.summary th:first-child,.summary td:first-child,.table-wrap th:first-child,.table-wrap td:first-child{text-align:left}.table-wrap{overflow:auto;max-height:260px}.table-wrap thead{position:sticky;top:0;background:#f7f8fa}.pos{color:#16813b}.neg{color:#c52a3a}code{font-size:12px}details{margin-top:10px}@media(max-width:950px){.grid{grid-template-columns:1fr}}
</style></head><body><main><div class="head"><h1>滚动测试 AUC 多实验对比</h1>
<p>每个点严格表示 checkpoint(D) 在 D+1 单日数据上的 AUC；固定窗口行已排除。差值单位为千分位（‰）。</p>
__ALIGNMENT__<details><summary>数据来源</summary><ul>__SOURCES__</ul></details></div><div class="grid">__CARDS__</div></main>
<script>
const P=__PAYLOAD__, C=__COLORS__;
function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function draw(task){
 const names=P.names, dates=P.dates, series=P.tasks[task], base=names[0];
 const vals=names.flatMap(n=>series[n]).filter(v=>v!=null), lo=Math.min(...vals), hi=Math.max(...vals), pad=Math.max((hi-lo)*.12,.00005);
 const W=760,H=350,L=66,R=20,T=26,B=58, ymin=lo-pad,ymax=hi+pad, x=i=>L+i*(W-L-R)/Math.max(dates.length-1,1), y=v=>T+(ymax-v)*(H-T-B)/(ymax-ymin);
 let svg=`<svg viewBox="0 0 ${W} ${H}" role="img">`;
 for(let i=0;i<6;i++){let v=ymin+(ymax-ymin)*i/5,yy=y(v);svg+=`<line x1="${L}" y1="${yy}" x2="${W-R}" y2="${yy}" stroke="#e7eaee"/><text x="${L-8}" y="${yy+4}" text-anchor="end" fill="#68727d" font-size="11">${v.toFixed(6)}</text>`;}
 dates.forEach((d,i)=>{svg+=`<text x="${x(i)}" y="${H-28}" transform="rotate(-35 ${x(i)} ${H-28})" text-anchor="end" fill="#68727d" font-size="11">${d}</text>`});
 names.forEach((n,j)=>{let pts=series[n].map((v,i)=>v==null?null:`${x(i)},${y(v)}`).filter(Boolean).join(' ');svg+=`<polyline points="${pts}" fill="none" stroke="${C[j%C.length]}" stroke-width="${j===0?3:2}"/>`;series[n].forEach((v,i)=>{if(v!=null)svg+=`<circle cx="${x(i)}" cy="${y(v)}" r="3" fill="${C[j%C.length]}"><title>${esc(n)} ${dates[i]}: ${v.toFixed(6)}</title></circle>`})});
 svg+='</svg>'; document.getElementById('chart-'+task).innerHTML=svg+`<div class="legend">${names.map((n,j)=>`<span style="--c:${C[j%C.length]}">${esc(n)}</span>`).join('')}</div>`;
 let sr='<table><thead><tr><th>experiment</th><th>mean AUC</th><th>mean Δ‰</th><th>胜/总天数</th><th>最差 Δ‰</th></tr></thead><tbody>';
 names.forEach(n=>{let a=series[n],ds=a.map((v,i)=>v==null||series[base][i]==null?null:(v-series[base][i])*1000).filter(v=>v!=null),mean=a.filter(v=>v!=null).reduce((s,v)=>s+v,0)/a.filter(v=>v!=null).length,dm=ds.length?ds.reduce((s,v)=>s+v,0)/ds.length:0,w=ds.filter(v=>v>0).length,mn=ds.length?Math.min(...ds):0;sr+=`<tr><td>${esc(n)}</td><td>${mean.toFixed(6)}</td><td class="${dm>=0?'pos':'neg'}">${n===base?'—':dm.toFixed(3)}</td><td>${n===base?'—':w+'/'+ds.length}</td><td>${n===base?'—':mn.toFixed(3)}</td></tr>`});sr+='</tbody></table>';document.getElementById('summary-'+task).innerHTML=sr;
 let tb='<table><thead><tr><th>test day</th>'+names.map(n=>`<th>${esc(n)}<br>AUC / Δ‰</th>`).join('')+'</tr></thead><tbody>';dates.forEach((d,i)=>{tb+=`<tr><td>${d}</td>`+names.map(n=>{let v=series[n][i];if(v==null)return '<td>missing</td>';let delta=(v-series[base][i])*1000;return `<td>${v.toFixed(6)}${n===base?'':`<br><span class="${delta>=0?'pos':'neg'}">${delta>=0?'+':''}${delta.toFixed(3)}‰</span>`}</td>`}).join('')+'</tr>'});tb+='</tbody></table>';document.getElementById('table-'+task).innerHTML=tb;
}
['buy','cat','click','ext'].forEach(draw);
</script></body></html>"""
    return (template.replace("__ALIGNMENT__", alignment)
            .replace("__SOURCES__", source_items)
            .replace("__CARDS__", cards)
            .replace("__PAYLOAD__", safe_payload)
            .replace("__COLORS__", json.dumps(COLORS)))


def main():
    parser = argparse.ArgumentParser(description="Compare rolling AUC curves and write a self-contained HTML report")
    parser.add_argument("baseline", help="baseline experiment name or absolute path")
    parser.add_argument("experiments", nargs="+", help="one or more experiment names or absolute paths")
    parser.add_argument("-o", "--output", default="rolling_auc_compare.html")
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR)
    args = parser.parse_args()

    data, sources = {}, []
    for exp in [args.baseline] + args.experiments:
        name, path, rows = load_metrics(exp, args.base_dir)
        if name in data:
            parser.error("duplicate experiment name: %s" % name)
        data[name], sources = rows, sources + [path]
        print("[INFO] %s: %d rolling metric rows <- %s" % (name, len(rows), path))
    errors = validate_alignment(data)
    for error in errors:
        print("[WARN] " + error, file=sys.stderr)
    report = render_html(build_payload(data), sources, errors)
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(report)
    print("[INFO] wrote self-contained HTML: %s" % os.path.abspath(args.output))
    return 2 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
