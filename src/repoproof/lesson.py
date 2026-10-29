"""Reproducible development lesson, not an independent accuracy benchmark.

All methods see the same English texts and query. Reference labels and Chinese
translations are attached only after rankings have been computed.
"""
from __future__ import annotations

import hashlib
import html
import json
import platform
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter

from .cli import lexical_rank
from .core import analyze
from .semantic import SemanticRetriever

METHODS = {"lexical": "按共同词语查找", "semantic": "AI 按意思查找", "hybrid": "结合两种排序"}
DECISIONS = {"needs_update": "需要修改", "keep": "此次不用改", "unsure": "证据不足"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reciprocal_rank_fusion(rankings: list[list[tuple[int, float]]], *, k: int = 60) -> list[tuple[int, float]]:
    """Standard RRF with equal weights, no fitted parameters or labels."""
    if k < 1:
        raise ValueError("RRF constant must be positive")
    scores: dict[int, float] = {}
    for ranking in rankings:
        if len({index for index, _ in ranking}) != len(ranking):
            raise ValueError("Each ranking must contain unique document indices")
        for position, (index, _) in enumerate(ranking, 1):
            scores[index] = scores.get(index, 0.0) + 1.0 / (k + position)
    return sorted(scores.items(), key=lambda row: (-row[1], row[0]))


def build_lesson(fixture: Path) -> dict:
    started = perf_counter()
    before, after, document = (fixture / name for name in ("before.py", "after.py", "guide.md"))
    result = analyze(before.read_text(encoding="utf-8-sig"), after.read_text(encoding="utf-8-sig"),
                     document.read_text(encoding="utf-8-sig"), source_path="after.py", document_path="guide.md")
    if len(result.changes) != 1:
        raise ValueError("This lesson expects exactly one code change")
    change = result.changes[0]
    blocks = list(result.blocks)
    texts = [" > ".join(block.heading_path) + "\n" + block.text for block in blocks]
    # Identical query construction to the main CLI; no expected answer is injected.
    query = (f"{change.symbol} parameter {change.parameter.replace('_', ' ')} "
             f"default value changed from {change.old.expression} to {change.new.expression}.")
    rank_start = perf_counter()
    lexical = lexical_rank(query, texts)
    lexical_seconds = perf_counter() - rank_start
    model_start = perf_counter()
    encoder = SemanticRetriever()
    load_seconds = perf_counter() - model_start
    model_start = perf_counter()
    semantic = encoder.rank(query, texts)
    inference_seconds = perf_counter() - model_start
    rankings = {"lexical": lexical, "semantic": semantic,
                "hybrid": reciprocal_rank_fusion([lexical, semantic])}
    # These are authored teaching references, deliberately not a model input.
    labels = json.loads((fixture / "labels.json").read_text(encoding="utf-8"))
    cases = {case["block_id"]: case for case in labels["cases"]}
    if set(cases) != {block.block_id for block in blocks}:
        raise ValueError("Reference labels no longer match document blocks")
    target = labels["target"]
    if (target["symbol"], target["parameter"], target["old"], target["new"]) != (
            change.symbol, change.parameter, change.old.value, change.new.value):
        raise ValueError("Code changes no longer match the lesson reference")
    expected = {key for key, case in cases.items() if case["reference_decision"] == "needs_update"}
    ranked = {}
    summaries = {}
    for name, rows in rankings.items():
        ranked[name] = [{"rank": position, "block_id": blocks[index].block_id, "score": score}
                        for position, (index, score) in enumerate(rows, 1)]
        selected = {row["block_id"] for row in ranked[name][:3]}
        summaries[name] = {"k": 3, "reference_update_count": len(expected),
                           "found_update_count": len(selected & expected),
                           "missed_update_ids": sorted(expected - selected)}
    root = Path(__file__).resolve().parents[2]
    code_paths = [Path(__file__), root / "src/repoproof/semantic.py", root / "src/repoproof/core.py",
                  root / "src/repoproof/cli.py", root / "scripts/run_semantic_lesson.py"]
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "status": "actual_local_neural_run",
        "data_scope": labels["origin"],
        "query": query, "top_k": 3, "rrf_k": 60,
        "changes": [change.to_dict()], "static_candidates": len(result.candidates),
        "blocks": [block.to_dict() for block in blocks], "rankings": ranked, "summaries": summaries,
        "reference_cases": labels["cases"], "reference_target": target,
        "model": {"model_id": encoder.manifest["model_id"], "revision": encoder.manifest["revision"],
                  "onnx_sha256": encoder.manifest["files"]["onnx/model.onnx"]["sha256"],
                  "providers": encoder.session.get_providers(), "language": "English", "threads": 4,
                  "pooling": encoder.manifest["pooling"]},
        "environment": {"python": sys.version, "platform": platform.platform()},
        "seconds": {"lexical": lexical_seconds, "model_load_and_verification": load_seconds,
                    "semantic_inference": inference_seconds, "total": perf_counter() - started},
        "input_sha256": {name: sha256(fixture / name) for name in ("before.py", "after.py", "guide.md", "labels.json")},
        "code_sha256": {path.relative_to(root).as_posix(): sha256(path) for path in code_paths},
        "human_review": "pending; this run does not establish that a participant reviewed anything",
    }


def render_lesson(payload: dict) -> str:
    esc = lambda value: html.escape(str(value), quote=True)
    blocks = {block["block_id"]: block for block in payload["blocks"]}
    reference = {case["block_id"]: case for case in payload["reference_cases"]}
    change = payload["changes"][0]
    cards = []
    for i, block in enumerate(payload["blocks"], 1):
        case = reference[block["block_id"]]
        options = '<option value="">请选择你的判断</option>' + ''.join(
            f'<option value="{key}">{value}</option>' for key, value in DECISIONS.items())
        cards.append(f'''<article class="case" data-id="{esc(block['block_id'])}">
<div class="line">说明 {i} · 文档 L{block['start_line']} · {esc(' / '.join(block['heading_path']))}</div>
<p class="english" lang="en">{esc(block['text'])}</p><p>{esc(case['translation'])}</p>
<label>我的判断 <select aria-label="说明{i}的判断">{options}</select></label>
<label class="why">理由（用自己的话）<input type="text" aria-label="说明{i}的理由" placeholder="写一句理由；不确定时说明还缺什么信息" maxlength="500"></label>
<details class="reference"><summary>查看教学参考（预先编写，待复核）</summary><p><strong>{esc(DECISIONS[case['reference_decision']])}</strong>：{esc(case['reason'])}</p></details></article>''')
    panels = []
    for method, title in METHODS.items():
        summary = payload["summaries"][method]
        rows = []
        for item in payload["rankings"][method][:3]:
            block = blocks[item["block_id"]]
            rows.append(f'<li><strong>第{item["rank"]}位 · 说明{list(blocks).index(item["block_id"])+1}</strong>'
                        f'<p lang="en">{esc(block["text"])}</p><p>{esc(reference[item["block_id"]]["translation"])}</p></li>')
        panels.append(f'<article><h3>{title}</h3><p>前3条中找回 <strong>{summary["found_update_count"]} / '
                      f'{summary["reference_update_count"]}</strong> 条待更新说明</p><ol class="ranked">{"".join(rows)}</ol></article>')
    metadata = {"generated_at": payload["generated_at"], "input_sha256": payload["input_sha256"],
                "code_sha256": payload["code_sha256"], "model": payload["model"],
                "schema_version": 1, "source": "semantic_lesson"}
    safe_json = json.dumps(metadata, ensure_ascii=True).replace("<", "\\u003c")
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>文证 · 换一种说法，能找到吗？</title><style>
:root{{font-family:"Microsoft YaHei",system-ui,sans-serif;color:#182f3b;background:#f3f6f7}}*{{box-sizing:border-box}}body{{margin:0}}main{{max-width:1120px;padding:44px 28px 80px;margin:auto}}.eyebrow{{color:#257665;letter-spacing:2px;font-size:13px;font-weight:bold}}h1{{font-size:36px;margin:12px 0}}h2{{font-size:24px;margin-top:38px}}h3{{font-size:18px}}p,li{{line-height:1.8}}.muted,.line{{color:#5b6b75;font-size:14px}}.notice{{background:#fff1cf;border-left:4px solid #d7ad4a;padding:14px 20px}}.fact{{background:#173f42;color:white;padding:24px;border-radius:10px}}.fact strong{{font-size:25px}}code{{font-family:Consolas,monospace;overflow-wrap:anywhere}}article{{background:#fff;border:1px solid #dbe4e7;border-radius:10px;padding:22px;margin:16px 0}}.english{{background:#f3f6f7;border-left:3px solid #72a697;padding:12px 16px;overflow-wrap:anywhere}}.why{{display:block;margin-top:12px}}select,input,button{{font:inherit;padding:10px;border:1px solid #a7b8bf;border-radius:5px;max-width:100%}}select{{margin-left:8px;background:white}}input{{display:block;width:100%;margin-top:7px}}details{{margin-top:16px}}summary{{cursor:pointer;color:#20685c;line-height:1.7}}button{{background:#21695d;color:white;cursor:pointer;border:0;padding:12px 20px}}button:hover{{background:#164d43}}.compare{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}}.compare article{{margin:0;padding:18px}}.ranked{{padding-left:0;list-style:none}}.ranked li{{border-top:1px solid #e1e9eb;padding:14px 0}}.ranked p{{font-size:14px;overflow-wrap:anywhere}}.actions{{background:#e4efeb;padding:20px;border-radius:10px}}#status{{min-height:24px}}.trace{{white-space:pre-wrap;word-break:break-all;font-size:12px}}footer{{border-top:1px solid #d4dfe2;padding-top:16px;margin-top:30px}}@media(max-width:760px){{.compare{{grid-template-columns:1fr}}main{{padding:26px 16px}}h1{{font-size:28px}}article{{padding:18px}}.fact strong{{font-size:21px}}}}
</style></head><body><main><div class="eyebrow">文证 / 第二次团队练习</div><h1>换一种说法，能找到吗？</h1>
<p>这一次，AI 负责寻找可能相关的说明。你们根据代码和句意，判断是否需要更新。</p>
<p class="notice">模型已在本机实际运行。以下8段是教学用合成数据，参考判断由AI辅助预先编写，尚待团队复核；不是检索模型的判断，也不是正式准确率。</p>
<div class="fact"><p>本次唯一的代码变化 · <code>fetch_records(timeout=…)</code></p>
<strong>默认等待 30 秒 → 10 秒</strong><p>单位“秒”来自函数自身的说明。调用时没有特别指定，就采用新的10秒；主动指定其他值仍然可以。</p>
<p>旧代码 L{change['before_line']} / 新代码 L{change['after_line']}。本练习只读取代码，不执行网络请求。</p></div>
<h2>1. 先用自己的话判断</h2><p>两人分别完成。每条选择“需要修改”“此次不用改”或“证据不足”，再写一句理由。优先独立判断；看过参考也请如实记录。</p>
{''.join(cards)}
<div class="actions"><label>我是 <select id="participant" aria-label="成员代号"><option value="">选择成员代号</option><option value="A">成员A</option><option value="B">成员B</option></select></label>
<p><label><input id="prior-reference" type="checkbox" style="display:inline;width:auto"> 我此前已看过参考答案，或与队友讨论过答案</label></p>
<button id="export" type="button">保存我的判断记录</button><p id="status" role="status" aria-live="polite"></p><p class="muted">只下载到你的电脑，不会联网提交。文件名为“文证第二次练习_成员A/B.json”。先保存，再与队友讨论分歧。</p></div>
<h2>2. 再看三种方法实际找到了什么</h2><details id="model-results"><summary>展开真实运行结果</summary><p>每种方法只推荐前3条。参照预先写好的教学标签，8条中有2条待更新。这里的“找回几条”只解释这个小样例；相关说明可能已经正确，也可能是历史说明。</p>
<div class="compare">{''.join(panels)}</div><p>先找相关段落，再核对描述对象、默认或主动设置、时间范围、单位和数值。推荐靠前不等于判定出错；模型没有自动改写文档。</p></details>
<details><summary>运行来源与复现信息</summary><p>运行时间：{esc(payload['generated_at'])}<br>模型：<a href="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2">{esc(payload['model']['model_id'])}</a>，CPU本地推理；只将英文原文和标题输入模型，中文为教学翻译。模型未在本练习上训练。</p>
<p>静态规则候选数：{payload['static_candidates']}。AI仍可能找到换说法的句子；这不自动增加“已确认错误”数。</p>
<p>相同查询：<code>{esc(payload['query'])}</code>。融合使用标准RRF常数60，不针对答案调参。完整8条排序、分数、版本与哈希保存在同目录的comparison.json中。</p>
<p>加载及校验 {payload['seconds']['model_load_and_verification']:.3f} 秒；语义推理 {payload['seconds']['semantic_inference']:.3f} 秒。这只是本机单次运行，不能作为性能基准。</p></details>
<footer class="muted">当前目标是理解与复核。完成本练习不等于完成独立评测、创新性证明或最终参赛作品。</footer></main>
<script type="application/json" id="run-metadata">{safe_json}</script><script>
const metadata=JSON.parse(document.getElementById('run-metadata').textContent);
let viewedReference=false, viewedResults=false;
document.querySelectorAll('.reference').forEach(el=>el.addEventListener('toggle',()=>{{if(el.open)viewedReference=true;}}));
document.getElementById('model-results').addEventListener('toggle',e=>{{if(e.target.open)viewedResults=true;}});
document.getElementById('export').addEventListener('click',()=>{{
 const participant=document.getElementById('participant').value;
 const status=document.getElementById('status');
 const judgments=Array.from(document.querySelectorAll('.case')).map(el=>({{block_id:el.dataset.id,decision:el.querySelector('select').value,reason:el.querySelector('input').value.trim()}}));
 if(!participant||judgments.some(row=>!row.decision||!row.reason)){{status.textContent='请先选择成员代号，并为8条说明填写判断和一句理由。';return;}}
 const record={{...metadata,participant,exported_at:new Date().toISOString(),judgments,
 viewed_reference_in_this_page:viewedReference,viewed_model_results_in_this_page:viewedResults,
 prior_reference_or_discussion:document.getElementById('prior-reference').checked,
 note:'Self-reported exercise record; not a formal blinded annotation or identity verification.'}};
 const blob=new Blob([JSON.stringify(record,null,2)],{{type:'application/json;charset=utf-8'}});
 const url=URL.createObjectURL(blob),a=document.createElement('a');
 a.href=url;a.download='文证第二次练习_成员'+participant+'.json';a.click();
 setTimeout(()=>URL.revokeObjectURL(url),1000);
 status.textContent='已发起下载。请在浏览器下载列表中确认文件已保存，再和队友比较理由。';
}});
</script></body></html>'''
