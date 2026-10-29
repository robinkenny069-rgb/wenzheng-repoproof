"""Local, read-only review. Neither repository code nor document commands run."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import html
import json
import re
from pathlib import Path

from .core import analyze


def terms(text: str) -> set[str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return set(re.findall(r"[a-z]+|[0-9]+", text.lower().replace("_", " ")))


def lexical_rank(query: str, documents: list[str]) -> list[tuple[int, float]]:
    """Explicit simple baseline. This is token overlap, not an AI model."""
    q = terms(query)
    scores = []
    for i, document in enumerate(documents):
        d = terms(document)
        score = len(q & d) / max(1, len(q | d))
        scores.append((i, score))
    return sorted(scores, key=lambda row: (-row[1], row[0]))


def build_review(before: str, after: str, markdown: str, *, retrieval: str = "lexical",
                 source_path: str = "api.py", document_path: str = "guide.md",
                 top_k: int = 3, duration_evidence: bool = False) -> dict:
    result = analyze(before, after, markdown, source_path=source_path,
                     document_path=document_path)
    payload = result.to_dict()
    payload["generated_at"] = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S 北京时间")
    payload["report_label"] = "文档核验报告"
    payload["prototype_notice"] = (
        "研发原型。输入数据不构成真实历史评测；候选需要人工核验。"
        "语义相似度不代表文档已出错。未执行待检查代码。"
    )
    payload["retrieval_backend"] = retrieval
    payload["source_file"] = source_path
    payload["document_file"] = document_path
    payload["ranked_candidates"] = []
    blocks = list(result.blocks)
    texts = [" > ".join(block.heading_path) + "\n" + block.text for block in blocks]
    ranker = lexical_rank
    if retrieval == "semantic":
        from .semantic import SemanticRetriever
        ranker = SemanticRetriever().rank
    elif retrieval != "lexical":
        raise ValueError(f"Unknown retrieval backend: {retrieval}")
    for change in result.changes:
        query = (f"{change.symbol} parameter {change.parameter.replace('_', ' ')} "
                 f"default value changed from {change.old.expression} "
                 f"to {change.new.expression}.")
        ranking = ranker(query, texts) if texts else []
        for index, score in ranking[:top_k]:
            block = blocks[index]
            payload["ranked_candidates"].append({
                "symbol": change.symbol, "parameter": change.parameter,
                "block_id": block.block_id, "score": float(score),
                "start_line": block.start_line, "end_line": block.end_line,
                "text": block.text, "heading_path": list(block.heading_path),
                "status": "retrieved_for_review",
                "query": query,
            })
    if duration_evidence:
        from .duration_evidence import DurationEvidenceChecker
        checker = DurationEvidenceChecker(before, after) if result.changes else None
        retrieved = {(row["symbol"], row["parameter"], row["block_id"]) for row in payload["ranked_candidates"]}
        rows = []
        for change in result.changes:
            for block in blocks:
                row = checker.check(change, block)
                row["in_retrieval_top_k"] = (change.symbol, change.parameter, block.block_id) in retrieved
                rows.append(row)
        payload["duration_evidence"] = rows
        payload["duration_evidence_scope"] = "Rules inspect every change/block pair. The top-k marker is separate; full-scan gains are not attributed to AI."
    return payload


def render_report(payload: dict) -> str:
    esc = lambda value: html.escape(str(value), quote=True)
    changes = payload.get("changes", [])
    candidates = payload.get("candidates", [])
    candidate_keys = {(row["change"]["symbol"], row["change"]["parameter"], row["block_id"]) for row in candidates}
    candidate_keys.update((row["symbol"], row["parameter"], row["block_id"])
                         for row in payload.get("duration_evidence", []) if row["status"] == "needs_update_candidate")
    blocks = {b["block_id"]: b for b in payload.get("blocks", [])}
    change_html = []
    for item in changes:
        old = item["old"].get("display", item["old"].get("expression", "?"))
        new = item["new"].get("display", item["new"].get("expression", "?"))
        change_html.append(
            f'<tr><td><code>{esc(item["symbol"])} · {esc(item["parameter"])}</code></td>'
            f'<td><code>{esc(old)}</code></td><td><code>{esc(new)}</code></td>'
            f'<td>旧 L{item["before_line"]} / 新 L{item["after_line"]}</td></tr>')
    evidence_html = []
    evidence_labels = {
        "symbol_context": "函数定位依据",
        "parameter_default_old_value": "文档中的旧默认值",
    }
    for candidate in candidates:
        change = candidate["change"]
        block = blocks.get(candidate["block_id"], {})
        evidence = "".join(
            f'<li><strong>{esc(evidence_labels.get(e["kind"], e["kind"]))}</strong> '
            f'L{e["start_line"]}–{e["end_line"]}: {esc(e["text"])}</li>'
            for e in candidate.get("evidence", []))
        evidence_html.append(
            f'<article><p class="tag">有静态线索 · 需人工确认</p>'
            f'<h3>{esc(change["symbol"])} / {esc(change["parameter"])}</h3>'
            f'<blockquote>{esc(block.get("text", ""))}</blockquote><ul>{evidence}</ul></article>')
    retrieval_html = []
    for row in payload.get("ranked_candidates", []):
        retrieval_html.append(
            f'<article><p class="tag">检索候选 · 不代表错误</p>'
            f'<h3>{esc(row["symbol"])} / {esc(row["parameter"])}</h3>'
            f'<p class="meta">L{row["start_line"]}–{row["end_line"]} · '
            f'相关度 {row["score"]:.4f} · {esc(" / ".join(row["heading_path"]))}</p>'
            f'<blockquote>{esc(row["text"])}</blockquote></article>')
    reason_labels = {
        "document_default_not_old_value": "文档值不等于旧默认值，未列为旧值候选；这也可能表示文档已经更新。",
        "explicit_default_claim_missing": "没有找到当前规则支持的明确默认值说明，需要结合上下文阅读。",
        "qualified_or_negated_claim": "句子含条件、否定或可能性表达，不能直接当作默认行为。",
        "historical_context": "这是历史或迁移说明，可能有意保留旧值。",
        "symbol_identity_not_established": "不能唯一确定这段话描述的是哪个函数。",
        "ambiguous_symbol_identity": "存在同名函数，不能确定对应关系。",
        "multiple_symbol_context": "上下文提到多个函数，需要人工确认描述对象。",
        "non_prose_block": "代码示例、引用或注释需要单独理解。",
        "document_literal_unknown": "文档值的含义或类型不够明确。",
        "conflicting_document_defaults": "同一参数出现不同默认值说明，需要人工核对。",
        "non_literal_default": "代码默认值是表达式，当前不会执行代码来求值。",
        "no_lexical_evidence": "未找到明确名称关联，仍可在相关段落中人工查找。",
        "source_parse_error": "源码解析失败，未进行默认值比较。",
        "duplicate_symbol_definition": "函数重复定义，无法唯一比较。",
        "unmatched_symbol": "函数新增或删除，超出本版默认值比较范围。",
        "parameter_added_or_removed": "参数新增或删除，超出本版默认值比较范围。",
        "default_added_or_removed": "参数从必填改为可选或相反，超出本版比较范围。",
    }
    abstention_html = []
    for row in payload.get("abstentions", []):
        block = blocks.get(row.get("block_id"), {})
        location = f' · 文档 L{block["start_line"]}–{block["end_line"]}' if block else ""
        label = reason_labels.get(row.get("reason"), row.get("detail") or "需要人工核对。")
        abstention_html.append(
            f'<li><code>{esc(row.get("symbol", ""))} / {esc(row.get("parameter", ""))}</code>'
            f'{location}：{esc(label)}</li>')
    abstentions = "".join(abstention_html)
    duration_html = ""
    if "duration_evidence" in payload:
        statuses = {"needs_update_candidate": "有旧值证据 · 待复核", "consistent_with_new_default": "与新声明默认值一致",
                    "excluded_from_change": "不能据这次变更改写", "manual_review": "证据不足 · 人工核对"}
        reasons = {
            "matches_old_declared_default": "文档中的时长等于旧默认值，与新默认值不同。单位来自源码参数说明。",
            "matches_new_declared_default": "文档时长与源码声明的新默认值一致。",
            "different_documented_symbol": "最近的函数标题指向另一个对象，不能套用本次函数的变化。",
            "different_function_in_text": "文字明确提到另一个函数。",
            "function_not_established": "无法确认这段话说的是本次变化的函数。",
            "historical_context": "存在历史或迁移语境，旧值可能是有意保留的。",
            "qualified_or_override_claim": "包含主动选择、条件、否定或近似表达，不能当作精确默认行为。",
            "default_behavior_not_established": "没有建立这句话与当前默认行为的关系。",
            "waiting_behavior_not_established": "没有建立这句话与等待上限的关系。",
            "source_unit_not_established": "源码参数说明未明确绑定时间单位，程序不猜测单位。",
            "ambiguous_wait_parameter": "同一个函数有多个等待参数，无法确定段落描述哪一个。",
            "different_or_multiple_wait_parameters": "句子涉及其他等待参数或同时涉及多个参数。",
            "multiple_sentences_require_review": "多句文字中的默认行为和时长关系需要人工核对。",
            "ambiguous_or_unsupported_document_duration": "没有唯一且受支持的明确时长表达。",
            "document_value_matches_neither_version": "文档值与旧、新默认值均不一致，不能直接归因于这次变更。",
            "ambiguous_function": "同名或多个函数造成归属歧义。",
            "source_function_ambiguous": "无法唯一确定源码函数。",
            "non_numeric_code_default": "本模块只处理数值型时间默认值。",
            "unsupported_code_duration": "源码时长不是受支持的非负有限值。",
            "non_prose_block": "代码示例、引用和注释需要单独理解。",
        }
        details = []
        for row in payload["duration_evidence"]:
            block = blocks[row["block_id"]]
            trail = []
            for item in row["evidence"]:
                if item["kind"] == "code_defaults_in_seconds":
                    text = f'源码声明：{item["old"]} 秒 → {item["new"]} 秒；旧 L{item["old_line"]} / 新 L{item["new_line"]}'
                elif item["kind"] == "document_duration":
                    text = f'文档 L{item["line"]}：{item["text"]}，换算为 {item["seconds"]} 秒'
                else:
                    label = {"documented_symbol": "文档函数标题/名称", "old_source_unit": "旧源码单位依据", "new_source_unit": "新源码单位依据"}[item["kind"]]
                    text = f'{label} L{item["line"]}：{item["text"]}'
                trail.append(f'<li>{esc(text)}</li>')
            details.append(f'<article><p class="tag">{esc(statuses[row["status"]])}</p>'
                           f'<h3>{esc(row["symbol"])} / {esc(row["parameter"])} · 文档 L{block["start_line"]}</h3>'
                           f'<p class="meta">进入当前检索前列：{"是" if row["in_retrieval_top_k"] else "否"}</p>'
                           f'<blockquote>{esc(block["text"])}</blockquote><p>{esc(reasons.get(row["reason"], row["reason"]))}</p>'
                           f'<ul>{"".join(trail)}</ul></article>')
        count = sum(row["status"] == "needs_update_candidate" for row in payload["duration_evidence"])
        duration_html = (f'<h2>05 时间说明的证据核对</h2><p>规则扫描全部{len(payload["duration_evidence"])}个变更与段落组合，找到{count}条旧值候选。'
                         '这项全量规则检查与前面的AI检索分别统计；不能将扩大检查范围带来的发现算作AI排名改善。'
                         '仅支持部分英文时长句式和明确单位的源码说明，候选仍需人工复核。</p>' + ''.join(details))
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(payload.get('report_label', '文档核验报告'))} · 文证</title><style>
:root{{font-family:"Microsoft YaHei",system-ui,sans-serif;color:#172638;background:#f4f6f8}}
body{{margin:0}}main{{max-width:1080px;margin:auto;padding:52px 28px 80px}}h1{{font-size:36px;letter-spacing:1px;margin:10px 0}}h2{{font-size:22px;margin:34px 0 18px}}h3{{font-size:17px;margin:10px 0}}p,li{{line-height:1.8}}.eyebrow{{color:#23675b;font-size:13px;font-weight:700;letter-spacing:3px}}.lead{{max-width:800px;color:#526070}}.notice{{padding:15px 18px;background:#fff3d7;border-radius:6px}}.stats{{display:flex;gap:40px;margin:30px 0}}.stats strong{{font-size:32px;display:block}}.meta,.tag{{font-size:13px;color:#526070}}.tag{{color:#23675b;margin:0}}table{{width:100%;border-collapse:collapse;background:white}}td,th{{padding:16px;text-align:left;border-bottom:1px solid #dce3e8}}th{{font-size:13px;color:#526070}}article{{background:white;padding:22px 26px;border:1px solid #dce3e8;border-radius:8px;margin:14px 0}}blockquote{{white-space:pre-wrap;margin:15px 0;padding:12px 18px;background:#f4f6f8;border-left:3px solid #90b4aa;line-height:1.8}}code{{font-family:Consolas,monospace;overflow-wrap:anywhere}}li{{overflow-wrap:anywhere}}footer{{border-top:1px solid #dce3e8;margin-top:36px;padding-top:16px;color:#526070;font-size:13px}}@media(max-width:600px){{main{{padding:25px 16px}}.stats{{gap:18px}}table{{font-size:12px}}td,th{{padding:8px}}}}
</style></head><body><main>
<div class="eyebrow">{esc(payload.get('report_label', '文档核验报告'))}</div><h1>文证 开源文档变更核验</h1>
<p class="lead">把代码变更、文档原文和待核验事项放在一起，帮助维护者决定哪些段落需要更新。</p>
<p class="notice">{esc(payload['prototype_notice'])}</p>
<p class="meta">生成时间：{esc(payload.get('generated_at', '旧报告未记录'))}<br>源码 {esc(payload['source_file'])} · 文档 {esc(payload['document_file'])} · 检索 {esc(payload['retrieval_backend'])}</p>
<div class="stats"><div><strong>{len(changes)}</strong>代码默认值变更</div><div><strong>{len(candidate_keys)}</strong>疑似未同步说明</div><div><strong>{len(payload.get('abstentions', []))}</strong>其他检查记录</div></div>
<p>第一项统计旧版和新版代码之间的变化，修改文档不会减少它。第二项统计有旧值证据、需要复核的文档说明，合并显式参数规则和已启用的时间核对结果，按函数、参数、段落去重。第三项记录显式参数规则未确定的检查情况；时间核对可能进一步找到证据，同一段落也可能产生多条记录，不能把它当作错误数。</p>
<details><summary>不熟悉这些词？查看一个具体例子</summary><p>默认值是调用者没有特别指定时，函数采用的设置。例如参数默认从30改成10后，说明书如果还写30，就值得检查。调用者主动指定30，或文档写“每页显示30行”，则不能仅凭数字相同判错。L4表示文件第4行。相关度只帮助排序，不是准确率或出错概率。</p></details>
<h2>01 代码事实</h2><table><thead><tr><th>符号与参数</th><th>原默认值</th><th>新默认值</th><th>代码位置</th></tr></thead><tbody>{''.join(change_html) or '<tr><td colspan="4">未检测到受支持的默认值变更。</td></tr>'}</tbody></table>
<h2>02 可回查的静态线索</h2>{''.join(evidence_html) or '<p>没有符合当前严格规则的候选。</p>'}
<h2>03 相关段落供人工复核</h2><p class="meta">相似度只用于排序，不能作为错误概率。词法模式为基线，semantic 模式才调用开源神经模型。</p>{''.join(retrieval_html) or '<p>没有可供检索的段落。</p>'}
<h2>04 支持边界与未确定事项</h2><ul>{abstentions or '<li>未生成额外拒判记录；这不等于全文档正确。</li>'}</ul>
{duration_html}
<footer>当前只处理部分 Python 字面默认值与 Markdown 文档。不能验证动态行为、历史版本说明、复杂示例执行或任意自然语言事实。此报告由真实程序输出生成，尚不是正式竞赛评测。</footer>
</main></body></html>'''


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only documentation review prototype")
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--docs", type=Path, required=True)
    parser.add_argument("--retrieval", choices=["lexical", "semantic"], default="lexical")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--duration-evidence", action="store_true", help="Run bounded duration evidence rules over all blocks; no automatic edits")
    parser.add_argument("--output", type=Path, default=Path("output/review"))
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    payload = build_review(args.before.read_text(encoding="utf-8-sig"),
                           args.after.read_text(encoding="utf-8-sig"),
                           args.docs.read_text(encoding="utf-8-sig"),
                           retrieval=args.retrieval, source_path=args.after.name,
                           document_path=args.docs.name, top_k=args.top_k, duration_evidence=args.duration_evidence)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "review.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "review.html").write_text(render_report(payload), encoding="utf-8")
    print(json.dumps({"retrieval": args.retrieval, "changes": len(payload['changes']),
                      "static_candidates": len(payload['candidates']),
                      "report": str(args.output / "review.html")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
