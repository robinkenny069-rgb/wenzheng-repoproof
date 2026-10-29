"""Run a fully specified, hash-checked source case without importing its code."""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import html
import json
from pathlib import Path
from time import perf_counter
from urllib.parse import urlparse

from .contract_graph import ContractGraph, Source
from .contract_docs import review_class

STATUS = {'default_mismatch_candidate': '默认值不一致候选', 'consistent_with_declared_default': '与声明默认值一致', 'abstain': '证据不足，保留核对'}
KINDS = {'import_binding': '读取明确的导入关系', 'export_binding': '核对模块导出', 'inherits': '追踪父类',
         'kwargs_alias': '参数字典传递', 'other_parameter_pop': '仅移除其他参数', 'other_parameter_write': '仅设置其他参数',
         'factory_target': '找到实际委托的类', 'kwargs_forward': '将参数转发给底层类', 'super_forward': '将参数交给父类',
         'declared_default': '找到参数声明的默认值'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def within(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Source path escapes the declared case directory')
    return path


def build_contract_review(case_path: Path) -> dict:
    started = perf_counter()
    config = json.loads(case_path.read_text(encoding='utf-8'))
    link = urlparse(config['upstream_url'])
    if link.scheme not in ('http', 'https') or not link.netloc:
        raise ValueError('Upstream link must be an HTTP(S) source URL')
    case_root = case_path.resolve().parent
    manifest_path = within(case_root, config['source_manifest'])
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    for relative, metadata in manifest['files'].items():
        if digest(within(case_root, relative)) != metadata['sha256']:
            raise ValueError(f'Source hash mismatch: {relative}')
    runs = {}
    for snapshot, mapping in config['snapshots'].items():
        sources = []
        for module, relative in mapping.items():
            if relative not in manifest['files']:
                raise ValueError(f'Module has no locked source provenance: {relative}')
            sources.append(Source(module, relative, within(case_root, relative).read_text(encoding='utf-8'), module in config['package_modules']))
        graph = ContractGraph(sources)
        rows = [row for entry in config['entry_classes'] for row in review_class(graph, entry)]
        runs[snapshot] = {'rows': rows, 'counts': dict(Counter(row['status'] for row in rows)), 'parameter_fields': len(rows)}
    project = Path(__file__).resolve().parents[2]
    files = ['src/repoproof/contract_graph.py', 'src/repoproof/contract_docs.py', 'src/repoproof/contract_review.py', 'src/repoproof/duration_evidence.py', 'scripts/run_contract_case.py']
    return {'schema_version': 1, 'generated_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
            'case_id': config['case_id'], 'title': config['title'], 'upstream_url': config['upstream_url'],
            'source_kind': config['source_kind'], 'entry_classes': config['entry_classes'], 'runs': runs,
            'elapsed_seconds': perf_counter()-started, 'code_executed': False, 'neural_model_called': False,
            'case_config_sha256': digest(case_path), 'source_manifest_sha256': digest(manifest_path),
            'source_sha256': {relative: metadata['sha256'] for relative, metadata in manifest['files'].items()},
            'code_sha256': {name: digest(project/name) for name in files},
            'assumptions': ['Explicit, manually specified module snapshot; not an automatically discovered installed environment.',
                            'No runtime monkeypatching, import side effects, or unsupported dynamic dispatch.',
                            'Declared defaults when the caller omits the parameter; not arbitrary runtime behavior.'],
            'evaluation_scope': 'One known upstream event, two related classes, before/after repair; development verification, not independent accuracy.',
            'team_review': 'pending for this new code and case'}


def render_contract_review(result: dict) -> str:
    esc = lambda value: html.escape(str(value), quote=True)
    runs = result['runs']
    before = runs['before']
    after = runs['after']
    after_by_key = {(row['class'], row['parameter']): row for row in after['rows']}
    highlights = []
    for row in before['rows']:
        if row['status'] != 'default_mismatch_candidate':
            continue
        corrected = after_by_key.get((row['class'], row['parameter']), {})
        trace_html = []
        for step in row['contract']['trace']:
            snippet = step.get('source_line', step['text']) if step['kind'] == 'declared_default' else step['text']
            trace_html.append(f'<li><strong>{esc(KINDS.get(step["kind"], step["kind"]))}</strong><p class="meta">{esc(step["path"])} · L{step["line"]}</p><pre>{esc(snippet)}</pre></li>')
        highlights.append(f'''<article><div class="tag">有文档和源码证据 · 仍需复核适用范围</div>
<h3>{esc(row['class'])} / {esc(row['parameter'])}</h3>
<div class="values"><div>修复前文档<strong>{esc(row['claim']['value'])} {esc(row['claim']['unit'] or '')}</strong></div>
<div>追踪到的代码声明<strong>{esc(row['contract']['value'])}</strong></div>
<div>修复后文档<strong>{esc(corrected.get('claim',{}).get('value','未确定'))} {esc(corrected.get('claim',{}).get('unit') or '')}</strong></div></div>
<p class="meta">旧文档：{esc(row['document']['path'])} · L{row['document']['start_line']}–{row['document']['end_line']}</p>
<blockquote>{esc(row['document']['text'])}</blockquote>
<p>修复后的检查结果：{esc(STATUS.get(corrected.get('status'),'未确定'))}。秒的单位依据来自底层类对应参数的说明。</p>
<details><summary>展开完整来源链：程序如何找到这个默认值</summary><ol>{''.join(trace_html)}</ol></details></article>''')
    tables = []
    for entry in result['entry_classes']:
        rows = []
        for row in before['rows']:
            if row['class'] != entry:
                continue
            corrected = after_by_key.get((entry, row['parameter']), {})
            reason_labels = {'no_unique_explicit_default_claim': '未找到当前支持的明确默认值句式',
                             'unsupported_unit_or_claim_continuation': '单位或后续限定暂不支持',
                             'sentinel_default_requires_behavior_review': 'None可能触发内部默认设置，需要继续追踪'}
            reason = '' if row['status'] != 'abstain' else f'<span class="reason">{esc(reason_labels.get(row["reason"], row["reason"]))}</span>'
            rows.append(f'<tr><td><code>{esc(row["parameter"])}</code></td><td>{esc(STATUS[row["status"]])}{reason}</td><td>{esc(STATUS.get(corrected.get("status"),"未确定"))}</td></tr>')
        tables.append(f'<h3>{esc(entry)}</h3><table><thead><tr><th>参数</th><th>修复前</th><th>修复后</th></tr></thead><tbody>{"".join(rows)}</tbody></table>')
    unknown = sum(run['counts'].get('abstain', 0) for run in runs.values())
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>文证 · 真实案例来源核对</title><style>
:root{{font-family:"Microsoft YaHei",system-ui,sans-serif;color:#18323f;background:#f4f7f8}}*{{box-sizing:border-box}}body{{margin:0}}main{{max-width:1120px;margin:auto;padding:40px 28px 70px}}h1{{font-size:34px}}h2{{margin-top:36px;font-size:23px}}h3{{font-size:18px;overflow-wrap:anywhere}}p,li{{line-height:1.8}}.tag{{font-size:13px;color:#1f7668;font-weight:700}}.notice{{background:#fff2d5;border-left:4px solid #d8b256;padding:15px 18px}}.meta,.reason{{font-size:13px;color:#5f7079;overflow-wrap:anywhere}}.stats,.values{{display:flex;gap:32px;flex-wrap:wrap}}.stats{{padding:24px;background:#173f42;color:#fff;border-radius:9px;margin:22px 0}}.stats strong,.values strong{{display:block;font-size:28px}}.values{{padding:16px 0}}article{{background:#fff;border:1px solid #d9e3e6;border-radius:10px;padding:24px;margin:18px 0}}blockquote,pre{{margin:16px 0;background:#eef3f4;padding:13px 16px;white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.7}}blockquote{{border-left:3px solid #74a797}}code,pre{{font-family:Consolas,monospace}}code{{overflow-wrap:anywhere}}details{{margin-top:18px}}summary{{color:#216d60;cursor:pointer;line-height:1.7}}table{{width:100%;border-collapse:collapse;background:white}}th,td{{text-align:left;border-bottom:1px solid #dbe4e7;padding:12px;vertical-align:top}}.reason{{display:block;margin-top:4px}}a{{color:#216d60}}.steps{{padding:20px 24px;background:#e5f0eb;border-radius:8px}}footer{{border-top:1px solid #d4dfe3;margin-top:32px;padding-top:16px}}@media(max-width:600px){{main{{padding:24px 14px}}h1{{font-size:27px}}article{{padding:18px}}.stats,.values{{gap:18px}}th,td{{font-size:12px;padding:7px}}ol{{padding-left:20px}}}}
</style></head><body><main><div class="tag">文证 / 第一个真实历史开发案例</div><h1>文档写的默认值，最终由谁决定？</h1>
<p>{esc(result['title'])} · <a href="{esc(result['upstream_url'])}">查看上游修复记录</a></p>
<p class="notice">这是已知的历史修复案例，现用于开发验证。两种服务实现属于同一事件，不能算两个独立项目。本页使用静态源码分析，没有调用语言模型，也没有运行第三方代码。</p>
<div class="stats"><div><strong>{before['counts'].get('default_mismatch_candidate',0)}</strong>修复前不一致候选</div><div><strong>{after['counts'].get('default_mismatch_candidate',0)}</strong>修复后不一致候选</div><div><strong>{unknown}</strong>两版本保留核对项</div></div>
<p>上层服务把参数传给底层库。程序沿导入、继承和参数转发关系，寻找最终声明的默认值，再与参数说明对照。只有明确支持的关系才给出候选；没有候选不等于所有文档正确。保留核对项统计暂时无法确定的记录，同一参数在两个版本会分别计数。</p>
<h2>1. 找到的不一致及修复对照</h2>{''.join(highlights) or '<p>没有找到受支持的不一致候选。</p>'}
<h2>2. 团队现在核对这三件事</h2><div class="steps"><ol><li>旧说明写的是多少秒？新的说明改成多少秒？</li><li>展开来源链，找到真正声明默认值的文件与行号；说明为什么不能只看上层文档。</li><li>解释为什么这只是一个真实开发案例，不能据此说整体准确率已经很高。</li></ol><p>两人分别阅读后，在聊天中写出自己的解释即可。本页不会自动记录你们已完成复核。</p></div>
<h2>3. 全部参数字段，包含正常项和无法判断的项</h2><details><summary>展开完整结果（没有只展示成功条目）</summary><p>两版本分别读取{before['parameter_fields']}、{after['parameter_fields']}个参数字段。没有明确默认值句式、内部占位值、单位或流向不确定的情况保留人工核对；英文原因码供研发追查。</p>{''.join(tables)}</details>
<h2>4. 支持范围</h2><p>当前支持明确的模块导入/导出、单继承、有限父类调用、直接返回类的工厂方法和未改写目标参数的字典转发。参数被改写、动态工厂、多继承、依赖缺失等会停止追踪。这里核对的是指定源码快照中的声明，不是当时用户电脑的完整运行环境。</p>
<p class="meta">生成时间：{esc(result['generated_at'])}<br>本次分析耗时：{result['elapsed_seconds']:.3f} 秒（单次日志，不是性能基准）。完整结果、文件哈希与代码版本记录在同目录review.json。</p>
<footer class="meta">来源文件按固定提交保存，保留两库MIT许可与版权声明。语义检索模块另有开发运行记录；这一页的检出不能归为AI模型带来的提升。</footer></main></body></html>'''
