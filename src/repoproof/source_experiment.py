"""Development-only full-file audit and same-corpus retrieval diagnostics."""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
from time import perf_counter

from .source_audit import audit_source
from .cli import lexical_rank
from .bm25 import BM25Index
from .lesson import reciprocal_rank_fusion
from .retrieval_text import focused_query


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lookup_queries(audit):
    """All unique, unconditional scalar declarations, including undocumented ones."""
    counts=Counter(function['symbol'] for function in audit['functions'])
    queries=[]
    for function in audit['functions']:
        if function['conditional'] or counts[function['symbol']]!=1:
            continue
        for parameter,default in function['parameters'].items():
            if default['status']!='parsed':
                continue
            queries.append({'query_id':f'{function["symbol"]}/{parameter}', 'symbol':function['symbol'],
                            'parameter':parameter,'line':default['line'],
                            'text':focused_query(function['symbol'],parameter)})
    return queries


def retrieval_probe(audit):
    from .semantic import SemanticRetriever
    started=perf_counter()
    blocks=audit['blocks']
    queries=lookup_queries(audit)
    texts=[block['symbol']+'\n'+block['text'] for block in blocks]
    bm25_index=BM25Index(texts,k1=1.2,b=0.75)
    encoder=SemanticRetriever()
    loaded=perf_counter()
    vectors=encoder.encode(texts+[query['text'] for query in queries])
    encoded=perf_counter()
    doc_vectors=vectors[:len(texts)]
    query_vectors=vectors[len(texts):]
    scores=query_vectors @ doc_vectors.T
    # Rankings are computed without reading labels, annotation files or the upstream diff.
    records=[]
    for qi,query in enumerate(queries):
        lexical=lexical_rank(query['text'],texts)
        bm25=bm25_index.rank(query['text'])
        semantic=[(idx,float(scores[qi,idx])) for idx in sorted(range(len(blocks)),key=lambda j:(-float(scores[qi,j]),j))]
        fusion=reciprocal_rank_fusion([lexical,semantic])
        bm25_fusion=reciprocal_rank_fusion([bm25,semantic])
        rankings={name:[{'block_id':blocks[idx]['block_id'],'score':float(score)} for idx,score in ranking]
                  for name,ranking in [('lexical',lexical),('bm25',bm25),('semantic',semantic),('rrf',fusion),('bm25_rrf',bm25_fusion)]}
        # This is syntactic owner/parameter correspondence, NOT defect truth labels.
        targets=[block['block_id'] for block in blocks if block['symbol']==query['symbol'] and query['parameter'] in block['parameters']]
        structural=targets if len(targets)==1 else []
        ranks={name:next((position for position,row in enumerate(ranking,1) if row['block_id'] in targets),None)
               for name,ranking in rankings.items()}
        records.append({**query,'rankings':rankings,'structural_reference_blocks':targets,
                        'reference_status':'unique_owner_parameter_mapping' if len(targets)==1 else 'missing_or_ambiguous',
                        'first_reference_rank':ranks,'structural_lookup':structural})
    eligible=[record for record in records if record['reference_status']=='unique_owner_parameter_mapping']
    summaries={}
    for method in ('lexical','bm25','semantic','rrf','bm25_rrf'):
        ranks=[record['first_reference_rank'][method] for record in eligible]
        summaries[method]={'reference_queries':len(eligible),
                           'hits':{str(k):sum(rank is not None and rank<=k for rank in ranks) for k in (1,3,5)},
                           'mean_reciprocal_rank':sum(1/rank for rank in ranks if rank)/len(ranks) if ranks else None}
    return {'scope':'Development diagnostic: retrieve syntactically linked parameter fields; not bug-detection accuracy or independent semantic relevance labels.',
            'corpus_blocks':len(blocks),'total_queries':len(queries),'unique_reference_queries':len(eligible),
            'bm25':bm25_index.specification(),
            'query_representation':'focused_source_identity_v1; values omitted; original document text retained',
            'queries_without_unique_reference':len(records)-len(eligible),'records':records,'summaries':summaries,
            'structural_lookup_hits':len(eligible),'structural_lookup_note':'Exact owner/parameter mapping defines the reference; its perfect correspondence is by construction, not learned performance.',
            'corpus_sha256':hashlib.sha256(json.dumps(texts,ensure_ascii=False).encode()).hexdigest(),
            'query_sha256':hashlib.sha256(json.dumps(queries,ensure_ascii=False).encode()).hexdigest(),
            'model':{'model_id':encoder.manifest['model_id'],'revision':encoder.manifest['revision'],
                     'onnx_sha256':encoder.manifest['files']['onnx/model.onnx']['sha256'],
                     'providers':encoder.session.get_providers(),'threads':4,'max_length':256,'rrf_k':60},
            'seconds':{'load_and_hash_check':loaded-started,'encode':encoded-loaded,'total':perf_counter()-started}}


def run_development_case(root: Path, *, with_semantic=True):
    started=perf_counter()
    case_id='matplotlib-14475'
    plan_path=root/'research/evaluation/plan-v1.json'
    plan=json.loads(plan_path.read_text(encoding='utf-8'))
    entry=next(case for case in plan['cases'] if case['case_id']==case_id)
    if entry['role']!='development':
        raise ValueError('This runner only permits the explicitly assigned development event')
    case_root=root/'research/cases'/case_id
    manifest_path=case_root/'manifest.json'
    if digest(manifest_path)!=entry['source_manifest_sha256']:
        raise ValueError('Source manifest differs from the recorded evaluation plan')
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    for relative,meta in manifest['files'].items():
        if digest(case_root/relative)!=meta['sha256']:
            raise ValueError(f'Source hash mismatch: {relative}')
    # No target function or expected defect is supplied to the audit.
    runs={version:audit_source((case_root/version/'lib/matplotlib/figure.py').read_text(encoding='utf-8'),
                                'lib/matplotlib/figure.py') for version in ('before','after')}
    probe=retrieval_probe(runs['before']) if with_semantic else {'status':'not_requested','model_called':False}
    files=['src/repoproof/source_audit.py','src/repoproof/source_experiment.py','src/repoproof/semantic.py',
           'src/repoproof/cli.py','src/repoproof/bm25.py','src/repoproof/lesson.py','src/repoproof/retrieval_text.py','scripts/run_source_audit.py']
    return {'schema_version':1,'generated_at_utc':datetime.now(timezone.utc).isoformat(),'case_id':case_id,
            'phase':'development','runs':runs,'retrieval_probe':probe,'model_called':with_semantic,
            'source_manifest_sha256':digest(manifest_path),'evaluation_plan_sha256':digest(plan_path),
            'source_sha256':{name:meta['sha256'] for name,meta in manifest['files'].items()},
            'code_sha256':{name:digest(root/name) for name in files},'target_code_executed':False,
            'seconds':perf_counter()-started,'human_review':'pending; R03 guided understanding is not full-file review',
            'limits':['One known development event; no hold-out repositories analyzed.',
                      'Only recognized NumPy-style parameter fields; docstring/template exclusions remain in the function inventory.',
                      'Comparison of lexical declarations, not execution or automatic repair.']}


def render_development_report(result):
    esc=lambda value:html.escape(str(value),quote=True)
    before,after=result['runs']['before'],result['runs']['after']
    status={'consistent_declared_value':'声明值一致','declared_default_mismatch_candidate':'声明值不一致候选','abstain':'暂不能确定'}
    reasons={'parameter_not_explicit_in_signature':'需要继续追踪参数转发','no_unique_explicit_header_default':'缺少唯一的明确默认值',
             'default_not_in_supported_header':'当前尚不支持正文中的默认值句式','non_literal_or_qualified_default':'默认值含配置、限定或表达式',
             'required_parameter_has_no_default':'声明中是必填参数','sentinel_requires_behavior_review':'None可能触发内部设置，需要进一步核对',
             'non_scalar_default':'默认值不是受支持的单个值','literal_values_agree':'声明与说明的默认值相符','literal_values_differ':'声明与说明的默认值不同'}
    after_map={(row['symbol'],row['parameter'],row['document']['start_line']):row for row in after['rows']}
    cards=[]
    for row in before['rows']:
        if row['status']!='declared_default_mismatch_candidate':continue
        repaired=after_map.get((row['symbol'],row['parameter'],row['document']['start_line']),{})
        cards.append(f'''<article><h2>{esc(row['symbol'])} / {esc(row['parameter'])}</h2>
<p>声明值：<strong>{esc(row['declaration']['expression'])}</strong> · 旧说明值：<strong>{esc(row['claim']['expression'])}</strong> · 修复后：{esc(status.get(repaired.get('status'),'未找到对应项'))}</p>
<p class="meta">代码 L{row['declaration']['line']}</p><pre>{esc(row['declaration']['source_line'])}</pre>
<p class="meta">原说明 L{row['document']['start_line']}–{row['document']['end_line']}</p><pre>{esc(row['document']['text'])}</pre></article>''')
    rows=[]
    for row in before['rows']:
        corrected=after_map.get((row['symbol'],row['parameter'],row['document']['start_line']),{})
        rows.append(f'<tr><td>{esc(row["symbol"])}<br><strong>{esc(row["parameter"])}</strong><br><span class="meta">L{row["document"]["start_line"]}</span></td><td>{esc(status[row["status"]])}</td><td>{esc(status.get(corrected.get("status"),"未找到对应项"))}</td><td>{esc(reasons.get(row["reason"],row["reason"]))}</td></tr>')
    probe=result['retrieval_probe']
    if result['model_called']:
        method_names={'lexical':'按共同词语查找','bm25':'BM25加权词语检索','semantic':'AI按意思查找','rrf':'共同词语 + AI融合','bm25_rrf':'BM25 + AI融合'}
        retrieval_rows=''.join(f'<tr><td>{method_names[name]}</td><td>{summary["hits"]["1"]} / {summary["reference_queries"]}</td><td>{summary["hits"]["3"]} / {summary["reference_queries"]}</td><td>{summary["hits"]["5"]} / {summary["reference_queries"]}</td></tr>' for name,summary in probe['summaries'].items())
        retrieval=f'''<p>真实本地AI模型已运行。各检索方法读取同一份{probe['corpus_blocks']}段参数说明及同一批{probe['total_queries']}个查询。其中{probe['unique_reference_queries']}个查询有唯一的结构关联参考，另外{probe['queries_without_unique_reference']}个无唯一参考，均保留在JSON中。BM25使用预先确定的k1=1.2、b=0.75，融合均为等权RRF，k=60。</p>
<table><thead><tr><th>方法</th><th>前1条找到对应说明</th><th>前3条</th><th>前5条</th></tr></thead><tbody>{retrieval_rows}</tbody></table>
<p>这只检查“把对应说明排在前面”的能力，不能当作文档判错准确率。参考来自明确的函数名和参数名，因此结构直接匹配按定义即可定位{probe['unique_reference_queries']}项；它不是模型学出来的分数。这类明确命名的说明不需要先依赖语义检索。</p>'''
    else:retrieval='<p>本次选择仅运行静态核对，没有调用模型。需要查看AI定位对照时，双击“启动整文件核验.cmd”。</p>'
    inventory=before['inventory']
    inventory_rows=''.join(f'<tr><td>{esc(f["symbol"])}</td><td>L{f["line"]}</td><td>{esc(f["documentation_status"])}</td></tr>' for f in before['functions'])
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>文证 · 整文件核验</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f3f6f7;color:#19313e;font-family:"Microsoft YaHei",system-ui,sans-serif}}main{{max-width:1100px;margin:auto;padding:36px 25px 64px}}h1{{font-size:30px}}h2{{font-size:22px}}p{{line-height:1.8}}.tag{{color:#216e60;font-size:13px;font-weight:700}}.notice{{padding:16px 20px;background:#fff2d5;border-left:4px solid #c49d46}}.stats{{display:flex;flex-wrap:wrap;gap:24px;background:#204b4b;color:white;padding:24px;margin:22px 0;border-radius:8px}}.stats strong{{display:block;font-size:28px}}.meta{{font-size:13px;color:#5b6f77;overflow-wrap:anywhere}}article{{padding:22px;background:white;border:1px solid #d9e2e5;border-radius:8px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef3f4;padding:14px;line-height:1.6}}table{{width:100%;border-collapse:collapse;background:white}}th,td{{padding:11px;text-align:left;vertical-align:top;border-bottom:1px solid #d5e0e4;overflow-wrap:anywhere}}td:first-child{{max-width:290px}}summary{{cursor:pointer;color:#216e60;line-height:1.8}}section{{margin:28px 0}}a{{color:#216e60}}@media(max-width:600px){{main{{padding:22px 12px}}h1{{font-size:25px}}th,td{{font-size:12px;padding:7px}}article{{padding:15px}}}}
</style></head><body><main><p class="tag">文证 / 开发组整文件实验</p><h1>让程序检查整份源码里的参数说明</h1>
<p class="notice">这是一条已知历史修复的开发实验，预留测试组没有参与运行。程序读取固定版本源码，不执行第三方项目。候选仍需复核；没有告警不等于所有说明正确。</p>
<div class="stats"><div><strong>{inventory['function_definitions']}</strong>发现的函数定义</div><div><strong>{inventory['parameter_rows']}</strong>识别到的参数记录</div><div><strong>{before['counts'].get('declared_default_mismatch_candidate',0)} → {after['counts'].get('declared_default_mismatch_candidate',0)}</strong>修复前后候选数</div><div><strong>{before['counts'].get('abstain',0)}</strong>旧版暂不能确定</div></div>
<p>以前先挑出6条供人阅读；这次直接读取整份文件，自动识别NumPy样式的参数说明。所有识别到的记录均保留，并列出无法解析或缺少此类说明的函数。</p>
<section><h2>1. 候选与源码依据</h2>{''.join(cards) or '<p>没有受支持的候选。</p>'}</section>
<section><h2>2. AI能否把对应说明找出来？</h2>{retrieval}</section>
<section><h2>3. 完整参数结果</h2><details><summary>展开全部{len(before['rows'])}条记录，包含正常项与拒判项</summary><table id="all-results"><thead><tr><th>函数与参数</th><th>修复前</th><th>修复后</th><th>原因</th></tr></thead><tbody>{''.join(rows)}</tbody></table></details></section>
<section><details><summary>查看{len(before['functions'])}个函数的解析范围</summary><table><thead><tr><th>函数</th><th>位置</th><th>参数说明识别状态</th></tr></thead><tbody>{inventory_rows}</tbody></table></details></section>
<p>下一步重点是完整候选池中的跨文件自然语言说明，以及模板、配置和参数转发等无法直接比较的情况。当前页面不证明整体准确率或AI创新优势。</p>
<p class="meta">固定来源：<a href="https://github.com/matplotlib/matplotlib/pull/14475">Matplotlib历史修复 #14475</a>。保留原许可证。生成时间：{esc(result['generated_at_utc'])}。JSON记录源码、代码哈希与逐项输出。单次总用时{result['seconds']:.2f}秒，不是性能基准。</p></main></body></html>'''
