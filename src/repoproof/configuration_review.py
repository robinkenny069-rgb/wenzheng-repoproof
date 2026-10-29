"""Scoped development checks of configuration objects and annotated fallbacks.

Document-to-API scope and legacy-name suggestions are explicit case inputs.
Neither neural rankings nor upstream expected outcomes determine comparison.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import html
import io
import json
from pathlib import Path
import re
import tokenize

from .contract_graph import Source
from .source_audit import scalar
from .duration_evidence import durations
from .retrieval_text import focused_query


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_path(root, relative):
    path=(root/relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Path outside the pinned source snapshot')
    return path


def read_case(root, case_id, plan):
    entry=next(item for item in plan['cases'] if item['case_id']==case_id)
    if entry['role']!='development':
        raise ValueError('This runner only permits registered development cases')
    directory=root/'research/cases'/case_id
    manifest_path=directory/'manifest.json'
    if digest(manifest_path)!=entry['source_manifest_sha256']:
        raise ValueError('Case manifest differs from the role plan')
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    for name,meta in manifest['files'].items():
        if digest(safe_path(directory,name))!=meta['sha256']:
            raise ValueError(f'Source hash mismatch: {case_id}/{name}')
    return directory,manifest


def class_node(source, qualified):
    name=qualified.rsplit('.',1)[-1]
    found=[node for node in ast.parse(source.text).body if isinstance(node,ast.ClassDef) and node.name==name]
    if len(found)!=1:
        raise ValueError('Configured entry class is not unique')
    return found[0]


def configuration_fields(source, qualified):
    found=set()
    for method in class_node(source,qualified).body:
        if isinstance(method,ast.FunctionDef) and method.name=='__init__':
            for node in ast.walk(method):
                if isinstance(node,ast.Attribute) and isinstance(node.ctx,ast.Store) and isinstance(node.value,ast.Name) and node.value.id=='self':
                    found.add(node.attr)
    return sorted(found)


def unqualified_scope(block, heading):
    """Configured heading binds only its own prose, never an implicit sub-scope."""
    headings=block.get('headings',[])
    return (bool(headings) and headings[-1]==heading and not block.get('incomplete',False)
            and not any(re.search(r'\b(?:previous|previously|former|formerly|historical|history|legacy|deprecated|conditional|when|if)\b',title,re.I) for title in headings))


def bullet_default_claims(blocks, heading):
    """Only explicit end-of-bullet default parentheses in a configured scope."""
    claims=[]
    for block in blocks:
        if block['kind']!='prose' or heading not in block['headings']:
            continue
        lines=block['text'].split('\n')
        starts=[i for i,line in enumerate(lines) if re.match(r'^\s*[-*]\s+',line)]
        for pos,start in enumerate(starts):
            end=starts[pos+1] if pos+1<len(starts) else len(lines)
            raw='\n'.join(lines[start:end]).rstrip()
            named=re.match(r'^\s*[-*]\s+`(?P<name>[A-Za-z_]\w*)`\s*[, :]',raw)
            if not named:
                continue
            tail=re.search(r'\(\s*Defaults?\s+([^()]+)\)\s*[.]?\s*$',raw,re.I)
            claim={'field':named['name'],'text':raw,'path':block['path'],
                   'start_line':block['start_line']+start,'end_line':block['start_line']+end-1,
                   'block_id':block['block_id'],'status':'unsupported','reason':'no_unique_unqualified_default'}
            if (unqualified_scope(block,heading) and tail and len(re.findall(r'\bdefaults?\b',raw,re.I))==1
                    and not re.search(r'\b(?:previously|formerly|historical|legacy|not|never|when|if)\b',raw,re.I)):
                claim.update(scalar(tail.group(1).strip()),expression=tail.group(1).strip())
                if claim['status']=='parsed':
                    claim['reason']='explicit_scalar_default'
            claims.append(claim)
    return claims


def duration_default_claims(blocks, heading):
    claims=[]
    for block in blocks:
        if block['kind']!='prose' or heading not in block['headings']:
            continue
        raw=block['text'].strip()
        match=re.fullmatch(r'(?:The\s+)?default\s+([A-Za-z][A-Za-z _-]*?)\s+is\s+(.+?)[.]?',raw,re.I)
        if not match:
            continue
        value_text=match.group(2).rstrip('.').strip()
        parsed=durations(value_text)
        claim={'field':re.sub(r'[ -]+','_',match.group(1).lower()),'text':raw,
               'path':block['path'],'start_line':block['start_line'],'end_line':block['end_line'],
               'block_id':block['block_id'],'status':'unsupported','reason':'non_simple_duration_sentence'}
        if unqualified_scope(block,heading) and len(parsed)==1 and parsed[0]['text'].strip().lower()==value_text.lower():
            claim.update(status='parsed',seconds=parsed[0]['seconds'],reason='simple_explicit_duration')
        claims.append(claim)
    return claims


def source_comment_duration(trace):
    """Annotations only; never infer an executable unit from a bare number."""
    annotations=[]
    for step in trace:
        if step.get('kind')!='local_class_fallback_binding' or step.get('end_line')!=step.get('line'):
            continue
        line=step.get('source_line','')
        try:
            tokens=list(tokenize.generate_tokens(io.StringIO(line).readline))
            if any(token.type==tokenize.OP and token.string==';' for token in tokens):
                continue
            tree=ast.parse(line.strip())
            original=ast.parse(step.get('text','').strip())
            if (len(tree.body)!=1 or not isinstance(tree.body[0],(ast.Assign,ast.AnnAssign))
                    or ast.dump(tree,include_attributes=False)!=ast.dump(original,include_attributes=False)):
                continue
            comments=[token.string[1:].strip().rstrip('.') for token in tokens if token.type==tokenize.COMMENT]
        except (SyntaxError,tokenize.TokenError,IndentationError):
            continue
        for comment in comments:
            values=durations(comment)
            if len(values)==1 and values[0]['text'].strip().lower()==comment.lower():
                annotations.append({'seconds':values[0]['seconds'],'text':comment,
                                    'path':step['path'],'line':step['line'],
                                    'scope':'source comment annotation, not runtime unit proof'})
    unique={(item['path'],item['line'],item['seconds']):item for item in annotations}
    return next(iter(unique.values())) if len(unique)==1 else None


def compare_configuration_claim(claim, facts, aliases):
    name=claim['field']
    exact=name in facts
    alias=aliases.get(name) if not exact else None
    field=alias.get('candidate_field') if alias else name
    fact=facts.get(field)
    result={'document':claim,'field':field,'source':fact,'status':'manual_review',
            'reason':'field_or_value_not_resolved','automatic_edit':False}
    if fact is None or fact['status']!='resolved' or claim['status']!='parsed':
        return result
    a,b=fact['value'],claim['value']
    equal=Decimal(str(a))==Decimal(str(b)) if type(a) in (int,float) and type(b) in (int,float) else type(a) is type(b) and a==b
    result['value_comparison']='equal' if equal else 'different'
    if alias:
        result.update(status='manual_alias_review',reason='legacy_name_identity_not_automatically_established',
                      alias_condition=alias,comparison_if_alias_is_valid='consistent' if equal else 'mismatch_candidate')
    else:
        result.update(status='consistent_configuration_declaration' if equal else 'configuration_mismatch_candidate',
                      reason='direct_field_and_declared_initial_value')
    return result


def compare_fallback_claim(claim, facts):
    fact=facts.get(claim['field'])
    result={'document':claim,'field':claim['field'],'source':fact,'status':'manual_review',
            'reason':'fallback_or_annotation_unresolved','automatic_edit':False}
    if not fact or fact['status']!='resolved' or claim['status']!='parsed':
        return result
    annotation=source_comment_duration(fact.get('trace',[]))
    if annotation is None:
        return result
    equal=Decimal(claim['seconds'])==Decimal(annotation['seconds'])
    result.update(annotation=annotation,status='consistent_source_annotation' if equal else 'source_annotation_mismatch_candidate',
                  reason='explicit_duration_compared_to_source_comment',
                  value_comparison='equal' if equal else 'different',
                  scope='Comparison to the source annotation associated with the fallback; not a runtime timing proof')
    return result


def reference_ranks(probe, queries, references):
    """Attach curator references only after every method has ranked the full pool."""
    records={row['query_id']:row for row in probe['records']}
    corpus={row['block_id'] for row in probe['corpus']}
    result=[]
    for reference in references:
        matches=[query for query in queries if query['field']==reference['field']]
        blocks=[block['block_id'] for block in probe['documents'] if block['block_id'] in corpus
                and block['start_line']<=reference['end_line'] and block['end_line']>=reference['start_line']]
        if len(matches)!=1 or len(blocks)!=1:
            result.append({'reference':reference,'status':'unresolved_reference','blocks':blocks})
            continue
        ranked=records[matches[0]['query_id']]
        ranks={method:next((row['rank'] for row in rows if row['block_id']==blocks[0]),None)
               for method,rows in ranked['rankings'].items()}
        result.append({'reference':reference,'query_id':matches[0]['query_id'],'block_id':blocks[0],
                       'status':'curated_development_reference','ranks':ranks})
    return {'reference_records':result,'total_queries':len(queries),
            'queries_without_curated_reference':len(queries)-sum(row['status']=='curated_development_reference' for row in result),
            'scope':'Selected upstream repair/neighbor statements; not independent relevance gold or defect accuracy'}


def build_configuration_review(root: Path, *, with_semantic=True):
    from .config_defaults import ConfigDefaults
    from .fallback_defaults import FallbackDefaults
    from .narrative_retrieval import split_document,rank_documents
    protocol_path=root/'research/evaluation/configuration-development-protocol-v1.json'
    protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    plan_path=root/protocol['role_plan']
    plan=json.loads(plan_path.read_text(encoding='utf-8'))
    results=[]
    for spec in protocol['cases']:
        directory,manifest=read_case(root,spec['case_id'],plan)
        runs={}
        for version in ('before','after'):
            source_map={}
            for module,path in spec['module_paths'].items():
                relative=path if path.startswith('dependency/') else f'{version}/{path}'
                source_map[module]=Source(module,relative,safe_path(directory,relative).read_text(encoding='utf-8'))
            sources=list(source_map.values())
            document_path=f'{version}/{spec["document"]}'
            blocks=split_document(safe_path(directory,document_path).read_text(encoding='utf-8'),document_path,spec['format'])
            if spec['source_kind']=='initial_configuration_object_declaration':
                engine=ConfigDefaults(sources)
                fields=configuration_fields(source_map[spec['configuration_class'].rsplit('.',1)[0]],spec['configuration_class'])
                facts={field:engine.resolve_constructor_field(spec['entry_class'],spec['parameter'],field) for field in fields}
                claims=bullet_default_claims(blocks,spec['document_scope_heading'])
                comparisons=[compare_configuration_claim(claim,facts,spec['alias_candidates']) for claim in claims]
            else:
                engine=FallbackDefaults(sources)
                fields=engine.method_names(spec['entry_class'])
                facts={field:engine.resolve(spec['entry_class'],field) for field in fields}
                claims=duration_default_claims(blocks,spec['document_scope_heading'])
                comparisons=[compare_fallback_claim(claim,facts) for claim in claims]
            queries=[{'query_id':spec['case_id']+':'+field,'field':field,
                      'text':focused_query(spec['entry_class'].rsplit('.',1)[-1],field)}
                     for field,fact in facts.items() if fact['status']=='resolved']
            runs[version]={'facts':facts,'document_blocks':blocks,'comparisons':comparisons,
                           'inventory':{'source_entries':len(facts),'resolved_entries':len(queries),
                                        'unresolved_entries':len(facts)-len(queries),'document_blocks':len(blocks),
                                        'scoped_default_claims':len(claims)},'queries':queries}
        # Ranking sees the before source-derived queries and the entire before document.
        # It never receives the comparison decisions, corrected text or curator references.
        probe=rank_documents(runs['before']['document_blocks'],runs['before']['queries'],with_semantic=with_semantic)
        if with_semantic and not probe['model'].get('actual_encode_completed'):
            raise RuntimeError('Requested semantic comparison did not complete; no success or lexical fallback is reported')
        diagnostic=reference_ranks(probe,runs['before']['queries'],spec['retrieval_references'])
        results.append({'case_id':spec['case_id'],'upstream_url':manifest['upstream_url'],
                        'runs':runs,'retrieval':probe,'retrieval_diagnostic':diagnostic,
                        'source_manifest_sha256':digest(directory/'manifest.json'),
                        'source_sha256':{path:meta['sha256'] for path,meta in manifest['files'].items()},
                        'scope_binding':spec['scope_binding_origin'],'licenses':manifest['license']})
    files=['src/repoproof/config_defaults.py','src/repoproof/fallback_defaults.py','src/repoproof/narrative_retrieval.py',
           'src/repoproof/configuration_review.py','src/repoproof/bm25.py','src/repoproof/semantic.py',
           'src/repoproof/source_audit.py','src/repoproof/duration_evidence.py','src/repoproof/cli.py',
           'src/repoproof/lesson.py','src/repoproof/contract_graph.py','src/repoproof/retrieval_text.py','scripts/run_configuration_review.py']
    return {'schema_version':1,'generated_at_utc':datetime.now(timezone.utc).isoformat(),'phase':'development',
            'protocol_sha256':digest(protocol_path),'role_plan_sha256':digest(plan_path),
            'cases':results,'target_code_executed':False,'reserved_cases_run':False,
            'query_representation':'focused_source_identity_v1; source values omitted; original document text retained',
            'query_selection_evidence':'research/evaluation/retrieval-text-selection-v1.json',
            'human_review':'pending for the new modules and cases; prior 3 workbench decisions do not cover them',
            'code_sha256':{name:digest(root/name) for name in files},'limits':protocol['constraints']}


def render_trace(trace,esc):
    labels={'consumer_default':'入口参数使用的初始配置','import_binding':'从另一文件导入配置',
            'configuration_binding':'预先创建配置对象','configuration_construction':'构造配置',
            'explicit_configuration_argument':'构造时明确填写该值','configuration_field_assignment':'保存为对象字段',
            'uninterpreted_decorator':'装饰器仅记录，未解释其运行行为','option_fallback_expression':'没有选项值时使用备用值',
            'local_class_fallback_binding':'同一类中声明的备用值'}
    groups=[]
    for step in trace:
        key=(step['path'],step['line'],step.get('end_line',step['line']))
        if groups and groups[-1]['key']==key:
            groups[-1]['labels'].append(labels.get(step['kind'],step['kind']))
        else:
            code=step.get('text','') if '\n' in step.get('text','') else step.get('source_line') or step.get('text','')
            groups.append({'key':key,'labels':[labels.get(step['kind'],step['kind'])],'code':code})
    return ''.join(f'<li><p>{esc(" → ".join(group["labels"]))}</p><p class="meta">{esc(group["key"][0])} · L{group["key"][1]}–{group["key"][2]}</p><pre>{esc(group["code"])}</pre></li>' for group in groups)


def render_configuration_review(result):
    esc=lambda value:html.escape(str(value),quote=True)
    names={'consistent_configuration_declaration':'该字段与初始配置声明一致','configuration_mismatch_candidate':'配置声明不一致候选',
           'manual_alias_review':'旧名称对应关系待人工确认','consistent_source_annotation':'与源码注释中的时长一致',
           'source_annotation_mismatch_candidate':'与源码注释中的时长不一致','manual_review':'证据不足，保留核对'}
    method_names={'lexical':'共同词语','bm25':'BM25','semantic':'AI语义','lexical_semantic_rrf':'共同词语+AI','bm25_semantic_rrf':'BM25+AI'}
    panels=[]
    for case in result['cases']:
        before,after=case['runs']['before'],case['runs']['after']
        later={row['field']:row for row in after['comparisons']}
        cards=[]
        for row in before['comparisons']:
            fact=row.get('source') or {}
            trace=render_trace(fact.get('trace',[]),esc)
            fixed=later.get(row['field'],{})
            caution=''
            if row['status']=='manual_alias_review':
                caution='<p class="notice">旧文档使用了不同名称。下方数值比较以两者指同一设置为前提；这项对应关系仍需人工确认，相似度不能代替身份依据。</p>'
            if 'annotation' in row:
                caution+='<p class="notice">这里比较的是源码声明旁的时长注释。注释也可能过时，不能将本项写成已经证明运行期等待时长。</p>'
            document=row['document']
            document_value=(str(document['value']) if document.get('status')=='parsed' and 'value' in document
                            else str(document['seconds'])+' 秒' if document.get('status')=='parsed' and 'seconds' in document else '未确定')
            cards.append(f'''<article><p class="tag">{esc(names[row['status']])}</p><h3>{esc(row['field'])}</h3>{caution}
<div class="values"><div>追踪到的声明值<strong>{esc(fact.get('value','未确定'))}</strong></div><div>说明中的值<strong>{esc(document_value)}</strong></div><div>附条件的对照<strong>{'不同' if row.get('value_comparison')=='different' else '相同' if row.get('value_comparison')=='equal' else '未确定'}</strong></div></div>
<p class="meta">修复前原文 · {esc(document['path'])} · L{document['start_line']}–{document['end_line']}</p><pre>{esc(document['text'])}</pre>
<details><summary>查看源码依据与适用条件</summary><ol>{trace}</ol><p class="meta">{esc(fact.get('scope',''))}</p><ul>{''.join('<li>'+esc(item)+'</li>' for item in fact.get('assumptions',[]))}</ul></details>
<details><summary>修复后的说明：{esc(names.get(fixed.get('status'),'未找到对应项'))}</summary><pre>{esc(fixed.get('document',{}).get('text','未找到唯一对应项'))}</pre></details></article>''')
        refs=case['retrieval_diagnostic']['reference_records']
        methods=list(next((row['ranks'].keys() for row in refs if 'ranks' in row),[]))
        rankrows=''.join('<tr><td>'+esc(row['reference']['field'])+'</td>'+''.join('<td>'+esc(row.get('ranks',{}).get(method,'未确定'))+'</td>' for method in methods)+'</tr>' for row in refs)
        inventory=''.join(f'<tr><td>{esc(field)}</td><td>{esc(fact["status"])}</td><td>{esc(fact.get("value",fact.get("reason","")))}</td></tr>' for field,fact in before['facts'].items())
        model=case['retrieval'].get('model',{})
        model_note=('本次已实际运行本地 MiniLM 模型，五种方法使用相同段落池。' if model.get('actual_encode_completed')
                    else '本次仅运行词语与 BM25 静态检索，未运行 AI 语义模型。')
        scope_note={'httpx-1535':'核对范围：人工指定 Client 的 limits 初始配置；调用者显式传入 limits 或 pool_limits 时，不适用本项结论。',
                    'celery-4476':'核对范围：上游修复所引用的固定 Kombu 源码，未复原当时完整安装环境。备用值1800旁边注明“30分钟”；本项比较文档与这条注释。'}.get(case['case_id'],case['scope_binding'])
        panels.append(f'''<section><h2>{esc(case['case_id'])}</h2><p><a href="{esc(case['upstream_url'])}">查看原始历史修复</a></p><p class="meta">{esc(scope_note)}</p>{''.join(cards)}
<div class="panel"><h3>完整文档中的定位排名</h3><p>{esc(model_note)}</p><p>对{len(case['retrieval']['corpus'])}个可检索段落、{len(before['queries'])}个源码查询实际排序。下表数字是对应原文的位置，越小越靠前；它不是错误概率。</p>
<div class="scroll"><table><thead><tr><th>开发参考</th>{''.join('<th>'+esc(method_names.get(method,method))+'</th>' for method in methods)}</tr></thead><tbody>{rankrows}</tbody></table></div>
<p class="meta">参考来自已知上游修复及邻近说明，模型排序之后才附加。相邻字段可能落在同一段，不能当成独立样本；无参考的查询也保留在JSON中。代码块和指令未用于自然段检索，完整排除清单已保存。</p></div>
<details><summary>全部{len(before['facts'])}个源条目，含无法追踪的情况</summary><table><thead><tr><th>字段/方法</th><th>追踪状态</th><th>声明值或原因</th></tr></thead><tbody>{inventory}</tbody></table></details></section>''')
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>文证 · 配置来源核验</title><style>
*{{box-sizing:border-box}}body{{font-family:"Microsoft YaHei",system-ui,sans-serif;background:#f3f6f5;color:#233d40;margin:0;line-height:1.75}}main{{max-width:1120px;margin:auto;padding:32px 26px 60px}}h1{{font-size:32px}}h2{{font-size:25px;border-bottom:1px solid #cedcd5;padding-bottom:10px}}h3{{font-size:18px;overflow-wrap:anywhere}}a,summary{{color:#17675c}}summary{{cursor:pointer}}.tag{{color:#17675c;font-weight:600;font-size:13px}}.notice{{background:#fff2d8;border-left:3px solid #c49337;padding:12px 16px}}.meta{{color:#5a6d70;font-size:13px;overflow-wrap:anywhere}}article,.panel{{background:white;border:1px solid #d7e2dd;border-radius:10px;padding:21px;margin:18px 0}}section{{margin:36px 0}}pre{{background:#eff4f1;padding:13px;white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}}.values{{display:flex;gap:50px;flex-wrap:wrap}}.values strong{{display:block;font-size:25px}}table{{border-collapse:collapse;width:100%;background:white}}th,td{{text-align:left;vertical-align:top;border-bottom:1px solid #dce4df;padding:10px;overflow-wrap:anywhere}}.scroll{{overflow-x:auto}}details{{margin-top:15px}}li{{margin-bottom:10px}}@media(max-width:600px){{main{{padding:20px 12px}}h1{{font-size:26px}}article,.panel{{padding:15px}}th,td{{font-size:12px;padding:6px}}ol{{padding-left:22px}}}}
</style></head><body><main><p class="tag">文证 / 配置对象与选项回退开发验证</p><h1>默认值不只写在函数括号里</h1><p>程序追踪预先创建的配置对象、类属性和明确的选项回退表达式，再把来源和文档放在一起核对。</p><p class="notice">这是两个已知历史修复的开发验证。没有执行第三方代码；初始配置、调用者覆盖、文档旧名称和源码注释各有边界，页面逐项保留条件。预留测试组未参与运行。</p><p><a href="../workbench/index.html">返回文证工作台</a> · <a href="review.json" download>下载完整结果</a></p>{''.join(panels)}<p class="meta">生成时间：{esc(result['generated_at_utc'])}。JSON包含固定来源、代码哈希、未解析条目和逐查询排名。本页尚未经过用户对新增实现与案例的复核。</p></main></body></html>'''
