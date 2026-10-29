"""Additive v2 product orchestration; frozen v1 analyzers remain unchanged.

Inputs are caller-selected source/doc text. Curated repair references never enter
this module. Each analysis family keeps its own inventory and comparison scope.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .source_audit import audit_source
from .source_experiment import lookup_queries
from .narrative_retrieval import split_document, rank_documents

MAX_RETRIEVAL_PAIRS = 20000


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode('utf-8')).hexdigest()


def source_inventory(sources):
    audits, errors, rows, queries, blocks = [], [], [], [], []
    for path, text in sorted(sources.items()):
        try:
            audit = audit_source(text, path)
        except (SyntaxError, ValueError, IndentationError, RecursionError) as error:
            errors.append({'path': path, 'reason': type(error).__name__, 'detail': str(error)})
            continue
        audits.append(audit)
        rows.extend(audit['rows'])
        for query in lookup_queries(audit):
            queries.append({**query, 'path': path, 'query_id': path+'::'+query['query_id']})
        blocks.extend({**{key: block[key] for key in ('block_id','path','start_line','end_line','text')},
                       'headings': [block['symbol']], 'kind': 'prose'} for block in audit['blocks'])
    return {'files': audits, 'parse_errors': errors, 'rows': rows, 'queries': queries, 'blocks': blocks,
            'inventory': {'source_files': len(sources), 'audited_files': len(audits), 'parse_errors': len(errors),
                          'functions': sum(len(a['functions']) for a in audits), 'parameter_rows': len(rows),
                          'queries': len(queries), 'parameter_blocks': len(blocks)},
            'counts': dict(Counter(row['status'] for row in rows))}


def snippet(sources, path, line, radius=2):
    if path not in sources or not isinstance(line, int) or line < 1:
        return None
    lines = sources[path].split('\n')
    start, end = max(1, line-radius), min(len(lines), line+radius)
    return {'path': path, 'start_line': start, 'end_line': end,
            'text': '\n'.join(lines[start-1:end])}


def display_row(row, origin, sources):
    result = {**row, 'origin': origin}
    doc = row.get('document', {})
    declaration = row.get('declaration', row.get('source', {})) or {}
    source_path = (declaration.get('path') or row.get('path') or row.get('source_path')
                   or (row.get('source') or {}).get('path')
                   or (doc.get('path') if origin != 'rst' else None))
    if isinstance(declaration.get('declaration'), dict):
        source_path = source_path or declaration.get('path')
        declaration = declaration['declaration']
    result['code_excerpt'] = snippet(sources, source_path, declaration.get('line', row.get('source_line')))
    result['record_id'] = digest({'origin': origin, 'symbol': row.get('symbol'), 'parameter': row.get('parameter'),
                                  'document': doc, 'source_path': source_path, 'declaration': declaration,
                                  'claim_id': row.get('claim_id'), 'claim': row.get('claim')})
    return result


def analyze_snapshot(sources, documents, *, with_semantic=True):
    from .rst_evidence_v2 import analyze_rst
    from .template_evidence_v2 import analyze_templates

    literal = source_inventory(sources)
    rst_documents = {p: t for p, t in documents.items() if p.lower().endswith('.rst')}
    rst = analyze_rst(sources, rst_documents)
    templates = analyze_templates(sources)
    display = [display_row(row, 'literal', sources) for row in literal['rows']]
    display.extend(display_row(row, 'rst', sources) for row in rst['rows'])
    display.extend(display_row(row, 'template', sources) for row in templates['rows'])
    template_functions = {(f['path'], f['symbol']): f for f in templates['functions']}
    for row in display:
        if row['origin'] != 'template':
            continue
        function = template_functions.get((row['document']['path'], row['symbol']), {})
        row['expansion_steps'] = function.get('expansion_steps', [])
        row['template_source_excerpts'] = []
        for span in row['document'].get('source_spans', []):
            if span['path'] in sources:
                lines = sources[span['path']].split('\n')
                row['template_source_excerpts'].append({**span, 'text': '\n'.join(lines[span['start_line']-1:span['end_line']])})
    external = list(rst['blocks'])
    for path, text in sorted(documents.items()):
        if path.lower().endswith('.md'):
            external.extend(split_document(text, path, 'markdown'))
    # Never build a retrieval pool from mismatches only. External documents keep
    # all eligible prose; source-only inputs retain the original full field pool.
    pool = literal['blocks'] + external
    eligible = sum(b['kind'] == 'prose' and bool(b['text'].strip()) and not b.get('incomplete') for b in pool)
    pairs = eligible * len(literal['queries'])
    if pairs > MAX_RETRIEVAL_PAIRS:
        raise ValueError(f'Retrieval size {pairs} exceeds {MAX_RETRIEVAL_PAIRS} query/document pairs; select a smaller snapshot directory. No results were silently truncated.')
    probe = rank_documents(pool, [{'query_id': q['query_id'], 'text': q['text']}
                                  for q in literal['queries']], with_semantic=with_semantic)
    if with_semantic and probe['corpus'] and literal['queries'] and not probe['model']['actual_encode_completed']:
        raise RuntimeError('Requested neural encoding failed; no successful AI run is claimed')
    return {'literal': literal, 'rst': rst, 'templates': templates, 'display_rows': display,
            'family_counts': {kind: dict(Counter(r['status'] for r in display if r['origin'] == kind))
                              for kind in ('literal','template','rst')},
            'retrieval': probe, 'retrieval_route': 'original_parameter_blocks_and_external_prose' if documents else 'original_parameter_blocks',
            'retrieval_note': 'Template expansion is evaluated as a separate static evidence feature; no neural gain is attributed to that feature.',
            'retrieval_budget': {'pairs': pairs, 'maximum_pairs': MAX_RETRIEVAL_PAIRS, 'truncated': False},
            'target_code_executed': False}


def analyze_case(case_id, before, after=None, *, with_semantic=True):
    snapshots = {'before': before}
    if after is not None:
        snapshots['after'] = after
    runs = {}
    for stage, snapshot in snapshots.items():
        runs[stage] = analyze_snapshot(snapshot['sources'], snapshot['documents'], with_semantic=with_semantic)
        runs[stage]['input_inventory'] = snapshot.get('inventory', {})
        runs[stage]['input_files'] = snapshot.get('files', [])
        runs[stage]['input_exclusions'] = snapshot.get('excluded', [])
    identity = {stage: [{'path': p, 'sha256': hashlib.sha256(t.encode('utf-8')).hexdigest()}
                       for p, t in sorted({**snap['sources'], **snap['documents']}.items())]
                for stage, snap in snapshots.items()}
    return {'case_id': case_id, 'source_identity': identity, 'runs': runs}


def render_product(payload, template):
    view = deepcopy(payload)
    for case in view.get('cases', []):
        for run in case['runs'].values():
            for record in run['retrieval']['records']:
                record['rankings'] = {method: ranking[:3] for method, ranking in record['rankings'].items()}
    view['view_scope'] = {'retrieval_rankings': 'first 3 per method; original ranks preserved',
                          'full_machine_result': 'review.json', 'presentation_only': True}
    encoded = json.dumps(view, ensure_ascii=False, separators=(',', ':')).replace('&','\\u0026').replace('<','\\u003c').replace('>','\\u003e').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
    text = Path(template).read_text('utf-8')
    if text.count('__PAYLOAD_JSON__') != 1:
        raise ValueError('Expected exactly one payload slot')
    return text.replace('__PAYLOAD_JSON__', encoded)
