"""BM25-first retrieval with a bounded, label-free NumPy owner shortcut.

An exact owner match establishes which original field documents a parameter;
it says nothing about whether that field is correct. No defect verdict, claim,
reference label, default value or semantic score participates in routing.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import re


_PLACEHOLDER = re.compile(r'%\([^)]+\)')

def has_unexpanded_placeholder(text):
    return bool(_PLACEHOLDER.search(text))


def _groups(items, key):
    grouped = defaultdict(list)
    for item in items:
        grouped[key(item)].append(item)
    return grouped


def _direct_owner(query, functions, fields, documents, corpus, bm25):
    identity = (query.get('path'), query.get('symbol'))
    parameter = query.get('parameter')
    if any(not isinstance(part, str) or not part for part in (*identity, parameter)):
        return None, 'missing_query_identity'
    owners = functions.get(identity, [])
    if len(owners) != 1:
        return None, 'missing_function_owner' if not owners else 'ambiguous_function_owner'
    owner, audit_path = owners[0]
    if audit_path != identity[0]:
        return None, 'inconsistent_source_inventory_path'
    if owner.get('duplicate', False) is not False:
        return None, 'ambiguous_function_owner'
    if owner.get('conditional') is not False:
        return None, 'conditional_or_unknown_function_scope'
    if type(owner.get('decorated')) is not bool:
        return None, 'unknown_decorator_inventory'
    if not isinstance(owner.get('parameters'), dict) or parameter not in owner['parameters']:
        return None, 'parameter_not_in_source_signature'
    # Count the complete original field inventory, not just retrieved/eligible
    # fields: excluding one duplicate from the pool must not invent uniqueness.
    matches = [(block, path) for block, path in fields.get(identity, [])
               if isinstance(block.get('parameters'), list) and parameter in block['parameters']]
    if len(matches) != 1:
        return None, 'missing_parameter_field' if not matches else 'ambiguous_parameter_fields'
    block, audit_path = matches[0]
    if audit_path != identity[0] or block['parameters'].count(parameter) != 1:
        return None, 'ambiguous_or_inconsistent_parameter_field'
    if block.get('kind', 'prose') != 'prose' or block.get('format') not in (None, 'numpy'):
        return None, 'unsupported_source_field_kind'
    if not isinstance(block.get('text'), str) or not block['text'].strip():
        return None, 'missing_parameter_field_text'
    if has_unexpanded_placeholder(block['text']):
        return None, 'unexpanded_template'
    block_id = block.get('block_id')
    if not isinstance(block_id, str) or not block_id:
        return None, 'missing_source_block_id'
    if len(corpus.get(block_id, [])) != 1:
        return None, 'source_block_not_unique_in_corpus'
    doc_matches = documents.get(block_id, [])
    if len(doc_matches) != 1:
        return None, 'source_block_not_unique_in_documents'
    document = doc_matches[0]
    if document.get('kind') != 'prose' or document.get('incomplete'):
        return None, 'unsupported_or_incomplete_pool_document'
    if document.get('format') not in (None, 'numpy'):
        return None, 'pool_document_is_not_original_numpy_field'
    for name in ('path', 'start_line', 'end_line', 'text'):
        if name not in block or document.get(name) != block[name]:
            return None, 'source_and_pool_document_differ'
    if (type(block['start_line']) is not int or type(block['end_line']) is not int
            or block['start_line'] < 1 or block['end_line'] < block['start_line']):
        return None, 'invalid_source_field_coordinates'
    if sum(item.get('block_id') == block_id for item in bm25) != 1:
        return None, 'source_block_not_unique_in_bm25'
    # A primary ranking must remain a permutation of this run's complete pool.
    ranked = [item.get('block_id') for item in bm25]
    if len(ranked) != len(set(ranked)) or set(ranked) != set(corpus):
        return None, 'bm25_does_not_cover_corpus_once'
    return block_id, 'unique_same_file_symbol_parameter_in_original_numpy_field'


def route_queries(run):
    """Route a complete product_v2 snapshot without modifying it.

    ``rankings.bm25`` is the untouched full baseline list. In
    ``rankings.structure_bm25`` only an eligible exact owner's position changes;
    other documents keep their BM25 order. Its score is still the original
    BM25 score, explicitly named by ``score_method``. Semantic rankings, if
    supplied, are copied separately and never influence the primary order.
    """
    literal, retrieval = run['literal'], run['retrieval']
    queries = literal['queries']
    if len({query['query_id'] for query in queries}) != len(queries):
        raise ValueError('query_id must be unique in the literal query inventory')
    functions, fields = defaultdict(list), defaultdict(list)
    for audit in literal['files']:
        for function in audit['functions']:
            functions[(function.get('path'), function.get('symbol'))].append((function, audit.get('path')))
        for block in audit['blocks']:
            fields[(block.get('path'), block.get('symbol'))].append((block, audit.get('path')))
    documents = _groups(retrieval['documents'], lambda item: item.get('block_id'))
    corpus = _groups(retrieval['corpus'], lambda item: item.get('block_id'))
    probes = _groups(retrieval['records'], lambda item: item.get('query_id'))
    records = []
    for query in queries:
        found = probes.get(query['query_id'], [])
        if len(found) > 1:
            raise ValueError('query_id must be unique in retrieval records')
        probe = found[0] if found else {}
        rankings = probe.get('rankings', {})
        bm25 = rankings.get('bm25', [])
        if not isinstance(bm25, list) or any(not isinstance(item, dict) for item in bm25):
            raise TypeError('BM25 ranking must be a list of document ranking records')
        if 'bm25' not in rankings:
            direct, reason = None, 'missing_bm25_ranking'
        else:
            direct, reason = _direct_owner(query, functions, fields, documents, corpus, bm25)
        order = list(range(len(bm25)))
        if direct is not None:
            first = next(i for i, item in enumerate(bm25) if item['block_id'] == direct)
            order = [first] + [i for i in order if i != first]
        primary = [{**deepcopy(bm25[index]), 'rank': rank, 'bm25_rank': index + 1,
                    'method': 'structure_bm25', 'score_method': 'bm25'}
                   for rank, index in enumerate(order, 1)]
        owners = functions.get((query.get('path'), query.get('symbol')), [])
        decorated = owners[0][0].get('decorated') if len(owners) == 1 else None
        records.append({**{key: query.get(key) for key in ('query_id', 'path', 'symbol', 'parameter')},
                        'direct_block_ids': [direct] if direct is not None else [],
                        'route': 'unique_structural_owner' if direct is not None else 'bm25_fallback',
                        'reason': reason,
                        'decorated_source': decorated if type(decorated) is bool else None,
                        'scope': 'raw source parameter field ownership only; decorators not executed, not runtime documentation',
                        'rankings': {'bm25': deepcopy(bm25), 'structure_bm25': primary},
                        'supplementary_rankings': {'semantic': deepcopy(rankings['semantic'])}
                            if 'semantic' in rankings else {}})
    structural = sum(row['route'] == 'unique_structural_owner' for row in records)
    return {'schema_version': 3, 'records': records,
            'summary': {'total': len(records), 'structural': structural, 'fallback': len(records) - structural,
                        'reasons': dict(Counter(row['reason'] for row in records))},
            'policy': {'method': 'unique_numpy_owner_then_bm25_v3',
                       'owner_evidence': 'literal.files functions and complete original parameter blocks only',
                       'identity': ['exact path', 'complete symbol', 'explicit parameter'],
                       'primary': 'unique supported owner first; all remaining BM25 positions retain relative order',
                       'scores': 'Original BM25 scores are retained; no structural or semantic score is invented.',
                       'semantic': 'Separate optional supplementary ranking; never changes the primary ranking.',
                       'scope': 'Original NumPy parameter fields only; no RST, expanded template or cross-file ownership inference.',
                       'interpretation': 'Exact owner lookup proves correspondence, not documentation correctness. High owner hit rates are by construction, not learned AI improvement.',
                       'correctness_labels_used': False, 'target_code_executed': False}}
