"""Source ownership first, unchanged BM25 fallback, separate neural suggestions.

Reuses v2 evidence checks unchanged. A retrieval route never changes a verdict.
"""
from copy import deepcopy

from .product_v2 import analyze_case as analyze_v2_case, render_product as render_v2
from .narrative_retrieval import rank_documents
from .retrieval_router_v3 import route_queries, has_unexpanded_placeholder


def attach_routes(run, *, with_semantic=True):
    """Add routes using source structure; no reference labels enter ranking."""
    router = route_queries(run)
    by_id = {r['query_id']: r for r in router['records']}
    unresolved = [q for q in run['literal']['queries'] if by_id[q['query_id']]['route'] == 'bm25_fallback']
    probe = run['retrieval']
    if with_semantic and unresolved and probe['corpus']:
        neural = rank_documents(probe['documents'], [{'query_id':q['query_id'],'text':q['text']} for q in unresolved])
        if not neural['model']['actual_encode_completed']:
            raise RuntimeError('Requested semantic supplement did not encode; no successful AI run is claimed')
        extras = {r['query_id']:r for r in neural['records']}
        probe['model'] = neural['model']
        probe['unavailable_methods'] = ['lexical_semantic_rrf','bm25_semantic_rrf']
        probe['semantic_supplement_timing'] = neural['timing']
        for record in probe['records']:
            extra = extras.get(record['query_id'])
            if extra:
                if extra['rankings']['bm25'] != record['rankings']['bm25']:
                    raise ValueError('BM25 changed between baseline and semantic supplement')
                # No RRF in the product recommendation; original v2 comparisons remain saved.
                record['rankings']['semantic'] = extra['rankings']['semantic']
    else:
        probe['model'].update(requested=bool(with_semantic), status='skipped_all_queries_have_direct_owner' if with_semantic and not unresolved else probe['model']['status'])
    for record in probe['records']:
        route = by_id[record['query_id']]
        record['rankings']['structure_bm25'] = route['rankings']['structure_bm25']
        primary_ids = {r['block_id'] for r in record['rankings']['structure_bm25'][:3]}
        # A distinct supplementary list, never an error verdict or an equal-budget superiority claim.
        record['semantic_supplement'] = [r for r in record['rankings'].get('semantic',[]) if r['block_id'] not in primary_ids][:3]
        record['semantic_status'] = ('completed' if 'semantic' in record['rankings'] else
                                     'skipped_direct_owner' if route['route']=='unique_structural_owner' and with_semantic else
                                     'disabled' if not with_semantic else 'not_called_empty_corpus')
    probe['method_availability'] = {'bm25':'all queries', 'structure_bm25':'all queries',
                                    'semantic':'fallback queries only when requested and corpus nonempty',
                                    'rrf':'not run in v3 recommendation'}
    probe['timing_scope'] = 'timing measures initial non-neural baseline; semantic_supplement_timing separately measures fallback probe, not an end-to-end latency benchmark'
    router['execution'] = {'semantic_requested':bool(with_semantic), 'total_queries':len(by_id),
                           'queries_needing_semantic':len(unresolved),
                           'queries_encoded':probe['model'].get('encoded_query_count',0),
                           'queries_avoided_by_structure':len(by_id)-len(unresolved),
                           'neural_query_reduction_observed':len(by_id)-len(unresolved) if with_semantic and probe['model']['actual_encode_completed'] else None,
                           'semantic_supplement_display_budget':3, 'primary_comparison_budget':3,
                           'latency_speedup_claimed':False}
    run['retrieval_router'] = router
    run['retrieval_note'] = 'Unique raw-source ownership is placed first; BM25 order is otherwise unchanged. Neural suggestions are separate and run only for fallback queries. No static verdict is changed.'
    run['routing_evaluation'] = evaluate_after_routing(run)
    return run


def evaluate_after_routing(run):
    """Post-hoc owner correspondence, explicitly not independent relevance gold."""
    blocks = [b for file in run['literal']['files'] for b in file['blocks']]
    rows = {r['query_id']:r for r in run['retrieval_router']['records']}
    details=[]
    for query in run['literal']['queries']:
        candidates=[b for b in blocks if b['path']==query['path'] and b['symbol']==query['symbol'] and query['parameter'] in b['parameters']]
        target=candidates[0] if len(candidates)==1 else None
        result={'query_id':query['query_id'],'reference_status':'unique_raw_source_owner' if target else 'missing_or_ambiguous',
                'reference_block':target['block_id'] if target else None,
                'reference_has_placeholder':bool(target and has_unexpanded_placeholder(target['text'])),
                'ranks':{}}
        for method,ranking in rows[query['query_id']]['rankings'].items():
            result['ranks'][method]=next((r['rank'] for r in ranking if target and r['block_id']==target['block_id']),None)
        details.append(result)
    unique=[r for r in details if r['reference_status']=='unique_raw_source_owner']
    summary={method:{'reference_queries':len(unique),
                      'hits':{str(k):sum(d['ranks'][method] is not None and d['ranks'][method]<=k for d in unique) for k in (1,3,5)}}
             for method in ('bm25','structure_bm25')}
    fallback_unchanged=all(r['rankings']['bm25']==r['rankings']['structure_bm25']
                           or [x['block_id'] for x in r['rankings']['bm25']]==[x['block_id'] for x in r['rankings']['structure_bm25']]
                           for r in rows.values() if r['route']=='bm25_fallback')
    return {'purpose':'same-corpus source-owner correspondence; references are defined by structure, not independent semantic or defect labels',
            'total_queries':len(details),'unique_references':len(unique),
            'references_with_unexpanded_placeholder':sum(d['reference_has_placeholder'] for d in unique),
            'queries_without_unique_reference':len(details)-len(unique), 'summaries':summary,'details':details,
            'fallback_bm25_order_unchanged':fallback_unchanged,
            'independent_gold':False,'neural_superiority_claimed':False}


def analyze_case(case_id,before,after=None,*,with_semantic=True):
    result=analyze_v2_case(case_id,before,after,with_semantic=False)
    for run in result['runs'].values(): attach_routes(run,with_semantic=with_semantic)
    return result


def render_product(payload,template):
    view=deepcopy(payload)
    for case in view['cases']:
        for run in case['runs'].values():
            for row in run['retrieval_router']['records']:
                row['rankings']={m:ranking[:3] for m,ranking in row['rankings'].items()}
    return render_v2(view,template)
