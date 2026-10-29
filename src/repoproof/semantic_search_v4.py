"""Interactive English behavior lookup over traceable v3 document evidence.

This is a new query task, not a replacement or relabeling of the frozen source
identity experiments. Search never changes the attached static evidence verdicts.
All rankers use the same complete-text pool, including verified template fields.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import math
import re
from threading import RLock
from time import perf_counter

from .bm25 import BM25Index, tokens
from .retrieval_router_v3 import has_unexpanded_placeholder
from .semantic import SemanticRetriever, SemanticUnavailable, MODEL_ID, REVISION


# A fixed auxiliary baseline, chosen before the new query evaluation. Keep
# negation, quantities, domain terms and all identifiers; no reference tuning.
STOPWORDS = frozenset('a an the is are was were be been being to of in for and or with by on at as do does did how can i which what this that it its my your from'.split())
LIMITS = [
    '英文用途检索；当前模型的中文效果未验证。',
    '候选相似不代表说明错误；附带证据是原v3静态核对结果，搜索不会改变判断。',
    '这是当前选定文件的局部语料，不代表项目全部文档；没有相关答案时仍可能返回相似候选。',
    '未展开模板及不完整结构不进入本检索池；所有方法使用同一可追溯正文。',
    '模型最长256个token；超长输入可能被截断。',
]


def digest(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()


def _normal(text):
    return ' '.join(text.split())


def _filtered(text):
    return ' '.join(t for t in tokens(text) if t not in STOPWORDS)


def _span(span):
    if (not isinstance(span, dict) or not isinstance(span.get('path'), str)
        or type(span.get('start_line')) is not int or type(span.get('end_line')) is not int
        or span['start_line'] < 1 or span['end_line'] < span['start_line']):
        raise ValueError('Invalid source span in search document')
    return deepcopy(span)


def build_pool(run):
    """Index supported original prose and bounded expanded template parameters.

    No evidence status, changed value, reference answer or query is used to
    select or represent a document. Exact duplicate fields retain all aliases.
    """
    originals = {b['block_id']: b for f in run['literal']['files'] for b in f['blocks']}
    # The old ranking packet intentionally strips RST owner metadata. Recover
    # it from the parser's complete block inventory, never from claim verdicts.
    for block in run.get('rst', {}).get('blocks', []):
        if block['block_id'] in originals and originals[block['block_id']].get('text') != block.get('text'):
            raise ValueError('Conflicting parser source blocks')
        originals.setdefault(block['block_id'], block)
    rows = {}
    for row in run.get('display_rows', []):
        rows.setdefault(row.get('block_id'), []).append(row)
    docs, excluded, seen, identities = [], [], {}, set()

    def add(block, origin):
        text = block.get('text')
        block_id = block.get('block_id')
        if not isinstance(block_id, str) or not block_id:
            raise ValueError('Missing source block identity')
        identity = origin + ':' + block_id
        if identity in identities:
            raise ValueError('Duplicate search input identity')
        identities.add(identity)
        if not isinstance(text, str) or not text.strip() or block.get('incomplete'):
            excluded.append({'block_id': block_id, 'reason': 'empty_or_incomplete', 'origin': origin})
            return
        if has_unexpanded_placeholder(text):
            excluded.append({'block_id': block_id, 'reason': 'unexpanded_template', 'origin': origin})
            return
        if origin == 'original' and block.get('kind') != 'prose':
            excluded.append({'block_id': block_id, 'reason': 'not_prose', 'origin': origin})
            return
        path = block.get('path')
        if not isinstance(path, str) or not path:
            raise ValueError('Missing document source path')
        original = originals.get(block_id, {})
        symbol = block.get('symbol') or original.get('symbol') or block.get('scope') or original.get('scope')
        parameters = block.get('parameters', original.get('parameters', []))
        parameter = block.get('parameter') or original.get('parameter')
        if not parameters and parameter:
            parameters = [parameter]
        if not isinstance(parameters, list) or any(not isinstance(x, str) for x in parameters):
            raise ValueError('Invalid parameter inventory')
        headings = block.get('headings', [symbol] if symbol else [])
        if not isinstance(headings, list) or any(not isinstance(x, str) for x in headings):
            raise ValueError('Invalid search headings')
        if origin == 'expanded_template':
            spans = [_span(s) for s in block.get('source_spans', [])]
            start, end = block.get('expanded_start_line'), block.get('expanded_end_line')
            if not spans or type(start) is not int or type(end) is not int or start < 1 or end < start:
                raise ValueError('Expanded field requires separate source and expansion coordinates')
            physical_start = physical_end = None
        else:
            spans = [_span(block)]
            spans = [{k: s[k] for k in ('path', 'start_line', 'end_line')} for s in spans]
            physical_start, physical_end = block['start_line'], block['end_line']
            start = end = None
        key = (path, symbol, tuple(parameters), _normal(text))
        # Unowned prose remains separate; identical fragments in unrelated
        # contexts must not be incorrectly combined into a single owner.
        if not symbol:
            key = (*key, identity)
        if key in seen:
            seen[key]['aliases'].append(block_id)
            return
        doc = {'document_id': identity, 'block_id': block_id, 'aliases': [block_id],
               'text': text, 'path': path, 'start_line': physical_start, 'end_line': physical_end,
               'headings': headings, 'symbol': symbol, 'parameters': parameters,
               'origin': origin, 'source_spans': spans,
               'expanded_start_line': start, 'expanded_end_line': end,
               'coordinate_system': 'expanded_docstring' if origin == 'expanded_template' else 'physical_source',
               'retrieval_text': '\n'.join([*headings, text])}
        seen[key] = doc
        docs.append(doc)

    for block in run['retrieval']['documents']:
        add(block, 'original')
    for block in run.get('templates', {}).get('blocks', []):
        add(block, 'expanded_template')
    if len(docs) > 20000:
        raise ValueError('Search pool exceeds the local 20000-document limit')
    pool_hash = digest(docs)
    # Attach verdicts only after computing the search representation. They are
    # display evidence, never rank features or query-selection labels.
    for doc in docs:
        evidence = [row for alias in doc['aliases'] for row in rows.get(alias, [])]
        doc['evidence'] = list({row['record_id']: deepcopy(row) for row in evidence}.values())
    return {'documents': docs, 'excluded': excluded, 'pool_hash': pool_hash,
            'expanded_documents': sum(d['origin'] == 'expanded_template' for d in docs)}


class SearchIndex:
    """Read-only in-memory index; optional pinned ONNX model, no target execution."""

    def __init__(self, payload, *, with_semantic=True):
        if not isinstance(payload, dict) or not isinstance(payload.get('cases'), list):
            raise ValueError('A v3 review packet with cases is required')
        self.packet_sha256 = payload.get('packet_sha256')
        self.input_digest = digest(payload)
        self.pools, self.bm25, self.bm25_stopwords, self.vectors = {}, {}, {}, {}
        self.source_identities = {}
        self.encoder = None
        self.lock = RLock()
        self.model = {'requested': bool(with_semantic), 'status': 'not_loaded' if with_semantic else 'disabled',
                      'model_id': MODEL_ID, 'revision': REVISION, 'actual_encode_completed': False}
        for case in payload['cases']:
            case_id = case.get('case_id')
            if not isinstance(case_id, str) or not case_id:
                raise ValueError('case_id must be a nonempty string')
            if case_id in self.source_identities or not isinstance(case.get('source_identity', {}), dict):
                raise ValueError('Duplicate case or invalid source identity')
            self.source_identities[case_id] = {
                'source_identity': deepcopy(case.get('source_identity', {})),
                'source_manifest_sha256': case.get('source_manifest_sha256')}
            for stage, run in case['runs'].items():
                if stage not in ('before', 'after') or (case_id, stage) in self.pools:
                    raise ValueError('Unknown or duplicate case snapshot')
                key = (case_id, stage)
                pool = self.pools[key] = build_pool(run)
                texts = [d['retrieval_text'] for d in pool['documents']]
                self.bm25[key] = BM25Index(texts)
                self.bm25_stopwords[key] = BM25Index([_filtered(t) for t in texts])

    def catalog(self):
        cases = []
        for case_id in dict.fromkeys(c for c, _ in self.pools):
            stages = [stage for case, stage in self.pools if case == case_id]
            cases.append({'case_id': case_id, 'snapshots': stages,
                          **deepcopy(self.source_identities[case_id]), 'pools': {
                stage: {'documents': len(self.pools[(case_id, stage)]['documents']),
                        'excluded': len(self.pools[(case_id, stage)]['excluded']),
                        'expanded_documents': self.pools[(case_id, stage)]['expanded_documents'],
                        'pool_hash': self.pools[(case_id, stage)]['pool_hash']} for stage in stages}})
        return {'cases': cases, 'model': deepcopy(self.model), 'limits': LIMITS,
                'packet_sha256': self.packet_sha256, 'input_digest': self.input_digest}

    def export_corpus(self):
        return {'schema_version': 4, 'packet_sha256': self.packet_sha256,
                'input_digest': self.input_digest,
                'pools': [{'case_id': case_id, 'snapshot': stage,
                           **{k: v for k, v in pool.items() if k != 'documents'},
                           'documents': [{k: v for k, v in d.items() if k != 'evidence'} for d in pool['documents']]}
                          for (case_id, stage), pool in self.pools.items()],
                'selection': 'complete supported prose and bounded expanded template fields; no status or labels',
                'limits': LIMITS}

    def search(self, query, case_id, snapshot='after', k=6):
        if (not isinstance(query, str) or not query.strip() or len(query) > 600
            or any(ord(c) < 32 and c not in '\t\n\r' for c in query)):
            raise ValueError('Query must contain 1–600 characters and no control codes')
        if type(k) is not int or not 1 <= k <= 20:
            raise ValueError('k must be an integer from 1 through 20')
        if not isinstance(case_id, str) or not isinstance(snapshot, str) or (case_id, snapshot) not in self.pools:
            raise ValueError('Unknown case or snapshot')
        started, key = perf_counter(), (case_id, snapshot)
        pool = self.pools[key]
        documents = pool['documents']
        ranks = {'bm25': self.bm25[key].rank(query),
                 'bm25_stopwords': self.bm25_stopwords[key].rank(_filtered(query))}
        timing = {'document_encoding_seconds': 0.0, 'query_encoding_seconds': 0.0,
                  'model_initialization_seconds': 0.0, 'document_vectors_reused': False,
                  'documents_encoded_this_request': 0, 'queries_encoded_this_request': 0}
        model = deepcopy(self.model)
        if self.model['requested'] and documents:
            with self.lock:
                try:
                    phase = perf_counter()
                    if self.encoder is None:
                        self.encoder = SemanticRetriever()
                        self.model.update(status='loaded', model_file_sha256=self.encoder.manifest['files']['onnx/model.onnx']['sha256'])
                    timing['model_initialization_seconds'] = perf_counter() - phase
                    phase = perf_counter()
                    if key not in self.vectors:
                        self.vectors[key] = self.encoder.encode([d['retrieval_text'] for d in documents])
                        timing['documents_encoded_this_request'] = len(documents)
                    else:
                        timing['document_vectors_reused'] = True
                    timing['document_encoding_seconds'] = perf_counter() - phase
                    phase = perf_counter()
                    qv = self.encoder.encode([query])[0]
                    timing['query_encoding_seconds'] = perf_counter() - phase
                    scores = self.vectors[key] @ qv
                    if len(scores) != len(documents) or any(not math.isfinite(float(s)) for s in scores):
                        raise SemanticUnavailable('Nonfinite or mismatched model scores')
                    ranks['semantic'] = sorted([(i, float(score)) for i, score in enumerate(scores)], key=lambda x: (-x[1], x[0]))
                    timing['queries_encoded_this_request'] = 1
                    self.model.update(status='completed', actual_encode_completed=True)
                    self.model.pop('error', None)
                    model = deepcopy(self.model)
                except (SemanticUnavailable, RuntimeError, ValueError, OSError) as error:
                    # Preserve BM25 results with an explicit unsuccessful model
                    # status. Never replace neural scores with lexical scores.
                    self.model.update(status='unavailable', actual_encode_completed=False, error=str(error))
                    model = deepcopy(self.model)
        elif self.model['requested'] and not documents:
            model.update(status='not_called_empty_corpus', actual_encode_completed=False)
        methods = {}
        for method, ranking in ranks.items():
            methods[method] = [{'rank': rank, 'score': score,
                                'score_method': 'cosine' if method == 'semantic' else method,
                                'document': deepcopy(documents[i])}
                               for rank, (i, score) in enumerate(ranking[:k], 1)]
        for missing in ('semantic', 'hybrid_supplement', 'rrf'):
            methods.setdefault(missing, [])
        if 'semantic' in ranks:
            # Fixed balanced candidate budget: ceil(k/2) BM25, then distinct
            # semantic candidates until the same k budget is reached.
            primary_count = (k + 1) // 2
            hybrid = [(i, score, 'bm25') for i, score in ranks['bm25'][:primary_count]]
            used = {i for i, _, _ in hybrid}
            for i, score in ranks['semantic']:
                if i not in used and len(hybrid) < k:
                    hybrid.append((i, score, 'semantic')); used.add(i)
            methods['hybrid_supplement'] = [{'rank': rank, 'score': score,
                    'score_method': 'bm25' if origin == 'bm25' else 'cosine', 'selected_by': origin,
                    'document': deepcopy(documents[i])} for rank, (i, score, origin) in enumerate(hybrid, 1)]
            fused = {}
            for method in ('bm25', 'semantic'):
                for rank, (i, _) in enumerate(ranks[method], 1):
                    fused[i] = fused.get(i, 0.0) + 1.0 / (60 + rank)
            methods['rrf'] = [{'rank': rank, 'score': score, 'score_method': 'rrf_k60_equal_weights',
                              'document': deepcopy(documents[i])}
                             for rank, (i, score) in enumerate(sorted(fused.items(), key=lambda x: (-x[1], x[0]))[:k], 1)]
        timing['total_seconds'] = perf_counter() - started
        provenance = self.source_identities[case_id]
        return {'schema_version': 4, 'query': query, 'case_id': case_id, 'snapshot': snapshot, 'k': k,
                'source_identity': deepcopy(provenance['source_identity'].get(snapshot, [])),
                'source_manifest_sha256': provenance['source_manifest_sha256'],
                'pool_hash': pool['pool_hash'], 'packet_sha256': self.packet_sha256, 'input_digest': self.input_digest,
                'documents_in_pool': len(documents), 'model': model, 'timing': timing, 'rankings': methods,
                'limits': LIMITS, 'automatic_verdict': False, 'target_code_executed': False,
                'purpose': 'interactive behavior lookup; not an automatic answer or independent benchmark'}
