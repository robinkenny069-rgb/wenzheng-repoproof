"""Combine development evidence and keep assisted human decisions separate.

This module does not run detectors or promote machine output into gold labels.
Review packets are content-addressed; exports are self reports, not signatures.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path


STATUS = {
    'declared_default_mismatch_candidate': 'candidate',
    'default_mismatch_candidate': 'candidate',
    'consistent_declared_value': 'consistent',
    'consistent_with_declared_default': 'consistent',
    'abstain': 'uncertain',
}
DECISIONS = {'pending', 'mismatch', 'consistent', 'uncertain'}
RECORD_KIND = 'assisted_development_review_self_report'


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def identity(row):
    return row.get('symbol', row.get('class')), row['parameter']


def summary(rows):
    counts = Counter(row['machine_status'] for row in rows)
    return {name: counts[name] for name in ('candidate', 'consistent', 'uncertain')}


def build_packet(source_result: dict, contract_result: dict, presentation_hashes=None):
    if source_result['case_id'] != 'matplotlib-14475' or contract_result['case_id'] != 'socketio-1255':
        raise ValueError('Only the two assigned development events are allowed in this workbench')
    if source_result['phase'] != 'development':
        raise ValueError('Expected development evidence')
    rows, cases = [], []
    specs = [
        ('Matplotlib', source_result, 'https://github.com/matplotlib/matplotlib/pull/14475'),
        ('Socket.IO / Engine.IO', contract_result, contract_result['upstream_url']),
    ]
    for title, result, upstream in specs:
        case_rows = []
        after_index = {}
        for row in result['runs']['after']['rows']:
            after_index.setdefault(identity(row), []).append(row)
        before_keys = Counter(identity(row) for row in result['runs']['before']['rows'])
        for row in result['runs']['before']['rows']:
            key = identity(row)
            matches = after_index.get(key, [])
            paired = matches[0] if len(matches) == 1 and before_keys[key] == 1 else None
            item = {
                'case_id': result['case_id'], 'project': title,
                'symbol': key[0], 'parameter': key[1],
                'machine_status': STATUS[row['status']],
                'after_status': STATUS[paired['status']] if paired else 'unpaired',
                'pairing': 'unique_symbol_parameter' if paired else 'not_unique_or_missing',
                'before': row, 'after': paired,
            }
            item['item_id'] = result['case_id'] + ':' + sha(item)[:20]
            case_rows.append(item)
        rows.extend(case_rows)
        cases.append({
            'case_id': result['case_id'], 'title': title, 'upstream_url': upstream,
            'rows_before': len(case_rows), 'counts_before': summary(case_rows),
            'counts_after': dict(Counter(STATUS[row['status']] for row in result['runs']['after']['rows'])),
            'rows_after': len(result['runs']['after']['rows']),
            'source_manifest_sha256': result['source_manifest_sha256'],
            'source_sha256': result['source_sha256'], 'code_sha256': result['code_sha256'],
        })
    if len({row['item_id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate evidence identities')
    probe = source_result['retrieval_probe']
    neural = source_result['model_called']
    retrieval = {'model_called': neural}
    if neural:
        retrieval.update({key: probe[key] for key in (
            'corpus_blocks', 'total_queries', 'unique_reference_queries',
            'queries_without_unique_reference', 'summaries', 'model',
            'corpus_sha256', 'query_sha256', 'structural_lookup_hits',
        )})
        gains = losses = 0
        for record in probe['records']:
            if record['reference_status'] != 'unique_owner_parameter_mapping':
                continue
            ranks = record['first_reference_rank']
            hit = lambda method: ranks[method] is not None and ranks[method] <= 3
            gains += hit('semantic') and not hit('lexical')
            losses += hit('lexical') and not hit('semantic')
        retrieval['semantic_vs_lexical_top3'] = {'gains': gains, 'losses': losses}
        if 'bm25' in probe:
            retrieval['bm25'] = probe['bm25']
            comparison = {'gains':0, 'losses':0}
            for record in probe['records']:
                if record['reference_status'] != 'unique_owner_parameter_mapping':
                    continue
                ranks = record['first_reference_rank']
                hit = lambda method: ranks[method] is not None and ranks[method] <= 3
                comparison['gains'] += hit('semantic') and not hit('bm25')
                comparison['losses'] += hit('bm25') and not hit('semantic')
            retrieval['semantic_vs_bm25_top3'] = comparison
    packet = {
        'schema_version': 1, 'packet_kind': 'development_workbench_evidence',
        'phase': 'development', 'review_scope': 'before snapshot; after shown as upstream repair context',
        'model_predictions_shown': True, 'independent_gold': False,
        'cases': cases, 'items': rows, 'counts_before': summary(rows), 'retrieval': retrieval,
        'presentation_sha256': presentation_hashes or {},
        'limits': [
            'Two known development events; three candidate fields do not mean three independent defects.',
            'Static declaration comparison, not runtime behavior proof. Cross-module tracing assumes the explicit source mapping.',
            'All recognized before fields are retained, including unsupported and required parameters.',
            'Retrieval relevance is separate from inconsistency evidence; structural references are not defect labels.',
            'Human review is assisted, self-reported and may be partial. It is not blind independent evaluation.',
        ],
    }
    packet['packet_sha256'] = sha(packet)
    return packet


def verify_packet(packet):
    content = {key: value for key, value in packet.items() if key != 'packet_sha256'}
    if packet.get('packet_sha256') != sha(content):
        raise ValueError('Evidence packet content does not match its hash')


def validate_review(record, packet):
    """Validate a partial assisted record. Never declare the author verified."""
    verify_packet(packet)
    if record.get('schema_version') != 1 or record.get('record_kind') != RECORD_KIND:
        raise ValueError('Not an assisted development review self report')
    if record.get('packet_sha256') != packet['packet_sha256']:
        raise ValueError('Review belongs to another evidence or program version')
    if record.get('model_predictions_shown') is not True or record.get('independent_gold') is not False:
        raise ValueError('Assisted review cannot be represented as blind gold labels')
    if record.get('participant') != 'current_operator':
        raise ValueError('This single-operator workflow records only the current operator')
    judgments = record.get('judgments')
    if not isinstance(judgments, list):
        raise ValueError('Missing judgments')
    expected = {row['item_id'] for row in packet['items']}
    seen = set()
    for row in judgments:
        if not isinstance(row, dict) or row.get('item_id') not in expected or row['item_id'] in seen:
            raise ValueError('Unknown or duplicate review item')
        seen.add(row['item_id'])
        decision, reason = row.get('decision'), row.get('reason')
        if decision not in DECISIONS or not isinstance(reason, str) or len(reason) > 3000:
            raise ValueError('Invalid decision or explanation')
        if decision != 'pending' and not reason.strip():
            raise ValueError('A human decision needs an explanation; uncertainty is allowed')
    if seen != expected:
        raise ValueError('Unreviewed items must remain explicitly pending')
    counts = Counter(row['decision'] for row in judgments)
    return {
        'record_kind': RECORD_KIND, 'participant_identity_verified': False,
        'independent_gold': False, 'total': len(expected),
        'reviewed': len(expected) - counts['pending'],
        'decisions': {name: counts[name] for name in sorted(DECISIONS)},
        'scope': 'Assisted review completeness, not accuracy or proof of human authorship.',
    }


def render_workbench(packet, template: Path):
    verify_packet(packet)
    # JSON is embedded as data, not executable source. Escape HTML parser delimiters.
    embedded = canonical(packet).replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
    return template.read_text(encoding='utf-8').replace('__PACKET_JSON__', embedded)
