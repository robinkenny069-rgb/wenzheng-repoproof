"""Build source-grounded review packets, without predictions or answer labels."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import re

DECISIONS = {'consistent', 'inconsistent', 'insufficient'}
SELECTIONS = [
    ('R01', 'matplotlib-14475', 'before', 'lib/matplotlib/figure.py', 'Figure.ginput', 'show_clicks'),
    ('R02', 'sklearn-34291', 'after', 'sklearn/cluster/_mean_shift.py', 'estimate_bandwidth', 'random_state'),
    ('R03', 'matplotlib-14475', 'before', 'lib/matplotlib/figure.py', 'Figure.ginput', 'timeout'),
    ('R04', 'sklearn-34291', 'before', 'sklearn/cluster/_mean_shift.py', 'estimate_bandwidth', 'random_state'),
    ('R05', 'matplotlib-14475', 'after', 'lib/matplotlib/figure.py', 'Figure.ginput', 'show_clicks'),
    ('R06', 'sklearn-34291', 'before', 'sklearn/cluster/_mean_shift.py', 'estimate_bandwidth', 'quantile'),
]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(data: dict) -> bytes:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')


def excerpt(text: str, symbol: str, parameter: str) -> dict:
    """Select exact numbered source lines, not rewritten or rendered docstrings."""
    node = ast.parse(text)
    for name in symbol.split('.'):
        matches = [child for child in node.body if isinstance(child, (ast.ClassDef, ast.FunctionDef)) and child.name == name]
        if len(matches) != 1:
            raise ValueError(f'Symbol missing or ambiguous: {symbol}')
        node = matches[0]
    if not isinstance(node, ast.FunctionDef) or not node.body or not isinstance(node.body[0], ast.Expr) or not isinstance(node.body[0].value, ast.Constant) or not isinstance(node.body[0].value.value, str):
        raise ValueError('Expected a function with a literal docstring')
    doc = node.body[0].value
    lines = text.splitlines()
    matches = [i for i in range(doc.lineno-1, doc.end_lineno) if re.match(r'^\s*' + re.escape(parameter) + r'\s*:', lines[i])]
    if len(matches) != 1:
        raise ValueError(f'Parameter field missing or ambiguous: {parameter}')
    start = matches[0]
    indent = len(lines[start])-len(lines[start].lstrip())
    end = start + 1
    while end < doc.end_lineno:
        line = lines[end]
        if line.strip() and len(line)-len(line.lstrip()) <= indent:
            break
        end += 1
    while end > start+1 and not lines[end-1].strip():
        end -= 1
    def numbered(first, last):
        return [{'line': i+1, 'text': lines[i]} for i in range(first-1, last)]
    return {'signature': numbered(node.lineno, doc.lineno-1),
            'document': numbered(start+1, end),
            'decorated': bool(node.decorator_list),
            'decorators': numbered(min(n.lineno for n in node.decorator_list), node.lineno-1) if node.decorator_list else []}


def build_packet(root: Path) -> dict:
    manifests = {}
    items = []
    for item_id, case_id, version, path, symbol, parameter in SELECTIONS:
        case_root = root/'research/cases'/case_id
        if case_id not in manifests:
            manifest_path = case_root/'manifest.json'
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            if manifest['status'] != 'complete':
                raise ValueError(f'Source collection incomplete: {case_id}')
            for relative, meta in manifest['files'].items():
                if digest((case_root/relative).read_bytes()) != meta['sha256']:
                    raise ValueError(f'Source hash mismatch: {case_id}/{relative}')
            manifests[case_id] = manifest
        manifest = manifests[case_id]
        relative = f'{version}/{path}'
        source = (case_root/relative).read_bytes()
        selected = excerpt(source.decode('utf-8'), symbol, parameter)
        items.append({'id': item_id, 'case_id': case_id, 'repository': manifest['repository'],
                      'commit': manifest[version], 'source_path': path, 'symbol': symbol, 'parameter': parameter,
                      'source_sha256': digest(source),
                      'source_url': f'https://github.com/{manifest["repository"]}/blob/{manifest[version]}/{path}',
                      'license_relative': f'../../research/cases/{case_id}/LICENSE.upstream', **selected})
    payload = {'schema_version': 1, 'packet_id': 'natural-defaults-review-v1',
               'scope': 'Compare documented default with declared signature default; no runtime proof.',
               'sampling': 'Convenience sample from known fixes; paired versions and unchanged controls, not representative prevalence.',
               'model_predictions_included': False, 'reference_answers_included': False,
               'items': items,
               'source_manifest_sha256': {cid: digest((root/'research/cases'/cid/'manifest.json').read_bytes()) for cid in manifests}}
    return {**payload, 'packet_sha256': digest(canonical(payload))}


def validate_record(record: dict, packet: dict) -> None:
    if record.get('schema_version') != 1 or record.get('packet_id') != packet['packet_id'] or record.get('packet_sha256') != packet['packet_sha256']:
        raise ValueError('Review packet identity does not match')
    if record.get('participant') not in ('A', 'B'):
        raise ValueError('Participant must be A or B')
    if record.get('prior_exposure') not in ('no', 'yes', 'unsure'):
        raise ValueError('Prior exposure must be explicitly reported')
    if record.get('record_kind') != 'human_source_review_self_report' or record.get('model_predictions_shown') is not False:
        raise ValueError('Not an eligible source review export')
    if not isinstance(record.get('source_links_opened'), list) or not set(record['source_links_opened']).issubset({item['id'] for item in packet['items']}):
        raise ValueError('Invalid source-link exposure record')
    rows = record.get('judgments', [])
    expected = {item['id'] for item in packet['items']}
    if len(rows) != len(expected) or {row.get('item_id') for row in rows} != expected:
        raise ValueError('Review has missing, duplicate, or unknown items')
    for row in rows:
        if row.get('decision') not in DECISIONS:
            raise ValueError('Invalid decision')
        if not isinstance(row.get('reason'), str) or len(row['reason'].strip()) < 4:
            raise ValueError('Provide a reason for each judgment')
        if not isinstance(row.get('evidence'), str) or not row['evidence'].strip():
            raise ValueError('Provide an evidence location or explain missing evidence')
