"""Compare explicit parameter documentation with a static constructor trace."""
from __future__ import annotations
import ast
from decimal import Decimal
import re

from .contract_graph import ContractGraph, TraceUnavailable, literal
from .duration_evidence import UNITS

_FIELD = re.compile(r"^\s*:param\s+([A-Za-z_]\w*)\s*:\s*(.*)$")
_BOUNDARY = re.compile(r"^\s*:(?:param|type|return|returns|rtype|raises|keyword)\b")
_CLAIM = re.compile(r"\b(?:the\s+)?default(?:\s+value)?\s+is\s+(`{1,2})?(?P<value>None|True|False|[+-]?(?:\d+(?:,\d{3})*(?:\.\d+)?))`{0,2}(?!\w|\.[\w.]|,\d)", re.I)
_HISTORY = re.compile(r"\b(?:previously|formerly|old\s+default|used\s+to|version\s+\d)\b", re.I)


def parameter_docs(graph: ContractGraph, class_name: str) -> list[dict]:
    info = graph.classes[class_name]
    if not info.node.body or not isinstance(info.node.body[0], ast.Expr):
        return []
    expression = info.node.body[0]
    if not isinstance(expression.value, ast.Constant) or not isinstance(expression.value.value, str):
        return []
    text = expression.value.value
    raw = ast.get_source_segment(info.source.text, expression) or ''
    if text.count('\n') != raw.count('\n'):
        return []  # Do not fabricate source lines for escaped/newline-concatenated literals.
    lines = text.splitlines()
    found = []
    current = None
    for offset, line in enumerate(lines):
        match = _FIELD.match(line)
        if match or _BOUNDARY.match(line):
            if current:
                current['text'] = '\n'.join(current.pop('_lines')).rstrip()
                found.append(current)
            current = None
        if match:
            current = {"class": class_name, "parameter": match.group(1), "path": info.source.path,
                       "start_line": expression.lineno + offset, "end_line": expression.lineno + offset,
                       "_lines": [line.strip()]}
        elif current:
            current['_lines'].append(line.strip())
            current['end_line'] = expression.lineno + offset
    if current:
        current['text'] = '\n'.join(current.pop('_lines')).rstrip()
        found.append(current)
    return found


def claim_from_doc(doc: dict) -> dict:
    text = doc['text']
    if _HISTORY.search(text):
        return {"status": "abstain", "reason": "historical_document_claim"}
    matches = list(_CLAIM.finditer(text))
    if len(matches) != 1:
        return {"status": "abstain", "reason": "no_unique_explicit_default_claim"}
    match = matches[0]
    # Negation, approximate values and expressions are not exact scalar claims.
    prefix = text[max(0, text.rfind('.', 0, match.start()) + 1):match.start()]
    if re.search(r"\b(?:not|never|if|may|might|approximately|about)\b", prefix, re.I):
        return {"status": "abstain", "reason": "qualified_default_claim"}
    tail = text[match.end():]
    if re.match(r"\s*(?:[+*/%<>=-]|(?:or|and|to)\b)", tail):
        return {"status": "abstain", "reason": "compound_default_claim"}
    word = match.group('value').replace(',', '')
    word = {'true': 'True', 'false': 'False', 'none': 'None'}.get(word.lower(), word)
    try:
        value = literal(ast.parse(word, mode='eval').body)
    except (SyntaxError, TraceUnavailable):
        return {"status": "abstain", "reason": "unsupported_document_literal"}
    units = '|'.join(sorted(UNITS, key=len, reverse=True))
    near = re.match(r"\s+(" + units + r")\b", tail, re.I)
    remainder = tail[near.end():] if near else tail
    if remainder.strip() and not re.match(r"\s*[.;]", remainder):
        return {"status": "abstain", "reason": "unsupported_unit_or_claim_continuation"}
    stated = list(re.finditer(r"\bin\s+(" + units + r")\b", text, re.I))
    unit = near.group(1).lower() if near else (stated[0].group(1).lower() if len(stated) == 1 else None)
    if near and any(UNITS[item.group(1).lower()] != UNITS[unit] for item in stated):
        return {"status": "abstain", "reason": "conflicting_document_units"}
    return {"status": "parsed", "value": value, "value_type": type(value).__name__, "unit": unit,
            "text": match.group(), "line": doc['start_line'] + text[:match.start()].count('\n')}


def compare_parameter(graph: ContractGraph, doc: dict) -> dict:
    claim = claim_from_doc(doc)
    result = {"class": doc['class'], "parameter": doc['parameter'], "document": doc,
              "claim": claim, "status": "abstain", "reason": "", "automatic_edit": False}
    if claim['status'] != 'parsed':
        result['reason'] = claim['reason']
        return result
    contract = graph.resolve_default(doc['class'], doc['parameter'])
    result['contract'] = contract
    if contract['status'] != 'resolved':
        result['reason'] = contract['reason']
        return result
    unit = claim['unit']
    provider = contract['provider_class']
    code_value, doc_value = contract['value'], claim['value']
    if code_value is None and doc_value is not None:
        result['reason'] = 'sentinel_default_requires_behavior_review'
        return result
    if unit:
        providers = [item for item in parameter_docs(graph, provider) if item['parameter'] == doc['parameter']]
        if len(providers) != 1:
            result['reason'] = 'provider_unit_not_established'
            return result
        provider_claim = claim_from_doc(providers[0])
        provider_unit = provider_claim.get('unit')
        # Unit attribution comes from the provider's parameter field, not an assumed seconds convention.
        if provider_claim['status'] != 'parsed' or not provider_unit:
            result['reason'] = 'provider_unit_not_established'
            return result
        if type(code_value) not in (int, float) or type(doc_value) not in (int, float):
            result['reason'] = 'non_numeric_duration'
            return result
        code_seconds = Decimal(str(code_value)) * UNITS[provider_unit]
        doc_seconds = Decimal(str(doc_value)) * UNITS[unit]
        if not code_seconds.is_finite() or not doc_seconds.is_finite():
            result['reason'] = 'non_finite_duration'
            return result
        result['unit_evidence'] = {"provider_document": providers[0], "provider_unit": provider_unit,
                                   "document_unit": unit, "code_seconds": str(code_seconds), "document_seconds": str(doc_seconds)}
        equal = code_seconds == doc_seconds
    else:
        equal = type(code_value) is type(doc_value) and code_value == doc_value
    result['status'] = 'consistent_with_declared_default' if equal else 'default_mismatch_candidate'
    result['reason'] = 'document_and_traced_default_agree' if equal else 'document_and_traced_default_differ'
    return result


def review_class(graph: ContractGraph, class_name: str) -> list[dict]:
    docs = parameter_docs(graph, class_name)
    seen = set()
    rows = []
    for doc in docs:
        if doc['parameter'] in seen or sum(item['parameter'] == doc['parameter'] for item in docs) > 1:
            rows.append({"class": class_name, "parameter": doc['parameter'], "document": doc,
                         "status": "abstain", "reason": "duplicate_parameter_documentation", "automatic_edit": False})
        else:
            rows.append(compare_parameter(graph, doc))
        seen.add(doc['parameter'])
    return rows
