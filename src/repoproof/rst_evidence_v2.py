"""Bounded RST parameter evidence, additive to the frozen v1 implementation.

This module reads source text only. It neither executes Python nor interprets
RST directives. A recognized API directive establishes a *documented* owner;
only a unique direct source declaration can bind it to code. Source annotations
support a limited unit/sentinel comparison, never a runtime-behaviour proof.
"""
from __future__ import annotations

import ast
from collections import Counter
from decimal import Decimal
import hashlib
import json
import re

from .narrative_retrieval import split_document
from .source_audit import declared_parameters, scalar


_DIRECTIVE = re.compile(r'^(?P<indent> *)\.\.\s+(?P<kind>[\w:.-]+)::\s*(?P<body>.*)$')
_API_KINDS = {'class', 'function', 'method', 'py:class', 'py:function', 'py:method'}
_PARAM = re.compile(r'^\s*:param\s+(?P<label>[^:]+):\s*(?P<body>.*)$')
_IDENTIFIER = re.compile(r'^[A-Za-z_]\w*$')
_NAME = re.compile(r'^([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*(?:\(|$)')
_DEFAULT_WORD = re.compile(r'\bdefaults?\b', re.I)
_QUALIFIED = re.compile(r"\b(?:if|when|unless|except|previously|formerly|usually|sometimes|may|might|not|never|nothing|neither|nor|only|isn't|wasn't|cannot|can't)\b", re.I)
_QUOTED = re.compile(r'''(?<!\w)(?:'(?:\\.|[^'\\])*'|"(?:\\.|[^"\\])*")(?!\w)''')
_DEFAULT_PREFIX = r'(?:By default\s+|(?:The\s+)?default(?:\s+is\s+|\s*:\s*)|Defaults? to\s+)'
_NUM = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)'
_UNIT_WORD = r'(?:milliseconds?|seconds?|minutes?|hours?|days?)'
_FACTORS = {'millisecond': Decimal('.001'), 'second': Decimal(1),
            'minute': Decimal(60), 'hour': Decimal(3600), 'day': Decimal(86400)}


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode('utf-8')).hexdigest()


def _indent(line):
    return len(line) - len(line.lstrip(' '))


def _plain(text):
    # Formatting cleanup must not silently change literal defaults such as
    # 'a*b', 'a  b', or a string containing an RST backtick.
    quoted = {}
    def protect(match):
        token = '\x00quoted' + str(len(quoted)) + '\x00'
        while token in text:
            token += '\x00'
        quoted[token] = match[0]
        return token
    text = _QUOTED.sub(protect, text)
    text = re.sub(r':[A-Za-z_:]+:`([^`]+)`', r'\1', text)
    text = re.sub(r'\s+', ' ', text.replace('`', '').replace('*', '')).strip()
    for token, value in quoted.items():
        text = text.replace(token, value)
    return text


def _unit(word):
    return word.lower().rstrip('s')


def _stop_indented(lines, start, indent):
    """Exclusive end of an explicitly indented construct, ignoring blank lines."""
    stop = start + 1
    while stop < len(lines):
        if lines[stop].strip() and _indent(lines[stop]) <= indent:
            break
        stop += 1
    while stop > start + 1 and not lines[stop - 1].strip():
        stop -= 1
    return stop


def _parameter_parts(raw, path, start, stop, scope):
    """Split one complete field into paragraphs, preserving its full context.

    Nested directives/literal examples remain excluded blocks. The original
    field stays attached to every paragraph, so a local sentence never erases
    a parameter-wide condition from the human evidence view.
    """
    lines = [line.removesuffix('\r') for line in raw]
    match = _PARAM.match(lines[start])
    label = match['label'].strip().split()
    parameter = label[-1].lstrip('*') if label else ''
    valid_parameter = bool(_IDENTIFIER.fullmatch(parameter))
    context = '\n'.join(raw[start:stop])
    field_id = _digest({'path': path, 'scope': scope['scope'], 'start_line': start + 1,
                        'end_line': stop, 'text': context})
    continuation = [_indent(lines[k]) for k in range(start + 1, stop)
                    if lines[k].strip() and not _DIRECTIVE.match(lines[k])]
    body_indent = min(continuation, default=_indent(lines[start]) + 3)
    blocks = []
    i = start
    literal_indent = None
    while i < stop:
        if not lines[i].strip():
            i += 1
            continue
        begin = i
        directive = _DIRECTIVE.match(lines[i]) if i != start else None
        reason = None
        if directive:
            i = min(stop, _stop_indented(lines, i, len(directive['indent'])))
            reason = 'nested_directive_not_interpreted'
        elif literal_indent is not None and _indent(lines[i]) > literal_indent:
            i = min(stop, _stop_indented(lines, i, _indent(lines[i]) - 1))
            reason = 'literal_example_not_a_document_claim'
            literal_indent = None
        elif lines[i].lstrip().startswith(('>>>', '...', '|', '+--', '+==')):
            i += 1
            while i < stop and lines[i].strip():
                i += 1
            reason = 'example_or_table_not_a_document_claim'
        else:
            literal_indent = None
            i += 1
            while i < stop and lines[i].strip() and not _DIRECTIVE.match(lines[i]):
                i += 1
            if begin != start and _indent(lines[begin]) > body_indent:
                reason = 'nested_indented_structure_not_interpreted'
            if any(lines[k].rstrip().endswith('::') for k in range(begin, i)):
                literal_indent = _indent(lines[i - 1])
                reason = 'literal_introduction_not_a_complete_claim'
        body = '\n'.join(raw[begin:i])
        if begin == start:
            claim_text = match['body'] + ('\n' + '\n'.join(raw[start + 1:i]) if i > start + 1 else '')
        else:
            claim_text = body
        if not valid_parameter:
            reason = 'unsupported_parameter_field_name'
        record = {'path': path, 'start_line': begin + 1, 'end_line': i, 'text': body,
                  'headings': [scope['scope'], parameter], 'kind': 'other' if reason else 'prose',
                  'format': 'rst', 'scope': scope['scope'], 'directive_kind': scope['kind'],
                  'directive_line': scope['line'], 'parameter': parameter,
                  'field_id': field_id, 'field_context': context,
                  'context_start_line': start + 1, 'context_end_line': stop,
                  'claim_text': claim_text, 'parse_status': reason or 'scoped_parameter_paragraph'}
        record['block_id'] = _digest({key: record[key] for key in ('path', 'start_line', 'end_line', 'text', 'scope', 'parameter', 'kind')})
        if reason:
            record.update(incomplete=True, parse_notes=[reason])
        blocks.append(record)
    return blocks


def extract_rst_parameters(text, path):
    """Read only explicit class/function/method parameter fields.

    Unknown enclosing directives, tabs and directive-less fields are retained
    in the v1 splitter but not promoted to v2 claim-bearing paragraphs.
    """
    raw = text.split('\n')
    lines = [line.removesuffix('\r') for line in raw]
    frames = []
    blocks, excluded = [], []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if '\t' in line[:len(line)-len(line.lstrip())]:
            excluded.append({'path': path, 'line': i + 1, 'reason': 'tab_indentation_not_interpreted'})
            i += 1
            continue
        indent = _indent(line)
        while frames and indent <= frames[-1]['indent']:
            frames.pop()
        directive = _DIRECTIVE.match(line)
        if directive:
            kind = directive['kind']
            name = _NAME.match(directive['body'])
            parent = next((f for f in reversed(frames) if not f['opaque']), None)
            opaque = kind not in _API_KINDS or name is None or any(f['opaque'] for f in frames)
            scope = name[1] if name else ''
            short_kind = kind.split(':')[-1]
            if not opaque and parent and parent['kind'] == 'class' and '.' not in scope:
                scope = parent['scope'] + '.' + scope
            frames.append({'indent': indent, 'kind': short_kind, 'scope': scope,
                           'line': i + 1, 'opaque': opaque})
            if opaque:
                excluded.append({'path': path, 'line': i + 1, 'reason': 'unsupported_enclosing_directive', 'directive': kind})
            i += 1
            continue
        match = _PARAM.match(line)
        if match:
            scope = next((f for f in reversed(frames) if not f['opaque']), None)
            if scope is None or any(f['opaque'] for f in frames):
                excluded.append({'path': path, 'line': i + 1, 'reason': 'parameter_without_supported_api_scope'})
                i += 1
                continue
            stop = i + 1
            while stop < len(lines):
                current = lines[stop]
                if current.strip() and _indent(current) <= indent:
                    break
                stop += 1
            while stop > i + 1 and not lines[stop - 1].strip():
                stop -= 1
            blocks.extend(_parameter_parts(raw, path, i, stop, scope))
            i = max(stop, i + 1)
            continue
        # A literal example may contain apparent API directives. Keep it opaque.
        if line.rstrip().endswith('::'):
            stop = _stop_indented(lines, i, indent)
            if stop > i + 1:
                excluded.append({'path': path, 'line': i + 1, 'end_line': stop,
                                 'reason': 'literal_example_not_interpreted'})
                i = stop
                continue
        i += 1
    return {'blocks': blocks, 'excluded': excluded}


def _docstring_lines(source, node):
    if not node.body or not isinstance(node.body[0], ast.Expr):
        return []
    literal = node.body[0].value
    if not isinstance(literal, ast.Constant) or not isinstance(literal.value, str):
        return []
    raw = ast.get_source_segment(source, literal) or ''
    if literal.value.count('\n') != raw.count('\n') or any(c in literal.value for c in ('\r','\u2028','\u2029')):
        return []
    return [(literal.lineno + index, line) for index, line in enumerate(literal.value.split('\n'))]


def _annotation(source, nodes, parameter, path):
    notes = []
    for node in nodes:
        lines = _docstring_lines(source, node)
        min_indent = min((_indent(line) for _, line in lines[1:] if line.strip()), default=0)
        for i, (line_no, line) in enumerate(lines):
            if i and _indent(line) != min_indent:
                continue
            match = re.match(r'^\s*' + re.escape(parameter) + r'\s*[-:]\s*(.+)$', line)
            sphinx = _PARAM.match(line)
            if sphinx and sphinx['label'].split()[-1].lstrip('*') == parameter:
                content = sphinx['body']
            elif match:
                content = match[1]
            else:
                continue
            end = i + 1
            while end < len(lines) and lines[end][1].strip() and _indent(lines[end][1]) > _indent(line):
                content += ' ' + lines[end][1].strip()
                end += 1
            note = {'path': path, 'start_line': line_no, 'end_line': lines[end - 1][0],
                    'text': '\n'.join(row[1] for row in lines[i:end]), 'meaning': _plain(content)}
            # A matching name in Methods/Attributes/Examples is not evidence
            # about the constructor's same-named parameter. Do not infer a
            # section or example's semantic ownership from its indentation.
            section = None
            for heading in range(i):
                title = lines[heading][1].strip()
                underline = lines[heading + 1][1].strip()
                if title and re.fullmatch(r'[-=~^]{3,}', underline):
                    section = title
                elif re.fullmatch(r'(?:Methods|Attributes|Returns|Yields|Examples|Notes|Parameters|Other Parameters):', title, re.I):
                    section = title[:-1]
            unsupported_context = (section is not None and section.lower() not in ('parameters', 'other parameters'))
            nested = any(_PARAM.match(row[1]) or re.match(r'^\s*[A-Za-z_]\w*\s*[-:]\s*', row[1])
                         for row in lines[i + 1:end])
            example = bool(re.search(r'\b(?:example|examples|for instance)\b|\be\.g\.|::', content, re.I))
            units = {_unit(m[0]) for m in re.finditer(r'\b' + _UNIT_WORD + r'\b', note['meaning'], re.I)}
            negative = bool(_QUALIFIED.search(note['meaning']))
            unit_binding = re.match(r'^(?:(?:max(?:imum)?|min(?:imum)?|total)\s+)?' + _UNIT_WORD + r'\b', note['meaning'], re.I)
            safe_context = not (unsupported_context or nested or example or negative)
            note['none_means_forever'] = bool(re.search(r'\bNone\s+(?:(?:means|is|for)\s+)?(?:cached\s+)?forever[.;]?\s*$', note['meaning'])) and safe_context
            if len(units) == 1 and safe_context and unit_binding:
                note.update(status='parsed', unit=next(iter(units)), reason='unique_nonnegated_parameter_unit_annotation')
            else:
                reason = ('annotation_section_does_not_establish_parameter_ownership' if unsupported_context
                          else 'nested_or_example_annotation_not_interpreted' if nested or example
                          else 'ambiguous_or_qualified_unit_annotation' if len(units) != 1 or negative
                          else 'unit_mention_not_explicitly_bound_to_parameter')
                note.update(status='unsupported', reason=reason if units else 'no_explicit_time_unit_annotation')
            notes.append(note)
    if len(notes) == 1:
        return notes[0]
    return {'status': 'unsupported', 'reason': 'multiple_parameter_annotations' if notes else 'no_parameter_annotation',
            'notes': notes, 'none_means_forever': False}


def collect_source_facts(sources):
    facts, apis, errors = [], [], []
    for path, source in sorted(sources.items()):
        try:
            tree = ast.parse(source, filename=path)
        except (SyntaxError, ValueError) as error:
            errors.append({'path': path, 'reason': type(error).__name__, 'message': str(error)})
            continue

        def walk(statements, prefix='', conditional=False, parent_class=None, ancestor_decorated=False):
            for node in statements:
                if isinstance(node, ast.ClassDef):
                    owner = prefix + node.name
                    decorated = ancestor_decorated or bool(node.decorator_list)
                    apis.append({'path': path, 'scope': owner, 'kind': 'class', 'line': node.lineno,
                                 'conditional': conditional, 'decorated': decorated})
                    walk(node.body, owner + '.', conditional, node, decorated)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbol = prefix + node.name
                    kind = 'method' if parent_class else 'function'
                    owner = prefix.rstrip('.') if parent_class and node.name == '__init__' else symbol
                    api_kind = 'class' if parent_class and node.name == '__init__' else kind
                    decorated = ancestor_decorated or bool(node.decorator_list)
                    apis.append({'path': path, 'scope': symbol, 'kind': kind, 'line': node.lineno,
                                 'conditional': conditional, 'decorated': decorated})
                    for parameter, declaration in declared_parameters(source, node).items():
                        if parameter in ('self', 'cls'):
                            continue
                        nodes = [node] + ([parent_class] if parent_class and node.name == '__init__' else [])
                        fact = {'source_id': f'{path}::{symbol}::{parameter}::{node.lineno}',
                                'path': path, 'symbol': symbol, 'owner': owner, 'api_kind': api_kind,
                                'parameter': parameter, 'line': node.lineno, 'declaration': declaration,
                                'unit_annotation': _annotation(source, nodes, parameter, path),
                                'conditional': conditional, 'decorated': decorated,
                                'status': 'resolved' if declaration['status'] == 'parsed' and not conditional and not decorated else 'abstain'}
                        fact['reason'] = ('conditional_or_decorated_declaration' if conditional or decorated
                                          else declaration.get('reason', 'literal_direct_declaration'))
                        facts.append(fact)
                    # Local functions are not module/class API declarations.
                else:
                    for child in ast.iter_child_nodes(node):
                        if isinstance(child, ast.stmt):
                            walk([child], prefix, True, parent_class, ancestor_decorated)
                        elif isinstance(child, (ast.ExceptHandler, ast.match_case)):
                            walk(child.body, prefix, True, parent_class, ancestor_decorated)
        walk(tree.body)
    return {'source_facts': facts, 'apis': apis, 'parse_errors': errors}


def _claim(sentence):
    """Recognize a small explicit grammar; unsupported default text stays visible."""
    result = {'status': 'unsupported', 'text': sentence, 'reason': 'unsupported_explicit_default_grammar'}
    if len(_DEFAULT_WORD.findall(sentence)) != 1:
        return {**result, 'reason': 'multiple_defaults_in_one_statement'}
    if _QUALIFIED.search(sentence):
        return {**result, 'reason': 'qualified_or_negated_default_statement'}
    # A by-default prefix permits a simple subject/predicate before forever.
    if re.match(r'^By default\b', sentence, re.I) and re.search(r'\bforever\b', sentence, re.I):
        # Additional numeric values or alternatives cannot be silently ignored.
        clause = sentence.split(',', 1)[0].rstrip('.;: ')
        if re.search(r'\b(?:or|otherwise|instead|but)\b|\d', sentence, re.I) or re.search(r'\band\b', clause, re.I):
            return {**result, 'reason': 'ambiguous_forever_default_clause'}
        if re.fullmatch(r'By default(?: [A-Za-z][A-Za-z _-]*)? forever', clause, re.I):
            return {'status': 'parsed', 'kind': 'forever', 'value': 'unbounded', 'text': sentence,
                    'reason': 'explicit_positive_forever_default'}
    # Only complete numeric-duration default statements are accepted.
    match = re.fullmatch(_DEFAULT_PREFIX + '(' + _NUM + r')\s+(' + _UNIT_WORD + r')[.;]?', sentence, re.I)
    if match:
        number = scalar(match[1])
        if number['status'] == 'parsed':
            return {'status': 'parsed', 'kind': 'duration', 'value': number['value'], 'unit': _unit(match[2]),
                    'text': sentence, 'reason': 'explicit_numeric_duration_default'}
    match = re.fullmatch(_DEFAULT_PREFIX + r'(.+?)[.;]?', sentence, re.I)
    if match is None:
        match = re.fullmatch(r'(.+?)\s+by default[.;]?', sentence, re.I)
    if match:
        parsed = scalar(match[1])
        if parsed['status'] == 'parsed':
            return {'status': 'parsed', 'kind': 'scalar', 'value': parsed['value'], 'value_type': parsed['value_type'],
                    'text': sentence, 'reason': 'explicit_literal_default'}
    return result


def extract_claims(blocks):
    claims = []
    for block in blocks:
        if 'field_id' not in block or block['kind'] != 'prose':
            continue
        plain = _plain(block['claim_text'])
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z`])', plain)
        for index, sentence in enumerate(sentences):
            if not _DEFAULT_WORD.search(sentence):
                continue
            claim = _claim(sentence)
            # A condition in an earlier paragraph can qualify a later default.
            earlier = block['field_context'].split('\n')[:block['start_line'] - block['context_start_line']]
            earlier_text = _plain('\n'.join(earlier) + ' ' + ' '.join(sentences[:index]))
            if _QUALIFIED.search(earlier_text) or re.search(r'\bin that case\b', earlier_text, re.I):
                claim.update(status='unsupported', reason='preceding_parameter_context_may_qualify_default')
            claims.append({'claim_id': _digest({'block_id': block['block_id'], 'sentence': index, 'text': sentence}),
                           'block_id': block['block_id'], 'symbol': block['scope'], 'parameter': block['parameter'],
                           'directive_kind': block['directive_kind'], 'claim': claim, 'document': block})
    return claims


def _scope_matches(scope, api):
    if scope == api['scope']:
        return True
    # An unqualified name may bind only if globally unique in the supplied inputs.
    if '.' not in scope:
        return api['scope'].split('.')[-1] == scope
    module = api['path'].replace('\\', '/').removesuffix('.py').replace('/', '.')
    full = module + '.' + api['scope']
    return full.endswith('.' + scope) or full == scope


def _compare(claim, fact):
    declaration = fact['declaration']
    if fact['status'] != 'resolved':
        return 'abstain', fact['reason'], {}
    if claim['status'] != 'parsed':
        return 'abstain', claim['reason'], {}
    value = declaration['value']
    kind = claim['kind']
    if kind == 'scalar':
        other = claim['value']
        if value is None and other is not None:
            return 'abstain', 'sentinel_default_requires_behavior_review', {}
        numeric = type(value) in (int, float) and type(other) in (int, float)
        same = Decimal(str(value)) == Decimal(str(other)) if numeric else type(value) is type(other) and value == other
        return ('consistent_declared_value' if same else 'declared_default_mismatch_candidate',
                'literal_values_agree' if same else 'literal_values_differ', {'comparison': 'literal_declaration'})
    note = fact['unit_annotation']
    if note['status'] != 'parsed':
        return 'abstain', 'source_time_unit_not_uniquely_supported', {}
    if kind == 'duration':
        if type(value) not in (int, float) or value < 0 or claim['value'] < 0:
            return 'abstain', 'non_numeric_or_sentinel_duration_default', {}
        a = Decimal(str(value)) * _FACTORS[note['unit']]
        b = Decimal(str(claim['value'])) * _FACTORS[claim['unit']]
        same = a == b
        return ('consistent_declared_value' if same else 'declared_default_mismatch_candidate',
                'annotated_durations_agree' if same else 'annotated_durations_differ',
                {'comparison': 'literal_default_plus_source_unit_annotation', 'source_seconds': str(a), 'document_seconds': str(b)})
    if kind == 'forever':
        if not note.get('none_means_forever'):
            return 'abstain', 'source_forever_semantics_not_explicitly_annotated', {}
        if value is None:
            return 'consistent_declared_value', 'none_default_has_explicit_forever_annotation', {'comparison': 'source_annotated_sentinel'}
        if type(value) in (int, float) and value > 0:
            return 'declared_default_mismatch_candidate', 'positive_duration_default_conflicts_with_forever_claim', {
                'comparison': 'literal_default_plus_source_unit_and_sentinel_annotation',
                'source_seconds': str(Decimal(str(value)) * _FACTORS[note['unit']])}
        return 'abstain', 'source_forever_mapping_for_this_value_not_supported', {}
    return 'abstain', 'unsupported_claim_kind', {}


def analyze_rst(sources, documents):
    """Return label-free scoped paragraphs and bounded comparison evidence.

    ``sources`` and ``documents`` map paths to original text. No references,
    expected outcomes, diff hints or known-case selectors enter this function.
    """
    if not isinstance(sources, dict) or not isinstance(documents, dict) or any(
            not isinstance(k, str) or not isinstance(v, str) for data in (sources, documents) for k, v in data.items()):
        raise TypeError('sources and documents must map string paths to string source text')
    source = collect_source_facts(sources)
    blocks, excluded = [], []
    for path, text in sorted(documents.items()):
        blocks.extend(split_document(text, path, 'rst'))
        parsed = extract_rst_parameters(text, path)
        blocks.extend(parsed['blocks'])
        excluded.extend(parsed['excluded'])
    claims = extract_claims(blocks)
    rows = []
    field_groups = {}
    for block in blocks:
        if 'field_id' in block:
            key = (block['path'], block['directive_line'], block['parameter'])
            field_groups.setdefault(key, set()).add(block['field_id'])
    for entry in claims:
        document = entry['document']
        row = {**entry, 'status': 'abstain', 'reason': 'no_unique_direct_source_api', 'source': None,
               'declaration': None, 'automatic_edit': False,
               'scope': 'explicit RST parameter assertion versus direct declaration and optional source unit annotation; not runtime proof'}
        apis = [api for api in source['apis'] if api['kind'] == entry['directive_kind'] and _scope_matches(entry['symbol'], api)]
        key = (document['path'], document['directive_line'], entry['parameter'])
        if len(field_groups[key]) > 1:
            row['reason'] = 'repeated_parameter_fields_require_review'
        elif len(apis) == 1:
            api = apis[0]
            candidates = [f for f in source['source_facts'] if f['path'] == api['path']
                          and f['owner'] == api['scope'] and f['api_kind'] == api['kind']
                          and f['parameter'] == entry['parameter']]
            if api['conditional'] or api['decorated']:
                row['reason'] = 'conditional_or_decorated_api'
            elif len(candidates) == 1:
                fact = candidates[0]
                status, reason, comparison = _compare(entry['claim'], fact)
                row.update(status=status, reason=reason, source=fact, declaration=fact['declaration'], **comparison)
                row['binding'] = {'document_scope': entry['symbol'], 'source_api': api,
                                  'constructor_mapping': api['kind'] == 'class',
                                  'basis': 'unique_direct_declaration_in_supplied_sources'}
            else:
                row['reason'] = 'no_unique_direct_parameter_declaration'
        elif len(apis) > 1:
            row['reason'] = 'ambiguous_api_across_source_definitions'
        rows.append(row)
    inventory = {'source_files': len(sources), 'source_parse_failures': len(source['parse_errors']),
                 'document_files': len(documents), 'source_facts': len(source['source_facts']),
                 'all_blocks': len(blocks), 'eligible_blocks': sum(b['kind'] == 'prose' for b in blocks),
                 'scoped_parameter_paragraphs': sum('field_id' in b and b['kind'] == 'prose' for b in blocks),
                 'excluded_scoped_blocks': sum('field_id' in b and b['kind'] != 'prose' for b in blocks),
                 'claims': len(claims), 'row_status_counts': dict(Counter(r['status'] for r in rows))}
    return {'schema_version': 2, 'method': 'bounded_rst_parameter_evidence_v2', 'blocks': blocks,
            'claims': claims, 'rows': rows, 'source_facts': source['source_facts'], 'inventory': inventory,
            'source_parse_errors': source['parse_errors'], 'excluded_constructs': excluded,
            'target_code_executed': False,
            'limitations': ['Only explicit supported API directives and parameter fields are promoted.',
                            'Unique names within supplied files do not prove installed-package import resolution.',
                            'Only direct declarations bind; inheritance, re-exports and runtime overrides are not inferred.',
                            'Units and forever/None meaning require a unique explicit source annotation.',
                            'Comparisons concern declarations and source annotations, not arbitrary runtime behaviour.',
                            'Unsupported clauses, examples, nested directives and ambiguous definitions retain abstention.',
                            'Old opaque directive blocks coexist with newly extracted subparagraphs; only prose is ranked.']}
