"""Bounded, source-only expansion of percent-mapping docstring decorators.

This is an additive v2 adapter. No imported module or target function is executed.
The recognised filldoc family is checked against reviewed AST structures, not
trusted by its module, function name, parameter name, or a desired defect label.
Expansion coordinates always refer to the expanded text; source spans are a
separate provenance list. Runtime mutation, arbitrary decorators and generated
templates remain unsupported.
"""
from __future__ import annotations

import ast
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from decimal import Decimal
import hashlib
from pathlib import PurePosixPath
import re

from .source_audit import declared_parameters, scalar


# ASTs omit docstrings, source coordinates and the outer function's name. These
# are contracts for a reviewed indentation/percent-formatting implementation,
# not case-specific template contents. Any code change fails the contract.
_HELPER_AST = {
    'docformat': '1b977b7e5189ba4649d9b3ded86cbc533641bf89c2e8b49cb8df948595ae9008',
    'indentcount_lines': '6ce6e6418f1db5177f9bcd6611ff6275f5663163f73b6a18f2c6ccdfc0edc5a8',
    'filldoc': '382b019f64fa7ab51ffcb4025540d916274e68f21d5f2d86f644321f6cd7151a',
    'unindent_dict': 'd8e2455bc89a27cc4d3ccbd2cdcd6f7147f9ad9a8bcf88352f231d0d7c54ccef',
    'unindent_string': 'adc98a118ade63d7f0ac2568268986f4e02611a54f1e2cf26cfbb90620fc2180',
}
_PLACEHOLDER = re.compile(r'%%|%\(([^)]+)\)s|%')
_HEADER = re.compile(r'^(?P<names>\*{0,2}[A-Za-z_]\w*(?:\s*,\s*\*{0,2}[A-Za-z_]\w*)*)\s*:\s*(?P<type>.+)$')
_LITERAL = r"(?:'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|None|True|False|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
_BODY_DEFAULT = re.compile(r'(?:^|[.!?]\s+)(?:The\s+)?[Dd]efault(?:\s+value)?\s+is\s+(?P<value>' + _LITERAL + r')\s*\.(?=\s|$)')
_LIMIT = 500_000


def _unique(items):
    result = []
    seen = set()
    for item in items:
        token = repr(sorted(item.items()))
        if token not in seen:
            seen.add(token); result.append(item)
    return result


def _span(path, node, kind='source_expression'):
    return {'path': path, 'start_line': node.lineno,
            'end_line': getattr(node, 'end_lineno', node.lineno), 'kind': kind}


@dataclass
class Value:
    kind: str
    data: object = None
    sources: list = field(default_factory=list)
    steps: list = field(default_factory=list)
    reason: str = ''
    valid: bool = True


def unknown(reason):
    return Value('unknown', reason=reason, valid=False)


def _body_hash(node):
    copy = deepcopy(node)
    if copy.body and isinstance(copy.body[0], ast.Expr) and isinstance(copy.body[0].value, ast.Constant) and isinstance(copy.body[0].value.value, str):
        copy.body = copy.body[1:]
    copy.name = '_adapter'
    return hashlib.sha256(ast.dump(copy, include_attributes=False).encode()).hexdigest()


def _simple_factory(node):
    """Prove the small ``f.__doc__ %= mapping`` factory by its complete AST."""
    if not isinstance(node, ast.FunctionDef): return False
    body = node.body[1:] if ast.get_docstring(node) is not None else node.body
    args = node.args
    if (node.decorator_list or args.posonlyargs or len(args.args) != 1 or args.defaults
            or args.kwonlyargs or args.vararg or args.kwarg or len(body) != 2):
        return False
    inner, ret = body
    if not isinstance(inner, ast.FunctionDef) or not isinstance(ret, ast.Return): return False
    if not isinstance(ret.value, ast.Name) or ret.value.id != inner.name: return False
    if inner.decorator_list or inner.args.posonlyargs or len(inner.args.args) != 1 or inner.args.defaults or inner.args.kwonlyargs or inner.args.vararg or inner.args.kwarg: return False
    if len(inner.body) != 2: return False
    assign, end = inner.body
    if not isinstance(assign, ast.Assign) or len(assign.targets) != 1 or not isinstance(end, ast.Return): return False
    target = assign.targets[0]
    parameter = inner.args.args[0].arg
    def docattr(n): return isinstance(n, ast.Attribute) and n.attr == '__doc__' and isinstance(n.value, ast.Name) and n.value.id == parameter
    return (docattr(target) and isinstance(assign.value, ast.BinOp) and isinstance(assign.value.op, ast.Mod)
            and docattr(assign.value.left) and isinstance(assign.value.right, ast.Name)
            and assign.value.right.id == args.args[0].arg
            and isinstance(end.value, ast.Name) and end.value.id == parameter)


def _reviewed_family(tree):
    definitions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    counts = Counter(n.name for n in tree.body if isinstance(n, ast.FunctionDef))
    if any(counts[name] != 1 or _body_hash(definitions[name]) != digest for name, digest in _HELPER_AST.items()):
        return False
    # The adapter's globals must not be rebound or mutated by module code.
    protected = set(_HELPER_AST) | {'sys', 'min', 'len', 'IndexError'}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            if node.decorator_list: return False
        elif isinstance(node, ast.Import):
            if any(a.name != 'sys' or a.asname for a in node.names): return False
        elif isinstance(node, ast.Assign):
            if any(isinstance(n, ast.Name) and n.id in protected for t in node.targets for n in ast.walk(t)): return False
            try: ast.literal_eval(node.value)
            except (ValueError, TypeError): return False
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            pass
        else:
            return False
    return True


class _Project:
    def __init__(self, sources):
        self.sources = {str(PurePosixPath(p.replace('\\', '/'))): s for p, s in sources.items()}
        self.trees = {}; self.errors = []; self.cache = {}; self.loading = set(); self.functions = []
        for path, source in self.sources.items():
            if not path.endswith('.py'): continue
            try: self.trees[path] = ast.parse(source, filename=path)
            except (SyntaxError, ValueError) as exc: self.errors.append({'path': path, 'reason': 'source_parse_failed', 'detail': str(exc)})

    def find(self, module):
        tail = module.replace('.', '/')
        candidates = [p for p in self.trees if p == tail + '.py' or p.endswith('/' + tail + '.py')
                      or p == tail + '/__init__.py' or p.endswith('/' + tail + '/__init__.py')]
        return candidates[0] if len(candidates) == 1 else None

    def imported(self, module):
        path = self.find(module)
        return Value('module', module, [_span(path, self.trees[path].body[0], 'module_source')] if path and self.trees[path].body else [])

    def attribute(self, base, name):
        if base.kind != 'module': return unknown('unsupported_attribute_access')
        path = self.find(base.data)
        if path:
            env = self.module(path)
            if name in env: return env[name]
        submodule = base.data + '.' + name
        return self.imported(submodule) if self.find(submodule) else unknown('imported_source_or_binding_missing')

    def module(self, path):
        if path in self.cache: return self.cache[path]
        if path in self.loading: return {'__cycle__': unknown('cyclic_import')}
        self.loading.add(path)
        context = _Context(self, path)
        for node in self.trees[path].body: context.statement(node)
        if context.tainted:
            for item in context.function_records:
                if item['status'] == 'expanded_static_template':
                    item.update(status='unsupported_template', reason='module_side_effect_or_control_flow_not_modelled',
                                fields=[], rows=[])
            context.env = {name: unknown('dependency_module_has_unmodelled_side_effects') for name in context.env}
        self.functions.extend(context.function_records)
        self.cache[path] = context.env
        self.loading.remove(path)
        return context.env


class _Context:
    def __init__(self, project, path):
        self.project = project; self.path = path; self.source = project.sources[path]
        self.env = {}; self.function_records = []; self.tainted = False
        self.family = _reviewed_family(project.trees[path])

    def value(self, node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str): return Value('text', node.value, [_span(self.path, node, 'string_literal')])
            if node.value is None or type(node.value) in (bool, int, float): return Value('scalar', node.value)
            return unknown('unsupported_constant')
        if isinstance(node, ast.Name): return self.env.get(node.id, unknown('unresolved_name:' + node.id))
        if isinstance(node, ast.Attribute): return self.project.attribute(self.value(node.value), node.attr)
        if isinstance(node, ast.Dict):
            values = {}
            for key, value in zip(node.keys, node.values):
                if not isinstance(key, ast.Constant) or not isinstance(key.value, str): return unknown('nonliteral_mapping_key')
                if key.value in values: return unknown('duplicate_mapping_key')
                values[key.value] = self.value(value)
            return Value('mapping', values, [_span(self.path, node, 'mapping_literal')])
        if isinstance(node, ast.Subscript):
            base = self.value(node.value)
            if base.kind != 'mapping' or not base.valid: return unknown('unresolved_or_mutated_mapping')
            if not isinstance(node.slice, ast.Constant) or not isinstance(node.slice.value, str): return unknown('dynamic_mapping_lookup')
            return base.data.get(node.slice.value, unknown('missing_mapping_key'))
        if isinstance(node, ast.Call):
            # Python evaluates all arguments even when this adapter rejects the
            # call's signature. Do not hide mutations in unsupported keywords.
            for argument in list(node.args) + [kw.value for kw in node.keywords]:
                self.value(argument)
            if isinstance(node.func, ast.Attribute) and node.func.attr in ('copy', 'replace'):
                base = self.value(node.func.value)
                if node.func.attr == 'copy' and base.kind == 'mapping' and base.valid and not node.args and not node.keywords:
                    result = Value('mapping', dict(base.data), list(base.sources), list(base.steps))
                    result.steps.append({'operation': 'mapping_copy', **_span(self.path, node)}); return result
                if (node.func.attr == 'replace' and base.kind == 'text' and base.valid and len(node.args) == 2
                        and not node.keywords and all(isinstance(a, ast.Constant) and isinstance(a.value, str) for a in node.args)):
                    old, new = [a.value for a in node.args]
                    if not old: return unknown('empty_replace_pattern_unsupported')
                    result = base.data.replace(old, new)
                    if len(result) > _LIMIT: return unknown('expanded_text_size_limit')
                    step = {'operation': 'literal_replace', 'old': old, 'new': new,
                            'occurrences': base.data.count(old), **_span(self.path, node)}
                    return Value('text', result, _unique(base.sources + [_span(self.path, node, 'replacement_expression')]), base.steps + [step])
            fn = self.value(node.func)
            if fn.kind == 'factory':
                if not 1 <= len(node.args) <= (2 if fn.data == 'indent_mapping_v1' else 1): return unknown('factory_arguments_unsupported')
                mapping = self.value(node.args[0])
                if mapping.kind != 'mapping' or not mapping.valid: return unknown('factory_mapping_unresolved')
                unindent = True
                if len(node.args) == 2:
                    flag = self.value(node.args[1])
                    if flag.kind != 'scalar' or type(flag.data) is not bool: return unknown('dynamic_unindent_option')
                    unindent = flag.data
                for keyword in node.keywords:
                    if keyword.arg != 'unindent_params' or fn.data != 'indent_mapping_v1' or len(node.args) == 2: return unknown('factory_keywords_unsupported')
                    flag = self.value(keyword.value)
                    if flag.kind != 'scalar' or type(flag.data) is not bool: return unknown('dynamic_unindent_option')
                    unindent = flag.data
                if fn.data == 'indent_mapping_v1' and unindent:
                    mapping = Value('mapping', {k: _unindent(v) for k, v in mapping.data.items()}, list(mapping.sources), list(mapping.steps))
                step = {'operation': 'create_formatter', 'contract': fn.data, 'unindent': unindent,
                        **_span(self.path, node)}
                return Value('decorator', {'mapping': mapping, 'contract': fn.data}, fn.sources, mapping.steps + [step])
            self.tainted = True
            return unknown('call_not_in_static_allowlist')
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left, right = self.value(node.left), self.value(node.right)
            if left.kind == right.kind == 'text' and left.valid and right.valid and len(left.data) + len(right.data) <= _LIMIT:
                return Value('text', left.data + right.data, _unique(left.sources + right.sources), left.steps + right.steps)
        # Literal lists are allowed as inert module metadata (e.g. __all__).
        try: return Value('literal', ast.literal_eval(node))
        except (ValueError, TypeError, SyntaxError):
            if any(isinstance(child, ast.Call) for child in ast.walk(node)): self.tainted = True
            return unknown('expression_not_in_static_allowlist')

    def bind(self, target, value):
        if isinstance(target, ast.Name): self.env[target.id] = value; return
        if isinstance(target, ast.Subscript):
            base = self.value(target.value)
            if base.kind == 'mapping' and base.valid and isinstance(target.slice, ast.Constant) and isinstance(target.slice.value, str):
                base.data[target.slice.value] = value
                base.steps.append({'operation': 'mapping_item_assignment', 'key': target.slice.value, **_span(self.path, target)})
                return
            if base.kind == 'mapping': base.valid = False
        self.tainted = True

    def statement(self, node):
        if isinstance(node, ast.ImportFrom):
            if any(a.name == '*' for a in node.names): self.tainted = True; return
            if node.level:
                parts = list(PurePosixPath(self.path).with_suffix('').parts)
                if parts[-1] != '__init__': parts.pop()
                else: parts.pop()
                if node.level > len(parts) + 1: self.tainted = True; return
                prefix = parts[:len(parts) - node.level + 1]
                module = '.'.join(prefix + ((node.module or '').split('.') if node.module else []))
            else: module = node.module or ''
            parent = self.project.imported(module)
            for alias in node.names: self.env[alias.asname or alias.name] = self.project.attribute(parent, alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                self.env[alias.asname or alias.name.split('.')[0]] = self.project.imported(alias.name if alias.asname else alias.name.split('.')[0])
        elif isinstance(node, ast.Assign):
            value = self.value(node.value)
            for target in node.targets: self.bind(target, value)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            self.bind(node.target, self.value(node.value))
        elif isinstance(node, ast.Delete):
            for target in node.targets:
                if isinstance(target, ast.Name): self.env.pop(target.id, None)
                else: self.tainted = True
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Defaults, annotations and decorators run at definition time, even
            # for an ordinary function whose docstring has no placeholders.
            expressions = list(node.args.defaults) + [v for v in node.args.kw_defaults if v is not None]
            expressions += [a.annotation for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs if a.annotation]
            expressions += [a.annotation for a in (node.args.vararg, node.args.kwarg) if a and a.annotation]
            if node.returns: expressions.append(node.returns)
            if any(isinstance(child, ast.Call) for expr in expressions for child in ast.walk(expr)):
                self.tainted = True
            for expression in node.decorator_list:
                if self.value(expression).kind != 'decorator': self.tainted = True
            if self.family and node.name == 'filldoc':
                self.env[node.name] = Value('factory', 'indent_mapping_v1', [_span(self.path, node, 'verified_formatter_definition')])
            elif _simple_factory(node):
                self.env[node.name] = Value('factory', 'plain_mapping_v1', [_span(self.path, node, 'verified_formatter_definition')])
            else: self.env[node.name] = unknown('ordinary_function_not_executed')
            self.function_records.append(self.function(node, node.name))
            self.nested_inventory(node.body, node.name + '.<locals>.')
        elif isinstance(node, ast.ClassDef):
            self.tainted = True
            self.env[node.name] = unknown('class_scope_not_interpreted')
            self.nested_inventory(node.body, node.name + '.')
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            pass
        elif isinstance(node, ast.Assert):
            condition = node.test
            if isinstance(condition, ast.Compare) and len(condition.ops) == 1 and isinstance(condition.ops[0], (ast.Eq, ast.NotEq)):
                left, right = self.value(condition.left), self.value(condition.comparators[0])
                if left.kind == right.kind == 'text' and left.valid and right.valid:
                    truth = (left.data == right.data) if isinstance(condition.ops[0], ast.Eq) else (left.data != right.data)
                    if truth: return
            self.tainted = True
        else:
            self.tainted = True
            # Conditional/nested function inventories are retained, never expanded.
            self.nested_inventory([node], '', conditional=True)

    def nested_inventory(self, statements, prefix, conditional=False):
        for node in statements:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                item = self.function(node, prefix + node.name, allow=False)
                item['conditional'] = conditional
                self.function_records.append(item)
                self.nested_inventory(node.body, prefix + node.name + '.<locals>.', conditional)
            elif isinstance(node, ast.ClassDef): self.nested_inventory(node.body, prefix + node.name + '.', conditional)
            else:
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, ast.stmt): self.nested_inventory([child], prefix, True)
                    elif isinstance(child, (ast.ExceptHandler, ast.match_case)): self.nested_inventory(child.body, prefix, True)

    def function(self, node, symbol, allow=True):
        raw = ast.get_docstring(node, clean=False)
        literal = node.body[0].value if raw is not None else None
        item = {'path': self.path, 'symbol': symbol, 'source_line': node.lineno,
                'docstring_source_range': _span(self.path, literal, 'raw_docstring') if literal else None,
                'raw_docstring': raw, 'expanded_docstring': None, 'expansion_steps': [],
                'fields': [], 'rows': [], 'status': 'no_template', 'reason': 'no_mapping_placeholder',
                'parameters': declared_parameters(self.source, node)}
        if raw is None or not re.search(r'%\([^)]+\)s', raw): return item
        item.update(status='unsupported_template', reason='unsupported_or_missing_decorator')
        if not allow: item['reason'] = 'nested_or_class_scope_template_not_interpreted'; return item
        if len(node.decorator_list) != 1: return item
        decorator = self.value(node.decorator_list[0])
        if decorator.kind != 'decorator' or not decorator.valid:
            item['reason'] = decorator.reason or 'unknown_decorator'; return item
        mapping = decorator.data['mapping']
        if not mapping.valid: item['reason'] = 'mutated_mapping'; return item
        if self.tainted: item['reason'] = 'module_side_effect_or_control_flow_not_modelled'; return item
        rawtext = Value('text', raw, [item['docstring_source_range']])
        expanded, line_sources, steps, reason = _expand(rawtext, mapping, decorator.data['contract'])
        if reason: item['reason'] = reason; return item
        item.update(status='expanded_static_template', reason='recognised_static_mapping_contract',
                    expanded_docstring=expanded, expanded_line_sources=line_sources,
                    expansion_steps=decorator.steps + steps,
                    formatter_source_spans=decorator.sources,
                    scope='static docstring assembly under a recognised source contract; not runtime behavior')
        item['fields'] = _fields(expanded, line_sources, self.path, symbol)
        names = Counter(p for block in item['fields'] for p in block['parameters'])
        for block in item['fields']:
            for parameter in block['parameters']:
                item['rows'].append(_compare(block, parameter, item['parameters'].get(parameter), names[parameter] > 1))
        return item


def _unindent(value):
    if value.kind != 'text' or not value.valid: return unknown('mapping_value_not_literal_text')
    lines = value.data.expandtabs().splitlines()
    indent = min((len(s) - len(s.lstrip()) for s in lines if s.strip()), default=0)
    text = '\n'.join(s[indent:] for s in lines) if indent else value.data
    return Value('text', text, value.sources, value.steps + [{'operation': 'remove_common_indent', 'columns': indent}])


def _expand(raw, mapping, contract):
    indent = min((len(line) - len(line.lstrip()) for line in raw.data.expandtabs().splitlines()[1:] if line.strip()), default=0)
    parts = []; steps = []; offset = 0
    for match in _PLACEHOLDER.finditer(raw.data):
        parts.append((raw.data[offset:match.start()], raw.sources))
        if match.group(0) == '%%': parts.append(('%', raw.sources))
        elif match.group(1) is None: return None, [], [], 'unsupported_percent_format'
        else:
            key = match.group(1); value = mapping.data.get(key)
            if value is None: return None, [], [], 'missing_mapping_key:' + key
            if value.kind != 'text' or not value.valid: return None, [], [], 'unresolved_template_fragment:' + key
            text = value.data
            if contract == 'indent_mapping_v1':
                lines = text.expandtabs().splitlines()
                if lines: text = '\n'.join([lines[0]] + [(' ' * indent) + line for line in lines[1:]])
            parts.append((text, _unique(value.sources + raw.sources)))
            steps.extend(value.steps)
            steps.append({'operation': 'substitute_mapping_key', 'key': key, 'source_spans': value.sources,
                          'placeholder_docstring_line': raw.data[:match.start()].count('\n') + 1})
        offset = match.end()
    parts.append((raw.data[offset:], raw.sources))
    output = ''.join(text for text, _ in parts)
    if len(output) > _LIMIT: return None, [], [], 'expanded_text_size_limit'
    line_sources = [[]]
    for text, spans in parts:
        segments = text.split('\n')
        for index, segment in enumerate(segments):
            if index: line_sources.append([])
            if segment: line_sources[-1].extend(spans)
    return output, [_unique(s) for s in line_sources], steps, None


def _fields(text, line_sources, path, symbol):
    lines = text.split('\n')
    indentation = lambda line: len(line) - len(line.lstrip())
    indent = min((indentation(s) for s in lines[1:] if s.strip()), default=0)
    headings = {i: lines[i].strip() for i in range(len(lines)-1)
                if lines[i].strip() and re.fullmatch('-{3,}', lines[i+1].strip())
                and (i == 0 or indentation(lines[i]) == indent) and indentation(lines[i+1]) == indent}
    fields = []
    for start, title in headings.items():
        if title not in ('Parameters', 'Other Parameters'): continue
        end = next((i for i in headings if i > start), len(lines))
        current = None
        for i in range(start+2, end):
            match = _HEADER.fullmatch(lines[i].strip()) if indentation(lines[i]) == indent else None
            if match:
                if current: fields.append(current)
                current = {'path': path, 'symbol': symbol, 'parameters': [n.strip() for n in match['names'].split(',')],
                           'header': lines[i].strip(), 'descriptor': match['type'], 'text': lines[i].strip(),
                           'expanded_start_line': i+1, 'expanded_end_line': i+1}
            elif current:
                current['text'] += '\n' + lines[i].rstrip(); current['expanded_end_line'] = i+1
        if current: fields.append(current)
    for item in fields:
        item['text'] = item['text'].rstrip()
        item['expanded_end_line'] = item['expanded_start_line'] + item['text'].count('\n')
        item['source_spans'] = _unique([span for spans in line_sources[item['expanded_start_line']-1:item['expanded_end_line']] for span in spans])
        item['coordinate_system'] = 'expanded_docstring_lines; source_spans are separate supporting ranges'
        item['block_id'] = hashlib.sha256((path + '::' + symbol + '::expanded:' + str(item['expanded_start_line']) + '::' + item['text']).encode()).hexdigest()
    return fields


def _claim(block):
    text = block['text']
    if len(re.findall(r'\bdefaults?\b', text, re.I)) != 1: return {'status': 'unsupported', 'reason': 'no_unique_default_statement'}
    # Keep qualifiers even when the standalone sentence looks simple.
    if re.search(r'\b(?:if|when|unless|provided|previous|previously|formerly|never|not|may|might|sometimes|examples?|only|except)\b', text, re.I):
        return {'status': 'unsupported', 'reason': 'conditional_negative_or_qualified_default'}
    if '>>>' in text or '::' in text: return {'status': 'unsupported', 'reason': 'example_or_directive_in_parameter_block'}
    header = re.search(r'\bdefault\s*[:=]\s*(.+)$', block['descriptor'], re.I)
    if header: expression = header.group(1).strip()
    else:
        body = '\n'.join(text.split('\n')[1:]).strip()
        matches = list(_BODY_DEFAULT.finditer(body))
        if len(matches) != 1: return {'status': 'unsupported', 'reason': 'no_supported_unconditional_default_sentence'}
        expression = matches[0].group('value')
    result = scalar(expression)
    result.update(expression=expression, evidence_scope='explicit unconditional scalar statement in expanded parameter field')
    return result


def _compare(block, parameter, declaration, ambiguous=False):
    claim = _claim(block)
    row = {'symbol': block['symbol'], 'parameter': parameter, 'block_id': block['block_id'],
           'document': block, 'claim': claim, 'declaration': declaration, 'status': 'abstain',
           'automatic_edit': False, 'scope': 'static expanded documentation versus lexical declaration; not runtime behavior'}
    if ambiguous: row['reason'] = 'ambiguous_documented_parameter'
    elif declaration is None: row['reason'] = 'parameter_not_explicit_in_signature'
    elif declaration['status'] != 'parsed': row['reason'] = declaration['reason']
    elif claim['status'] != 'parsed': row['reason'] = claim['reason']
    elif declaration['value'] is None and claim['value'] is not None: row['reason'] = 'sentinel_requires_behavior_review'
    else:
        a, b = declaration['value'], claim['value']
        numeric = type(a) in (int, float) and type(b) in (int, float)
        equal = Decimal(str(a)) == Decimal(str(b)) if numeric else type(a) is type(b) and a == b
        row.update(status='consistent_declared_value' if equal else 'declared_default_mismatch_candidate',
                   reason='expanded_literal_values_agree' if equal else 'expanded_literal_values_differ')
    return row


def analyze_templates(sources: dict[str, str]):
    """Inspect a single version's Python source mapping without executing it.

    ``functions`` inventories every function; only successfully expanded ones
    contribute ``blocks``/``rows``. A separate v1 literal audit may cover the rest.
    Missing imports, duplicate paths/modules, changed helpers and opaque module
    side effects abstain. Known historical cases reused here are development
    evidence, never an independent held-out benchmark.
    """
    project = _Project(sources)
    records = []
    # Each entry receives an isolated import context. An unrelated input file
    # must never mutate a cached dependency later used by another entry.
    for path in sorted(project.trees):
        entry = _Project(sources)
        entry.module(path)
        records.extend(item for item in entry.functions if item['path'] == path)
    counts = Counter((item['path'], item['symbol']) for item in records)
    for item in records:
        if counts[(item['path'], item['symbol'])] > 1 and item['status'] == 'expanded_static_template':
            item.update(status='unsupported_template', reason='duplicate_function_definition', fields=[], rows=[])
    functions = sorted(records, key=lambda item: (item['path'], item['source_line'], item['symbol']))
    blocks = [block for item in functions if item['status'] == 'expanded_static_template' for block in item['fields']]
    rows = [row for item in functions if item['status'] == 'expanded_static_template' for row in item['rows']]
    return {'schema_version': 2, 'method': 'bounded_template_source_adapter_v2', 'functions': functions,
            'blocks': blocks, 'rows': rows, 'counts': dict(Counter(row['status'] for row in rows)),
            'inventory': {'source_files': len(project.trees), 'parse_failures': len(project.errors),
                          'functions': len(functions), 'template_statuses': dict(Counter(item['status'] for item in functions)),
                          'expanded_parameter_blocks': len(blocks), 'expanded_parameter_rows': len(rows)},
            'diagnostics': project.errors, 'code_executed': False,
            'limitations': ['Only recognised percent-mapping factory structures are interpreted; no target functions are executed.',
                            'Module-level straight-line literal mappings, import aliases, copy and literal replace are supported; arbitrary calls and control flow abstain.',
                            'Source ranges support provenance, not a claim that expanded lines occur verbatim at those file line numbers.',
                            'Nested/class template scopes, arbitrary decorators, runtime mutation, unknown imports and conditional statements are not resolved.',
                            'Simple scalar default statements are checked; other natural-language behavior and units remain unsupported.']}
